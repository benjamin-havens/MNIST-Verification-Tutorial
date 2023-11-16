"""
This script using Linear Programming to verify model performance. Relies on the triangle relaxation for ReLU.
Gives bounds only, so if the model passes verification on a given image, performance is guaranteed; however, if the model
    fails verification on a given image, it not actually misclassify any images perturbed in plus or minus epsilon. That
    is to say, false negatives are impossible, but false positives are possible.
"""

import numpy as np
import torch
from gurobipy import GRB, quicksum, max_
from tqdm import trange

from gurobi_utils import get_gurobi_model
from mnist import validation_dataset, load_MNIST_model


class MNISTModelVerifier:
    """
    This class uses Linear Programming to verify performance of an MNISTModel.
    """

    def __init__(self, model=None):
        """
        Saves or creates an MNISTModel for later verification.
        :param model: The MNISTModel instance to verify. If None, will attempt to load from the default configuration
                        (2 layers of 100 hidden units, located at mnist_model.pt)
        """
        self.net = model or load_MNIST_model()

    def verify(self, dataset, epsilon, silence_gurobi=True, silence_tqdm=False, silence_print=True):
        """
        Verify performance of self.model on dataset with perturbation level epsilon, return the indices of verified images.
        :param dataset: An MNIST dataset--iterable of (image, label) tensors. Should support indexing or return value
                        will not be meaningful.
        :param epsilon: Perturbation level. Each pixel may be independently perturbed within [pixel - epsilon, pixel +
                        epsilon], clamped at [0, 1], creating an L-infinity ball with the image at the center.
        :param silence_gurobi: Whether to silence output from the gurobi LP solver. Default True.
        :param silence_tqdm: Whether to silence the progress bars. Default False.
        :param silence_print: Whether to silence print statements. Default True.
        :return: List of verified images from the dataset.
        """
        loop = trange(len(dataset), disable=silence_tqdm, leave=False)
        loop.set_description("Verification")

        # The main loop over the dataset
        verified = [index for index in loop if self.verify_one(*dataset[index], epsilon, silence_gurobi=silence_gurobi)]

        if not silence_print:
            print(f"Verified {len(verified) / len(dataset):.2%} of the dataset at epsilon={epsilon}.", end=" ")
            if len(verified) >= len(dataset) / 2:
                print("Failed: ", list(filter(lambda i: i not in verified, range(len(dataset)))))
            else:
                print("Verified: ", verified)

        return verified

    def verify_one(self, image, label, epsilon, save_solution_filename=None, silence_gurobi=True):
        """
        Verify performance on a single instance.
        :param image: Image tensor from the MNIST dataset. Should have 28x28 pixels.
        :param label: The corresponding label.
        :param epsilon: Perturbation level. Each pixel may be independently perturbed within [pixel - epsilon, pixel +
                        epsilon], clamped at [0, 1]
        :param save_solution_filename: If provided, will save all variables from the solver to this location.
        :param silence_gurobi: Whether to silence output from the gurobi LP solver.
        :return: True if the model's performance was verified on this image, else False.
        """
        bound = self._calculate_bound(image, label, epsilon, save_solution_filename=save_solution_filename,
                                      silence_gurobi=silence_gurobi)
        return bound > 0

    def _calculate_bound(self, image, label, epsilon, save_solution_filename=None, silence_gurobi=True):
        m = get_gurobi_model(silence_gurobi)

        # Add the input neuron variables and constraints
        image = image.flatten()
        n_in = image.shape[0]
        lb = (image - epsilon * np.ones(n_in)).clip(min=0, max=1)
        ub = (image + epsilon * np.ones(n_in)).clip(min=0, max=1)
        activations = np.array(list(m.addVars(n_in, lb=lb, ub=ub, name="x").values()))
        pre_activation = None  # We assume the network doesn't start with a ReLU

        # Go through sequential net and tighten bounds using LP at each layer
        for index, module in enumerate(self.net.net):
            if isinstance(module, torch.nn.Linear):
                # For linear layers, we extract parameters from the model and create exact constraints between
                # activations from the previous layer and pre-activation values for the next layer.
                parameters = dict(module.named_parameters())
                weight = parameters["weight"].detach().numpy()
                bias = parameters["bias"].detach().numpy()
                n_out, n_in = weight.shape

                pre_activation = np.array(list(m.addVars(n_out, lb=float("-inf"), name=f"z{index}").values()))
                m.addConstrs((
                    pre_activation[i] == quicksum([weight[i][j] * activations[j] for j in range(n_in)]) + bias[i]
                    for i in range(n_out)),
                    f"{index}.Linear",
                )

                n_in = n_out
            elif isinstance(module, torch.nn.ReLU):
                # For ReLU layers, we use the LP solver to create bounds on pre_activation variables, and use the triangle
                # relaxation to turn those bounds into bounds on activation variables. This is where inexactness is
                # introduced, meaning that failed verification may be a false postive.

                # Compute bounds
                lb, ub = np.zeros_like(pre_activation), np.zeros_like(pre_activation)
                for idx, neuron in enumerate(pre_activation):
                    m.setObjective(neuron, sense=GRB.MINIMIZE)
                    m.optimize()
                    lb[idx] = m.ObjVal
                    m.setObjective(neuron, sense=GRB.MAXIMIZE)
                    m.optimize()
                    ub[idx] = m.ObjVal

                activations = np.array(list(m.addVars(n_in, name=f"a{index}").values()))

                for i in range(n_in):
                    if lb[i] >= 0:
                        m.addConstr(activations[i] == pre_activation[i], name=f"{index}.ReLU.activation_{i}")
                    elif ub[i] <= 0:
                        m.addConstr(activations[i] == 0, name=f"{index}.ReLU.activation_{i}")
                    else:  # lb[i] < 0 < ub[i]:
                        # Triangle relaxation for ReLU: a is greater than 0, greater than z, and less than the line connecting
                        # the lower bound's activation and the upper bound's activation.
                        # The first constraint is handled by default--see docs for addVars
                        # m.addConstr(activations[i] >= 0, name=f"{index}.ReLU.activation_{i}_triangle_1")
                        m.addConstr(activations[i] >= pre_activation[i], name=f"{index}.ReLU.activation_{i}_triangle_2")
                        m.addConstr(
                            activations[i] <= (pre_activation[i] - lb[i]) * ub[i] / (ub[i] - lb[i]),
                            name=f"{index}.ReLU.activation_{i}_triangle_3",
                        )
            elif isinstance(module, torch.nn.Flatten):
                pass
            else:
                raise TypeError(f"Verifier not equipped to handle layer of type {type(module)}")

        # Minimize the gap between the correct output logit and the largest incorrect logit.
        output_neurons = pre_activation  # We assume no ReLU on output neurons
        max_incorrect_logit = m.addVar(lb=float("-inf"), name="max_incorrect_logit")
        m.addConstr(max_incorrect_logit == max_([var for i, var in enumerate(output_neurons) if i != label]),
                    name="max_incorrect_logit")
        m.setObjective(output_neurons[label] - max_incorrect_logit, GRB.MINIMIZE)
        m.optimize()

        if save_solution_filename:
            try:
                with open(save_solution_filename, "w") as f:
                    for v in m.getVars():
                        f.write(f"{v.varName} {v.x:.6}\n")
            except FileNotFoundError:
                raise ValueError(f"Could not write solution to {save_solution_filename}")

        return m.ObjVal


if __name__ == "__main__":
    # Example usage:

    # 1. Verify a portion of the validation dataset
    verifier = MNISTModelVerifier()
    dataset = [validation_dataset[i] for i in range(10)]
    verifier.verify(dataset=dataset, epsilon=0.05, silence_print=False)

    # 2. Verify a specific image, saving the solution to a file
    verified = verifier.verify_one(*dataset[0], epsilon=0.05, save_solution_filename="LP_solution.txt")

