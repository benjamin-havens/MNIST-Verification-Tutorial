"""
This script using Mixed Integer Linear Programming to verify model performance.
Because MILP gives exact solutions, rather than bounds as LP and IBP, it can also return the perturbed input that gives
    the worst performance, functioning as a counterexample to the model's reliability. If any inputs are misclassified
    (at a given epsilon), the counterexample will be, and by the largest margin.
The cost of this improved accuracy and functionality is a drastically increased evaluation time. Even on a small network,
    this method takes ~10 seconds.
"""

import torch
from gurobipy import GRB, quicksum, max_
from torchvision.transforms.v2.functional import to_pil_image
from tqdm import trange

from gurobi_utils import get_gurobi_model
from mnist import validation_dataset, load_MNIST_model
from printing_utils import delete_last_line


class MNISTModelVerifier:
    """
    This class uses Mixed Integer Linear Programming to verify performance of an MNISTModel.
    """

    def __init__(self, model=None):
        """
        Saves or creates an MNISTModel for later verification.
        :param model: The MNISTModel instance to verify. If None, will attempt to load from the default configuration
                        (2 layers of 100 hidden units, located at mnist_model.pt)
        """
        try:
            self.net = model or load_MNIST_model()
        except FileNotFoundError:
            raise ValueError("No model passed in and no model found at mnist_model.pt.")

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

    def verify_one(self, image, label, epsilon, save_solution_filename=None, silence_gurobi=True,
                   return_counterexample=False):
        """
        Verify performance on a single instance.
        :param image: Image tensor from the MNIST dataset. Should have 28x28 pixels.
        :param label: The corresponding label.
        :param epsilon: Perturbation level. Each pixel may be independently perturbed within [pixel - epsilon, pixel +
                        epsilon], clamped at [0, 1]
        :param save_solution_filename: If provided, will save all variables from the solver to this location.
        :param silence_gurobi: Whether to silence output from the gurobi LP solver.
        :param return_counterexample: If True, will return a tuple of (verified, counterexample).
        :return: True if the model's performance was verified on this image, else False. See also return_counterexample.
        """
        if return_counterexample:
            bound, counterexample = self._calculate_bound(image, label, epsilon,
                                                          save_solution_filename=save_solution_filename,
                                                          silence_gurobi=silence_gurobi,
                                                          return_counterexample=True)
            return bound > 0, counterexample
        bound = self._calculate_bound(image, label, epsilon, save_solution_filename=save_solution_filename,
                                      silence_gurobi=silence_gurobi)
        return bound > 0

    def _calculate_bound(self, image, label, epsilon, save_solution_filename=None, silence_gurobi=True,
                         return_counterexample=False):
        m = get_gurobi_model(silence_gurobi)

        # Add the input neuron variables and constraints
        image = image.flatten()
        n_in = image.shape[0]
        lb = (image - epsilon * torch.ones(n_in)).clip(min=0, max=1)
        ub = (image + epsilon * torch.ones(n_in)).clip(min=0, max=1)
        activations = list(m.addVars(n_in, lb=lb, ub=ub, name="x").values())
        pre_activation = None  # We assume the network doesn't start with a ReLU

        # Go through sequential net and bound variables at each layer
        for index, module in enumerate(self.net.net):
            if isinstance(module, torch.nn.Linear):
                # For linear layers, we extract parameters from the model and create exact constraints between
                # activations from the previous layer and pre-activation values for the next layer.
                parameters = dict(module.named_parameters())
                weight = parameters["weight"].detach().numpy()
                bias = parameters["bias"].detach().numpy()
                n_out, n_in = weight.shape

                pre_activation = list(m.addVars(n_out, lb=float("-inf"), name=f"z{index}").values())
                m.addConstrs((
                    pre_activation[i] == quicksum([weight[i][j] * activations[j] for j in range(n_in)]) + bias[i]
                    for i in range(n_out)),
                    f"{index}.Linear",
                )

                n_in = n_out
            elif isinstance(module, torch.nn.ReLU):
                # For ReLU layers, we use a binary variable to 'branch' into active or inactive neuron states.

                # Indicator variables for pre-activations
                pre_activation_indicators = list(m.addVars(n_in, name=f"ind{index}", vtype=GRB.BINARY).values())
                # Activation variables
                activations = list(m.addVars(n_in, name=f"a{index}").values())

                for i in range(n_in):
                    # The binary variable should be high when the pre-activation sum is geq 0, else low.
                    m.addGenConstrIndicator(pre_activation_indicators[i], True, pre_activation[i] >= 0)
                    m.addGenConstrIndicator(pre_activation_indicators[i], False, pre_activation[i] <= 0)
                    # If the binary variable is high, activations = pre-activations, else they equal 0.
                    m.addGenConstrIndicator(pre_activation_indicators[i], True, activations[i] == pre_activation[i])
                    m.addGenConstrIndicator(pre_activation_indicators[i], False, activations[i] == 0)
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

        if return_counterexample:
            input_variables = filter(lambda v: v.varName.startswith("x"), m.getVars())
            counterexample = torch.tensor([v.x for v in input_variables])
            return m.ObjVal, counterexample.unsqueeze(0)

        return m.ObjVal


if __name__ == "__main__":
    # Example usages:
    # 1. Loop through a dataset, checking performance on each image with the same epsilon. If the model fails verification,
    # show the original image, perturbed image, and predicted vs actual label.
    # You can break out of the loop by entering 'q' when prompted
    verifier = MNISTModelVerifier()
    epsilon = 0.01
    print(f"Using {epsilon=}")

    for idx, (original_image, label) in enumerate(validation_dataset):
        if input("q to quit, anything else to continue: ").lower().strip() == "q":
            break
        delete_last_line()

        verified, counterexample = verifier.verify_one(original_image, label, epsilon=epsilon, silence_gurobi=True,
                                                       return_counterexample=True)
        if not verified:
            to_pil_image(original_image.reshape(1, 28, 28)).resize((280, 280)).show()
            to_pil_image(counterexample.reshape(1, 28, 28)).resize((280, 280)).show()
            print(
                f"Index {idx} failed: real label is {label} but model predicts {verifier.net(counterexample).argmax(1).item()}")
        else:
            assert verifier.net(counterexample).argmax(1).item() == label
            print(f"Index {idx} successfully verified.")

    # 2. Verify a portion of the validation dataset
    dataset = [validation_dataset[i] for i in range(10)]
    verifier.verify(dataset=dataset, epsilon=0.05, silence_print=False)

    # 3. Verify a specific image, saving the solution to a file
    verified = verifier.verify_one(*dataset[0], epsilon=0.05, save_solution_filename="MILP_solution.txt")
