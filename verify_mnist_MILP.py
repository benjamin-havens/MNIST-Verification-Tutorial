import torch
from gurobipy import GRB, quicksum, max_
from torchvision.transforms.v2.functional import to_pil_image
from tqdm import trange

from mnist import validation_dataset, load_MNIST_model
from gurobi_utils import get_gurobi_model


class MNISTModelVerifier:
    def __init__(self, model=None):
        self.net = model or load_MNIST_model()

    def verify(self, dataset, epsilon, silence_gurobi=True, silence_tqdm=False, silence_print=True):
        loop = trange(len(dataset), disable=silence_tqdm, leave=False)
        loop.set_description("Verification")
        verified = [index for index in loop if self.verify_one(*dataset[index], epsilon, silence_gurobi=silence_gurobi)]
        if not silence_print:
            print(f"Verified {len(verified) / len(dataset):.2%} of the dataset at epsilon={epsilon}.", end=" ")
            if len(verified) >= len(dataset) / 2:
                print("Failed: ", list(filter(lambda i: i not in verified, range(len(dataset)))))
            else:
                print("Verified: ", verified)
        return verified

    def verify_one(self, image, label, epsilon, save_solution=False, silence_gurobi=True, return_counterexample=False):
        if return_counterexample:
            bound, counterexample = self._calculate_bound(image, label, epsilon, save_solution=save_solution,
                                                          silence_gurobi=silence_gurobi,
                                                          return_counterexample=True)
            return bound > 0, counterexample
        bound = self._calculate_bound(image, label, epsilon, save_solution=save_solution,
                                      silence_gurobi=silence_gurobi)
        return bound > 0

    def _calculate_bound(self, image, label, epsilon, save_solution=False, silence_gurobi=True,
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
                # Indicator variables for pre-activations
                pre_activation_indicators = list(m.addVars(n_in, name=f"ind{index}", vtype=GRB.BINARY).values())
                # Activation variables
                activations = list(m.addVars(n_in, name=f"a{index}").values())

                for i in range(n_in):
                    m.addGenConstrIndicator(pre_activation_indicators[i], True, activations[i] == pre_activation[i])
                    m.addGenConstrIndicator(pre_activation_indicators[i], True, pre_activation[i] >= 0)
                    m.addGenConstrIndicator(pre_activation_indicators[i], False, activations[i] == 0)
                    m.addGenConstrIndicator(pre_activation_indicators[i], False, pre_activation[i] <= 0)
            elif isinstance(module, torch.nn.Flatten):
                pass
            elif isinstance(module, torch.nn.BatchNorm1d):
                pass  # TODO: account for batchnorm
            else:
                raise TypeError(f"Verifier not equipped to handle layer of type {type(module)}")

        # Minimize the gap between the correct output logit and the largest incorrect logit.
        output_neurons = pre_activation  # We assume no ReLU on output neurons
        max_incorrect_logit = m.addVar(lb=float("-inf"), name="max_incorrect_logit")
        m.addConstr(max_incorrect_logit == max_([var for i, var in enumerate(output_neurons) if i != label]),
                    name="max_incorrect_logit")
        m.setObjective(output_neurons[label] - max_incorrect_logit, GRB.MINIMIZE)
        m.optimize()

        if save_solution:
            with open("outputs/MILP_solution.txt", "w") as f:
                for v in m.getVars():
                    f.write(f"{v.varName} {v.x:.6}\n")

        if return_counterexample:
            input_variables = filter(lambda v: v.varName.startswith("x"), m.getVars())
            counterexample = torch.tensor([v.x for v in input_variables])
            return m.ObjVal, counterexample.unsqueeze(0)

        return m.ObjVal


if __name__ == "__main__":
    verifier = MNISTModelVerifier()
    # dataset = [validation_dataset[i] for i in range(10)]
    # verifier.verify(dataset=dataset, epsilon=0.05, silence_gurobi=True, silence_print=False)
    # verified, counterexample = verifier.verify_one(*dataset[2], epsilon=0.05, silence_gurobi=True, save_solution=True,
    #                                                return_counterexample=True)

    epsilon = 0.08
    for idx, (original_image, label) in enumerate(validation_dataset):
        verified, counterexample = verifier.verify_one(original_image, label, epsilon=epsilon, silence_gurobi=True,
                                                       save_solution=True,
                                                       return_counterexample=True)
        if not verified:
            to_pil_image(original_image.reshape(1, 28, 28)).resize((256, 256)).show()
            to_pil_image(counterexample.reshape(1, 28, 28)).resize((256, 256)).show()
            print(f"Real label is {label}, model predicts {verifier.net(counterexample).argmax(1).item()}")
        else:
            assert verifier.net(counterexample).argmax(1).item() == label
            print(f"Index {idx} successfully verified at {epsilon=}")
        if input("q to quit: ") == "q":
            break
