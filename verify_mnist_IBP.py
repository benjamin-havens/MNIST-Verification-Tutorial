import numpy as np
import torch
from gurobipy import Model, GRB, quicksum, max_
from tqdm import trange

from mnist import train_dataset, validation_dataset, load_MNIST_model
from silence_output import suppress_stdout

GRB_STATUS_MEANING = {
    GRB.Status.LOADED: "LOADED",
    GRB.Status.OPTIMAL: "OPTIMAL",
    GRB.Status.INFEASIBLE: "INFEASIBLE",
    GRB.Status.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.Status.UNBOUNDED: "UNBOUNDED",
    GRB.Status.CUTOFF: "CUTOFF",
    GRB.Status.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.Status.NODE_LIMIT: "NODE_LIMIT",
    GRB.Status.TIME_LIMIT: "TIME_LIMIT",
    GRB.Status.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.Status.INTERRUPTED: "INTERRUPTED",
    GRB.Status.NUMERIC: "NUMERIC",
    GRB.Status.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.Status.INPROGRESS: "INPROGRESS",
    GRB.Status.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
}


# noinspection PyArgumentList
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

    def verify_one(self, image, label, epsilon, save_solution=False, silence_gurobi=True):
        m = self._get_gurobi_model(image, label, epsilon, silence_gurobi=silence_gurobi)
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            if save_solution:
                with open("outputs/IBP_solution.txt", "w") as f:
                    for v in m.getVars():
                        f.write(f"{v.varName} {v.x:.6}\n")
                m.write("model.mps")
            return m.ObjVal > 0
        elif m.status == GRB.Status.INFEASIBLE:
            m.computeIIS()
            m.write("model.ilp")
            m.write("model.mps")
            raise RuntimeError("Got solver status INFEASIBLE")
        else:
            m.write("model.mps")
            status = GRB_STATUS_MEANING.get(m.status, "UNKNOWN STATUS")
            raise RuntimeError(f"Got solver status {status}")

    def _get_gurobi_model(self, image, label, epsilon, silence_gurobi=True):
        if silence_gurobi:
            with suppress_stdout():
                m = Model("MNIST_verifier")
        else:
            m = Model("MNIST_verifier")
        m.setParam("OutputFlag", int(not silence_gurobi))

        # Add the input neuron variables and constraints
        image = image.flatten()
        n_in = image.shape[0]
        lb = (image - epsilon * np.ones(n_in)).clip(min=0, max=1)
        ub = (image + epsilon * np.ones(n_in)).clip(min=0, max=1)
        activations = np.array(list(m.addVars(n_in, lb=lb, ub=ub, name="x").values()))
        pre_activation = None

        # Go through sequential net and add constraints and variables
        for index, module in enumerate(self.net.net):
            if isinstance(module, torch.nn.Linear):
                parameters = dict(module.named_parameters())
                weight = parameters["weight"].detach().numpy()
                bias = parameters["bias"].detach().numpy()
                n_out, n_in = weight.shape

                # Update bounds
                new_lb = np.zeros(n_out)
                new_ub = np.zeros(n_out)
                for i in range(n_out):
                    for j in range(n_in):
                        if weight[i, j] > 0:
                            new_lb[i] += weight[i, j] * lb[j]
                            new_ub[i] += weight[i, j] * ub[j]
                        else:
                            new_lb[i] += weight[i, j] * ub[j]
                            new_ub[i] += weight[i, j] * lb[j]
                    new_lb[i] += bias[i]
                    new_ub[i] += bias[i]
                lb, ub = np.minimum(new_lb, new_ub), np.maximum(new_lb, new_ub)

                pre_activation = np.array(list(m.addVars(n_out, lb=lb, ub=ub, name=f"z{index}").values()))
                m.addConstrs((
                    pre_activation[i] == quicksum([weight[i][j] * activations[j] for j in range(n_in)]) + bias[i]
                    for i in range(n_out)),
                    f"{index}.Linear",
                )

                n_in = n_out
            elif isinstance(module, torch.nn.ReLU):
                # Update bounds
                new_lb = lb.clip(min=0)
                new_ub = ub.clip(min=0)
                new_lb, new_ub = np.minimum(new_lb, new_ub), np.maximum(new_lb, new_ub)

                activations = np.array(list(m.addVars(n_in, lb=new_lb, ub=new_ub, name=f"a{index}").values()))

                for i in range(n_in):
                    if lb[i] >= 0:
                        m.addConstr(activations[i] == pre_activation[i], name=f"{index}.ReLU.activation_{i}")
                    elif ub[i] <= 0:
                        m.addConstr(activations[i] == 0, name=f"{index}.ReLU.activation_{i}")
                    else:  # lb[i] < 0 < ub[i]:
                        # Triangle relaxation for ReLU: a is greater than 0, greater than z, and less than the line connecting
                        # the lower bound's activation and the upper bound's activation
                        m.addConstr(activations[i] >= 0, name=f"{index}.ReLU.activation_{i}_triangle_1")
                        m.addConstr(activations[i] >= pre_activation[i], name=f"{index}.ReLU.activation_{i}_triangle_2")
                        m.addConstr(
                            activations[i] <= (pre_activation[i] - lb[i]) * ub[i] / (ub[i] - lb[i]),
                            name=f"{index}.ReLU.activation_{i}_triangle_3",
                        )

                lb, ub = new_lb, new_ub
            elif isinstance(module, torch.nn.Flatten):
                pass
            elif isinstance(module, torch.nn.BatchNorm1d):
                pass  # TODO: account for batchnorm
            else:
                raise TypeError(f"Verifier not equipped to handle layer of type {type(module)}")

        # Minimize the gap between the correct output logit and the largest incorrect logit.
        output_neurons = pre_activation  # We assume no ReLU on output neurons
        max_incorrect_logit = m.addVar(lb=min(lb), ub=max(ub), name="max_incorrect_logit")
        m.addConstr(max_incorrect_logit == max_([var for i, var in enumerate(output_neurons) if i != label]),
                    name="max_incorrect_logit")
        m.setObjective(output_neurons[label] - max_incorrect_logit, GRB.MINIMIZE)

        return m


if __name__ == "__main__":
    verifier = MNISTModelVerifier()
    dataset = [validation_dataset[i] for i in range(50)]
    # verifier.verify(dataset=dataset, epsilon=0.05, silence_gurobi=True, silence_print=False)
    verifier.verify_one(*dataset[0], epsilon=0.05, silence_gurobi=True, save_solution=True)
