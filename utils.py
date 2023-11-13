import contextlib
import os
import sys

from gurobipy import Model, GRB

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


@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as fnull:
        original_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = original_stdout


def get_gurobi_model(silence=True):
    if silence:
        with suppress_stdout():
            m = Model("MNIST_verifier")
    else:
        m = Model("MNIST_verifier")
    m.setParam("OutputFlag", int(not silence))
    return m
