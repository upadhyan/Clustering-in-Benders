

import itertools


import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from networkx.utils import py_random_state


@py_random_state("seed")
def tang_instance(n, m, seed=0):
    """
    Generates a max-cut instance as described in A.2 in [1].
    Parameters
    ----------
    n: int
        Number of nodes
    m: int
        Number of edges
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] Tang, Y., Agrawal, S., & Faenza, Y. (2019). Reinforcement learning for integer
    programming: Learning to cut. arXiv preprint arXiv:1906.04859.
    """
    graph = nx.generators.gnm_random_graph(n, m, seed=seed)
    weights = tang_params(graph, seed=0)
    for (_, _, data), weight in zip(graph.edges(data=True), weights):
        data["weight"] = weight
    _, model = naive(graph)
    return model


@py_random_state("seed")
def tang_params(graph, seed=0):
    """
    Generates max-cut instance params as described in A.2 in [1].
    Parameters
    ----------
    graph: nx.Graph
        Networkx graph
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    weights: list[int]
        Weight for each edge
    References
    ----------
    .. [1] Tang, Y., Agrawal, S., & Faenza, Y. (2019). Reinforcement learning for integer
    programming: Learning to cut. arXiv preprint arXiv:1906.04859.
    """
    weights = []
    for _ in graph.edges:
        weights.append(seed.randint(0, 10))
    return weights
def undirected_edge_name(u, v) -> str:
    """
    :return The name of an undirected edge as "(u,v)" with u <= v.
    """
    u_i, v_i = int(u), int(v)
    if u_i > v_i:
        u_i, v_i = v_i, u_i
    return f"({u_i},{v_i})"

def naive(graph):
    model = gp.Model("Naive MaxCut")

    node_variables = {}
    for v in graph.nodes():
        node_variables[v] = model.addVar(lb=0, ub=1, obj=0, name=str(v), vtype="B")

    edge_variables = {}
    all_non_negative = True
    for u, v, d in graph.edges(data=True):
        edge_name = undirected_edge_name(u, v)
        weight = d["weight"]
        edge_variables[edge_name] = model.addVar(
            lb=0, ub=1, obj=weight, name=edge_name, vtype="B"
        )
        if weight < 0:
            all_non_negative = False

    #model.setMaximize()
    model.setAttr("ModelSense", -1)

    for u, v, d in graph.edges(data=True):
        edge_name = undirected_edge_name(u, v)
        model.addConstr(
            node_variables[u] + node_variables[v] + edge_variables[edge_name] <= 2
        )
        model.addConstr(
            -node_variables[u] - node_variables[v] + edge_variables[edge_name] <= 0
        )
        if not all_non_negative:
            model.addConstr(
                node_variables[u] - node_variables[v] - edge_variables[edge_name] <= 0
            )
            model.addConstr(
                -node_variables[u] + node_variables[v] - edge_variables[edge_name] <= 0
            )

    return (node_variables, edge_variables), model


def triangle(graph):
    model = gp.Model("Triangle MaxCut")

    edge_variables = {}

    for u, v in itertools.combinations(graph.nodes(), 2):
        edge_name = undirected_edge_name(u, v)
        if graph.has_edge(u, v):
            weight = graph.get_edge_data(u, v)["weight"]
        else:
            weight = 0
        edge_variables[edge_name] = model.addVar(
            lb=0, ub=1, obj=weight, name=edge_name, vtype="B"
        )

    #model.setMaximize()
    model.setAttr("ModelSense", -1)

    for i, j, k in itertools.combinations(graph.nodes(), 3):
        x_ij = _get_edge_variable(i, j, edge_variables)
        x_ik = _get_edge_variable(i, k, edge_variables)
        x_kj = _get_edge_variable(k, j, edge_variables)
        model.addCons(x_ij <= x_ik + x_kj)
        model.addCons(x_ij + x_ik + x_kj <= 2)

    return edge_variables, model


def _get_edge_variable(u, v, edge_variables):
    edge_name = undirected_edge_name(u, v)
    return edge_variables[edge_name]