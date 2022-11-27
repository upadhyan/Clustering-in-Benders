import networkx as nx
import gurobipy as gp

from networkx.utils import py_random_state


def independent_set(graph, name="Independent Set"):
    """
    Generates an independent set instance according to [1].
    Parameters
    ----------
    graph: nx.Graph
        Networkx undirected graph
    name: str
        Name of the generated model
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] https://www.princeton.edu/~aaa/Public/Teaching/ORF523/S16/ORF523_S16_Lec11_gh.pdf
    """
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    model = gp.Model(name)

    vars = {
        str(node): model.addVar(lb=0, ub=1, obj=1, name=str(node), vtype="B")
        for node in graph.nodes
    }

    for u, v in graph.edges:
        model.addConstr(vars[str(u)] + vars[str(v)] <= 1)

    #model.setMaximize()

    model.setAttr("ModelSense", -1)

    return model


def _get_cliques(graph):
    """
    Partition the graph into cliques using a greedy algorithm, this code is
    based on the code from [1].
    Parameters
    ----------
    graph: nx.Graph
        Networkx undirected graph
    Returns
    -------
    cliques: list[set]
        The resulting clique partition
    References
    ----------
    .. [1] https://github.com/ds4dm/learn2branch/blob/master/01_generate_instances.py
    """
    cliques = []

    # sort nodes in descending order of degree
    leftover_nodes = sorted(list(graph.nodes), key=lambda node: -graph.degree[node])

    while leftover_nodes:
        clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
        clique = {clique_center}
        neighbors = set(graph.neighbors(clique_center)).intersection(leftover_nodes)
        densest_neighbors = sorted(neighbors, key=lambda node: -graph.degree[node])
        for neighbor in densest_neighbors:
            if all(
                [neighbor in graph.neighbors(clique_node) for clique_node in clique]
            ):
                clique.add(neighbor)
        cliques.append(clique)
        leftover_nodes = [node for node in leftover_nodes if node not in clique]

    return cliques


def clique_independent_set(graph, name="Clique Independent Set"):
    """
    Generates an independent set instance according to [1, 4.6.4].
    Parameters
    ----------
    graph: nx.Graph
        Networkx undirected graph
    name: str
        Name of the generated model
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] David Bergman, Andre A. Cire, Willem-Jan Van Hoeve, and John Hooker. Decision diagrams
    for optimization. Springer, 2016.
    """
    graph = nx.relabel.convert_node_labels_to_integers(graph)
    model = gp.Model(name)

    cliques = _get_cliques(graph)

    vars = {
        str(node): model.addVar(lb=0, ub=1, obj=1, name=str(node), vtype="B")
        for node in graph.nodes
    }

    for clique in cliques:
        model.addConstr(gp.quicksum(vars[str(node)] for node in clique) <= 1)

    #model.setMaximize()

    model.setAttr("ModelSense", -1)

    return model

@py_random_state("seed")
def gasse_params(n, p, seed=0):
    """
    Generates a maximum independent set instance as described in [1].
    Parameters
    ----------
    n: int
        Number of nodes.
    p: float
        Edge probability
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    graph: nx.Graph
        An Erdos-Renyi Networkx graph
    References
    ----------
    .. [1] "Exact Combinatorial Optimization with Graph Convolutional Neural Networks" (2019)
      Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin and Andrea Lodi
      Advances in Neural Information Processing Systems 32 (2019)
    """
    return nx.generators.erdos_renyi_graph(n, p, seed)


@py_random_state("seed")
def gasse_instance(n, p, seed=0):
    """
    Generates a maximum independent set instance as described in [1].
    Parameters
    ----------
    n: int
        Number of nodes.
    p: float
        Edge probability
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the instance.
    References
    ----------
    .. [1] "Exact Combinatorial Optimization with Graph Convolutional Neural Networks" (2019)
      Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin and Andrea Lodi
      Advances in Neural Information Processing Systems 32 (2019)
    """
    return clique_independent_set(
        gasse_params(n, p, seed), name="Gasse Independent Set"
    )

@py_random_state("seed")
def barabasi_albert_params(n, m, seed=0):
    """
    Generates a maximum independent set instance params of graphs described in [1].
    Parameters
    ----------
    n: int
        Number of nodes
    m: int
        Number of edges to attach from a new node to existing nodes
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    graph: nx.Graph
        A Barabasi-Albert Networkx graph
    References
    ----------
    .. [1] A. L. Barabási and R. Albert “Emergence of scaling in random networks”, Science 286, pp 509-512, 1999.
    """
    return nx.generators.barabasi_albert_graph(n, m, seed)


@py_random_state("seed")
def barabasi_albert_instance(n, m, seed=0):
    """
    Generates a maximum independent set instance of graphs described in [1].
    Parameters
    ----------
    n: int
        Number of nodes
    m: int
        Number of edges to attach from a new node to existing nodes
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the instance.
    References
    ----------
    .. [1] A. L. Barabási and R. Albert “Emergence of scaling in random networks”, Science 286, pp 509-512, 1999.
    """
    return independent_set(
        barabasi_albert_params(n, m, seed), name="Barabasi-Albert Independent Set"
    )




