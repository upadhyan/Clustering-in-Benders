import gurobipy as gp
import numpy as np
from networkx.utils import np_random_state
import scipy.sparse
from networkx.utils import py_random_state

def set_cover(costs, sets, name="Set Cover"):
    """
    Generates basic set cover formulation.
    Parameters
    ----------
    costs: list[float]
        Cost for covering each element
    sets: list[set]
        Set constraints for elements
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    """
    model = gp.Model(name)

    # add variables and their cost
    variables = [
        model.addVar(lb=0, ub=1, obj=c, name=f"v_{i}", vtype="B")
        for i, c in enumerate(costs)
    ]

    # add constraints
    for s in sets:
        model.addConstr(gp.quicksum(variables[i] for i in s) >= 1)

    #model.setMinimize()

    model.setAttr("ModelSense", 1)

    return model









@np_random_state("seed")
def gasse_instance(nrows, ncols, density, max_coef=100, seed=0):
    """
    Generates instance for set cover generation as described in [1].
    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    max_coef: int
        Maximum objective coefficient (>=1)
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.
    """
    return set_cover(
        *gasse_params(nrows, ncols, density, max_coef, seed), name="Gasse Set Cover"
    )


@np_random_state("seed")
def gasse_params(nrows, ncols, density, max_coef=100, seed=0):
    """
    Generates instance params for set cover generation as described in [1],
    based on the code from [2].
    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    max_coef: int
        Maximum objective coefficient (>=1)
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    costs: list[int]
        Element costs in objective function
    sets: list[set]
        Definition of element requirement for each set
    References
    ----------
    .. [1] E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.
    .. [2] https://github.com/ds4dm/learn2branch/blob/master/01_generate_instances.py
    """
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at least 2 rows per col

    # compute number of rows per column
    indices = seed.choice(ncols, size=nnzrs)  # random column indexes
    indices[: 2 * ncols] = np.repeat(
        np.arange(ncols), 2
    )  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = seed.permutation(nrows)  # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i: i + n] = seed.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(
                np.arange(nrows), indices[i:nrows], assume_unique=True
            )
            indices[nrows: i + n] = seed.choice(
                remaining_rows, size=i + n - nrows, replace=False
            )

        i += n
        indptr.append(i)

    # objective coefficients
    c = seed.randint(max_coef, size=ncols) + 1

    # sparse CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
        (np.ones(len(indices), dtype=int), indices, indptr), shape=(nrows, ncols)
    ).tocsr()
    indices = A.indices
    indptr = A.indptr

    costs = list(c)
    sets = [list(indices[indptr[i]: indptr[i + 1]]) for i in range(nrows)]
    return costs, sets

@py_random_state("seed")
def yang_instance(m, seed=0):
    """
    Generates instance for set cover generation as described in [1].
    Parameters
    ----------
    m: int
        Number of set constraints
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] Yu Yang, Natashia Boland, Bistra Dilkina, Martin Savelsbergh,
    "Learning Generalized Strong Branching for Set Covering,
    Set Packing, and 0-1 Knapsack Problems", 2020.
    """
    return set_cover(*yang_params(m, seed))


@py_random_state("seed")
def yang_params(m, seed=0):
    """
    Generates instance params for set cover generation as described in [1].
    Parameters
    ----------
    m: int
        Number of set constraints
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    costs: list[int]
        Element costs in objective function
    sets: list[set]
        Definition of element requirement for each set
    References
    ----------
    .. [1] Yu Yang, Natashia Boland, Bistra Dilkina, Martin Savelsbergh,
    "Learning Generalized Strong Branching for Set Covering,
    Set Packing, and 0-1 Knapsack Problems", 2020.
    """
    n = 10 * m

    costs = [seed.randint(1, 100) for _ in range(n)]

    sets = []
    for _ in range(m):
        num_nonzero = seed.randint(2 * n // 25 + 1, 3 * n // 25 - 1)
        sets.append(set(j for j in seed.sample(range(n), k=num_nonzero)))

    return costs, sets


def _sun_costs(n, seed):
    return [seed.randint(1, 100) for _ in range(n)]


def _sun_sets(n, m, seed, initial_sets=None):
    if not initial_sets:
        sets = [set() for _ in range(m)]
    else:
        sets = list(initial_sets)

    p = 0.05
    for e in range(n):
        # enforce element to appear in at least 2 sets
        for s in (sets[i] for i in seed.sample(range(m), k=2)):
            s.add(e)

        # add element to set with probability p
        for s in sets:
            if seed.random() < p:
                s.add(e)

    return sets


@py_random_state("seed")
def sun_instance(n, m, seed=0):
    """
    Generates instance for set cover generation as described in [1].
    Parameters
    ----------
    n: int
        Number of elements
    m: int
        Number of set constraints
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] Haoran Sun, Wenbo Chen, Hui Li, & Le Song (2021).
         Improving Learning to Branch via Reinforcement Learning. In Submitted to
         International Conference on Learning
    """
    return set_cover(*sun_params(n, m, seed))


@py_random_state("seed")
def sun_params(n, m, seed=0):
    """
    Generates instance params for set cover generation as described in [1].
    Parameters
    ----------
    n: int
        Number of elements
    m: int
        Number of set constraints
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    costs: list[int]
        Element costs in objective function
    sets: list[set]
        Definition of element requirement for each set
    References
    ----------
    .. [1] Haoran Sun, Wenbo Chen, Hui Li, & Le Song (2021).
         Improving Learning to Branch via Reinforcement Learning. In Submitted to
         International Conference on Learning
    """
    return _sun_costs(n, seed), _sun_sets(n, m, seed, initial_sets=None)


@py_random_state("seed")
def expand_sun_params(new_params, base_result, seed=0):
    """
    Implements the expansion from an existing set cover instance as described in [1].
    Parameters
    ----------
    new_params: tuple
        New params for sun_params
    base_result: tuple
        Tuple of (costs, sets) that represent instance params of backbone
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    costs: list[int]
        Element costs in objective function
    sets: list[set]
        Definition of element requirement for each set
    References
    __________
    .. [1] Haoran Sun, Wenbo Chen, Hui Li, & Le Song (2021).
     Improving Learning to Branch via Reinforcement Learning. In Submitted to
     International Conference on Learning
    """
    n, *_ = new_params
    base_costs, base_sets = base_result
    assert n > len(base_costs)

    costs = list(base_costs)
    costs += _sun_costs(n - len(base_costs), seed)

    return costs, _sun_sets(n, len(base_sets), seed, initial_sets=base_sets)
