import gurobipy as gp

from networkx.utils import py_random_state

def uncapacitated_lot_sizing(
    T, M, initial_storage, final_storage, p, h, q, d, name="Production Planning"
):
    """
    Generates an uncapacitated lot-sizing MIP instance instance as in 2.1 of [1].
    Parameters
    ----------
    T: int
        Time horizon
    M: int
        Maximum lot size at any time step
    initial_storage: int
        Initial available storage
    final_storage: int
        Storage available at the last time step
    p: list[int]
        Unit production cost at each time step
    h: list[int]
        Unit inventory cost at each time step
    q: list[int]
        Fixed production cost at each time step
    d: list[int]
        Demand at each time step
    name: str
        Name to be given to the generated model
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] Pochet, Y. and Wolsey, L. A. (2006). Production planning by
    mixed integer programming. Springer Science & Business Media.
    """
    model = gp.Model(name)
    # add variables and their cost
    production_vars = []
    produce_or_not_vars = []
    storage_vars = []
    for i in range(T + 1):
        var = model.addVar(lb=0, obj=p[i], name=f"x_{i}", vtype="I")
        production_vars.append(var)

        var = model.addVar(lb=0, ub=1, obj=h[i], name=f"y_{i}", vtype="B")
        produce_or_not_vars.append(var)

        var = model.addVar(lb=0, obj=q[i], name=f"s_{i}", vtype="I")
        storage_vars.append(var)

    # remove unneeded var
    model.remove(production_vars[0])

    # add constraints
    for i in range(1, T + 1):
        model.addConstr(
            storage_vars[i - 1] + production_vars[i] == d[i] + storage_vars[i]
        )
        model.addConstr(production_vars[i] <= M * produce_or_not_vars[i])

    model.addConstr(storage_vars[0] == initial_storage)
    model.addConstr(storage_vars[T] == final_storage)


    #model.setMinimize()

    model.setAttr("ModelSense", 1)



    return model



@py_random_state("seed")
def tang_instance(T, seed=0):
    """Generates a production planning instance as described in A.2 in [1].
    Parameters
    ----------
    T: int
        Time horizon
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
    return uncapacitated_lot_sizing(
        T, *tang_params(T, seed), name="Tang Production Planning"
    )


@py_random_state("seed")
def tang_params(T, seed=0):
    """Generates production planning instance params as described in A.2 in [1].
    Parameters
    ----------
    T: int
        Time horizon
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    M: int
        Maximum lot size at any time step
    initial_storage: int
        Initial available storage
    final_storage: int
        Storage available at the last time step
    p: list[int]
        Unit production cost at each time step
    h: list[int]
        Unit inventory cost at each time step
    q: list[int]
        Fixed production cost at each time step
    d: list[int]
        Demand at each time step
    References
    ----------
    .. [1] Tang, Y., Agrawal, S., & Faenza, Y. (2019). Reinforcement learning for integer
    programming: Learning to cut. arXiv preprint arXiv:1906.04859.
    """
    initial_storage = 0
    final_storage = 20
    M = 100
    p = []
    h = []
    q = []
    d = []
    for i in range(T + 1):
        p.append(seed.randint(1, 10))
        h.append(seed.randint(1, 10))
        q.append(seed.randint(1, 10))
        d.append(seed.randint(1, 10))


    return M, initial_storage, final_storage, p, h, q, d