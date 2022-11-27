import itertools
import gurobipy as gp
import numpy as np
from networkx.utils import py_random_state


def capacitated_facility_location(
    n_customers,
    n_facilities,
    transportation_cost,
    demands,
    fixed_costs,
    capacities,
    name="Capacitated Facility Location",
):
    """
    Generate a Capacitated Facility Location MIP formulation following [1].
    Parameters
    ----------
    n_customers: int
        The desired number of customers
    n_facilities: int
        The desired number of facilities
    transportation_cost: numpy array [float]
        Matrix of transportation costs from customer i to facility j [i,j]
    demands: numpy array [int]
        Demands of each customer
    fixed_costs: numpy array [int]
        Fixed costs of operating each facility
    capacities: numpy array [int]
        Capacities of each facility
    name: str
        Name of the model
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.
    """
    model = gp.Model(name)

    total_demand = demands.sum()

    #model.setMinimize()
    model.setAttr("ModelSense", 1)

    customer_facility_vars = {}
    facility_vars = []
    # add customer-facility vars
    for i, j in itertools.product(range(n_customers), range(n_facilities)):
        var = model.addVar(
            lb=0, ub=1, obj=transportation_cost[i, j], name=f"x_{i}_{j}", vtype="B"
        )
        customer_facility_vars[i, j] = var
    # add facility vars
    for j in range(n_facilities):
        var = model.addVar(lb=0, ub=1, obj=fixed_costs[j], name=f"y_{j}", vtype="B")
        facility_vars.append(var)

    # add constraints
    for i in range(n_customers):
        model.addConstr(
            gp.quicksum(customer_facility_vars[i, j] for j in range(n_facilities))
            >= 1
        )
    for j in range(n_facilities):
        model.addConstr(
            gp.quicksum(
                demands[i] * customer_facility_vars[i, j] for i in range(n_customers)
            )
            <= capacities[j] * facility_vars[j]
        )

    # optional constraints

    # total capacity constraint
    model.addConstr(
        gp.quicksum(capacities[j] * facility_vars[j] for j in range(n_facilities))
        >= total_demand
    )

    # affectation constraints
    for i, j in itertools.product(range(n_customers), range(n_facilities)):
        model.addConstr(customer_facility_vars[i, j] <= facility_vars[j])

    return model


def capacitated_warehouse_location(
    n_customers,
    n_facilities,
    transportation_cost,
    demands,
    fixed_costs,
    capacities,
    name="Capacitated Warehouse Location",
):
    """
    Generate a Capacitated Warehouse Location MIP formulation following [1].
    Parameters
    ----------
    n_customers: int
        The desired number of customers
    n_facilities: int
        The desired number of warehouses
    transportation_cost: numpy array [float]
        Matrix of transportation costs from customer i to warehouse j [i,j]
    demands: numpy array [int]
        Demands of each customer
    fixed_costs: numpy array [int]
        Fixed costs of operating each warehouse
    capacities: numpy array [int]
        Capacities of each warehouse
    name: str
        Name of the model
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] J.E. Beasley, An algorithm for solving large capacitated warehouse location problems,
        European Journal of Operational Research, Volume 33, Issue 3, 1988,
        Pages 314-325, ISSN 0377-2217,
        https://doi.org/10.1016/0377-2217(88)90175-0.
    """
    model = gp.Model(name)

    #model.setMinimize()
    model.setAttr("ModelSense", 1)

    customer_facility_vars = {}
    facility_vars = []

    # add customer-facility vars
    for i, j in itertools.product(range(n_customers), range(n_facilities)):
        var = model.addVar(
            lb=0, ub=1, obj=transportation_cost[i, j], name=f"x_{i}_{j}", vtype="C"
        )
        customer_facility_vars[i, j] = var

    # add facility vars
    for j in range(n_facilities):
        var = model.addVar(lb=0, ub=1, obj=fixed_costs[j], name=f"y_{j}", vtype="B")
        facility_vars.append(var)

    # add constraints

    # constraints (2)
    for i in range(n_customers):
        model.addConstr(
            gp.quicksum(customer_facility_vars[i, j] for j in range(n_facilities))
            >= 1
        )

    # constraints (3)
    for j in range(n_facilities):
        model.addConstr(
            gp.quicksum(
                demands[i] * customer_facility_vars[i, j] for i in range(n_customers)
            )
            <= capacities[j] * facility_vars[j]
        )

    # constraints (4) and (5) are skipped because no data of bounds are given in problem data in OR-Library

    # constraints (6)
    for i, j in itertools.product(range(n_customers), range(n_facilities)):
        model.addConstr(
            customer_facility_vars[i, j]
            <= min(1, capacities[j] / demands[i]) * facility_vars[j]
        )

    return model


@py_random_state("seed")
def cornuejols_instance(n_customers, n_facilities, ratio, seed=0):
    """
    Generates a Capacitated Facility Location MIP formulation following [1].
    Parameters
    ----------
    n_customers: int
        Number of customers
    n_facilities: int
        Number of facilities
    ratio: float
        Capacity / demand ratio
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    model: scip.Model
        A pyscipopt model of the generated instance
    References
    ----------
    .. [1] Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.
    """
    return capacitated_facility_location(
        n_customers,
        n_facilities,
        *cornuejols_params(n_customers, n_facilities, ratio, seed),
    )


@py_random_state("seed")
def cornuejols_params(n_customers, n_facilities, ratio, seed=0):
    """
    Generates a Capacitated Facility Location instance params following [1].
    This code is heavily based on the code available in [1] which was used in [2] and
    the generation techniques in [3].
    Parameters
    ----------
    n_customers: int
        Number of customers
    n_facilities: int
        Number of facilities
    ratio: float
        Capacity / demand ratio
    seed: integer, random_state, or None
        Indicator of random number generation state
    Returns
    -------
    trans_costs: numpy array [float]
        Matrix of transportation costs from customer i to facility j [i,j]
    demands: numpy array [int]
        Demands of each customer
    fixed_costs: numpy array [int]
        Fixed costs of operating each facility
    capacities: numpy array [int]
        Capacities of each facility
     References
    ----------
    .. [1] https://github.com/ds4dm/learn2branch
    .. [2] "Exact Combinatorial Optimization with Graph Convolutional Neural Networks" (2019)
      Maxime Gasse, Didier Chételat, Nicola Ferroni, Laurent Charlin and Andrea Lodi
      Advances in Neural Information Processing Systems 32 (2019)
    .. [3] Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.
    """
    # locations for customers
    c_x = np.array([seed.random() for _ in range(n_customers)])
    c_y = np.array([seed.random() for _ in range(n_customers)])

    # locations for facilities
    f_x = np.array([seed.random() for _ in range(n_facilities)])
    f_y = np.array([seed.random() for _ in range(n_facilities)])

    demands = np.array([seed.randint(5, 35 + 1) for _ in range(n_customers)])
    capacities = np.array([seed.randint(10, 160 + 1) for _ in range(n_facilities)])
    fixed_costs = np.array(
        [seed.randint(100, 110 + 1) for _ in range(n_facilities)]
    ) * np.sqrt(capacities) + np.array(
        [seed.randint(0, 90 + 1) for _ in range(n_facilities)]
    )
    fixed_costs = fixed_costs.astype(int)

    # adjust capacities according to ratio
    total_demand = demands.sum()
    total_capacity = capacities.sum()
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)

    # transportation cost
    trans_costs = (
        np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2
        )
        * 10
        * demands.reshape((-1, 1))
    )
    return trans_costs, demands, fixed_costs, capacities