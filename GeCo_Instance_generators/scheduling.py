import itertools
import math
from networkx.utils import py_random_state

import gurobipy as gp

@py_random_state("seed")
def hooker_params(number_of_facilities, number_of_tasks, seed=0):
    """Generates late tasks mip instance described in section 4 in [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    seed: int, random object or None
        for randomization
    Returns
    -------
    processing_times: dict[int,int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[int,int]
        cost of assigning a task to a facility
    release_times: list[int]
        time step at which a job is released
    deadlines: dict[int, float]
        deadline (time step) to finish a job
    References
    ----------
    .. [1] Hooker, John. (2005). Planning and Scheduling to Minimize
     Tardiness. 314-327. 10.1007/11564751_25.
    """
    return generate_params(number_of_facilities, number_of_tasks, seed)[:-1]


@py_random_state("seed")
def hooker_instance(number_of_facilities, number_of_tasks, time_steps, seed=0):
    """Generates late tasks mip instance described in section 4 in [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    time_steps:
        the number of time steps starting from 0 (corresponds to "N" in the paper)
    Returns
    -------
        model: SCIP model of the late tasks instance
    References
    ----------
    .. [1] Hooker, John. (2005). Planning and Scheduling to Minimize
     Tardiness. 314-327. 10.1007/11564751_25.
    """
    return late_tasks_formulation(
        number_of_facilities,
        number_of_tasks,
        time_steps,
        *hooker_params(number_of_facilities, number_of_tasks, seed),
        name="Hooker Scheduling Instance",
    )


def generate_hookers_instances():
    number_of_tasks = [10 + 2 * i for i in range(7)]
    time_steps = [10, 100]
    seeds = range(10)
    for n, t, seed in itertools.product(number_of_tasks, time_steps, seeds):
        params = 3, n, t, seed
        yield params, hooker_instance(*params)


def _common_hooker_params(number_of_facilities, number_of_tasks, seed):
    capacities = [10] * number_of_facilities
    resource_requirements = {}
    for i in range(number_of_tasks):
        cur_res_requirement = seed.randrange(1, 10)
        for j in range(number_of_facilities):
            resource_requirements[i, j] = cur_res_requirement
    return capacities, resource_requirements


@py_random_state("seed")
def c_instance_params(seed=0):
    for m, n in itertools.product(range(2, 4 + 1), range(10, 38 + 1, 2)):
        yield c_params_generator(m, n, seed)


@py_random_state("seed")
def c_params_generator(number_of_facilities, number_of_tasks, seed=0):
    """
    Generate instance parameters for the c problem set mentioned in [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    seed: int, random object or None
        for randomization
    Returns
    -------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    processing_times: dict[(int,int),int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[(int,int),int]
        cost of assigning a task to a facility
    release_times: list[int]
        time step at which a job is released
    deadlines: dict[int, int]
        deadline (time step) to finish a job
    resource_requirements: dict[(int,int),int]
        resources required for each task assigned to a facility
    References
    ----------
    ..[1] http://public.tepper.cmu.edu/jnh/instances.htm
    """
    capacities, resource_requirements = _common_hooker_params(
        number_of_facilities, number_of_tasks, seed
    )

    release_dates = [0] * number_of_tasks
    due_dates = [
        _due_date_helper(1 / 3, number_of_facilities, number_of_tasks)
    ] * number_of_tasks

    processing_times = {}
    for i in range(number_of_facilities):
        for j in range(number_of_tasks):
            processing_times[j, i] = seed.randrange(i + 1, 10 * (i + 1))

    processing_costs = {}
    for i in range(number_of_facilities):
        for j in range(number_of_tasks):
            processing_costs[j, i] = seed.randrange(
                2 * (number_of_facilities - i), 20 * (number_of_facilities - i)
            )

    return (
        number_of_facilities,
        number_of_tasks,
        processing_times,
        capacities,
        processing_costs,
        release_dates,
        due_dates,
        resource_requirements,
    )


@py_random_state("seed")
def e_instance_params(seed=0):
    for m in range(2, 10 + 1):
        yield e_params_generator(m, 5 * m, seed)


@py_random_state("seed")
def e_params_generator(number_of_facilities, number_of_tasks, seed=0):
    """
    Generate instance parameters for the e problem set mentioned in [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    seed: int, random object or None
        for randomization
    Returns
    -------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    processing_times: dict[(int,int),int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[(int,int),int]
        cost of assigning a task to a facility
    release_times: list[int]
        time step at which a job is released
    deadlines: dict[int, int]
        deadline (time step) to finish a job
    resource_requirements: dict[(int,int),int]
        resources required for each task assigned to a facility
    References
    ----------
    ..[1] http://public.tepper.cmu.edu/jnh/instances.htm
    """
    capacities, resource_requirements = _common_hooker_params(
        number_of_facilities, number_of_tasks, seed
    )

    release_dates = [0] * number_of_tasks
    due_dates = [33] * number_of_tasks

    processing_times = {}
    for i in range(number_of_facilities):
        for j in range(number_of_tasks):
            processing_times[j, i] = seed.randrange(
                2, int(25 - i * (10 / (number_of_facilities - 1)))
            )

    processing_costs = {}
    for i in range(number_of_facilities):
        for j in range(number_of_tasks):
            processing_costs[j, i] = seed.randrange(
                math.floor(400 / (25 - i * (10 / (number_of_facilities - 1)))),
                math.ceil(800 / (25 - i * (10 / (number_of_facilities - 1)))),
            )

    return (
        number_of_facilities,
        number_of_tasks,
        processing_times,
        capacities,
        processing_costs,
        release_dates,
        due_dates,
        resource_requirements,
    )


@py_random_state("seed")
def de_instance_params(seed=0):
    for n in range(14, 28 + 1, 2):
        yield de_params_generator(3, n, seed)


@py_random_state("seed")
def de_params_generator(number_of_facilities, number_of_tasks, seed=0):
    """
    Generate instance parameters for the de problem set mentioned in [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    seed: int, random object or None
        for randomization
    Returns
    -------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    processing_times: dict[(int,int),int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[(int,int),int]
        cost of assigning a task to a facility
    release_times: list[int]
        time step at which a job is released
    deadlines: dict[int, int]
        deadline (time step) to finish a job
    resource_requirements: dict[(int,int),int]
        resources required for each task assigned to a facility
    References
    ----------
    ..[1] http://public.tepper.cmu.edu/jnh/instances.htm
    """
    capacities, resource_requirements = _common_hooker_params(
        number_of_facilities, number_of_tasks, seed
    )

    release_dates = [0] * number_of_tasks
    due_dates = [
        seed.randrange(
            _due_date_helper((1 / 4) * (1 / 3), number_of_facilities, number_of_tasks),
            _due_date_helper(1 / 3, number_of_facilities, number_of_tasks),
        )
        for _ in range(number_of_tasks)
    ]

    processing_times = {}
    range_start = 2 if number_of_facilities <= 20 else 5  # P1 in the reference website
    for i in range(number_of_facilities):
        for j in range(number_of_tasks):
            processing_times[j, i] = seed.randrange(range_start, 30 - i * 5)

    processing_costs = {}
    for i in range(number_of_facilities):
        for j in range(number_of_tasks):
            processing_costs[j, i] = seed.randrange(10 + 10 * i, 40 + 10 * i)

    return (
        number_of_facilities,
        number_of_tasks,
        processing_times,
        capacities,
        processing_costs,
        release_dates,
        due_dates,
        resource_requirements,
    )


@py_random_state("seed")
def df_instance_params(seed=0):
    for n in range(14, 28 + 1, 2):
        yield df_params_generator(3, n, seed)


@py_random_state("seed")
def df_params_generator(number_of_facilities, number_of_tasks, seed=0):
    """
    Generate instance parameters for the df problem set mentioned in [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    seed: int, random object or None
        for randomization
    Returns
    -------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    processing_times: dict[(int,int),int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[(int,int),int]
        cost of assigning a task to a facility
    release_times: list[int]
        time step at which a job is released
    deadlines: dict[int, int]
        deadline (time step) to finish a job
    resource_requirements: dict[(int,int),int]
        resources required for each task assigned to a facility
    References
    ----------
    ..[1] http://public.tepper.cmu.edu/jnh/instances.htm
    """
    capacities, resource_requirements = _common_hooker_params(
        number_of_facilities, number_of_tasks, seed
    )

    release_dates = [0] * number_of_tasks

    random_release_time = seed.choice(release_dates)
    due_dates = [
        seed.randrange(
            random_release_time
            + _due_date_helper(1 / 4 * 1 / 2, number_of_facilities, number_of_tasks),
            random_release_time
            + _due_date_helper(1 / 2, number_of_facilities, number_of_tasks),
        )
        for _ in range(number_of_tasks)
    ]

    processing_times = {}
    range_start = 2 if number_of_facilities <= 20 else 5  # P1 in the reference website
    for i in range(number_of_facilities):
        for j in range(number_of_tasks):
            processing_times[j, i] = seed.randrange(range_start, 30 - i * 5)

    processing_costs = {}
    for i in range(number_of_facilities):
        for j in range(number_of_tasks):
            processing_costs[j, i] = seed.randrange(10 + 10 * i, 40 + 10 * i)

    return (
        number_of_facilities,
        number_of_tasks,
        processing_times,
        capacities,
        processing_costs,
        release_dates,
        due_dates,
        resource_requirements,
    )


def _due_date_helper(a, number_of_facilities, number_of_tasks):
    return math.ceil(
        5 * a * number_of_tasks * (number_of_facilities + 1) / number_of_facilities
    )

def late_tasks_formulation(
    number_of_facilities,
    number_of_tasks,
    time_steps,
    processing_times,
    capacities,
    assignment_costs,
    release_dates,
    deadlines,
    name="Hooker Scheduling Late Tasks Formulation",
):
    """Generates late tasks mip formulation described in section 4 in [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    time_steps:
        the number of time steps starting from 0 (corresponds to "N" in the paper)
    processing_times: dict[int,int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[int,int]
        cost of assigning a task to a facility
    release_dates: list[int]
        time step at which a job is released
    deadlines: dict[int, float]
        deadline (time step) to finish a job
    name: str
        assigned name to generated instance
    Returns
    -------
        model: SCIP model of the late tasks instance
    References
    ----------
    .. [1] Hooker, John. (2005). Planning and Scheduling to Minimize
     Tardiness. 314-327. 10.1007/11564751_25.
    """
    model = gp.Model(name)

    start_time = min(release_dates)
    time_steps = range(start_time, start_time + time_steps)

    # add variables and their cost
    L = []
    for i in range(number_of_tasks):
        var = model.addVar(lb=0, ub=1, obj=1, name=f"L_{i}", vtype="B")
        L.append(var)

    # assignment vars
    x = {}
    for j, i, t in itertools.product(
        range(number_of_tasks), range(number_of_facilities), time_steps
    ):
        var = model.addVar(lb=0, ub=1, obj=0, name=f"x_{j}_{i}_{t}", vtype="B")
        x[j, i, t] = var

    # add constraints
    # constraint (a)
    for j, t in itertools.product(range(number_of_tasks), time_steps):
        model.addConstr(
            len(time_steps) * L[j]
            >= gp.quicksum(
                (
                    (t + processing_times[j, i]) * x[j, i, t] - deadlines[j]
                    for i in range(number_of_facilities)
                )
            )
        )

    # constraint (b)
    for j in range(number_of_tasks):
        vars = (
            x[j, i, t]
            for i, t in itertools.product(range(number_of_facilities), time_steps)
        )
        model.addConstr(gp.quicksum(vars) == 1)

    # constraint (c)
    for i, t in itertools.product(range(number_of_facilities), time_steps):
        vars = []
        for j in range(number_of_tasks):
            vars += [
                assignment_costs[j, i] * x[j, i, t_prime]
                for t_prime in range(t - processing_times[j, i] + 1, t + 1)
                if (j, i, t_prime) in x
            ]
        model.addConstr(gp.quicksum(vars) <= capacities[i])

    # constraint (d)
    for i, j, t in itertools.product(
        range(number_of_facilities), range(number_of_tasks), time_steps
    ):
        if t < release_dates[j] or t > len(time_steps) - processing_times[j, i]:
            model.addConstr(x[j, i, t] == 0)

    #model.setMinimize()

    model.setAttr("ModelSense", 1)



    return model


def heinz_formulation(
    number_of_facilities,
    number_of_tasks,
    processing_times,
    capacities,
    assignment_costs,
    release_dates,
    deadlines,
    resource_requirements,
    name="Heinz Scheduling Formulation",
):
    """Generates scheduling MIP formulation according to Model 4 in [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    processing_times: dict[(int,int),int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[(int,int),int]
        cost of assigning a task to a facility
    release_dates: list[int]
        time step at which a job is released
    deadlines: dict[int, float]
        deadline (time step) to finish a job
    resource_requirements: dict[(int,int),int]
        resources required for each task assigned to a facility
    name: str
        assigned name to generated instance
    Returns
    -------
        model: SCIP model of generated instance
    References
    ----------
    .. [1] Heinz, J. (2013). Recent Improvements Using Constraint Integer Programming for Resource Allocation and Scheduling.
    In Integration of AI and OR Techniques in Constraint Programming for Combinatorial Optimization Problems
    (pp. 12–27). Springer Berlin Heidelberg.
    """
    model = gp.Model(name)

    time_steps = range(min(release_dates), int(max(deadlines)))

    # objective function
    x = {}
    for j, k in itertools.product(range(number_of_tasks), range(number_of_facilities)):
        var = model.addVar(
            lb=0, ub=1, obj=assignment_costs[j, k], name=f"x_{j}_{k}", vtype="B"
        )
        x[j, k] = var

    # y vars
    y = {}
    for j, k, t in itertools.product(
        range(number_of_tasks), range(number_of_facilities), time_steps
    ):
        if release_dates[j] <= t <= deadlines[j] - processing_times[j, k]:
            var = model.addVar(lb=0, ub=1, obj=0, name=f"y_{j}_{k}_{t}", vtype="B")
            y[j, k, t] = var

    # add constraints
    # constraint (12)
    for j in range(number_of_tasks):
        model.addConstr(gp.quicksum(x[j, k] for k in range(number_of_facilities)) == 1)

    # constraint (13)
    for j, k in itertools.product(range(number_of_tasks), range(number_of_facilities)):
        model.addConstr(
            gp.quicksum(
                y[j, k, t]
                for t in range(
                    release_dates[j], int(deadlines[j]) - processing_times[j, k]
                )
                if t < len(time_steps)
            )
            == x[j, k]
        )

    # constraint (14)
    for k, t in itertools.product(range(number_of_facilities), time_steps):
        model.addConstr(
            gp.quicksum(
                resource_requirements[j, k] * y[j, k, t_prime]
                for j in range(number_of_tasks)
                for t_prime in range(t - processing_times[j, k], t + 1)
                if (j, k, t_prime) in y
            )
            <= capacities[k]
        )

    # constraint (15)
    epsilon = filter(
        lambda ts: ts[0] < ts[1], itertools.product(release_dates, deadlines)
    )
    for k, (t1, t2) in itertools.product(range(number_of_facilities), epsilon):
        model.addConstr(
            gp.quicksum(
                processing_times[j, k] * resource_requirements[j, k] * x[j, k]
                for j in range(number_of_tasks)
                if t1 <= release_dates[j] and t2 >= deadlines[j]
            )
            <= capacities[k] * (t2 - t1)
        )

    return model


def hooker_cost_formulation(
    number_of_facilities,
    number_of_tasks,
    processing_times,
    capacities,
    assignment_costs,
    release_dates,
    deadlines,
    resource_requirements,
    name="Hooker Cost Scheduling Formulation",
):
    """Generates scheduling MIP formulation according to [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    processing_times: dict[(int,int),int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[(int,int),int]
        cost of assigning a task to a facility
    release_dates: list[int]
        time step at which a job is released
    deadlines: dict[int, float]
        deadline (time step) to finish a job
    resource_requirements: dict[(int,int),int]
        resources required for each task assigned to a facility
    name: str
        assigned name to generated instance
    Returns
    -------
        model: SCIP model of generated instance
    References
    ----------
    .. [1] J. N. Hooker, A hybrid method for planning and scheduling, CP 2004.
    """
    model = gp.Model(name)

    time_steps = range(min(release_dates), int(max(deadlines)))

    # objective function
    x = {}
    for j, i, t in itertools.product(
        range(number_of_tasks), range(number_of_facilities), time_steps
    ):
        var = model.addVar(
            lb=0, ub=1, obj=assignment_costs[j, i], name=f"x_{j}_{i}_{t}", vtype="B"
        )
        x[j, i, t] = var

    # add constraints
    # constraints (a)
    for j in range(number_of_tasks):
        model.addCons(
            gp.quicksum(
                x[j, i, t]
                for i, t in itertools.product(range(number_of_facilities), time_steps)
            )
            == 1
        )

    # constraints (b)
    for i, t in itertools.product(range(number_of_facilities), time_steps):
        model.addConstr(
            gp.quicksum(
                resource_requirements[j, i] * x[j, i, t] for j in range(number_of_tasks)
            )
            <= capacities[i]
        )

    # constraints (c)
    for j, i, t in itertools.product(
        range(number_of_tasks), range(number_of_facilities), time_steps
    ):
        if (
            deadlines[j] - processing_times[j, i] < t < release_dates[j]
            or t > number_of_tasks - processing_times[j, i]
        ):
            model.addCons(x[j, i, t] == 0)

    return model


@py_random_state("seed")
def generate_params(number_of_facilities, number_of_tasks, seed=0):
    """
    Generic instance parameter generator for heinz [1] and hooker [2] formulations.
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    seed: int, random object or None
        for randomization
    Returns
    -------
    processing_times: dict[int,int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[int,int]
        cost of assigning a task to a facility
    release_times: list[int]
        time step at which a job is released
    deadlines: dict[int, float]
        deadline (time step) to finish a job
    resource_requirements: dict[int,int]
        resources required for each task assigned to a facility
    References
    ----------
    .. [1] Heinz, J. (2013). Recent Improvements Using Constraint Integer Programming for Resource Allocation and Scheduling.
    In Integration of AI and OR Techniques in Constraint Programming for Combinatorial Optimization Problems
    (pp. 12–27). Springer Berlin Heidelberg.
    .. [2] Hooker, John. (2005). Planning and Scheduling to Minimize
     Tardiness. 314-327. 10.1007/11564751_25.
    """
    processing_times = {}

    for j, i in itertools.product(range(number_of_tasks), range(number_of_facilities)):
        if number_of_tasks < 22:
            processing_times[j, i] = seed.randint(2, 20 + 5 * i)
        else:
            processing_times[j, i] = seed.randint(5, 20 + 5 * i)

    capacities = [10] * number_of_facilities

    assignment_costs = {}
    for j in range(number_of_tasks):
        value = seed.randint(1, 10)
        for i in range(number_of_facilities):
            assignment_costs[j, i] = value

    release_times = [0] * number_of_tasks

    beta = 20 / 9
    deadlines = [
        seed.uniform(beta * number_of_tasks / 4, beta * number_of_tasks)
        for _ in range(number_of_tasks)
    ]

    resource_requirements = {}
    for j, k in itertools.product(range(number_of_tasks), range(number_of_facilities)):
        resource_requirements[j, k] = seed.randint(1, 9)

    return (
        processing_times,
        capacities,
        assignment_costs,
        release_times,
        deadlines,
        resource_requirements,
    )




@py_random_state("seed")
def heinz_params(number_of_facilities, number_of_tasks, seed=0):
    """Generates scheduling MIP instance params according to [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    seed: int, random object or None
        for randomization
    Returns
    -------
    processing_times: dict[(int,int),int]
        time steps to process each task
    capacities: list[int]
        capacity of each facility
    assignment_costs: dict[(int,int),int]
        cost of assigning a task to a facility
    release_times: list[int]
        time step at which a job is released
    deadlines: dict[int,int]
        deadline (time step) to finish a job
    resource_requirements: dict[(int,int),int]
        resources required for each task assigned to a facility
    References
    ----------
    .. [1] Heinz, J. (2013). Recent Improvements Using Constraint Integer Programming for Resource Allocation and Scheduling.
    In Integration of AI and OR Techniques in Constraint Programming for Combinatorial Optimization Problems
    (pp. 12–27). Springer Berlin Heidelberg.
    """
    return generate_params(number_of_facilities, number_of_tasks, seed)


@py_random_state("seed")
def heinz_instance(number_of_facilities, number_of_tasks, seed=0):
    """Generates scheduling MIP instance according to [1].
    Parameters
    ----------
    number_of_facilities: int
        the number of facilities to schedule on
    number_of_tasks: int
        the number of tasks to assign to facilities
    seed: int, random object or None
        for randomization
    Returns
    -------
        model: SCIP model of generated instance
    References
    ----------
    .. [1] Heinz, J. (2013). Recent Improvements Using Constraint Integer Programming for Resource Allocation and Scheduling.
    In Integration of AI and OR Techniques in Constraint Programming for Combinatorial Optimization Problems
    (pp. 12–27). Springer Berlin Heidelberg.
    """
    return heinz_formulation(
        number_of_facilities,
        number_of_tasks,
        *heinz_params(number_of_facilities, number_of_tasks, seed),
        name="Heinz Scheduling Instance",
    )