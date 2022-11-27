import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from utiliT.io import dump_yaml

def multi_cut_bender(problem):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0
    x = MP.addMVar((problem.s1_n_var,), name="x")
    eta = MP.addMVar((problem.k,), name="eta", ub=problem.eta_bounds[1], lb=problem.eta_bounds[0])
    if problem.s1_direction == GRB.MAXIMIZE:
        MP.modelSense = GRB.MAXIMIZE
    else:
        MP.modelSense = GRB.MINIMIZE

    MP.setObjective(
        problem.c @ x + np.array([1 / problem.k] * problem.k) @ eta
    )

    c1 = MP.addConstr(
        problem.A @ x <= problem.b
    )
    cut_found = True
    n_iters = 0
    n_cuts = 0
    MP_solve_time = 0
    BL_solve_time = 0
    t1 = time.time()
    status = "optimal"
    highest_LB = 0
    UB = problem.eta_bounds[0] * 10000
    t1 = time.time()
    while cut_found:
        curr_time = time.time()
        if curr_time - t1 >= 300:
            status = "timelimit"
            break
        n_iters = n_iters + 1
        cut_found = False
        MP.update()
        t_mp_1 = time.time()
        MP.optimize()
        t_mp_2 = time.time()
        MP_solve_time = MP_solve_time + t_mp_2 - t_mp_1
        UB = MP.ObjVal
        x_i = x.x
        LB = problem.c @ x_i
        eta_i = eta.x
        t_bl_1 = time.time()
        for s in range(problem.k):
            SP = gp.Model("SP")
            SP.Params.outputFlag = 0  # turn off output
            SP.Params.method = 1  # dual simplex
            y = SP.addMVar((problem.s2_n_var,), name="y", obj=problem.q_list[s])
            SP.modelSense = problem.s2_direction
            res = SP.addConstr(
                problem.W_list[s] @ y <= problem.h_list[s] - (problem.T_list[s] @ x_i)
            )
            SP.optimize()
            Q = SP.ObjVal
            pi = res.Pi
            LB = LB + Q / problem.k
            if np.abs(Q - eta_i[s]) > 0.00001:
                if Q < eta_i[s]:
                    cut_found = True
                    p1 = pi @ problem.h_list[s]
                    p2 = pi @ problem.T_list[s]
                    MP.addConstr(
                        eta[s] <= p1 - gp.quicksum(p2[a] * x[a] for a in range(problem.s1_n_var))
                    )
                    n_cuts = n_cuts + 1
        t_bl_2 = time.time()
        BL_solve_time = BL_solve_time + t_bl_2 - t_bl_1
        if LB > highest_LB:
            highest_LB = LB
    t2 = time.time()
    elapsed_time = t2 - t1
    results = {
        "method": "multi-cut",
        "n_cuts": n_cuts,
        "n_iterations": n_iters,
        "avg_mp_solve": MP_solve_time / n_iters,
        "avg_benders_loop_solve": BL_solve_time / n_iters,
        "status": status,
        "primal_gap": highest_LB - UB,
        "primal_gap_perc": (UB - highest_LB) / UB,
        "runtime": elapsed_time,
        "n1": problem.s1_n_var,
        "n2": problem.s2_n_var,
        "m1":problem.s1_n_constr,
        "m2": problem.s2_n_constr,
        "k": problem.k,
        "distribution": problem.distribution
    }
    return results