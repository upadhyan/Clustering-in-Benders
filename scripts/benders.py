import gc
from sklearn.cluster import SpectralClustering, KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
from sklearn.decomposition import PCA
from scipy.spatial import distance


def evaluate_solution(problem, x):
    obj_val = problem.c @ x
    p = [1 / problem.k] * problem.k
    for s in range(problem.k):
        SP = gp.Model("SP")
        SP.Params.outputFlag = 0  # turn off output
        SP.Params.method = 1  # dual simplex
        y = SP.addMVar((problem.s2_n_var,), name="y", obj=problem.q_list[s])
        SP.modelSense = problem.s2_direction
        res = SP.addConstr(
            problem.W_list[s] @ y <= problem.h_list[s] - (problem.T_list[s] @ x)
        )
        SP.optimize()
        Q = SP.ObjVal
        obj_val = obj_val + p[s] * Q
    return obj_val


def multi_cut(problem):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0
    x = MP.addMVar((problem.s1_n_var,), name="x")
    eta = MP.addMVar((problem.k,), name="eta", ub=problem.eta_bounds[1], lb=problem.eta_bounds[0])
    MP.modelSense = problem.s1_direction
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
    status = "optimal"
    highest_LB = 0
    UB = np.abs(problem.eta_bounds[0]) * 10000
    t1 = time.time()
    LB = 0
    primal_gap_perc = 1
    primal_gap = UB - LB
    SP = gp.Model("SP")
    SP.Params.outputFlag = 0  # turn off output
    SP.Params.method = 1  # dual simplex
    y = SP.addMVar((problem.s2_n_var,), name="y", obj=problem.q_list[0])
    SP.modelSense = problem.s2_direction
    x_i = np.zeros(problem.s1_n_var)
    res = SP.addConstr(
        problem.W_list[0] @ y <= problem.h_list[0] - (problem.T_list[0] @ x_i)
    )
    while cut_found:
        gc.collect()
        curr_time = time.time()
        if curr_time - t1 >= 300:
            status = "timelimit"
            break
        if primal_gap_perc < 0.000001:
            status = "optimal"
            primal_gap = 0
            primal_gap_perc = 0
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
            RHS = problem.h_list[s] - (problem.T_list[s] @ x_i)
            res.rhs = RHS
            SP.update()
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
        primal_gap = UB - highest_LB
        primal_gap_perc = primal_gap / UB
    t2 = time.time()
    elapsed_time = t2 - t1
    results = {
        "grouping_method": f"None",
        "obj_val": evaluate_solution(problem, x.x),
        "dr":"None",
        "cut_method": "multi",
        "n_cuts": n_cuts,
        "n_iterations": n_iters,
        "avg_mp_solve": MP_solve_time / n_iters,
        "avg_benders_loop_solve": BL_solve_time / n_iters,
        "status": status,
        "primal_gap": primal_gap,
        "primal_gap_perc": primal_gap_perc,
        "runtime": elapsed_time,
        "n1": problem.s1_n_var,
        "n2": problem.s2_n_var,
        "m1": problem.s1_n_constr,
        "m2": problem.s2_n_constr,
        "k": problem.k,
        "distribution": problem.distribution,
        "clustering_runtime": 0
    }
    return results


def single_cut(problem):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0
    x = MP.addMVar((problem.s1_n_var,), name="x")
    theta = MP.addMVar((1,), name="eta", ub=problem.eta_bounds[1], lb=problem.eta_bounds[0])
    MP.modelSense = problem.s1_direction
    MP.setObjective(
        problem.c @ x + theta
    )

    c1 = MP.addConstr(
        problem.A @ x <= problem.b
    )
    cut_found = True
    n_iters = 0
    n_cuts = 0
    MP_solve_time = 0
    BL_solve_time = 0
    status = "optimal"
    highest_LB = 0
    UB = np.abs(problem.eta_bounds[1] * problem.k)
    t1 = time.time()
    p = [1 / problem.k] * problem.k
    LB = 0
    primal_gap = UB - highest_LB
    primal_gap_perc = primal_gap / UB

    SP = gp.Model("SP")
    SP.Params.outputFlag = 0  # turn off output
    SP.Params.method = 1  # dual simplex
    y = SP.addMVar((problem.s2_n_var,), name="y", obj=problem.q_list[0])
    SP.modelSense = problem.s2_direction
    res = SP.addConstr(
        problem.W_list[0] @ y <= np.zeros(problem.s2_n_constr)
    )
    while cut_found:
        gc.collect()
        curr_time = time.time()
        if curr_time - t1 >= 300:
            status = "timelimit"
            break
        if primal_gap_perc < 0.000001:
            status = "optimal"
            primal_gap = 0
            primal_gap_perc = 0
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
        theta_i = theta.x
        t_bl_1 = time.time()
        p1 = 0
        p2 = np.zeros(problem.s1_n_var)
        sub_problem = 0
        for s in range(problem.k):
            RHS = problem.h_list[s] - (problem.T_list[s] @ x_i)
            res.rhs = RHS
            SP.optimize()
            Q = SP.ObjVal
            pi_k = res.Pi
            sub_problem = sub_problem + Q / problem.k
            p1 = p1 + pi_k @ problem.h_list[s] * p[s]
            p2 = p2 + pi_k @ problem.T_list[s] * p[s]
        if np.abs(sub_problem - theta_i) > 0.00001:
            if sub_problem < theta_i:
                cut_found = True
                MP.addConstr(
                    theta + p2 @ x <= p1
                )
                n_cuts = n_cuts + 1
        LB = LB + sub_problem
        t_bl_2 = time.time()
        BL_solve_time = BL_solve_time + t_bl_2 - t_bl_1
        if LB > highest_LB:
            highest_LB = LB
        primal_gap = UB - highest_LB
        primal_gap_perc = primal_gap / UB
    t2 = time.time()
    elapsed_time = t2 - t1
    results = {
        "grouping_method": f"None",
        "obj_val": evaluate_solution(problem, x.x),
        "dr":"None",
        "cut_method": "single",
        "n_cuts": n_cuts,
        "n_iterations": n_iters,
        "avg_mp_solve": MP_solve_time / n_iters,
        "avg_benders_loop_solve": BL_solve_time / n_iters,
        "status": status,
        "primal_gap": primal_gap,
        "primal_gap_perc": primal_gap_perc,
        "runtime": elapsed_time,
        "n1": problem.s1_n_var,
        "n2": problem.s2_n_var,
        "m1": problem.s1_n_constr,
        "m2": problem.s2_n_constr,
        "k": problem.k,
        "distribution": problem.distribution,
        "clustering_runtime": 0
    }
    return results


def clustering_scenarios(problem, method, dr=True):
    k = problem.k
    if dr:
        cvar = PCA(n_components=10).fit_transform(problem.clust_vars)
    else:
        cvar = problem.clust_vars
    n_clust = [int(k * x / 100) for x in range(2, 21)]
    if method == 'kmeans':
        labels = [KMeans(n_clusters=n).fit_predict(cvar) for n in n_clust]
        scores = [silhouette_score(cvar, label) for label in labels]
    elif method == 'hierarchical':
        labels = [AgglomerativeClustering(n_clusters=n).fit_predict(cvar) for n in n_clust]
        scores = [silhouette_score(cvar, label) for label in labels]
    elif method == 'spectral':
        labels = [
            SpectralClustering(n_clusters=n, assign_labels='kmeans', random_state=0).fit_predict(cvar)
            for n in n_clust]
        scores = [silhouette_score(cvar, label) for label in labels]
    elif method == 'affinity':
        labels = [AffinityPropagation(max_iter=1000, damping=0.9).fit_predict(cvar)]
        scores = [12]
    elif method == 'random':
        n_clust = int((np.random.randint(2, 21) * k / 100))
        # n_clust = int(5 * k / 100)
        missing = True
        label_set = np.random.randint(0, n_clust, problem.k)
        while missing:
            missing = False
            for i in range(n_clust):
                if int(i) not in label_set:
                    missing = True
            if missing:
                label_set = np.random.randint(0, n_clust, problem.k)
        labels = [label_set]
        scores = [12]
    else:
        raise ValueError("clustering type not given")
    label = labels[np.argmax(np.argmin(scores))]
    unique_labels = np.unique(label)
    label_dic = {}
    for l in unique_labels:
        label_dic[l] = []

    for i, label in enumerate(label):
        label_dic[label].append(i)
    n_label = int(max(label_dic.keys()) + 1)

    q_dict = {}
    W_dict = {}
    h_dict = {}
    T_dict = {}
    representative_scenarios = {}
    max_range = 0
    for key, value in label_dic.items():
        q_dict[key] = np.array(problem.q_list)[value]
        W_dict[key] = np.array(problem.W_list)[value]
        h_dict[key] = np.array(problem.h_list)[value]
        T_dict[key] = np.array(problem.T_list)[value]
        rel_cvar = np.array(cvar)[value]
        col_means = np.mean(rel_cvar, axis=0)
        dist_vector = [0] * len(value)
        for i in range(len(value)):
            dist_vector[i] = np.linalg.norm(rel_cvar[i] - col_means)
        representative_scenarios[key] = np.argmin(dist_vector)
    return q_dict, W_dict, h_dict, T_dict, n_label, representative_scenarios


def dropout_cut(problem, method, dr=False):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0
    t1 = time.time()
    q_dict, W_dict, h_dict, T_dict, n_label, representative_scenarios = clustering_scenarios(problem, method, dr=dr)
    tclust = time.time()
    clustering_runtime = tclust - t1
    p = []
    q_cluster, W_cluster, h_cluster, T_cluster = [], [], [], []
    for i in range(n_label):
        r = representative_scenarios[i]
        p.append(len(q_dict[i]) / problem.k)
        q_cluster.append(q_dict[i][r])
        W_cluster.append(W_dict[i][r])
        h_cluster.append(h_dict[i][r])
        T_cluster.append(T_dict[i][r])

    p = np.array(p)
    q_cluster = np.array(q_cluster)
    W_cluster = np.array(W_cluster)
    h_cluster = np.array(h_cluster)
    T_cluster = np.array(T_cluster)

    x = MP.addMVar((problem.s1_n_var,), name="x")
    eta = MP.addMVar((n_label,), name="eta", ub=problem.eta_bounds[1], lb=problem.eta_bounds[0])
    if problem.s1_direction == GRB.MAXIMIZE:
        MP.modelSense = GRB.MAXIMIZE
    else:
        MP.modelSense = GRB.MINIMIZE

    MP.setObjective(
        problem.c @ x + p @ eta
    )

    c1 = MP.addConstr(
        problem.A @ x <= problem.b
    )
    cut_found = True
    n_iters = 0
    n_cuts = 0
    MP_solve_time = 0
    BL_solve_time = 0
    status = "optimal"
    highest_LB = 0
    UB = np.abs(problem.eta_bounds[0]) * 10000
    primal_gap = UB - highest_LB
    primal_gap_perc = primal_gap / UB
    SP = gp.Model("SP")
    SP.Params.outputFlag = 0  # turn off output
    SP.Params.method = 1  # dual simplex
    y = SP.addMVar((problem.s2_n_var,), name="y", obj=q_cluster[0])
    SP.modelSense = problem.s2_direction
    res = SP.addConstr(
        W_cluster[0] @ y <= np.zeros(problem.s2_n_constr)
    )
    while cut_found:
        gc.collect()
        curr_time = time.time()
        if curr_time - t1 >= 300:
            status = "timelimit"
            break
        if primal_gap_perc < 0.000001:
            status = "optimal"
            primal_gap = 0
            primal_gap_perc = 0
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
        for s in range(n_label):
            RHS = h_cluster[s] - (T_cluster[s] @ x_i)
            res.rhs = RHS
            SP.update()
            SP.optimize()
            Q = SP.ObjVal
            pi = res.Pi
            LB = LB + Q * p[s]
            if np.abs(Q - eta_i[s]) > 0.00001:
                if Q < eta_i[s]:
                    cut_found = True
                    p1 = pi @ h_cluster[s]
                    p2 = pi @ T_cluster[s]
                    MP.addConstr(
                        eta[s] <= p1 - gp.quicksum(p2[a] * x[a] for a in range(problem.s1_n_var))
                    )
                    n_cuts = n_cuts + 1
        t_bl_2 = time.time()
        BL_solve_time = BL_solve_time + t_bl_2 - t_bl_1
        if LB > highest_LB:
            highest_LB = LB
        primal_gap = UB - highest_LB
        primal_gap_perc = primal_gap / UB
    t2 = time.time()
    elapsed_time = t2 - t1
    results = {
        "grouping_method": f"{method}",
        "obj_val": evaluate_solution(problem, x.x),
        "dr":dr,
        "cut_method": "dropout",
        "n_cuts": n_cuts,
        "n_iterations": n_iters,
        "avg_mp_solve": MP_solve_time / n_iters,
        "avg_benders_loop_solve": BL_solve_time / n_iters,
        "status": status,
        "primal_gap": primal_gap,
        "primal_gap_perc": primal_gap_perc,
        "runtime": elapsed_time,
        "n1": problem.s1_n_var,
        "n2": problem.s2_n_var,
        "m1": problem.s1_n_constr,
        "m2": problem.s2_n_constr,
        "k": problem.k,
        "distribution": problem.distribution,
        "clustering_runtime": clustering_runtime
    }
    return results


def hybrid(problem, method, dr=False):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0
    t1 = time.time()
    q_dict, W_dict, h_dict, T_dict, n_label, max_range = clustering_scenarios(problem, method, dr=dr)
    t_clust = time.time()
    clustering_runtime = t_clust - t1
    p = []
    for i in range(n_label):
        p.append(len(q_dict[i]) / problem.k)
    p = np.array(p)

    x = MP.addMVar((problem.s1_n_var,), name="x")
    theta = MP.addMVar((n_label,), name="theta", ub=problem.eta_bounds[1],
                       lb=problem.eta_bounds[0])
    if problem.s1_direction == GRB.MAXIMIZE:
        MP.modelSense = GRB.MAXIMIZE
    else:
        MP.modelSense = GRB.MINIMIZE

    MP.setObjective(
        problem.c @ x + p @ theta
    )

    c1 = MP.addConstr(
        problem.A @ x <= problem.b
    )
    cut_found = True
    n_iters = 0
    n_cuts = 0
    MP_solve_time = 0
    BL_solve_time = 0
    status = "optimal"
    highest_LB = 0
    UB = np.abs(problem.eta_bounds[0]) * 10000
    primal_gap = UB - highest_LB
    primal_gap_perc = primal_gap / UB
    SP = gp.Model("SP")
    SP.Params.outputFlag = 0  # turn off output
    SP.Params.method = 1  # dual simplex
    y = SP.addMVar((problem.s2_n_var,), name="y", obj=problem.q_list[0])
    SP.modelSense = problem.s2_direction
    res = SP.addConstr(
        problem.W_list[0] @ y <= np.zeros(problem.s2_n_constr)
    )
    while cut_found:
        gc.collect()
        curr_time = time.time()
        if curr_time - t1 >= 300:
            status = "timelimit"
            break
        if primal_gap_perc < 0.000001:
            status = "optimal"
            primal_gap = 0
            primal_gap_perc = 0
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
        theta_i = theta.x
        t_bl_1 = time.time()
        sub_problem_lst = []
        for i in range(n_label):
            q_list, W_list, h_list, T_list = q_dict[i], W_dict[i], h_dict[i], T_dict[i]
            n_scenarios = len(q_list)
            p1 = 0
            p2 = 0
            sub_problem = 0
            for s in range(n_scenarios):
                RHS = h_list[s] - (T_list[s] @ x_i)
                res.rhs = RHS
                SP.optimize()
                Q = SP.ObjVal
                pi_k = res.Pi
                sub_problem = sub_problem + Q / n_scenarios
                p1 = p1 + pi_k @ h_list[s] / n_scenarios
                p2 = p2 + pi_k @ T_list[s] / n_scenarios
            sub_problem_lst.append(sub_problem)
            if np.abs(sub_problem - theta_i[i]) > 0.00001:
                if sub_problem < theta_i[i]:
                    cut_found = True
                    MP.addConstr(
                        theta[i] <= p1 - gp.quicksum(p2[a] * x[a] for a in range(problem.s1_n_var))
                    )
                    n_cuts = n_cuts + 1
        LB = LB + np.array(sub_problem_lst) @ p
        t_bl_2 = time.time()
        BL_solve_time = BL_solve_time + t_bl_2 - t_bl_1
        if LB > highest_LB:
            highest_LB = LB
        primal_gap = UB - highest_LB
        primal_gap_perc = primal_gap / UB
    t2 = time.time()
    elapsed_time = t2 - t1
    results = {
        "grouping_method": f"{method}",
        "obj_val": evaluate_solution(problem, x.x),
        "dr":dr,
        "cut_method": "hybrid",
        "n_cuts": n_cuts,
        "n_iterations": n_iters,
        "avg_mp_solve": MP_solve_time / n_iters,
        "avg_benders_loop_solve": BL_solve_time / n_iters,
        "status": status,
        "primal_gap": primal_gap,
        "primal_gap_perc": primal_gap_perc,
        "runtime": elapsed_time,
        "n1": problem.s1_n_var,
        "n2": problem.s2_n_var,
        "m1": problem.s1_n_constr,
        "m2": problem.s2_n_constr,
        "k": problem.k,
        "distribution": problem.distribution,
        "clustering_runtime": clustering_runtime
    }
    return results
