from sklearn.cluster import SpectralClustering, KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import gc


def clustering_scenarios(problem, method, multi=True):
    k = problem.k
    n_clust = [int(k * x / 100) for x in range(5, 15 + 1)]
    if method == 'kmeans':
        labels = [KMeans(n_clusters=n).fit_predict(problem.clust_vars) for n in n_clust]
        scores = [silhouette_score(problem.clust_vars, label) for label in labels]
        # clustering = KMeans(n_clusters=n_cluster, random_state=0).fit(problem.clust_vars)
    elif method == 'hierarchical':
        labels = [AgglomerativeClustering(n_clusters=n).fit_predict(problem.clust_vars) for n in n_clust]
        scores = [silhouette_score(problem.clust_vars, label) for label in labels]
    elif method == 'spectral':
        labels = [
            SpectralClustering(n_clusters=n, assign_labels='cluster_qr', random_state=0).fit_predict(problem.clust_vars)
            for n in n_clust]
        scores = [silhouette_score(problem.clust_vars, label) for label in labels]
        # clustering = SpectralClustering(n_clusters=n_cluster, assign_labels='cluster_qr', random_state=0).fit(
        #    problem.clust_vars)
    elif method == 'affinity':
        labels = [AffinityPropagation(random_state=0).fit_predict(problem.clust_vars)]
        scores = [12]
        # clustering = AffinityPropagation(random_state=0).fit(problem.clust_vars)
    else:
        raise ValueError("clustering type not given")

    label = labels[np.argmin(np.argmin(scores))]

    label_dic = {}
    for i in range(len(label)):
        if label[i] not in label_dic.keys():
            label_dic[label[i]] = [i]
        else:
            label_dic[label[i]].append(i)

    n_label = int(max(label_dic.keys()) + 1)
    
    q_dict = {}
    W_dict = {}
    h_dict = {}
    T_dict = {}
    max_range = 0
    for key, value in label_dic.items():
        if max_range < len(value) + 1:
            max_range = len(value) + 1
        q_dict[key] = np.array(problem.q_list)[value]
        W_dict[key] = np.array(problem.W_list)[value]
        h_dict[key] = np.array(problem.h_list)[value]
        T_dict[key] = np.array(problem.T_list)[value]
    return q_dict, W_dict, h_dict, T_dict, n_label, max_range

    if multi:
        q_list = []
        W_list = []
        h_list = []
        T_list = []
        for key, value in label_dic.items():
            index = value[0]
            q_list.append(problem.q_list[index])
            W_list.append(problem.W_list[index])
            h_list.append(problem.h_list[index])
            T_list.append(problem.T_list[index])

        return q_list, W_list, h_list, T_list, n_label

    else:
        q_dict = {}
        W_dict = {}
        h_dict = {}
        T_dict = {}
        max_range = 0
        for key, value in label_dic.items():
            if max_range < len(value) + 1:
                max_range = len(value) + 1
            q_dict[key] = np.array(problem.q_list)[value]
            W_dict[key] = np.array(problem.W_list)[value]
            h_dict[key] = np.array(problem.h_list)[value]
            T_dict[key] = np.array(problem.T_list)[value]
        return q_dict, W_dict, h_dict, T_dict, n_label, max_range


def dropout_cut(problem, type, n_cluster):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0

    q_dict, W_dict, h_dict, T_dict, n_label, _ = clustering_scenarios(problem, type, n_cluster)
    p = []
    q_cluster, W_cluster, h_cluster, T_cluster = [], [], [], []
    for i in range(n_label):
        p.append([len(q_dict[i])])
        q_cluster.append(q_dict[i][0])
        W_cluster.append(W_dict[i][0])
        h_cluster.append(h_dict[i][0])
        T_cluster.append(T_dict[i][0])


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
    t1 = time.time()
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
            SP = gp.Model("SP")
            SP.Params.outputFlag = 0  # turn off output
            SP.Params.method = 1  # dual simplex
            y = SP.addMVar((problem.s2_n_var,), name="y", obj=q_cluster[s])
            SP.modelSense = problem.s2_direction
            res = SP.addConstr(
                W_cluster[s] @ y <= h_cluster[s] - (T_cluster[s] @ x_i)
            )
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
    t2 = time.time()
    elapsed_time = t2 - t1
    results = {
        "method": "dropout-cut",
        "obj_val": MP.ObjVal,
        "n_cuts": n_cuts,
        "n_iterations": n_iters,
        "avg_mp_solve": MP_solve_time / n_iters,
        "avg_benders_loop_solve": BL_solve_time / n_iters,
        "status": status,
        "primal_gap": UB - highest_LB,
        "primal_gap_perc": (UB - LB) / UB,
        "runtime": elapsed_time,
        "n1": problem.s1_n_var,
        "n2": problem.s2_n_var,
        "m1": problem.s1_n_constr,
        "m2": problem.s2_n_constr,
        "k": problem.k,
        "distribution": problem.distribution
    }
    return results


def hybrid(problem, type, n_cluster):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0

    q_dict, W_dict, h_dict, T_dict, n_label, max_range = clustering_scenarios(problem, type, n_cluster, multi=False)

    x = MP.addMVar((problem.s1_n_var,), name="x")
    theta = MP.addMVar((n_label,), name="theta", ub=problem.eta_bounds[1] * max_range,
                       lb=problem.eta_bounds[0] * max_range)
    if problem.s1_direction == GRB.MAXIMIZE:
        MP.modelSense = GRB.MAXIMIZE
    else:
        MP.modelSense = GRB.MINIMIZE

    MP.setObjective(
        problem.c @ x + np.array([1 / n_label] * n_label) @ theta
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

    while cut_found:
        gc.collect()
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
                SP = gp.Model("SP")
                SP.Params.outputFlag = 0  # turn off output
                SP.Params.method = 1  # dual simplex
                y = SP.addMVar((problem.s2_n_var,), name="y", obj=q_list[s])
                SP.modelSense = problem.s2_direction
                res = SP.addConstr(
                    W_list[s] @ y <= h_list[s] - (T_list[s] @ x_i)
                )
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
        LB = LB + sum(sub_problem_lst)
        t_bl_2 = time.time()
        BL_solve_time = BL_solve_time + t_bl_2 - t_bl_1
        if LB > highest_LB:
            highest_LB = LB
    t2 = time.time()
    elapsed_time = t2 - t1
    results = {
        "method": "hybrid-cut",
        "obj_val": MP.ObjVal,
        "n_cuts": n_cuts,
        "n_iterations": n_iters,
        "avg_mp_solve": MP_solve_time / n_iters,
        "avg_benders_loop_solve": BL_solve_time / n_iters,
        "status": status,
        "primal_gap": UB - LB,
        "primal_gap_perc": (UB - LB) / UB,
        "runtime": elapsed_time,
        "n1": problem.s1_n_var,
        "n2": problem.s2_n_var,
        "m1": problem.s1_n_constr,
        "m2": problem.s2_n_constr,
        "k": problem.k,
        "distribution": problem.distribution
    }
    return results

