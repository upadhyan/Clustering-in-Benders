from sklearn.cluster import SpectralClustering, KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import time
import gc
import random


def clustering_scenarios(problem, method):
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



def dynamic_clustering(dual_var, method):
    random = [dual - min(dual) / (max(dual) - min(dual)) for dual in dual_var]
    k = len(random)
    n_clust = [int(k * x / 100) for x in range(5, 15 + 1)]
    if method == 'kmeans':
        labels = [KMeans(n_clusters=n).fit_predict(random) for n in n_clust]
        scores = [silhouette_score(random, label) for label in labels]
        # clustering = KMeans(n_clusters=n_cluster, random_state=0).fit(problem.clust_vars)
    elif method == 'hierarchical':
        labels = [AgglomerativeClustering(n_clusters=n).fit_predict(random) for n in n_clust]
        scores = [silhouette_score(random, label) for label in labels]
    elif method == 'spectral':
        labels = [
            SpectralClustering(n_clusters=n, assign_labels='cluster_qr', random_state=0).fit_predict(random)
            for n in n_clust]
        scores = [silhouette_score(random, label) for label in labels]
        # clustering = SpectralClustering(n_clusters=n_cluster, assign_labels='cluster_qr', random_state=0).fit(
        #    problem.clust_vars)
    elif method == 'affinity':
        labels = [AffinityPropagation(random_state=0).fit_predict(random)]
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

    p = []
    center_clust = []
    for key, value in label_dic.items():
        p.append(len(value) / len(dual_var))
        center = np.mean(np.array(dual_var)[value], axis=0)
        center_clust.append(random)

    return label, n_label, p, center_clust

def find_neareat(dual_var_lst, lst):
    # lst: [point, p1, p2]
    index_lst = []
    random = [dual - min(dual) / (max(dual) - min(dual)) for dual in dual_var_lst]
    for item in lst:
        distance = [np.sum((np.array(point)-np.array(item[0]))**2, axis=0) for point in random]
        index = np.argmin(distance)
        index_lst.append(index)

    return index_lst



def dropout_cut(problem, method):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0
    t1 = time.time()

    q_dict, W_dict, h_dict, T_dict, n_label, _ = clustering_scenarios(problem, method)
    p = []
    q_cluster, W_cluster, h_cluster, T_cluster = [], [], [], []
    for i in range(n_label):
        p.append(len(q_dict[i]) / problem.k)
        q_cluster.append(q_dict[i][0])
        W_cluster.append(W_dict[i][0])
        h_cluster.append(h_dict[i][0])
        T_cluster.append(T_dict[i][0])

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
        primal_gap = UB - highest_LB
        primal_gap_perc = primal_gap / UB
    t2 = time.time()
    elapsed_time = t2 - t1
    results = {
        "method": f"{method}-ND-dropout",
        "obj_val": MP.ObjVal,
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
        "distribution": problem.distribution
    }
    return results


def hybrid(problem, method):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0
    t1 = time.time()

    q_dict, W_dict, h_dict, T_dict, n_label, max_range = clustering_scenarios(problem, method)
    p = []
    for i in range(n_label):
        p.append(len(q_dict[i]) / problem.k)
    p = np.array(p)

    x = MP.addMVar((problem.s1_n_var,), name="x")
    theta = MP.addMVar((n_label,), name="theta", ub=problem.eta_bounds[1] * max_range,
                       lb=problem.eta_bounds[0] * max_range)
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
        LB = LB + np.array(sub_problem_lst) @ p
        t_bl_2 = time.time()
        BL_solve_time = BL_solve_time + t_bl_2 - t_bl_1
        if LB > highest_LB:
            highest_LB = LB
        primal_gap = (UB - highest_LB)
        primal_gap_perc = primal_gap / UB
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
        "primal_gap": primal_gap,
        "primal_gap_perc": primal_gap_perc,
        "runtime": elapsed_time,
        "n1": problem.s1_n_var,
        "n2": problem.s2_n_var,
        "m1": problem.s1_n_constr,
        "m2": problem.s2_n_constr,
        "k": problem.k,
        "distribution": problem.distribution
    }
    return results



def dynamic_dropout(problem, method):
    t1 = time.time()
    cut_found = True
    n_iters = 0
    n_cuts = 0
    MP_solve_time = 0
    BL_solve_time = 0
    status = "optimal"
    highest_LB = 0
    UB = np.abs(problem.eta_bounds[0]) * 10000
    primal_gap_perc = 1
    primal_gap = UB - highest_LB
    n_scenarios = problem.k
    p = [1 / n_scenarios] * n_scenarios
    label = np.arange(n_scenarios)
    constraint_info = [] # list of [center_cluster, p1, p2]
    while cut_found:
        MP = gp.Model("MP")
        MP.Params.outputFlag = 0
        x = MP.addMVar((problem.s1_n_var,), name="x")
        eta = MP.addMVar((n_scenarios,), name="eta", ub=problem.eta_bounds[1], lb=problem.eta_bounds[0])
        eta_list = np.zeros(problem.k)
        if problem.s1_direction == GRB.MAXIMIZE:
            MP.modelSense = GRB.MAXIMIZE
        else:
            MP.modelSense = GRB.MINIMIZE


        MP.setObjective(
            problem.c @ x + np.array(p) @ eta 
        )

        c1 = MP.addConstr(
            problem.A @ x <= problem.b
        )
        
        if len(constraint_info) != 0:
            index_lst = find_neareat(pi_cluster, constraint_info)
            for i in range(len(index_lst)):
                p1 = constraint_info[i][1]
                p2 = constraint_info[i][2]
                MP.addConstr(
                    eta[index_lst[i]] <= p1 - gp.quicksum(p2[a] * x[a] for a in range(problem.s1_n_var))
                        )

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
        for i in range(n_scenarios):
            index = np.where(np.array(label) == i)
            for j in index:
                eta_list[j] = eta_i[i]
        t_bl_1 = time.time()
        dual_var = []
        Q_lst = []
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
            dual_var.append(pi)
            Q_lst.append(Q)
            LB = LB + Q / problem.k
        # cluster based on pi
        label, n_scenarios, p, center_clust = dynamic_clustering(dual_var, method)
        h_cluster = []
        T_cluster = []
        pi_cluster = []
        Q_cluster = []
        eta_cluster = []
        for i in range(n_scenarios):
            index = random.choice(np.where(np.array(label) == i)[0])
            h_cluster.append(problem.h_list[index])
            T_cluster.append(problem.W_list[index])
            pi_cluster.append(dual_var[index])
            Q_cluster.append(Q_lst[index])
            eta_cluster.append(eta_list[index])
        for s_clust in range(n_scenarios):
            if np.abs(Q_cluster[s_clust] - eta_cluster[s_clust]) > 0.00001:
                if Q_cluster[s_clust] < eta_cluster[s_clust]:
                    cut_found = True
                    p1 = pi_cluster[s_clust] @ h_cluster[s_clust]
                    p2 = pi_cluster[s_clust] @ T_cluster[s_clust]
                    constraint_info.append([center_clust[s_clust], p1, p2])
                    n_cuts = n_cuts + 1

        t_bl_2 = time.time()
        BL_solve_time = BL_solve_time + t_bl_2 - t_bl_1
        if LB > highest_LB:
            highest_LB = LB
        primal_gap = (UB - highest_LB)
        primal_gap_perc = primal_gap / UB
    t2 = time.time()
    elapsed_time = t2 - t1
    results = {
        "method": "dynamic-cut",
        "obj_val": MP.ObjVal,
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
        "distribution": problem.distribution
    }
    return results