from sklearn.cluster import SpectralClustering, KMeans, AffinityPropagation, AgglomerativeClustering
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def clustering_scenarios(problem, type, n_cluster, multi = True):
    if type == 'kmeans':
        clustering = KMeans(n_clusters=n_cluster, random_state=0).fit(problem.clust_vars)
    elif type == 'hierarchical':
        clustering = AgglomerativeClustering().fit(problem.clust_vars)
    elif type == 'spectral':
        clustering = SpectralClustering(n_clusters=n_cluster, assign_labels='cluster_qr', random_state=0).fit(problem.clust_vars)
    elif type == '':
        clustering = AffinityPropagation(random_state=0).fit(problem.clust_vars)
    else:
        raise ValueError("clustering type not given")

    label = clustering.predict(problem.clust_vars)
    
    label_dic = {}
    for i in range(len(label)):
        if label[i] not in label_dic.keys():
            label_dic[label[i]] = [i]
        else:
            label_dic[label[i]].append(i)
    
    n_label = int(max(label_dic.keys()) + 1)
    
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



def dropout_multicut(problem, type, n_cluster):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0

    q_list, W_list, h_list, T_list, k = clustering_scenarios(problem, type, n_cluster)

    x = MP.addMVar((problem.s1_n_var,), name = "x")
    eta = MP.addMVar((k,), name = "eta", ub = problem.eta_bounds[1], lb = problem.eta_bounds[0])

    if problem.s1_direction == GRB.MAXIMIZE:
        MP.modelSense = GRB.MAXIMIZE
    else:
        MP.modelSense = GRB.MINIMIZE

    MP.setObjective(
        problem.c @ x + np.array([1/k] * k) @ eta 
    )

    c1 = MP.addConstr(
        problem.A @ x <= problem.b
    )


    cut_found = True
    n_iters = 0
    try: 
        while cut_found:
            cut_found = False
            q_vals = []
            if n_iters < 200:
                n_iters = n_iters + 1
                print(f"On Iteration: {n_iters}")
            else:
                break
            MP.update()
            MP.optimize()
            
            x_i = x.x
            eta_i = eta.x
            LB = problem.c @ x_i

            for s in range(k):
                SP = gp.Model("SP")
                SP.Params.outputFlag = 0  # turn off output
                SP.Params.method = 1      # dual simplex
                y = SP.addMVar((problem.s2_n_var,), name = "y", obj = q_list[s])
                if problem.s2_direction == GRB.MAXIMIZE:
                    SP.modelSense = GRB.MAXIMIZE
                else:
                    SP.modelSense = GRB.MINIMIZE
                res = SP.addConstr(
                    W_list[s] @ y <= h_list[s] - (T_list[s] @ x_i)
                )
                SP.optimize()
                Q = SP.ObjVal
                q_vals.append(Q)
                pi = res.Pi
                BestLB = 0
                LB = LB + Q / k
                if np.abs(Q - eta_i[s]) > 0.00001:
                    if Q < eta_i[s]:
                        cut_found = True
                        p1 = pi @ h_list[s]
                        p2 = pi @ T_list[s]
                        if problem.s2_direction == GRB.MAXIMIZE:
                            MP.addConstr(
                                eta[s] <= p1 - gp.quicksum(p2[a] * x[a] for a in range(problem.s1_n_var))
                            )
                        else:
                            MP.addConstr(
                                eta[s] >= p1 - gp.quicksum(p2[a] * x[a] for a in range(problem.s1_n_var))
                            )
                if(LB > BestLB):
                    BestLB = LB        
            print(f"Iteration {n_iters}: LB = {BestLB}. UB = {MP.ObjVal}")
    except:
        print(f"Errored out on iteration: {n_iters} scenario {s}")



def single_cut(problem):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0

    x = MP.addMVar((problem.s1_n_var,), name = "x")
    theta = MP.addMVar(1, name = "theta", ub = problem.eta_bounds[1] * problem.k, lb = problem.eta_bounds[0] * problem.k)

    if problem.s1_direction == GRB.MAXIMIZE:
        MP.modelSense = GRB.MAXIMIZE
    else:
        MP.modelSense = GRB.MINIMIZE

    MP.setObjective(
        problem.c @ x + theta 
    )

    c1 = MP.addConstr(
        problem.A @ x <= problem.b
    )


    cut_found = True
    n_iters = 0
    try: 
        while cut_found:
            cut_found = False
            q_vals = []
            if n_iters < 200:
                n_iters = n_iters + 1
                print(f"On Iteration: {n_iters}")
            else:
                break
            MP.update()
            MP.optimize()
            LB = 0
            x_i = x.x
            theta = theta.x
            p1 = 0
            p2 = 0
            for s in range(problem.k):
                SP = gp.Model("SP")
                SP.Params.outputFlag = 0  # turn off output
                SP.Params.method = 1      # dual simplex
                y = SP.addMVar((problem.s2_n_var,), name = "y", obj = problem.q_list[s])
                if problem.s2_direction == GRB.MAXIMIZE:
                    SP.modelSense = GRB.MAXIMIZE
                else:
                    SP.modelSense = GRB.MINIMIZE
                res = SP.addConstr(
                    problem.W_list[s] @ y <= problem.h_list[s] - (problem.T_list[s] @ x_i)
                )
                SP.optimize()
                Q = SP.ObjVal
                q_vals.append(Q)
                pi = res.Pi
                p1 = p1 + pi @ problem.h_list[s] / problem.k
                p2 = p2 + pi @ problem.T_list[s] / problem.k
                LB = LB + Q / problem.k
            if np.abs(LB - theta) > 0.00001:
                if LB < theta:
                    cut_found = True
                    MP.addConstr(theta + p2 @ x <= p1)
            if(LB > BestLB):
                BestLB = LB        
            print(f"Iteration {n_iters}: LB = {BestLB}. UB = {MP.ObjVal}")
    except:
        print(f"Errored out on iteration: {n_iters} scenario {s}")



def hybrid(problem, type, n_cluster):
    MP = gp.Model("MP")
    MP.Params.outputFlag = 0
    x = MP.addMVar((problem.s1_n_var,), name = "x")

    q_dict, W_dict, h_dict, T_dict, n_label, max_range = clustering_scenarios(problem, type, n_cluster, multi = False)

    theta = MP.addMVar((n_label,), name = "theta", ub = problem.eta_bounds[1] * max_range, lb = problem.eta_bounds[0] * max_range)

    if problem.s1_direction == GRB.MAXIMIZE:
        MP.modelSense = GRB.MAXIMIZE
    else:
        MP.modelSense = GRB.MINIMIZE

    MP.setObjective(
        problem.c @ x + theta 
    )

    c1 = MP.addConstr(
        problem.A @ x <= problem.b
    )


    cut_found = True
    n_iters = 0
    try: 
        while cut_found:
            cut_found = False
            q_vals = []
            n_iters = n_iters + 1
            print(f"On Iteration: {n_iters}")
            MP.update()
            MP.optimize()
            LB = 0
            x_i = x.x
            theta_i = theta.x
            p1 = 0
            p2 = 0
            for s in range(problem.k):
                SP = gp.Model("SP")
                SP.Params.outputFlag = 0  # turn off output
                SP.Params.method = 1      # dual simplex
                y = SP.addMVar((problem.s2_n_var,), name = "y", obj = problem.q_list[s])
                if problem.s2_direction == GRB.MAXIMIZE:
                    SP.modelSense = GRB.MAXIMIZE
                else:
                    SP.modelSense = GRB.MINIMIZE
                res = SP.addConstr(
                    problem.W_list[s] @ y <= problem.h_list[s] - (problem.T_list[s] @ x_i)
                )
                SP.optimize()
                Q = SP.ObjVal
                q_vals.append(Q)
                pi = res.Pi
                p1 = p1 + pi @ problem.h_list[s] / problem.k
                p2 = p2 + pi @ problem.h_list[s] / problem.k
                LB = LB + Q / problem.k
                if np.abs(Q - theta_i) > 0.00001:
                    if Q < theta_i[s]:
                        cut_found = True
                        MP.addConstr(theta - p2 @ x <= p1)
            print(f"Iteration {n_iters}: LB = {LB}. UB = {MP.ObjVal}")
    except:
        print(f"Errored out on iteration: {n_iters} scenario {s}")
