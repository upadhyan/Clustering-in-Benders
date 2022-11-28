import sys

sys.path.append('../')
from GeCo_Instance_generators.packing import tang_instance_packing

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from pandas_datareader import data, wb
from datetime import date
import random
from tqdm.notebook import trange


class Instance:
    def __init__(self, A, b, c, q_list, W_list, h_list, T_list, clust_vars,
                 s1_direction, s2_direction, eta_bounds, distribution, other_info=None):
        self.A = A
        self.b = b
        self.c = c
        self.q_list = q_list
        self.W_list = W_list
        self.h_list = h_list
        self.T_list = T_list
        self.clust_vars = clust_vars
        self.s1_n_var = len(c)
        self.s2_n_var = len(q_list[0])
        self.s1_n_constr = A.shape[0]
        self.s2_n_constr = h_list[0].shape[0]
        self.k = len(q_list)
        self.s1_direction = s1_direction
        self.s2_direction = s2_direction
        self.other_info = other_info
        self.eta_bounds = eta_bounds
        self.distribution = distribution


class StochasticBinPackerGenerator:
    def __init__(self):
        pass

    def generate_uniform_stochastics(self, n1, n2, m2):
        h_vals = np.random.uniform(9 * n2, 10 * n2, m2)
        T_vals = np.random.uniform(5, 30, (m2, n1))
        cluster_vals = np.sum(T_vals, axis=1) + h_vals
        T_vals = T_vals * -1
        return h_vals, T_vals, cluster_vals

    def generate_normal_stochastic(self, n1, n2, m2):
        h_vals = np.random.normal(9.5 * n2, n2, m2)
        h_vals = np.clip(h_vals, 0, None)
        T_vals = np.random.normal(17, 5, (m2, n1))
        T_vals = np.clip(T_vals, 0, None)
        cluster_vals = np.sum(T_vals, axis=1) + h_vals
        T_vals = T_vals * -1
        return h_vals, T_vals, cluster_vals

    def generate_gamma_stochastic(self, n1, n2, m2):
        h_vals = np.random.gamma(9.5, n2, m2)
        h_vals = np.clip(h_vals, 0, None)
        shape = np.random.uniform(5, 30)
        scale = 17 / shape
        T_vals = np.random.gamma(shape, scale, (m2, n1))
        T_vals = np.clip(T_vals, 0, None)
        cluster_vals = np.sum(T_vals, axis=1) + h_vals
        T_vals = T_vals * -1
        return h_vals, T_vals, cluster_vals

    def generate_multi_peak(self, n1, n2, m2):
        ratio_h = int(m2 * 0.5)
        h_vals = np.concatenate((np.random.normal(7 * n2, n2, ratio_h),
                                 np.random.normal(12 * n2, n2, m2 - ratio_h)))
        np.random.shuffle(h_vals)
        h_vals = np.clip(h_vals, 0, None)
        t_size = m2 * n1
        ratio_t = int(t_size * 0.5)
        T_vals = np.concatenate((np.random.normal(10, 5, ratio_t),
                                 np.random.normal(24, 5, t_size - ratio_t)))
        T_vals = np.clip(T_vals, 0, None)
        T_vals = np.reshape(T_vals, (m2, n1))
        cluster_vals = np.sum(T_vals, axis=1) + h_vals
        T_vals = T_vals * -1
        return h_vals, T_vals, cluster_vals

    def generate_problem(self, n1, n2, m1, m2, k, distribution):
        s1_problem = tang_instance_packing(n1, m1)
        s1_problem.update()
        s2_problem = tang_instance_packing(n2, m2)
        s2_problem.update()
        s2_problem = s2_problem.relax()
        A1 = s1_problem.getA().toarray().astype(np.float32)
        A2 = s2_problem.getA().toarray().astype(np.float32)

        c1 = np.array(s1_problem.getAttr('Obj', s1_problem.getVars()))
        c2 = np.array(s2_problem.getAttr('Obj', s2_problem.getVars()))
        bounds = (np.sum(c2) * -5000, np.sum(c2) * 5000)
        b1 = np.array(s1_problem.getAttr('RHS', s1_problem.getConstrs()))
        b2 = np.array(s2_problem.getAttr('RHS', s2_problem.getConstrs()))

        other_info = {
            'original_c2': c2,
            'original_A2': A2,
            'original_b2': b2
        }
        q_list = [None] * k
        W_list = [None] * k
        h_list = [None] * k
        T_list = [None] * k
        clust_vars = [None] * k
        for i in range(k):
            W_list[i] = A2
            q_list[i] = c2
            if distribution == "multipeak":
                h_xi, T_xi, c_vars = self.generate_multi_peak(n1, n2, m2)
            elif distribution == "normal":
                h_xi, T_xi, c_vars = self.generate_normal_stochastic(n1, n2, m2)
            elif distribution == "gamma":
                h_xi, T_xi, c_vars = self.generate_gamma_stochastic(n1, n2, m2)
            else:
                h_xi, T_xi, c_vars = self.generate_uniform_stochastics(n1, n2, m2)
            h = b2 + h_xi
            h_list[i] = h
            T_list[i] = T_xi
            clust_vars[i] = c_vars

        clust_vars = np.stack(clust_vars)
        return Instance(A1, b1, c1,
                        q_list, W_list, h_list, T_list, clust_vars,
                        s1_problem.modelSense, s2_problem.modelSense, bounds, distribution=distribution,
                        other_info=other_info)

    def batch_generator(self):
        n_instances = 3 * 3 * 3 * 4
        print(f"Generating {n_instances} instances")
        instance_list = [None] * n_instances
        count = 0
        for dist in ["multipeak","normal","uniform"]:  # 3
            for n1 in [100, 150, 200]:  # 50 100, 150, 200
                for n2 in [100, 150, 200]:  # 50, 100, 150, 200
                    for k in [100,200,300,400]:  # 100, 200, 300 ,400
                        instance_list[count] = self.generate_problem(n1, n2, n1 * 2, n2 * 2, k, dist)
                        count = count + 1
        return instance_list
