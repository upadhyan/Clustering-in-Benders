import sys
sys.path.append('../')
from GeCo_Instance_generators.packing import tang_instance_packing

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from pandas_datareader import data,wb
from datetime import date
import random
from tqdm import trange


class Instance:
    def __init__(self, A,b,c,q_list, W_list, h_list, T_list, clust_vars,
                 s1_direction, s2_direction, eta_bounds, distribution, other_info = None):
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

    def generate_uniform_instance(self, n1, n2, m1, m2, k, max_penalty = 3):
        s1_problem = tang_instance_packing(n1, m1)
        s1_problem.update()
        s2_problem = tang_instance_packing(n2, m2)
        s2_problem.update()
        print(s1_problem.modelSense)
        print(GRB.MAXIMIZE)
        s2_problem = s2_problem.relax()
        A1 = s1_problem.getA().toarray().astype(np.float32)
        A2 = s2_problem.getA().toarray().astype(np.float32)

        c1 = np.array(s1_problem.getAttr('Obj', s1_problem.getVars()))
        c2 = np.array(s1_problem.getAttr('Obj', s1_problem.getVars())) / 2
        bounds = (np.sum(c2) * -500, np.sum(c2) * 500)
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
            h_list[i] = b2 + np.random.uniform(9 * n2, 10 * n2, )
            T = np.random.uniform(5, 30, (m2, n1)) * -1
            clust_vars[i] = np.sum(T, axis = 1)
            T_list[i] = T

            q_list[i] = c2
        clust_vars = np.stack(clust_vars)
        return Instance(A1, b1, c1,
                        q_list, W_list, h_list, T_list, clust_vars,
                        s1_problem.modelSense, s2_problem.modelSense, bounds, distribution ="uniform", other_info = other_info)

