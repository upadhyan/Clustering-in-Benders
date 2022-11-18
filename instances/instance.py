import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from pandas_datareader import data,wb
from datetime import date
import random
from tqdm import trange
class Instance: 
    def __init__(self, A,b,c,q_list, W_list, h_list, T_list, clust_vars, eta_bound, s1_direction, s2_direction, other_info = None):
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
        self.eta_bound = eta_bound
        self.s1_direction = s1_direction
        self.s2_direction = s2_direction
        self.other_info = other_info

class InvestmentGenerator:
    def __init__(self):
        print("Start")
        symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].to_list()
        n = len(symbols)
        data_list = [None] *n 
        startdate = pd.to_datetime('2010-01-01')
        enddate = pd.to_datetime(date.today())
        col_names = []
        for i in trange(n):
            try:
                data_list[i] = data.DataReader(symbols[i], 'yahoo', startdate, enddate)['Open']
                col_names.append(symbols[i])
            except:
                pass
        final_data = pd.concat(data_list, axis = 1)
        final_data.columns = col_names
        final_data = final_data.interpolate()
        final_data.to_csv('stock_data.csv')
        # https://medium.com/wealthy-bytes/5-lines-of-python-to-automate-getting-the-s-p-500-95a632e5e567

        
        self.stocks = list(col_names)
        self.price_data = final_data
        self.means = final_data.pct_change(periods = 120).mean()
        self.stds = final_data.pct_change(periods = 120).std()
        self.max_stocks = len(col_names)
    def __init__(self, csv):
        final_data = pd.read_csv(csv,index_col='Date')
        self.stocks = list(final_data.columns)
        self.price_data = final_data
        self.means = final_data.pct_change(periods = 120).mean()
        self.stds = final_data.pct_change(periods = 120).std()
        self.max_stocks = len(self.stocks)

    def create_instance(self, n, k):
        n = max(10, n)
        n = min(self.max_stocks, n)
        k = max(k, 0)
        contains_nan = True
        while contains_nan:
            chosen_stocks = random.sample(self.stocks, n)
            relevant_prices = self.price_data[chosen_stocks]
            relevant_means = self.means.loc[chosen_stocks]
            relevant_stds = self.stds.loc[chosen_stocks]
            price_series = relevant_prices.sample().iloc[0]
            contains_nan = price_series.isna().any()
        
        initial_prices = np.round(price_series.to_numpy(), 2)
        
        m = np.round(initial_prices / np.random.uniform(2,10), 2)

        C = np.sum(initial_prices) * np.random.uniform(5,10)

        I_c = np.random.uniform(50,70) / 100 
        
        I_u = np.random.uniform(10,30) / 100 

        R_l = np.mean(self.means) # R_l: Beat the S&P

        R_c =  R_l / np.random.uniform(1,3)

        c = np.zeros(n) # no initial cost
        A1 = m
        b1 = [I_c * C] # Sum of the margin is less than our allowed investment
        A2 = (np.identity(n) - np.ones((n,n)) * I_u) * m
        b2 = [0] * n
        A = np.vstack([A1, A2])
        b = np.array(b1 + b2)

        cluster_vars = [None] * k
        pct_changes = [None] * k
        W_list = [None] * k
        h_list = [None] * k
        T_list = [None] * k
        q_list = [None] * k
        for s in range(k):
            pct_changes = np.random.normal(relevant_means, relevant_stds, n)
            end_price = (pct_changes + 1) * initial_prices
            return_p = np.round(end_price,2) - initial_prices 
            q_s = np.array([1] + [1] * n + [-1] * n + [-1])
            
            w_1 = np.array([1] + [0] * n + [0] * n  + [0])
            h_1 = [R_c * C]
            t_1 = m * C

            w_2 = np.hstack([np.zeros((n,1)), np.ones((n,n)), -1 * np.ones((n,n)), np.zeros((n,1))])
            h_2 = [0] * n
            t_2 = -1 * np.ones((n,n)) * return_p

            w_3 = np.array([0] + [-1] * n + [1] * n + [-2])
            h_3 = [0]
            t_3 = m * R_l * -1

            W_s = np.vstack([w_1,w_2,w_3])
            h_s = np.array(h_1 + h_2 + h_3)
            T_s = np.vstack([t_1, t_2, t_3])
            
            q_list[s] = q_s
            W_list[s] = W_s
            h_list[s] = h_s
            T_list[s] = T_s
            cluster_vars[s] = return_p
        other_info = {
            'initial_prices':initial_prices,
            'I_c':I_c,
            'I_u': I_u,
            'R_l':R_l,
            'R_c':R_c,
            'capital':C,
            'pct_changes': pct_changes 
        }
        return Instance(A,b,c,q_list, W_list, h_list, T_list, np.vstack(cluster_vars), C * 20, GRB.MAXIMIZE, GRB.MAXIMIZE, other_info=other_info)

