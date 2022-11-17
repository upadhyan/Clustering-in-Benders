import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from pandas_datareader import data,wb
from datetime import date
import random
from tqdm import trange
class Instance: 
    def __init__(self, A,b,c,q_list, W_list, h_list, T_list, clust_vars, eta_bound, direction):
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
        self.direction = direction

class InvestmentGenerator:
    def __init__(self):
        print("Start")
        symbols = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].to_list()
        n = len(symbols)
        data_list = [None] *n 
        startdate = pd.to_datetime('2015-07-15')
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
        self.means = final_data.pct_change(periods = 7).mean()
        self.stds = final_data.pct_change(periods = 7).std()
        self.max_stocks = len(col_names)
    def __init__(self, csv):
        final_data = pd.read_csv(csv,index_col='Date')
        self.stocks = list(final_data.columns)
        self.price_data = final_data
        self.means = final_data.pct_change(periods = 7).mean()
        self.stds = final_data.pct_change(periods = 7).std()
        self.max_stocks = len(self.stocks)

    def create_instance(self, n, k):
        n = max(10, n)
        n = min(self.max_stocks, n)
        k = max(k, 0)

        chosen_stocks = random.sample(self.stocks, n)
        relevant_prices = self.price_data[chosen_stocks]
        relevant_means = self.means.loc[chosen_stocks]
        relevant_stds = self.stds.loc[chosen_stocks]


        initial_prices = np.round(relevant_prices.sample().iloc[0].to_numpy(), 2)
        margin = np.round(initial_prices / np.random.uniform(2,10), 2)
        capital = np.sum(initial_prices) * np.random.uniform(3,7)
        max_spend = (60  + np.random.uniform(-20,20)) / 100 # max spend %
        single_asset_cap = np.random.uniform(3,n-1) / n  # I_u: Percent
        capital_return =  (6 + np.random.uniform(-3,3))/100 #R_c: Capital Return percent
        minimum_return = np.mean(self.means) # R_l: Beat the S&P


        return_prices = np.array([None] * k)
        for i in range(k):
            pct_changes = np.random.normal(relevant_means, relevant_stds, n)
            end_price = (pct_changes + 1) * initial_prices
            return_prices[i] = initial_prices - np.round(end_price,2)
        return_prices = np.stack(return_prices)

        c = np.zeros(n)
        A1 = margin
        b1 = [max_spend * capital]
        A2 = np.identity(n) * margin - np.ones((n,n)) * single_asset_cap * margin
        b2 = [0] * n
        A = np.vstack([A1, A2])
        b = np.array(b1 + b2)

        q = [1] + [1] * n + [-1] * n + [-1]
        q = np.array(q)
        q_list = [q] * k

        W_list = [None] * k
        h_list = [None] * k
        T_list = [None] * k

        for i in range(k):
            W_s = np.zeros((n + 2, 2 * n + 2))
            h_s = np.zeros(n+2)
            T_s = np.zeros((n+2, n))

            # constraint 1 -------------------
            W_s[0,:] = [1] + [0] * (2 * n + 1)

            h_s[0] = capital * capital_return
            T_s[0,:] = -1 * capital_return * margin

            # constraint 2 -------------------
            W_s[1:(n+1),1:(n+1)] = np.identity(n) 
            W_s[1:(n+1), (n+1):(2*n+1)] = -1 * np.identity(n)


            T_s[1:(n+1),:] = np.identity(n) * return_prices[i,:]

            # constraint 3 -------------------
            W_s[(n+1),:] = np.array([0] + [1] * n + [-1] * n + [2]) * -1
            T_s[(n+1),:] = margin * minimum_return * -1
            W_list[i] = W_s
            h_list[i] = h_s
            T_list[i] = T_s
        return Instance(A,b,c,q_list, W_list, h_list, T_list, return_prices, capital * 20,GRB.MAXIMIZE)

