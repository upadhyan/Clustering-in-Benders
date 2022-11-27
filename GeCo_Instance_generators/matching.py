'''
Websites used:

    -
    -
    -

'''
from fileinput import filename
from gurobipy import GRB
import gurobipy as gp
import tsplib95
import sys
import os
sys.path.insert(1, '/Users/arnauddeza/Documents/2022_l2c_summer/ip')

from gurobi_utils import is_integer
def get_files(dir):
    all_files = []
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            filepath = subdir + os.sep + file
            if filepath.endswith(".tsp"):
                all_files.append(filepath)
    return all_files
all_tsps = get_files('/Users/arnauddeza/Documents/2022_l2c_summer/data/TSPLIB')
for files_ in all_tsps:
    problem = tsplib95.load(files_)
    filename = files_.split('TSPLIB/')[1].split('.tsp')[0]
    graph = problem.get_graph()
    edges , nodes = graph.edges , graph.nodes
    num_vars = len(edges)
    num_constraints = len(nodes)

    if num_vars > 6000:
        print("skipped")
    if num_vars < 6000:


        print("Creating matching problem")
        matching = gp.Model("Matching")

        # add variables and their cost
        vars = []
        for i, edge in enumerate(edges):
            cost = graph.get_edge_data(edge[0],edge[1])['weight']
            var = matching.addVar(lb=0,ub=1, obj=cost, name=f"v_{i}", vtype="I")
            vars.append(var)

        # add constraints
        for i ,node in enumerate(nodes):
            adjacent_edges = graph.edges(node)
            adjacent_edges_index = [j for j,edge in enumerate(edges) if edge in adjacent_edges]
            
            matching.addConstr(gp.quicksum(vars[k] for k in adjacent_edges_index) == 2)


        matching.setAttr("ModelSense", 1)
        matching.Params.OutputFlag = 0
        matching.update()
    
        lp = matching.relax()

        matching.optimize()

        if matching.status == GRB.OPTIMAL:
            print('Optimal objective: %g' % matching.objVal)
            lp.optimize()
        if matching.status == GRB.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
            sys.exit(0)
        elif matching.status == GRB.INFEASIBLE:
            print('Model is infeasible')
            sys.exit(0)
        elif matching.status == GRB.UNBOUNDED:
            print('Model is unbounded')
            sys.exit(0)

        ip_solution = [var.X for var in matching.getVars()]
        lp_solution = [var.X for var in lp.getVars()]
        lp_objval = lp.ObjVal
        ip_objval = matching.ObjVal

        print("\n \n \n \n ")
        print("n,m          {}      {}".format(matching.getAttr('NumVars'),matching.getAttr('NumConstrs')))
        print(ip_objval)
        print(lp_objval)
        print(is_integer(lp_solution))
        print(set(lp_solution))

        matching.write("/Users/arnauddeza/Documents/2022_l2c_summer/data/matching/{}_matching_n__m__{}__{}_is_int_{}.mps".format(filename,matching.getAttr('NumVars'),matching.getAttr('NumConstrs'),is_integer(lp_solution)))
        






