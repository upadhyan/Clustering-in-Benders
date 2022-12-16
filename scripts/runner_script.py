import os
import sys
import gc
from tqdm import tqdm
from utiliT.io import dump_pickle, read_pickle
import yaml

sys.path.append('../')
from benders import *
from instance import *


def generate_instances():
    generator = StochasticBinPackerGenerator()
    instance_list = generator.batch_generator()
    for instance in instance_list:
        n1 = instance.s1_n_var
        n2 = instance.s2_n_var
        k = instance.k
        dist = instance.distribution
        file_name = f"../data/{dist}_{n1}_{n2}_{k}.pkl"
        dump_pickle(instance, file_name)


def runner(function, files=None, n_seeds=3):
    bad_instances = []
    if files is None:
        files = os.listdir("../data")
    for file in tqdm(files):
        for seed in range(n_seeds):
            instance_name = file[:-4]
            instance = read_pickle(f"../data/{file}")
            try:
                result = function(instance)
                result['instance_name'] = instance_name
                result['run_number'] = seed
                file_name = f"../results/run_results/{result['instance_name']}_{result['cut_method']}_{result['grouping_method']}_" \
                            f"{result['dr']}_{seed}.pkl"
                dump_pickle(result, file_name)
                del result
                gc.collect()
            except Exception as e:
                bad_instances.append(file)
                print(file)
                print(e)


def multi_runner(functions, files=None):
    print(f"Running {len(functions)} functions")
    i = 1
    for function in functions:
        print(f"Running function: {i}")
        i = i + 1
        runner(function, files=files)
        gc.collect()


def clean_results(terms):
    for term in terms:
        files = os.listdir("../results/run_results")
        deleting = [x for x in files if term in x]
        for f in deleting:
            os.remove(f"../results/run_results/{f}")


def run_hybrid():
    functions = [
        lambda x: hybrid(x, "affinity"),
        lambda x: hybrid(x, "hierarchical"),
        lambda x: hybrid(x, "spectral"),
        lambda x: hybrid(x, "random")
    ]
    multi_runner(functions)


def run_dropout():
    functions = [
        lambda x: dropout_cut(x, "random"),
        lambda x: dropout_cut(x, "affinity"),
        lambda x: dropout_cut(x, "hierarchical"),
        lambda x: dropout_cut(x, "kmeans"),
        lambda x: dropout_cut(x, "spectral")
    ]
    multi_runner(functions)


def run_baselines():
    functions = [
        multi_cut,
        single_cut
    ]
    multi_runner(functions)


def main():
    run_dropout()
    run_hybrid()
    run_baselines()


if __name__ == "__main__":
    main()
