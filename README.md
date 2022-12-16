# Clustering in Benders
Considering the uncertain variables involved in practical scenarios, stochastic programming is a method to tackle optimization problems. Even if Sample Average Approximation is applied to solve the two-stage Stochastic programming (2SP) deterministically, it is not computationally efficient to solve the problem directly. Hence, Bender’s decomposition is introduced to solve the problems, which divides the 2SP into a master problem and one subproblem for each scenario. Through passing the candidate decision into subproblems, cuts are generated to either constrain the surrogate variables within certain values or to bound the first-stage decisions depending
on feasibilty in the subproblems. In this paper, we cluster all scenarios into several groups using different clustering techniques. We compare the performance of the hybrid method and the scenario reduction method using different clustering techniques. Our results show that Affinity Propagation clustering provides performance improvements for problems with complicated uncertainty distributions. 

---
To replicate our results, there are a few main steps
1. Generating data: Unfortunately our instances are too large to store on GitHub, but its easy to recreate them!
2. Running Dropout Cut
3. Running Hybrid Cut
4. Running the Baseline

There are 2 methods to repeat the steps:

### Notebooks
1. Navigate to `instance_creation_runner.ipynb` and run all the cells
2. Navigate to `runner.ipynb` and run all the cells 

### Scripts

Run `runner_script.py`


## More information
All Benders routines can be found in `benders.py`.
The instance creation procedure can be found in `instance.py`