''' 
This file requires the user install highspy (pip install highspy)
to access the HiGHs solver. HiGHs documentation can be found here: 
https://highs.dev/ or https://github.com/ERGO-Code/HiGHS 
'''

import pickle as pkl
import highspy
import time

import src
from src.reload import deep_reload

#Change this to point to your version of cbc or use another solver
solver_name='appsi_highs'

# import the model data to data_inputs.py
def load_data(filepath):        

# Load the pickled file
    with open(filepath, 'rb') as pickle_file:
        loaded_dicts_with_names = pkl.load(pickle_file)

    return loaded_dicts_with_names

model_data = load_data('src/dicts_with_names.pkl')

# Load in the set indices and the model parameters from data_inputs.py
set_inputs = src.data_inputs.extract_sets(model_data)
param_inputs = src.data_inputs.extract_params(model_data)

# run problem
t0=time.time()
problem=src.good.model_opt(set_inputs, param_inputs)

#solve problem
t1=time.time()
solution = problem.Solve(solver_name)
t2=time.time()

print(problem.model.objective(),t1-t0,t2-t1)
