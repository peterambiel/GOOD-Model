import pickle as pkl



import src
from src.reload import deep_reload

#Change this to point to your version of cbc or use another solver
solver_kwargs={'_name':'cbc','executable':'src/cbc'}

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
problem=src.good.model_opt(set_inputs, param_inputs)
