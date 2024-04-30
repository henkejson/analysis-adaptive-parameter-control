import pymc as pm
import numpy as np
import arviz as az


def logit_to_probability(logit_vals):
    """Apply logistic transformation to logit values."""
    return 1 / (1 + np.exp(-logit_vals))

def log_to_exp(log_values):
    return np.exp(log_values)

def create_inference_data_from_trace(trace,  variable_list, transformations=None):
    """
    Create a summary from a PyMC trace.
    
    Parameters:
    - trace: PyMC trace object.
    - variable_list: List of variable names to include in the summary.
    - transformations: Optional dictionary mapping variable names to functions that will be applied to transform these variables.
    - hdi: 
    """

    posterior_dict = {}
    if transformations is None:
        transformations = {}
    
    # Extract variables and apply transformations if specified
    for variable in variable_list:
        data = trace.posterior[variable].values
        if variable in transformations:
            # Apply the transformation function if specified
            data = transformations[variable](data)
        posterior_dict[variable] = data

    # Create an InferenceData object
    return az.from_dict(posterior=posterior_dict)













