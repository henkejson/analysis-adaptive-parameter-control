import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
from Notebooks.utils.design_matrix_creator import get_design_matricies



if __name__ == '__main__':
    # Load the CSV file
    data = pd.read_csv("single_parameter/combined_data/statistics.csv")

    # Select columns for the coverage timeline
    coverage_data = data.filter(regex='^CoverageTimeline_T')

    # Calculate the integral for each row using the trapezoidal rule
    data['IntegralValue'] = coverage_data.apply(lambda row: np.trapz(row, dx=1), axis=1)
                
    observation_matrix, module_matrix, parameter_matrix, interaction_matrix = get_design_matricies(data, 'IntegralValue')

    with pm.Model():
        # Global Intercept and standard deviation for Modules
        a_bar = pm.Normal('a_bar', mu=150, sigma=30)
        
        # Standard Deviations for modules, marameters and nteractions
        sigma_a = pm.Exponential('sigma_a', 0.03)
        sigma_b = pm.Exponential('sigma_b', 0.1)
        sigma_g = pm.Exponential('sigma_g', 0.1)
        
        # Non-centered parameterizations for module, parameter and interaction effect.
        a_offset = pm.Normal('a_offset', mu=0, sigma=1, shape=24)
        a_m = pm.Deterministic('a_m', a_bar + sigma_a * a_offset)

        b_offset = pm.Normal('b_offset', mu=0, sigma=1, shape=12)
        b_p = pm.Deterministic('b_p', sigma_b * b_offset)

        g_offset = pm.Normal('g_offset', mu=0, sigma=1, shape=288)
        g_mp = pm.Deterministic('g_mp', sigma_g * g_offset)

        
        # Activate the correct dummy variables
        identity_a = pm.math.dot(module_matrix, a_m)
        identity_b = pm.math.dot(parameter_matrix, b_p)
        identity_g = pm.math.dot(interaction_matrix, g_mp)

        # Link function (identity)
        mu = pm.Deterministic('mu', identity_a + identity_b + identity_g)
        
        # Normal distribution likelihood 
        sigma = pm.Exponential('sigma', 0.1)
        Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=observation_matrix['IntegralValue'])
        
        # Sample from the model
        trace = pm.sample(5000, chains=4, return_inferencedata=True, progressbar=True, target_accept=0.95)
        log_lik = pm.compute_log_likelihood(trace)

    print("Model building complete. Saving results...")

    az.to_netcdf(trace, "coverage_rate_model_single_parameter.nc")
    print("Results saved!")