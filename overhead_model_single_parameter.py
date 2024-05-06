import pymc as pm
import pandas as pd
import arviz as az
from Notebooks.utils.design_matrix_creator import get_design_matricies

if __name__ == '__main__':
    data = pd.read_csv("single_parameter/combined_data/statistics.csv")
    observation_matrix, module_matrix, parameter_matrix, interaction_matrix = get_design_matricies(data, 'AlgorithmIterations')

    with pm.Model():
        # Global Intercept and standard deviation for Modules
        a_bar = pm.Normal('a_bar', mu=7, sigma=0.5)
        
        # Standard Deviations for modules, parameters and interactions
        sigma_a = pm.Exponential('sigma_a', 4.0)
        sigma_b = pm.Exponential('sigma_b', 4.0)
        sigma_g = pm.Exponential('sigma_g', 4.0)
        
        # Non-centered parameterizations for module, parameter and interaction effect.
        a_offset = pm.Normal('a_offset', mu=0, sigma=1, shape=24)
        a_m = pm.Deterministic('a_m', a_bar + sigma_a * a_offset)

        b_offset = pm.Normal('b_offset', mu=0, sigma=1, shape=12)
        b_p = pm.Deterministic('b_p', sigma_b * b_offset)

        g_offset = pm.Normal('g_offset', mu=0, sigma=1, shape=288)
        g_mp = pm.Deterministic('g_mp', sigma_g * g_offset)

        
        # Activate the correct dummy variables
        log_a = pm.math.dot(module_matrix, a_m)
        log_b = pm.math.dot(parameter_matrix, b_p)
        log_g = pm.math.dot(interaction_matrix, g_mp)

        # Link function (log)
        mu = pm.Deterministic('mu', pm.math.exp(log_a + log_b + log_g))
        
        # Negative Binomial likelihood
        theta = pm.Gamma('theta', alpha=5.0, beta=0.1) # dispersion parameter
        Y_obs = pm.NegativeBinomial('Y_obs', n=theta, p=(theta)/(mu + theta), observed=observation_matrix['AlgorithmIterations'])
        
        # Sample from the model
        trace = pm.sample(5000, chains=4,return_inferencedata=True, progressbar=True, target_accept=0.95)
        log_lik = pm.compute_log_likelihood(trace)

    print("Model building complete. Saving results...")

    az.to_netcdf(trace, "overhead_model_single_parameter.nc")
    print("Results saved!")