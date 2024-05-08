import formulaic
import pymc as pm
import pandas as pd
import arviz as az
from Notebooks.utils.design_matrix_creator import get_design_matricies

if __name__ == '__main__':
    data = pd.read_csv("multi_parameter/combined_data/statistics.csv")

    observation_matrix, module_matrix, parameter_matrix, _ = get_design_matricies(data, 'Coverage')

    with pm.Model() as model:
        # Global Intercept and standard deviation for Modules
        a_bar = pm.Normal('a_bar', mu=0, sigma=1.5)
        
        # Standard Deviations for modules, marameters and nteractions
        sigma_a = pm.Exponential('sigma_a', 2.0)
        sigma_b = pm.Exponential('sigma_b', 4.0)

        # Non-centered parameterizations for module, parameter and interaction effect.
        a_offset = pm.Normal('a_offset', mu=0, sigma=1, shape=24)
        a_m = pm.Deterministic('a_m', a_bar + sigma_a * a_offset)

        b_offset = pm.Normal('b_offset', mu=0, sigma=1, shape=66)
        b_p = pm.Deterministic('b_p', sigma_b * b_offset)
        
        # Activate the correct dummy variables
        logit_a = pm.math.dot(module_matrix, a_m)
        logit_b = pm.math.dot(parameter_matrix, b_p)

        # Link function (logit), unbounded to (0,1) probability
        p = pm.Deterministic('p', pm.math.sigmoid(logit_a + logit_b))
        
        # Beta distribution likelihood 
        theta = pm.Gamma('theta', alpha=6, beta= 0.1) # Disperion parameter
        Y_obs = pm.Beta('Y_obs', alpha=p*theta, beta=(1-p)*theta, observed=observation_matrix['Coverage'])
        
        # Sample from the model
        trace = pm.sample(5000, chains=4, return_inferencedata=True, progressbar=True, target_accept=0.95)
        log_lik = pm.compute_log_likelihood(trace)

    
    print("Model building complete. Saving results...")

    az.to_netcdf(trace, "final_coverage_multi_parameter.nc")
    print("Results saved!")
    
  
    
