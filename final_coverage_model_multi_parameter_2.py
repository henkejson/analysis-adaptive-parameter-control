import formulaic
import pymc as pm
import pandas as pd
import arviz as az

if __name__ == '__main__':
    data = pd.read_csv("multi_parameter/combined_data/statistics.csv")

    # Dummy variables for Module and Parameters
    model_formula = 'Coverage ~ 0 + C(TargetModule) + C(TuningParameters, contr.treatment("NONE"))'
    design_matrix = formulaic.model_matrix(model_formula, data=data)

    module_matrix = design_matrix.rhs.iloc[:, :24]
    parameter_matrix = design_matrix.rhs.iloc[:, 24:]

    with pm.Model():
        # Global Intercept and standard deviation for Modules
        a_bar = pm.Normal('a_bar', mu=0, sigma=1.5)
        
        # Standard Deviations for modules, marameters and nteractions
        sigma_a = pm.Exponential('sigma_a', 2.0)
        sigma_b = pm.Exponential('sigma_b', 5.0)
        sigma_g = pm.Exponential('sigma_g', 5.0)
        
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
        theta = pm.Uniform('theta', 10, 200) # Disperion parameter
        Y_obs = pm.Beta('Y_obs', alpha=p*theta, beta=(1-p)*theta, observed=design_matrix.lhs['Coverage'])
        
        # Sample from the model
        trace = pm.sample(7000, chains=4, return_inferencedata=True, progressbar=True, target_accept=0.95)
        log_lik = pm.compute_log_likelihood(trace)


    
    print("Model building complete. Saving results...")

    az.to_netcdf(trace, "final_coverage_model_multi_parameter_2.nc")
    print("Results saved!")
    
  
    
