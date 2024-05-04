import formulaic
import pymc as pm
import pandas as pd

def get_design_matricies(data, observation_column):
    # Dummy variables for Module and Parameters
    model_formula = f'{observation_column} ~ 0 + C(TargetModule) + C(TuningParameters, contr.treatment("NONE"))'
    design_matrix = formulaic.model_matrix(model_formula, data=data)

    module_matrix = design_matrix.rhs.iloc[:, :24]
    print(f"Module Matrix shape: {module_matrix.shape}")
    parameter_matrix = design_matrix.rhs.iloc[:, 24:]
    print(f"Parameter Matrix shape: {parameter_matrix.shape}")

    # Dummy variables for interaction terms
    model_formula = f'{observation_column} ~ 0 + C(TargetModule) : C(TuningParameters)'
    design_matrix = formulaic.model_matrix(model_formula, data=data)

    # Filter out columns that contain 'T.NONE' in their name
    columns_to_drop = [col for col in design_matrix.rhs.columns if 'T.NONE' in col]

    # Drop the identified columns
    design_matrix.rhs.drop(columns=columns_to_drop, axis=1, inplace=True)
    interaction_matrix = design_matrix.rhs.iloc[:,:]
    print(f"Interaction Matrix shape: {interaction_matrix.shape}")
    print(f"Observation Matrix shape: {design_matrix.lhs.shape}")


    return design_matrix.lhs, module_matrix, parameter_matrix, interaction_matrix
