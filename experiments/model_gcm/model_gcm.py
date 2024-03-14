from dowhy import gcm
import networkx as nx
import numpy as np
import pandas as pd


def main() -> None:
    
    # Create DAG to model a causal graph
    causal_graph = nx.DiGraph([('X', 'Y'), ('Y', 'Z')])

    # Create a structural causal model from the causal graph
    causal_model = gcm.StructuralCausalModel(causal_graph)

    # Generate synthetic data for the structural causal model
    
    x = np.random.normal(size=1000, scale=1, loc=0)
    y = 2 * x + np.random.normal(size=1000, scale=1, loc=0)
    z = 3 * y + np.random.normal(size=1000, scale=1, loc=0)

    data = pd.DataFrame({'X': x, 'Y': y, 'Z': z}) 
    print(data.head())

    # Assign causal mechanisms to the structural causal model according to the data
    # auto_assignment_summary = gcm.auto.assign_causal_mechanisms(causal_model, data)
    # print(auto_assignment_summary)

    # It is also possible to set the nature of each causal equation manually
    # (In this particular example, auto assign correctly assigns the nature of the causal equations)
    causal_model.set_causal_mechanism('X', gcm.EmpiricalDistribution())
    causal_model.set_causal_mechanism('Y', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))
    causal_model.set_causal_mechanism('Z', gcm.AdditiveNoiseModel(gcm.ml.create_linear_regressor()))

    gcm.fit(causal_model, data)
    
    # Gives metrics to evaluate the causal model
    print(gcm.evaluate_causal_model(causal_model, data))

    samples = gcm.interventional_samples(causal_model,
                                     {'Y': lambda y: 2.34 },
                                     num_samples_to_draw=1000)
    print(samples.head())

    
