import numpy as np

from dowhy import CausalModel
import dowhy.datasets

from utils.time import time_func

GIVE_GRAPH = True
RANDOM_SEED = 0
WORKERS = 4


def main():

    data = time_func(dowhy.datasets.linear_dataset, {
        "beta": 10,
        "num_common_causes": 5,
        "num_instruments": 2,
        "num_effect_modifiers": 1,
        "num_samples": 5_000,
        "treatment_is_binary": True,
        "stddev_treatment_noise": 10,
        "num_discrete_common_causes": 1,
    }
    )

    print(data["df"].head())
    
    if GIVE_GRAPH:
        # Ia. Create a causal model from the data and given graph.
        model =time_func(CausalModel,
            {
                "data": data["df"],
                "treatment": data["treatment_name"],
                "outcome": data["outcome_name"],
                "graph": data["gml_graph"]
            }
        )
    else:
        # Ia. Create a causal model from the data and witouh a graph (but gives common causes and effect modifiers).
        model =time_func(CausalModel,
            {
                "data": data["df"],
                "treatment": data["treatment_name"],
                "outcome": data["outcome_name"],
                "common_causes":data["common_causes_names"],
                "effect_modifiers":data["effect_modifier_names"]
            }
        )

    model.view_model()

    # II. Identify causal effect and return target estimands

    identified_estimand = time_func(model.identify_effect)
    print(identified_estimand)

    # III. Estimate the target estimand using a statistical method.
    estimate_ate = time_func(model.estimate_effect,
            {
            "identified_estimand":identified_estimand,
            "method_name":"backdoor.propensity_score_stratification"
        }
    )
    estimate_att = time_func(model.estimate_effect,
            {
            "identified_estimand":identified_estimand,
            "method_name":"backdoor.propensity_score_stratification",
            "target_units":"att"
        }
    )
    estimate_atc = time_func(model.estimate_effect,
            {
            "identified_estimand":identified_estimand,
            "method_name":"backdoor.propensity_score_stratification",
            "target_units":"atc"
        }
    )

    print("==========================ATE==========================")
    print(estimate_ate)
    print("Causal Estimate is " + str(estimate_ate.value))
    print("==========================ATT==========================")
    print(estimate_att)
    print("Causal Estimate is " + str(estimate_att.value))
    print("==========================ATC==========================")
    print(estimate_atc)
    print("Causal Estimate is " + str(estimate_atc.value))
    # # IV. Refute the obtained estimate using multiple robustness checks.
    
    res_random = time_func(model.refute_estimate, 
                            {
                            "estimand":identified_estimand, 
                            "estimate": estimate_ate,
                            "method_name":"random_common_cause",
                            "show_progress_bar":True,
                            "random_seed": RANDOM_SEED,
                            "n_jobs": WORKERS
                            }
                    )
    print(res_random)
        
    res_placebo = time_func(model.refute_estimate, 
                            {
                            "estimand":identified_estimand, 
                            "estimate": estimate_ate,
                            "method_name":"placebo_treatment_refuter",
                            "show_progress_bar":True,
                            "placebo_type":"permute",
                            "random_seed": RANDOM_SEED,
                            "n_jobs": WORKERS
                            }
                    )
    print(res_placebo)

    res_subset = time_func(model.refute_estimate, 
                        {
                        "estimand":identified_estimand, 
                        "estimate": estimate_ate,
                        "method_name":"data_subset_refuter",
                        "show_progress_bar":True,
                        "subset_fraction":0.9,
                        "random_seed": RANDOM_SEED,
                        "n_jobs": WORKERS
                        }
                )
    print(res_subset)

    
    res_unobserved = time_func(model.refute_estimate, 
                        {
                        "estimand":identified_estimand, 
                        "estimate": estimate_ate,
                        "method_name":"add_unobserved_common_cause",
                        "confounders_effect_on_treatment": "binary_flip",
                        "confounders_effect_on_outcome": "linear",
                        "effect_strength_on_treatment":0.01,
                        "effect_strength_on_outcome":0.02,
                        "show_progress_bar":True,
                        "random_seed": RANDOM_SEED,
                        "n_jobs": WORKERS
                        }
                )
    print(res_unobserved)

    res_unobserved_range = time_func(model.refute_estimate, 
                        {
                        "estimand":identified_estimand, 
                        "estimate": estimate_ate,
                        "method_name":"add_unobserved_common_cause",
                        "confounders_effect_on_treatment": "binary_flip",
                        "confounders_effect_on_outcome": "linear",
                        "effect_strength_on_treatment":[0.001, 0.005, 0.01, 0.02],
                        "effect_strength_on_outcome":0.01,
                        "show_progress_bar":True,
                        "random_seed": RANDOM_SEED,
                        "n_jobs": WORKERS
                        }
                )
    print(res_unobserved_range)

    res_unobserved_drange = time_func(model.refute_estimate, 
                        {
                        "estimand":identified_estimand, 
                        "estimate": estimate_ate,
                        "method_name":"add_unobserved_common_cause",
                        "confounders_effect_on_treatment": "binary_flip",
                        "confounders_effect_on_outcome": "linear",
                        "effect_strength_on_treatment":np.linspace(0.001, 0.1, 20),
                        "effect_strength_on_outcome":np.linspace(0.001, 0.1, 20),
                        "show_progress_bar":True,
                        "random_seed": RANDOM_SEED,
                        "n_jobs": WORKERS
                        }
                )
    print(res_unobserved_drange)

    

                            
    