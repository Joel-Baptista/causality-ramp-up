import numpy as np

from dowhy import CausalModel
import dowhy.datasets

import timeit

GIVE_GRAPH = True

def time_func(func, kargs={}):
    start_time = timeit.default_timer()
    res = func(**kargs)
    elapsed_time = timeit.default_timer() - start_time
    print("Execution time of", func.__name__, ":", elapsed_time, "seconds")
    
    return res

def main():

    data = time_func(dowhy.datasets.linear_dataset, {
        "beta": 10,
        "num_common_causes": 2,
        "num_instruments": 0,
        "num_effect_modifiers": 1,
        "num_samples": 5_000,
        "treatment_is_binary": True,
        "stddev_treatment_noise": 10,
        "num_discrete_common_causes": 1
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
            "method_name":"backdoor.propensity_score_matching"
        }
    )
    estimate_att = time_func(model.estimate_effect,
            {
            "identified_estimand":identified_estimand,
            "method_name":"backdoor.propensity_score_matching",
            "target_units":"att"
        }
    )
    estimate_atc = time_func(model.estimate_effect,
            {
            "identified_estimand":identified_estimand,
            "method_name":"backdoor.propensity_score_matching",
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
    # refute_results = time_func(model.refute_estimate, 
    #                         {
    #                         "estimand":identified_estimand, 
    #                         "estimate": estimate,
    #                         "method_name":"random_common_cause"
    #                         }
    #                 )
                            