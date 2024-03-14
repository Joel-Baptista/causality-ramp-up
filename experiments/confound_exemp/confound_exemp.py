import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import dowhy
from dowhy import CausalModel
import dowhy.datasets, dowhy.plotter

import logging.config

PLOT = False
DEFAULT_LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'loggers': {
        '': {
            'level': 'INFO',
        },
    }
}
logging.config.dictConfig(DEFAULT_LOGGING)

def main() -> None:
    
    rvar = 1 if np.random.uniform() > 0.5 else 0
    data_dict = dowhy.datasets.xy_dataset(
        10000, 
        effect=rvar, 
        sd_error=0.2
    )

    df = data_dict['df']
    print(df[["Treatment", "Outcome", "w0"]].head())

    if PLOT:
        dowhy.plotter.plot_treatment_outcome(
            df[data_dict["treatment_name"]], 
            df[data_dict["outcome_name"]],
            df[data_dict["time_val"]]
        )

    model = CausalModel(
        data = df,
        treatment=data_dict["treatment_name"],
        outcome=data_dict["outcome_name"],
        common_causes=data_dict["common_causes_names"],
        instruments=data_dict["instrument_names"]
    )

    # model.view_model(layout="dot")

    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    estimate = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression")
    print("Causal Estimate is " + str(estimate.value))

    if PLOT:
        dowhy.plotter.plot_causal_effect(
            estimate, 
            df[data_dict["treatment_name"]], 
            df[data_dict["outcome_name"]]
        )

    print("DoWhy estimate is " + str(estimate.value))
    print ("Actual true causal effect was {0}".format(rvar))

    res_random=model.refute_estimate(identified_estimand, estimate, method_name="random_common_cause")
    res_placebo=model.refute_estimate(identified_estimand, estimate,
    method_name="placebo_treatment_refuter", placebo_type="permute")
    res_subset=model.refute_estimate(identified_estimand, estimate,
    method_name="data_subset_refuter", subset_fraction=0.9)
    
    print(res_random)
    print(res_placebo)
    print(res_subset)




