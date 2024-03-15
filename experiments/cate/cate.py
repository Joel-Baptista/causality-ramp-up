import numpy as np
import pandas as pd
import logging

import dowhy
from dowhy import CausalModel
import dowhy.datasets
from IPython.display import Image, display

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor

import econml
from econml.inference import BootstrapInference

import warnings
warnings.filterwarnings('ignore')

BETA = 10

def main() -> None:

    data = dowhy.datasets.linear_dataset(
        BETA, 
        num_common_causes=4, 
        num_samples=10000,
        num_instruments=2, 
        num_effect_modifiers=2,
        num_treatments=1,
        treatment_is_binary=False,
        num_discrete_common_causes=2,
        num_discrete_effect_modifiers=0,
        one_hot_encode=False
    )

    df=data['df']
    print(df.head())
    print("True causal estimate is", data["ate"])

    model = CausalModel(
        data=data["df"],
        treatment=data["treatment_name"], outcome=data["outcome_name"],
        graph=data["gml_graph"]
    )
    model.view_model()
    display(Image(filename="causal_model.png"))

    identified_estimand= model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    linear_estimate = model.estimate_effect(identified_estimand,
                                        method_name="backdoor.linear_regression",
                                       control_value=0,
                                       treatment_value=1)
    print(linear_estimate)


    
    dml_estimate = model.estimate_effect(
        identified_estimand, 
        method_name="backdoor.econml.dml.DML",
        control_value = 0,
        treatment_value = 1,
        target_units = lambda df: df["X0"]>1,  # condition used for CATE
        confidence_intervals=False,
        method_params={"init_params":{'model_y':GradientBoostingRegressor(),
        'model_t': GradientBoostingRegressor(),
        "model_final":LassoCV(fit_intercept=False),
        'featurizer':PolynomialFeatures(degree=1, include_bias=False)},
        "fit_params":{}}
    )

    print(dml_estimate)

    print("True causal estimate is", data["ate"])

    dml_estimate = model.estimate_effect(identified_estimand, method_name="backdoor.econml.dml.DML",
                                     control_value = 0,
                                     treatment_value = 1,
                                 target_units = 1,  # condition used for CATE
                                 confidence_intervals=False,
                                method_params={"init_params":{'model_y':GradientBoostingRegressor(),
                                                              'model_t': GradientBoostingRegressor(),
                                                              "model_final":LassoCV(fit_intercept=False),
                                                              'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                                               "fit_params":{}})
    print(dml_estimate)

    dml_estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.econml.dml.DML",
                                     target_units = "ate",
                                     confidence_intervals=True,
                                     method_params={"init_params":{'model_y':GradientBoostingRegressor(),
                                                              'model_t': GradientBoostingRegressor(),
                                                              "model_final": LassoCV(fit_intercept=False),
                                                              'featurizer':PolynomialFeatures(degree=1, include_bias=True)},
                                               "fit_params":{
                                                               'inference': BootstrapInference(n_bootstrap_samples=100, n_jobs=-1),
                                                            }
                                              })
    print(dml_estimate)