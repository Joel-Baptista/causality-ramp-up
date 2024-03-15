import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import dowhy

from IPython.display import Image, display

import warnings
import os
warnings.filterwarnings('ignore')

EXPECTED_COUNTS = False

def main() -> None:
    dataset = pd.read_csv('https://raw.githubusercontent.com/Sid-darthvader/DoWhy-The-Causal-Story-Behind-Hotel-Booking-Cancellations/master/hotel_bookings.csv')
    print(dataset.head())
    print("-------------Old columns------------------")
    print(dataset.columns)

    # Total stay in nights
    dataset['total_stay'] = dataset['stays_in_week_nights']+dataset['stays_in_weekend_nights']
    # Total number of guests
    dataset['guests'] = dataset['adults']+dataset['children'] +dataset['babies']
    # Creating the different_room_assigned feature
    dataset['different_room_assigned']=0
    slice_indices =dataset['reserved_room_type']!=dataset['assigned_room_type']
    dataset.loc[slice_indices,'different_room_assigned']=1
    # Deleting older features
    dataset = dataset.drop(['stays_in_week_nights','stays_in_weekend_nights','adults','children','babies'
                            ,'reserved_room_type','assigned_room_type'],axis=1)


    dataset.isnull().sum() # Country,Agent,Company contain 488,16340,112593 missing entries
    dataset = dataset.drop(['agent','company'],axis=1)  

    # Replacing missing countries with most freqently occuring countries
    dataset['country']= dataset['country'].fillna(dataset['country'].mode()[0])

    dataset = dataset.drop(['reservation_status','reservation_status_date','arrival_date_day_of_month'],axis=1)
    dataset = dataset.drop(['arrival_date_year'],axis=1)
    dataset = dataset.drop(['distribution_channel'], axis=1)

    # Replacing 1 by True and 0 by False for the experiment and outcome variables
    dataset['different_room_assigned']= dataset['different_room_assigned'].replace(1,True)
    dataset['different_room_assigned']= dataset['different_room_assigned'].replace(0, False)
    dataset['is_canceled']= dataset['is_canceled'].replace(1,True)
    dataset['is_canceled']= dataset['is_canceled'].replace(0, False)
    dataset.dropna(inplace=True)
    print("-------------New columns------------------")
    print(dataset.columns)
    print(dataset.iloc[:, 5:20].head(100))

    dataset = dataset[dataset.deposit_type=="No Deposit"]
    print(dataset.groupby(['deposit_type','is_canceled']).count())
          
    dataset_copy = dataset.copy(deep=True)

    if EXPECTED_COUNTS:
        counts_sum=0
        for i in range(1,10000):
            counts_i = 0
            rdf = dataset.sample(1000)
            counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
            counts_sum+= counts_i
        print(f"Average cancelation when different rooms assigned: {counts_sum/10000}") # 588.6

        counts_sum=0
        for i in range(1,10000):
            counts_i = 0
            rdf = dataset[dataset["booking_changes"]==0].sample(1000)
            counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
            counts_sum+= counts_i
        print(f"Average cancelation when different rooms assigned, with no booking changes: {counts_sum/10000}")
        # 572.8
        counts_sum=0
        for i in range(1,10000):
            counts_i = 0
            rdf = dataset[dataset["booking_changes"]>0].sample(1000)
            counts_i = rdf[rdf["is_canceled"]== rdf["different_room_assigned"]].shape[0]
            counts_sum+= counts_i
        print(f"Average cancelation when different rooms assigned, with booking changes: {counts_sum/10000}")
        # 666.0 -> There is some influence more than expected. Booking changes might confound this two variables

    
    # 1 - Create Causal Graph
    
    # Get the current file path
    file_path = os.path.abspath(__file__)

    # Specify the location of the graph file
    graph_file = os.path.join(os.path.dirname(file_path), "graph.txt")

    with open(graph_file, 'r') as file:
        causal_graph = file.read()

    model= dowhy.CausalModel(
        data = dataset,
        graph=causal_graph.replace("\n", " "),
        treatment="different_room_assigned",
        outcome='is_canceled')
    model.view_model()

    display(Image(filename="causal_model.png"))

    #Identify the causal effect
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)


    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_weighting",
        target_units="ate")
    # ATE = Average Treatment Effect
    # ATT = Average Treatment Effect on Treated (i.e. those who were assigned a different room)
    # ATC = Average Treatment Effect on Control (i.e. those who were not assigned a different room)
    print(estimate)


    refute1_results=model.refute_estimate(identified_estimand, estimate,
        method_name="random_common_cause")
    print(refute1_results)

    refute2_results=model.refute_estimate(identified_estimand, estimate,
        method_name="placebo_treatment_refuter")
    print(refute2_results)

    refute3_results=model.refute_estimate(identified_estimand, estimate,
        method_name="data_subset_refuter")
    print(refute3_results)
