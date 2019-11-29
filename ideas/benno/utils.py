import stuett
from stuett.global_config import get_setting, setting_exists
import argparse
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt

def load_data(fname):
    
    # change here to get data from another sensor
    rock_temperature_file = "MH30_temperature_rock_2017.csv"
    #rock_temperature_file= "MH10_resistivity_rock_2017.csv"
    #rock_temperature_file= "MH25_vaisalawxt520windpth_2017.csv"
    # Getting cloud data
    account_name = (
        get_setting("azure")["account_name"]
        if setting_exists("azure")
        else "storageaccountperma8980"
    )
    account_key = (
        get_setting("azure")["account_key"] if setting_exists("azure") else None
    )
    store = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix="timeseries_derived_data_products",
        account_name=account_name,
        account_key=account_key, 
    )


    rock_temperature_node = stuett.data.CsvSource(fname,store=store)
    rock_temperature = rock_temperature_node()

    # return data
    rock_temperature = rock_temperature.drop(dim="name", labels=["position"])
    return rock_temperature