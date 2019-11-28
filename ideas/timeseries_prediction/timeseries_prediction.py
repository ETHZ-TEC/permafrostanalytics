"""MIT License

Copyright (c) 2019, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer, Stefan Draskovic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

import stuett
from stuett.global_config import get_setting, setting_exists
import argparse
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import numpy as np

#################################################################################
## The following part parses data from the csv files.
## Note the name of the file we read from, and the time period we are focusing on
## >> You can change what data you read, or specify training and test datasets here!
#################################################################################
parser = argparse.ArgumentParser(description="Time series prediction")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "..", "data/")),
    help="The path to the folder containing the permafrost hackathon data",
)
parser.add_argument("-l", "--local", action="store_true", help="Only use local files and not data from Azure")
args = parser.parse_args()

data_path = Path(args.path)

# change here to get data from another sensor
rock_temperature_file = "MH30_temperature_rock_2017.csv"

# Getting either cloud or local data file
if not args.local:
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
else:
    timeseries_folder = Path(data_path).joinpath("timeseries_derived_data_products").resolve()
    store = stuett.DirectoryStore(timeseries_folder)
    if rock_temperature_file not in store:
        raise RuntimeError(
            "Please provide a valid path to the permafrost data or see README how to download it"
        )


rock_temperature_node = stuett.data.CsvSource(rock_temperature_file,store=store)

# Specify a subset of the data
rock_temperature = rock_temperature_node(
    {"start_time": datetime(2017, 4, 1), "end_time": datetime(2017, 4, 15)}
)

# Drop the position of the node, as it is the same for all data points and is not used here
rock_temperature = rock_temperature.drop(dim="name", labels=["position"])

## End of data parsing ##########################################################

#################################################################################
## The following part features two simple prediction algoritms
## Note that the prediction at time t is derived from values known before time t
## >> You add your algorithms here!
#################################################################################
dumb_prediction = np.asarray(
    [[None] * len(rock_temperature[1, :]) for i in range(len(rock_temperature[:, 1]))]
)
for depth in range(len(rock_temperature[1, :])):
    dumb_prediction[1, depth] = None
    for element in range(1, len(dumb_prediction[:, depth])):
        dumb_prediction[element, depth] = rock_temperature[element - 1, depth]

diff_prediction = np.asarray(
    [[None] * len(rock_temperature[1, :]) for i in range(len(rock_temperature[:, 1]))]
)
for depth in range(len(rock_temperature[1, :])):
    diff_prediction[1, depth] = None
    for element in range(1, len(dumb_prediction[:, depth])):
        if (
            diff_prediction[element - 1, depth] == None
            or np.isnan(diff_prediction[element - 1, depth])
            or element <= 25
        ):
            diff_prediction[element, depth] = rock_temperature.values[
                element - 1, depth
            ]
        else:
            diff_prediction[element, depth] = rock_temperature.values[
                element - 1, depth
            ] + (
                rock_temperature.values[element - 24, depth]
                - rock_temperature.values[element - 25, depth]
            )

## End of prediction algorithms #################################################

#################################################################################
## The following part is for evaulation of the prediction algoritms
## This is how we judge whether one prediction algorithm is better than another
## Note: this is not a perfect test (i.e. RMSE depends on the test dataset), but it's an OK first test
## >> You don't need to change anything here, just add your scheme
#################################################################################


def absolute_error(prediction, original):
    _ae = np.asarray([[None] * len(original[1, :]) for i in range(len(original[:, 1]))])
    for depth in range(len(_ae[1, :])):
        for element in range(len(_ae[:, depth])):
            if not (
                prediction[element, depth] == None
                or np.isnan(prediction[element, depth])
            ):
                _ae[element, depth] = np.abs(
                    original[element, depth] - prediction[element, depth]
                )
    return _ae


def rmse(_ae):
    _rmse = np.asarray([None] * len(_ae[1, :]))
    for depth in range(len(_ae[1, :])):
        filtered_ae = np.asarray(
            [i for i in _ae[:, depth] if not (i == None or np.isnan(i))]
        )
        _rmse[depth] = np.sqrt((filtered_ae ** 2).mean())

    return _rmse


# Add your scheme here
dumb_ae = absolute_error(dumb_prediction, rock_temperature)
diff_ae = absolute_error(diff_prediction, rock_temperature)

print("The RMSE of DUMB prediction is:")
print(rmse(dumb_ae))
print("The RMSE of DIFF prediction is:")
print(rmse(diff_ae))

## End of evaluation of prediction algorithms ###################################

#################################################################################
## The following part plots some data nicely
## >> This is just to help you visualize stuff
#################################################################################

# Create figure

fig = go.Figure()
for i in range(1):  # rock_temperature.shape[1]):
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(rock_temperature["time"].values),
            y=rock_temperature.values[:, i],
            name="actual_"
            + rock_temperature["name"].values[i]
            + " ["
            + rock_temperature["unit"].values[i]
            + "]",
        )
    )

for i in range(1):
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(rock_temperature["time"].values),
            y=np.array(dumb_prediction[:, i]).tolist(),
            name="dumb_pred_"
            + rock_temperature["name"].values[i]
            + " ["
            + rock_temperature["unit"].values[i]
            + "]",
        )
    )

for i in range(1):
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(rock_temperature["time"].values),
            y=np.array(diff_prediction[:, i]).tolist(),
            name="diff_pred_"
            + rock_temperature["name"].values[i]
            + " ["
            + rock_temperature["unit"].values[i]
            + "]",
        )
    )

for i in range(1):
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(rock_temperature["time"].values),
            y=np.array(dumb_ae[:, i]).tolist(),
            name="dumb_a_error_"
            + rock_temperature["name"].values[i]
            + " ["
            + rock_temperature["unit"].values[i]
            + "]",
        )
    )

for i in range(1):
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(rock_temperature["time"].values),
            y=np.array(diff_ae[:, i]).tolist(),
            name="diff_a_error_"
            + rock_temperature["name"].values[i]
            + " ["
            + rock_temperature["unit"].values[i]
            + "]",
        )
    )


# Set title
fig.update_layout(title_text="Time series with range slider and selectors")

# Add range slider
fig.update_layout(
    xaxis=go.layout.XAxis(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    )
)

fig.show()

## Have a nice day ##############################################################
