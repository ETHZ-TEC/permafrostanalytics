import stuett
import argparse
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

parser = argparse.ArgumentParser(description="Time series plot")
parser.add_argument(
    "data_folder",
    metavar="folder_to_data",
    type=str,
    help="The path to the folder containing the permafrost hackathon data",
)
args = parser.parse_args()

if not args.data_folder:
    raise RuntimeError("Please provide a path to the dataset folder")

timeseries_folder = Path(args.data_folder).joinpath("timeseries_derived_data_products")
rock_temperature_file = timeseries_folder.joinpath("MH30_temperature_rock_2017.csv")
rock_temperature_node = stuett.data.CsvSource(rock_temperature_file)

rock_temperature = rock_temperature_node()

# Create figure
# rock_temperature = rock_temperature.loc[:,['temperature_5cm','temperature_10cm','temperature_100cm']]
rock_temperature = rock_temperature.drop(dim="name", labels=["position"])

fig = go.Figure()
for i in range(rock_temperature.shape[1]):
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(rock_temperature["time"].values),
            y=rock_temperature.values[:, i],
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
