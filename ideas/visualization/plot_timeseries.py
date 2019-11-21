'''MIT License

Copyright (c) 2019, Swiss Federal Institute of Technology (ETH Zurich)

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
SOFTWARE.'''

import stuett
import argparse
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

parser = argparse.ArgumentParser(description="Time series plot")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "..", "data")),
    help="The path to the folder containing the permafrost hackathon data",
)
args = parser.parse_args()

data_path = Path(args.path)

timeseries_folder = Path(data_path).joinpath("timeseries_derived_data_products")
rock_temperature_file = timeseries_folder.joinpath("MH30_temperature_rock_2017.csv")

if not rock_temperature_file.exists():
    raise RuntimeError('Please provide a valid path to the permafrost data or see README how to download it')

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
            name=rock_temperature['name'].values[i] + ' [' + rock_temperature['unit'].values[i] + ']'
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
