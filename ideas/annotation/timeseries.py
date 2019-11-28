"""MIT License

Copyright (c) 2019, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer, Romain Jacob

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

import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd 

parser = argparse.ArgumentParser(description="Seismic time series and spectogram plot")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "data/")),
    help="The path to the folder containing the permafrost hackathon data",
)
parser.add_argument("-l", "--local", action="store_true", help="Load data from local storage")
args = parser.parse_args()

data_path = Path(args.path)
annotations_path = data_path.joinpath("annotations")

if not args.local:
    account_name = (
        get_setting("azure")["account_name"]
        if setting_exists("azure")
        else "storageaccountperma8980"
    )
    account_key = (
        get_setting("azure")["account_key"] if setting_exists("azure") else None
    )
    annotation_store = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix="annotations",
        account_name=account_name,
        account_key=account_key,
        blob_service_kwargs={},
    )
else:
    annotation_store = stuett.DirectoryStore(annotations_path)


df = stuett.read_csv_with_store(annotation_store, "automatic_labels_mountaineers.csv")
df['start_time'] = pd.to_datetime(df['start_time'] )
df['end_time'] = pd.to_datetime(df['end_time'] )
df.index = df['start_time']
df = df['2017-08-01':'2017-08-02']


fig = go.Figure(
    layout=dict(
        xaxis={"type": "date"},
        xaxis_range=[
            pd.to_datetime("2017-08-01"),
            pd.to_datetime("2017-08-02"),
        ],
    )
)

for i,row in df.iterrows():
    if(pd.isnull(row['__target'])):
        continue
    fig.add_trace(
        go.Scatter(
            x=[
                row['start_time'],
                row['end_time'],
                row['end_time'],
                row['start_time'],
            ],
            y=[0, 0, 1, 1],
            fill="toself",
            fillcolor="darkviolet",
            # marker={'size':0},
            mode="lines",
            hoveron="points+fills",  # select where hover is active
            line_color="darkviolet",
            showlegend=False,
            # line_width=0,
            opacity=0.5,
            text=str(row['__target']),
            hoverinfo="text+x+y",
        )
    )
fig.show()