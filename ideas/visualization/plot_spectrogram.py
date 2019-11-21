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
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np 

parser = argparse.ArgumentParser(description="Seismic time series and spectogram plot")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "..", "data/")),
    help="The path to the folder containing the permafrost hackathon data",
)
args = parser.parse_args()

data_path = Path(args.path)

seismic_folder = Path(data_path).joinpath("seismic_data/4D/")

if not seismic_folder.exists():
    raise RuntimeError('Please provide a valid path to the permafrost data or see README how to download it')

seismic_node = stuett.data.SeismicSource(
    path=seismic_folder,
    station="MH36",
    channel=["EHE",'EHN','EHZ'],
    start_time="2017-08-01 10:00:00",
    end_time="2017-08-01 10:01:00",
)

seismic_data = seismic_node()

print(seismic_data)
# Create figure

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing = 0.1)
fig.update_layout(title_text="Time series and spectrogram")

for i,seed_id in enumerate(seismic_data['seed_id']):
    for j,stream_id in enumerate(seismic_data['stream_id']):
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(seismic_data["time"].values),
                y=seismic_data.sel(seed_id=seed_id,stream_id=stream_id).values,
            ),
        row=1, col=1)


spectrogram = stuett.data.Spectrogram(nfft=512, stride=64, dim="time")
spec = spectrogram(seismic_data)

# select only one channel
spec = spec.sel(seed_id='4D.MH36.A.EHE', stream_id=0)

trace_hm = go.Heatmap(
    x=pd.to_datetime(spec['time'].values),
    y=spec['frequency'].values,
    z=np.log(spec.values),
    colorscale="Jet",
    hoverinfo="none",
    colorbar={"title": "Power Spectrum/dB"},
)
fig.add_trace(trace_hm,row=2,col=1)

fig.show()