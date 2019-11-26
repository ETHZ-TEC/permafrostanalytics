"""MIT License

Copyright (c) 2019, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

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
from stuett.global_config import get_setting, setting_exists, set_setting

import argparse
from pathlib import Path

import numpy as np
import json
import pandas as pd
import os


parser = argparse.ArgumentParser(description="Seismic time series and spectogram plot")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "..", "data/")),
    help="The path to the folder containing the permafrost hackathon data",
)
parser.add_argument("-a", "--azure", action="store_true", help="Load data from Azure")
args = parser.parse_args()

data_path = Path(args.path)

prefix = "timelapse_images"
if args.azure:
    from stuett.global_config import get_setting, setting_exists

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
        prefix=prefix,
        account_name=account_name,
        account_key=account_key,
        blob_service_kwargs={},
    )
else:
    folder = Path(data_path).joinpath(prefix)
    store = stuett.DirectoryStore(folder)
    if not folder.exists():
        raise RuntimeError(
            "Please provide a valid path to the permafrost data or see README how to download it"
        )


# Setting a user directory to speed up image lookup
set_setting(
    "user_dir",
    str(Path(__file__).absolute().parent.joinpath("..", "..", "data", "user_dir")),
)
os.makedirs(get_setting("user_dir"), exist_ok=True)

node = stuett.data.MHDSLRImages(
    base_directory=folder,
    output_format="xarray",
    start_time=pd.to_datetime("2017-05-10 11:00:00"),
    end_time=pd.to_datetime("2017-05-12"),
)
# node = stuett.data.MHDSLRFilenames(base_directory=folder, start_time = pd.to_datetime('2017-05-10 11:00:00'), end_time = pd.to_datetime('2017-05-12'))


filename = Path(data_path).joinpath("annotations", "boundingbox_timeseries.csv")
filename = Path(data_path).joinpath("annotations", "boundingbox_images.csv")
label = stuett.data.BoundingBoxAnnotation(filename)


# TODO: new class extends SegmentedDataset that gets as Input filenames and provides images
dataset = stuett.data.SegmentedDataset(
    node,
    label,
    dataset_slice={"time": slice("2017-05-10", "2017-05-12")},
    batch_dims={"time": pd.to_timedelta(1, "H")},
)

x = dataset[0]
print(x.shape)
