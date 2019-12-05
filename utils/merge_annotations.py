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

parser = argparse.ArgumentParser(description="Seismic time series and spectogram plot")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "data/")),
    help="The path to the folder containing the permafrost hackathon data",
)
parser.add_argument(
    "-u",
    "--user_annotations",
    type=str,
    default=str(
        Path(__file__)
        .absolute()
        .parent.joinpath("..", "data", "user_dir", "annotations")
    ),
    help="The path where the user annotations are stored",
)
parser.add_argument("-a", "--azure", action="store_true", help="Load data from Azure")
parser.add_argument(
    "-d", "--to_data_storage", action="store_true", help="Store on data storage"
)
args = parser.parse_args()

data_path = Path(args.path)
annotations_path = data_path.joinpath("annotations")

if args.azure:
    account_name = (
        get_setting("azure")["account_name"]
        if setting_exists("azure")
        else "storageaccountperma8980"
    )
    account_key = (
        get_setting("azure")["account_key"] if setting_exists("azure") else None
    )
    if args.to_data_storage:
        output_store = stuett.ABSStore(
            container="hackathon-on-permafrost",
            prefix="annotations",
            account_name=account_name,
            account_key=account_key,
            blob_service_kwargs={},
        )
    else:
        output_store = stuett.DirectoryStore(annotations_path)
        input_store = stuett.ABSStore(
            container="hackathon-public-rw",
            prefix="",
            account_name=account_name,
            account_key=account_key,
        )
else:
    input_store = stuett.DirectoryStore(args.user_annotations)
    output_store = stuett.DirectoryStore(annotations_path)

print(args.user_annotations)

annotation_dict = {}
for key in input_store.keys():
    fn = Path(key)
    if fn.suffix != ".csv":
        continue
    df = stuett.read_csv_with_store(input_store, key, dict(nrows=3))
    df["__session"] = str(fn.parent)
    filename = fn.name
    if filename not in annotation_dict:
        annotation_dict[filename] = {
            "key": key,
            "modif_ts": pd.to_datetime(df["__creation_time"].iloc[0]),
        }
    else:
        current_time = annotation_dict[filename]["modif_ts"]
        new_time = pd.to_datetime(df["__creation_time"].iloc[0])
        if new_time > current_time:
            print("Newer annotation found for %s. Replace in dict" % file_name)
            annotation_dict[filename] = annotation_dict[filename] = {
                "key": key,
                "modif_ts": new_time,
            }


df_list = []
for filename in annotation_dict:
    key = annotation_dict[filename]["key"]
    df = stuett.read_csv_with_store(input_store, key)
    df_list.append(df)

df = pd.concat(df_list)
print("Saving...")
stuett.to_csv_with_store(output_store, "annotations.csv", df)
