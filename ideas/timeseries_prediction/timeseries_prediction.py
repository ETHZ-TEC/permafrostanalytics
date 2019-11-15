import stuett
import argparse
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser(description="Time series prediction")
parser.add_argument(
    "data_folder",
    metavar="folder_to_data",
    type=str,
    help="The path to the folder containing the permafrost hackathon data",
)
args = parser.parse_args()

if not args.data_folder:
    raise RuntimeError("Please provide a path with the -d argument")


timeseries_folder = Path(args.data_folder).joinpath("timeseries_derived_data_products")
rock_temperature_file = timeseries_folder.joinpath("MH30_temperature_rock_2017.csv")

rock_temperature_node = stuett.data.CsvSource(rock_temperature_file)

rock_temperature = rock_temperature_node(
    {"start_time": datetime(2017, 8, 1), "end_time": datetime(2017, 9, 1)}
)

print(rock_temperature)
