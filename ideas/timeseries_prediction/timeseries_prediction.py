import stuett
import argparse
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser(description="Time series prediction")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..","..", "data/")),
    help="The path to the folder containing the permafrost hackathon data",
)
args = parser.parse_args()

data_path = Path(args.path)


timeseries_folder = Path(data_path).joinpath("timeseries_derived_data_products")
rock_temperature_file = timeseries_folder.joinpath("MH30_temperature_rock_2017.csv")

if not rock_temperature_file.exists():
    raise RuntimeError('Please provide a valid path to the permafrost data or see README how to download it')

rock_temperature_node = stuett.data.CsvSource(rock_temperature_file)

rock_temperature = rock_temperature_node(
    {"start_time": datetime(2017, 8, 1), "end_time": datetime(2017, 9, 1)}
)

print(rock_temperature)
