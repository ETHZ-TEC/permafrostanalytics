import wget
from pathlib import Path 
import os
import argparse
import zipfile


parser = argparse.ArgumentParser(description="How to stuett")
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "data/")),
    help="The path to where the data should be downloaded",
)
parser.add_argument(
    "-f",
    "--files",
    type=str,
    nargs='+',
    help="The files to be downloaded",
)

args = parser.parse_args()

data_dir = Path(args.path)
print(f"Downloading files to {data_dir}")
os.makedirs(data_dir,exist_ok=True)

base_url = 'https://storageaccountperma8980.blob.core.windows.net/hackathon-on-permafrost/'

if not args.files:
    # download all files
    print('WARNING: Downloading all files, which are several GB of data. '
          'Make sure you are using an appropriate internet connection and have enough space left on your disk')
    files = ['timeseries_derived_data_products.zip']
else:
    files = args.files


for item in files:
    filename = data_dir / item
    if filename.exists():
        print(f'Not downloading {item}. It already exists.')
        continue
    print(f'Downloading {item}')
    wget.download(base_url + item, str(filename))

for item in files:
    filename = data_dir / item
    with zipfile.ZipFile(filename, 'r') as zip_file:
        zip_file.extractall(data_dir)
