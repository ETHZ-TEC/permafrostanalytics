import stuett

import argparse
from pathlib import Path 
from datetime import datetime 

parser = argparse.ArgumentParser(description='How to stuett')
parser.add_argument('data_folder', metavar='folder_to_data', type=str, help='The path to the folder containing the permafrost hackathon data')
args = parser.parse_args()

if not args.data_folder:
    raise RuntimeError('Please provide a path with the -d argument')

# First load our data. In this case rock temperature data
timeseries_folder = Path(args.data_folder).joinpath('timeseries_derived_data_products')
rock_temperature_file = timeseries_folder.joinpath('MH30_temperature_rock_2017.csv')

# In stuett we can choose from different data loaders. Here, we choose the csv loader
# First we instantiate our loader with the data file. It will not load the file immediately but instead create a node
rock_temperature_node = stuett.data.CsvSource(rock_temperature_file)       

#### Data loading ####
# There are multiple ways to load the data.
# 1. When nothing is given as an argument to the node it will load all data available.
rock_temperature = rock_temperature_node()

# 2. We can also select a time frame providing a request (a dict containing e.g. start_time and end_time). 
# Note: Since we are loading csv files the whole file will still be loaded but then cropped to the desired time frame.
#       Other nodes, e.g. seismic nodes, will only load the data from the specified period
rock_temperature = rock_temperature_node({'start_time':'2017-08-01', 'end_time':'2017-09-01'})

# 3. If we want to provide the same data for every call but do not want to send a request each time we can simply
# configure the node with a request on initialization 
rock_temperature_node = stuett.data.CsvSource(**{'filename':rock_temperature_file,'start_time':'2017-08-01', 'end_time':'2017-09-01'})       
rock_temperature = rock_temperature_node()

#### Data structure ####
# The stuett nodes (unless specifically configured differently) will return xarray DataArrays.
# The DataArrays give you additional information e.g. regarding time or units.
print(rock_temperature)

# DataArrays provide you with advanced indexing capabilities 
# For more information see http://xarray.pydata.org/en/stable/indexing.html

# Time indexing: The following three lines accomplish the same goal.
one_day = rock_temperature.sel(time=(slice(datetime(2017,8,1),datetime(2017,8,2))))
one_day = rock_temperature.loc[datetime(2017,8,1):datetime(2017,8,2)]
one_day = rock_temperature.loc['2017-08-01':'2017-08-02']
print(one_day)

# Since the rock temperature sensor gives us the temperature for different depths
# We would like to select one specific depth (i.e. one specific column from the data)
one_day_5cm = rock_temperature.sel(time=slice(datetime(2017,8,1),datetime(2017,8,2)),name='temperature_5cm')
one_day_5cm = rock_temperature.loc['2017-08-01':'2017-08-02','temperature_5cm']
print(one_day_5cm)
# or from multiple depths
one_day_5cm_10cm = rock_temperature.loc['2017-08-01':'2017-08-02',['temperature_5cm','temperature_10cm']]
print(one_day_5cm_10cm)


#### Data formats ####
# Note: The timezone used within stuett is UTC, but the datetime objects you receive are not timezone aware.
#       Timezone issues are a recurring problem so be careful when working with the data.

#TODO: Seismic (as xarray and obspy)
#TODO: Images
#TODO: Rock temperature
#TODO: Wind
#TODO: Precipitation
#TODO: Radiation
#TODO: Mountaineer Annotations
#TODO: Annotations

#### Graph ####
