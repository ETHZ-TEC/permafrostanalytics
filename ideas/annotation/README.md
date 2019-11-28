# Idea: Integrated Annotation Tool For Images


## Quickstart
From the repositories base directory execute
```
python ideas/annotation/images.py
```
In the console you will see a web-address. Click on it or enter it into your browser.

The data will be loaded from Azure. You have the option to either load sparse (in time) high-resolution images or low resolution images with a higher capture frequency (this has been done to minimize transfer costs; all images are available in high-resolution by adding the argument `-hq` when invoking the script).
You can also download the data (6.7 GB) yourself and use it (again executed from the base directory).

```
python utils/download_files.py -f timelapse_images_fast.zip
python ideas/annotation/images.py --local
```

## Annotations

## Data collection
The tool stores your labels in the cloud and locally. Locally this is done by using one ID per browser session (if you close and open the browser you cannot see your old labels). In frequent intervals we will merge all your labels (with the script `utils/merge_annotations.py`) and load them to the data folder on Azure. The image labeling tool will automatically download the annotations done by the all participants and you can fine-tune, label in addition and of course use all the labels.

## What next?
There are many tools available for image labeling, but what if we want to label timeseries data? How do we even label timeseries data? Can we simply draw on the plot of a time series or are there certain characteristics which are hidden and can only revealed with another view on the data?
A way to plot annotations over time is presented in [timeseries.py](./timeseries.py)