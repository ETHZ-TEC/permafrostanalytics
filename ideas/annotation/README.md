# Idea: Integrated Annotation Tool For Images


## Quickstart
From the repositories base directory execute
```
python ideas/annotation/images.py
```

The data will be loaded from Azure. You have the option to either load sparse (in time) high-resolution images or low resolution images with a higher capture frequency. (This has been done to minimize transfer costs; all images are available in high-resolution upon request)
You can also download the data (6.7 GB) yourself and use it (again executed from the base directory).

```
python utils/download_files.py -f timelapse_images_fast.zip
python ideas/annotation/images.py --local
```

## What next?
There are many tools available for image labeling, but what if we want to label timeseries data? How do we even label timeseries data? Can we simply draw on the plot of a time series or are there certain characteristics which are hidden and can only revealed with another view on the data?
A way to plot annotations over time is presented in [timeseries.py](./timeseries.py)