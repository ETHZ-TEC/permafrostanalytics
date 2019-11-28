# PermafrostAnalytics: Repository for the Permafrost Hackathon

## Organisation

### Communication
For discussions during and after the hackathon please use the [chat room](https://riot.im/app/#/room/!DncqFOaoXsgUnageDH:matrix.ee.ethz.ch?via=matrix.ee.ethz.ch).

### Data
During the hackathon, the data will be available [here](https://storageaccountperma8980.blob.core.windows.net/hackathon-on-permafrost/README.md). We will move the data to a permanent, public storage location after the hackathon.

### Code
The code you will need lives in this repository.

## Quickstart

To use the framework, you require the following tools as prerequisites:

* Python 3.7
* git
* gcc

We recommend [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), the latter being a minimal (but sufficient) version of the Anaconda distribution. The following instructions will be based on Miniconda, but you can use any other Python environment.

### Create a new conda environment

After the installation of Anaconda, open a terminal (on Windows `Anaconda Prompt`) and create a new environment by typing:

```
conda create -n permafrost python==3.7 git curl numcodecs -y
conda activate permafrost
pip install --pre poetry
```

### Download this repository and install the required packages

On Linux, do the following:

```
git clone https://gitlab.ethz.ch/tec/public/employees/matthias-meyer/permafrostanalytics
cd permafrostanalytics
poetry install -vvv
```

##### Windows installation

As Windows has issues with the Python packages we use, we require some additional steps:

```
git clone https://gitlab.ethz.ch/tec/public/employees/matthias-meyer/permafrostanalytics
cd permafrostanalytics
conda install -n permafrost pyyaml pytorch torchvision cpuonly -c pytorch
conda install -n permafrost -c conda-forge xarray-extras -y
poetry install -vvv
```

##### Installation hints

*Conda:* In case `conda` should throw errors, try updating it to its newest version:

    conda update -n base -c defaults conda
    
*Long installation time:* If the installation script is stuck for a prolonged period of time (> 30 seconds), simply try pressing the `Enter` key; it occasionally waits for affirmation which are not displayed properly.

*Reset:* If you would like to setup a new Python environment (e.g. in case your package management is corrupted because you installed new packets which are not compatible anymore, or due to a `MemoryError()`), you can quickly reset by removing the `permafrost` environment and starting from scratch:

```
conda deactivate
conda remove -n permafrost --all
```

#### Alternative Repository
If you prefer to work with Github, we have mirrored the repository [here](https://github.com/ETHZ-TEC/permafrostanalytics) and you can clone it
```
git clone https://github.com/ETHZ-TEC/permafrostanalytics
```

### Try out an example
```
python ideas/visualization/plot_timeseries.py
```

The script connects to the data repository on Azure. If you want to use the script without an internet connection you can download the files once, and run the script with the local option as shown below.

```
python utils/download_files.py -f timeseries_derived_data_products.zip
python ideas/visualization/plot_timeseries.py -l
```

In case your default browser does not render the page correctly, consider trying another browser. We recommend [Mozilla Firefox](https://www.mozilla.org/en-US/firefox/new/), which delivers the most stable performance.
On Windows, you can switch between default programs using `Control Panel → Programs → Default Programs → Set your default programs`.

### Discover
There are more examples in the [_ideas_](./ideas) folder and some tutorials in the [_tutorial_](./tutorial) folder which we will partially go through during the hackathon.

### Create your own idea
We encourage you to create your own idea in this repository. To do so, create a new git branch and folder with the name of your idea. You can also share any code in this folder if you want to open source it.
To make your idea publicly available for the discussion with others, we suggest you create a merge request after you are finished so we can include your idea directly inside the main repository.

```
git checkout -b youridea_name
mkdir ideas/youridea_name
cd ideas/youridea_name
```

## Need more...

### References

### Data
* Using the scripts from this repo, most of the data from [here](https://doi.pangaea.de/10.1594/PANGAEA.897640?format=html#download) can be used plug and play.
* If you have access to the [arclink service](http://arclink.ethz.ch), you can use it to download seismic streams. There is an explanation in the file [ideas/visualization/plot_spectrogram.py](./ideas/visualization/plot_spectrogram.py).


### Code
*  If you want to dig deeper you can also have a look at the code for our data management package [stuett](https://gitlab.ethz.ch/tec/public/employees/matthias-meyer/stuett).

### Tools
Check out the many open-source tools available online. The ones we use and found most useful are
* [python](https://www.python.org/)
* [xarray](http://xarray.pydata.org/)
* [dask](https://dask.org/)
* [pytorch](https://pytorch.org/)
* [plotly](https://plot.ly)/[dash](https://plot.ly/dash/)
* [zarr](https://zarr.readthedocs.io/en/stable/index.html)
* [pandas](https://pandas.pydata.org/)
* [obspy](https://github.com/obspy/obspy/wiki)/[obsplus](https://github.com/niosh-mining/obsplus)