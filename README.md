
# PermafrostAnalytics: Repository for the Permafrost Hackathon

## Organisation

### Communication
For discussions during and after the hackathon please use the [chat room](https://matrix.to/#/!DncqFOaoXsgUnageDH:matrix.ee.ethz.ch?via=matrix.ee.ethz.ch)

### Data
During the hackathon the data will be available [here](). We will move the data to a permanent, public storage location after the hackathon.

### Code
The code you will need lives in this repository. If you want to dig deeper you can also have a look at the code for our data management package [stuett](linktostuett).

## Quickstart

* Python
* git

We recommend [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), the latter being a minimal (but sufficient) version of the Anaconda distribution. The following will be based on miniconda but you can use any other python environment.

### Create a new conda environment
Open a terminal (e.g. Anaconda Terminal) and create a new environment with
```
conda create -n permafrost python==3.7 poetry
conda activate permafrost
```

### Download this repository and install the required packages
```
git clone [TODO:address]
poetry install
```

### Try out an example
```
python utils/download_files.py -f timeseries_derived_data_products.zip
python ideas/visualization/plot_timeseries.py
```

### Discover
There are more examples in the _ideas_ folder and some tutorials in the tutorials _folder_ which we will partially go through during the hackthon.

### Create your own idea
We encourage you to create your idea in this repository. Create a git branch and folder with the name of your idea. You can also share any code in this folder if you want to open source it.

```
git checkout -b youridea_name
mkdir ideas/youridea_name
```
