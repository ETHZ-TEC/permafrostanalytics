# PermafrostAnalytics: Repository for the Permafrost Hackathon

## Organisation

### Communication
For discussions during and after the hackathon please use the [chat room](https://matrix.to/#/!DncqFOaoXsgUnageDH:matrix.ee.ethz.ch?via=matrix.ee.ethz.ch).

### Data
During the hackathon, the data will be available [here](). We will move the data to a permanent, public storage location after the hackathon.

### Code
The code you will need lives in this repository. If you want to dig deeper you can also have a look at the code for our data management package [stuett](https://gitlab.ethz.ch/tec/public/employees/matthias-meyer/stuett).

## Quickstart

To use the framework, you require the following tools:

* Python
* git

We recommend [Anaconda](https://www.anaconda.com/distribution/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), the latter being a minimal (but sufficient) version of the Anaconda distribution. The following instructions will be based on Miniconda, but you can use any other Python environment.

### Create a new conda environment

After the installation of Anaconda, open a terminal (on Windows `Anaconda Prompt`) and create a new environment by typing:

```
conda create -n permafrost python==3.7 git curl -y
conda activate permafrost
pip install --pre poetry
```

### Download this repository and install the required packages

On Linux, do the following:

```
git clone https://gitlab.ethz.ch/tec/public/employees/matthias-meyer/permafrostanalytics
cd permafrostanaltics
poetry install -vvv
```

##### Windows installation

As Windows has issues with the Python packages we use, we require some additional steps:

```
git clone https://gitlab.ethz.ch/tec/public/employees/matthias-meyer/permafrostanalytics
cd permafrostanaltics
conda install -n permafrost numcodecs pyyaml -y
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

### Try out an example
```
python utils/download_files.py -f timeseries_derived_data_products.zip
python ideas/visualization/plot_timeseries.py
```

In case your default browser doesnt render the page correctly, consider trying another browser. We recommend [Mozilla Firefox](https://www.mozilla.org/en-US/firefox/new/), which delivers the most stable performance.
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
