import stuett
import torch
import numpy as np
import scipy
import argparse
import datetime as dt
import os
import pandas as pd
import xarray as xr

from datasets import SeismicDataset, DatasetFreezer, DatasetMerger
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

import stuett
from stuett.global_config import get_setting, setting_exists, set_setting

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from ignite.metrics import Accuracy

from pathlib import Path

from PIL import Image

import numpy as np
import json
import pandas as pd
import os
from skimage import io as imio
import io, codecs

from models import SimpleCNN

from dateutil import rrule
from datetime import datetime, timedelta


account_name = (
    get_setting("azure")["account_name"]
    if setting_exists("azure")
    else "storageaccountperma8980"
)
account_key = (
    get_setting("azure")["account_key"] if setting_exists("azure") else None
)
store = stuett.ABSStore(
    container="hackathon-on-permafrost",
    prefix="seismic_data/4D/",
    account_name=account_name,
    account_key=account_key,
)

def get_seismic_transform():
    def to_db(x,min_value=1e-10,reference=1.0):
        value_db = 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, x))
        value_db -= 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, reference))
        return value_db

    spectrogram = stuett.data.Spectrogram(nfft=512, stride=512, dim="time", sampling_rate=1000)

    transform = transforms.Compose([
        lambda x: x / x.max(),                          # rescale to -1 to 1
        spectrogram,                                    # spectrogram
        lambda x: to_db(x).values.squeeze(),
        lambda x: Tensor(x)
        ])

    return transform


def transform_hour(data):
    pass

def transform_minute(data):
    pass


# Load the data source
def load_seismic_source(start, end):
    output = []
    for date in rrule.rrule(rrule.HOURLY, dtstart=start, until=end):
        seismic_node = stuett.data.SeismicSource(
            store=store,
            station="MH36",
            channel=["EHE", "EHN", "EHZ"],
            start_time=date,
            end_time=date + timedelta(hours=1),
        )
        output.append(transform_hour(seismic_node()))
    return output

def load_image_source():
    image_node = stuett.data.MHDSLRFilenames(
        store=store,
        force_write_to_remote=True,
        as_pandas=False,
    )
    return image_node, 3

transform = get_seismic_transform()
data_node, num_channels = load_seismic_source()
data = data_node()
print(type(data))
print(data)
