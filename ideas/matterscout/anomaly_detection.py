import stuett
import torch
import numpy as np
import scipy
import argparse
import datetime as dt
import os
import pandas as pd
import xarray as xr

#from datasets import SeismicDataset, DatasetFreezer, DatasetMerger
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
#from ignite.metrics import Accuracy

from pathlib import Path

from PIL import Image

import numpy as np
import json
import pandas as pd
import os
from skimage import io as imio
import io, codecs

#from models import SimpleCNN

from dateutil import rrule
from datetime import datetime, timedelta

from scipy.fftpack import fft


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

#calculates entropy on the measurements
def calculate_entropy(v):
    counter_values = Counter(v).most_common()
    probabilities = [elem[1] / len(v) for elem in counter_values]
    entropy = scipy.stats.entropy(probabilities)
    return entropy

#extracts statistical features
def min_max_estractor:
    return  [np.min(row), np.max(row), np.var(row), np.rms(row), entropy(row),
            np.percentile(row, 1), np.percentile(row, 5), np.percentile(row, 25),
            np.percentile(row, 95), np.percentile(row,95), np.percentile(row, 99)]

#computes fourier transform of the signal and extracts features
def fourier_extractor(x):
    sampling_freq = 250
    N=len(x)
    f_values = np.linspace(0.0, sampling_freq/2, N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])

    coeff_0=fft_values[0] #coefficient at 0Hz
    peak_70=0 #coefficient around 70 Hz
    coeff = np.zeros(20) #max coefficient from each 2 Hz interval (0-40)
    integral40 = 0 #integral from 0 to 40 Hz
    integral125 = np.avg(fft_values) #integral over the whole transform
    for i in range(0, len(f_values)):
        if f_values[i]>69 and f_values[i]<72 and fft_values[i]>peak_70:
            peak_70=fft_values[i]
        if f_values[i]<40:
            integral40+=fft_values[i]
            if fft_values[i] > coeff[int(i/2)]:
                coeff[int(i/2)]=fft_values[i]
    return coeff + [coeff_0, peak_70, integral40, integral125]

#extracts features from an hour worth of seismic data from three sensors
def transform_hour(data):
    data = np.array(data)
    features=[]
    for row in data:
        for extractor in [min_max_estractor, fourier_extractor]:
            for element in extractor(row):
                features.append(element)
    return features

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
