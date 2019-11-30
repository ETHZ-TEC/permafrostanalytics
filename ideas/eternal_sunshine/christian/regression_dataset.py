"""MIT License

Copyright (c) 2019, Christian Henning

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Note, some code snippets in this file are inspired by implementations from
Matthias Meyer as provided for the purpose of this hackathon,.
"""

import stuett
from stuett.global_config import get_setting, setting_exists
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch
import io


class PermaRegressionDataset(Dataset):
    """A dataset that maps images and meta information (such as radiation,
    surface temperature, ...) onto the temperature below surface."""

    def __init__(self, local, data_path='../data', transform=None,
                 time_slice={"start_time": "2017-01-01",
                             "end_time": "2017-12-31"}):
        """
        Args:
            local (bool): Whether to read the dataset from a local storage
                location or from a public Azure share.
            data_path (str, optional): If the data should be read from a local
                location, then this folder will denote the location of the
                dataset.
            transform (callable, optional): Optional transform to be applied
                on images.
            time_slice (dict): Can be used to create a different train and test
                set. Note, this is not a pretty solution, especially because
                time values are not interleaved. I.e., if time information is
                used as input to a network, but the network has never seen
                values from the corresponding month, then it can't make
                confident predictions.
        """
        if transform is not None:
            raise NotImplementedError("transform not implemented!")
        self.transform = transform

        # This sensor contains near-surface temperature readings and is on the
        # south side and therefore receives a lot of sunshine.
        rock_temperature_file_mh10 = "MH10_temperature_rock_2017.csv"  # South

        radiation_file = "MH15_radiometer__conv_2017.csv"

        if not local:
            account_name = (
                get_setting("azure")["account_name"]
                if setting_exists("azure")
                else "storageaccountperma8980"
            )
            account_key = (
                get_setting("azure")["account_key"] if setting_exists(
                    "azure") else None
            )

            ts_store = stuett.ABSStore(
                container="hackathon-on-permafrost",
                prefix="timeseries_derived_data_products",
                account_name=account_name,
                account_key=account_key,
            )

            img_store = stuett.ABSStore(
                container="hackathon-on-permafrost",
                prefix="timelapse_images_fast",
                account_name=account_name,
                account_key=account_key,
            )

        else:
            timeseries_folder = Path(data_path).joinpath(
                "timeseries_derived_data_products").resolve()
            ts_store = stuett.DirectoryStore(timeseries_folder)
            if rock_temperature_file_mh10 not in store:
                raise RuntimeError('Please provide a valid path to the ' +
                                   'permafrost data!')

            img_store = stuett.DirectoryStore(Path(data_path).joinpath( \
                'timelapse_images_fast'))
            if "2017-01-01/20170101_080018.JPG" not in store:
                raise RuntimeError('Please provide a valid path to the ' +
                                   'permafrost images.')

        #self._ts_store = ts_store
        self._img_store = img_store

        ### Load timeseries data.
        rock_temperature_node_mh10 = stuett.data.CsvSource(
            rock_temperature_file_mh10, store=ts_store)
        rock_temp_mh10 = rock_temperature_node_mh10(time_slice)

        radiation_node = stuett.data.CsvSource(radiation_file, store=ts_store)
        radiation = radiation_node(time_slice)

        net_radiation = radiation.loc[:, ['net_radiation']]
        surface_temp = rock_temp_mh10.loc[:, ['temperature_nearsurface_t2']]
        target_temp = rock_temp_mh10.loc[:, ['temperature_10cm']]

        ### Load image filenames.
        image_node = stuett.data.MHDSLRFilenames(
            store=img_store,
            force_write_to_remote=True,
            as_pandas=False,
        )
        image_fns = image_node(time_slice)

        ### Find image filenames that were captured close to temperature
        ### measures.
        # With close we mean within a 20min window.
        # Temperature/radiation values that have no corresponding image are
        # ignored.

        # Sanity check!
        #for t1, t2 in zip(radiation['time'], rock_temp_mh10['time']):
        #    assert (t1 == t2)

        j = 0
        n = len(image_fns['time'])

        measurement_pairs = []

        for i, t in enumerate(rock_temp_mh10['time'].values):
            while j < n:
                # Translate difference in timestamps to minutes before casting
                # to int.
                diff = (image_fns['time'][j] - t).values.astype( \
                    'timedelta64[m]').astype(np.int)


                if diff > 10:
                    # Image too far in the future, ignore sensor value.
                    break

                absdiff = np.abs(diff)
                if absdiff < 10:
                    # The image is very close, simply check whether the next
                    # picture is even closer. Otherwise, we take the current
                    # image.
                    if j + 1 < n:
                        absdiff2 = np.abs(
                            (image_fns['time'][j + 1] - t).values.astype(
                                'timedelta64[m]').astype(np.int))
                    else:
                        absdiff2 = None

                    if absdiff2 is None or absdiff < absdiff2:
                        measurement_pairs.append((i, j))
                        j += 1
                    else:
                        measurement_pairs.append((i, j + 1))
                        j += 2

                    break

                j += 1

        ### Build dataset (make sure that there are no None values in the
        ### timeseries measurements).
        self._img_fns = []
        self._surface_temp = []
        self._target_temp = []
        self._timestamps = []
        self._radiation = []

        # This is coarse time information that one may provide as additional
        # information. We encode the (normalized) month and daytime information,
        # as this information may be quite helpful when judging temperature
        # values.
        # Though, it might also tempt the regression system to ignore all
        # other information and solely predict based on this information
        # (as a strong local minimum).
        self._month = []
        self._daytime = []

        assert(np.all(~np.isnan(net_radiation.values)))
        assert(np.all(~np.isnan(surface_temp.values)))
        #assert(np.all(~np.isnan(target_temp.values)))

        for i, j in measurement_pairs:
            if np.any(np.isnan(target_temp.values[i, 0])):
                continue

            self._target_temp.append(target_temp.values[i, 0])
            self._surface_temp.append(surface_temp.values[i, 0])
            self._radiation.append(net_radiation.values[i, 0])

            self._timestamps.append(target_temp['time'].values[i])
            ts = pd.to_datetime(self._timestamps[-1])
            self._month.append(ts.month)
            self._daytime.append(ts.hour*60 + ts.minute)

            self._img_fns.append(str(image_fns.values[0, j]))

        self._target_temp = np.array(self._target_temp, dtype=np.float32)
        self._surface_temp = np.array(self._surface_temp, dtype=np.float32)
        self._radiation = np.array(self._radiation, dtype=np.float32)

        self._month = np.array(self._month, dtype=np.float32)
        self._daytime = np.array(self._daytime, dtype=np.float32)

        # Normalize regression values.
        self.target_temp_mean = self._target_temp.mean()
        self.target_temp_std = self._target_temp.std()

        self.surface_temp_mean = self._surface_temp.mean()
        self.surface_temp_std = self._surface_temp.std()

        self.radiation_mean = self._radiation.mean()
        self.radiation_std = self._radiation.std()

        self._target_temp = (self._target_temp - self.target_temp_mean) / \
            self.target_temp_std

        self._surface_temp = (self._surface_temp - self.surface_temp_mean) / \
            self.surface_temp_std

        self._radiation = (self._radiation - self.radiation_mean) / \
            self.radiation_std

        self._month = (self._month - self._month.mean()) / self._month.std()
        self._daytime = (self._month - self._daytime.mean()) / \
            self._daytime.std()

        print('dataset contains %d samples.' % len(self._img_fns))

    def __len__(self):
        return len(self._img_fns)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if isinstance(idx, list):
            # TODO read multiple images
            raise NotImplementedError()
        else:
            img = Image.open(io.BytesIO(self._img_store[self._img_fns[idx]]))
            img = img.rotate(90, expand=1)
            data = np.array(img.convert('RGB')).transpose([2, 0, 1])
            data = data.astype(np.float32)

            ts =  self._timestamps[idx]

        sample = {
            'img': data,
            'surface_temp': self._surface_temp[idx].reshape(-1, 1),
            'target_temp': self._target_temp[idx].reshape(-1, 1),
            'radiation': self._radiation[idx].reshape(-1, 1),
            'month': self._month[idx].reshape(-1, 1),
            'daytime': self._daytime[idx].reshape(-1, 1),
            # Just for the user, not meant to be used as input to a neural net.
            #'timestamp': ts,
            'idx': idx
        }

        return sample