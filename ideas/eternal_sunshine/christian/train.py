#!/usr/bin/env python3
# Copyright 2019 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train a resnet on images and time series data to predict below surface
temperatures.

Note, this implementation does not represent a Bayesian Neural Network yet (not
enough time).

The timeseries data is fed into the network via a fully-connected network that
produces the batch norm weights.
"""

from argparse import Namespace
import numpy as np
import random
from time import time
import torch
from torch.utils.data import DataLoader

from regression_dataset import PermaRegressionDataset
from resnet import ResNet
from simple_bn_generator import BNGenerator

if __name__ == '__main__':
    script_start = time()

    # FIXME
    ### User config
    args = Namespace()
    args.batch_size = 32
    args.lr = 0.01
    args.random_seed = 42
    args.local = False
    args.num_workers = 4

    ### Deterministic computation
    # Note, doesn't hold when using GPU or multiple workers that load the
    # dataset.
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    ### Select device.
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Using cuda: ' + str(use_cuda))

    ### Generate networks.
    # FIXME Downsample images to speed up computation.
    img_net = ResNet(in_shape=[536, 356, 3], num_outs=1, no_weights=False,
                     use_batch_norm=True)

    # We include the following sensory information:
    # - surface temperature
    # - radiation
    # - month
    # - daytime
    n_in = 4

    # Shapes of batchnorm layer weights.
    # We will produce these weights with an auxiliary fully-connected network,
    # that takes the timeseries data as input.
    bn_shapes = []
    for l in img_net._batchnorm_layers:
        bn_shapes.extend(l.param_shapes)

    # FIXME Our current implementation doesn't allow efficient batch processing.
    # Neither the underlying hnet allows the usage of multiple task embeddings
    # nor does the batch norm layer support a batch of weights.
    ts_net = BNGenerator(bn_shapes, 1, layers=[100,100], te_dim=n_in)

    ### Generate datasets.
    train_data = PermaRegressionDataset(args.local,
        time_slice={"start_time": "2017-01-01", "end_time": "2017-06-30"})
    test_data = PermaRegressionDataset(args.local,
                                       time_slice={"start_time": "2017-07-01",
                                                   "end_time": "2017-07-31"})

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    ### Train and test.
    raise NotImplementedError('Training and testing not implemented yet.')

    print('Program finished successfully in %f sec.' % (time() - script_start))