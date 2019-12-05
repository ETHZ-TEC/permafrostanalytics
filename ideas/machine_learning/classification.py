"""MIT License

Copyright (c) 2019, Swiss Federal Institute of Technology (ETH Zurich), Matthias Meyer

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
SOFTWARE."""

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

parser = argparse.ArgumentParser(description="Pytorch Neural Network Classification")
parser.add_argument(
    "--classifier",
    type=str,
    default="image",
    help="Classification type either `image` or `seismic`",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="input batch size for training (default: 16)",
)
parser.add_argument(
    "--epochs", type=int, default=500, help="number of epochs to train (default: 500)"
)
parser.add_argument(
    "--lr", type=float, default=0.001, help="learning rate for optimizer"
)
parser.add_argument(
    "--linear_decrease_start_epoch",
    type=int,
    default=100,
    help="At which epoch to start the linear decrease",
)
parser.add_argument(
    "--use_frozen",
    action="store_true",
    default=False,
    help="Using cached/preprocessed dataset",
)
parser.add_argument(
    "--reload_frozen",
    action="store_true",
    default=False,
    help="Reloads the cached/preprocessed dataset",
)
parser.add_argument(
    "--reload_all",
    action="store_true",
    default=False,
    help="Reloads the cached/preprocessed dataset, the labels",
)
parser.add_argument(
    "--resume", type=str, default=None, help="Resume from given model checkpoint"
)
parser.add_argument(
    "--augment", action="store_true", default=False, help="augment data at runtime"
)
parser.add_argument(
    "--tmp_dir",
    default=str(
        Path(__file__).absolute().parent.joinpath("..", "..", "data", "user_dir", "tmp")
    ),
    help="folder to store logs and model checkpoints",
)
parser.add_argument(
    "--run_id",
    default=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
    help="id for this run. If not provided it will be the current timestamp",
)
parser.add_argument(
    "-p",
    "--path",
    type=str,
    default=str(Path(__file__).absolute().parent.joinpath("..", "..", "data/")),
    help="The path to the folder containing the permafrost hackathon data",
)
parser.add_argument(
    "-l",
    "--local",
    action="store_true",
    help="Only use local files and not data from Azure",
)
args = parser.parse_args()

################## PARAMETERS ###################
#################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_path = Path(args.path)
label_filename = "automatic_labels_mountaineers.csv"
tmp_dir = Path(args.tmp_dir)
os.makedirs(tmp_dir, exist_ok=True)


if args.classifier == "image":
    prefix = "timelapse_images_fast"
elif args.classifier == "seismic":
    prefix = "seismic_data/4D/"
else:
    raise RuntimeError("Please specify either `image` or `seismic` classifier")

if args.reload_all:
    args.reload_frozen = True

############ SETTING UP DATA LOADERS ############
#################################################
if not args.local:
    from stuett.global_config import get_setting, setting_exists

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
        prefix=prefix,
        account_name=account_name,
        account_key=account_key,
    )
    annotation_store = stuett.ABSStore(
        container="hackathon-on-permafrost",
        prefix="annotations",
        account_name=account_name,
        account_key=account_key,
    )
else:
    store = stuett.DirectoryStore(Path(data_path).joinpath(prefix))
    if (
        "2017-01-01/20170101_080018.JPG" not in store
        and "MH36/2017/EHE.D/4D.MH36.A.EHE.D.20171231_230000.miniseed" not in store
    ):
        raise RuntimeError(
            f"Please provide a valid path to the permafrost {prefix} data or see README how to download it"
        )
    annotation_store = stuett.DirectoryStore(Path(data_path).joinpath("annotations"))
    if label_filename not in annotation_store:
        print(
            "WARNING: Please provide a valid path to the permafrost annotation data or see README how to download it"
        )


################## START OF IDEA ################
#################################################
def get_seismic_transform():
    def to_db(x, min_value=1e-10, reference=1.0):
        value_db = 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, x))
        value_db -= 10.0 * xr.ufuncs.log10(xr.ufuncs.maximum(min_value, reference))
        return value_db

    spectrogram = stuett.data.Spectrogram(
        nfft=512, stride=512, dim="time", sampling_rate=1000
    )

    transform = transforms.Compose(
        [
            lambda x: x / x.max(),  # rescale to -1 to 1
            spectrogram,  # spectrogram
            lambda x: to_db(x).values.squeeze(),
            lambda x: Tensor(x),
        ]
    )

    return transform


def get_image_transform():
    # TODO: add image transformations
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = transforms.Compose([transforms.ToTensor(), normalize])

    return transform


########## Annotation Balancing #################
#################################################
# Load the labels
label = stuett.data.BoundingBoxAnnotation(
    filename=label_filename, store=annotation_store
)()
# we are not interest in any x or y position (since there are none in the labels)
label = label.drop_vars(["start_x", "end_x", "start_y", "end_y"])

# Currently, the dataset contains of only one label 'mountaineer'
# The labelled section without mountaineer outnumber the sections with one (approx 10:1)
# To train succesfully we need a balanced dataset of positive and negative examples
# Here, we balance it by choosing number of random non-mountaineer sections which
# is approximatly the same number as the mountaineer sections.
# NOTE: Adjust this section if you want to train with different label classes!!
no_label_mask = label.isnull()
label_mask = label.notnull()
ratio = (no_label_mask.sum() / label_mask.sum()).values.astype(int)
no_label_indices = np.argwhere(no_label_mask.values)[::ratio].squeeze()
label_mask[no_label_indices] = True
label = label[label_mask]
print("Number of labels which are checked against the data", len(label))

# here we load a predefined list from our server
# If you want to regenerate your list add reload_all as an argument to the script
label_list_file = tmp_dir.joinpath(f"{args.classifier}_list.csv").resolve()
if not label_list_file.exists() and not args.reload_all:
    # load from server
    with open(label_list_file, "wb") as f:
        f.write(annotation_store[f"{args.classifier}_list.csv"])


###### SELECTING A CLASSIFIER TYPE ##############
#################################################
# Load the data source
def load_seismic_source():
    seismic_channels = ["EHE", "EHN", "EHZ"]
    seismic_node = stuett.data.SeismicSource(
        store=store, station="MH36", channel=seismic_channels,
    )
    return seismic_node, len(seismic_channels)


def load_image_source():
    image_node = stuett.data.MHDSLRFilenames(
        store=store, force_write_to_remote=True, as_pandas=False,
    )
    return image_node, 3


if args.classifier == "image":
    from datasets import ImageDataset as Dataset

    transform = None
    data_node, num_channels = load_image_source()
elif args.classifier == "seismic":
    from datasets import SeismicDataset as Dataset

    transform = get_seismic_transform()
    data_node, num_channels = load_seismic_source()


############# LOADING DATASET ###################
#################################################
bypass_freeze = not args.use_frozen
print("Setting up training dataset")
train_dataset = Dataset(
    label_list_file=label_list_file,
    transform=transform,
    store=store,
    mode="train",
    label=label,
    data=data_node,
    dataset_slice={"time": slice("2017-01-01", "2017-12-31")},
    batch_dims={"time": stuett.to_timedelta(10, "minutes")},
)
print("Using cached training data: ", args.use_frozen)
train_frozen = DatasetFreezer(
    train_dataset, path=tmp_dir.joinpath("frozen", "train"), bypass=bypass_freeze
)
train_frozen.freeze(reload=args.reload_frozen)

print("Setting up test dataset")
train_dataset = Dataset(
    label_list_file=label_list_file,
    transform=transform,
    store=store,
    mode="test",
    label=label,
    data=data_node,
    dataset_slice={"time": slice("2017-01-01", "2017-12-31")},
    batch_dims={"time": stuett.to_timedelta(10, "minutes")},
)
print("Using cached test data: ", args.use_frozen)
test_frozen = DatasetFreezer(
    train_dataset, path=tmp_dir.joinpath("frozen", "test"), bypass=bypass_freeze
)
test_frozen.freeze(reload=args.reload_frozen)

# Set up pytorch data loaders
shuffle = True
train_sampler = None
train_loader = DataLoader(
    train_frozen,
    batch_size=args.batch_size,
    shuffle=shuffle,
    sampler=train_sampler,
    # drop_last=True,
    num_workers=0,
)

validation_sampler = None
test_loader = DataLoader(
    test_frozen,
    batch_size=args.batch_size,
    shuffle=shuffle,
    sampler=validation_sampler,
    # drop_last=True,
    num_workers=0,
)


def train(epoch, model, train_loader, writer):
    model.train()

    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader), 0):
        # get the inputs
        data = data
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # calculate loss and backpropagate
        loss = criterion(outputs, targets)
        loss.backward()
        # optimize
        optimizer.step()

        # for statistics
        running_loss += loss.item()

    writer.add_scalar(
        "Train Loss", running_loss / len(train_loader.sampler), int((epoch + 1))
    )
    print(
        "\nTrain set: Average loss: {:.4f}\n".format(
            running_loss / len(train_loader.sampler)
        )
    )


def test(epoch, model, test_loader, writer, embeddings=None):
    model.eval()
    test_loss = 0
    correct = 0

    acc = Accuracy()
    acc.reset()

    all_targets = []
    all_results = []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)

            # perform prediction
            output = model(data)

            test_loss += criterion(output, targets).item()

            # Since during training sigmoid is applied in BCEWithLogitsLoss
            # we also need to apply it here
            output = torch.sigmoid(output)

            # Make a hard decision threshold at 0.5
            output[output > 0.5] = 1
            output[output <= 0.5] = 0

            acc.update((output, targets))

    acc_value = acc.compute()
    test_loss /= len(test_loader.sampler)
    writer.add_scalar("Test Loss", test_loss, int((epoch + 1)))
    writer.add_scalar("Test Acc", acc_value, int((epoch + 1)))

    print(
        "Test set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n".format(
            test_loss, acc_value * 100
        )
    )


if __name__ == "__main__":
    writer = SummaryWriter(tmp_dir.joinpath(args.run_id, "log"))

    model = SimpleCNN(num_targets=len(train_dataset.classes), num_channels=num_channels)
    model = model.to(device)

    # we choose binary cross entropy loss with logits (i.e. sigmoid applied before calculating loss)
    # because we want to detect multiple events concurrently (at least later on, when we have more labels)
    criterion = nn.BCEWithLogitsLoss()

    # for most cases adam is a good choice
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    start_epoch = 0

    # optionally, resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("no checkpoint found at '{}'".format(args.resume))

    for epoch in range(0, args.epochs):
        if epoch > args.linear_decrease_start_epoch:
            for g in optimizer.param_groups:
                g["lr"] = args.lr - args.lr * (
                    epoch - args.linear_decrease_start_epoch
                ) / (args.epochs - args.linear_decrease_start_epoch)

        tqdm.write(str(epoch))
        tqdm.write("Training")
        train(epoch, model, train_loader, writer)
        tqdm.write("Testing")
        test(epoch, model, test_loader, writer)

        # is_best = True
        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        checkpoint_dir = tmp_dir.joinpath(args.run_id, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        filename = checkpoint_dir.joinpath("checkpoint.pth")
        torch.save(state, filename)
