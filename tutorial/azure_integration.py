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
from stuett.global_config import get_setting, setting_exists

from azure.storage.blob import (
    BlockBlobService,
)  # make sure to install 2.1 version with `pip install azure-storage-blob==2.1.0`
import zarr


# Get the account name which is "storageaccountperma8980" for the hackathon
# If you stored it in a config file you it will beloaded
if setting_exists("azure"):
    account_name = get_setting("azure")["account_name"]
else:
    account_name = "storageaccountperma8980"
account_key = get_setting("azure")["account_key"] if setting_exists("azure") else None

if account_key is not None:
    print("using credentials")


# Create a blob service and list all data available
block_blob_service = BlockBlobService(
    account_name=account_name, account_key=account_key
)
print("\nList blobs in the container")
generator = block_blob_service.list_blobs("hackathon-on-permafrost")
for i, blob in enumerate(generator):
    print("\t Blob name: " + blob.name)
    if i == 5:
        break

print("List some documents")
# In stuett we can use a a zarr store and load the data from there
store = stuett.ABSStore(
    container="hackathon-on-permafrost",
    prefix="docs/",
    account_name=account_name,
    account_key=account_key,
    blob_service_kwargs={},
)

for i, key in enumerate(store.keys()):
    print(key)
    if i == 5:
        break

# # Currently, stuett (or zarr in the backend) only support azure-storage-blob==2.1.0`
# # But a newer version is available which you can use independently
# # using azure-storage-blob==12.0.0
# # untested
# from azure.storage.blob import BlobServiceClient
# service = BlobServiceClient(account_url="https://storageaccountperma8980.blob.core.windows.net/",credentials=account_key)
# client = service.get_container_client('hackathon-on-permafrost')
# blob_client = client.get_blob_client(blob='dataset/timeseries_derived_data_products/MH30_temperature_rock_2017.csv')
