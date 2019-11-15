from azure.storage.blob import  BlockBlobService # make sure to install 2.1 version with `pip install azure-storage-blob==2.1.0`
import zarr 
from stuett.global_config import get_setting, setting_exists


account_name = get_setting('azure')['account_name'] if setting_exists('azure') else "storageaccountperma8980"
account_key  = get_setting('azure')['account_key']  if setting_exists('azure')  else None

if account_key is not None:
    print('using credentials')

block_blob_service = BlockBlobService(account_name="storageaccountperma8980",account_key=account_key)
print("\nList blobs in the container")
generator = block_blob_service.list_blobs('hackathon-on-permafrost')
for blob in generator:
    print("\t Blob name: " + blob.name)


account_name = get_setting('azure')['account_name'] if setting_exists('azure') else "storageaccountperma8980"
account_key  = get_setting('azure')['account_key']  if setting_exists('azure')  else None

store = zarr.ABSStore(container='hackathon-on-permafrost', prefix='dataset/', account_name=account_name, account_key=account_key, blob_service_kwargs={})
root = zarr.group(store=store, overwrite=False)

# # using azure-storage-blob==12.0.0
# # untested
# from azure.storage.blob import BlobServiceClient
# service = BlobServiceClient(account_url="https://storageaccountperma8980.blob.core.windows.net/",credentials=account_key)
# client = service.get_container_client('hackathon-on-permafrost')
# blob_client = client.get_blob_client(blob='dataset/timeseries_derived_data_products/MH30_temperature_rock_2017.csv')