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
from stuett.data import annotations_to_slices

import warnings
import sys, os
from os.path import join
import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from ast import literal_eval as make_tuple
from tqdm import tqdm 

from torch import Tensor
from scipy.io import wavfile
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from PIL import Image
import io

from pathlib import Path 

class PytorchDataset(stuett.data.SegmentedDataset):
    def __init__(self, data=None, label=None, label_list_file=None, store=None, transform=None, mode="train", dataset_slice=None, batch_dims=None,random_seed=1045,train_split=0.7 ):
        super().__init__(data=data, label=label,dataset_slice=dataset_slice,batch_dims=batch_dims, discard_empty=True)

        self.store = store
        self.transform = transform

        self.label_list = None
        if isinstance(label_list_file,str) or isinstance(label_list_file,Path):
            self.load_list(label_list_file)
        elif isinstance(label_list_file,pd.DataFrame):
            self.label_list = label_list_file
        if self.label_list is None:
            print('Computing labels. This might take long...')
            # if something we couldn't load a file
            # recompute and save
            self.label_list = self.compute_label_list()
            self.store_list(label_list_file)

        # Train and test set must get the same list otherwise it's not guaranteed
        # that sets are non-overlapping
        np.random.seed(random_seed)
        indices = np.arange(len(self.label_list))  
        np.random.shuffle(indices)
        split = np.floor(len(self.label_list)*train_split).astype(np.int)

        if mode is "train":
            self.label = self.label_list.iloc[indices[:split]]
        elif mode is "test":
            self.label = self.label_list.iloc[indices[split:]]

    def store_list(self,path):
        row_list = []
        print(self.label_list)
        for key, row in self.label_list.iterrows():
            row_dict = {}
            row_dict['id'] = key
            for dim in row['indexers']:
                start_item  = 'start_'+str(dim)
                end_item  = 'end_'+str(dim)
                row_dict[start_item] = str(row['indexers'][dim].start)
                row_dict[end_item] = str(row['indexers'][dim].stop)
            for label in row['labels']:
                if pd.isnull(label):
                    label = ''
                row_dict_copy = row_dict.copy()
                row_dict_copy['__target'] = label
                row_list.append(row_dict_copy)

        df = pd.DataFrame(row_list)
        df.to_csv(path,index=False)
            
    def load_list(self,path):
        try: 
            x = stuett.data.BoundingBoxAnnotation(filename=path)()
        except:
            return

        dims, slices = annotations_to_slices(x)

        classes = []
        row_dict = {}
        for i in range(len(x)):
            j = x['id'].values[i]
            if j not in row_dict:
                row_dict[j] = {}
                row_dict[j]['indexers'] = slices[i]
                row_dict[j]['labels'] = [x.values[i]]
                
            else:
                row_dict[j]['labels'] += [x.values[i]]
            if x.values[i] not in classes and pd.notnull(x.values[i]):
                classes.append(x.values[i])
        df = pd.DataFrame([row_dict[key] for key in row_dict])
        self.label_list = df
        self.classes = {class_name: i for i, class_name in enumerate(classes)}

    def __len__(self):
        return len(self.label)

class ImageDataset(PytorchDataset):
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label_info  = self.label.iloc[idx]
        indexers    = label_info['indexers']
        label       = label_info['labels']

        target = np.zeros((len(self.classes),))
        for l in label:
            if pd.notnull(l):
                target[self.classes[str(l)]] = 1

        filenames = self.get_data(indexers)
        # print(filenames)
        if filenames.size > 1:
            key = str(filenames.squeeze().values[0])
        else:
            key = str(filenames.squeeze().values)

        # print(key)
        img = Image.open(io.BytesIO(self.store[key]))
        data = np.array(img.convert('RGB')).transpose([2,0,1])
        data = data.astype(np.float32)

        if self.transform is not None:
            data = self.transform(data)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return data, target        


class SeismicDataset(PytorchDataset):
    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        label_info = self.label.iloc[idx]
        indexers   = label_info['indexers']
        label   = label_info['labels']

        target = np.zeros((len(self.classes),))
        for l in label:
            if pd.notnull(l):
                target[self.classes[l]] = 1

#        print(label_info)

        # print(indexers['time'])
        data = self.get_data(indexers)
        
        if self.transform is not None:
            data = self.transform(data)

        # if "shape" in self.__dict__:
        #     self.shape = data.shape
        # elif data.shape != self.shape:
        #     warnings.warn(f"Inconsistency in the data for item {indexers['time']}, its shape {data.shape} does not match shape {shape}")
        #     padded_data = torch.zeros(self.shape)
        #     pad = data.shape - self.shape
        #     padded_data = torch.nn.functional.pad(data)

        return data, target


class DatasetMerger(Dataset):
    def __init__(self,list_of_datasets):
        self.__list_of_datasets = list_of_datasets


        self.__info = np.zeros((len(self),2),dtype=np.int)

        set_index = 0
        for i, dataset in enumerate(self.__list_of_datasets):
            self.__info[set_index:set_index+len(dataset),0] = np.arange(0,len(dataset))
            self.__info[set_index:set_index+len(dataset),1] = i
            set_index = len(dataset)

    def get_dataset(self,idx):
        """get dataset given the idx
        
        Arguments:
            idx {integer} -- the index for which the dataset should be retrieved
        """
        dataset_id = self.__info[idx,1]
        return self.__list_of_datasets[dataset_id] 

    def __len__(self):
        length = 0
        for dataset in self.__list_of_datasets:
            length += len(dataset)
        return length

    def __getitem__(self, idx):
        dataset_id = self.__info[idx,1]
        dataset_idx = self.__info[idx,0]
        return self.__list_of_datasets[dataset_id][dataset_idx]



class DatasetFreezer(Dataset):
    def __init__(self, dataset, path=None, ignore_list=[], bypass=False):
        """Given a dataset the DatasetFreezer has the capability to store the (preprocessed) dataset to disk.

        
        Arguments:
            dataset {[type]} -- The dataset to be frozen
        """
        self.__dataset = dataset
        self.__frozen = False
        self.__path = Path(path)
        self.__ignore_list = ignore_list
        self.__bypass = bypass

        # if path is not None:
        #     self.freeze()

    def freeze(self,reload=False):
        if self.__bypass:
            return

        os.makedirs(self.__path,exist_ok=True)

        # if not os.path.isdir(self.__path):
        #     raise FileNotFoundError('The folder {} cannot be found.'.format(self.__path))

        # TODO: parquet is not ideal, rather go for zarr
        filename = self.__path.joinpath('{}.parquet'.format('datafile'))
        if os.path.isfile(filename) and not reload:
            self.parquet_file = pq.ParquetFile(filename)
            assert len(self) == self.parquet_file.num_row_groups, 'Potentially corrupted file: Preprocessed data file does not match the length of the dataset'
            self.__frozen = True
            return

        pqwriter = None
        for i in tqdm(range(len(self))):
            value = self[i]               
            df = pd.DataFrame()
            if isinstance(value,tuple):
                for j,element in enumerate(value):
                    df['data_%d'%j] = [np.array(element).flatten()]
            else:
                df['data_0'] = [np.array(value).flatten()]

            table = pa.Table.from_pandas(df)
            metadata = table.schema.metadata
            if metadata is None:
                metadata = collections.OrderedDict()

            if isinstance(value,tuple):
                for j,element in enumerate(value):
                    metadata['shape_%d'%j] = str(np.array(element).shape)
            else:
                metadata['shape_0'] = str(np.array(value).shape)

            table = table.replace_schema_metadata(metadata)

            if i == 0:
                pqwriter = pq.ParquetWriter(filename, table.schema)
                pqwriter.write_table(table)
            else:
                pqwriter.write_table(table)

        if pqwriter:
            pqwriter.close()

        self.parquet_file = pq.ParquetFile(filename)
        self.__frozen = True
        # assert len(self) == self.parquet_file.num_row_groups

    def get_frozen(self,idx):
        row_group = self.parquet_file.read_row_group(idx)

        row_df = row_group.to_pandas()

        # print(row_df)
        # return_tuple = ()
        return_list = []
        for j, element in enumerate(list(row_df)):
            data_shape = make_tuple(str(row_group.schema.metadata[b'shape_%d'%j].decode('ascii')))
            tensor = Tensor(row_df['data_%d'%j].values[0].reshape(data_shape))
            # return_tuple += tuple(tensor)
            return_list.append(tensor)

        
        # print(return_tuple)
        # print(type(return_tuple))
        # TODO: make this more generic. Currenlty, we are only expecting data or data+target
        if len(return_list) == 1:
            return return_list[0]
        else:
            # return return_list[0], torch.Tensor([self.targets[idx]])
            return return_list[0], return_list[1]


    def __getattr__(self, item):
       result = getattr(self.__dataset, item)
    #    if callable(result):
    #        result = tocontainer(result)
       return result

    def __len__(self):
        return len(self.__dataset)

    def __getitem__(self, idx):
        if self.__frozen and not self.__bypass:
            return self.get_frozen(idx)

        return self.__dataset[idx]
