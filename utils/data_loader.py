# -- coding : utf-8 -*-

r"""
Copyright (c) 2023-present, Dankook Univ, South Korea
Allright Reserved.
Author : Sooin Kim
Last modified : 2023.03.09
Code version : 1.0
"""

import h5py
import os

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self,
        data_path:str,
        scaling:bool=True,
        class_converter:bool=True,
        label_converter:bool=True
    ):
        
        self.data_path = data_path
        assert os.path.isfile(self.data_path) is True, 'There is no data file in your path. Please make sure there is any files'
     
        self.scaling=scaling
        self.class_converter=class_converter
        self.label_converter=label_converter


    def __len__(self):
        
        with h5py.File(self.data_path,'r') as dataset:
            length = len(dataset['inputs'])
        dataset.close()

        return length
    
    
    def __getitem__(self,
            index: int
            ):
        
        with h5py.File(self.data_path,'r') as dataset:

            features = dataset['inputs'][f'{index}']
            features = features[()]
            features = torch.from_numpy(features)
            features = torch.unsqueeze(features, 0)

            labels = dataset['outputs'][f'{index}']
            labels = labels[()]
            labels = torch.from_numpy(labels)

            classes = dataset['classes'][f'{index}']
            classes = classes[()]
            classes = torch.from_numpy(classes)

        if self.scaling:
            features = features/255.

        
        if self.class_converter:
            all_classes = list()
            for k in classes:
                idx = [0,0,0,0,0,0,0,0]
                idx[int(k)] = 1
                all_classes.append(idx)
            classes = torch.tensor(all_classes)

        if self.label_converter:
            labels = torch.argmax(labels, dim=0)

        return features, classes, labels
        