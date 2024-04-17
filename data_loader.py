import open3d as o3d
import os, sys, glob
import time
from tqdm import tqdm
import numpy as np
import h5py
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME
from data_processing.data_utils import read_h5, read_h5_label, read_ply_ascii

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None: new_list_data.append(data)
        else: num_removed += 1
    list_data = new_list_data
    if len(list_data) == 0: raise ValueError('No data in the batch')
    if len(list_data[0])==2:
        coords, feats = list(zip(*list_data))
        coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

        return coords_batch, feats_batch

    elif len(list_data[0])==3:
        coords, feats, label = list(zip(*list_data))
        coords_batch, feats_batch, label_batch = ME.utils.sparse_collate(coords, feats, label)

        return coords_batch, feats_batch, label_batch


class PCDataset(torch.utils.data.Dataset):

    def __init__(self, files, have_label=False):
        self.files = []
        self.cache = {}
        # self.last_cache_percent = 0
        self.files = files
        self.have_label = have_label

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir = self.files[idx]
        if not self.have_label:
            if idx in self.cache:
                coords, feats = self.cache[idx]
            else:
                if filedir.endswith('.h5'): coords, feats = read_h5(filedir)
                if filedir.endswith('.ply'): coords, feats = read_ply_ascii(filedir)
                # cache
                self.cache[idx] = (coords, feats)
                # cache_percent = int((len(self.cache) / len(self)) * 100)
            coords = coords.astype("int32")
            feats = feats.astype("float32")/255.

            return (coords, feats)

        else:
            if idx in self.cache:
                coords, feats, label = self.cache[idx]
            else:
                coords, feats, label = read_h5_label(filedir)
                # cache
                self.cache[idx] = (coords, feats, label)
                # cache_percent = int((len(self.cache) / len(self)) * 100)
            coords = coords.astype("int32")
            feats = feats.astype("float32")/255.
            label = label.astype("float32")/255.

            return (coords, feats, label)


def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False, 
                    collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader

