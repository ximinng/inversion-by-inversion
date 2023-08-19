# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:


import os
from PIL import Image
from typing import Any, Callable, List, Optional, Union

from omegaconf import ListConfig
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from .utils import is_image_file


#################################################################################
#                              abstract class                                   #
#################################################################################

class VisionDataset(Dataset):

    def __init__(
            self,
            path: str,
            split: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """ Initialize the class; save the options in the class

        Args:
            path: data path
            split: "train", "test" or combining "train" and "test".
        """
        self.path = path
        self.split = split
        assert split in ["train", "test", "all"], f"{split} not exist."

        # for backwards-compatibility
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError


class VisionMixinSketchDataset(Dataset):

    def __init__(
            self,
            path: str,
            split: str,
            photo_transform: Optional[Callable] = None,
            sketch_transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """ Initialize the class; save the options in the class

        Args:
            path: data path
            split: "train", "test" or combining "train" and "test".
        """
        self.path = path
        self.split = split
        assert split in ["train", "test", "all"], f"{split} not exist."

        # for backwards-compatibility
        self.photo_transform = photo_transform
        self.sketch_transform = sketch_transform

        self.target_transform = target_transform

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            (Any): Sample and meta data, optionally transformed by the respective transforms.
        """
        raise NotImplementedError


#################################################################################
#                                  utils                                        #
#################################################################################

def _make_dataset(dir: Union[List, str], max_dataset_size=float("inf")):
    images = []
    if isinstance(dir, list):
        for i in range(len(dir)):
            dir_i = dir[i]
            for root, _, fnames in sorted(os.walk(dir_i, followlinks=True)):
                for fname in fnames:
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        images.append(path)
    else:
        for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
    return images[:min(max_dataset_size, len(images))]


class SingleDomainDataset(Dataset):
    """
    Takes a dataset path as input and returns the images only.

    Args:
        path: dataset path
        transform(optional): data transformations
   """

    def __init__(self, path: str, transform: Union[transforms.Compose, nn.Module]):
        self.paths = sorted(_make_dataset(path))
        self.size = len(self.paths)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img


class SingleDomainWithFileNameDataset(Dataset):
    """
    Takes a dataset path as input and returns the images and img_file name.

    Args:
        path: dataset path
        transform(optional): data transformations
    """

    def __init__(self, path: str, transform: Union[transforms.Compose, nn.Module]):
        self.paths = sorted(_make_dataset(path))
        self.size = len(self.paths)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')

        fname = os.path.basename(path)
        fname_wo_extension = os.path.splitext(fname)[0]

        if self.transform is not None:
            img = self.transform(img)
        return img, fname_wo_extension


class MultiDomainDataset(Dataset):
    """
    Takes a list of dataset path as input and returns a concatenated dataset.
    Noting: use the folder name as the label name.

    Args:
        paths: a list contains dataset paths
        transform(optional): data transformations
    """

    def __init__(self,
                 paths: Union[List, str],
                 transform: Union[transforms.Compose, nn.Module],
                 img_convert: str = "RGB"):

        labels = []
        label_idx = 0
        if isinstance(paths, list) or isinstance(paths, ListConfig):
            paths = list(paths)
            self.paths = []
            for domain_i in range(len(paths)):
                path_i = sorted(_make_dataset(paths[domain_i]))
                assert len(path_i) >= 0, print(f"data path: {paths[domain_i]} is empty.")
                self.paths = self.paths + path_i
                label_di = [str(domain_i).split("/")[0] for i in range(len(path_i))]
                labels = labels + label_di
                label_idx += 1
        else:
            self.paths = sorted(_make_dataset(paths))
            # use the folder name as the label name
            labels = [str(paths).split("/")[0] for i in range(len(self.paths))]

        self.classes = set(labels)
        self.class_to_idx = {clss: idx for idx, clss in enumerate(self.classes)}
        self.label = [self.class_to_idx[l] for l in labels]
        self.size = len(self.paths)

        self.transform = transform
        self.img_convert = img_convert

        self.get_meta_info()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.paths[index]
        fname = path.split('/')[-1]

        label = self.label[index]
        if self.img_convert == 'gray':
            img = Image.open(path).convert('L')
        elif self.img_convert == 'RGB':
            img = Image.open(path).convert(self.img_convert)
        else:
            img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        sample = {
            "image": img, "label": label, "fname": fname, "path": path
        }
        return sample

    def get_meta_info(self):
        print(
            f"{str('-' * 50)}\n"
            f"Total number of samples: {self.size}\n"
            f"Total number of classes: {len(self.classes)}\n"
            f"{str('-' * 50)}"
        )
