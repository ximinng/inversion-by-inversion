# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:
from typing import (Union, Tuple, List, Dict, Iterable, Callable, Optional)

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


#################################################################################
#                                ImageNet 1K                                    #
#################################################################################

def build_imagenet(
        data_path: Union[str, List],
        image_size: int,
        split: str = 'train',
        data_transforms: Union[transforms.Compose, torch.nn.Module, Dict] = None,
        get_dataloader: bool = True,
        batch_size: int = 4,
        shuffle: bool = False,
        drop_last: bool = True,
        pin_memory: bool = True,
        num_workers: int = 1,
):
    assert split in ['train', 'test', 'val']

    from .imagenet import ImageNetDataset

    if split in ['train']:
        if data_transforms is None:
            train_transforms = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            train_transforms = data_transforms

        dataset = ImageNetDataset(root=data_path, split='train', transform=train_transforms)

    elif split in ['test', 'val']:
        if data_transforms is None:
            test_transforms = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            test_transforms = data_transforms

        dataset = ImageNetDataset(root=data_path, split="val", transform=test_transforms)

    assert len(dataset) > 0, print("the number of dataset cannot be 0.")
    num_classes = len(dataset.classes)

    if not get_dataloader:
        return dataset, num_classes
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        return dataloader, num_classes


#################################################################################
#                         Image-to-image translation                            #
#################################################################################

def build_image2image_translation_dataset(
        dataset: str,
        data_path: Union[str, List],
        split: str = 'train',
        data_transforms: Optional[Union[transforms.Compose, torch.nn.Module, Dict]] = None,
        image_size: int = None,
        get_dataloader: bool = True,
        batch_size: int = 4,
        shuffle: bool = False,
        drop_last: bool = True,
        pin_memory: bool = True,
        num_workers: int = 1,
) -> Tuple[Dataset, Dataset, int] or Tuple[Dataset, None, int]:
    assert dataset in ['cat2dog', 'wild2dog', 'afhq_multi2dog', 'male2female']
    assert split in ['train', 'test', 'val']

    """build dataset"""
    from .base_dataset import MultiDomainDataset

    if split in ['train'] and data_transforms is None:
        assert image_size is not None and image_size >= 0
        transform = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
    elif split in ['test', 'val'] and data_transforms is None:
        assert image_size is not None and image_size >= 0
        transform = transforms.Compose([
            transforms.Resize([image_size, image_size]),
            transforms.ToTensor()
        ])
    else:
        transform = data_transforms

    dataset = MultiDomainDataset(data_path, transform)
    assert len(dataset) > 0, print("the number of dataset cannot be 0.")

    if not get_dataloader:
        return dataset
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        return dataloader


#################################################################################
#                               Any dataset                                     #
#################################################################################

def load_data_folders(
        data_paths: Union[str, List],
        image_size: int = None,
        use_transforms: bool = True,
        data_transforms: Optional[Union[transforms.Compose, torch.nn.Module, object]] = None,
        img_convert: str = "RGB",
        get_dataloader: bool = True,
        batch_size: int = 4,
        shuffle: bool = False,
        drop_last: bool = False,
        pin_memory: bool = True,
        num_workers: int = 1,
) -> Union[Dataset, DataLoader]:
    """build dataset"""
    from .base_dataset import MultiDomainDataset

    if use_transforms:
        if data_transforms is None:
            assert image_size is not None and image_size > 0
            data_transforms = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.RandomCrop(image_size),
                transforms.ToTensor(),
            ])
        else:
            data_transforms = data_transforms

    dataset = MultiDomainDataset(data_paths, data_transforms, img_convert=img_convert)
    assert len(dataset) > 0, "the number of dataset cannot be 0."

    if not get_dataloader:
        return dataset
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        return dataloader


#################################################################################
#                              Dummy dataset                                    #
#################################################################################

def build_dummy_dataset(
        size: int = 1000,
        image_shape: Tuple[int, int, int] = (3, 224, 224),
        num_classes: int = 10,
        transform: Optional[Callable] = None,
        random_offset: int = 0,
):
    from torchvision.datasets import FakeData

    if transform is None:
        dataset = FakeData(size, image_shape, num_classes, transforms.ToTensor(), random_offset=random_offset)
    else:
        dataset = FakeData(size, image_shape, num_classes, transform, random_offset=random_offset)
    return dataset


#################################################################################
#                              Concat datasets                                  #
#################################################################################

def concat_dataset(datasets: Iterable[Dataset]):
    from torch.utils.data.dataset import ConcatDataset

    merged_dataset = ConcatDataset(datasets)
    return merged_dataset
