# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

# sketch dataset
from .mnist import MNISTDataset
from .sketchx_shoe_chairV2 import SketchXShoeAndChairCoordDataset, SketchXShoeAndChairPhotoDataset
from .sketchy import SketchyDataset
# real image dataset
from .cifar10 import CIFAR10Dataset
from .imagenet import ImageNetDataset
# common
from .base_dataset import MultiDomainDataset, SingleDomainDataset, SingleDomainWithFileNameDataset

# utils
from .base_dataset import is_image_file
