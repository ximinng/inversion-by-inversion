# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import os
from argparse import Namespace
from typing import Union, Dict
from functools import lru_cache
from omegaconf import OmegaConf

__all__ = ["photo2sketch_model_build_util", "photo2sketch_available_models"]

_METHODS = ["PhotoSketching", "InformativeDrawings"]


def photo2sketch_available_models():
    return _METHODS


@lru_cache()
def default_config_path(dir_name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name, "default_config.yaml")


def photo2sketch_model_build_util(
        method: str = "PhotoSketching",
        model_config: Union[Namespace, Dict] = None
):
    assert method in _METHODS, f"Model {method} not recognized."

    if model_config is None:  # load default configuration
        config_path = default_config_path(method)
        model_config = OmegaConf.load(config_path)

    if method == "PhotoSketching":
        from .PhotoSketching.networks import ResnetGenerator, get_norm_layer
        norm_layer = get_norm_layer(norm_type=model_config.norm)
        model = ResnetGenerator(model_config.input_nc, model_config.output_nc,
                                model_config.ngf, norm_layer, model_config.use_dropout,
                                model_config.n_blocks)
        return model
    elif method == "InformativeDrawings":
        from .InformativeDrawings.model import Generator
        model = Generator(model_config.input_nc, model_config.output_nc, model_config.n_blocks)
        return model
    else:
        raise ModuleNotFoundError("Model [%s] not recognized." % method)
