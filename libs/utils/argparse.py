# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import os
from typing import Tuple
from functools import reduce

import argparse
from argparse import Namespace
from omegaconf import DictConfig, OmegaConf


#################################################################################
#                            practical argparse utils                           #
#################################################################################

def accelerate_parser():
    parser = argparse.ArgumentParser(add_help=False)

    # Device
    parser.add_argument("-cpu", "--use_cpu", action="store_true",
                        help="Whether or not disable cuda")

    # Gradient Accumulation
    parser.add_argument("-cumgard", "--gradient-accumulate-step",
                        type=int, default=1)
    parser.add_argument("--split-batches", action="store_true",
                        help="Whether or not the accelerator should split the batches "
                             "yielded by the dataloaders across the devices.")

    # Nvidia-Apex and GradScaler
    parser.add_argument("-mprec", "--mixed-precision",
                        type=str, default='no', choices=['no', 'fp16', 'bf16'],
                        help="Whether to use mixed precision. Choose"
                             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
                             "and an Nvidia Ampere GPU.")
    parser.add_argument("--init-scale",
                        type=float, default=65536.0,
                        help="Default value: `2.**16 = 65536.0` ,"
                             "For ImageNet experiments, '2.**20 = 1048576.0' was a good default value."
                             "the others: `2.**17 = 131072.0` ")
    parser.add_argument("--growth-factor", type=float, default=2.0)
    parser.add_argument("--backoff-factor", type=float, default=0.5)
    parser.add_argument("--growth-interval", type=int, default=2000)

    # Gradient Normalization
    parser.add_argument("-gard_norm", "--max_grad_norm", type=float, default=-1)

    # Trackers
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--project-name", type=str, default="SketchGeneration")
    parser.add_argument("--entity", type=str, default="ximinng")
    parser.add_argument("--tensorboard", action="store_true")

    # reproducibility
    parser.add_argument("-d", "--seed", default=42, type=int)

    # result path
    parser.add_argument("-respath", "--results_path",
                        type=str, default="",
                        help="If it is None, it is automatically generated.")

    # timing
    parser.add_argument("-log_step", "--log_step", default=1000, type=int,
                        help="can be use to control log.")
    parser.add_argument("-eval_step", "--eval_step", default=5000, type=int,
                        help="can be use to calculate some metrics.")
    parser.add_argument("-save_step", "--save_step", default=5000, type=int,
                        help="can be use to control saving checkpoint.")

    # update configuration interface
    # example: python main.py -c main.yaml -update "nnet.depth=16 batch_size=16"
    parser.add_argument("-update",
                        type=str, default=None,
                        help="modified hyper-parameters of config file.")
    return parser


def ema_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ema', action='store_true', help='enable EMA model')
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--ema_update_after_step", type=int, default=100)
    parser.add_argument("--ema_update_every", type=int, default=10)
    return parser


def base_data_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-spl", "--split",
                        default='test', type=str,
                        choices=['train', 'val', 'test', 'all'],
                        help="which part of the data set, 'all' means combine training and test sets.")
    parser.add_argument("-j", "--num_workers",
                        default=6, type=int,
                        help="how many subprocesses to use for data loading.")
    parser.add_argument("--shuffle",
                        action='store_true',
                        help="how many subprocesses to use for data loading.")
    parser.add_argument("--drop_last",
                        action='store_true',
                        help="how many subprocesses to use for data loading.")
    return parser


def base_training_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-tbz", "--train_batch_size",
                        default=32, type=int,
                        help="how many images to sample during training.")
    parser.add_argument("-lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("-wd", "--weight_decay", default=0, type=float)
    return parser


def base_sampling_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-vbz", "--valid_batch_size",
                        default=1, type=int,
                        help="how many images to sample during evaluation")
    parser.add_argument("-ts", "--total_samples",
                        default=2000, type=int,
                        help="the total number of samples, can be used to calculate FID.")
    parser.add_argument("-ns", "--num_samples",
                        default=4, type=int,
                        help="number of samples taken at a time, "
                             "can be used to repeatedly induce samples from a generation model "
                             "from a fixed guided information, "
                             "eg: `one latent to ns samples` (1 latent to 5 photo generation) ")
    return parser


#################################################################################
#                             merge yaml and argparse                           #
#################################################################################

def register_resolver():
    OmegaConf.register_new_resolver(
        "add", lambda *numbers: sum(numbers)
    )
    OmegaConf.register_new_resolver(
        "multiply", lambda *numbers: reduce(lambda x, y: x * y, numbers)
    )
    OmegaConf.register_new_resolver(
        "sub", lambda n1, n2: n1 - n2
    )


def _merge_args_and_config(
        cmd_args: Namespace,
        yaml_config: DictConfig,
        read_only: bool = False
) -> Tuple[DictConfig, DictConfig, DictConfig]:
    # convert cmd line args to OmegaConf
    cmd_args_dict = vars(cmd_args)
    cmd_args_list = []
    for k, v in cmd_args_dict.items():
        cmd_args_list.append(f"{k}={v}")
    cmd_args_conf = OmegaConf.from_cli(cmd_args_list)

    # The following overrides the previous configuration
    # cmd_args_list > configs
    args_ = OmegaConf.merge(yaml_config, cmd_args_conf)

    if read_only:
        OmegaConf.set_readonly(args_, True)

    return args_, cmd_args_conf, yaml_config


def merge_configs(args, method_cfg_path):
    """merge command line args (argparse) and config file (OmegaConf)"""
    yaml_config_path = os.path.join("./", "config", method_cfg_path)
    try:
        yaml_config = OmegaConf.load(yaml_config_path)
    except FileNotFoundError as e:
        print(f"error: {e}")
        print(f"input file path: `{method_cfg_path}`")
        print(f"config path: `{yaml_config_path}` not found.")
        raise FileNotFoundError(e)
    return _merge_args_and_config(args, yaml_config, read_only=False)


def update_configs(source_args, update_nodes, strict=True, remove_update_nodes=True):
    """update config file (OmegaConf) with dotlist"""
    if update_nodes is None:
        return source_args

    update_args_list = str(update_nodes).split()
    if len(update_args_list) < 1:
        return source_args

    # check update_args
    for item in update_args_list:
        item_key_ = str(item).split('=')[0]  # get key

        if strict:
            # Tests if a key is existing
            assert OmegaConf.select(source_args, item_key_) is not None, f"{item_key_} is not existing."
            # Tests if a value is missing
            assert not OmegaConf.is_missing(source_args, item_key_), f"the value of {item_key_} is missing."

    # merge
    update_nodes = OmegaConf.from_dotlist(update_args_list)
    merged_args = OmegaConf.merge(source_args, update_nodes)

    # remove update_args
    if remove_update_nodes:
        OmegaConf.update(merged_args, 'update', '')
    return merged_args


def update_if_exist(source_args, update_nodes):
    """update config file (OmegaConf) with dotlist"""
    if update_nodes is None:
        return source_args

    source_args_list = str(update_nodes).split()
    if len(source_args_list) < 1:
        return source_args

    update_args_list = []
    for item in source_args_list:
        item_key_ = str(item).split('=')[0]  # get key

        # if a key is existing
        if OmegaConf.select(source_args, item_key_) is not None:
            update_args_list.append(item)

    # update source_args if key be selected
    if len(update_args_list) < 1:
        merged_args = source_args
    else:
        update_nodes = OmegaConf.from_dotlist(update_args_list)
        merged_args = OmegaConf.merge(source_args, update_nodes)

    return merged_args


def merge_and_update_config(args):
    register_resolver()

    # if yaml_config is exist, then merge command line args and yaml_config
    # if os.path.isfile(args.config) and args.config is not None:
    if args.config is not None and str(args.config).endswith('.yaml'):
        merged_args, cmd_args, yaml_config = merge_configs(args, args.config)
    else:
        merged_args, cmd_args, yaml_config = args, args, None

    # update the yaml_config with the cmd '-update' flag
    update_nodes = args.update
    final_args = update_configs(merged_args, update_nodes)

    # for logs
    yaml_config_update = update_if_exist(yaml_config, update_nodes)
    cmd_args_update = update_if_exist(cmd_args, update_nodes)
    cmd_args_update.update = ""  # clear update params

    final_args.yaml_config = yaml_config_update
    final_args.cmd_args = cmd_args_update

    # update seed
    if final_args.seed <= -1:
        import random
        final_args.seed = random.randint(0, 65535)

    return final_args
