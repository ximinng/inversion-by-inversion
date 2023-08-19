# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import os
import sys
import argparse

from accelerate.utils import set_seed

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from libs.utils.argparse import (merge_and_update_config, accelerate_parser, base_data_parser, base_sampling_parser)
from dpm_nn.guided_dpm.build_ADMs import ADMs_build_util
from sketch_nn.dataset.build import build_image2image_translation_dataset


def main(args):
    assert len(args.data_folder) > 0, "Insufficient dataset entry!"

    args.batch_size = args.valid_batch_size

    if args.task == "base":  # SDEdit - image to image translation
        from pipelines.inversion.SDEdit_pipeline import SDEditPipeline

        dataloader = build_image2image_translation_dataset(args.dataset, args.data_folder,
                                                           split=args.split,
                                                           image_size=args.image_size,
                                                           batch_size=args.valid_batch_size,
                                                           shuffle=args.shuffle, drop_last=True,
                                                           num_workers=args.num_workers)

        sde_model, _ = ADMs_build_util(args.image_size, args.num_classes, args.model, args.diffusion)

        SDEdit = SDEditPipeline(args, sde_model, args.sdepath, dataloader, args.use_dpm_solver)
        SDEdit.sample()

    elif args.task == "mask":  # TODO: SDEdit - image editing
        pass

    elif args.task == "ref":
        from pipelines.inversion.SDEdit_iter_pipeline import IterativeSDEditPipeline

        src_dataloader = build_image2image_translation_dataset(args.dataset, args.data_folder,
                                                               split=args.split,
                                                               image_size=args.image_size,
                                                               batch_size=args.valid_batch_size,
                                                               shuffle=args.shuffle, drop_last=True,
                                                               num_workers=args.num_workers)
        ref_dataloader = build_image2image_translation_dataset(args.dataset, args.ref_data_folder,
                                                               split=args.split,
                                                               image_size=args.image_size,
                                                               batch_size=args.valid_batch_size,
                                                               shuffle=args.shuffle, drop_last=True,
                                                               num_workers=args.num_workers)

        sde_model, _ = ADMs_build_util(args.image_size, args.num_classes, args.model, args.diffusion)

        SDEdit = IterativeSDEditPipeline(args, sde_model, args.sdepath, src_dataloader, ref_dataloader)
        SDEdit.sample()


if __name__ == '__main__':
    """ 
    ## cat2dog, base sampling, SDEdit:
    CUDA_VISIBLE_DEVICES=0 python run/run_SDEdit.py -c SDEdit/cat2dog-img256.yaml -sdepath ./checkpoint/InvSDE/afhq_dog_4m.pt -dpath ./dataset/afhq/val/cat -respath ./workdir/sdedit_cat -vbz 32 -final -ts 500
    CUDA_VISIBLE_DEVICES=0 python run/run_SDEdit.py -c SDEdit/cat2dog-img256.yaml -sdepath ./checkpoint/InvSDE/afhq_dog_4m.pt -dpath ./dataset/afhq/val/dog -respath ./workdir/sdedit_dog -vbz 32 -final -ts 500
    
    ## SDEdit + ref:
    CUDA_VISIBLE_DEVICES=0 python run/run_SDEdit.py -c SDEdit/iter-cat2dog-img256-p400-k33-dN32.yaml --task ref -sdepath ./checkpoint/afhq_dog_4m.pt -dpath ./dataset/afhq/train/cat -rdpath ./dataset/afhq/train_edge_map/dog -respath /data2/xingxm/skgruns/ -vbz 8
    """

    parser = argparse.ArgumentParser(
        description="SDEdit",
        parents=[accelerate_parser(), base_data_parser(), base_sampling_parser()]
    )

    # flag
    parser.add_argument("-tk", "--task",
                        default="base", type=str, choices=["base", "mask", "ref"],
                        help="guided image synthesis and editing.")
    # config
    parser.add_argument("-c", "--config",
                        required=True, type=str,
                        default="SDEdit/cat2dog-img256.yaml",
                        help="YAML/YML file for configuration.")
    # data path
    parser.add_argument("-dpath", "--data_folder",
                        nargs="+", type=str,
                        # default==['./dataset/afhq/val/cat'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/dog'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/wild', './dataset/afhq/train/dog'],
                        help="single input for single-domain, multi inputs for multi-domain")
    parser.add_argument("-rdpath", "--ref_data_folder",
                        nargs="+", type=str, default=None,
                        # default==['./dataset/afhq/val/cat'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/dog'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/wild', './dataset/afhq/train/dog'],
                        help="single input for single-domain, multi inputs for multi-domain")
    # model path
    parser.add_argument("-sdepath",
                        default="./checkpoint/afhq_dog_4m.pt", type=str,
                        help="place pretrained model in `./checkpoint/afhq_dog_4m.pt`, "
                             "if None, then train from scratch")
    # use dpm-solver
    parser.add_argument("-uds", "--use_dpm_solver",
                        action='store_true',
                        help="use dpm_solver accelerates sampling.")
    # sampling mode
    parser.add_argument("-final", "--get_final_results",
                        action='store_true',
                        help="visualize intermediate results or just get final output.")

    args = parser.parse_args()
    args = merge_and_update_config(args)

    set_seed(args.seed)
    main(args)
