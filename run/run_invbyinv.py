# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import os
import sys
import ast
import argparse

import omegaconf
from torchvision.transforms import transforms
from accelerate.utils import set_seed

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from libs.utils.argparse import (merge_and_update_config, accelerate_parser, base_data_parser, base_sampling_parser)
from dpm_nn.guided_dpm.build_ADMs import ADMs_build_util
from dpm_nn.inversion.egsde_model import create_sde_and_dse
from sketch_nn.dataset.build import build_image2image_translation_dataset


def main(args):
    assert len(args.data_folder) > 0, "Insufficient dataset entry!"

    # preprocess args
    if not isinstance(args.lam_shape, omegaconf.ListConfig):
        args.lam_shape = ast.literal_eval(args.lam_shape)
    if not isinstance(args.lam_pixel_texture, omegaconf.ListConfig):
        args.lam_pixel_texture = ast.literal_eval(args.lam_pixel_texture)
    if not isinstance(args.lam_feature_texture, omegaconf.ListConfig):
        args.lam_feature_texture = ast.literal_eval(args.lam_feature_texture)

    args.batch_size = args.valid_batch_size
    if args.task == "sample":
        from pipelines.inversion.invbyinv_pipeline import InversionByInversionPipeline

        transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor()
        ])
        src_dataloader = build_image2image_translation_dataset(args.dataset,
                                                               args.data_folder,
                                                               split='test',
                                                               data_transforms=transform,
                                                               batch_size=args.valid_batch_size,
                                                               shuffle=args.shuffle,
                                                               drop_last=True,
                                                               num_workers=args.num_workers)
        ref_dataloader = build_image2image_translation_dataset(args.dataset,
                                                               args.ref_data_folder,
                                                               split='test',
                                                               data_transforms=transform,
                                                               batch_size=args.valid_batch_size,
                                                               shuffle=args.shuffle,
                                                               drop_last=True,
                                                               num_workers=args.num_workers)

        dse_model = None
        if args.dataset in ['male2female']:
            from sketch_nn.model.ddpm_model import Model
            sde_model = Model(args.model, args.image_size, args.diffusion.timesteps)
        elif args.dataset in ['cat2dog', 'wild2dog']:
            if args.dsepath is not None and (True in args.use_dse):
                sde_model, dse_model = create_sde_and_dse(args)
            else:
                sde_model, _ = ADMs_build_util(args.image_size, args.num_classes, args.model, args.diffusion)
        else:
            raise NotImplementedError

        shape_expert = None
        if args.get('use_shape', False) and True in args.use_shape:
            from sketch_nn.photo2sketch import photo2sketch_model_build_util
            shape_expert = photo2sketch_model_build_util(method='InformativeDrawings')

        style_vgg_encoder, style_decoder = None, None
        if args.get('use_style', False) and True in args.use_style:
            from style_transfer.AdaIN import net
            style_decoder = net.decoder
            style_vgg_encoder = net.vgg

        inversionSDE = InversionByInversionPipeline(
            args,
            sde_model, args.sdepath,  # SDE model
            src_dataloader, ref_dataloader,  # load sketch and exemplar
            shape_expert,
            style_vgg_encoder, style_decoder,
            dse_model, args.dsepath  # load domain classifier
        )
        inversionSDE.sample()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inversion by Inversion",
        parents=[accelerate_parser(), base_data_parser(), base_sampling_parser()]
    )

    # flag
    parser.add_argument("-tk", "--task",
                        default="sample", type=str, choices=["sample", "eval"],
                        help="sampling or evaluating the results of sampling.")
    # config
    parser.add_argument("-c", "--config",
                        required=True, type=str,
                        default="SDEdit/cat2dog-img256.yaml",
                        help="YAML/YML file for configuration.")
    # data path
    parser.add_argument("-dpath", "--data_folder",
                        nargs="+", type=str,
                        # default=['./dataset/afhq/train/cat'],
                        default=['./dataset/afhq/val/cat'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/dog'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/wild', './dataset/afhq/train/dog'],
                        help="single input for single-domain, multi inputs for multi-domain")
    parser.add_argument("-rdpath", "--ref_data_folder",
                        nargs="+", type=str,
                        # default=['./dataset/afhq/train_anime_edge/dog'],
                        default=['./dataset/afhq/val_anime_edge/dog'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/dog'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/wild', './dataset/afhq/train/dog'],
                        help="single input for single-domain, multi inputs for multi-domain")
    parser.add_argument("-gtpath", "--GT_data_folder",
                        type=str,
                        default='./dataset/afhq/val/dog',
                        help="ground-truth samples used to calculate FID.")
    parser.add_argument("-retpath", "--results_data_folder",
                        type=str,
                        default='',
                        help="the path to store the results.")
    # model path
    parser.add_argument("-sdepath",
                        default="./checkpoint/InvSDE/afhq_dog_4m.pt", type=str,
                        help="provide by EGSDE repo")
    # use dpm-solver
    parser.add_argument("-uds", "--use_dpm_solver",
                        action='store_true',
                        help="use dpm_solver accelerates sampling.")
    # sampling mode
    parser.add_argument("-final", "--get_final_results",
                        action='store_true',
                        help="get a final output instead of visualizing intermediate results.")
    parser.add_argument("-save_inter", "--save_intermediate",
                        action='store_true',
                        help="saving intermediate results.")
    parser.add_argument("-vfp", "--visual_fpath",
                        type=str, default="./two-stage-grad",
                        help="specifies the path where the visualize intermediate results are stored.")

    args = parser.parse_args()
    args = merge_and_update_config(args)

    set_seed(args.seed)
    main(args)
