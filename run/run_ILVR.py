# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import os
import sys
import argparse

from accelerate.utils import set_seed

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from sketch_nn.dataset.build import build_image2image_translation_dataset
from dpm_nn.guided_dpm.build_ADMs import ADMs_build_util
from libs.utils.argparse import (merge_and_update_config, accelerate_parser, base_data_parser, base_sampling_parser)


def main(args):
    if args.task == "ilvr":  # original method
        from dpm_nn.inversion.ILVR import build_spaced_gaussian_diffusion
        from pipelines.inversion.ILVR import ILVRPipeline

        args.batch_size = args.valid_batch_size
        dpm_cfg = args.diffusion
        model_cfg = args.model
        num_classes = args.num_classes

        eps_model, _ = ADMs_build_util(args.image_size, num_classes, model_cfg, dpm_cfg)

        diffusion = build_spaced_gaussian_diffusion(
            timesteps=dpm_cfg.timesteps,
            noise_schedule=dpm_cfg.beta_schedule,
            learn_sigma=dpm_cfg.learn_sigma,
            sigma_small=dpm_cfg.sigma_small,
            use_kl=dpm_cfg.use_kl,
            predict_xstart=dpm_cfg.predict_xstart,
            rescale_timesteps=dpm_cfg.rescale_timesteps,
            rescale_learned_sigmas=dpm_cfg.rescale_learned_sigmas,
            timestep_respacing=dpm_cfg.timestep_respacing,
        )

        dataloader = build_image2image_translation_dataset(args.dataset, args.data_folder,
                                                           args.image_size, split=args.split,
                                                           batch_size=args.valid_batch_size,
                                                           shuffle=args.shuffle, drop_last=True,
                                                           num_workers=args.num_workers)

        ILVR = ILVRPipeline(args, eps_model, args.eps_model_path, diffusion, dataloader)
        ILVR.sample()

    elif args.task == "ilvr_mixup":  # mix source input and reference input up
        from sketch_nn.methods.inversion.ILVR_mixup import build_spaced_gaussian_diffusion
        from pipelines.inversion.ILVR_mixup import ILVRMixupPipeline

        args.batch_size = args.valid_batch_size
        dpm_cfg = args.diffusion
        model_cfg = args.model
        num_classes = args.num_classes

        eps_model, _ = ADMs_build_util(args.image_size, num_classes, model_cfg, dpm_cfg)

        diffusion = build_spaced_gaussian_diffusion(
            timesteps=dpm_cfg.timesteps,
            noise_schedule=dpm_cfg.beta_schedule,
            learn_sigma=dpm_cfg.learn_sigma,
            sigma_small=dpm_cfg.sigma_small,
            use_kl=dpm_cfg.use_kl,
            predict_xstart=dpm_cfg.predict_xstart,
            rescale_timesteps=dpm_cfg.rescale_timesteps,
            rescale_learned_sigmas=dpm_cfg.rescale_learned_sigmas,
            timestep_respacing=dpm_cfg.timestep_respacing,
        )

        # build source and reference dataset
        src_dataloader = build_image2image_translation_dataset(args.dataset, args.data_folder,
                                                               args.image_size, split=args.split,
                                                               batch_size=args.valid_batch_size,
                                                               shuffle=args.shuffle, drop_last=True,
                                                               num_workers=args.num_workers)
        ref_dataloader = build_image2image_translation_dataset(args.dataset, args.extra_data_folder,
                                                               args.image_size, split=args.split,
                                                               batch_size=args.valid_batch_size,
                                                               shuffle=args.shuffle, drop_last=True,
                                                               num_workers=args.num_workers)

        ILVR = ILVRMixupPipeline(args, eps_model, args.eps_model_path, diffusion,
                                 src_dataloader, ref_dataloader)
        ILVR.sample()


if __name__ == '__main__':
    """ 
    # ILVR ffhq:
    CUDA_VISIBLE_DEVICES=0 python run/run_ILVR.py -tk ilvr -c ILVR/ffhq-p2-img256-respace100-down32-Rt20.yaml -vbz 4 -dpath ./dataset/ref_imgs/face -epspath /data5/xingxm/checkpoint/ffhq_p2.pt -respath /data5/xingxm/skgruns/ 
    # ILVR cat2dog:
    CUDA_VISIBLE_DEVICES=0 python run/run_ILVR.py -tk ilvr -c ILVR/cat2dog-img256-respace250-down32-Rt20.yaml -vbz 32 -dpath ./dataset/afhq/train/cat -epspath ./checkpoint/afhqdog_p2.pt -respath /data2/xingxm/skgruns/ILVR-cat2dog -final -ts 3000
    # ILVR wild2dog:
    CUDA_VISIBLE_DEVICES=0 python run/run_ILVR.py -tk ilvr -c ILVR/wild2dog-img256-respace250-down32-Rt20.yaml -vbz 32 -dpath ./dataset/afhq/train/wild -epspath ./checkpoint/afhqdog_p2.pt -respath /data2/xingxm/skgruns/ILVR-wild2dog -final -ts 3000
    
    # ILVR ffhq, sketch and photo mixup input
    CUDA_VISIBLE_DEVICES=5 python run/run_ILVR.py -tk ilvr_mixup -c ILVR/ffhq-p2-img256-respace100-down32-Rt20.yaml -vbz 32 -epspath ./checkpoint/ffhq_p2.pt -dpath ./dataset/celeba_hq/train/male -epath ./dataset/face_new/pre_female -respath /data2/xingxm/skgruns/ILVR_mixed -ts 3000 -final --fusion_scale 0.3
    # ILVR cat2dog, sketch and photo mixup input
    CUDA_VISIBLE_DEVICES=5 python run/run_ILVR.py -tk ilvr_mixup -c ILVR/cat2dog-img256-respace100-down32-Rt20.yaml -vbz 32 -epspath ./checkpoint/afhqdog_p2.pt -dpath ./dataset/afhq/train/cat -epath ./dataset/afhq/train_edge_map/dog -respath /data2/xingxm/skgruns/ILVR_mixed -ts 3000 -final --fusion_scale 0.3
    # ILVR wild2dog, sketch and photo mixup input
    CUDA_VISIBLE_DEVICES=5 python run/run_ILVR.py -tk ilvr_mixup -c ILVR/wild2dog-img256-respace100-down32-Rt20.yaml -vbz 32 -epspath ./checkpoint/afhqdog_p2.pt -dpath ./dataset/afhq/train/wild -epath ./dataset/afhq/train_edge_map/dog -respath /data2/xingxm/skgruns/ILVR_mixed -ts 3000 -final --fusion_scale 0.3
    """

    parser = argparse.ArgumentParser(
        description="ILVR: Conditioning Method for Denoising Diffusion Probabilistic Models",
        parents=[accelerate_parser(), base_sampling_parser(), base_data_parser()]
    )
    # flag
    parser.add_argument(
        "-tk", "--task",
        default="ilvr", type=str,
        choices=["ilvr", "ilvr_shape"]
    )
    # configs
    parser.add_argument(
        "-c", "--config",
        required=True, type=str,
        default="ILVR/face-img256-respace100-down32-Rt20.yaml",
        help="YAML/YML file for configuration."
    )
    # data paths
    parser.add_argument(
        "-dpath", "--data_folder",
        type=str, default="./dataset/ref_imgs/face",
        help="place to guide the generated dataset path."
    )
    parser.add_argument(
        "-epath", "--extra_data_folder",
        type=str, default="./dataset/ref_imgs/face",
        help="an additional data set."
    )
    # log path
    parser.add_argument(
        "-respath", "--results_path",
        default="", type=str,
        help="If it is None, it is automatically generated."
    )
    # model path
    parser.add_argument(
        "-epspath", "--eps_model_path",
        default="./checkpoint/ffhq_10m.pt", type=str,
        help="pretrained u-net model."
    )
    # method
    parser.add_argument("--down_N", default=32, type=int, help="")
    parser.add_argument("--range_t", default=20, type=int, help="")
    parser.add_argument("--fuse_scale", default=0.3, type=int, help="The ratio of image blending")
    # sampling mode
    parser.add_argument("-final", "--get_final_results",
                        action='store_true',
                        help="visualize intermediate results or just get final output.")

    args = parser.parse_args()
    args = merge_and_update_config(args)

    set_seed(args.seed)
    main(args)
