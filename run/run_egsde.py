# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import os
import sys
import argparse
from copy import deepcopy
from functools import partial

from accelerate.utils import set_seed

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from libs.utils.argparse import (merge_and_update_config, accelerate_parser, base_data_parser,
                                 base_sampling_parser, ema_parser)
from dpm_nn.guided_dpm import build_spaced_gaussian_diffusion
from dpm_nn.inversion.egsde_model import create_sde_and_dse, create_dse
from sketch_nn.dataset.build import build_image2image_translation_dataset


def single_domain_sample(args):
    from pipelines.inversion.egsde_pipeline import EnergyGuidedSDEPipeline
    sde_model, dse_model = create_sde_and_dse(args)

    dataloader = build_image2image_translation_dataset(args.dataset, args.data_folder,
                                                       args.image_size, split=args.split,
                                                       batch_size=args.valid_batch_size,
                                                       shuffle=args.shuffle, drop_last=args.drop_last,
                                                       num_workers=args.num_workers)

    EGSDE = EnergyGuidedSDEPipeline(args, sde_model, args.sdepath, dse_model, args.dsepath, dataloader,
                                    args.use_dpm_solver)
    EGSDE.sample()


def domain_specific_extractor_train(args, train_dataloader, valid_dataloader=None):
    from pipelines.inversion.egsde_pipeline import DSETrainer

    dpm_cfg = args.diffusion
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

    dse_cfg = args.dse
    dse_model = create_dse(image_size=args.image_size,
                           num_class=args.num_classes,
                           classifier_use_fp16=dse_cfg.use_fp16,
                           classifier_width=dse_cfg.model_channels,
                           classifier_depth=dse_cfg.num_res_blocks,
                           classifier_attention_resolutions=dse_cfg.attention_resolutions,
                           classifier_use_scale_shift_norm=dse_cfg.use_scale_shift_norm,
                           classifier_resblock_updown=dse_cfg.resblock_updown,
                           classifier_pool=dse_cfg.pool,
                           phase=args.split)

    trainer = DSETrainer(
        diffusion,
        dse_model,
        train_dataloader,
        args,
        clf_model_path=args.sdepath,
        class_cond=args.model.class_cond,
        use_ddim=dpm_cfg.use_ddim,
        noised=args.noised,
        train_num_steps=args.train_num_steps,
        train_batch_size=args.train_batch_size,
        train_lr=args.lr,
        adam_betas=list(args.adam_betas),
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        ckpt_steps=args.ckpt_steps,
        num_workers=args.num_workers
    )
    trainer.train()


def main(args):
    assert len(args.data_folder) > 0, "Insufficient dataset entry!"

    dataloader = partial(build_image2image_translation_dataset,
                         dataset=args.dataset,
                         image_size=args.image_size,
                         split=args.split,
                         batch_size=args.valid_batch_size,
                         shuffle=args.shuffle, drop_last=True,
                         num_workers=args.num_workers)

    if args.task == "single":  # single-domain train/sampling
        if args.split == "test":
            args.batch_size = args.valid_batch_size
            single_domain_sample(args)
        else:  # train single-domain dse (clf):
            assert len(args.data_folder) == 2, f"get dataset entry: {args.data_folder}, excepted: {2}"
            args.batch_size = args.train_batch_size
            train_dataloader = dataloader(data_path=args.data_folder)
            domain_specific_extractor_train(args, train_dataloader, valid_dataloader=None)

    elif args.task == "multi":  # multi-domain train/sampling
        if args.split == "test":
            args.batch_size = args.valid_batch_size
            # exec `single_domain_sample` N times
            data_folders = deepcopy(args.data_folder)
            for cur_data_folder in data_folders:
                args.data_folder = cur_data_folder
                print(f"Current dataset: {args.data_folder}")
                single_domain_sample(args)
        else:  # train multi-domain dse (clf):
            assert len(args.data_folder) > 2, "More than one dataset should be entered!"
            train_dataloader = dataloader(data_path=args.data_folder)
            domain_specific_extractor_train(args, train_dataloader, valid_dataloader=None)


if __name__ == '__main__':
    """ 
    # sample single-domain:
    CUDA_VISIBLE_DEVICES=1 python run/run_egsde.py -c EGSDE/cat2dog-img256.yaml -vbz 8 -dpath ./dataset/afhq/val/cat -sdepath ./checkpoint/afhq_dog_4m.pt -dsepath ./checkpoint/cat2dog_dse.pt -respath /data2/xingxm/skgruns/
    
    # sample multi-domain:
    CUDA_VISIBLE_DEVICES=1 python run/run_egsde.py -tk multi -c EGSDE/cat2dog-img256.yaml -dpath './dataset/afhq/train/cat' './dataset/afhq/train/wild' './dataset/afhq/train/dog' -vbz 8 -sdepath ./checkpoint/afhq_dog_4m.pt -dsepath ./checkpoint/cat2dog_dse.pt -respath /data2/xingxm/skgruns/
    
    # train ref_extractor:
    CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --nproc_per_node 2 --use_env run/run_egsde.py -spl train -tk single -c EGSDE/cat2dog-img256.yaml -sdepath ./checkpoint/256x256_classifier.pt -dpath ./dataset/afhq/train_edge_map/dog ./dataset/afhq/train/dog -tbz 16 -cumgard 1 -mprec no -respath /data2/xingxm/skgruns/ --ckpt_steps 500 --shuffle
    """
    parser = argparse.ArgumentParser(
        description="EGSDE -- Energy-Guided Stochastic Differential Equations",
        parents=[accelerate_parser(), ema_parser(), base_sampling_parser(), base_data_parser()]
    )

    # flag
    parser.add_argument("-tk", "--task",
                        default="single", type=str, choices=["single", "multi"],
                        help="run EGSDE for one/two-domain image translation.")
    # config
    parser.add_argument("-c", "--config",
                        required=True, type=str,
                        default="EGSDE/cat2dog-img256.yaml",
                        help="YAML/YML file for configuration.")
    # data path
    parser.add_argument("-dpath", "--data_folder",
                        nargs="+", type=str,
                        # default==['./dataset/afhq/val/cat'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/dog'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/wild', './dataset/afhq/train/dog'],
                        help="single input for single-domain, multi inputs for multi-domain")
    # log path
    parser.add_argument("-respath", "--results_path",
                        default="", type=str,
                        help="If it is None, it is automatically generated.")
    # model paths
    parser.add_argument("-sdepath",
                        default="./checkpoint/afhq_dog_4m.pt", type=str,
                        help="place pretrained model in `./checkpoint/afhq_dog_4m.pt`, "
                             "if None, then train from scratch")
    parser.add_argument("-dsepath",
                        default="./checkpoint/cat2dog_dse.pt", type=str,
                        help="place pretrained model in `./checkpoint/cat2dog_dse.pt`, "
                             "if None, then train from scratch")
    # training
    parser.add_argument("-tbz", "--train_batch_size", default=32, type=int)
    parser.add_argument("--train_num_steps", default=700000, type=int, help="total training steps.")
    # use dpm solver
    parser.add_argument("-uds", "--use_dpm_solver",
                        action='store_true',
                        help="use dpm_solver accelerates sampling.")

    args = parser.parse_args()
    args = merge_and_update_config(args)

    set_seed(args.seed)
    main(args)
