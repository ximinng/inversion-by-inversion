# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import os
import sys
import argparse
from multiprocessing import cpu_count

from accelerate.utils import set_seed

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from dpm_nn.guided_dpm.clf_guided_trainer import ClassifierGuidedDPMTrainer, ClassifierTrainer
from dpm_nn.guided_dpm.clf_guided_sampler import ClassifierGuidedSampler
from dpm_nn.guided_dpm import ADMs_build_util, build_spaced_gaussian_diffusion
from sketch_nn.dataset.build import build_imagenet
from libs.utils.argparse import (
    merge_and_update_config,
    accelerate_parser,
    base_data_parser,
    base_training_parser,
    base_sampling_parser,
    ema_parser
)


def sampling(args):
    num_classes = args.num_classes
    dpm_cfg = args.diffusion
    model_cfg = args.model
    clf_cfg = args.classifier
    eps_model, clf_model = ADMs_build_util(args.image_size,
                                           num_classes,
                                           model_cfg,
                                           dpm_cfg,
                                           build_clf=True,
                                           clf_cfg=clf_cfg)

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

    sampler = ClassifierGuidedSampler(
        diffusion, eps_model, args.dpm_path, clf_model, args.clf_path, args,
        classifier_scale=args.classifier_scale,
        clip_denoised=args.diffusion.clip_denoised,
        total_samples=args.total_samples,
        mixed_precision=args.mixed_precision,
    )
    sampler.sampling()


def classifier_train(args, train_loader, valid_loader):
    num_classes = args.num_classes
    dpm_cfg = args.diffusion
    model_cfg = args.model
    clf_cfg = args.classifier
    eps_model, clf_model = ADMs_build_util(args.image_size,
                                           num_classes,
                                           model_cfg,
                                           dpm_cfg,
                                           build_clf=True,
                                           clf_cfg=clf_cfg)

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

    trainer = ClassifierTrainer(
        diffusion,
        eps_model,  # Here eps_model does not involve training and is only used for sampling
        args.dpm_path,
        clf_model,
        train_loader,
        valid_loader,
        args,
        classifier_scale=args.classifier_scale,
        clf_model_path=args.clf_path,
        train_batch_size=args.batch_size,
        valid_batch_size=args.batch_size,
        adam_betas=args.adam_betas,
        weight_decay=args.weight_decay,
        train_num_steps=args.train_num_steps,
        num_samples=args.total_samples,
        ckpt_steps=args.ckpt_steps,
        vis_steps=args.vis_steps,
    )
    trainer.train()


def diffusion_model_train(args, dataloader):
    num_classes = args.num_classes
    dpm_cfg = args.diffusion
    model_cfg = args.model
    eps_model, _ = ADMs_build_util(args.image_size, num_classes,
                                   model_cfg, dpm_cfg, build_clf=False)

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
        input_pertub=args.get('input_pertub', 0),
        p2_gamma=args.get('p2_gamma', 0),
        p2_k=args.get('p2_k', 1.0)
    )

    trainer = ClassifierGuidedDPMTrainer(
        diffusion,
        eps_model,
        dataloader,
        args,
        eps_model_path=args.dpm_path,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        adam_betas=args.adam_betas,
        weight_decay=args.weight_decay,
        train_num_steps=args.train_num_steps,
        num_samples=args.total_samples,
        ckpt_steps=args.ckpt_steps,
        vis_steps=args.vis_steps,
    )
    trainer.train()


def get_dataloader(args):
    if args.dataset == "imagenet-1k":
        dataloader = build_imagenet(args.data_folder,
                                    args.image_size, split=args.split,
                                    batch_size=args.train_batch_size,
                                    shuffle=True, drop_last=False,
                                    num_workers=cpu_count() if args.num_workers == 0 else args.num_workers)
    elif args.dataset in ["cat2dog"]:
        from sketch_nn.dataset.build import load_data_folders
        dataloader = load_data_folders(args.dataset, args.data_folder,
                                       args.image_size,
                                       batch_size=args.valid_batch_size,
                                       shuffle=True, drop_last=False,
                                       num_workers=cpu_count() if args.num_workers == 0 else args.num_workers)
    else:
        print(f"{args.dataset} is not currently supported.")

    return dataloader


def main(args):
    args.batch_size = args.train_batch_size
    if args.task == "diffusion":
        dataloader = get_dataloader(args)
        diffusion_model_train(args, dataloader)

    elif args.task == "classifier":
        train_dataloader = build_imagenet(args.data_folder,
                                          args.image_size, split="train",
                                          batch_size=args.train_batch_size,
                                          shuffle=True, drop_last=True,
                                          num_workers=cpu_count() if args.num_workers == 0 else args.num_workers)
        val_dataloader = build_imagenet(args.data_folder,
                                        args.image_size, split="test",
                                        batch_size=args.valid_batch_size,
                                        shuffle=False, drop_last=False,
                                        num_workers=cpu_count() if args.num_workers == 0 else args.num_workers)
        classifier_train(args, train_dataloader, val_dataloader)

    elif args.task == "sample":
        args.batch_size = args.valid_batch_size
        sampling(args)


if __name__ == '__main__':
    """ 
    - train DDPM from scratch:
    python run/run_ADM.py --lr 1e-4 --image-size 128 --num-classes 125  --batch-size 6 --mixed-precision fp16 --grad-cumprod 2 --num-channels 256 --num-res-blocks 2 --num-heads 4 --learn-sigma --clip-denoised --data-folder ./dataset/sketchy --results-path ./results_clf_guided_Sketchy_128 --ckpt-steps 3000 --vis-steps 1000
    
    - fine-tune DDPM:
    python run/run_ADM.py --lr 1e-4 --image-size 128 --num-classes 125  --batch-size 6 --mp fp16 --grad-cumprod 2 --num-channels 256 --num-res-blocks 2 --num-heads 4 --learn-sigma --clip-denoised --data-folder ./dataset/sketchy --dpm-path ./checkpoint/128x128_diffusion.pt --results-path ./results_ft_dpm_clfguid_sketchy_128 --ckpt-steps 3000 --vis-steps 1000
    CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch run/run_ADM.py --lr 1e-4 --image-size 128 --num-classes 125 --batch-size 4 --mp fp16 --grad-cumprod 2 --num-channels 256 --num-res-blocks 2 --num-heads 4 --learn-sigma --clip-denoised --data-folder ./dataset/sketchy --dpm-path ./checkpoint/128x128_diffusion.pt --results-path ./results_ft_dpm_clfguid_sketchy_128 --ckpt-steps 3000 --vis-steps 500 --num-workers 4
    
    - train classifier from scratch: 
    python run/run_ADM.py -tk classifier -tbz 4 -cumgard 2 -mprec fp16 --classifier-scale 0.5 --mp fp16 -dpath /data2/xingxm/ILSVRC2012 -respath /data2/xingxm/results_clf_imagenet_128 --dpm-path ./checkpoint/128x128_diffusion.pt --ckpt-steps 30000 --vis-steps 10000 --ema --ema-decay 0.9999

    - sampling:
    CUDA_VISIBLE_DEVICES=1 python run/run_ADM.py -tk sample -c clfguided/imagenet-img256-respace250.yaml -vbz 16 -mprec no --clf_path ./checkpoint/ADM/256x256_classifier.pt --dpm_path ./checkpoint/ADM/256x256_diffusion.pt -respath ./skgruns/ --total_samples 10000
    """

    parser = argparse.ArgumentParser(
        description="classifier guided diffusion model",
        parents=[accelerate_parser(), ema_parser(), base_data_parser(), base_training_parser(), base_sampling_parser()]
    )

    # flag
    parser.add_argument("-tk", "--task",
                        default="diffusion", type=str,
                        choices=["diffusion", "classifier", "sample"],
                        help="which part of training: 'diffusion' or 'classifier'")
    # config
    parser.add_argument("-c", "--config",
                        required=True, type=str,
                        default="clfguided/imagenet-img128-respace250.yaml",
                        help="YAML/YML file for configuration.")
    # data path
    parser.add_argument("-dpath", "--data_folder",
                        nargs="+", type=str,
                        # default==['./dataset/sketchy'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/dog'],
                        # default=['./dataset/afhq/train/cat', './dataset/afhq/train/wild', './dataset/afhq/train/dog']
                        )
    # classifier and diffusion model paths
    parser.add_argument("--clf_path",
                        default="", type=str,
                        help="place pretrained model in `./checkpoint/64x64_classifier.pt`, "
                             "if None, then train from scratch")
    parser.add_argument("--dpm_path",
                        default="", type=str,
                        help="place pretrained model in `./checkpoint/64x64_diffusion.pt`, "
                             "if None, then train from scratch")
    # evaluate
    parser.add_argument("--vis_steps", default=1000, type=int, help="Sampling steps.")
    # methods
    parser.add_argument("--classifier_scale", default=0.5, type=float)
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--timestep_respacing", default="", type=str, help="respacing trick: `250`, `ddim-250`.")

    args = parser.parse_args()
    args = merge_and_update_config(args)

    set_seed(args.seed)
    main(args)
