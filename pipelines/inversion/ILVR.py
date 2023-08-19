# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from argparse import Namespace

import torch
import torch.nn as nn
from torchvision import utils as tv_util
from tqdm import tqdm

from libs.engine import ModelState
from libs.utils import cycle
from sketch_nn.augment.resizer import Resizer


class ILVRPipeline(ModelState):

    def __init__(
            self,
            args: Namespace,
            eps_model: nn.Module,
            eps_model_path: str,
            diffusion: nn.Module,
            dataloader: torch.utils.data.DataLoader
    ):
        super().__init__(args)
        self.args = args

        # set log path
        self.results_path = self.results_path.joinpath(f"{args.task}-sample")
        self.results_path.mkdir(exist_ok=True)

        self.diffusion = diffusion

        # create eps_model
        self.print(f"loading SDE from `{eps_model_path}` ....")
        self.eps_model = self.load_ckpt_model_only(eps_model, eps_model_path)
        if args.model.use_fp16:
            self.eps_model.convert_to_fp16()
        self.eps_model.eval()
        self.print(f"-> eps_model Params: {(sum(p.numel() for p in self.eps_model.parameters()) / 1e6):.3f}M")

        self.eps_model, self.dataloader = self.accelerator.prepare(self.eps_model, dataloader)
        self.dataloader = cycle(self.dataloader)

    def sample(self):
        device = self.accelerator.device
        accelerator = self.accelerator

        sample = next(iter(self.dataloader))
        batch_size = sample["image"].shape[0]  # get real batch_size
        image_size = self.args.image_size

        down_N = self.args.down_N
        shape = (batch_size, 3, image_size, image_size)
        shape_d = (
            batch_size, 3, int(image_size / down_N), int(image_size / down_N)
        )
        down = Resizer(shape, 1 / down_N).to(device)
        up = Resizer(shape_d, down_N).to(device)
        resizers = (down, up)

        extra_kwargs = {}
        model_kwargs = {}

        i = 0
        with tqdm(initial=i, total=self.args.total_samples, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.args.total_samples:
                sample = next(self.dataloader)
                ref_img, name = sample["image"], sample["fname"]
                extra_kwargs["ref_img"] = ref_img

                sample = self.diffusion.p_sample_loop(
                    self.eps_model,
                    (batch_size, 3, image_size, image_size),
                    clip_denoised=self.args.diffusion.clip_denoised,
                    model_kwargs=model_kwargs,
                    resizers=resizers,
                    range_t=self.args.range_t,
                    extra_kwargs=extra_kwargs
                )

                if self.accelerator.is_main_process:
                    sample = self.accelerator.gather(sample)
                    sample = (sample + 1) / 2

                    if self.args.get_final_results:
                        for b in range(sample.shape[0]):
                            name_ = name[b].split(".")[0]  # Remove file suffixes
                            save_path = self.results_path / f"{int(self.step + b)}-{name_}.png"
                            tv_util.save_image(sample[b], save_path)
                    else:
                        for b in range(sample.shape[0]):
                            save_path = self.results_path.joinpath(
                                f"i-{i}-b-{b}-t-down_N-{down_N}-rt-{self.args.range_t}.png"
                            )
                            # (x0, sampled)
                            save_grids = torch.cat(
                                [ref_img[b].unsqueeze_(0), sample[b].unsqueeze_(0)],
                                dim=0
                            )
                            tv_util.save_image(save_grids.float(), save_path, nrow=sample.shape[0])

                i += batch_size
                pbar.update(1)
        self.close()
