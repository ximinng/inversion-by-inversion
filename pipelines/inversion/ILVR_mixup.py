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

class ILVRMixupPipeline(ModelState):

    def __init__(
            self,
            args: Namespace,
            eps_model: nn.Module,
            eps_model_path: str,
            diffusion: nn.Module,
            src_dataloader: torch.utils.data.DataLoader,
            ref_dataloader: torch.utils.data.DataLoader
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

        self.eps_model = self.accelerator.prepare(self.eps_model)
        self.src_dataloader, self.ref_dataloader = self.accelerator.prepare(src_dataloader, ref_dataloader)
        self.src_dataloader = cycle(self.src_dataloader)
        self.ref_dataloader = cycle(self.ref_dataloader)

    def sample(self):
        device = self.accelerator.device

        sample = next(iter(self.src_dataloader))
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

        model_kwargs = {}
        i = 0

        with tqdm(initial=i, total=self.args.total_samples, disable=not self.accelerator.is_local_main_process) as pbar:
            while i < self.args.total_samples:
                src_sample = next(self.src_dataloader)
                src_input, src_name = src_sample["image"], src_sample["fname"]
                ref_sample = next(self.ref_dataloader)
                ref_input, ref_name = ref_sample["image"], ref_sample["fname"]

                extra_kwargs = {
                    "src_input": src_input,
                    "ref_input": ref_input,
                    "fuse_scale": self.args.fuse_scale
                }

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
                            s_name_ = src_name[b].split(".")[0]  # Remove file suffixes
                            r_name_ = ref_name[b].split(".")[0]  # Remove file suffixes
                            save_path = self.results_path / f"{int(self.step + b)}-{s_name_}_to_{r_name_}.png"
                            tv_util.save_image(sample[b], save_path)
                        else:
                            for b in range(sample.shape[0]):
                                save_path = self.results_path.joinpath(
                                    f"i-{i}-b-{b}-t-down_N-{down_N}-rt-{self.args.range_t}.png"
                                )
                                # (x0, sampled)
                                save_grids = torch.cat(
                                    [ref_input[b].unsqueeze_(0), sample[b].unsqueeze_(0)],
                                    dim=0
                                )
                                tv_util.save_image(save_grids.float(), save_path, nrow=sample.shape[0])

                i += batch_size
                pbar.update(1)

        self.close()
