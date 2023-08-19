# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from datetime import datetime

import torch
from torchvision import utils as tv_util
from tqdm import tqdm

from libs.engine import ModelState
from libs.utils import cycle
from sketch_nn.augment.resizer import Resizer
from sketch_nn.methods.inversion.SDEdit_iter import IterativeSDEdit


class IterativeSDEditPipeline(ModelState):

    def __init__(self, args, sde_model, sde_path, src_dataloader, ref_dataloader):
        super().__init__(args)
        self.args = args

        self.print(f"loading SDE from `{sde_path}` ....")
        self.sde_model = self.load_ckpt_model_only(sde_model, sde_path)
        self.print(f"-> SDE Params: {(sum(p.numel() for p in sde_model.parameters()) / 1e6):.3f}M")

        self.results_path = self.results_path.joinpath(f"{args.dataset}-{args.task}-sample-seed-{args.seed}")
        self.results_path.mkdir(exist_ok=True)

        dpm_cfg = args.diffusion
        self.SDEdit = IterativeSDEdit(dpm_cfg.timesteps, dpm_cfg.beta_schedule, dpm_cfg.var_type)

        self.SDEdit, self.sde_model = self.accelerator.prepare(self.SDEdit, self.sde_model)
        self.src_dataloader, self.ref_dataloader = self.accelerator.prepare(src_dataloader, ref_dataloader)
        self.src_dataloader = cycle(self.src_dataloader)
        self.ref_dataloader = cycle(self.ref_dataloader)

        self.print()

    def sample(self):
        accelerator = self.accelerator
        device = self.accelerator.device

        sample = next(iter(self.src_dataloader))
        batch_size = sample["image"].shape[0]  # online batch_size
        image_size = self.args.image_size

        s_down_N = self.args.src_down_N
        shape = (batch_size, 3, image_size, image_size)
        s_shape_d = (
            batch_size, 3, int(image_size / s_down_N), int(image_size / s_down_N)
        )
        src_down = Resizer(shape, 1 / s_down_N).to(device)
        src_up = Resizer(s_shape_d, s_down_N).to(device)
        low_passer = (src_down, src_up)

        model_kwargs = {}
        iter_kwargs = {
            'low_passer': low_passer,
            'fusion_scale': self.args.fusion_scale
        }
        i = 0
        with tqdm(initial=i, total=self.args.total_samples, disable=not accelerator.is_main_process) as pbar:
            while i < self.args.total_samples:
                src_sample = next(self.src_dataloader)
                src_input, name = src_sample["image"], src_sample["fname"]
                ref_sample = next(self.ref_dataloader)
                ref_input = ref_sample["image"]

                model_kwargs['step'] = i

                start_time = datetime.now()
                results = self.SDEdit.iterative_sampling_progressive(
                    src_input,
                    ref_input,
                    self.args.iter_step,
                    iter_kwargs,
                    list(self.args.repeat_step),
                    list(self.args.perturb_step),
                    model=self.sde_model,
                    model_kwargs=model_kwargs,
                    device=device,
                    recorder=pbar
                )
                pbar.set_description(f"one batch time: {datetime.now() - start_time}, "
                                     f"total_iter: {self.args.iter_step}")

                if accelerator.is_main_process:
                    results = accelerator.gather(results)
                    # gather final result
                    for b in range(batch_size):
                        all_iter_grids = []
                        for ith in range(self.args.iter_step):
                            for kth in range(self.args.repeat_step[ith]):
                                x0, final, perturb_x0, blurred_x0 = results[f"{ith}-{kth}-th"]
                                # (x0, perturbed_x0, kth_translated_x0)
                                # (x0, perturbed_x0, src, blurred_x0, kth_translated_x0)
                                save_grids = torch.cat([x0[b].unsqueeze_(0),
                                                        perturb_x0[b].unsqueeze_(0),
                                                        src_input[b].unsqueeze_(0)
                                                        if ith != 0 else torch.zeros_like(src_input[b]).unsqueeze_(0),
                                                        blurred_x0[b].unsqueeze_(0),
                                                        final[b].unsqueeze_(0)], dim=0)
                                all_iter_grids.append(save_grids)
                        # visual
                        img_name = name[b].split(".")[0]  # Remove file suffixes
                        save_path = self.results_path.joinpath(
                            f"{i}-{img_name}-iter-{ith + 1}-K-{kth + 1}-t-{self.args.perturb_step}.png"
                        )
                        tv_util.save_image(torch.cat(all_iter_grids, dim=0), save_path, nrow=5)

                i += 1
                pbar.update(1)

        self.close()
