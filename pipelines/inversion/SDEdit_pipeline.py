# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from datetime import datetime

import torch
from torchvision import utils as tv_util
from tqdm import tqdm

from libs.engine import ModelState
from dpm_nn.inversion.SDEdit import SDEdit


class SDEditPipeline(ModelState):

    def __init__(self, args, sde_model, sde_path, dataloader, use_dpm_solver: bool = False):
        super().__init__(args)
        self.args = args
        self.use_dpm_solver = use_dpm_solver

        self.print(f"loading SDE from `{sde_path}` ....")
        self.sde_model = self.load_ckpt_model_only(sde_model, sde_path)
        self.print(f"-> SDE Params: {(sum(p.numel() for p in sde_model.parameters()) / 1e6):.3f}M")

        dpm_cfg = args.diffusion
        self.SDEdit = SDEdit(dpm_cfg.timesteps, dpm_cfg.beta_schedule, dpm_cfg.var_type)

        self.SDEdit, self.sde_model = \
            self.accelerator.prepare(self.SDEdit, self.sde_model)
        self.dataloader = self.accelerator.prepare(dataloader)

        self.print()

    def sample(self):
        accelerator = self.accelerator
        device = self.accelerator.device

        sample = next(iter(self.dataloader))
        batch_size = sample["image"].shape[0]  # online batch_size
        image_size = self.args.image_size

        model_kwargs = {}
        with tqdm(self.dataloader, disable=not accelerator.is_local_main_process) as pbar:
            for i, sample in enumerate(pbar):
                start_time = datetime.now()

                src_input, name = sample["image"], sample["fname"]

                model_kwargs['step'] = i
                results = self.SDEdit.sampling_progressive(
                    src_input,
                    mask=sample.get('mask', None),  # editing mask
                    repeat_step=self.args.repeat_step,
                    perturb_step=self.args.perturb_step,
                    model=self.sde_model,
                    model_kwargs=model_kwargs,
                    device=device,
                    recorder=pbar,
                    use_dpm_solver=self.use_dpm_solver
                )

                pbar.set_description(f"time per batch: {datetime.now() - start_time}")
                # pbar.write(f"Running time: {datetime.now() - start_time} | batch_size: {batch_size} \n")

                if accelerator.is_main_process:
                    results = accelerator.gather(results)
                    # gather final result
                    for b in range(batch_size):
                        all_iter_grids = []
                        for kth in range(len(results)):
                            x0, final, perturb_x0 = results[f"{kth}-th"]
                            # (x0, perturbed_x0, kth_translated_x0)
                            save_grids = torch.cat(
                                [x0[b].unsqueeze_(0), perturb_x0[b].unsqueeze_(0), final[b].unsqueeze_(0)],
                                dim=0
                            )
                            all_iter_grids.append(save_grids)
                        # visual
                        img_name = name[b].split(".")[0]  # Remove file suffixes
                        save_path = self.results_path.joinpath(
                            f"i-{i}-{img_name}-B-{b}-K-{kth + 1}-t-{self.args.perturb_step}.png"
                        )
                        tv_util.save_image(torch.cat(all_iter_grids, dim=0), save_path, nrow=3)

        self.close()
