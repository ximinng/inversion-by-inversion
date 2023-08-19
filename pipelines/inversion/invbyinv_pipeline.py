# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from datetime import datetime
from pathlib import Path

import torch
from torchvision import utils as tv_util
from tqdm.auto import tqdm

from libs.engine import ModelState
from libs.utils import cycle, sum_params
from dpm_nn.dpm_solver import NoiseScheduleVP, DPM_Solver, model_wrapper
from sketch_nn.augment.resizer import Resizer
from sketch_nn.methods.inversion.invbyinv import InversionByInversion


class InversionByInversionPipeline(ModelState):

    def __init__(
            self,
            args,
            sde_model: torch.nn.Module,
            sde_path: str,
            src_dataloader: torch.utils.data.DataLoader,
            ref_dataloader: torch.utils.data.DataLoader,
            shape_expert: torch.nn.Module = None,
            style_vgg_encoder: torch.nn.Module = None,
            style_decoder: torch.nn.Module = None,
            dse_model: torch.nn.Module = None,
            dse_path: str = None
    ):
        super().__init__(
            args,
            log_path_suffix=f"sd{args.seed}"
                            f"-S{args.n_stage}K{''.join(str(item) for item in args.repeat_step)}"
                            f"-p{max(args.perturb_step)}"
                            f"{'-uds' if args.use_dpm_solver else ''}"
                            f"{'-final' if args.get_final_results else ''}"
        )
        self.args = args

        self.print(f"loading SDE from `{sde_path}` ....")
        self.sde_model = self.load_ckpt_model_only(sde_model, sde_path,
                                                   rm_module_prefix=True if args.dataset in ['male2female'] else False)
        self.print(f"-> SDE Params: {sum_params(sde_model):.3f}M")

        # load photo2sketch
        self.shape_expert = shape_expert
        if True in args.use_shape and self.shape_expert:
            path_ = Path(args.shape_expert_root) / args.shape_style / "netG_A_latest.pth"
            assert path_.exists(), f"{path_} is not exist."
            self.print(f"loading shape_expert from `{path_}` ....")
            shape_expert.load_state_dict(torch.load(path_))
            shape_expert.eval()
            self.print(f"-> shape_expert Params: {sum_params(shape_expert):.3f}M")
            self.shape_expert = self.accelerator.prepare(shape_expert)

        # load inception_v3 model
        self.texture_feat_model = None
        if True in args.feature_texture:
            self.print(f"load feature_texture model ....")
            if self.args.feature_texture_model == 'inceptionV3':
                from libs.modules.vision import inception_v3
                texture_feat_model = inception_v3(pretrained=True, progress=True)
                self.print(f"-> inception_v3 Params: {sum_params(texture_feat_model):.3f}M")
            elif self.args.feature_texture_model == 'VGG':
                from libs.modules.vision import vgg16
                texture_feat_model = vgg16(pretrained=True, progress=True)
                self.print(f"-> VGG16 Params: {sum_params(texture_feat_model):.3f}M")
            elif self.args.feature_texture_model == 'CLIP':
                from libs.metric.clip_score import CLIPScoreWrapper
                texture_feat_model = CLIPScoreWrapper(clip_model_name="RN101",
                                                      device=self.device,
                                                      visual_score=True,
                                                      feats_loss_type="l2",
                                                      feats_loss_weights=[0, 0, 1.0, 1.0, 0],
                                                      fc_loss_weight=0.1)
                self.print(f"-> CLIP RN101")
            else:
                raise NotImplementedError(f"The specified model does not exist: {self.args.feature_texture_model}")

            texture_feat_model.eval()
            self.texture_feat_model = self.accelerator.prepare(texture_feat_model)

        # load AdaIn model
        self.style_vgg_encoder, self.style_decoder = style_vgg_encoder, style_decoder
        if (True in args.use_style) and style_vgg_encoder and style_decoder:
            self.print(f"-> style_vgg_encoder Params: {sum_params(style_vgg_encoder):.3f}M")
            style_vgg_encoder.load_state_dict(torch.load(args.style_vgg))
            style_vgg_encoder = torch.nn.Sequential(*list(style_vgg_encoder.children())[:31])
            style_vgg_encoder.eval()
            self.print(f"-> style_decoder Params: {sum_params(style_decoder):.3f}M")
            style_decoder.load_state_dict(torch.load(args.style_decoder))
            style_decoder.eval()
            self.style_vgg_encoder, self.style_decoder = self.accelerator.prepare(style_vgg_encoder, style_decoder)

        # load DSE model
        self.dse_model = None
        if (True in args.use_dse) and dse_model and dse_path:
            self.print(f"loading domain-specific extractor from `{dse_path}` ....")
            self.dse_model = self.load_ckpt_model_only(dse_model, dse_path)
            self.dse_model.eval()
            self.print(f"-> DSE Params: {(sum(p.numel() for p in sde_model.parameters()) / 1e6):.3f}M")
            self.dse_model = self.accelerator.prepare(self.dse_model)

        dpm_cfg = args.diffusion
        self.inversionSDE = InversionByInversion(dpm_cfg.timesteps, dpm_cfg.beta_schedule, dpm_cfg.var_type, args)

        self.inversionSDE, self.sde_model = \
            self.accelerator.prepare(self.inversionSDE, self.sde_model)
        self.src_dataloader, self.ref_dataloader = self.accelerator.prepare(src_dataloader, ref_dataloader)
        self.src_dataloader = cycle(self.src_dataloader)
        self.ref_dataloader = cycle(self.ref_dataloader)

        self.step = 0

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

        model_kwargs = {
            'low_passer': low_passer,
            'geom_expert': self.shape_expert,
            "texture_feat_model": self.texture_feat_model,
            "style_vgg_encoder": self.style_vgg_encoder,
            "style_decoder": self.style_decoder,
            "dse": self.dse_model
        }

        # init dpm_solver
        dpm_solver_kwargs = {}
        if self.args.use_dpm_solver:
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.inversionSDE.betas)
            kwargs_ = {'split': True}
            model_fn = model_wrapper(
                self.sde_model,
                noise_schedule=noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs=kwargs_,
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

            # set dpm_solver
            t_guided_, t_dpm_solver_ = [], []
            solver_cfg = self.args.dpm_solver
            for i in range(1, int(self.args.n_stage) + 1):
                t_guided_.append(
                    list(solver_cfg.t_guided.get(i))
                )

                assert len(list(solver_cfg.t_dpm_solver_dense.get(i))) == 2
                d_left, d_right = list(solver_cfg.t_dpm_solver_dense.get(i))
                t_dpm_solver_.append(
                    list(range(d_left, d_right, -1)) + list(solver_cfg.t_dpm_solver_spare.get(i))
                )

            dpm_solver_kwargs['dpm_solver'] = dpm_solver
            dpm_solver_kwargs['t_guided'] = t_guided_  # guided denoising step
            dpm_solver_kwargs['t_dpm_solver'] = t_dpm_solver_  # dpm_solver denoising step without guided

        with tqdm(initial=self.step, total=self.args.total_samples, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.args.total_samples:

                src_sample = next(self.src_dataloader)
                src_input, src_name = src_sample["image"], src_sample["fname"]
                ref_sample = next(self.ref_dataloader)
                ref_input, ref_name = ref_sample["image"], ref_sample["fname"]

                model_kwargs['step'] = self.step

                start_time = datetime.now()
                results = self.inversionSDE.inversion_step(
                    src_input,
                    ref_input,
                    self.args.n_stage,
                    list(self.args.repeat_step),
                    list(self.args.perturb_step),
                    model=self.sde_model,
                    model_kwargs=model_kwargs,
                    dpm_solver_kwargs=dpm_solver_kwargs,
                    device=device,
                    recorder=pbar,
                    log_fn=self.accelerator.log
                )
                pbar.set_description(f"one batch time: {datetime.now() - start_time}, "
                                     f"total_iter: {self.args.n_stage}")

                if accelerator.is_local_main_process:
                    if self.args.get_final_results:  # gather the final, used to calculate the FID
                        final = results["out"]
                        for b in range(batch_size):
                            src_name_ = src_name[b].split(".")[0]  # Remove file suffixes
                            ref_name_ = ref_name[b].split(".")[0]  # Remove file suffixes
                            save_path = self.results_path / f"{int(self.step + b)}-{src_name_}-to-{ref_name_}.png"
                            tv_util.save_image(final[b], save_path)
                    else:
                        # used for debugging
                        # visualize intermediate results
                        for b in range(batch_size):
                            all_iter_grids = []

                            for ith in range(self.args.n_stage):
                                for kth in range(self.args.repeat_step[ith]):
                                    # x0, final, perturb_x0, blurred_x0, style = results[f"{ith}-{kth}-th"]
                                    x0, final, perturb_x0, blurred_x0 = results[f"{ith}-{kth}-th"]

                                    # (x0, perturbed_x0, ref, src, depth_input, kth_translated_x0)
                                    src_vis = src_input[b].unsqueeze_(0) if ith != 0 \
                                        else torch.zeros_like(src_input[b]).unsqueeze_(0)
                                    save_grids = torch.cat([x0[b].unsqueeze_(0),
                                                            ref_input[b].unsqueeze_(0),
                                                            perturb_x0[b].unsqueeze_(0),
                                                            src_vis,
                                                            blurred_x0[b].unsqueeze_(0),
                                                            final[b].unsqueeze_(0)], dim=0)
                                    all_iter_grids.append(save_grids)

                            # visual
                            src_name_ = src_name[b].split(".")[0]  # Remove file suffixes
                            ref_name_ = ref_name[b].split(".")[0]  # Remove file suffixes
                            save_path = self.results_path.joinpath(
                                f"{self.step + b}-{src_name_}-to-{ref_name_}-I-{ith + 1}-K-{kth + 1}.png"
                            )
                            tv_util.save_image(torch.cat(all_iter_grids, dim=0), save_path, nrow=6)

                self.step += self.actual_batch_size
                pbar.update(1)

        self.close()
