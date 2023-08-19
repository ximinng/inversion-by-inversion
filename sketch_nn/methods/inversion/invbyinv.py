# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Description:
from pathlib import Path
from typing import Union, Dict, List, Callable
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torchvision import transforms
import numpy as np
from tqdm.auto import tqdm

from libs.metric.lpips_origin import LPIPS
from dpm_nn.guided_dpm.beta_schedule import get_named_beta_schedule
from dpm_nn.inversion.egsde import RequiresGradContext, cosine_similarity
from dpm_nn.utils import normalize, unnormalize, extract
from style_transfer.AdaIN import adaptive_instance_normalization, coral


class InversionByInversion(nn.Module):

    def __init__(
            self,
            num_timesteps: int,
            beta_schedule: str,
            var_type: str,
            args
    ):
        super().__init__()
        self.args = args

        betas = get_named_beta_schedule(
            schedule_name=beta_schedule,
            beta_start=0.0001,
            beta_end=0.02,
            timesteps=num_timesteps,
        )
        betas = torch.from_numpy(betas)

        self.num_timesteps = num_timesteps

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        alphas = 1. - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]], dim=0)
        assert alphas_cumprod_prev.shape == (self.num_timesteps,)

        register_buffer('betas', betas)
        register_buffer('alphas', alphas)
        register_buffer('alphas_cumprod', alphas_cumprod)  # $\bar{\alpha}_t$, alpha_bar, alpha_bar_t
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)  # $\bar{\alpha}_{t-1}$, alpha_bar_t-1

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))  # $\sqrt{alpha_bar}}$
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))  # $\sqrt{1 - alpha_bar}$

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        if var_type == "fixedlarge":
            # register_buffer('logvar', torch.log(betas))
            register_buffer('logvar', torch.from_numpy(
                np.log(np.append(posterior_variance[1], betas[1:]))
            ))
        elif var_type == "fixedsmall":
            register_buffer('logvar', torch.log(posterior_variance.clamp(min=1e-20)))

        # structural similarity
        if self.args.shape_metric == 'lpips':
            self.lpips_fn = LPIPS(net='vgg', verbose=True)

    def denoising_step(
            self,
            y: torch.Tensor,  # perturbed condition
            ref: torch.Tensor,  # conditional input
            x0: torch.Tensor,  # exemplar without noise
            xt: torch.Tensor,  # exemplar with noise
            t: torch.Tensor,
            sde_model: nn.Module,
            model_name: str = 'ADM',
            model_kwargs: Dict = None,
            dpm_solver_step_kwargs: Dict = None,
            record: Dict = None,
            vis_stats: Dict = None
    ) -> torch.Tensor:
        """ Sample from p(x_{t-1} | x_t) """
        idx_ = model_kwargs['iter-th']

        # mean for SDE
        with torch.no_grad():
            if model_name == 'ADM':
                dpm_solver = dpm_solver_step_kwargs.get('fn', None)

                # without guide denoising
                if dpm_solver:
                    t_start = dpm_solver_step_kwargs['pre_t'] / 1000
                    t_end = t[:1].item() / 1000
                    if t[0].item() == 0:
                        t_end = None
                    model_output = dpm_solver.sample(x=y, steps=10, t_start=t_start, t_end=t_end, order=3,
                                                     skip_type="time_uniform", method="singlestep")
                    return model_output

                # guide denoising
                model_output = sde_model(y, t)
                model_output, _ = torch.split(model_output, 3, dim=1)
            elif model_name == 'DDPM':
                model_output = sde_model(y, t)

        weighted_score = self.betas / torch.sqrt(1 - self.alphas_cumprod)
        mean = extract(1 / torch.sqrt(self.alphas), t, y.shape) * (
                y - extract(weighted_score, t, y.shape) * model_output)

        if vis_stats is not None:
            vis_stats['mean_init'] = mean.clone().detach().cpu().numpy()

        # inversion-guidance
        weight_energy = self.betas / torch.sqrt(self.alphas)
        weight_t = extract(weight_energy, t, y.shape)

        # sketch shape control
        if self.args.use_shape[idx_] and model_kwargs.get('geom_expert', False):
            with RequiresGradContext(y, requires_grad=True):
                # generate y
                mean_ = extract(1 / torch.sqrt(self.alphas), t, y.shape) * (
                        y - extract(weighted_score, t, y.shape) * model_output)
                logvar_ = extract(self.logvar, t, y.shape)
                noise_ = torch.randn_like(y)
                mask_ = 1 - (t == 0).float()
                mask_ = mask_.reshape((y.shape[0],) + (1,) * (len(y.shape) - 1))
                sample_ = mean_ + mask_ * torch.exp(0.5 * logvar_) * noise_
                sample_ = sample_.float()

                if self.args.ys_rescale:
                    sample_ = unnormalize(sample_)

                # y to sketch
                geom_expert = model_kwargs['geom_expert']
                # photo2sketch, the single channel replicates the three channels
                Y_geom = geom_expert(sample_).repeat(1, 3, 1, 1)
                R_geom = ref

                # feature reshape
                if len(Y_geom.shape) == 4 and self.args.shape_metric != 'lpips':
                    b, c, h, w = Y_geom.shape
                    Y_geom = Y_geom.reshape(b, c, h * w)
                    R_geom = ref.reshape(b, c, h * w)

                # geometry inversion
                if self.args.shape_metric == 'l2':
                    energy = F.mse_loss(Y_geom, R_geom, reduction="none")
                elif self.args.shape_metric == 'l1':
                    energy = F.l1_loss(Y_geom, R_geom, reduction="none")
                elif self.args.shape_metric == 'lpips':
                    energy = self.lpips_fn(Y_geom, R_geom)
                elif self.args.shape_metric == 'cosine':
                    energy = F.cosine_similarity(Y_geom, R_geom)

                grad = autograd.grad(energy.sum(), y)[0]
            mean = mean - self.args.lam_shape[idx_] * weight_t * grad.detach()

            if vis_stats is not None:
                vis_stats['grad_shape'] = grad.clone().detach().cpu().numpy()

            record['l_shape'].append(energy.mean().item())
            del mean_, logvar_, noise_, mask_, sample_

        # texture control -- pixel-level
        if self.args.pixel_texture[idx_]:
            down, up = model_kwargs['low_passer']
            with RequiresGradContext(y, requires_grad=True):
                # generate y
                mean_ = extract(1 / torch.sqrt(self.alphas), t, y.shape) * (
                        y - extract(weighted_score, t, y.shape) * model_output)
                logvar_ = extract(self.logvar, t, y.shape)
                noise_ = torch.randn_like(y)
                mask_ = 1 - (t == 0).float()
                mask_ = mask_.reshape((y.shape[0],) + (1,) * (len(y.shape) - 1))
                sample_ = mean_ + mask_ * torch.exp(0.5 * logvar_) * noise_
                sample_ = sample_.float()
                y_inv = sample_
                # y_inv = unnormalize(sample_)

                # blur generated y or not
                Y = up(down(y_inv)) if self.args.blur_y else y_inv
                X = up(down(xt)) if self.args.blur_xt else xt

                energy = F.mse_loss(Y, X, reduction="none")
                grad = autograd.grad(energy.sum(), y)[0]
            mean = mean - self.args.lam_pixel_texture[idx_] * weight_t * grad.detach()

            if vis_stats is not None:
                vis_stats['grad_texture'] = grad.clone().detach().cpu().numpy()

            record['l_texture'].append(energy.mean().item())
            del mean_, logvar_, noise_, mask_, sample_, y_inv

        # texture control -- feature-map-level
        if self.args.feature_texture[idx_] and model_kwargs.get('texture_feat_model', False):
            _feat_model = model_kwargs['texture_feat_model']

            with RequiresGradContext(y, requires_grad=True):
                if self.args.feature_texture_model == 'CLIP':
                    y_aug, x0_aug = self.drawing_pair_augment(y, x0, im_res=224)
                    l_clip_fc, l_clip_conv = _feat_model.compute_visual_distance(
                        y_aug, x0_aug, clip_norm=False
                    )
                    energy = sum(l_clip_conv) + l_clip_fc
                else:
                    feats_y, _ = _feat_model(y)
                    feats_x, _ = _feat_model(x0)

                    b, c, h, w = feats_x.shape
                    feats_y_ = feats_y.reshape(b, c, h * w)
                    feats_x_ = feats_x.reshape(b, c, h * w)

                    if self.args.feature_texture_metric == "l1":
                        energy = F.l1_loss(feats_y_, feats_x_, reduction="none")
                    elif self.args.feature_texture_metric == "l2":
                        energy = F.mse_loss(feats_y_, feats_x_, reduction="none")
                    elif self.args.feature_texture_metric == "cosine":
                        energy = F.cosine_similarity(feats_y_, feats_x_)
                    energy = energy.sum()

                grad = autograd.grad(energy, y)[0]
            mean = mean - self.args.lam_feature_texture[idx_] * weight_t * grad.detach()

            record['l_feat_texture'].append(energy.mean().item())

        # domain-specific features
        if self.args.use_dse[idx_] and model_kwargs.get('dse', False):
            dse = model_kwargs.get('dse')
            with RequiresGradContext(y, requires_grad=True):
                Y = dse(y, t)  # shape: [4, 512, 8, 8]
                X = dse(xt, t)
                energy = cosine_similarity(Y, X)
                grad = autograd.grad(energy.sum(), y)[0]
            mean = mean - self.args.lam_dse[idx_] * weight_t * grad.detach()

            record['l_dse'].append(energy.mean().item())

        # add noise
        logvar = extract(self.logvar, t, y.shape)
        noise = torch.randn_like(y)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((y.shape[0],) + (1,) * (len(y.shape) - 1))
        sample = mean + mask * torch.exp(0.5 * logvar) * noise
        sample = sample.float()

        return sample

    def inversion_step(self,
                       src_input: torch.tensor,
                       ref_input: torch.tensor,
                       n_stage: int = 0,
                       repeat_step: List = None,
                       perturb_step: List = None,
                       model: nn.Module = None,
                       model_kwargs: Dict = None,
                       dpm_solver_kwargs: Dict = None,
                       device: torch.device = None,
                       recorder: Union[tqdm] = None,
                       log_fn: Callable = None) -> OrderedDict:
        bs = src_input.size(0)

        # normalize to [-1, 1]
        src_input = normalize(src_input)
        ref_input = normalize(ref_input)

        y_iter = None
        repeat_results_record = OrderedDict()

        for ith in range(n_stage):
            model_kwargs['iter-th'] = ith

            # get dpm_solver args if used
            dpm_solver, dpm_solver_step_kwargs = None, {}
            t_guided_, t_dpm_solver_ = None, None
            pre_t = perturb_step[ith]  # recode last step, init pre_t
            if self.args.use_dpm_solver:
                dpm_solver = dpm_solver_kwargs['dpm_solver']
                t_guided_ = dpm_solver_kwargs['t_guided'][ith]
                t_dpm_solver_ = dpm_solver_kwargs['t_dpm_solver'][ith]

            if ith == 0:  # iter1: shape reconstruction
                y0 = ref_input
            else:  # iter2: color reconstruction
                # The input of iter2 is the output of iter1
                y0 = y_iter

            # style injection
            if self.args.use_style[ith]:
                if self.args.preserve_color:
                    src_input = coral(y0, src_input)
                y0 = style_transfer(model_kwargs['style_vgg_encoder'], model_kwargs['style_decoder'],
                                    y0, src_input, self.args.alpha)

            # SDEdit by repeating it K times.
            for kth in range(repeat_step[ith]):
                step_recorder = {'l_shape': [], 'l_depth': [], 'l_texture': [], 'l_feat_texture': [], 'l_dse': []}

                perturb_step_ = perturb_step[ith]
                eps = torch.randn_like(y0)
                # the start point M: y ∼ qM|0(y|x0)
                y = y0 * self.sqrt_alphas_cumprod[perturb_step_ - 1] \
                    + eps * self.sqrt_one_minus_alphas_cumprod[perturb_step_ - 1]
                perturb_x0 = y.detach()

                if self.args.save_intermediate:
                    stats_steps_ = {'mean_init': [], 'mean_shape': [], 'mean_texture': [],
                                    'grad_shape': [], 'grad_texture': []}

                for ri in reversed(range(perturb_step_)):
                    if dpm_solver and (ri not in t_guided_) and (ri not in t_dpm_solver_):
                        continue

                    # dpm_solver:
                    if dpm_solver:
                        fn = None if ri in t_dpm_solver_ else dpm_solver
                        dpm_solver_step_kwargs = {'fn': fn, 'pre_t': pre_t}

                    batched_t = torch.full(size=(bs,), fill_value=ri, device=device, dtype=torch.long)
                    src_noised = src_input * self.sqrt_alphas_cumprod[ri] + eps * self.sqrt_one_minus_alphas_cumprod[ri]

                    stats_ = {} if self.args.save_intermediate else None

                    y_ = self.denoising_step(
                        y=y,
                        ref=ref_input,
                        x0=src_input,  # without noise
                        xt=src_noised,
                        t=batched_t,
                        sde_model=model,
                        model_name='ADM' if self.args.dataset != "male2female" else 'DDPM',
                        model_kwargs=model_kwargs,
                        dpm_solver_step_kwargs=dpm_solver_step_kwargs,
                        record=step_recorder,
                        vis_stats=stats_
                    )
                    y = y_

                    pre_t = ri  # record dpm_solver last step

                    if self.args.save_intermediate:  # collect data
                        for k, v in stats_.items():
                            stats_steps_[k].append(stats_[k])

                    # log denoising step
                    if recorder is not None:
                        dict_ = {"ith": ith, "kth": kth, "ri": ri}
                        for k, v in step_recorder.items():
                            if len(v) > 0:
                                dict_[k] = np.array(v).mean()
                        recorder.set_postfix(**dict_)

                # visual for debugging
                if not self.args.get_final_results:
                    vis_optional_ = torch.zeros_like(src_input)

                    repeat_results_record[f"{ith}-{kth}-th"] = [
                        unnormalize(y0),
                        unnormalize(y),
                        unnormalize(perturb_x0),
                        unnormalize(vis_optional_)
                    ]

                if self.args.save_intermediate:
                    fpath = Path(self.args.visual_fpath)
                    if not fpath.exists():
                        fpath.mkdir(parents=True, exist_ok=True)
                    for k, v in stats_steps_.items():
                        if len(v) > 0:
                            np.save(file=f"{fpath}/{ith}-{kth}-{str(k)}", arr=v)

                # kth repeat
                y0 = y

            # ith repeat
            y_iter = y0

            if self.args.get_final_results:
                repeat_results_record["out"] = unnormalize(y_iter)

        return repeat_results_record

    def drawing_pair_augment(self,
                             x: torch.Tensor,
                             y: torch.Tensor,
                             im_res: int):
        # CLIP inputs
        clip_norm_resize = transforms.Compose([
            transforms.Resize(im_res, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(im_res),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        # make augmentation pairs
        x_augs, y_augs = [clip_norm_resize(x)], [clip_norm_resize(y)]
        xs = torch.cat(x_augs, dim=0)
        ys = torch.cat(y_augs, dim=0)
        return xs, ys


def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(content.device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)
