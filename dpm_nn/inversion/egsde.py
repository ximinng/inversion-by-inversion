# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Description:

from typing import Union, List, Dict
from copy import deepcopy
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from dpm_nn.guided_dpm.beta_schedule import get_named_beta_schedule
from dpm_nn.utils import normalize, unnormalize, extract
from dpm_nn.dpm_solver import NoiseScheduleVP, DPM_Solver, model_wrapper


class EnergyGuidedSDE(nn.Module):

    def __init__(
            self,
            num_timesteps: int,
            beta_schedule: str,
            var_type: str,
            expert_kwargs: Union[DictConfig, Dict] = None
    ):
        super().__init__()

        betas = get_named_beta_schedule(
            schedule_name=beta_schedule,
            beta_start=0.0001,
            beta_end=0.02,
            timesteps=num_timesteps,
        )
        betas = torch.from_numpy(betas)

        self.num_timesteps = num_timesteps
        self.expert_kwargs = expert_kwargs

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
            register_buffer('logvar', torch.log(betas))
        elif var_type == "fixedsmall":
            register_buffer('logvar', torch.log(posterior_variance.clamp(min=1e-20)))

    def denoising_step(
            self,
            y: torch.Tensor,  # perturbed condition
            t: torch.Tensor,
            xt: torch.Tensor,  # noise at t step
            experts: Dict,
            sde_model: nn.Module,
            model_name: str = 'ADM',
            record: OrderedDict = None,
            dpm_solver: nn.Module = None,
            pre_T: int = 0
    ) -> torch.Tensor:
        dse, die = experts['dse'], experts['die']
        lam_s, lam_i = self.expert_kwargs.lam_s, self.expert_kwargs.lam_i
        realistic_metric, faithful_metric = self.expert_kwargs.s1, self.expert_kwargs.s2

        # mean for SDE
        with torch.no_grad():
            if model_name == 'ADM':
                if dpm_solver:
                    t_start = pre_T / 1000
                    t_end = t[:1].item() / 1000
                    if t[:1].item() == 0:
                        t_end = None
                        # args' detail : ./dpm_solver.py
                    model_output = dpm_solver.sample(
                        x=y,
                        steps=10,
                        t_start=t_start,
                        t_end=t_end,
                        order=3,
                        skip_type="time_uniform",
                        method="singlestep",
                    )
                    return model_output
                else:
                    model_output = sde_model(y, t)
                    model_output, _ = torch.split(model_output, 3, dim=1)
            elif model_name == 'DDPM':
                model_output = sde_model(y, t)
        weighted_score = self.betas / torch.sqrt(1 - self.alphas_cumprod)
        mean = extract(1 / torch.sqrt(self.alphas), t, y.shape) * (
                y - extract(weighted_score, t, y.shape) * model_output)

        # inversion-guidance
        weight_energy = self.betas / torch.sqrt(self.alphas)
        weight_t = extract(weight_energy, t, y.shape)

        # realistic expert based on domain-specified extractor
        with RequiresGradContext(y, requires_grad=True):
            Y = dse(y, t)  # shape: [4, 512, 8, 8]
            X = dse(xt, t)
            if realistic_metric == 'cosine':
                energy = cosine_similarity(Y, X)
            if realistic_metric == 'neg_l2':
                energy = - mse(Y, X)
            grad = autograd.grad(energy.sum(), y)[0]
        mean = mean - lam_s * weight_t * grad.detach()

        record['ds_g'].append(grad.mean().item())
        record['ds_m'].append(mean.mean().item())

        # faithful expert based on domain-independent extractor
        down, up = die
        with RequiresGradContext(y, requires_grad=True):
            Y = up(down(y))
            X = up(down(xt))
            if faithful_metric == 'cosine':
                energy = - cosine_similarity(X, Y)
            if faithful_metric == 'neg_l2':
                energy = mse(X, Y)
            grad = autograd.grad(energy.sum(), y)[0]
        mean = mean - lam_i * weight_t * grad.detach()

        record['di_g'].append(grad.mean().item())
        record['di_m'].append(mean.mean().item())

        # add noise
        logvar = extract(self.logvar, t, y.shape)
        noise = torch.randn_like(y)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((y.shape[0],) + (1,) * (len(y.shape) - 1))
        sample = mean + mask * torch.exp(0.5 * logvar) * noise
        sample = sample.float()
        return sample

    def sampling_progressive(self,
                             src_input: torch.tensor,
                             experts: Dict,
                             repeat_step: int = 1,
                             perturb_step: int = 500,
                             model: nn.Module = None,
                             model_kwargs: Dict = None,
                             device: torch.device = None,
                             recorder: Union[tqdm] = None,
                             use_dpm_solver: bool = False) -> OrderedDict:
        bs = src_input.size(0)
        # normalize
        y0 = normalize(src_input).to(device)
        # let x0 be source image
        x0 = y0

        repeat_results_record = OrderedDict()
        # EGSDE by repeating it K times. (usually set 1) (see Appendix A.2)
        dpm_solver = None
        if use_dpm_solver:
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

            # 2. Convert your discrete-time `model` to the continuous-time noise prediction model
            kwargs = {'split': True}
            model_fn = model_wrapper(
                model,
                noise_schedule=noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs=kwargs,
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

        for kth in range(repeat_step):
            inner_record = OrderedDict({'ds_g': [], 'ds_m': [], 'di_g': [], 'di_m': []})

            eps = torch.randn_like(y0)
            # the start point M: y ∼ qM|0(y|x0)
            y = y0 * self.sqrt_alphas_cumprod[perturb_step - 1] \
                + eps * self.sqrt_one_minus_alphas_cumprod[perturb_step - 1]
            perturb_x0 = deepcopy(y)

            # The parameters t_start and t_end of dpm_solver are the time at which noise removal begins and the time at which noise removal ends. 
            # The input noise graph must correspond to t_start
            # The pre_T is the t_start multiply N, N is the betas' length
            pre_T = perturb_step - 1
            # example 499-498-497-496-400(dpm_solver)-
            dpm_solver_t = [400, 300, 200, 100, 0]
            wo_dpm_solver_t = [499, 498, 497, 496, 399, 398, 397, 396, 299, 298, 297, 296, 199, 198, 197, 196, 99, 98,
                               97, 96]

            for ri in reversed(range(perturb_step)):
                if dpm_solver and (ri not in dpm_solver_t) and (ri not in wo_dpm_solver_t):
                    continue
                batched_t = torch.full(size=(bs,), fill_value=ri, device=device, dtype=torch.long)
                # sample perturbed source image from the perturbation kernel: x ∼ qs|0(x|x0)
                xt = x0 * self.sqrt_alphas_cumprod[ri] + eps * self.sqrt_one_minus_alphas_cumprod[ri]
                # egsde update (see VP-EGSDE in Appendix A.3)
                if dpm_solver_t:
                    fn = dpm_solver
                    if ri in wo_dpm_solver_t:
                        fn = None
                    y_ = self.denoising_step(
                        y=y,
                        t=batched_t,
                        xt=xt,
                        experts=experts,
                        sde_model=model,
                        model_name='ADM',
                        record=inner_record,
                        dpm_solver=fn,
                        pre_T=pre_T
                    )
                    pre_T = ri
                else:
                    y_ = self.denoising_step(
                        y=y,
                        t=batched_t,
                        xt=xt,
                        experts=experts,
                        sde_model=model,
                        model_name='ADM',
                        record=inner_record
                    )
                y = y_

                if recorder is not None:
                    recorder.set_postfix(
                        kth=kth,
                        ri=ri,
                        ds_g=np.array(inner_record['ds_g']).mean(),
                        ds_m=np.array(inner_record['ds_m']).mean(),
                        di_g=np.array(inner_record['di_g']).mean(),
                        di_m=np.array(inner_record['di_m']).mean()
                    )

            y0 = y
            # unnormalize
            y = unnormalize(y)

            repeat_results_record[f"{kth}-th"] = [y, unnormalize(perturb_x0)]
            if recorder is not None:
                recorder.write(
                    f"Step: {model_kwargs['step']} | k: {kth}, "
                    f"ds_g: {np.array(inner_record['ds_g']).mean()}, "
                    f"ds_m: {np.array(inner_record['ds_m']).mean()}, "
                    f"di_g: {np.array(inner_record['di_g']).mean()}, "
                    f"di_m: {np.array(inner_record['di_m']).mean()}"
                )
        return repeat_results_record


def cosine_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    compute cosine similarity for each pair of image
    Args:
        X: shape: [B, C, H, W]
        Y: shape: [B, C, H, W]

    Returns:
         similarity, shape: [B, 1]
    """

    def _norm(t: torch.Tensor):
        return F.normalize(t, dim=1, eps=1e-10)

    b, c, h, w = X.shape
    X = X.reshape(b, c, h * w)
    Y = Y.reshape(b, c, h * w)
    corr = _norm(X) * _norm(Y)  # shape: [B, C, H*W]
    similarity = corr.sum(dim=1).mean(dim=1)
    return similarity


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).square().sum(dim=(1, 2, 3))


def truncation(global_step, ratio=0.5):
    part = int(global_step * ratio)
    weight_l = torch.zeros(part).reshape(-1, 1)
    weight_r = torch.ones(global_step - part).reshape(-1, 1)
    weight = torch.cat((weight_l, weight_r), dim=0)
    return weight


def judge_requires_grad(obj):
    if isinstance(obj, torch.Tensor):
        return obj.requires_grad
    elif isinstance(obj, nn.Module):
        return next(obj.parameters()).requires_grad
    elif isinstance(obj, torch.nn.parallel.DataParallel):
        return next(obj.module.parameters()).requires_grad
    elif isinstance(obj, torch.nn.parallel.DistributedDataParallel):
        return next(obj.module.parameters()).requires_grad
    else:
        raise TypeError


class RequiresGradContext(object):
    """Specifies the tensors that requires gradient tracking."""

    def __init__(self, *objs, requires_grad: Union[bool, List]):
        self.objs = objs
        self.backups = [judge_requires_grad(obj) for obj in objs]

        if isinstance(requires_grad, bool):
            self.requires_grads = [requires_grad] * len(objs)
        elif isinstance(requires_grad, list):
            self.requires_grads = requires_grad
        else:
            raise TypeError

        assert len(self.objs) == len(self.requires_grads)

    def __enter__(self):
        for obj, requires_grad in zip(self.objs, self.requires_grads):
            obj.requires_grad_(requires_grad)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for obj, backup in zip(self.objs, self.backups):
            obj.requires_grad_(backup)
