# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Description:

from typing import Union, Dict, List
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from dpm_nn.guided_dpm.beta_schedule import get_named_beta_schedule
from dpm_nn.utils import normalize, unnormalize, extract


class IterativeSDEdit(nn.Module):

    def __init__(
            self,
            num_timesteps: int,
            beta_schedule: str,
            var_type: str
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

    def denoising_step(
            self,
            y: torch.Tensor,  # perturbed condition
            t: torch.Tensor,
            sde_model: nn.Module,
            model_name: str = 'ADM',
    ) -> torch.Tensor:
        """ Sample from p(x_{t-1} | x_t) """

        # mean for SDE
        with torch.no_grad():
            if model_name == 'ADM':
                model_output = sde_model(y, t)
                model_output, _ = torch.split(model_output, 3, dim=1)
            elif model_name == 'DDPM':
                model_output = sde_model(y, t)
        weighted_score = self.betas / torch.sqrt(1 - self.alphas_cumprod)
        mean = extract(1 / torch.sqrt(self.alphas), t, y.shape) * (
                y - extract(weighted_score, t, y.shape) * model_output)

        # add noise
        logvar = extract(self.logvar, t, y.shape)
        noise = torch.randn_like(y)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((y.shape[0],) + (1,) * (len(y.shape) - 1))
        sample = mean + mask * torch.exp(0.5 * logvar) * noise
        sample = sample.float()
        return sample

    def iterative_sampling_progressive(self,
                                       src_input: torch.tensor,
                                       ref_input: torch.tensor,
                                       iter_step: int = 0,
                                       iter_kwargs: Dict = None,
                                       repeat_step: List = None,
                                       perturb_step: List = None,
                                       model: nn.Module = None,
                                       model_kwargs: Dict = None,
                                       device: torch.device = None,
                                       recorder: Union[tqdm] = None) -> OrderedDict:
        bs = src_input.size(0)

        # normalize
        src_input = normalize(src_input)
        ref_input = normalize(ref_input)

        y_iter = None
        blurred_src = None
        repeat_results_record = OrderedDict()

        for ith in range(iter_step):
            if ith == 0:  # iter1: shape reconstruction
                y0 = ref_input
            else:  # iter2: color reconstruction
                down, up = iter_kwargs['low_passer']
                blurred_src = up(down(src_input))

                # resize
                if blurred_src.shape[2] != src_input.shape[2] and blurred_src.shape[3] != src_input.shape[3]:
                    from torchvision.transforms import Resize
                    resizer = Resize(src_input.shape[2])
                    blurred_src = resizer(blurred_src)

                # mixin iter1_product and source image
                y0 = blurred_src * iter_kwargs['fusion_scale'] + y_iter * (1 - iter_kwargs['fusion_scale'])

            # SDEdit by repeating it K times.
            for kth in range(repeat_step[ith]):
                perturb_step_ = perturb_step[ith]
                eps = torch.randn_like(y0)
                # the start point M: y ∼ qM|0(y|x0)
                y = y0 * self.sqrt_alphas_cumprod[perturb_step_ - 1] \
                    + eps * self.sqrt_one_minus_alphas_cumprod[perturb_step_ - 1]
                perturb_x0 = deepcopy(y)

                for ri in reversed(range(perturb_step_)):
                    batched_t = torch.full(size=(bs,), fill_value=ri, device=device, dtype=torch.long)

                    y_ = self.denoising_step(
                        y,
                        batched_t,
                        sde_model=model,
                        model_name='ADM',
                    )
                    y = y_

                    if recorder is not None:
                        dict_ = {"ith": ith, "kth": kth, "ri": ri}
                        recorder.set_postfix(**dict_)

                # visual
                repeat_results_record[f"{ith}-{kth}-th"] = [
                    unnormalize(y0),
                    unnormalize(y),
                    unnormalize(perturb_x0),
                    unnormalize(blurred_src) if blurred_src is not None else torch.zeros_like(perturb_x0)
                ]
                # kth repeat
                y0 = y

            # ith repeat
            y_iter = y0

        return repeat_results_record
