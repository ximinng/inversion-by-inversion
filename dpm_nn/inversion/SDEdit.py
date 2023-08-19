# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Description:

from typing import Union, Dict
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from dpm_nn.guided_dpm.beta_schedule import get_named_beta_schedule
from dpm_nn.utils import normalize, unnormalize, extract
from dpm_nn.dpm_solver import NoiseScheduleVP, DPM_Solver, model_wrapper


class SDEdit(nn.Module):

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
            dpm_solver: nn.Module = None,
            pre_T: int = 0
    ) -> torch.Tensor:
        """ Sample from p(x_{t-1} | x_t) """

        # mean for SDE
        with torch.no_grad():
            if model_name == 'ADM':
                if dpm_solver:
                    t_start = pre_T / 1000
                    t_end = t[:1].item() / 1000
                    if t[:1].item() == 0:
                        t_end = None
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
                             mask: torch.tensor = None,
                             repeat_step: int = 1,
                             perturb_step: int = 500,
                             model: nn.Module = None,
                             model_kwargs: Dict = None,
                             device: torch.device = None,
                             recorder: Union[tqdm] = None,
                             use_dpm_solver: bool = False) -> OrderedDict:
        bs = src_input.size(0)

        y0 = normalize(src_input)
        # default: y0 = (y0 - 0.5) * 2. ?

        repeat_results_record = OrderedDict()
        # SDEdit by repeating it K times.
        dpm_solver = None
        if use_dpm_solver:
            noise_schedule = NoiseScheduleVP(schedule='discrete', betas=self.betas)

            kwargs = {'split': True}
            model_fn = model_wrapper(
                model,
                noise_schedule=noise_schedule,
                model_type="noise",  # or "x_start" or "v" or "score"
                model_kwargs=kwargs,
            )
            dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")

        for kth in range(repeat_step):
            eps = torch.randn_like(y0)
            # the start point M: y ∼ qM|0(y|x0)
            y = y0 * self.sqrt_alphas_cumprod[perturb_step - 1] \
                + eps * self.sqrt_one_minus_alphas_cumprod[perturb_step - 1]
            perturb_x0 = deepcopy(y)

            # The parameters t_start and t_end of dpm_solver are the time at which noise removal begins and the time at which noise removal ends. 
            # The input noise graph must correspond to t_start
            # The pre_T is the t_start multiply N, N is the betas' length
            pre_T = perturb_step - 1
            # example 499-498-497-496-400 (dpm_solver)
            dpm_solver_t = [400, 300, 200, 100, 0]
            wo_dpm_solver_t = [499, 498, 497, 496, 399, 398, 397, 396, 299, 298, 297, 296, 199, 198, 197, 196, 99, 98,
                               97, 96]

            for ri in reversed(range(perturb_step)):
                if dpm_solver and (ri not in dpm_solver_t) and (ri not in wo_dpm_solver_t):
                    continue

                batched_t = torch.full(size=(bs,), fill_value=ri, device=device, dtype=torch.long)
                if dpm_solver_t:
                    fn = dpm_solver
                    if ri in wo_dpm_solver_t:
                        fn = None
                    y_ = self.denoising_step(
                        y=y,
                        t=batched_t,
                        sde_model=model,
                        model_name='ADM',
                        dpm_solver=fn,
                        pre_T=pre_T
                    )
                    pre_T = ri
                else:
                    y_ = self.denoising_step(
                        y=y,
                        t=batched_t,
                        sde_model=model,
                        model_name='ADM',
                    )

                if mask is not None:  # edit the masked image
                    y = y0 * self.sqrt_alphas_cumprod[ri] + eps * self.sqrt_one_minus_alphas_cumprod[ri]
                    y[:, (mask != 1.)] = y_[:, (mask != 1.)]

                y = y_

                if recorder is not None:
                    dict_ = {"kth": kth, "ri": ri}
                    recorder.set_postfix(**dict_)

            # visual
            repeat_results_record[f"{kth}-th"] = [unnormalize(y0), unnormalize(y), unnormalize(perturb_x0)]
            # kth repeat
            y0 = y

        return repeat_results_record
