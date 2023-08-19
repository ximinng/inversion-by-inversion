# -*- coding: utf-8 -*-
# Description:

import math
import numpy as np


def get_named_beta_schedule(
        schedule_name: str,
        timesteps: int,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "scaled_linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / timesteps
        beta_start = scale * beta_start
        beta_end = scale * beta_end
        betas = np.linspace(
            beta_start, beta_end, timesteps, dtype=np.float64
        )
    elif schedule_name == "linear":
        betas = np.linspace(
            beta_start, beta_end, timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        betas = _betas_for_alpha_bar(
            timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    elif schedule_name == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, timesteps, 0.1)
    elif schedule_name == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, timesteps, 0.5)
    elif schedule_name == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5, beta_end ** 0.5, timesteps,
                    dtype=np.float64
                ) ** 2
        )
    elif schedule_name == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            timesteps, 1, timesteps, dtype=np.float64
        )
    elif schedule_name == "sigmoid":
        def sigmoid(x):
            s = 1 / (1 + np.exp(-x))
            return s

        betas = np.linspace(-6, 6, timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

    assert betas.shape == (timesteps,)
    return betas


def _betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(
            min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)
        )
    return np.array(betas)


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas
