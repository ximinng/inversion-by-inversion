# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

from omegaconf import DictConfig

from .ADMs import UNetModel as ClfGuidedUnet, EncoderUNetModel

__all__ = ['ADMs_build_util']


def ADMs_build_util(
        image_size,
        num_classes,  # class conditional
        model_cfg: DictConfig,
        dpm_cfg: DictConfig,
        build_clf: bool = False,
        clf_cfg: DictConfig = None
):
    if image_size == 512:
        channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    elif image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size in [128, 96]:
        channel_mult = (1, 1, 2, 3, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported image size: {image_size}")
    model_cfg.channel_mult = channel_mult  # add key

    _attention_resolutions = model_cfg.attention_resolutions  # example: [32, 16, 8]
    attention_ds = []
    for res in model_cfg.attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))
    model_cfg.attention_ds = attention_ds  # add key

    model_cfg.out_channels = (3 if not dpm_cfg.learn_sigma else 6)  # update key

    eps_model = ClfGuidedUnet(
        image_size=image_size,
        in_channels=model_cfg.in_channels,
        num_res_blocks=model_cfg.num_res_blocks,  # depth
        model_channels=model_cfg.num_channels,  # width
        out_channels=model_cfg.out_channels,
        attention_resolutions=tuple(model_cfg.attention_ds),
        dropout=model_cfg.dropout,
        channel_mult=model_cfg.channel_mult,
        num_classes=(num_classes if model_cfg.class_cond else None),
        num_heads=model_cfg.num_heads,
        num_head_channels=model_cfg.num_head_channels,
        use_scale_shift_norm=model_cfg.use_scale_shift_norm,
        resblock_updown=model_cfg.resblock_updown,
        use_new_attention_order=model_cfg.use_new_attention_order,
        use_fp16=model_cfg.use_fp16
    )

    if (clf_cfg is not None) and build_clf:
        clf_cfg.channel_mult = channel_mult  # add key

        clf_attention_ds = []
        for res in clf_cfg.attention_resolutions.split(","):
            clf_attention_ds.append(image_size // int(res))
        clf_cfg.attention_ds = attention_ds  # add key

        clf_model = EncoderUNetModel(
            image_size=image_size,
            in_channels=clf_cfg.in_channels,
            model_channels=clf_cfg.model_channels,  # width
            num_res_blocks=clf_cfg.num_res_blocks,  # depth
            out_channels=clf_cfg.out_channels,
            attention_resolutions=tuple(clf_cfg.attention_ds),
            dropout=clf_cfg.dropout,
            num_head_channels=clf_cfg.num_head_channels,
            channel_mult=clf_cfg.channel_mult,
            use_scale_shift_norm=clf_cfg.use_scale_shift_norm,
            resblock_updown=clf_cfg.resblock_updown,
            pool=clf_cfg.pool,
            use_fp16=clf_cfg.use_fp16
        )
        return eps_model, clf_model
    else:
        return eps_model, None
