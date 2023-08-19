# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
from torchvision import utils as tv_utils

from libs.engine import ModelState


class ClassifierGuidedSampler(ModelState):
    """
    Sampling for evaluation
    """

    def __init__(
            self,
            diffusion_model,
            eps_model,
            dpm_path,
            classifier,
            clf_path,
            args,
            *,
            classifier_scale=1.0,
            clip_denoised=True,
            total_samples=2000,
            mixed_precision="no",
            use_dpm_solver=False
    ):
        super().__init__(args)

        self.print(f"total sample: {total_samples}")

        self.diffusion_model = diffusion_model
        self.print(f"\nloading DDPM from `{dpm_path}` ....")
        self.eps_model = self.load_ckpt_model_only(eps_model, dpm_path)
        if args.model.use_fp16:
            self.eps_model.convert_to_fp16()
        self.eps_model.eval()
        self.print(f"-> U-Net Params: {(sum(p.numel() for p in eps_model.parameters()) / 1e6):.3f}M")

        self.print(f"loading classifier from `{clf_path}` ....")
        self.classifier = classifier
        self.classifier = self.load_ckpt_model_only(self.classifier, clf_path)
        if args.classifier.use_fp16:
            self.classifier.convert_to_fp16()
        self.classifier.eval()
        self.print(f"-> Classifier Params: {(sum(p.numel() for p in classifier.parameters()) / 1e6):.3f}M")
        self.use_dpm_solver = use_dpm_solver
        # update results_path
        self.results_path = self.results_path.joinpath("sample")
        self.results_path.mkdir(exist_ok=True)

        self.image_size = args.image_size
        self.num_classes = args.num_classes

        self.classifier_scale = classifier_scale
        self.class_cond = args.model.class_cond
        self.clip_denoised = clip_denoised

        self.use_ddim: bool = self.args.diffusion.use_ddim if self.args.use_ddim is False else self.args.use_ddim
        self.total_samples, self.batch_size = total_samples, args.batch_size

        # step counter state
        self.step = 0

        # prepare models
        self.classifier, self.eps_model \
            = self.accelerator.prepare(self.classifier, self.eps_model)

    def cond_fn(self, x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)  # shape: [B, num_classes]
            selected = log_probs[range(len(logits)), y.view(-1)]  # index ground-truth dimensions
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale  # the score of log p(y | x_t)

    def model_fn(self, x, t, y=None):
        assert y is not None
        return self.eps_model(x, t, y if self.class_cond else None)

    @torch.no_grad()
    def sampling(self):
        self.accelerator.print("\nStart sampling...")

        accelerator = self.accelerator
        device = accelerator.device

        img_size = self.image_size

        all_images = []  # for evaluation
        all_labels = []
        idx = 0
        while len(all_images) < self.total_samples:
            model_kwargs = {}
            self.classifier.eval()
            self.eps_model.eval()

            classes = torch.randint(
                low=0, high=self.num_classes, size=(self.batch_size,), device=device
            )
            model_kwargs["y"] = classes

            sample_fn = (
                self.diffusion_model.p_sample_loop
                if not self.use_ddim else self.diffusion_model.ddim_sample_loop
            )
            sample = sample_fn(
                self.model_fn,
                (self.batch_size, 3, img_size, img_size),
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=self.cond_fn,
                progress=True,
                device=device
            )

            sample_copy = deepcopy(sample)
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()  # shape: [B, W, H, C]

            # visualize photos
            if self.accelerator.is_main_process:
                out_path = self.results_path.joinpath(f"{img_size}-sampled-{idx}.png")

                for i in range(sample_copy.shape[0]):
                    image_tensor = sample_copy[i].unsqueeze(0)
                    if i == 0:
                        image_tensor_last = image_tensor
                        continue
                    image_tensor_last = torch.cat((image_tensor_last, image_tensor), dim=0)
                images_tensor = (image_tensor_last + 1) / 2
                tv_utils.save_image(images_tensor.float(), out_path,
                                    nrow=int(math.sqrt(images_tensor.shape[0])),
                                    padding=0, normalize=False)
                print(f"saving to `{out_path}`")

            # gather images
            gathered_samples = self.accelerator.gather(sample)
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            # gather labels
            gathered_labels = self.accelerator.gather(classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            self.print(f"created {len(all_images)} samples")

            idx += 1

        # saving `.npz` file for evaluating
        image_arr = np.concatenate(all_images, axis=0)
        image_arr = image_arr[: self.total_samples]
        label_arr = all_labels[: self.total_samples]

        if self.accelerator.is_main_process:
            shape_str = "x".join([str(x) for x in image_arr.shape])
            save_path = self.results_path.joinpath(f"samples_{shape_str}.npz")
            np.savez(save_path, image_arr, label_arr)
            print(f"saving to `{save_path}`")

        self.print('Sampling complete!')
