# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torchvision import utils as tv_utils

from libs.engine import ModelState
from libs.utils import cycle
from libs.metric.lpips_origin import LPIPS
from sketch_nn.dataset.build import build_image2image_translation_dataset
from sketch_nn.photo2sketch import photo2sketch_model_build_util


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
        self.eps_model.train()
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
        self.all_results_path = self.results_path.joinpath("all")
        self.all_results_path.mkdir(exist_ok=True)
        self.results_path = self.results_path.joinpath("samples")
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

        # build
        resnet = torchvision.models.resnet50(pretrained=True)
        image_feature_model = torch.nn.Sequential(*list(resnet.children())[:-2])
        image_feature_model.requires_grad_(True)
        # for param in image_feature_model.parameters():
        #     param.requires_grad = False
        self.image_feature_model = self.accelerator.prepare(image_feature_model)

        # build dataset and dataloaders
        transform = transforms.Compose([
            transforms.Resize([args.image_size, args.image_size]),
            transforms.ToTensor()
        ])
        self.src_dataloader = build_image2image_translation_dataset(args.dataset,
                                                                    args.data_folder,
                                                                    split='test',
                                                                    data_transforms=transform,
                                                                    batch_size=args.valid_batch_size,
                                                                    shuffle=False,
                                                                    drop_last=True,
                                                                    num_workers=args.num_workers)
        self.ref_dataloader = build_image2image_translation_dataset(args.dataset,
                                                                    args.ref_data_folder,
                                                                    split='test',
                                                                    data_transforms=transform,
                                                                    batch_size=args.valid_batch_size,
                                                                    shuffle=False,
                                                                    drop_last=True,
                                                                    num_workers=args.num_workers)
        self.src_dataloader, self.ref_dataloader = \
            self.accelerator.prepare(self.src_dataloader, self.ref_dataloader)
        self.src_dataloader = cycle(self.src_dataloader)
        self.ref_dataloader = cycle(self.ref_dataloader)

        photo2sketch_model = photo2sketch_model_build_util(method='InformativeDrawings')
        self.photo2sketch_model = self.accelerator.prepare(photo2sketch_model)

        self.lpips_loss_fn = LPIPS(net='alex').to(self.device)
        self.image_loss = torch.nn.L1Loss()

        self.optimizer = torch.optim.SGD(self.eps_model.parameters(), lr=0.001, momentum=0.9)

    @torch.no_grad()
    def image_feature_stop_grad(self, image_data):
        output = self.image_feature_model(image_data)
        return output

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
            self.eps_model.train()
            self.classifier.eval()
            model_kwargs = {}

            classes = torch.randint(
                low=0, high=self.num_classes, size=(self.batch_size,), device=device
            )
            model_kwargs["y"] = classes

            src_sample = next(self.src_dataloader)
            src_input, src_name = src_sample["image"], src_sample["fname"]
            ref_sample = next(self.ref_dataloader)
            ref_input, ref_name = ref_sample["image"], ref_sample["fname"]
            src_input, ref_input = src_input.to(self.device), ref_input.to(self.device)

            sample_fn = (
                self.diffusion_model.p_sample_loop
                if not self.use_ddim else self.diffusion_model.ddim_sample_loop
            )
            sample = sample_fn(
                self.model_fn,
                (self.batch_size, 3, img_size, img_size),
                clip_denoised=self.clip_denoised,
                model_kwargs=model_kwargs,
                # cond_fn=self.cond_fn,
                progress=True,
                device=device
            )
            sample.requires_grad_ = True

            g_image_feature = self.image_feature_stop_grad(sample)
            o_image_feature = self.image_feature_model(src_input)

            g_sketch = self.photo2sketch_model(sample).requires_grad_()
            g_sketch.requires_grad_ = True

            loss_image = self.image_loss(g_image_feature, o_image_feature)
            loss_percept = self.lpips_loss_fn(g_sketch, ref_input).to(self.device)
            loss = loss_image + loss_percept

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

                for i in range(len(images_tensor.shape[0])):
                    out_path = self.all_results_path / f"I-{idx}-S{i}-{src_name[i]}-to-{ref_name[i]}.png"
                    tv_utils.save_image(images_tensor.float(),
                                        out_path,
                                        nrow=1, padding=0, normalize=False)

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
