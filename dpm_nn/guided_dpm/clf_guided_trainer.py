# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import utils as tv_utils
from tqdm import tqdm
from torchmetrics import Accuracy

from libs.engine import ModelState
from libs.utils import has_int_squareroot, cycle, exists
from libs.utils.imshow import save_grid_images_and_labels
from libs.utils.meter import AverageMeter
from .loss_resample import UniformSampler, LossAwareSampler


#################################################################################
#                            Classifier Trainer                                 #
#################################################################################

class ClassifierTrainer(ModelState):

    def __init__(
            self,
            diffusion_model,
            eps_model,  # Here eps_model does not involve training and is only used for sampling
            eps_model_path,
            classifier,
            train_dataloader,
            valid_dataloader,
            args,
            *,
            clf_model_path=None,  # fine-tune flag
            class_cond=True,
            classifier_scale=0.5,
            noised=True,
            use_ddim=False,
            train_batch_size=16,
            valid_batch_size=16,
            train_lr=1e-4,
            weight_decay=0,
            adam_betas=(0.9, 0.999),
            anneal_lr=False,
            train_num_steps=100000,
            max_grad_norm=None,
            schedule_sampler=None,
            ckpt_steps=3000,
            vis_steps=1000,
            num_samples=25,
    ):
        super().__init__(args)

        assert has_int_squareroot(num_samples), self.print('number of samples must have an integer square root')

        self.print(f"-> Learning rate: {train_lr}, adam_betas: {adam_betas} ,weight_decay: {weight_decay}")
        self.print(f"-> use DDIM: {use_ddim}, noised training: {noised}, class_cond: {class_cond}")

        self.results_path = self.results_path.joinpath(f"clf-train-result")
        self.results_path.mkdir(exist_ok=True)
        self.print(f"-> The results will be saved in '{self.results_path}'")

        self.num_classes = args.num_classes
        self.num_samples = num_samples

        # load pretrained eps model
        self.print(f"\nloading DDPM from `{eps_model_path}` ....")
        self.diffusion_model = diffusion_model
        ckpt = torch.load(eps_model_path, map_location=self.accelerator.device)

        try:
            eps_model.load_state_dict(ckpt)
        except RuntimeError as err:
            ckpt = ckpt["model"]
            eps_model.load_state_dict(ckpt)
        self.print(f"-> U-Net Params: {(sum(p.numel() for p in eps_model.parameters()) / 1e6):.3f}M")
        del ckpt

        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion_model)
        self.eps_model = eps_model

        self.ckpt_steps = ckpt_steps
        self.vis_steps = vis_steps
        self.eval_steps = vis_steps  # evaluating classifier

        if clf_model_path is not None:
            # fine-tune the classifier
            self.print(f"loading clf from `{clf_model_path}` ....")
            if self.args.dse.load_share_weights:
                # when the network and the pre-training model architecture is different
                self.classifier = self.load_shared_weights(classifier, clf_model_path)
            else:
                self.classifier = self.load_ckpt_model_only(classifier, clf_model_path)
            self.print("Start fine-tuning classifier ...")
        else:
            self.print("Training the classifier from scratch ...")
            self.classifier = classifier
        self.print(f"-> clf Params: {(sum(p.numel() for p in classifier.parameters()) / 1e6):.3f}M")

        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size if valid_batch_size >= self.n_gpus else train_batch_size
        self.train_num_steps = train_num_steps
        self.image_size = args.image_size
        self.max_grad_norm = max_grad_norm

        self.use_ddim = use_ddim
        self.class_cond = class_cond
        self.noised = noised
        self.classifier_scale = classifier_scale

        # optimizer
        self.optim = AdamW(self.no_decay_params(self.classifier, args.weight_decay),
                           lr=train_lr, betas=adam_betas)
        self.train_lr, self.anneal_lr = train_lr, anneal_lr

        if self.use_ema and self.accelerator.is_main_process:
            self.ema = self.ema_wrapper(model=self.classifier)

        # step counter state
        self.step, self.resume_step = 0, 0

        # prepare model, dataloader, optimizer with accelerator
        self.classifier, self.diffusion_model, self.eps_model, \
            = self.accelerator.prepare(self.classifier, self.diffusion_model, self.eps_model)
        self.train_dataloader, self.valid_dataloader = self.accelerator.prepare(train_dataloader, valid_dataloader)
        self.train_dataloader = cycle(train_dataloader)
        self.valid_dataloader = cycle(valid_dataloader)

    def cond_fn(self, x, t, y=None):
        assert y is not None
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = self.classifier(x_in.float(), t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            # calculate classifier score
            return torch.autograd.grad(selected.sum(), x_in)[0] * self.classifier_scale

    @torch.no_grad()
    def model_fn(self, x, t, y=None):
        assert y is not None
        return self.eps_model(x.float(), t, y if self.class_cond else None)

    def train(self):
        self.accelerator.print("\nStart Training...")

        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.classifier.train()
                self.eps_model.eval()

                if self.anneal_lr:
                    set_annealed_lr(self.optim, self.train_lr, (self.step + self.resume_step) / self.args.lr_iters)

                total_loss = torch.tensor(0., device=device, dtype=torch.float32)
                top1, top5 = AverageMeter('Acc@1', ':6.2f'), AverageMeter('Acc@5', ':6.2f')
                train_accuracy = Accuracy()

                with accelerator.accumulate(self.classifier):  # And perform gradient accumulation
                    batch = next(self.train_dataloader)
                    images, labels = batch["image"], batch["label"]
                    b = images.size(0)

                    # mixed_x, y1, y2, lam = \
                    #     self.mixup.dual_domains_mix(sketch_img, photo_img, photo_label, sketch_label, 1.0)

                    if self.noised:  # noisy dataset
                        t, weights = self.schedule_sampler.sample(b, device)
                        images = self.diffusion_model.q_sample(images, t)
                    else:
                        t = torch.zeros(images.shape[0], dtype=torch.long, device=device)

                    photo_logits = self.classifier(images.float(), timesteps=t)
                    ce_loss = F.cross_entropy(photo_logits, labels, reduction="none")

                    all_photo_logits, all_photo_label = accelerator.gather_for_metrics((photo_logits, labels))
                    acc1, acc5 = train_accuracy(all_photo_logits, all_photo_label)
                    top1.update(acc1[0], b)
                    top5.update(acc5[0], b)

                    # compute loss without autocast
                    ce_loss = ce_loss.mean()
                    total_loss += ce_loss

                    self.accelerator.backward(total_loss)
                    self.optim.step()
                    self.optim.zero_grad()

                    if self.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.classifier.parameters(), self.max_grad_norm)

                    pbar.set_description(
                        f'L_total: {total_loss.item():.3f}, L_ce: {ce_loss.item():.3f} '
                        f'Acc@1: {top1.avg:.3f}, Acc@5: {top5.avg:.3f}'
                    )

                if self.use_ema and accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                # diffusion model evaluation
                if self.step != 0 and self.step % self.vis_steps == 0 and accelerator.is_main_process:
                    model_kwargs = {}
                    with torch.no_grad():
                        milestone = self.step // self.vis_steps

                        sampled_labels = torch.randint(
                            low=0, high=self.num_classes, size=(self.num_samples,), device=device
                        )
                        model_kwargs['y'] = sampled_labels

                        sample_fn = (
                            self.diffusion_model.p_sample_loop
                            if not self.use_ddim else self.diffusion_model.ddim_sample_loop
                        )
                        sample = sample_fn(
                            self.model_fn,
                            (self.num_samples, 3, self.image_size, self.image_size),
                            clip_denoised=True,
                            model_kwargs=model_kwargs,
                            cond_fn=self.cond_fn,
                            progress=True,
                            device=device
                        )

                    for i in range(sample.shape[0]):
                        image_tensor = sample[i].unsqueeze(0)
                        if i == 0:
                            image_tensor_last = image_tensor
                            continue
                        image_tensor_last = torch.cat((image_tensor_last, image_tensor), dim=0)

                    images_tensor = (image_tensor_last + 1) / 2
                    tv_utils.save_image(images_tensor.float(),
                                        str(self.results_path / f'sample-{milestone}.png'),
                                        nrow=int(math.sqrt(self.num_samples)),
                                        padding=0, normalize=False)

                    t, weights = self.schedule_sampler.sample(self.num_samples, device)
                    sampled_logits = self.classifier(image_tensor_last.float(), timesteps=t)
                    save_grid_images_and_labels(image_tensor_last.float(), sampled_logits, sampled_labels,
                                                self.train_set.classes,
                                                str(self.results_path / f'sample-{milestone}.png'),
                                                nrow=int(math.sqrt(self.num_samples)),
                                                normalize=True)

                    if self.step % self.ckpt_steps == 0:
                        milestone = self.step // self.ckpt_steps
                        self.save(milestone, {
                            'step': self.step,
                            'model': self.accelerator.get_state_dict(self.classifier)
                            if not self.use_ema else self.ema.ema_model.state_dict(),
                            'optim': self.optim.state_dict(),
                            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
                        })

                # Distributed evaluation: classifier
                if self.step != 0 and self.step % self.eval_steps == 0:
                    val_accuracy = Accuracy()
                    with torch.no_grad():
                        cur_model = self.ema if self.use_ema else self.classifier
                        cur_model.eval()

                        batch = next(self.valid_dataloader)
                        # batch = {k: v.to(accelerator.device) for k, v in batch.items()}
                        v_input, v_label = batch["image"], batch["label"]
                        t, _ = self.schedule_sampler.sample(v_input.shape[0], device)
                        predictions = cur_model(v_input, t)
                        # Gather all predictions and targets
                        all_predictions, all_targets = accelerator.gather_for_metrics((predictions, v_label))
                        eval_acc1, eval_acc5 = val_accuracy(all_predictions, all_targets)

                        pbar.write(
                            f"Test | Acc@1: {eval_acc1.cpu().numpy().tolist()[0]:.3f} | Acc@5: {eval_acc5.cpu().numpy().tolist()[0]:.3f}"
                        )
                        self.accelerator.print("-" * 50)

                self.accelerator.wait_for_everyone()

                self.step += 1
                pbar.update(1)

        self.close()


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


#################################################################################
#                                ADM Trainer                                    #
#################################################################################

class ClassifierGuidedDPMTrainer(ModelState):
    def __init__(
            self,
            diffusion_model,
            eps_model,
            train_dataloader,
            args,
            *,
            eps_model_path=None,  # fine-tune flag
            train_batch_size=16,
            train_lr=1e-4,
            weight_decay=0,
            adam_betas=(0.9, 0.999),
            train_num_steps=100000,
            max_grad_norm=None,
            schedule_sampler=None,
            ckpt_steps=3000,
            vis_steps=1000,
            num_samples=25,
    ):
        super().__init__(args)

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'

        # banner
        if self.accelerator.is_main_process:
            print(f"-> U-Net dim: {args.num_channels}, num_res_blocks: {args.num_res_blocks}, "
                  f"num-heads: {args.num_heads}, attn-size: {args.attention_ds}, channel_mult: {args.channel_mult}")
            print(f"-> U-Net in_channels: {args.in_channels}, out_channels: {args.out_channels}")
            print(f"-> U-Net Params: {(sum(p.numel() for p in eps_model.parameters()) / 1e6):.3f}M")
            print(f"-> Learning rate: {train_lr}, adam_betas: {adam_betas} ,weight_decay: {weight_decay}")

        self.diffusion_model = diffusion_model
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion_model)
        self.max_grad_norm = max_grad_norm

        self.num_classes = args.num_classes
        self.num_samples = num_samples
        self.ckpt_steps = ckpt_steps
        self.vis_steps = vis_steps

        # if fine-tuning is True, load the model to be fine-tuned
        if eps_model_path is not None:
            ckpt = torch.load(eps_model_path, map_location=self.accelerator.device)
            ckpt["label_emb.weight"] = nn.Embedding(self.num_classes, args.num_channels * 4).weight
            self.eps_model = self.accelerator.unwrap_model(eps_model)
            self.eps_model.load_state_dict(ckpt)
            self.accelerator.print("Start fine-tuning DPM model...")
        else:
            self.eps_model = eps_model

        self.train_batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.image_size = args.image_size

        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

        # optimizer
        self.optim = AdamW(
            self.no_decay_params(self.eps_model, args.weight_decay),
            lr=args.lr, betas=adam_betas
        )

        if self.use_ema and self.accelerator.is_main_process:
            self.ema = self.ema_wrapper(model=self.eps_model)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.eps_model, self.optim, self.train_dataloader = \
            self.accelerator.prepare(self.eps_model, self.optim, train_dataloader)

    def train(self):
        self.accelerator.print("\nStart Training...")

        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                pbar.set_description(f'DPM train: [{self.step}/{self.train_num_steps}]')

                self.eps_model.train()

                total_loss = 0.

                with accelerator.accumulate(self.eps_model):  # And perform gradient accumulation
                    batch = next(self.train_dataloader)
                    input, label = batch['positive_img'], batch['positive_label']

                    model_kwargs = {"y": label}
                    bz = input.size(0)
                    t, weights = self.schedule_sampler.sample(bz, device)

                    terms = self.diffusion_model.training_losses(self.eps_model, x_start=input, t=t,
                                                                 model_kwargs=model_kwargs)

                    with accelerator.autocast():  # And apply automatic mixed-precision
                        loss = terms["loss"] * weights
                        loss = loss.mean()
                        unweighted_loss = terms["loss"].mean()

                    if isinstance(self.schedule_sampler, LossAwareSampler):
                        self.schedule_sampler.update_with_local_losses(
                            t, unweighted_loss.detach()
                        )

                    total_loss += loss.item()

                    self.accelerator.backward(loss)
                    self.optim.step()
                    self.optim.zero_grad()

                    if self.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.eps_model.parameters(), self.max_grad_norm)

                    pbar.set_postfix(loss=total_loss, uw_loss=unweighted_loss.item())

                if self.use_ema and accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                # diffusion model evaluation
                if self.step != 0 and (self.step % self.vis_steps == 0) and accelerator.is_main_process:
                    cur_model = self.ema if self.use_ema else self.eps_model
                    cur_model.eval()

                    with torch.no_grad():
                        milestone = self.step
                        model_kwargs['y'] = torch.randint(
                            low=0, high=self.num_classes, size=(self.num_samples,), device=device
                        )
                        sample_fn = (
                            self.diffusion_model.p_sample_loop
                            if not self.args.use_ddim else self.diffusion_model.ddim_sample_loop
                        )
                        sample = sample_fn(
                            cur_model,
                            shape=(self.num_samples, 3, self.image_size, self.image_size),
                            clip_denoised=True,
                            device=device, progress=True,
                            model_kwargs=model_kwargs
                        )

                    for i in range(sample.shape[0]):
                        image_tensor = sample[i].unsqueeze(0)
                        if i == 0:
                            image_tensor_last = image_tensor
                            continue
                        image_tensor_last = torch.cat((image_tensor_last, image_tensor), dim=0)
                    images_tensor = (image_tensor_last + 1) / 2
                    tv_utils.save_image(images_tensor.float(),
                                        str(self.results_path / f'sample-{milestone}.png'),
                                        nrow=int(math.sqrt(self.num_samples)),
                                        padding=0, normalize=False)

                    if self.step % self.ckpt_steps == 0:
                        milestone = self.step // self.ckpt_steps
                        self.save(milestone, {
                            'step': self.step,
                            'model': self.accelerator.get_state_dict(self.eps_model)
                            if not self.use_ema else self.ema.ema_model.state_dict(),
                            'optim': self.optim.state_dict(),
                            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
                        })

                pbar.write(f'DPM train: [{self.step}/{self.train_num_steps}], loss={total_loss}')

                self.step += 1
                pbar.update(1)

        self.close()
