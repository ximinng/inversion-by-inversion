# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description: EGSDE: Unpaired Image-to-Image Translation via Energy-Guided Stochastic Differential Equations

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torchvision import utils as tv_util
from tqdm import tqdm

from libs.engine import ModelState
from libs.solver.lr_scheduler import get_scheduler
from libs.utils import cycle, exists
from dpm_nn.guided_dpm.loss_resample import UniformSampler
from dpm_nn.inversion.egsde import EnergyGuidedSDE
from sketch_nn.augment.resizer import Resizer


#################################################################################
#                                EGSDE sampling                                 #
#################################################################################

class EnergyGuidedSDEPipeline(ModelState):

    def __init__(self, args, sde_model, sde_path, dse_model, dse_path, dataloader, use_dpm_solver: bool = False):
        super().__init__(args)
        self.args = args
        self.use_dpm_solver = use_dpm_solver

        self.print(f"loading SDE from `{sde_path}` ....")
        self.sde_model = self.load_ckpt_model_only(sde_model, sde_path)
        self.print(f"-> SDE Params: {(sum(p.numel() for p in sde_model.parameters()) / 1e6):.3f}M")

        self.print(f"loading domain-specific extractor from `{dse_path}` ....")
        try:
            self.dse_model = self.load_ckpt_model_only(dse_model, dse_path)
        except Exception as e:
            ckpt = torch.load(dse_path, map_location=self.accelerator.device)
            self.dse_model = self.accelerator.unwrap_model(dse_model)
            self.dse_model.load_state_dict(ckpt['model'])
        self.print(f"-> DSE Params: {(sum(p.numel() for p in sde_model.parameters()) / 1e6):.3f}M")

        self.results_path = self.results_path.joinpath(f"{args.dataset}-{args.task}-sample")
        self.results_path.mkdir(exist_ok=True)

        dpm_cfg = args.diffusion
        self.EGSDE = EnergyGuidedSDE(dpm_cfg.timesteps, dpm_cfg.beta_schedule, dpm_cfg.var_type,
                                     expert_kwargs=self.args.expert)

        self.EGSDE, self.sde_model, self.dse_model = \
            self.accelerator.prepare(self.EGSDE, self.sde_model, self.dse_model)
        self.dataloader = self.accelerator.prepare(dataloader)

    def sample(self):
        accelerator = self.accelerator
        device = self.accelerator.device

        sample = next(iter(self.dataloader))
        batch_size = sample["image"].shape[0]  # online batch_size
        image_size = self.args.image_size

        # load domain-independent feature extractor
        down_N = self.args.expert.down_N
        shape = (batch_size, 3, image_size, image_size)
        shape_d = (
            batch_size, 3, int(image_size / down_N), int(image_size / down_N)
        )
        down = Resizer(shape, 1 / down_N).to(device)
        up = Resizer(shape_d, down_N).to(device)
        die = (down, up)

        experts = {
            "dse": self.dse_model,
            "die": die,
        }
        model_kwargs = {}

        with tqdm(self.dataloader, disable=not accelerator.is_local_main_process) as pbar:
            for i, sample in enumerate(pbar):
                src_input, name = sample["image"], sample["fname"]
                pbar.set_description(f"Sampling[{i}/{len(self.dataloader)}]")

                model_kwargs['step'] = i
                results = self.EGSDE.sampling_progressive(
                    src_input,
                    experts,
                    self.args.repeat_step,
                    self.args.perturb_step,
                    self.sde_model,
                    model_kwargs,
                    device,
                    recorder=pbar,
                    use_dpm_solver=self.use_dpm_solver
                )

                if accelerator.is_main_process:
                    results = accelerator.gather(results)
                    # gather final result
                    for kth in range(len(results)):
                        for b in range(src_input.shape[0]):
                            final, perturb_x0 = results[f"{kth}-th"]
                            img_name = name[b].split(".")[0]  # Remove file suffixes
                            save_path = self.results_path.joinpath(
                                f"i-{i}-{img_name}-B-{b}-K-{kth}-t-{self.args.perturb_step}.png"
                            )
                            # (x0, perturbed_x0, kth_translated_x0)
                            save_grids = torch.cat(
                                (src_input[b].unsqueeze_(0), perturb_x0[b].unsqueeze_(0), final[b].unsqueeze_(0)),
                                dim=0
                            )
                            tv_util.save_image(save_grids, save_path, nrow=save_grids.shape[0])

        self.close()


#################################################################################
#                                DSE Trainer                                    #
#################################################################################

class DSETrainer(ModelState):

    def __init__(
            self,
            diffusion_model,
            classifier,
            train_dataloader,
            args,
            *,
            clf_model_path=None,  # fine-tune flag
            class_cond=True,
            noised=True,
            use_ddim=False,
            schedule_sampler=None,
            train_num_steps=100000,
            train_batch_size=32,
            train_lr=3e-4,
            weight_decay=0.5,
            adam_betas=(0.9, 0.999),
            lr_scheduler: str = "constant",
            lr_warmup_steps: int = 100,
            max_grad_norm=None,
            ckpt_steps=3000,
            num_workers=0
    ):
        super().__init__(args)

        self.print(f"-> Learning rate: {train_lr}, adam_betas: {adam_betas} ,weight_decay: {weight_decay}")
        self.print(f"-> LR Scheduler: {lr_scheduler}, lr_warmup_steps: {lr_warmup_steps}")
        self.print(f"-> use DDIM: {use_ddim}, noised training: {noised}, class_cond: {class_cond}")

        self.results_path = self.results_path.joinpath(f"{args.dataset}-{args.task}-DSE-train-out")
        self.results_path.mkdir(exist_ok=True)
        self.print(f"-> The results will be saved in '{self.results_path}'")

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

        self.num_classes = args.num_classes
        self.ckpt_steps = ckpt_steps

        self.train_batch_size = train_batch_size
        self.train_num_steps = train_num_steps
        self.image_size = args.image_size
        self.max_grad_norm = max_grad_norm

        self.use_ddim = use_ddim
        self.class_cond = class_cond
        self.noised = noised
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion_model)

        # optimizer
        self.optim = AdamW(self.no_decay_params(self.classifier, args.weight_decay),
                           lr=train_lr, betas=adam_betas)
        self.train_lr = train_lr

        # learning rate scheduler
        self.lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=self.optim,
            num_warmup_steps=lr_warmup_steps * args.gradient_accumulate_step,
            num_training_steps=train_num_steps * args.gradient_accumulate_step)

        if self.use_ema and self.accelerator.is_main_process:
            self.ema = self.ema_wrapper(model=self.classifier)

        # step counter state
        self.step, self.resume_step = 0, 0

        self.diffusion_model = diffusion_model
        self.classifier, self.optim, self.lr_scheduler, train_dataloader = self.accelerator.prepare(
            self.classifier, self.optim, self.lr_scheduler, train_dataloader
        )
        self.train_dataloader = cycle(train_dataloader)

    def train(self):
        self.accelerator.print("\nStart Training...")

        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.classifier.train()

                total_loss = torch.tensor(0., device=device, dtype=torch.float32)

                with accelerator.accumulate(self.classifier):  # And perform gradient accumulation
                    batch = next(self.train_dataloader)
                    images, labels = batch["image"], batch["label"]
                    images = 2 * images - 1.0
                    b = images.size(0)

                    if self.noised:  # noisy img
                        t, weights = self.schedule_sampler.sample(b, device)
                        images = self.diffusion_model.q_sample(images, t)
                    else:
                        t = torch.zeros(images.shape[0], dtype=torch.long, device=device)

                    photo_logits = self.classifier(images.float(), timesteps=t)
                    ce_loss = F.cross_entropy(photo_logits, labels, reduction="none")

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(ce_loss.repeat(self.train_batch_size)).mean()
                    total_loss += avg_loss.item() / self.args.gradient_accumulate_step

                    self.accelerator.backward(avg_loss)
                    if self.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(self.classifier.parameters(), self.args.max_grad_norm)
                    self.optim.step()
                    self.lr_scheduler.step()
                    self.optim.zero_grad()

                    pbar.set_description(
                        f"L_total: {total_loss.item():.3f}, "
                        f"init_lr: {self.train_lr}, cur_lr: {self.optim.state_dict()['param_groups'][0]['lr']}"
                    )

                if self.use_ema and accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                # save model
                if self.step % self.ckpt_steps == 0 and self.step != 0 and accelerator.is_main_process:
                    milestone = self.step // self.ckpt_steps
                    self.save(milestone, {
                        'step': self.step,
                        'model': self.accelerator.get_state_dict(self.classifier),
                        'ema': self.ema.ema_model.state_dict() if self.use_ema else None,
                        'optim': self.optim.state_dict(),
                        'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
                    })

                self.step += 1
                pbar.update(1)

        self.close()
