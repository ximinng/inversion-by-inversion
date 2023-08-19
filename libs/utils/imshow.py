# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import pathlib
from pathlib import Path
from typing import Union, List, Text, BinaryIO

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

__all__ = [
    'show_tensor_image',
    'show_images',
    'simulate_forward_diffusion',
    'save_grid_images_and_labels',
    'save_grid_images_and_captions'
]

reverse_transforms = transforms.Compose([
    # unnormalizing to [0,1]
    transforms.Lambda(lambda t: torch.clamp((t + 1) / 2, min=0.0, max=1.0)),
    # Add 0.5 after unnormalizing to [0, 255]
    transforms.Lambda(lambda t: torch.clamp(t * 255. + 0.5, min=0, max=255)),
    # CHW to HWC
    transforms.Lambda(lambda t: t.permute(1, 2, 0)),
    # to numpy ndarray, dtype int8
    transforms.Lambda(lambda t: t.to('cpu', torch.uint8).numpy()),
    # Converts a numpy ndarray of shape H x W x C to a PIL Image
    transforms.ToPILImage(),
])


def show_tensor_image(image, title="", f_name=None):
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))
    plt.title(title)

    if f_name is not None and Path(f_name).is_file():
        plt.savefig(f_name)
    plt.close()


def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(num_samples / cols + 1, cols, i + 1)
        plt.imshow(img[0])
    plt.close()


def simulate_forward_diffusion(
        image,
        dataloader: torch.utils.data.DataLoader,
        T: int,
        ddpm: torch.nn.Module,
        num_images: int,
):
    """ Simulate forward diffusion
    Args:
        image: add noise to this image
               image = next(iter(dataloader))[0]
        dataloader:
        T:
        ddpm:
        num_images:
    """
    plt.figure(figsize=(15, 15))
    plt.axis('off')

    stepsize = int(T / num_images)

    for idx in range(0, T, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images + 1, (idx / stepsize) + 1)
        image, noise = ddpm.q_sample(image, t)
        show_tensor_image(image)

    plt.savefig(f"forward-step-{stepsize}.png")
    plt.close()


@torch.no_grad()
def save_grid_images_and_labels(
        images: Union[torch.Tensor, List[torch.Tensor]],
        probs: Union[torch.Tensor, List[torch.Tensor]],
        labels: Union[torch.Tensor, List[torch.Tensor]],
        classes: Union[torch.Tensor, List[torch.Tensor]],
        fp: Union[Text, pathlib.Path, BinaryIO],
        nrow: int = 4,
        normalize: bool = True
) -> None:
    """Save a given Tensor into an image file.
    """
    num_images = len(images)
    num_rows, num_cols = get_subplot_shape(num_images, nrow)

    fig = plt.figure(figsize=(25, 20))

    for i in range(num_images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)

        image, true_label, prob = images[i], labels[i], probs[i]

        true_prob = prob[true_label]
        incorrect_prob, incorrect_label = torch.max(prob, dim=0)
        true_class = classes[true_label]

        incorrect_class = classes[incorrect_label]

        if normalize:
            image = reverse_transforms(image)

        ax.imshow(image)
        title = f'true label: {true_class} ({true_prob:.3f})\n ' \
                f'pred label: {incorrect_class} ({incorrect_prob:.3f})'
        ax.set_title(title, fontsize=20)
        ax.axis('off')

    fig.subplots_adjust(hspace=0.3)

    plt.savefig(fp)
    plt.close()


@torch.no_grad()
def save_grid_images_and_captions(
        images: Union[torch.Tensor, List[torch.Tensor]],
        captions: List,
        fp: Union[Text, pathlib.Path, BinaryIO],
        nrow: int = 4,
        normalize: bool = True
) -> None:
    """
    Save a grid of images and their captions into an image file.

    Args:
        images (Union[torch.Tensor, List[torch.Tensor]]): A list of images to display.
        captions (List): A list of captions for each image.
        fp (Union[Text, pathlib.Path, BinaryIO]): The file path to save the image to.
        nrow (int, optional): The number of images to display in each row. Defaults to 4.
        normalize (bool, optional): Whether to normalize the image or not. Defaults to False.
    """
    num_images = len(images)
    num_rows, num_cols = get_subplot_shape(num_images, nrow)

    fig = plt.figure(figsize=(25, 20))

    for i in range(num_images):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)
        image, caption = images[i], captions[i]

        if normalize:
            image = reverse_transforms(image)

        ax.imshow(image)
        title = f'"{caption}"' if num_images > 1 else f'"{captions}"'
        title = insert_newline(title)
        ax.set_title(title, fontsize=20)
        ax.axis('off')

    fig.subplots_adjust(hspace=0.3)

    plt.savefig(fp)
    plt.close()


def get_subplot_shape(num_images, nrow):
    """
    Calculate the number of rows and columns required to display images in a grid.

    Args:
        num_images (int): The total number of images to display.
        nrow (int): The maximum number of images to display in each row.

    Returns:
        Tuple[int, int]: The number of rows and columns required to display images in a grid.
    """
    num_cols = min(num_images, nrow)
    num_rows = (num_images + num_cols - 1) // num_cols
    return num_rows, num_cols


def insert_newline(string, point=9):
    # split by blank
    words = string.split()
    if len(words) <= point:
        return string

    word_chunks = [words[i:i + point] for i in range(0, len(words), point)]
    new_string = "\n".join(" ".join(chunk) for chunk in word_chunks)
    return new_string
