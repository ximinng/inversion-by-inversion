# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing

"""
Description: How to install
# all:
pip install omegaconf tqdm scipy opencv-python einops BeautifulSoup4 timm matplotlib torchmetrics accelerate diffusers triton transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

# CLIP:
pip install git+https://github.com/openai/CLIP.git -i https://pypi.tuna.tsinghua.edu.cn/simple

# torch 1.13.1:
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# xformers (python=3.10):
conda install xformers -c xformers
xFormers - Toolbox to Accelerate Research on Transformers:
https://github.com/facebookresearch/xformers
"""

from setuptools import setup, find_packages

setup(
    name='SketchGuidedGeneration',
    packages=find_packages(),
    version='0.0.13',
    license='MIT',
    description='Sketch Guided Content Generation',
    author='XiMing Xing',
    author_email='ximingxing@gmail.com',
    url='https://github.com/ximinng/SketchGeneration/',
    long_description_content_type='text/markdown',
    keywords=[
        'artificial intelligence',
        'generative models',
        'sketch'
    ],
    install_requires=[
        'omegaconf',  # YAML processor
        'accelerate',  # Hugging Face - pytorch distributed configuration
        'diffusers',  # Hugging Face - diffusion models
        'transformers',  # Hugging Face - transformers
        'einops',
        'pillow',
        'torch>=1.13.1',
        'torchvision',
        'tensorboard',
        'torchmetrics',
        'tqdm',  # progress bar
        'timm',  # computer vision models
        "numpy",  # numpy
        'matplotlib',
        'scikit-learn',
        'omegaconf',  # configs
        'Pillow',  # keep the PIL.Image.Resampling deprecation away,
        'wandb',  # weights & Biases
        'opencv-python',  # cv2
        'BeautifulSoup4'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)
