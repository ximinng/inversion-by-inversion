## Inversion-by-Inversion: Exemplar-based Sketch-to-Photo Synthesis via Stochastic Differential Equations without Training

[![arXiv](https://img.shields.io/badge/arXiv-2308.07665-b31b1b.svg)](https://arxiv.org/abs/2308.07665)

#### [Project Link](https://ximinng.github.io/inversion-by-inversion-project/)

Our Inversion-by-Inversion method for exemplar-based sketch-to-photo synthesis addresses the challenge of generating
photo-realistic images from mostly white-space sketches. It includes shape-enhancing and full-control inversion, which
generate an uncolored photo for shape control and add color and texture using an appearance-energy function to create
the final RGB photo. Our pipeline works for different exemplars and does not require task-specific training or trainable
hyper-network, making it a versatile solution.
<br>
<br>

![VCT examples](assets/teaser.png?raw=true)

## Setup

To set up the environment, please run

```bash
conda create -n inv-by-inv python=3.10
conda activate inv-by-inv
pip install -r requirements.txt
```

We test our method on both Nvidia RTX3090 and V100 GPU. However, it should work in any GPU with 4G memory (
when `valid_batch_size=1`).

## Dataset

Please download the AFHQ dataset and put them in `dataset/`.

> Download Link: [afhq dataset](https://drive.google.com/file/d/18b0cz38KugVrqFgEe0lZvZgO5ACDdhHZ/view?usp=sharing)

We also provide some demo images in `data/afhq_demo/` for quick start.

## Pretrained Models

To synthesize the image, a pre-trained diffusion model is required.

> Download
> Link: [pretrained models](https://drive.google.com/drive/folders/1zt2YzcJUPTxWNKAm2wX4GBK4BB3gQD0C?usp=sharing)

- In contrast, you need to download the models pretrained on other datasets in the table and put it
  in `./checkpoint/InvSDE/` folder.
- You can manually revise the checkpoint paths and names in `./config/inversion/afhq-cat2dog-ADM.yaml` file.

## Usage

After downloading the dataset, to use the inv-by-inv for **cat-to-dog** tasks, please run

```bash
python run/run_invbyinv.py -c inversion/afhq-cat2dog-ADM.yaml -respath ./workdir/invbyinv/ -vbz 8
```

The `-vbz` indicates `--valid_batch_size`.

**Note: This version includes more detailed content and optimizes image quality, so the sampling time will be longer.**

Specify the data path to run,

```bash
python run/run_invbyinv.py -c inversion/afhq-cat2dog-ADM.yaml \
    -dpath ./data/afhq_demo/cat \ # examplar
    -rdpath ./data/afhq_demo/dog_sketch \ # sketch
    -respath ./workdir/invbyinv/ \ 
    -vbz 8
```

Please put your exemplar image into `-dpath`, and sketch images into `-rdpath`.
The translated images will be saved in `-respath`.

If you need to speed up sampling, dpm-solver can be called as follows,

```bash
python run/run_invbyinv.py -c inversion/afhq-cat2dog-ADM.yaml -respath ./workdir/invbyinv/ -vbz 8 -uds
```

The `-uds` indicates `--use_dpm_solver`.

```bash
python run/run_invbyinv.py -c inversion/afhq-cat2dog-ADM.yaml -respath ./workdir/invbyinv/ -vbz 8 -ts 30000 -final
```

The `-ts` is the total number of samples (eg. 30000) and `-final` indicates that intermediate results are skipped.

To use the inv-by-inv for **wild-to-dog** tasks, please run,

```bash
python run/run_invbyinv.py -c inversion/afhq-wild2dog-ADM.yaml -dpath ./dataset/afhq/train/wild -respath ./workdir/invbyinv/ -vbz 8
```

## Citation

If this code is useful for your work, please cite our paper:

```
@article{xing2023inversion,
  title={Inversion-by-Inversion: Exemplar-based Sketch-to-Photo Synthesis via Stochastic Differential Equations without Training},
  author={Xing, Ximing and Wang, Chuang and Zhou, Haitao and Hu, Zhihao and Li, Chongxuan and Xu, Dong and Yu, Qian},
  journal={arXiv preprint arXiv:2308.07665},
  year={2023}
}
```