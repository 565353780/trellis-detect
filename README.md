# TRELLIS Detect

## Download

### DINOv2 Model

```bash
https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
```

### TRELLIS Models

```bash
sudo apt install git-lfs
git lfs install

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/microsoft/TRELLIS-image-large
cd TRELLIS-image-large
git lfs pull
```

## Setup

```bash
conda create -n trellis python=3.10
conda activate trellis
./setup.sh
```

## Run

```bash
python demo.py
```

## Enjoy it~
