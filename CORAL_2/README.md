

<p align="center">
	<img src="./image/coral-final.gif" alt="preview" width="30" />
</p>

<p align="center">
	<strong>CORAL: Correspondence Alignment for Improved Virtual Try On</strong>
</p>

<p align="center">
	<a href="https://arxiv.org/abs/XXXX.XXXXX"><img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red" alt="arXiv" /></a>
	<a href="https://cvlab-kaist.github.io/CORAL"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project-blue" alt="Project" /></a>
	<a href="https://huggingface.co/chimaharicox/coral_vt"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF_ckpts-VITON--HD-orange" alt="HuggingFace VITON-HD" /></a>
	<a href="https://huggingface.co/chimaharicox/coral_dc"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF_ckpts-DressCode-orange" alt="HuggingFace DressCode" /></a>
</p>
<!-- Teaser image (PNG) - preserve original aspect ratio and limit size -->
<div style="width:100%; margin:20px 0; text-align:center;">
	<img src="./image/teaser.png" alt="teaser" style="width:100%; height:auto; display:block; margin:0 auto;" />
</div>

# <h1 style=" margin-bottom:16px;">🎤 Intro</h1>
We introduce CORrespondence ALignment (CORAL), which explicitly enhances person–garment correspondences by improving query–key matching in the full 3D attention of the DiT. For more details and results, please visit our [project page](https://cvlab-kaist.github.io/CORAL)!


<h1 style="margin-top:36px; margin-bottom:12px; text-align:center;">🔥 TODO</h1>
<div style="padding:8px 0; color:#ffffff; text-align:center;">
<ul style="list-style:none; padding:0; margin:0; display:inline-block; text-align:left;">
	<li style="margin:10px 0; font-size:16px;">☑️ Inference Code Release</li>
	<li style="margin:10px 0; font-size:16px;">☑️ Checkpoints for VITON-HD, DressCode Release</li>
	<li style="margin:10px 0; font-size:16px;">⬜ HuggingFace🤗 Demo Release</li>
	<li style="margin:10px 0; font-size:16px;">⬜ Training Code Release</li>
</ul>
</div>

# <h1 style="margin-top:36px; margin-bottom:12px;">🛠️ Installation</h1>

Prepare a conda environment and install required libraries in `requirments.txt`.

```bash
conda create -n coral python=3.10 -y
conda activate coral

# Clone the CORAL repository and install requirements
git clone https://github.com/cvlab-kaist/CORAL.git
cd CORAL
pip install -r requirements.txt

# Install diffusers from source
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .

# Return to CORAL
cd ../
```
# <h1 style="margin-top:36px; margin-bottom:12px;">📁 Data Preparation</h1>

<h2 style="margin-top:28px; margin-bottom:10px;">👚 VITON-HD</h2>

<h3 style="margin-top:20px; margin-bottom:8px;">1. Download VITON-HD</h3>

Download [VITON-HD](https://github.com/shadow2496/VITON-HD) and place it in your dataset directory.

### 2. Preprocessed files

The VITON-HD benchmark already includes the required preprocessed files under `test/` (folders below). If any required folder is missing, inference cannot be run.

Required folder structure:

```
viton-hd/
├── test/
│   ├── agnostic-mask/
│   ├── cloth/
│   ├── image/
│   ├── image-densepose/
│   ├── image-parse-v3/
├── train/
│   ...
```



### 3. Test pair lists

The test set requires two separate pair lists, one for paired setting and one for unpaired setting. The list for unpaired setting is included in the [VITON-HD](https://github.com/shadow2496/VITON-HD) dataset.

Example lines for a paired list:

```text
00006_00.jpg 00006_00.jpg
00008_00.jpg 00008_00.jpg
00013_00.jpg 00013_00.jpg
00017_00.jpg 00017_00.jpg
00034_00.jpg 00034_00.jpg
```

Example lines for an unpaired list:

```text
05006_00.jpg 11001_00.jpg
02532_00.jpg 14096_00.jpg
03921_00.jpg 08015_00.jpg
12419_00.jpg 01944_00.jpg
12562_00.jpg 14025_00.jpg
```

<h2 style="margin-top:28px; margin-bottom:10px;">👚 DressCode</h2>

<h3 style="margin-top:20px; margin-bottom:8px;">1. Download DressCode</h3>

Download the DressCode dataset and place it in your dataset directory.

### 2. Preprocessed files

The DressCode benchmark already includes the required preprocessed files under `test/` (folders below). If any required folder is missing, inference cannot be run.

Required folder structure:

```
DressCode/
├── test/
│   ├── upper_body/
│   │   ├── images/
│   │   ├── keypoints/
│   │   ├── label_maps/
│   │   ├── image-densepose/
│   ├── lower_body/
│   │   ├── images/
│   │   ├── keypoints/
│   │   ├── label_maps/
│   │   ├── image-densepose/
│   ├── dresses/
│   │   ├── images/
│   │   ├── keypoints/
│   │   ├── label_maps/
│   │   ├── image-densepose/
├── train/
│   ...
```

Download the pre-computed densepose images from [here](https://drive.google.com/file/d/1OTRjo0yHsYvn-sevt83VN_-BgjE7fMqQ/view?usp=sharing) and place them in the `image-densepose/` folder. The original source is [here](https://github.com/yisol/IDM-VTON).

<a id="dresscode-test-pair-lists"></a>

### 3. Test pair lists

The test set requires two separate pair lists: one for the paired setting and one for the unpaired setting.

Example lines for a paired list:

```text
048392_0.jpg	048392_1.jpg	0
048393_0.jpg	048393_1.jpg	0
048394_0.jpg	048394_1.jpg	0
048395_0.jpg	048395_1.jpg	0
048396_0.jpg	048396_1.jpg	0
```

Example lines for an unpaired list:

```text
048392_0.jpg	049114_1.jpg	0
048393_0.jpg	048724_1.jpg	0
048408_0.jpg	048433_1.jpg	0
048409_0.jpg	049910_1.jpg	0
048410_0.jpg	048647_1.jpg	0
```


# <h1 style="margin-top:36px; margin-bottom:12px;">🏃 Inference</h1>
We provide separate inference scripts for DressCode and the VITON-HD benchmark.

Running inference requires a GPU with at least 40 GB of VRAM.


## 👚 VITON-HD

In the `CORAL` directory run the provided script:

```bash
cd CORAL
bash inference_vt.sh
```

Sample `inference_vt.sh`:

```bash
CUDA_VISIBLE_DEVICES=0 python ./inference_vton.py \
	--pretrained_model_name_or_path="black-forest-labs/FLUX.1-Fill-dev" \
	--coral_model_path="chimaharicox/coral_vt" \
	--weight_dtype="bf16" \
	--width=768 \
	--height=1024 \
	--seed="42" \
	--dataroot="/path/to/viton-hd"  \
	--dataset_type="vt" \
	--output_dir="/path/to/output_dir" \
	--data_list="/path/to/test_pair_list.txt"
```

Before running, update the following three arguments in `CORAL/inference_vt.sh`:

- `--dataroot`: Set the path to the VITON-HD dataset root you downloaded [here](#1-download-viton-hd).
- `--data_list`: Set the path to the test pair list file to use. Choose either the paired or unpaired list prepared [here](#3-test-pair-lists).
- `--output_dir`: set where inference outputs will be saved in.

## 👚 DressCode
In the `CORAL` directory run the provided script for DressCode:

```bash
cd CORAL
bash inference_dc.sh
```

Sample `inference_dc.sh` (update paths before running):

```bash
CUDA_VISIBLE_DEVICES=1 python ./inference_vton.py \
	--pretrained_model_name_or_path="black-forest-labs/FLUX.1-Fill-dev" \
	--coral_model_path="chimaharicox/coral_dc" \
	--weight_dtype="bf16" \
	--width=768 \
	--height=1024 \
	--seed="42" \
	--dataroot="/path/to/DressCode"  \
	--dataset_type="dc" \
	--output_dir="/path/to/output_dir" \
	--data_list="/path/to/test_pairs_paired_00_2.txt"
```

Before running, update the following three arguments in `CORAL/inference_dc.sh`:

- `--dataroot`: Set the path to the DressCode dataset root you downloaded  [here](#1-download-dresscode).
- `--data_list`: Set the path to the test pair list file to use. Choose either the paired or unpaired list prepared [here](#dresscode-test-pair-lists).
- `--output_dir`: Set where inference outputs will be saved.


