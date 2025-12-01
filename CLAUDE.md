# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

JiT (Just image Transformer) is a PyTorch/GPU re-implementation of pixel-space diffusion models for high-resolution image generation. This is a research codebase implementing the paper "Back to Basics: Let Denoising Generative Models Denoise" (arXiv:2511.13720). The original implementation was in JAX+TPU; this version is in PyTorch+GPU.

**Extended Feature**: This implementation now supports text-to-image generation using Japanese LLMs (e.g., LLM-jp) as text encoders, in addition to the original ImageNet class conditioning. See README_ja.md for Japanese documentation.

## Environment Setup

Create and activate the conda environment:
```bash
conda env create -f environment.yaml
conda activate jit
```

If you encounter `undefined symbol: iJIT_NotifyEvent` when importing torch:
```bash
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

**For text conditioning (Text-to-Image)**, install additional dependencies:
```bash
pip install transformers
```

## Training Commands

Training uses PyTorch distributed training via `torchrun`. All examples below are configured for 8 GPUs.

**JiT-B/16 on ImageNet 256x256:**
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

**JiT-B/32 on ImageNet 512x512:**
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 512 --noise_scale 2.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

**JiT-H/16 on ImageNet 256x256:**
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-H/16 \
--proj_dropout 0.2 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

### Text Conditioning (Text-to-Image)

**JiT-B/16-Text with LLM-jp on custom captions (256x256):**
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16-Text \
--llm_model_name llm-jp/llm-jp-3-3.7b \
--use_text_conditioning \
--freeze_llm \
--max_text_len 77 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${TEXT_DATASET_PATH} --online_eval
```

Text conditioning requires a dataset with this structure:
```
${TEXT_DATASET_PATH}/
  images/
    img_0001.jpg
    img_0002.jpg
    ...
  captions/
    img_0001.txt
    img_0002.txt
    ...
```

## Evaluation

Evaluate a trained checkpoint:
```bash
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--img_size 256 --noise_scale 1.0 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

Evaluation uses a customized `torch-fidelity` library (https://github.com/LTH14/torch-fidelity) to compute FID and Inception Score. Reference statistics are provided in `fid_stats/` for 256x256 and 512x512 resolutions.

## Code Architecture

### Core Components

**main_jit.py** - Entry point and training orchestration
- Handles argument parsing, distributed setup, data loading, and training loop
- Uses PyTorch DistributedDataParallel (DDP) for multi-GPU training
- Manages EMA (Exponential Moving Average) tracking with two decay rates (0.9999 and 0.9996)
- Checkpoints are saved as `checkpoint-last.pth` every `save_last_freq` epochs and `checkpoint-{epoch}.pth` every 100 epochs
- Online evaluation runs every `eval_freq` epochs when `--online_eval` is enabled

**denoiser.py** - Denoiser wrapper implementing the diffusion process
- Wraps the JiT transformer model from model_jit.py
- Implements the forward diffusion process: `z = t * x + (1 - t) * e`
- Velocity prediction: `v = (x - z) / (1 - t)` with clamping via `t_eps`
- Generation uses ODE solvers (Euler or Heun) with classifier-free guidance (CFG)
- CFG is applied only within the interval `[interval_min, interval_max]` of the timestep
- Manages two EMA parameter sets that are updated after each training step

**model_jit.py** - JiT transformer architecture
- Vision Transformer with adaptive layer norm (adaLN) conditioning on timestep and class
- Uses RoPE (Rotary Position Embeddings) for spatial awareness
- Bottleneck patch embedding: Conv2d → Conv2d (reduces dimensions before transformer)
- In-context learning: injects class-conditioned tokens at a specific layer depth (controlled by `in_context_start`)
- Available models: JiT-B/16, JiT-B/32, JiT-L/16, JiT-L/32, JiT-H/16, JiT-H/32
- Attention uses custom scaled dot product with QK normalization (RMSNorm)
- MLP uses SwiGLU activation
- Dropout is applied only to middle layers (between depth//4 and depth*3//4)

**engine_jit.py** - Training and evaluation loops
- `train_one_epoch()`: Training step with bfloat16 mixed precision, per-iteration LR scheduling, and EMA updates
- `evaluate()`: Generates images distributed across GPUs, computes FID/IS, and saves to temporary folder
- Images are normalized to [-1, 1] for training and denormalized back to [0, 255] for saving
- Evaluation switches to EMA parameters (first EMA by default), generates images, then switches back
- Now supports both class-based and text-based conditioning

**text_image_dataset.py** - Text-conditioned image datasets (NEW)
- `TextImageDataset`: Loads images and corresponding text captions from files
- Expects `images/` and `captions/` subdirectories with matching filenames
- Handles tokenization using transformers tokenizers
- `ImageNetWithCaptions`: Optional wrapper that generates simple Japanese captions from ImageNet class indices

### Utility Modules

**util/misc.py** - Distributed training utilities, metric tracking, checkpoint saving
**util/lr_sched.py** - Learning rate scheduling (constant or cosine with warmup)
**util/model_util.py** - RoPE, positional embeddings, RMSNorm implementations
**util/crop.py** - Center crop data augmentation

### Key Architectural Details

**Diffusion Process:**
- Uses velocity prediction formulation rather than noise prediction
- Timestep sampling from logit-normal distribution: `t = sigmoid(N(P_mean, P_std))`
- Default parameters: P_mean=-0.8, P_std=0.8
- Loss is L2 distance between true and predicted velocity: `||v - v_pred||^2`

**In-Context Mechanism:**
- At layer `in_context_start`, prepends `in_context_len` learnable tokens (default 32) to the sequence
- These tokens are initialized with class embeddings plus learnable positional embeddings
- Requires switching RoPE from `feat_rope` to `feat_rope_incontext` after injection
- Tokens are removed before the final layer

**EMA and Checkpointing:**
- Two EMA copies of model parameters are maintained with different decay rates
- Checkpoints include: model weights, optimizer state, both EMA states, epoch number
- Resume from `checkpoint-last.pth` in the resume directory if it exists

**Classifier-Free Guidance:**
- During training, labels are randomly dropped with probability `label_drop_prob` (default 0.1)
- Dropped labels are replaced with the unconditional token (class index = num_classes)
- During generation, both conditional and unconditional predictions are computed
- CFG formula: `v_uncond + cfg_scale * (v_cond - v_uncond)` applied only within timestep interval

**Text Conditioning (NEW):**
- `LLMTextEncoder` in model_jit.py: Uses pretrained LLMs (e.g., LLM-jp) as text encoders
- Architecture: LLM → masked mean pooling → linear projection to JiT hidden size
- LLM weights are frozen by default (only projection layer is trained)
- Text dropout for CFG: replaces text with empty tokens (all zeros) during training
- At inference, generates both conditional (with text) and unconditional (empty text) predictions
- Text models available: JiT-B/16-Text, JiT-B/32-Text, JiT-L/16-Text, JiT-L/32-Text, JiT-H/16-Text, JiT-H/32-Text

## Model Variants

### Class Conditioning (Original)

Each model variant has specific hyperparameters:

| Model     | Depth | Hidden Size | Heads | Bottleneck | In-Context Start | Patch Size | Typical Dropout |
|-----------|-------|-------------|-------|------------|------------------|------------|-----------------|
| JiT-B/16  | 12    | 768         | 12    | 128        | 4                | 16         | 0.0             |
| JiT-B/32  | 12    | 768         | 12    | 128        | 4                | 32         | 0.0             |
| JiT-L/16  | 24    | 1024        | 16    | 128        | 8                | 16         | 0.0             |
| JiT-L/32  | 24    | 1024        | 16    | 128        | 8                | 32         | 0.0             |
| JiT-H/16  | 32    | 1280        | 16    | 256        | 10               | 16         | 0.2             |
| JiT-H/32  | 32    | 1280        | 16    | 256        | 10               | 32         | 0.2             |

### Text Conditioning (NEW)

Text-conditioned variants use the same architecture as class-conditioned models but replace `LabelEmbedder` with `LLMTextEncoder`:

| Model           | Architecture Base | LLM Default          | Frozen LLM | Resolution |
|----------------|-------------------|---------------------|------------|------------|
| JiT-B/16-Text  | JiT-B/16          | llm-jp/llm-jp-3-3.7b | Yes        | 256x256    |
| JiT-B/32-Text  | JiT-B/32          | llm-jp/llm-jp-3-3.7b | Yes        | 512x512    |
| JiT-L/16-Text  | JiT-L/16          | llm-jp/llm-jp-3-3.7b | Yes        | 256x256    |
| JiT-L/32-Text  | JiT-L/32          | llm-jp/llm-jp-3-3.7b | Yes        | 512x512    |
| JiT-H/16-Text  | JiT-H/16          | llm-jp/llm-jp-3-3.7b | Yes        | 256x256    |
| JiT-H/32-Text  | JiT-H/32          | llm-jp/llm-jp-3-3.7b | Yes        | 512x512    |

Note: `noise_scale` is typically 1.0 for 256x256 images and 2.0 for 512x512 images. LLM model can be changed via `--llm_model_name` argument.

## Dataset

### For Class Conditioning (Original)

Requires ImageNet dataset in standard torchvision ImageFolder format:
```
${IMAGENET_PATH}/
  train/
    n01440764/
      n01440764_10026.JPEG
      ...
    ...
```

### For Text Conditioning (NEW)

Requires a custom dataset with images and corresponding text captions:
```
${TEXT_DATASET_PATH}/
  images/
    img_0001.jpg
    img_0002.jpg
    img_0003.jpg
    ...
  captions/
    img_0001.txt  (contains: "青い空と緑の草原の風景写真")
    img_0002.txt  (contains: "白い猫が窓辺で日向ぼっこをしている写真")
    img_0003.txt  (contains: "富士山の夕焼けの景色")
    ...
```

**Important notes:**
- Each `.txt` file should contain exactly one line of text caption
- Filenames (excluding extensions) must match between `images/` and `captions/` directories
- Supported image formats: .jpg, .jpeg, .png
- Text can be in Japanese or English (depending on the LLM used)

### Evaluation

The `prepare_ref.py` script can be used to prepare reference images for FID evaluation.
