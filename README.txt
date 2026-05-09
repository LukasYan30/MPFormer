Project README

Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- CUDA-enabled GPU (recommended)

---

Quick Start

Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py
Testing
CUDA_VISIBLE_DEVICES=0 python test.py
Dataset Configuration (Important)

Dataset paths are not passed via command line.
You must manually set them inside the code.

Training (train.py)
if args.dataset == "EUVP-s":
    args.train_root = ""
    args.val_root = ""

elif args.dataset == "UIEB":
    args.train_root = ""
    args.val_root = ""

elif args.dataset == "UFO":
    args.train_root = ""
    args.val_root = ""

Fill in:

args.train_root = "path/to/train"
args.val_root   = "path/to/val"
Testing (test.py)
if args.dataset == "UIEB":
    args.test_root = ""

elif args.dataset == "UFO":
    args.test_root = ""

elif args.dataset == "EUVP-s":
    args.test_root = ""

Fill in:

args.test_root = "path/to/test"
Supported Datasets
UIEB
UFO
EUVP-s

Select via:

--dataset UIEB
Training Details

The training pipeline includes:

Multi-stage training strategy
Optional knowledge distillation
Multiple loss functions:
L1 Loss
SSIM Loss
Perceptual Loss
Edge Loss
PSNR Loss
Distillation (Optional)

Enabled by default:

--distill_enable 1

Supported teachers:

DINOv3
SigLIP v2
Depth Anything v2

Configurable via arguments (e.g., weights, stages, targets).

Testing

Example:

python test.py \
    --resume path/to/checkpoint \
    --ckpt_mode fold
--resume: path to model weights
--ckpt_mode: default is fold (recommended)
