import argparse
import os
import time
import random
import numpy as np

import torch
from tqdm import tqdm

from utils.dataset import get_loader
from model import myModel
from utils.metrics import Evaluator


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataset_config(args):
    if args.dataset == "EUVP-s":
        args.test_root = ""
        args.datasize = 256
        args.resize = True

    elif args.dataset == "UIEB":
        args.test_root = ""

        args.datasize = 256
        args.resize = True

    elif args.dataset == "UFO":
        args.test_root = ""
        args.datasize = 256
        args.resize = True

    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return args


class FoldTester(object):
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.evaluator = Evaluator()


        self.in_channels = 3
        self.feature_channels = int(args.feature_channels)
        self.use_white_balance = bool(int(args.use_white_balance))


        self.model = myModel(
            in_channels=self.in_channels,
            feature_channels=self.feature_channels,
            use_white_balance=self.use_white_balance,
        ).to(self.device)


        self.load_by_mode(args.resume, args.ckpt_mode)

        self.model.eval()

    def _read_checkpoint(self, ckpt_path: str):
        if ckpt_path is None or str(ckpt_path).strip() == "":
            raise ValueError("Please provide a valid checkpoint path for --resume.")

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"[INFO] Loading checkpoint from: {ckpt_path}")

        try:
            checkpoint = torch.load(
                ckpt_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(
                ckpt_path,
                map_location=self.device,
            )

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]

        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Checkpoint should be a state_dict-like dict, but got {type(checkpoint)}"
            )

        cleaned_ckpt = {}
        for k, v in checkpoint.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned_ckpt[nk] = v

        return cleaned_ckpt

    def _print_mismatch(self, missing, unexpected, title=""):
        if title:
            print(title)

        print(f"[WARNING] Missing keys: {len(missing)}")
        print(f"[WARNING] Unexpected keys: {len(unexpected)}")

        if len(missing) > 0:
            print("[WARNING] Missing key examples:")
            for k in missing[:40]:
                print("  ", k)

        if len(unexpected) > 0:
            print("[WARNING] Unexpected key examples:")
            for k in unexpected[:40]:
                print("  ", k)

    def _strict_load_or_raise(self, state_dict, err_prefix: str):
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        if len(missing) > 0 or len(unexpected) > 0:
            self._print_mismatch(missing, unexpected)
            raise RuntimeError(
                f"{err_prefix} does not fully match current model structure. "
                f"Please check model.py version, feature_channels, use_white_balance, and checkpoint source."
            )

    def convert_to_fold_infer(self):
        print("[INFO] Converting model to folded inference structure...")

        if hasattr(self.model, "set_join_lambda"):
            self.model.set_join_lambda(1.0)

        if hasattr(self.model, "structural_reparameterize_absorb_ln"):
            self.model.structural_reparameterize_absorb_ln()

        if hasattr(self.model, "fold_model"):
            self.model.fold_model(inplace=True)

        if hasattr(self.model, "_is_fully_folded"):
            print(f"[INFO] Fully folded: {self.model._is_fully_folded()}")

        print("[INFO] Folded inference model is ready.")

    def load_by_mode(self, ckpt_path: str, ckpt_mode: str):
        ckpt_mode = str(ckpt_mode).strip().lower()
        if ckpt_mode not in ("train", "fold"):
            raise ValueError(f"--ckpt_mode must be 'train' or 'fold', but got: {ckpt_mode}")

        ckpt = self._read_checkpoint(ckpt_path)
        print(f"[INFO] Checkpoint mode: {ckpt_mode}")

        if ckpt_mode == "train":

            self._strict_load_or_raise(
                ckpt,
                err_prefix="Train checkpoint"
            )
            print("[INFO] Train checkpoint loaded successfully into TRAIN structure.")
            self.convert_to_fold_infer()

        elif ckpt_mode == "fold":

            self.convert_to_fold_infer()
            self._strict_load_or_raise(
                ckpt,
                err_prefix="Fold checkpoint"
            )
            print("[INFO] Fold checkpoint loaded successfully into FOLD structure.")

    @torch.no_grad()
    def test(self):
        test_loader = get_loader(
            self.args.test_root,
            self.args.eval_batch_size,
            self.args.datasize,
            train=False,
            resize=self.args.resize,
            num_workers=self.args.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        loop = tqdm(enumerate(test_loader), total=len(test_loader), leave=False)
        for _, (x, label, _) in loop:
            x = x.to(self.device, non_blocking=True)
            label_np = label.numpy().astype(np.float32).transpose(0, 2, 3, 1)

            pred = self.model(x).clamp(0.0, 1.0)
            pred_np = pred.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)

            self.evaluator.evaluation(pred_np, label_np)
            loop.set_description("[Testing Fold/Infer Model]")

        ssim, psnr = self.evaluator.getMean()
        print("=" * 60)
        print(f"[RESULT] Dataset: {self.args.dataset}")
        print(f"[RESULT] Checkpoint mode: {self.args.ckpt_mode}")
        print(f"[RESULT] Folded Model -> SSIM: {ssim:.4f}, PSNR: {psnr:.4f}")
        print("=" * 60)

        return ssim, psnr


def main():
    parser = argparse.ArgumentParser()

    # dataset / loader
    parser.add_argument(
        "--dataset",
        type=str,
        default="UFO",
        choices=["UIEB",  "UFO", "EUVP-s", ],
    )
    parser.add_argument("--eval_batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)


    parser.add_argument(
        "--feature_channels",
        type=int,
        default=24,
        help="must match training setting",
    )
    parser.add_argument(
        "--use_white_balance",
        type=int,
        default=0,
        help="must match training setting",
    )

    # checkpoint args
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to checkpoint weights",
    )
    parser.add_argument(
        "--ckpt_mode",
        type=str,
        default="fold",
        choices=["train", "fold"],
        help="checkpoint structure type: train or fold",
    )

    args = parser.parse_args()
    args = build_dataset_config(args)

    tester = FoldTester(args)
    tester.test()


if __name__ == "__main__":
    start = time.time()
    seed_everything(7)
    main()
    end = time.time()
    print("Total test time:", end - start)