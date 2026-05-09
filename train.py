import argparse
import os
import time
import random
import math
import numpy as np

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils.dataset import get_loader
from model import myModel
from datetime import datetime
from utils.metrics import Evaluator
from utils.loss_funcs import (
    EdgeAwareLoss,
    SSIMLoss,
    L1_Charbonnier_loss,
    PerceptualLoss,
)
from utils.CIDNet import CIDNet
from utils.distill import (
    DistillManager,
    ClipConfig,
    DinoConfig,
    Siglip2Config,
    DepthConfig,
)



def seed_everything(seed):
    import random, os, numpy as np, torch
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4
        imdff = pred - target
        rmse = ((imdff ** 2).mean(dim=(1, 2, 3)) + 1e-8).sqrt()
        loss = 20 * torch.log10(1 / rmse).mean()
        loss = (50.0 - loss) / 100.0
        return self.loss_weight * loss


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.evaluator = Evaluator()

        # ---- model construction params (for creating temp folded eval model) ----
        self._in_channels = 3

        if not hasattr(args, "feature_channels") or args.feature_channels is None:
            args.feature_channels = 32
        self._feature_channels = int(args.feature_channels)

        if not hasattr(args, "use_white_balance") or args.use_white_balance is None:
            args.use_white_balance = True
        self._use_white_balance = bool(args.use_white_balance)

        self.deep_model = myModel(
            in_channels=self._in_channels,
            feature_channels=self._feature_channels,
            use_white_balance=self._use_white_balance,
        ).to("cuda")

        # ---- model stage meta (used by multi-stage distill plan) ----
        if not hasattr(self.deep_model, "get_stage_distill_meta"):
            raise RuntimeError(
                "Current myModel does not provide get_stage_distill_meta(). "
                "Please use the updated model version first."
            )

        self._stage_names = tuple(self.deep_model.STAGE_NAMES)
        self._stage_meta = self.deep_model.get_stage_distill_meta()

        # ---- stage-wise distill loss weights ----
        if not hasattr(args, "distill_stage_w_enc1") or args.distill_stage_w_enc1 is None:
            args.distill_stage_w_enc1 = 0.0
        if not hasattr(args, "distill_stage_w_enc2") or args.distill_stage_w_enc2 is None:
            args.distill_stage_w_enc2 = 0.2
        if not hasattr(args, "distill_stage_w_bottleneck") or args.distill_stage_w_bottleneck is None:
            args.distill_stage_w_bottleneck = 1.0
        if not hasattr(args, "distill_stage_w_dec1") or args.distill_stage_w_dec1 is None:
            args.distill_stage_w_dec1 = 0.2
        if not hasattr(args, "distill_stage_w_dec2") or args.distill_stage_w_dec2 is None:
            args.distill_stage_w_dec2 = 0.0

        self._distill_stage_weights = {
            "enc1": float(args.distill_stage_w_enc1),
            "enc2": float(args.distill_stage_w_enc2),
            "bottleneck": float(args.distill_stage_w_bottleneck),
            "dec1": float(args.distill_stage_w_dec1),
            "dec2": float(args.distill_stage_w_dec2),
        }

        # ---- HVI net ----
        self.hvi_net = CIDNet().cuda()
        pth = r"utils/CIDNet_weight_LOLv2_bestSSIM.pth"
        self.hvi_net.load_state_dict(torch.load(pth, map_location="cuda"))
        self.hvi_net.eval()

        # ---- save dir ----
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(
            args.save_path, args.model_name, args.dataset, now_str
        )
        os.makedirs(self.model_save_path, exist_ok=True)

        # ---- resume (weights only) ----
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError(f"=> no checkpoint found at '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location="cuda")
            state_dict = self.deep_model.state_dict()
            model_dict = {k: v for k, v in checkpoint.items() if k in state_dict}
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        # -------------------------
        # Distillation setup
        # -------------------------
        if not hasattr(args, "distill_enable"):
            args.distill_enable = 0
        self.distill_enable = bool(int(args.distill_enable))

        if not hasattr(args, "distill_log_every") or args.distill_log_every is None:
            args.distill_log_every = 50
        self.distill_log_every = int(args.distill_log_every)

        # global distill args
        if not hasattr(args, "distill_target") or args.distill_target is None:
            args.distill_target = "mix_gt_out"
        if not hasattr(args, "distill_depth_target") or args.distill_depth_target is None:
            args.distill_depth_target = "gt"
        if not hasattr(args, "distill_mix_ratio") or args.distill_mix_ratio is None:
            args.distill_mix_ratio = 0.5
        if not hasattr(args, "distill_w_clip") or args.distill_w_clip is None:
            args.distill_w_clip = 0.1
        if not hasattr(args, "distill_w_dino") or args.distill_w_dino is None:
            args.distill_w_dino = 0.2
        if not hasattr(args, "distill_w_siglip2") or args.distill_w_siglip2 is None:
            args.distill_w_siglip2 = 0.1
        if not hasattr(args, "distill_w_depth") or args.distill_w_depth is None:
            args.distill_w_depth = 0.0
        if not hasattr(args, "distill_alpha") or args.distill_alpha is None:
            args.distill_alpha = 1.0
        if not hasattr(args, "distill_token_grid") or args.distill_token_grid is None:
            args.distill_token_grid = "14,14"
        if not hasattr(args, "distill_dino_source") or args.distill_dino_source is None:
            args.distill_dino_source = "out"
        if not hasattr(args, "distill_token_loss") or args.distill_token_loss is None:
            args.distill_token_loss = "cos"
        if not hasattr(args, "distill_depth_grad_weight") or args.distill_depth_grad_weight is None:
            args.distill_depth_grad_weight = 0.5

        if not hasattr(args, "clip_model") or args.clip_model is None:
            args.clip_model = "ViT-B-32"
        if not hasattr(args, "clip_pretrained") or args.clip_pretrained is None:
            args.clip_pretrained = "openai"
        if not hasattr(args, "clip_input") or args.clip_input is None:
            args.clip_input = 224

        # ---- DINO / DINOv3 args ----
        if not hasattr(args, "dino_model") or args.dino_model is None:
            args.dino_model = "facebook/dinov3-vits16-pretrain-lvd1689m"
        if not hasattr(args, "dino_input") or args.dino_input is None:
            args.dino_input = 224
        if not hasattr(args, "dino_backend") or args.dino_backend is None:
            args.dino_backend = "hf"
        if not hasattr(args, "dino_remove_prefix_tokens") or args.dino_remove_prefix_tokens is None:
            args.dino_remove_prefix_tokens = 1
        if not hasattr(args, "dino_num_register_tokens") or args.dino_num_register_tokens is None:
            args.dino_num_register_tokens = 0
        if not hasattr(args, "dino_patch_size") or args.dino_patch_size is None:
            args.dino_patch_size = 16
        if not hasattr(args, "dino_strict_token_check") or args.dino_strict_token_check is None:
            args.dino_strict_token_check = 1

        if not hasattr(args, "siglip2_model") or args.siglip2_model is None:
            args.siglip2_model = "google/siglip2-base-patch16-224"
        if not hasattr(args, "siglip2_input") or args.siglip2_input is None:
            args.siglip2_input = 224

        if not hasattr(args, "depth_model") or args.depth_model is None:
            args.depth_model = "depth-anything/Depth-Anything-V2-Small-hf"
        if not hasattr(args, "depth_input") or args.depth_input is None:
            args.depth_input = 518

        # legacy fallback
        if not hasattr(args, "distill_branch_map") or args.distill_branch_map is None:
            args.distill_branch_map = "0:dino,1:clip,2:none,3:none"

        if not hasattr(args, "distill_stage_mask") or args.distill_stage_mask is None:
            args.distill_stage_mask = "00100"

        if not hasattr(args, "distill_branch_map_enc1") or args.distill_branch_map_enc1 is None:
            args.distill_branch_map_enc1 = "0:none,1:none,2:none,3:none"
        if not hasattr(args, "distill_branch_map_enc2") or args.distill_branch_map_enc2 is None:
            args.distill_branch_map_enc2 = "0:none,1:none,2:none,3:none"
        if not hasattr(args, "distill_branch_map_bottleneck") or args.distill_branch_map_bottleneck is None:
            args.distill_branch_map_bottleneck = str(args.distill_branch_map)
        if not hasattr(args, "distill_branch_map_dec1") or args.distill_branch_map_dec1 is None:
            args.distill_branch_map_dec1 = "0:none,1:none,2:none,3:none"
        if not hasattr(args, "distill_branch_map_dec2") or args.distill_branch_map_dec2 is None:
            args.distill_branch_map_dec2 = "0:none,1:none,2:none,3:none"

        self.distill_managers = {}
        self._distill_primed = False

        # multi-stage distill plan
        self._distill_plan = self._build_distill_plan()
        self._active_distill_stages = [
            s for s in self._stage_names if self._distill_plan[s]["enabled"]
        ]

        # model distill request:
        # only request teacher-active branches to reduce unnecessary branch feature export
        self._model_distill_request = {
            s: (
                list(self._distill_plan[s]["active_teacher_branches"])
                if self._distill_plan[s]["enabled"]
                else None
            )
            for s in self._stage_names
        }

        # runtime counters
        self._distill_seen_steps = 0
        self._distill_nonzero_steps = 0
        self._distill_branchouts_none_steps = 0

        self._distill_stage_seen_steps = {s: 0 for s in self._stage_names}
        self._distill_stage_nonzero_steps = {s: 0 for s in self._stage_names}
        self._distill_stage_branchouts_none_steps = {s: 0 for s in self._stage_names}

        if self.distill_enable and len(self._active_distill_stages) > 0:
            tg = str(args.distill_token_grid).strip().split(",")
            token_grid = (int(tg[0]), int(tg[1])) if len(tg) == 2 else (14, 14)

            self._clip_cfg = ClipConfig(
                model_name=str(args.clip_model),
                pretrained=str(args.clip_pretrained),
                input_size=int(args.clip_input),
            )
            self._dino_cfg = DinoConfig(
                model_name=str(args.dino_model),
                input_size=int(args.dino_input),
                backend=str(args.dino_backend),
                prefer_patch_tokens=True,
                remove_prefix_tokens=bool(int(args.dino_remove_prefix_tokens)),
                num_register_tokens=int(args.dino_num_register_tokens),
                patch_size=int(args.dino_patch_size),
                strict_token_check=bool(int(args.dino_strict_token_check)),
            )
            self._siglip2_cfg = Siglip2Config(
                model_name=str(args.siglip2_model),
                input_size=int(args.siglip2_input),
            )
            self._depth_cfg = DepthConfig(
                model_name=str(args.depth_model),
                input_size=int(args.depth_input),
            )
            self._token_grid = token_grid

            for stage_name in self._active_distill_stages:
                plan = self._distill_plan[stage_name]

                use_clip_here = any(v == "clip" for v in plan["branch_map_dict"].values())
                use_dino_here = any(v == "dino" for v in plan["branch_map_dict"].values())
                use_siglip2_here = any(v == "siglip2" for v in plan["branch_map_dict"].values())
                use_depth_here = any(v == "depth" for v in plan["branch_map_dict"].values())

                manager = DistillManager(
                    branches=int(plan["branches"]),
                    branch_map=str(plan["branch_map_str"]),
                    student_channels=int(plan["student_channels"]),
                    clip_cfg=self._clip_cfg if (use_clip_here and float(args.distill_w_clip) > 0) else None,
                    dino_cfg=self._dino_cfg if (use_dino_here and float(args.distill_w_dino) > 0) else None,
                    siglip2_cfg=self._siglip2_cfg if (use_siglip2_here and float(args.distill_w_siglip2) > 0) else None,
                    depth_cfg=self._depth_cfg if (use_depth_here and float(args.distill_w_depth) > 0) else None,
                    weights={
                        "clip": float(args.distill_w_clip),
                        "dino": float(args.distill_w_dino),
                        "siglip2": float(args.distill_w_siglip2),
                        "depth": float(args.distill_w_depth),
                    },
                    target=str(args.distill_target),
                    depth_target=str(args.distill_depth_target),
                    mix_ratio=float(args.distill_mix_ratio),
                    token_grid=token_grid,
                    token_loss=str(args.distill_token_loss),
                    dino_source=str(args.distill_dino_source),
                    depth_grad_weight=float(args.distill_depth_grad_weight),
                    device="cuda",
                ).to("cuda")

                self.distill_managers[stage_name] = manager

            print("[DISTILL] enabled=1")
            print(f"[DISTILL] stage_mask={args.distill_stage_mask}")
            print(f"[DISTILL] active_stages={self._active_distill_stages}")
            print(
                "[DISTILL] stage_loss_weights="
                f"enc1={self._distill_stage_weights['enc1']:.4f}, "
                f"enc2={self._distill_stage_weights['enc2']:.4f}, "
                f"bottleneck={self._distill_stage_weights['bottleneck']:.4f}, "
                f"dec1={self._distill_stage_weights['dec1']:.4f}, "
                f"dec2={self._distill_stage_weights['dec2']:.4f}"
            )
            for s in self._active_distill_stages:
                p = self._distill_plan[s]
                print(
                    f"[DISTILL][{s}] branches={p['branches']} "
                    f"student_channels={p['student_channels']} "
                    f"branch_map={p['branch_map_str']} "
                    f"active_teacher_branches={p['active_teacher_branches']}"
                )

            print(
                f"[DISTILL] target={args.distill_target} "
                f"depth_target={args.distill_depth_target} "
                f"dino_source={str(args.distill_dino_source)}"
                f"token_loss={str(args.distill_token_loss)}"
            )
            print(
                f"[DISTILL] w_clip={float(args.distill_w_clip):.4f} "
                f"w_dino={float(args.distill_w_dino):.4f} "
                f"w_siglip2={float(args.distill_w_siglip2):.4f} "
                f"w_depth={float(args.distill_w_depth):.4f} "
                f"alpha={float(args.distill_alpha):.4f} "
                f"depth_grad_weight={float(args.distill_depth_grad_weight):.4f}"
            )
            print(
                f"[DISTILL] token_grid={token_grid} "
                f"clip=({args.clip_model},{args.clip_pretrained},{args.clip_input}) "
                f"dino=({args.dino_model},{args.dino_backend},{args.dino_input},"
                f"rm_prefix={int(args.dino_remove_prefix_tokens)},"
                f"reg={int(args.dino_num_register_tokens)},"
                f"patch={int(args.dino_patch_size)},"
                f"strict={int(args.dino_strict_token_check)}) "
                f"siglip2=({args.siglip2_model},{args.siglip2_input}) "
                f"depth=({args.depth_model},{args.depth_input})"
            )
            print(f"[DISTILL] log_every={self.distill_log_every}")
        else:
            if self.distill_enable and len(self._active_distill_stages) == 0:
                print("[DISTILL] enabled=1 but no valid active stages after normalization; distill will be skipped.")
            else:
                print("[DISTILL] enabled=0 (set --distill_enable 1 to turn on)")

        # ---- dataset config ----
        if  args.dataset == "EUVP-s":
            args.train_root = ""
            args.val_root = ""
            args.datasize = 256
            args.resize = True
        elif args.dataset == "UIEB":
            args.train_root = ""
            args.val_root = ""

            args.datasize = 256
            args.resize = True
        elif args.dataset == "UFO":
            args.train_root = ""
            args.val_root = ""
            args.datasize = 256
            args.resize = True


        # -------------------------
        # IMPORTANT FIX:
        # Prime distill adapters BEFORE optimizer/scheduler creation.
        # -------------------------
        self._prime_distill_modules_with_dummy_input()

        # ---- optimizer / scheduler ----
        self.optim = self._build_optimizer(lr=args.lr)
        self.scheduler = self._build_scheduler(self.optim)

        # ---- losses ----
        self.vggL = PerceptualLoss()
        self.L1L = L1_Charbonnier_loss()
        self.ssimL = SSIMLoss(device="cuda", window_size=5)
        self.edgeL = EdgeAwareLoss(loss_type="l2", device="cuda")
        self.psnrL = PSNRLoss(reduction="mean", toY=False)

        # ---- schedule points ----
        self._p1, self._p2 = self._parse_stage_points()
        self._dp1, self._dp2 = self._parse_distill_points()

        # ---- runtime states used by validation ----
        self._global_step = 0
        self._total_steps = 1
        self._folded_done = False

        if not hasattr(self.args, "dual_val_lambda") or self.args.dual_val_lambda is None:
            self.args.dual_val_lambda = 0.5

        print(f"[DUAL-VAL] dual_val_lambda={float(self.args.dual_val_lambda):.3f}")
        print(f"[STAGES] stage_points={self.args.stage_points}")
        print(f"[DISTILL] distill_points={self.args.distill_points}")

    # =========================================================
    # Distill plan helpers
    # =========================================================

    def _normalize_stage_mask(self, mask_str: str):
        s = str(mask_str).strip()
        if len(s) != len(self._stage_names):
            raise ValueError(
                f"distill_stage_mask must have length {len(self._stage_names)} over "
                f"{list(self._stage_names)}, but got '{s}'"
            )
        if any(ch not in ("0", "1") for ch in s):
            raise ValueError(
                f"distill_stage_mask must contain only '0' or '1', but got '{s}'"
            )
        return {stage_name: (s[idx] == "1") for idx, stage_name in enumerate(self._stage_names)}

    def _parse_branch_map_str(self, branch_map_str: str):
        """
        Parse "0:dino,1:clip,2:siglip2,3:depth,4:none"
        -> {0:"dino",1:"clip",2:"siglip2",3:"depth",4:"none"}

        Unknown keys / malformed items are ignored.
        """
        out = {}
        if branch_map_str is None:
            return out

        s = str(branch_map_str).strip()
        if len(s) == 0:
            return out

        parts = [p.strip() for p in s.split(",") if len(p.strip()) > 0]
        for part in parts:
            if ":" not in part:
                continue
            k, v = part.split(":", 1)
            k = k.strip()
            v = v.strip().lower()
            if not k.isdigit():
                continue
            if v not in ("clip", "dino", "siglip2", "depth", "none"):
                continue
            out[int(k)] = v
        return out

    def _parse_distill_points(self):
        if not hasattr(self.args, "distill_points") or self.args.distill_points is None:
            raise ValueError("Missing args.distill_points. Please add --distill_points like '0.3,0.55'.")
        s = str(self.args.distill_points).strip()
        a, b = s.split(",")
        p1 = max(0.0, min(1.0, float(a)))
        p2 = max(0.0, min(1.0, float(b)))
        if p2 < p1:
            p2 = p1
        return p1, p2

    def _in_distill_window(self, global_step, total_steps, p1, p2):
        total_steps = max(1, int(total_steps))
        s1 = int(round(p1 * total_steps))
        s2 = int(round(p2 * total_steps))
        s1 = max(0, min(total_steps, s1))
        s2 = max(0, min(total_steps, s2))
        return (global_step >= s1) and (global_step < s2)

    def _branch_map_dict_to_full_str(self, branch_map_dict, num_branches: int):
        """
        Convert normalized dict to fixed full string for DistillManager.
        Always expand to [0, ..., num_branches-1].
        """
        items = []
        for idx in range(int(num_branches)):
            role = branch_map_dict.get(idx, "none")
            items.append(f"{idx}:{role}")
        return ",".join(items)

    def _build_distill_plan(self):
        """
        Build a normalized multi-stage distillation plan from:
            - distill_stage_mask
            - distill_branch_map_enc1 / enc2 / bottleneck / dec1 / dec2
            - model stage metadata

        Supported teacher roles:
            - clip
            - dino
            - siglip2
            - depth
            - none

        Rules:
            - stage mask controls whether a stage is allowed to distill
            - overflow branch ids are silently ignored
            - if a stage has no valid non-none teacher branches after normalization,
              it is treated as disabled
        """
        stage_mask = self._normalize_stage_mask(self.args.distill_stage_mask)

        raw_stage_branch_map = {
            "enc1": str(self.args.distill_branch_map_enc1),
            "enc2": str(self.args.distill_branch_map_enc2),
            "bottleneck": str(self.args.distill_branch_map_bottleneck),
            "dec1": str(self.args.distill_branch_map_dec1),
            "dec2": str(self.args.distill_branch_map_dec2),
        }

        valid_roles = ("clip", "dino", "siglip2", "depth", "none")

        plan = {}
        for stage_name in self._stage_names:
            meta = self._stage_meta[stage_name]
            num_branches = int(meta["branches"])
            student_channels = int(meta["vit_dim"])

            raw_map_dict = self._parse_branch_map_str(raw_stage_branch_map[stage_name])

            normalized_map_dict = {}
            for idx in range(num_branches):
                role = raw_map_dict.get(idx, "none")
                if role not in valid_roles:
                    role = "none"
                normalized_map_dict[idx] = role

            active_teacher_branches = [
                idx for idx, role in normalized_map_dict.items() if role != "none"
            ]

            enabled = bool(stage_mask[stage_name]) and (len(active_teacher_branches) > 0)

            plan[stage_name] = {
                "enabled": enabled,
                "branches": num_branches,
                "student_channels": student_channels,
                "branch_map_dict": normalized_map_dict,
                "branch_map_str": self._branch_map_dict_to_full_str(
                    normalized_map_dict, num_branches
                ),
                "active_teacher_branches": active_teacher_branches,
            }

        return plan

    def _prime_distill_modules_with_dummy_input(self):
        """
        Prime every active stage-specific DistillManager BEFORE optimizer creation.
        """
        if self._distill_primed:
            return

        if (not self.distill_enable) or (len(self._active_distill_stages) == 0):
            self._distill_primed = True
            print("[DISTILL] PRIME skipped (disabled or no active stages).")
            return

        print("[DISTILL] PRIME(start) with dummy input before optimizer creation")

        prev_model_mode = self.deep_model.training
        prev_manager_modes = {k: m.training for k, m in self.distill_managers.items()}

        self.deep_model.eval()
        for m in self.distill_managers.values():
            m.eval()

        dummy_h = int(self.args.datasize)
        dummy_w = int(self.args.datasize)
        dummy_x = torch.rand(
            1, self._in_channels, dummy_h, dummy_w, device="cuda", dtype=torch.float32
        )
        dummy_gt = dummy_x.clone()

        with torch.no_grad():
            out_tmp = self.deep_model(
                dummy_x,
                return_dict=True,
                distill_request=self._model_distill_request,
                branch_collect="last",
            )
            pred_tmp = out_tmp["pred"].clamp(0.0, 1.0)
            branch_outs_tmp_all = out_tmp.get("branch_outs", None)
            branch_qs_tmp_all = out_tmp.get("branch_qs", None)

            if branch_outs_tmp_all is None and branch_qs_tmp_all is None:
                raise RuntimeError(
                    "[DISTILL][PRIME] both branch_outs and branch_qs are None during dummy priming. "
                    "This usually means the model did not return structured stage branch dicts, "
                    "or the model is already folded too early."
                )

            teacher_feature_cache = {}

            for stage_name in self._active_distill_stages:
                manager = self.distill_managers[stage_name]
                bo = None if branch_outs_tmp_all is None else branch_outs_tmp_all.get(stage_name, None)
                bq = None if branch_qs_tmp_all is None else branch_qs_tmp_all.get(stage_name, None)

                dtmp = manager(
                    x_in=dummy_x,
                    x_out=pred_tmp,
                    x_gt=dummy_gt,
                    branch_outs=bo,
                    branch_qs=bq,
                    stage="prime",
                    feature_cache=teacher_feature_cache,
                )

                sd = manager.state_dict()
                adapter_keys = [k for k in sd.keys() if ("adapter" in k or "adapters" in k)]

                print(
                    f"[DISTILL][PRIME][{stage_name}] "
                    f"loss={float(dtmp['loss']):.6f} "
                    f"clip={float(dtmp['clip']):.6f} "
                    f"dino={float(dtmp['dino']):.6f} "
                    f"siglip2={float(dtmp['siglip2']):.6f} "
                    f"depth={float(dtmp['depth']):.6f} "
                    f"adapter_like_keys={len(adapter_keys)}"
                )

        self._distill_primed = True

        if prev_model_mode:
            self.deep_model.train()
        else:
            self.deep_model.eval()

        for k, m in self.distill_managers.items():
            if prev_manager_modes[k]:
                m.train()
            else:
                m.eval()

        print("[DISTILL] PRIME(done) all lazy adapters created before optimizer creation")

    def _prime_distill_once(self, x, global_step, epoch, step_idx, lam, phase_name):
        """
        Compatibility guard.
        """
        if self._distill_primed:
            return

        raise RuntimeError(
            "[DISTILL] _prime_distill_once() was called during training, which means "
            "distill adapters were not initialized before optimizer creation. "
            "This would cause optimizer/scheduler state mismatch. "
            "Please ensure __init__ calls _prime_distill_modules_with_dummy_input() "
            "before _build_optimizer()."
        )

    # =========================================================
    # Optim / schedule helpers
    # =========================================================

    def _build_optimizer(self, lr: float):
        params = list(self.deep_model.parameters())

        if self.distill_enable and len(self.distill_managers) > 0:
            for stage_name in self._active_distill_stages:
                params += list(self.distill_managers[stage_name].parameters())

        opt = optim.AdamW(
            params,
            lr=float(lr),
            weight_decay=self.args.weight_decay,
            betas=(0.9, 0.999),
        )
        return opt

    def _build_scheduler(self, optimizer, last_epoch: int = -1):
        if self.args.scheduler == "cosine":
            eta_min = self.args.cosine_eta_min
            if eta_min is None:
                eta_min = self.args.lr * 1e-4

            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.args.epoch,
                eta_min=eta_min,
                last_epoch=last_epoch,
            )
        elif self.args.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.args.decay_epoch,
                gamma=self.args.decay_rate,
                last_epoch=last_epoch,
            )
        elif self.args.scheduler == "constant":
            return optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 1.0,
                last_epoch=last_epoch,
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.args.scheduler}")

    def _parse_stage_points(self):
        if not hasattr(self.args, "stage_points") or self.args.stage_points is None:
            raise ValueError("Missing args.stage_points. Please add --stage_points like '0.3,0.9'.")
        s = str(self.args.stage_points).strip()
        a, b = s.split(",")
        p1 = max(0.0, min(1.0, float(a)))
        p2 = max(0.0, min(1.0, float(b)))
        if p2 < p1:
            p2 = p1
        return p1, p2

    def _lambda_by_3stage(self, global_step, total_steps, p1, p2):
        total_steps = max(1, int(total_steps))
        s1 = int(round(p1 * total_steps))
        s2 = int(round(p2 * total_steps))
        s1 = max(0, min(total_steps, s1))
        s2 = max(0, min(total_steps, s2))

        if global_step < s1:
            return 0.0, "stage1"
        if global_step >= s2:
            return 1.0, "stage3"

        ramp_len = max(1, s2 - s1)
        lam = float((global_step - s1) / ramp_len)
        return lam, "stage2"

    def _apply_fold_once(self, epoch):
        if hasattr(self.deep_model, "structural_reparameterize_absorb_ln"):
            self.deep_model.structural_reparameterize_absorb_ln()
        if hasattr(self.deep_model, "fold_model"):
            self.deep_model.fold_model(inplace=True)

        cur_lr = float(self.optim.param_groups[0]["lr"])
        self.optim = self._build_optimizer(lr=cur_lr)

        for pg in self.optim.param_groups:
            pg.setdefault("initial_lr", cur_lr)

        self.scheduler = self._build_scheduler(self.optim, last_epoch=epoch - 1)

        print("[FOLD] applied: absorb_ln + fold_model + rebuild optimizer/scheduler")

    def _build_folded_eval_model_from_current(self):
        m = myModel(
            in_channels=self._in_channels,
            feature_channels=self._feature_channels,
            use_white_balance=self._use_white_balance,
        ).to("cuda").eval()

        m.load_state_dict(self.deep_model.state_dict(), strict=True)

        if hasattr(m, "set_join_lambda"):
            m.set_join_lambda(1.0)
        if hasattr(m, "structural_reparameterize_absorb_ln"):
            m.structural_reparameterize_absorb_ln()
        if hasattr(m, "fold_model"):
            m.fold_model(inplace=True)

        m.eval()
        return m

    # =========================================================
    # Training / validation
    # =========================================================

    def training(self):
        best_train_psnr = -1.0
        best_fold_psnr = -1.0

        best_train_round = {}
        best_fold_round = {}

        torch.cuda.empty_cache()

        train_data_loader = get_loader(
            self.args.train_root,
            self.args.train_batch_size,
            self.args.datasize,
            train=True,
            resize=self.args.resize,
            num_workers=self.args.num_workers,
            shuffle=True,
            pin_memory=True,
        )

        p1, p2 = self._p1, self._p2
        dp1, dp2 = self._dp1, self._dp2

        steps_per_epoch = len(train_data_loader)
        total_steps = int(self.args.epoch * steps_per_epoch)

        global_step = 0

        self._total_steps = max(1, total_steps)
        self._global_step = 0
        self._folded_done = False

        print(f"[STAGES] p1={p1:.2f}, p2={p2:.2f}, total_steps={total_steps}")
        print(f"[DISTILL] distill window = [{dp1:.2f}, {dp2:.2f})")
        print("[TRAIN] main model stays in train-time structure for the whole training.")
        print("[TRAIN] folded model is only built as a temporary copy during validation.")

        self.deep_model.train()
        for epoch in range(1, self.args.epoch + 1):
            loop = tqdm(enumerate(train_data_loader), total=len(train_data_loader), leave=False)
            loss_mean = 0.0

            for step_idx, (x, label, _) in loop:
                x = x.to("cuda", non_blocking=True)
                label = label.to("cuda", non_blocking=True)
                label_for_teacher = label.clamp(0.0, 1.0)

                lam, phase_name = self._lambda_by_3stage(global_step, total_steps, p1, p2)

                if hasattr(self.deep_model, "set_join_lambda"):
                    self.deep_model.set_join_lambda(lam)

                use_distill_now = (
                    self.distill_enable
                    and (len(self._active_distill_stages) > 0)
                    and self._in_distill_window(global_step, total_steps, dp1, dp2)
                )

                self.optim.zero_grad(set_to_none=True)

                distill_stage_dict = {}
                distill_loss = x.new_tensor(0.0)

                if use_distill_now:
                    out = self.deep_model(
                        x,
                        return_dict=True,
                        distill_request=self._model_distill_request,
                        branch_collect="last",
                    )
                    pred = out["pred"]
                    branch_outs_all = out.get("branch_outs", None)
                    branch_qs_all = out.get("branch_qs", None)

                    self._distill_seen_steps += 1

                    if branch_outs_all is None and branch_qs_all is None:
                        self._distill_branchouts_none_steps += 1

                    teacher_feature_cache = {}

                    for stage_name in self._active_distill_stages:
                        self._distill_stage_seen_steps[stage_name] += 1

                        stage_branch_outs = None if branch_outs_all is None else branch_outs_all.get(stage_name, None)
                        stage_branch_qs = None if branch_qs_all is None else branch_qs_all.get(stage_name, None)

                        if stage_branch_outs is None and stage_branch_qs is None:
                            self._distill_stage_branchouts_none_steps[stage_name] += 1

                        manager = self.distill_managers[stage_name]
                        d_stage = manager(
                            x_in=x,
                            x_out=pred.clamp(0.0, 1.0),
                            x_gt=label_for_teacher,
                            branch_outs=stage_branch_outs,
                            branch_qs=stage_branch_qs,
                            stage=phase_name,
                            feature_cache=teacher_feature_cache,
                        )
                        distill_stage_dict[stage_name] = d_stage

                        stage_w = float(self._distill_stage_weights.get(stage_name, 1.0))
                        distill_loss = (
                            distill_loss
                            + float(self.args.distill_alpha) * stage_w * d_stage["loss"]
                        )

                        if float(d_stage["loss"].item()) > 1e-12:
                            self._distill_stage_nonzero_steps[stage_name] += 1

                    if float(distill_loss.item()) > 1e-12:
                        self._distill_nonzero_steps += 1

                    if all(
                        (
                            (branch_outs_all is None or branch_outs_all.get(stage_name, None) is None)
                            and
                            (branch_qs_all is None or branch_qs_all.get(stage_name, None) is None)
                        )
                        for stage_name in self._active_distill_stages
                    ):
                        self._distill_branchouts_none_steps += 1

                else:
                    pred = self.deep_model(x)

                pred_clamped = pred.clamp(0.0, 1.0)

                with torch.no_grad():
                    label_hvi = self.hvi_net.trans.HVIT(label)
                    pred_hvi = self.hvi_net.trans.HVIT(pred.clamp(0.0, 1.0))

                hvi_loss = self.L1L(pred_hvi, label_hvi)
                l1_loss = self.L1L(pred, label)
                vgg_loss = self.vggL(pred, label)
                ssim_loss = self.ssimL(pred, label)
                edge_loss = self.edgeL(pred, label)
                psnr_loss = self.psnrL(pred_clamped, label.clamp(0.0, 1.0))

                task_loss = (
                    l1_loss
                    + 0.5 * hvi_loss
                    + 0.1 * ssim_loss
                    + 0.1 * vgg_loss
                    + 0.1 * edge_loss
                    + psnr_loss
                )

                final_loss = task_loss + distill_loss

                loss_mean += float(final_loss.item())
                final_loss.backward()
                self.optim.step()

                if use_distill_now and len(distill_stage_dict) > 0:
                    if (self._distill_seen_steps == 1) or (self._distill_seen_steps % self.distill_log_every == 0):
                        stage_msgs = []
                        for stage_name in self._active_distill_stages:
                            d_stage = distill_stage_dict[stage_name]
                            stage_w = float(self._distill_stage_weights.get(stage_name, 1.0))
                            weighted_stage_loss = float(self.args.distill_alpha) * stage_w * float(d_stage["loss"])
                            stage_none = (
                                ((branch_outs_all is None) or (branch_outs_all.get(stage_name, None) is None))
                                and
                                ((branch_qs_all is None) or (branch_qs_all.get(stage_name, None) is None))
                            )
                            stage_msgs.append(
                                f"{stage_name}(w={stage_w:.3f},"
                                f"loss={float(d_stage['loss']):.6f},"
                                f"weighted={weighted_stage_loss:.6f},"
                                f"clip={float(d_stage['clip']):.6f},"
                                f"dino={float(d_stage['dino']):.6f},"
                                f"siglip2={float(d_stage['siglip2']):.6f},"
                                f"depth={float(d_stage['depth']):.6f},"
                                f"none={stage_none})"
                            )
                        print(
                            f"[DISTILL][RUN] step={global_step} phase={phase_name} "
                            f"lam={lam:.3f} in_window={use_distill_now} "
                            f"sum_loss={float(distill_loss):.6f} | " + " | ".join(stage_msgs)
                        )

                global_step += 1
                self._global_step = global_step
                self._folded_done = False

                loop.set_description(f"[{epoch}/{self.args.epoch}]")
                if use_distill_now and len(distill_stage_dict) > 0:
                    show_stage = self._active_distill_stages[0]
                    show_dict = distill_stage_dict[show_stage]
                    loop.set_postfix(
                        loss=f"{final_loss.item():.4f}",
                        task=f"{task_loss.item():.4f}",
                        dis=f"{float(distill_loss.item()):.4f}",
                        psnrL=f"{float(psnr_loss.item()):.4f}",
                        show=f"{show_stage}",
                        clip=f"{float(show_dict['clip'].item()):.4f}",
                        dino=f"{float(show_dict['dino'].item()):.4f}",
                        siglip2=f"{float(show_dict['siglip2'].item()):.4f}",
                        depth=f"{float(show_dict['depth'].item()):.4f}",
                        lam=f"{lam:.3f}",
                        phase=phase_name,
                    )
                else:
                    loop.set_postfix(
                        loss=f"{final_loss.item():.4f}",
                        lam=f"{lam:.3f}",
                        psnrL=f"{float(psnr_loss.item()):.4f}",
                        distill="off",
                        phase=phase_name,
                    )

            print(
                f"[{epoch}/{self.args.epoch}] avg_loss={loss_mean / len(train_data_loader):.6f} "
                f"lr={self.optim.param_groups[0]['lr']:.6e}"
            )

            if self.distill_enable:
                print(
                    f"[DISTILL][EPOCH] seen_steps={self._distill_seen_steps} "
                    f"nonzero_steps={self._distill_nonzero_steps} "
                    f"branchouts_none_steps={self._distill_branchouts_none_steps} "
                    f"primed={self._distill_primed}"
                )
                for s in self._active_distill_stages:
                    print(
                        f"[DISTILL][EPOCH][{s}] seen_steps={self._distill_stage_seen_steps[s]} "
                        f"nonzero_steps={self._distill_stage_nonzero_steps[s]} "
                        f"branchouts_none_steps={self._distill_stage_branchouts_none_steps[s]}"
                    )

            if epoch % self.args.epoch_val == 0:
                gc.collect()
                torch.cuda.empty_cache()

                self.deep_model.eval()
                ret = self.validation()

                gc.collect()
                torch.cuda.empty_cache()

                with open(os.path.join(self.model_save_path, "records.txt"), "a") as f:
                    if isinstance(ret, tuple) and len(ret) == 4:
                        ssim_t, psnr_t, ssim_f, psnr_f = ret
                        f.write(
                            f"[epoch:{epoch}] DUAL | "
                            f"TrainPSNR:{psnr_t:.3f} FoldPSNR:{psnr_f:.3f} "
                            f"TrainSSIM:{ssim_t:.4f} FoldSSIM:{ssim_f:.4f}\n"
                        )

                        #逐epoch保存
                        # torch.save(
                        #     self.deep_model.state_dict(),
                        #     os.path.join(self.model_save_path, f"model_{epoch}.pth"),
                        # )

                        if psnr_t > best_train_psnr:
                            torch.save(
                                self.deep_model.state_dict(),
                                os.path.join(self.model_save_path, "best_train_model.pth"),
                            )
                            best_train_psnr = psnr_t
                            best_train_round = {
                                "best_epoch": epoch,
                                "best_PSNR": best_train_psnr,
                                "best_SSIM": ssim_t,
                            }
                            f.write(f"## BEST TRAIN ## {best_train_round}\n")

                        if psnr_f > best_fold_psnr:
                            m_fold_best = self._build_folded_eval_model_from_current()
                            torch.save(
                                m_fold_best.state_dict(),
                                os.path.join(self.model_save_path, "best_fold_model.pth"),
                            )
                            best_fold_psnr = psnr_f
                            best_fold_round = {
                                "best_epoch": epoch,
                                "best_PSNR": best_fold_psnr,
                                "best_SSIM": ssim_f,
                            }
                            f.write(f"## BEST FOLD ## {best_fold_round}\n")

                    else:
                        ssim_t, psnr_t = ret
                        f.write(f"[epoch:{epoch}] TRAIN | PSNR:{psnr_t:.3f} SSIM:{ssim_t:.4f}\n")

                        # torch.save(
                        #     self.deep_model.state_dict(),
                        #     os.path.join(self.model_save_path, f"model_{epoch}.pth"),
                        # )

                        if psnr_t > best_train_psnr:
                            torch.save(
                                self.deep_model.state_dict(),
                                os.path.join(self.model_save_path, "best_train_model.pth"),
                            )
                            best_train_psnr = psnr_t
                            best_train_round = {
                                "best_epoch": epoch,
                                "best_PSNR": best_train_psnr,
                                "best_SSIM": ssim_t,
                            }
                            f.write(f"## BEST TRAIN ## {best_train_round}\n")

                self.deep_model.train()

            self.scheduler.step()

        if self.distill_enable:
            if len(self._active_distill_stages) == 0:
                print("[DISTILL][FINAL][WARNING] distill_enable=1 but no valid active stages after normalization.")
            elif self._distill_seen_steps == 0:
                print(
                    "[DISTILL][FINAL][WARNING] distill_enable=1 but distill never ran (seen_steps=0). "
                    "Check distill_points / stage condition."
                )
            elif self._distill_nonzero_steps == 0:
                print(
                    "[DISTILL][FINAL][WARNING] distill ran but summed loss always zero. "
                    "Check branch maps / teacher weights / branch features."
                )
            elif self._distill_branchouts_none_steps > 0:
                print(
                    "[DISTILL][FINAL][WARNING] some distill steps had all-stage branch features None. "
                    "Model may be folded or not returning branches."
                )
            else:
                print("[DISTILL][FINAL] distill appears ACTIVE and EFFECTIVE (nonzero summed loss observed).")

            for s in self._active_distill_stages:
                if self._distill_stage_seen_steps[s] == 0:
                    print(f"[DISTILL][FINAL][{s}][WARNING] stage never ran.")
                elif self._distill_stage_nonzero_steps[s] == 0:
                    print(f"[DISTILL][FINAL][{s}][WARNING] stage ran but loss always zero.")
                elif self._distill_stage_branchouts_none_steps[s] > 0:
                    print(f"[DISTILL][FINAL][{s}][WARNING] some steps had branch features=None.")
                else:
                    print(f"[DISTILL][FINAL][{s}] OK.")

        print("The accuracy of the best train round is ", best_train_round)
        print("The accuracy of the best fold round is ", best_fold_round)

    def validation(self):
        val_data_loader = get_loader(
            self.args.val_root,
            self.args.eval_batch_size,
            self.args.datasize,
            train=False,
            resize=self.args.resize,
            num_workers=1,
            shuffle=False,
            pin_memory=True,
        )
        torch.cuda.empty_cache()

        lam_val, phase_val = self._lambda_by_3stage(
            self._global_step, self._total_steps, self._p1, self._p2
        )

        if hasattr(self.deep_model, "set_join_lambda"):
            self.deep_model.set_join_lambda(float(lam_val))

        dual_on = float(lam_val) > float(self.args.dual_val_lambda)

        evaluator_train = Evaluator()
        evaluator_fold = Evaluator() if dual_on else None
        m_fold = self._build_folded_eval_model_from_current() if dual_on else None

        with torch.inference_mode():
            loop = tqdm(enumerate(val_data_loader), total=len(val_data_loader), leave=False)
            for _, (x, label, _) in loop:
                x = x.to("cuda", non_blocking=True)
                label_np = label.numpy().astype(np.float32).transpose(0, 2, 3, 1)

                pred_t = self.deep_model(x).clamp(0.0, 1.0)
                pred_t_np = pred_t.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                evaluator_train.evaluation(pred_t_np, label_np)

                if dual_on:
                    pred_f = m_fold(x).clamp(0.0, 1.0)
                    pred_f_np = pred_f.detach().cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
                    evaluator_fold.evaluation(pred_f_np, label_np)
                    loop.set_description(
                        f"[Val lam={lam_val:.3f} {phase_val} DUAL thr={self.args.dual_val_lambda:.2f}]"
                    )
                else:
                    loop.set_description(f"[Val lam={lam_val:.3f} {phase_val}]")

        ssim_t, psnr_t = evaluator_train.getMean()
        print("[Validation-Train] SSIM: %.4f, PSNR: %.4f" % (ssim_t, psnr_t))

        if dual_on:
            ssim_f, psnr_f = evaluator_fold.getMean()
            print("[Validation-Fold ] SSIM: %.4f, PSNR: %.4f" % (ssim_f, psnr_f))
            return (ssim_t, psnr_t, ssim_f, psnr_f)

        return (ssim_t, psnr_t)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epoch", type=int, default=1000, help="epoch number")
    parser.add_argument("--epoch_val", type=int, default=1, help="training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--train_batch_size", type=int, default=5)
    parser.add_argument("--eval_batch_size", type=int, default=5)
    parser.add_argument("--decay_rate", type=float, default=0.1, help="decay rate of learning rate")
    parser.add_argument("--decay_epoch", type=int, default=50, help="every n epochs decay learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="constant",
        choices=["cosine", "step", "constant"]
    )
    parser.add_argument(
        "--cosine_eta_min",
        type=float,
        default=1e-5,
        help="minimum learning rate for cosine annealing"
    )

    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--dataset", type=str, default="UIEB", choices=["UIEB", "UFO", "EUVP-s"])
    parser.add_argument("--model_name", type=str, default="WWE-UIE")
    parser.add_argument("--save_path", type=str, default="./output/")

    parser.add_argument("--resume", type=str)

    parser.add_argument("--dual_val_lambda", type=float, default=0.05,
                        help="when lambda > this, validate both train-model and folded-model")
    parser.add_argument("--feature_channels", type=int, default=24,
                        help="base feature width of myModel (e.g., 16/24/32). Must keep bottleneck dim even.")
    parser.add_argument("--use_white_balance", type=int, default=0,
                        help="1: enable GrayWorldRetinex module, 0: disable")
    parser.add_argument("--stage_points", type=str, default="0.5,0.7",
                        help="two ratios like '0.5,0.8' for (fold_start, fold_end)")

    # ---- distill ----
    parser.add_argument("--distill_enable", type=int, default=1,
                        help="1: enable distillation in stage2, 0: disable")
    parser.add_argument("--distill_log_every", type=int, default=134,
                        help="print distill runtime stats every N distill steps")
    parser.add_argument("--distill_points", type=str, default="0.3,0.5",
                        help="two ratios like '0.3,0.55' for (distill_start, distill_end)")

    # new stage-level controls: order is [enc1, enc2, bottleneck, dec1, dec2]
    parser.add_argument("--distill_stage_mask", type=str, default="11111",
                        help="5-char stage mask over [enc1,enc2,bottleneck,dec1,dec2], "
                             "e.g. '00100' means only bottleneck distills")

    parser.add_argument("--distill_branch_map_enc1", type=str, default="0:dino,1:siglip2,2:depth,3:none",
                        help="branch map for enc1, overflow branch ids are ignored automatically")
    parser.add_argument("--distill_branch_map_enc2", type=str, default="0:dino,1:siglip2,2:depth,3:none",
                        help="branch map for enc2, overflow branch ids are ignored automatically")
    parser.add_argument("--distill_branch_map_bottleneck", type=str, default="0:dino,1:siglip2,2:depth,3:none",
                        help="branch map for bottleneck")
    parser.add_argument("--distill_branch_map_dec1", type=str, default="0:dino,1:siglip2,2:depth,3:none",
                        help="branch map for dec1, overflow branch ids are ignored automatically")
    parser.add_argument("--distill_branch_map_dec2", type=str, default="0:dino,1:siglip2,2:depth,3:none",
                        help="branch map for dec2, overflow branch ids are ignored automatically")
    # e.g. 0:dino,1:siglip2,2:depth,3:clip none

    parser.add_argument("--distill_stage_w_enc1", type=float, default=0.05,
                        help="stage weight for enc1 distill loss")
    parser.add_argument("--distill_stage_w_enc2", type=float, default=0.3,
                        help="stage weight for enc2 distill loss")
    parser.add_argument("--distill_stage_w_bottleneck", type=float, default=1.0,
                        help="stage weight for bottleneck distill loss")
    parser.add_argument("--distill_stage_w_dec1", type=float, default=0.3,
                        help="stage weight for dec1 distill loss")
    parser.add_argument("--distill_stage_w_dec2", type=float, default=0.05,
                        help="stage weight for dec2 distill loss")

    parser.add_argument("--distill_target", type=str, default="gt",
                        choices=["in", "out", "gt", "mix", "mix_gt_out", "mix_in_gt"],
                        help="teacher target for clip/dino/siglip2: in / out / gt / mix(in,out) / mix(gt,out) / mix(in,gt)")
    parser.add_argument("--distill_depth_target", type=str, default="gt",
                        choices=["in", "out", "gt", "mix", "mix_gt_out", "mix_in_gt"],
                        help="teacher target for depth teacher")
    parser.add_argument("--distill_mix_ratio", type=float, default=0.5,
                        help="mix ratio for mix targets")
    parser.add_argument("--distill_alpha", type=float, default=0.08,
                        help="overall distill multiplier")

    parser.add_argument("--distill_dino_source", type=str, default="q", choices=["out", "q"],
                        help="DINO student feature source: out=branch output, q=branch query feature")
    parser.add_argument("--distill_token_loss", type=str, default="rel_l1",
                        choices=["cos", "mse", "rel_l1", "rel_mse"],
                        help="DINO token loss. 'cos' is the softest start; relation losses are harder.")
    parser.add_argument("--distill_depth_grad_weight", type=float, default=0.2,
                        help="gradient weight inside dense depth distillation loss")

    parser.add_argument("--distill_w_clip", type=float, default=0.1,
                        help="weight for CLIP distill inside manager")
    parser.add_argument("--distill_w_dino", type=float, default=0.1,
                        help="weight for DINO distill inside manager")
    parser.add_argument("--distill_w_siglip2", type=float, default=0.1,
                        help="weight for SigLIP2 distill inside manager")
    parser.add_argument("--distill_w_depth", type=float, default=0.03,
                        help="weight for Depth teacher distill inside manager")

    parser.add_argument("--distill_token_grid", type=str, default="14,14",
                        help="token grid for student token adapter, e.g. '14,14'")

    parser.add_argument("--clip_model", type=str, default="ViT-B-32",
                        help="open_clip model name")
    parser.add_argument("--clip_pretrained", type=str, default="openai",
                        help="open_clip pretrained tag")
    parser.add_argument("--clip_input", type=int, default=224,
                        help="CLIP teacher input size")

    # ---- DINO / DINOv3 ----
    parser.add_argument("--dino_model", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m",
                        help="DINO teacher model name. Recommended default is HF DINOv3 small.")
    parser.add_argument("--dino_input", type=int, default=224,
                        help="DINO teacher input size")
    parser.add_argument("--dino_backend", type=str, default="hf", choices=["timm", "hf"],
                        help="DINO backend: 'hf' for DINOv3, 'timm' for legacy DINO / DINOv2.")
    parser.add_argument("--dino_remove_prefix_tokens", type=int, default=1, choices=[0, 1],
                        help="1: remove CLS/register prefix tokens before token distill, 0: keep them")
    parser.add_argument("--dino_num_register_tokens", type=int, default=0,
                        help="number of extra register tokens before patch tokens")
    parser.add_argument("--dino_patch_size", type=int, default=16,
                        help="DINO patch size used for token-grid sanity checks")
    parser.add_argument("--dino_strict_token_check", type=int, default=1, choices=[0, 1],
                        help="1: enforce strict DINO token count check, 0: use best-effort fallback")

    parser.add_argument("--siglip2_model", type=str, default="google/siglip2-base-patch16-224",
                        help="SigLIP2 model name from HuggingFace")
    parser.add_argument("--siglip2_input", type=int, default=224,
                        help="SigLIP2 teacher input size")

    parser.add_argument("--depth_model", type=str, default="depth-anything/Depth-Anything-V2-Small-hf",
                        help="Depth teacher model name from HuggingFace")
    parser.add_argument("--depth_input", type=int, default=518,
                        help="Depth teacher input size")

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":

    start = time.time()
    seed_everything(7)

    main()

    end = time.time()
    print("The total training time is:", end - start)
