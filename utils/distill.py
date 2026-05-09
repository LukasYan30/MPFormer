# utils/distill.py
# Distillation utilities for branch-wise distillation (CLIP / DINO / SigLIP2)
# in underwater enhancement.
# - Lazy-load teachers only when needed.
# - Shared teacher pool across multiple DistillManager instances.
# - Optional per-step feature cache so the same teacher only runs once per step.
# - Robust to branch_outs containing None entries (for selective branch export).

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Helper: normalization
# -------------------------

def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def cosine_distance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # returns (1 - cosine) averaged over batch (and token if present)
    a = l2norm(a, dim=-1, eps=eps)
    b = l2norm(b, dim=-1, eps=eps)
    # supports shapes: [B,D] or [B,N,D]
    cos = (a * b).sum(dim=-1)
    return (1.0 - cos).mean()


def pairwise_relation_matrix(
    x: torch.Tensor,
    eps: float = 1e-6,
    remove_diagonal: bool = True,
) -> torch.Tensor:
    """
    x: [B, N, D]
    return: [B, N, N]
    Relation matrix based on cosine similarity between tokens.
    """
    if x.ndim != 3:
        raise ValueError(f"pairwise_relation_matrix expects [B,N,D], got {tuple(x.shape)}")

    x = l2norm(x, dim=-1, eps=eps)
    rel = torch.matmul(x, x.transpose(1, 2))  # [B,N,N]

    if remove_diagonal:
        n = rel.shape[1]
        eye = torch.eye(n, device=rel.device, dtype=torch.bool).unsqueeze(0)  # [1,N,N]
        rel = rel.masked_fill(eye, 0.0)

    return rel


def _infer_square_hw_from_token_count(n_tokens: int) -> Optional[Tuple[int, int]]:
    if n_tokens <= 0:
        return None
    s = int(round(float(n_tokens) ** 0.5))
    if s * s == n_tokens:
        return (s, s)
    return None


def _normalize_token_hw(hw: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    if hw is None:
        return None
    if not isinstance(hw, (tuple, list)) or len(hw) != 2:
        return None
    h, w = int(hw[0]), int(hw[1])
    if h <= 0 or w <= 0:
        return None
    return (h, w)


def match_teacher_tokens_to_student(
    student_t: torch.Tensor,
    teacher_t: torch.Tensor,
    token_grid: Tuple[int, int],
    teacher_token_hw: Optional[Tuple[int, int]] = None,
    strict: bool = False,
) -> torch.Tensor:
    """
    Match teacher tokens [B,Nt,D] to student tokens [B,Ns,D] in token count.

    Priority:
      1. use explicit teacher_token_hw if provided and valid
      2. infer square grid from Nt if possible
      3. fallback to truncate / pad
      4. if strict=True and teacher tokens cannot be interpreted as a spatial grid
         when spatial pooling is needed, raise an error instead of silent mismatch
    """
    if teacher_t is None:
        return teacher_t

    if student_t.ndim != 3 or teacher_t.ndim != 3:
        raise ValueError(
            f"Expected student_t and teacher_t to be [B,N,D], got "
            f"{tuple(student_t.shape)} and {tuple(teacher_t.shape)}"
        )

    if student_t.shape[1] == teacher_t.shape[1]:
        return teacher_t

    B, Nt, D = teacher_t.shape
    Ns = student_t.shape[1]
    gh, gw = int(token_grid[0]), int(token_grid[1])

    teacher_token_hw = _normalize_token_hw(teacher_token_hw)
    if teacher_token_hw is not None and (teacher_token_hw[0] * teacher_token_hw[1] != Nt):
        if strict:
            raise RuntimeError(
                f"teacher_token_hw={teacher_token_hw} is inconsistent with teacher token count Nt={Nt}."
            )
        teacher_token_hw = None

    # First try explicit teacher grid, then square inference
    inferred_hw = teacher_token_hw
    if inferred_hw is None:
        inferred_hw = _infer_square_hw_from_token_count(Nt)

    # Spatial remap path
    if inferred_hw is not None and Ns == gh * gw:
        th, tw = inferred_hw
        t2 = teacher_t.view(B, th, tw, D).permute(0, 3, 1, 2).contiguous()  # [B,D,th,tw]
        t2 = F.adaptive_avg_pool2d(t2, (gh, gw))
        teacher_t = t2.permute(0, 2, 3, 1).contiguous().view(B, gh * gw, D)
        return teacher_t

    # If student is spatial tokens but teacher tokens cannot be safely spatialized
    if Ns == gh * gw and inferred_hw is None and strict:
        raise RuntimeError(
            f"Cannot safely align teacher tokens to student tokens: "
            f"teacher Nt={Nt} is not square and no valid teacher_token_hw was provided."
        )

    # Conservative fallback: truncate / pad
    if Nt > Ns:
        teacher_t = teacher_t[:, :Ns, :]
    else:
        pad = Ns - Nt
        teacher_t = torch.cat(
            [teacher_t, teacher_t[:, -1:, :].repeat(1, pad, 1)],
            dim=1,
        )

    return teacher_t

def normalize_depth_map(depth: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Normalize depth map per-image to reduce scale / shift ambiguity.

    depth: [B,1,H,W]
    return: [B,1,H,W]
    """
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"normalize_depth_map expects [B,1,H,W], got {tuple(depth.shape)}")

    mean = depth.mean(dim=(-2, -1), keepdim=True)
    std = depth.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (depth - mean) / (std + eps)


def spatial_gradients_2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [B,C,H,W]
    returns:
      gx: horizontal gradient, [B,C,H,W]
      gy: vertical gradient,   [B,C,H,W]
    """
    if x.ndim != 4:
        raise ValueError(f"spatial_gradients_2d expects [B,C,H,W], got {tuple(x.shape)}")

    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]

    gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    return gx, gy


def parse_branch_map(spec: Union[str, Dict[int, str]], branches: int) -> Dict[int, str]:
    """
    spec can be:
      - dict: {0:"dino",1:"clip",2:"siglip2",3:"depth",4:"none"}
      - str : "0:dino,1:clip,2:siglip2,3:depth,4:none"
    """
    valid_roles = ("clip", "dino", "siglip2", "depth", "none")

    if isinstance(spec, dict):
        out = {int(k): str(v).lower() for k, v in spec.items()}
    else:
        out = {}
        s = str(spec).strip()
        if len(s) == 0:
            out = {}
        else:
            items = [x.strip() for x in s.split(",") if x.strip()]
            for it in items:
                if ":" not in it:
                    continue
                k, v = it.split(":", 1)
                k = k.strip()
                v = v.strip().lower()
                if not k.isdigit():
                    continue
                if v not in valid_roles:
                    continue
                out[int(k)] = v

    # fill missing with "none"
    for b in range(branches):
        out.setdefault(b, "none")
    return out
# -------------------------
# Configs
# -------------------------

@dataclass(frozen=True)
class ClipConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    input_size: int = 224
    amp: bool = False


@dataclass(frozen=True)
class DinoConfig:
    # Recommended DINOv3 small ViT teacher
    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    input_size: int = 224
    amp: bool = False
    prefer_patch_tokens: bool = True

    # backend:
    #   - "auto": use HF DINOv3 path when model_name contains "dinov3", else use timm path
    #   - "hf"  : force HuggingFace/Transformers path
    #   - "timm": force legacy timm path
    backend: str = "auto"

    # DINOv3 ViT models return cls + register + patch tokens.
    # This flag removes prefix tokens and keeps only pure patch tokens.
    remove_prefix_tokens: bool = True

    # Optional manual override for the number of register tokens.
    # If None, code will try config inference; for DINOv3 ViT it falls back to 4.
    num_register_tokens: Optional[int] = None

    # Optional manual override for patch size.
    # If None, code will try to infer it from model config.
    patch_size: Optional[int] = 16

    # When True, token/grid ambiguity raises explicit errors instead of silent fallback.
    strict_token_check: bool = True


@dataclass(frozen=True)
class Siglip2Config:
    model_name: str = "google/siglip2-base-patch16-224"
    input_size: int = 224
    amp: bool = False
    prefer_patch_tokens: bool = False


@dataclass(frozen=True)
class DepthConfig:
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"
    input_size: int = 518
    amp: bool = False

# -------------------------
# Student adapters
# -------------------------

class GlobalAdapter(nn.Module):
    """
    Map feature map [B,C,H,W] -> global vector [B,D].

    Compared with pure GAP+MLP, this version uses:
      - learnable spatial attention pooling
      - auxiliary average pooling
      - fused projection

    This is more suitable for CLIP / SigLIP2 style global semantic distillation:
    it keeps the stability of average pooling, while letting each branch learn
    a task-aware semantic readout.
    """

    def __init__(self, in_channels: int, out_dim: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = int(hidden) if hidden is not None else int(in_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.attn_pre = nn.Conv2d(in_channels, hidden, kernel_size=1, bias=True)
        self.attn_act = nn.GELU()
        self.attn_score = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)

        self.value_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=True)

        self.proj = nn.Sequential(
            nn.Linear(in_channels * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"GlobalAdapter expects [B,C,H,W], got {tuple(x.shape)}")

        B, C, H, W = x.shape

        # stable branch-global summary
        avg_feat = self.avg_pool(x).view(B, C)  # [B,C]

        # learnable semantic readout
        attn_logits = self.attn_score(self.attn_act(self.attn_pre(x)))  # [B,1,H,W]
        attn = torch.softmax(attn_logits.view(B, 1, H * W), dim=-1)     # [B,1,HW]

        value = self.value_proj(x).view(B, C, H * W).transpose(1, 2).contiguous()  # [B,HW,C]
        attn_feat = torch.bmm(attn, value).squeeze(1)  # [B,C]

        fused = torch.cat([avg_feat, attn_feat], dim=-1)  # [B,2C]
        return self.proj(fused)


class TokenAdapter(nn.Module):
    """
    Map feature map [B,C,H,W] -> token sequence [B,N,D]
    by pooling to fixed grid (gh,gw), then flatten to N=gh*gw tokens, then linear to D.
    """
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        grid: Tuple[int, int] = (14, 14),
        hidden: Optional[int] = None,
    ):
        super().__init__()
        self.gh, self.gw = int(grid[0]), int(grid[1])
        hidden = int(hidden) if hidden is not None else int(in_channels)
        self.pool = nn.AdaptiveAvgPool2d((self.gh, self.gw))
        self.proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)  # [B,C,gh,gw]
        B, C, gh, gw = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, gh * gw, C)  # [B,N,C]
        x = self.proj(x)  # [B,N,D]
        return x

class DepthAdapter(nn.Module):
    """
    Map feature map [B,C,H,W] -> dense relative depth map [B,1,H,W].

    Design goal:
      - keep it lightweight
      - keep dense spatial structure
      - suitable for branch-wise geometry distillation
    """

    def __init__(self, in_channels: int, hidden: Optional[int] = None):
        super().__init__()
        hidden = int(hidden) if hidden is not None else int(in_channels)

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"DepthAdapter expects [B,C,H,W], got {tuple(x.shape)}")
        return self.body(x)

# -------------------------
# Teacher wrappers (lazy)
# -------------------------

class TeacherBase(nn.Module):
    """
    forward_features(img) -> dict with possible keys:
      - "global": [B,D] or None
      - "tokens": [B,N,D] or None
      - "depth" : [B,1,H,W] or None

    Optional extra metadata keys:
      - "token_hw": (H_tokens, W_tokens) or None
      - "num_prefix_tokens": int
      - "backend": str
    """
    def __init__(self):
        super().__init__()
        self._initialized = False

    def maybe_init(self, device: str = "cuda"):
        raise NotImplementedError

    @torch.no_grad()
    def forward_features(self, img: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        raise NotImplementedError


class OpenCLIPTeacher(TeacherBase):
    def __init__(self, cfg: ClipConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.model = None
        self.mean = None
        self.std = None

    def maybe_init(self, device: str = "cuda"):
        if self._initialized:
            return

        try:
            import open_clip
        except Exception as e:
            raise ImportError(
                "OpenCLIPTeacher requires open_clip_torch.\n"
                "Install: pip install open_clip_torch\n"
                f"Import error: {repr(e)}"
            )

        model, _, _ = open_clip.create_model_and_transforms(
            self.cfg.model_name, pretrained=self.cfg.pretrained
        )
        model.eval()
        model.to(device)
        model.requires_grad_(False)

        try:
            cfg_dict = open_clip.get_model_config(self.cfg.model_name)
            mean = cfg_dict.get("mean", None)
            std = cfg_dict.get("std", None)
        except Exception:
            mean, std = None, None

        if mean is None or std is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)

        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.device = device
        self._initialized = True

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 3:
            raise ValueError(f"CLIP teacher expects 3-channel RGB input, got shape {tuple(img.shape)}")
        x = F.interpolate(
            img,
            size=(self.cfg.input_size, self.cfg.input_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def forward_features(self, img: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        self.maybe_init(self.device)
        x = self._prep(img)
        if self.cfg.amp and x.is_cuda:
            with torch.cuda.amp.autocast():
                g = self.model.encode_image(x)
        else:
            g = self.model.encode_image(x)
        return {"global": g, "tokens": None}


class LegacyTimmDinoTeacher(TeacherBase):
    """
    Backward-compatible DINO teacher for legacy timm models.
    """

    def __init__(self, cfg: DinoConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.model = None
        self.mean = None
        self.std = None

    def maybe_init(self, device: str = "cuda"):
        if self._initialized:
            return

        try:
            import timm
        except Exception as e:
            raise ImportError(
                "LegacyTimmDinoTeacher requires timm.\n"
                "Install: pip install timm\n"
                f"Import error: {repr(e)}"
            )

        model = timm.create_model(self.cfg.model_name, pretrained=True)
        model.eval()
        model.to(device)
        model.requires_grad_(False)

        dc = getattr(model, "default_cfg", {}) or {}
        mean = dc.get("mean", (0.485, 0.456, 0.406))
        std = dc.get("std", (0.229, 0.224, 0.225))

        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.device = device
        self._initialized = True

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 3:
            raise ValueError(f"DINO teacher expects 3-channel RGB input, got shape {tuple(img.shape)}")
        x = F.interpolate(
            img,
            size=(self.cfg.input_size, self.cfg.input_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        return x

    def _infer_token_hw(self, tokens: Optional[torch.Tensor]) -> Optional[Tuple[int, int]]:
        if tokens is None:
            return None
        if tokens.ndim != 3:
            return None
        return _infer_square_hw_from_token_count(int(tokens.shape[1]))

    @torch.no_grad()
    def forward_features(self, img: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        self.maybe_init(self.device)
        x = self._prep(img)

        if self.cfg.amp and x.is_cuda:
            with torch.cuda.amp.autocast():
                feat = self.model.forward_features(x)
        else:
            feat = self.model.forward_features(x)

        global_vec = None
        tokens = None

        if isinstance(feat, dict):
            if self.cfg.prefer_patch_tokens:
                for k in ["x_norm_patchtokens", "patch_tokens", "x_patchtokens", "tokens"]:
                    if k in feat and torch.is_tensor(feat[k]):
                        tokens = feat[k]
                        break
            for k in ["x_norm_clstoken", "cls_token", "x_clstoken", "global"]:
                if k in feat and torch.is_tensor(feat[k]):
                    global_vec = feat[k]
                    break

        elif torch.is_tensor(feat):
            if feat.ndim == 2:
                global_vec = feat
            elif feat.ndim == 3:
                global_vec = feat[:, 0, :]
                tokens = feat[:, 1:, :]
        else:
            raise RuntimeError(f"Unsupported forward_features output type: {type(feat)}")

        token_hw = self._infer_token_hw(tokens)

        return {
            "global": global_vec,
            "tokens": tokens,
            "depth": None,
            "token_hw": token_hw,
            "num_prefix_tokens": 0,
            "backend": "timm",
        }


class HFDinoV3Teacher(TeacherBase):
    """
    HuggingFace DINOv3 teacher.

    Design choices:
      - use AutoModel.from_pretrained(...)
      - use AutoImageProcessor only for normalization metadata
      - return pure patch tokens when prefer_patch_tokens=True
      - robustly remove prefix tokens (cls + register tokens) by candidate search
      - support non-square token grids when necessary
    """

    def __init__(self, cfg: DinoConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.model = None
        self.processor = None
        self.mean = None
        self.std = None

        self.patch_size = None
        self.num_register_tokens = None

    def maybe_init(self, device: str = "cuda"):
        if self._initialized:
            return

        try:
            from transformers import AutoModel, AutoImageProcessor
        except Exception as e:
            raise ImportError(
                "HFDinoV3Teacher requires transformers with DINOv3 support.\n"
                "Install or upgrade: pip install -U transformers\n"
                f"Import error: {repr(e)}"
            )

        try:
            processor = AutoImageProcessor.from_pretrained(self.cfg.model_name)
            model = AutoModel.from_pretrained(self.cfg.model_name)
        except Exception as e:
            raise RuntimeError(
                "Failed to load DINO/DINOv3 teacher from HuggingFace. "
                "Please check model_name, transformers version, and model access/license."
            ) from e

        model.eval()
        model.to(device)
        model.requires_grad_(False)

        mean = getattr(processor, "image_mean", None)
        std = getattr(processor, "image_std", None)

        if mean is None or std is None:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        patch_size = self.cfg.patch_size
        if patch_size is None:
            patch_size = getattr(getattr(model, "config", None), "patch_size", None)
        if patch_size is not None:
            patch_size = int(patch_size)

        num_register_tokens = self.cfg.num_register_tokens
        if num_register_tokens is None:
            num_register_tokens = getattr(getattr(model, "config", None), "num_register_tokens", None)
        if num_register_tokens is not None:
            num_register_tokens = int(num_register_tokens)

        self.processor = processor
        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.device = device
        self._initialized = True

    def _effective_input_size(self) -> int:
        size = int(self.cfg.input_size)
        if self.patch_size is None:
            return size
        if size < self.patch_size:
            raise ValueError(
                f"DINOv3 input_size={size} is smaller than patch_size={self.patch_size}."
            )
        size = (size // self.patch_size) * self.patch_size
        if size <= 0:
            raise ValueError(
                f"Invalid effective DINOv3 input size derived from input_size={self.cfg.input_size} "
                f"and patch_size={self.patch_size}."
            )
        return size

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 3:
            raise ValueError(f"DINOv3 teacher expects 3-channel RGB input, got shape {tuple(img.shape)}")

        size = self._effective_input_size()
        x = F.interpolate(
            img,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        return x

    def _infer_num_register_tokens(self) -> int:
        if self.cfg.num_register_tokens is not None:
            return int(self.cfg.num_register_tokens)
        if self.num_register_tokens is not None:
            return int(self.num_register_tokens)

        # Conservative fallback for ViT-based DINOv3 models
        name = str(self.cfg.model_name).lower()
        if "dinov3" in name:
            return 4
        return 0

    def _extract_sequence_tensor(self, out) -> Optional[torch.Tensor]:
        if out is None:
            return None

        if torch.is_tensor(out):
            return out

        if hasattr(out, "last_hidden_state"):
            val = getattr(out, "last_hidden_state")
            if torch.is_tensor(val):
                return val

        if isinstance(out, dict):
            val = out.get("last_hidden_state", None)
            if torch.is_tensor(val):
                return val

        return None

    def _extract_global_tensor(self, out, seq: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if out is not None:
            if hasattr(out, "pooler_output"):
                val = getattr(out, "pooler_output")
                if torch.is_tensor(val):
                    return val
            if isinstance(out, dict):
                val = out.get("pooler_output", None)
                if torch.is_tensor(val):
                    return val

        if seq is None or not torch.is_tensor(seq):
            return None

        if seq.ndim == 3:
            return seq[:, 0, :]
        if seq.ndim == 4:
            return seq.mean(dim=(-2, -1))

        return None

    def _factor_hw_from_num_tokens(
        self,
        n_tokens: int,
        target_hw: Optional[Tuple[int, int]] = None,
    ) -> Optional[Tuple[int, int]]:
        """
        Factor n_tokens into (h, w). Prefer shapes close to target_hw if provided.
        """
        if n_tokens <= 0:
            return None

        pairs = []
        s = int(n_tokens ** 0.5)
        for h in range(1, s + 1):
            if n_tokens % h == 0:
                w = n_tokens // h
                pairs.append((h, w))
                if h != w:
                    pairs.append((w, h))

        if len(pairs) == 0:
            return None

        if target_hw is None:
            pairs.sort(key=lambda x: abs(x[0] - x[1]))
            return pairs[0]

        th, tw = int(target_hw[0]), int(target_hw[1])

        def score(hw):
            h, w = hw
            return abs(h - th) + abs(w - tw)

        pairs.sort(key=score)
        return pairs[0]

    def _extract_patch_tokens(
        self,
        seq: Optional[torch.Tensor],
        input_hw: Tuple[int, int],
    ) -> Tuple[Optional[torch.Tensor], Optional[Tuple[int, int]], int]:
        if seq is None or (not torch.is_tensor(seq)):
            return None, None, 0

        if seq.ndim == 4:
            # Conv-like spatial map fallback: flatten as tokens
            b, c, h, w = seq.shape
            tokens = seq.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
            return tokens, (h, w), 0

        if seq.ndim != 3:
            return None, None, 0

        total_tokens = int(seq.shape[1])

        target_hw = None
        if self.patch_size is not None:
            gh = int(input_hw[0]) // int(self.patch_size)
            gw = int(input_hw[1]) // int(self.patch_size)
            if gh > 0 and gw > 0:
                target_hw = (gh, gw)

        # Candidate prefix counts:
        # 1) configured/inferred cls + register
        # 2) cls only
        # 3) 0 prefix
        # 4) small brute-force fallback
        candidate_prefixes = []

        inferred_regs = self._infer_num_register_tokens()
        candidate_prefixes.extend([
            1 + inferred_regs,
            1,
            0,
        ])

        for p in range(0, min(9, total_tokens)):
            candidate_prefixes.append(int(p))

        # remove duplicates while keeping order
        seen = set()
        uniq_prefixes = []
        for p in candidate_prefixes:
            if p not in seen:
                seen.add(p)
                uniq_prefixes.append(p)

        best_prefix = None
        best_hw = None
        best_score = None

        for prefix in uniq_prefixes:
            remain = total_tokens - int(prefix)
            if remain <= 0:
                continue

            hw = self._factor_hw_from_num_tokens(remain, target_hw=target_hw)
            if hw is None:
                continue

            h, w = hw
            score = 0

            if target_hw is not None:
                th, tw = target_hw
                score += abs(h - th) + abs(w - tw)

                # strongly prefer exact expected patch grid
                if (h, w) == (th, tw):
                    score -= 1000

            # prefer smaller prefix under same grid quality
            score += 0.01 * prefix

            if (best_score is None) or (score < best_score):
                best_score = score
                best_prefix = int(prefix)
                best_hw = (int(h), int(w))

        if best_prefix is None or best_hw is None:
            if self.cfg.strict_token_check:
                raise RuntimeError(
                    f"DINOv3 patch token extraction failed for total_tokens={total_tokens}."
                )
            return seq, None, 0

        h, w = best_hw
        remain = h * w
        tokens = seq[:, best_prefix:best_prefix + remain, :]

        if int(tokens.shape[1]) != remain:
            if self.cfg.strict_token_check:
                raise RuntimeError(
                    f"DINOv3 token slicing mismatch: expected {remain} patch tokens, "
                    f"got {int(tokens.shape[1])}."
                )
            return seq, None, 0

        return tokens, (h, w), int(best_prefix)

    @torch.no_grad()
    def forward_features(self, img: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        self.maybe_init(self.device)
        x = self._prep(img)

        if self.cfg.amp and x.is_cuda:
            with torch.cuda.amp.autocast():
                out = self.model(
                    pixel_values=x,
                    output_hidden_states=False,
                    return_dict=True,
                )
        else:
            out = self.model(
                pixel_values=x,
                output_hidden_states=False,
                return_dict=True,
            )

        seq = self._extract_sequence_tensor(out)
        global_vec = self._extract_global_tensor(out, seq)

        tokens = None
        token_hw = None
        num_prefix_tokens = 0
        if self.cfg.prefer_patch_tokens:
            tokens, token_hw, num_prefix_tokens = self._extract_patch_tokens(
                seq=seq,
                input_hw=(int(x.shape[-2]), int(x.shape[-1])),
            )

        return {
            "global": global_vec,
            "tokens": tokens,
            "depth": None,
            "token_hw": token_hw,
            "num_prefix_tokens": int(num_prefix_tokens),
            "backend": "hf",
        }


class Siglip2Teacher(TeacherBase):
    """
    SigLIP2 teacher.

    Correct usage:
      - global embedding: prefer get_image_features(pixel_values=...)
      - token features  : use vision_model(...) if available

    This wrapper normalizes different HuggingFace return formats into:
      {
          "global": Tensor[B, D] or None,
          "tokens": Tensor[B, N, D] or None,
      }
    """

    def __init__(self, cfg: Siglip2Config, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.model = None
        self.mean = None
        self.std = None

    def maybe_init(self, device: str = "cuda"):
        if self._initialized:
            return

        try:
            from transformers import AutoModel
        except Exception as e:
            raise ImportError(
                "Siglip2Teacher requires transformers.\n"
                "Install: pip install transformers\n"
                f"Import error: {repr(e)}"
            )

        model = AutoModel.from_pretrained(self.cfg.model_name)
        model.eval()
        model.to(device)
        model.requires_grad_(False)

        # SigLIP/SigLIP2 commonly use [-1, 1] style normalization
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.device = device
        self._initialized = True

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 3:
            raise ValueError(
                f"SigLIP2 teacher expects 3-channel RGB input, got shape {tuple(img.shape)}"
            )
        x = F.interpolate(
            img,
            size=(self.cfg.input_size, self.cfg.input_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        return x

    def _extract_global_tensor(self, out) -> Optional[torch.Tensor]:
        """
        Normalize possibly different HF outputs into a Tensor[B, D].
        """
        if out is None:
            return None

        if torch.is_tensor(out):
            if out.ndim == 2:
                return out
            if out.ndim == 3:
                return out.mean(dim=1)
            raise RuntimeError(f"Unsupported tensor shape for global feature: {tuple(out.shape)}")

        # HF object fields
        for key in ["image_embeds", "pooler_output", "last_hidden_state"]:
            if hasattr(out, key):
                val = getattr(out, key)
                if val is None:
                    continue
                if torch.is_tensor(val):
                    if val.ndim == 2:
                        return val
                    if val.ndim == 3:
                        return val.mean(dim=1)

        # dict-like fallback
        if isinstance(out, dict):
            for key in ["image_embeds", "pooler_output", "last_hidden_state"]:
                if key in out and out[key] is not None:
                    val = out[key]
                    if torch.is_tensor(val):
                        if val.ndim == 2:
                            return val
                        if val.ndim == 3:
                            return val.mean(dim=1)

        return None

    def _extract_tokens_tensor(self, out) -> Optional[torch.Tensor]:
        """
        Best-effort extraction of Tensor[B, N, D].
        """
        if out is None:
            return None

        if torch.is_tensor(out):
            if out.ndim == 3:
                return out
            return None

        if hasattr(out, "last_hidden_state"):
            val = getattr(out, "last_hidden_state")
            if torch.is_tensor(val) and val.ndim == 3:
                return val

        if isinstance(out, dict):
            val = out.get("last_hidden_state", None)
            if torch.is_tensor(val) and val.ndim == 3:
                return val

        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            hs = out.hidden_states[-1]
            if torch.is_tensor(hs) and hs.ndim == 3:
                return hs

        return None

    @torch.no_grad()
    def forward_features(self, img: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        self.maybe_init(self.device)
        x = self._prep(img)

        # -------------------------
        # 1) global image embedding
        # -------------------------
        global_vec = None

        if not hasattr(self.model, "get_image_features"):
            raise RuntimeError(
                "Loaded SigLIP2 model does not provide get_image_features(). "
                "Please check transformers version / model name."
            )

        if self.cfg.amp and x.is_cuda:
            with torch.cuda.amp.autocast():
                global_out = self.model.get_image_features(pixel_values=x)
        else:
            global_out = self.model.get_image_features(pixel_values=x)

        global_vec = self._extract_global_tensor(global_out)

        # -------------------------
        # 2) optional token features
        # -------------------------
        tokens = None
        if hasattr(self.model, "vision_model") and self.cfg.prefer_patch_tokens:
            try:
                if self.cfg.amp and x.is_cuda:
                    with torch.cuda.amp.autocast():
                        vision_out = self.model.vision_model(
                            pixel_values=x,
                            output_hidden_states=True,
                            return_dict=True,
                        )
                else:
                    vision_out = self.model.vision_model(
                        pixel_values=x,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                tokens = self._extract_tokens_tensor(vision_out)

            except Exception:
                tokens = None

        if global_vec is None:
            # fallback: try recovering global from vision tower output
            if hasattr(self.model, "vision_model"):
                try:
                    if self.cfg.amp and x.is_cuda:
                        with torch.cuda.amp.autocast():
                            vision_out = self.model.vision_model(
                                pixel_values=x,
                                output_hidden_states=True,
                                return_dict=True,
                            )
                    else:
                        vision_out = self.model.vision_model(
                            pixel_values=x,
                            output_hidden_states=True,
                            return_dict=True,
                        )

                    if tokens is None:
                        tokens = self._extract_tokens_tensor(vision_out)

                    global_vec = self._extract_global_tensor(vision_out)

                except Exception:
                    pass

        if global_vec is None:
            raise RuntimeError(
                "SigLIP2 forward output format is unsupported for global feature extraction."
            )

        return {"global": global_vec, "tokens": tokens}

class DepthAnythingTeacher(TeacherBase):
    """
    Dense depth teacher based on HuggingFace depth-estimation models.

    Recommended model:
        depth-anything/Depth-Anything-V2-Small-hf

    Returns:
        {
            "global": None,
            "tokens": None,
            "depth": Tensor[B,1,H,W],
        }
    """

    def __init__(self, cfg: DepthConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.model = None
        self.processor = None
        self.mean = None
        self.std = None

    def maybe_init(self, device: str = "cuda"):
        if self._initialized:
            return

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except Exception as e:
            raise ImportError(
                "DepthAnythingTeacher requires transformers with depth-estimation support.\n"
                "Install: pip install transformers\n"
                f"Import error: {repr(e)}"
            )

        processor = AutoImageProcessor.from_pretrained(self.cfg.model_name)
        model = AutoModelForDepthEstimation.from_pretrained(self.cfg.model_name)

        model.eval()
        model.to(device)
        model.requires_grad_(False)

        mean = getattr(processor, "image_mean", None)
        std = getattr(processor, "image_std", None)

        if mean is None or std is None:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

        self.processor = processor
        self.model = model
        self.mean = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.device = device
        self._initialized = True

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[1] != 3:
            raise ValueError(
                f"Depth teacher expects 3-channel RGB input, got shape {tuple(img.shape)}"
            )

        x = F.interpolate(
            img,
            size=(self.cfg.input_size, self.cfg.input_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        return x

    def _extract_depth_tensor(self, out) -> Optional[torch.Tensor]:
        """
        Normalize model outputs into Tensor[B,1,H,W].
        """
        if out is None:
            return None

        val = None

        if hasattr(out, "predicted_depth"):
            val = getattr(out, "predicted_depth")
        elif isinstance(out, dict) and ("predicted_depth" in out):
            val = out["predicted_depth"]
        elif torch.is_tensor(out):
            val = out

        if val is None or (not torch.is_tensor(val)):
            return None

        if val.ndim == 3:
            val = val.unsqueeze(1)  # [B,1,H,W]
        elif val.ndim == 4:
            if val.shape[1] != 1:
                val = val.mean(dim=1, keepdim=True)
        else:
            raise RuntimeError(f"Unsupported depth tensor shape: {tuple(val.shape)}")

        return val

    @torch.no_grad()
    def forward_features(self, img: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        self.maybe_init(self.device)
        x = self._prep(img)

        if self.cfg.amp and x.is_cuda:
            with torch.cuda.amp.autocast():
                out = self.model(pixel_values=x)
        else:
            out = self.model(pixel_values=x)

        depth = self._extract_depth_tensor(out)
        if depth is None:
            raise RuntimeError(
                "Depth teacher output format is unsupported for depth extraction."
            )

        depth = F.interpolate(
            depth,
            size=img.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        return {"global": None, "tokens": None, "depth": depth}

# -------------------------
# Shared teacher pool
# -------------------------

class SharedTeacherPool:
    """
    Global registry:
    - same CLIP config   -> same CLIP teacher object
    - same DINO config   -> same DINO teacher object
    - same SigLIP2 config-> same SigLIP2 teacher object
    - same Depth config  -> same Depth teacher object
    """
    _clip_pool: Dict[Tuple[Any, ...], OpenCLIPTeacher] = {}
    _dino_pool: Dict[Tuple[Any, ...], TeacherBase] = {}
    _siglip2_pool: Dict[Tuple[Any, ...], Siglip2Teacher] = {}
    _depth_pool: Dict[Tuple[Any, ...], DepthAnythingTeacher] = {}

    @classmethod
    def _clip_key(cls, cfg: ClipConfig, device: str) -> Tuple[Any, ...]:
        return (
            str(device),
            str(cfg.model_name),
            str(cfg.pretrained),
            int(cfg.input_size),
            bool(cfg.amp),
        )

    @classmethod
    def _dino_key(cls, cfg: DinoConfig, device: str) -> Tuple[Any, ...]:
        return (
            str(device),
            str(cfg.model_name),
            int(cfg.input_size),
            bool(cfg.amp),
            bool(cfg.prefer_patch_tokens),
            str(cfg.backend).lower(),
            bool(cfg.remove_prefix_tokens),
            None if cfg.num_register_tokens is None else int(cfg.num_register_tokens),
            None if cfg.patch_size is None else int(cfg.patch_size),
            bool(cfg.strict_token_check),
        )

    @classmethod
    def _siglip2_key(cls, cfg: Siglip2Config, device: str) -> Tuple[Any, ...]:
        return (
            str(device),
            str(cfg.model_name),
            int(cfg.input_size),
            bool(cfg.amp),
            bool(cfg.prefer_patch_tokens),
        )

    @classmethod
    def _depth_key(cls, cfg: DepthConfig, device: str) -> Tuple[Any, ...]:
        return (
            str(device),
            str(cfg.model_name),
            int(cfg.input_size),
            bool(cfg.amp),
        )

    @classmethod
    def get_clip_teacher(cls, cfg: ClipConfig, device: str = "cuda") -> OpenCLIPTeacher:
        key = cls._clip_key(cfg, device)
        teacher = cls._clip_pool.get(key, None)
        if teacher is None:
            teacher = OpenCLIPTeacher(cfg, device=device)
            teacher.maybe_init(device)
            cls._clip_pool[key] = teacher
        return teacher

    @classmethod
    def get_dino_teacher(cls, cfg: DinoConfig, device: str = "cuda") -> TeacherBase:
        key = cls._dino_key(cfg, device)
        teacher = cls._dino_pool.get(key, None)
        if teacher is not None:
            return teacher

        backend = str(cfg.backend).lower()
        if backend == "auto":
            name = str(cfg.model_name).lower()
            backend = "hf" if ("dinov3" in name or name.startswith("facebook/dinov3")) else "timm"

        if backend == "hf":
            teacher = HFDinoV3Teacher(cfg, device=device)
        elif backend == "timm":
            teacher = LegacyTimmDinoTeacher(cfg, device=device)
        else:
            raise ValueError(f"Unsupported DinoConfig.backend='{cfg.backend}'. Use 'auto', 'hf', or 'timm'.")

        teacher.maybe_init(device)
        cls._dino_pool[key] = teacher
        return teacher

    @classmethod
    def get_siglip2_teacher(cls, cfg: Siglip2Config, device: str = "cuda") -> Siglip2Teacher:
        key = cls._siglip2_key(cfg, device)
        teacher = cls._siglip2_pool.get(key, None)
        if teacher is None:
            teacher = Siglip2Teacher(cfg, device=device)
            teacher.maybe_init(device)
            cls._siglip2_pool[key] = teacher
        return teacher

    @classmethod
    def get_depth_teacher(cls, cfg: DepthConfig, device: str = "cuda") -> DepthAnythingTeacher:
        key = cls._depth_key(cfg, device)
        teacher = cls._depth_pool.get(key, None)
        if teacher is None:
            teacher = DepthAnythingTeacher(cfg, device=device)
            teacher.maybe_init(device)
            cls._depth_pool[key] = teacher
        return teacher

    @classmethod
    def summary(cls) -> Dict[str, int]:
        return {
            "clip_teachers": len(cls._clip_pool),
            "dino_teachers": len(cls._dino_pool),
            "siglip2_teachers": len(cls._siglip2_pool),
            "depth_teachers": len(cls._depth_pool),
        }


# -------------------------
# DistillManager
# -------------------------

class DistillManager(nn.Module):
    """
    Branch-wise distillation manager.

    Supports:
      - CLIP global distill
      - DINO token distill / relation distill
      - SigLIP2 global distill
      - Depth dense geometry distill

    Key upgrades:
      1. branch_map supports "depth"
      2. depth uses independent dense loss on branch_outs
      3. DINO can use branch_qs when dino_source="q"
      4. depth_target is separated from target for CLIP / DINO / SigLIP2
      5. DINO token alignment now supports explicit teacher token grid metadata
      6. DINOv3 HF path removes prefix tokens (cls + registers) explicitly

    token_loss:
      - "cos"      : token cosine
      - "mse"      : token mse
      - "rel_l1"   : pairwise token relation L1
      - "rel_mse"  : pairwise token relation MSE
    """

    def __init__(
        self,
        *,
        branches: int,
        branch_map: Union[str, Dict[int, str]],
        student_channels: int,
        clip_cfg: Optional[ClipConfig] = None,
        dino_cfg: Optional[DinoConfig] = None,
        siglip2_cfg: Optional[Siglip2Config] = None,
        depth_cfg: Optional[DepthConfig] = None,
        weights: Optional[Dict[str, float]] = None,
        target: str = "in",
        depth_target: str = "gt",
        mix_ratio: float = 0.5,
        token_grid: Tuple[int, int] = (14, 14),
        token_loss: str = "cos",
        global_loss: str = "cos",
        dino_source: str = "out",
        depth_grad_weight: float = 0.5,
        device: str = "cuda",
    ):
        super().__init__()
        self.device = device
        self.branches = int(branches)
        self.branch_map = parse_branch_map(branch_map, branches=self.branches)

        self.target = str(target).lower()
        valid_targets = ("in", "out", "gt", "mix", "mix_gt_out", "mix_in_gt")
        if self.target not in valid_targets:
            raise ValueError(f"target must be one of {valid_targets}, got {self.target}")

        self.depth_target = str(depth_target).lower()
        if self.depth_target not in valid_targets:
            raise ValueError(
                f"depth_target must be one of {valid_targets}, got {self.depth_target}"
            )

        self.mix_ratio = float(mix_ratio)
        if not (0.0 <= self.mix_ratio <= 1.0):
            raise ValueError(f"mix_ratio must be in [0,1], got {self.mix_ratio}")

        self.dino_source = str(dino_source).lower()
        if self.dino_source not in ("out", "q"):
            raise ValueError(f"dino_source must be one of ('out','q'), got {self.dino_source}")

        self.depth_grad_weight = float(depth_grad_weight)
        if self.depth_grad_weight < 0.0:
            raise ValueError(f"depth_grad_weight must be >= 0, got {self.depth_grad_weight}")

        self.weights = weights or {}
        self.w_clip = float(self.weights.get("clip", 0.0))
        self.w_dino = float(self.weights.get("dino", 0.0))
        self.w_siglip2 = float(self.weights.get("siglip2", 0.0))
        self.w_depth = float(self.weights.get("depth", 0.0))

        self.token_loss = str(token_loss).lower()
        self.global_loss = str(global_loss).lower()

        valid_token_losses = ("cos", "mse", "rel_l1", "rel_mse")
        valid_global_losses = ("cos", "mse")

        if self.token_loss not in valid_token_losses:
            raise ValueError(f"token_loss must be one of {valid_token_losses}, got {self.token_loss}")
        if self.global_loss not in valid_global_losses:
            raise ValueError(f"global_loss must be one of {valid_global_losses}, got {self.global_loss}")

        self.student_channels = int(student_channels)
        self.token_grid = (int(token_grid[0]), int(token_grid[1]))

        self.clip_cfg = clip_cfg
        self.dino_cfg = dino_cfg
        self.siglip2_cfg = siglip2_cfg
        self.depth_cfg = depth_cfg

        # lazy-created student adapters
        self.global_adapters = nn.ModuleDict()  # keys: f"b{idx}_clip" / f"b{idx}_siglip2"
        self.token_adapters = nn.ModuleDict()   # keys: f"b{idx}_dino"
        self.depth_adapters = nn.ModuleDict()   # keys: f"b{idx}_depth"

    def _get_clip_teacher(self) -> OpenCLIPTeacher:
        if self.clip_cfg is None:
            raise RuntimeError("CLIP teacher requested but clip_cfg is None.")
        return SharedTeacherPool.get_clip_teacher(self.clip_cfg, device=self.device)

    def _get_dino_teacher(self) -> TeacherBase:
        if self.dino_cfg is None:
            raise RuntimeError("DINO teacher requested but dino_cfg is None.")
        return SharedTeacherPool.get_dino_teacher(self.dino_cfg, device=self.device)

    def _get_siglip2_teacher(self) -> Siglip2Teacher:
        if self.siglip2_cfg is None:
            raise RuntimeError("SigLIP2 teacher requested but siglip2_cfg is None.")
        return SharedTeacherPool.get_siglip2_teacher(self.siglip2_cfg, device=self.device)

    def _get_depth_teacher(self) -> DepthAnythingTeacher:
        if self.depth_cfg is None:
            raise RuntimeError("Depth teacher requested but depth_cfg is None.")
        return SharedTeacherPool.get_depth_teacher(self.depth_cfg, device=self.device)

    def _ensure_global_adapter(self, branch_idx: int, teacher_name: str, D: int, device: torch.device):
        key = f"b{branch_idx}_{teacher_name}"
        if key in self.global_adapters:
            self.global_adapters[key] = self.global_adapters[key].to(device)
            return
        self.global_adapters[key] = GlobalAdapter(self.student_channels, D).to(device)

    def _ensure_dino_adapter(self, branch_idx: int, D: int, device: torch.device):
        key = f"b{branch_idx}_dino"
        if key in self.token_adapters:
            self.token_adapters[key] = self.token_adapters[key].to(device)
            return
        self.token_adapters[key] = TokenAdapter(
            self.student_channels,
            D,
            grid=self.token_grid,
        ).to(device)

    def _ensure_depth_adapter(self, branch_idx: int, device: torch.device):
        key = f"b{branch_idx}_depth"
        if key in self.depth_adapters:
            self.depth_adapters[key] = self.depth_adapters[key].to(device)
            return
        self.depth_adapters[key] = DepthAdapter(self.student_channels).to(device)

    @torch.no_grad()
    def _teacher_features(
        self,
        img: torch.Tensor,
        teacher_name: str,
        feature_cache: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        cache_key: Optional[str] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        feature_cache:
            dict shared across multiple managers in the same step.
            Recommended cache keys:
                "clip_in", "clip_out", "clip_gt",
                "dino_in", "dino_out", "dino_gt",
                "siglip2_in", "siglip2_out", "siglip2_gt",
                "depth_in", "depth_out", "depth_gt"
        """
        if feature_cache is not None and cache_key is not None and cache_key in feature_cache:
            return feature_cache[cache_key]

        teacher_name = str(teacher_name).lower()
        if teacher_name == "clip":
            feat = self._get_clip_teacher().forward_features(img)
        elif teacher_name == "dino":
            feat = self._get_dino_teacher().forward_features(img)
        elif teacher_name == "siglip2":
            feat = self._get_siglip2_teacher().forward_features(img)
        elif teacher_name == "depth":
            feat = self._get_depth_teacher().forward_features(img)
        else:
            raise ValueError(f"Unknown teacher_name: {teacher_name}")

        if feature_cache is not None and cache_key is not None:
            feature_cache[cache_key] = feat
        return feat

    def _target_requires_in(self, target_mode: str) -> bool:
        return target_mode in ("in", "mix", "mix_in_gt")

    def _target_requires_out(self, target_mode: str) -> bool:
        return target_mode in ("out", "mix", "mix_gt_out")

    def _target_requires_gt(self, target_mode: str) -> bool:
        return target_mode in ("gt", "mix_gt_out", "mix_in_gt")

    def _mix_target(
        self,
        a: Optional[torch.Tensor],
        b: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        r = self.mix_ratio
        return (1.0 - r) * a + r * b

    def _resolve_target_tensor(
        self,
        *,
        target_mode: str,
        t_in: Optional[torch.Tensor],
        t_out: Optional[torch.Tensor],
        t_gt: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if target_mode == "in":
            return t_in
        if target_mode == "out":
            return t_out
        if target_mode == "gt":
            return t_gt
        if target_mode == "mix":
            return self._mix_target(t_in, t_out)
        if target_mode == "mix_gt_out":
            return self._mix_target(t_gt, t_out)
        if target_mode == "mix_in_gt":
            return self._mix_target(t_in, t_gt)
        raise RuntimeError(f"Unsupported target mode: {target_mode}")

    def _resolve_token_hw(
        self,
        hw_in: Optional[Tuple[int, int]],
        hw_out: Optional[Tuple[int, int]],
        hw_gt: Optional[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        hws = []
        for hw in (hw_in, hw_out, hw_gt):
            hw = _normalize_token_hw(hw)
            if hw is not None:
                hws.append(hw)

        if len(hws) == 0:
            return None

        ref = hws[0]
        for hw in hws[1:]:
            if hw != ref:
                strict = bool(getattr(self.dino_cfg, "strict_token_check", False)) if self.dino_cfg is not None else False
                if strict:
                    raise RuntimeError(
                        f"Inconsistent DINO token grids across targets: {hws}"
                    )
        return ref

    def _loss_global(self, student_g: torch.Tensor, teacher_g: torch.Tensor) -> torch.Tensor:
        if self.global_loss == "cos":
            return cosine_distance(student_g, teacher_g)
        student_g = l2norm(student_g, dim=-1)
        teacher_g = l2norm(teacher_g, dim=-1)
        return F.mse_loss(student_g, teacher_g)

    def _loss_tokens_value(
        self,
        student_t: torch.Tensor,
        teacher_t: torch.Tensor,
        teacher_token_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if teacher_t is None:
            return student_t.new_tensor(0.0)

        strict = bool(getattr(self.dino_cfg, "strict_token_check", False)) if self.dino_cfg is not None else False
        teacher_t = match_teacher_tokens_to_student(
            student_t=student_t,
            teacher_t=teacher_t,
            token_grid=self.token_grid,
            teacher_token_hw=teacher_token_hw,
            strict=strict,
        )

        if self.token_loss == "cos":
            return cosine_distance(student_t, teacher_t)

        student_t = l2norm(student_t, dim=-1)
        teacher_t = l2norm(teacher_t, dim=-1)
        return F.mse_loss(student_t, teacher_t)

    def _loss_tokens_relation(
        self,
        student_t: torch.Tensor,
        teacher_t: torch.Tensor,
        teacher_token_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if teacher_t is None:
            return student_t.new_tensor(0.0)

        strict = bool(getattr(self.dino_cfg, "strict_token_check", False)) if self.dino_cfg is not None else False
        teacher_t = match_teacher_tokens_to_student(
            student_t=student_t,
            teacher_t=teacher_t,
            token_grid=self.token_grid,
            teacher_token_hw=teacher_token_hw,
            strict=strict,
        )

        rel_s = pairwise_relation_matrix(student_t, remove_diagonal=True)  # [B,N,N]
        rel_t = pairwise_relation_matrix(teacher_t, remove_diagonal=True)  # [B,N,N]

        if self.token_loss == "rel_l1":
            return F.l1_loss(rel_s, rel_t)
        return F.mse_loss(rel_s, rel_t)

    def _loss_tokens(
        self,
        student_t: torch.Tensor,
        teacher_t: torch.Tensor,
        teacher_token_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if teacher_t is None:
            return student_t.new_tensor(0.0)

        if self.token_loss in ("cos", "mse"):
            return self._loss_tokens_value(student_t, teacher_t, teacher_token_hw=teacher_token_hw)

        if self.token_loss in ("rel_l1", "rel_mse"):
            return self._loss_tokens_relation(student_t, teacher_t, teacher_token_hw=teacher_token_hw)

        raise RuntimeError(f"Unsupported token_loss mode: {self.token_loss}")

    def _loss_depth(self, student_depth: torch.Tensor, teacher_depth: torch.Tensor) -> torch.Tensor:
        """
        Dense relative-depth loss:
          - normalize teacher / student depth per-image
          - L1 on normalized depth
          - gradient consistency on normalized depth
        """
        if teacher_depth is None:
            return student_depth.new_tensor(0.0)

        if student_depth.ndim != 4 or student_depth.shape[1] != 1:
            raise ValueError(
                f"student_depth must be [B,1,H,W], got {tuple(student_depth.shape)}"
            )

        if teacher_depth.ndim == 3:
            teacher_depth = teacher_depth.unsqueeze(1)
        elif teacher_depth.ndim != 4:
            raise ValueError(
                f"teacher_depth must be [B,1,H,W] or [B,H,W], got {tuple(teacher_depth.shape)}"
            )

        if teacher_depth.shape[1] != 1:
            teacher_depth = teacher_depth.mean(dim=1, keepdim=True)

        teacher_depth = teacher_depth.to(device=student_depth.device, dtype=student_depth.dtype)

        if teacher_depth.shape[-2:] != student_depth.shape[-2:]:
            teacher_depth = F.interpolate(
                teacher_depth,
                size=student_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        student_n = normalize_depth_map(student_depth)
        teacher_n = normalize_depth_map(teacher_depth)

        loss_value = F.l1_loss(student_n, teacher_n)

        gx_s, gy_s = spatial_gradients_2d(student_n)
        gx_t, gy_t = spatial_gradients_2d(teacher_n)
        loss_grad = F.l1_loss(gx_s, gx_t) + F.l1_loss(gy_s, gy_t)

        return loss_value + self.depth_grad_weight * loss_grad

    def _select_dino_branch_feature(
        self,
        branch_idx: int,
        branch_outs: Optional[List[Optional[torch.Tensor]]],
        branch_qs: Optional[List[Optional[torch.Tensor]]],
    ) -> Optional[torch.Tensor]:
        """
        If dino_source == "q", prefer branch_qs.
        If branch_qs is unavailable, gracefully fall back to branch_outs.
        """
        if self.dino_source == "q":
            if branch_qs is not None and branch_idx < len(branch_qs):
                feat_q = branch_qs[branch_idx]
                if feat_q is not None:
                    return feat_q

        if branch_outs is not None and branch_idx < len(branch_outs):
            return branch_outs[branch_idx]

        return None

    def forward(
        self,
        x_in: torch.Tensor,
        x_out: torch.Tensor,
        x_gt: Optional[torch.Tensor] = None,
        branch_outs: Optional[List[Optional[torch.Tensor]]] = None,
        branch_qs: Optional[List[Optional[torch.Tensor]]] = None,
        stage: str = "stage2",
        feature_cache: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict:
          {
            "loss": total_distill_loss,
            "clip": clip_loss,
            "dino": dino_loss,
            "siglip2": siglip2_loss,
            "depth": depth_loss,
          }
        """
        _ = str(stage).lower()

        z = x_out.new_tensor(0.0)

        no_branch_outs = branch_outs is None or (
            isinstance(branch_outs, (list, tuple)) and len(branch_outs) == 0
        )
        no_branch_qs = branch_qs is None or (
            isinstance(branch_qs, (list, tuple)) and len(branch_qs) == 0
        )

        if no_branch_outs and no_branch_qs:
            return {"loss": z, "clip": z, "dino": z, "siglip2": z, "depth": z}

        need_clip = any(v == "clip" for v in self.branch_map.values()) and (self.w_clip > 0)
        need_dino = any(v == "dino" for v in self.branch_map.values()) and (self.w_dino > 0)
        need_siglip2 = any(v == "siglip2" for v in self.branch_map.values()) and (self.w_siglip2 > 0)
        need_depth = any(v == "depth" for v in self.branch_map.values()) and (self.w_depth > 0)

        need_any_gt = (
            ((need_clip or need_dino or need_siglip2) and self._target_requires_gt(self.target))
            or (need_depth and self._target_requires_gt(self.depth_target))
        )
        if need_any_gt and x_gt is None:
            raise ValueError("Current distill configuration requires x_gt, but x_gt is None.")

        clip_t_in = clip_t_out = clip_t_gt = None
        dino_t_in = dino_t_out = dino_t_gt = None
        dino_hw_in = dino_hw_out = dino_hw_gt = None
        siglip2_t_in = siglip2_t_out = siglip2_t_gt = None
        depth_t_in = depth_t_out = depth_t_gt = None

        if need_clip:
            if self._target_requires_in(self.target):
                feat_in = self._teacher_features(
                    x_in, "clip", feature_cache=feature_cache, cache_key="clip_in"
                )
                clip_t_in = feat_in["global"]

            if self._target_requires_out(self.target):
                feat_out = self._teacher_features(
                    x_out, "clip", feature_cache=feature_cache, cache_key="clip_out"
                )
                clip_t_out = feat_out["global"]

            if self._target_requires_gt(self.target):
                feat_gt = self._teacher_features(
                    x_gt, "clip", feature_cache=feature_cache, cache_key="clip_gt"
                )
                clip_t_gt = feat_gt["global"]

        if need_dino:
            if self._target_requires_in(self.target):
                feat_in = self._teacher_features(
                    x_in, "dino", feature_cache=feature_cache, cache_key="dino_in"
                )
                dino_t_in = feat_in["tokens"]
                dino_hw_in = feat_in.get("token_hw", None)

            if self._target_requires_out(self.target):
                feat_out = self._teacher_features(
                    x_out, "dino", feature_cache=feature_cache, cache_key="dino_out"
                )
                dino_t_out = feat_out["tokens"]
                dino_hw_out = feat_out.get("token_hw", None)

            if self._target_requires_gt(self.target):
                feat_gt = self._teacher_features(
                    x_gt, "dino", feature_cache=feature_cache, cache_key="dino_gt"
                )
                dino_t_gt = feat_gt["tokens"]
                dino_hw_gt = feat_gt.get("token_hw", None)

        if need_siglip2:
            if self._target_requires_in(self.target):
                feat_in = self._teacher_features(
                    x_in, "siglip2", feature_cache=feature_cache, cache_key="siglip2_in"
                )
                siglip2_t_in = feat_in["global"]

            if self._target_requires_out(self.target):
                feat_out = self._teacher_features(
                    x_out, "siglip2", feature_cache=feature_cache, cache_key="siglip2_out"
                )
                siglip2_t_out = feat_out["global"]

            if self._target_requires_gt(self.target):
                feat_gt = self._teacher_features(
                    x_gt, "siglip2", feature_cache=feature_cache, cache_key="siglip2_gt"
                )
                siglip2_t_gt = feat_gt["global"]

        if need_depth:
            if self._target_requires_in(self.depth_target):
                feat_in = self._teacher_features(
                    x_in, "depth", feature_cache=feature_cache, cache_key="depth_in"
                )
                depth_t_in = feat_in["depth"]

            if self._target_requires_out(self.depth_target):
                feat_out = self._teacher_features(
                    x_out, "depth", feature_cache=feature_cache, cache_key="depth_out"
                )
                depth_t_out = feat_out["depth"]

            if self._target_requires_gt(self.depth_target):
                feat_gt = self._teacher_features(
                    x_gt, "depth", feature_cache=feature_cache, cache_key="depth_gt"
                )
                depth_t_gt = feat_gt["depth"]

        clip_target = None
        if need_clip:
            clip_target = self._resolve_target_tensor(
                target_mode=self.target,
                t_in=clip_t_in,
                t_out=clip_t_out,
                t_gt=clip_t_gt,
            )

        dino_target = None
        dino_target_hw = None
        if need_dino:
            dino_target = self._resolve_target_tensor(
                target_mode=self.target,
                t_in=dino_t_in,
                t_out=dino_t_out,
                t_gt=dino_t_gt,
            )
            dino_target_hw = self._resolve_token_hw(dino_hw_in, dino_hw_out, dino_hw_gt)

        siglip2_target = None
        if need_siglip2:
            siglip2_target = self._resolve_target_tensor(
                target_mode=self.target,
                t_in=siglip2_t_in,
                t_out=siglip2_t_out,
                t_gt=siglip2_t_gt,
            )

        depth_target = None
        if need_depth:
            depth_target = self._resolve_target_tensor(
                target_mode=self.depth_target,
                t_in=depth_t_in,
                t_out=depth_t_out,
                t_gt=depth_t_gt,
            )

        clip_loss_total = z
        dino_loss_total = z
        siglip2_loss_total = z
        depth_loss_total = z

        valid_clip_branches = 0
        valid_dino_branches = 0
        valid_siglip2_branches = 0
        valid_depth_branches = 0

        for b in range(self.branches):
            tag = self.branch_map.get(b, "none")

            feat_b_out = None
            if branch_outs is not None and b < len(branch_outs):
                feat_b_out = branch_outs[b]

            if tag == "clip" and need_clip and clip_target is not None:
                if feat_b_out is None:
                    continue
                D = int(clip_target.shape[-1])
                self._ensure_global_adapter(b, "clip", D, device=feat_b_out.device)
                key = f"b{b}_clip"
                student_g = self.global_adapters[key](feat_b_out)
                clip_loss_total = clip_loss_total + self._loss_global(student_g, clip_target)
                valid_clip_branches += 1

            elif tag == "dino" and need_dino and dino_target is not None:
                feat_b_dino = self._select_dino_branch_feature(
                    branch_idx=b,
                    branch_outs=branch_outs,
                    branch_qs=branch_qs,
                )
                if feat_b_dino is None:
                    continue
                D = int(dino_target.shape[-1])
                self._ensure_dino_adapter(b, D, device=feat_b_dino.device)
                key = f"b{b}_dino"
                student_t = self.token_adapters[key](feat_b_dino)
                dino_loss_total = dino_loss_total + self._loss_tokens(
                    student_t,
                    dino_target,
                    teacher_token_hw=dino_target_hw,
                )
                valid_dino_branches += 1

            elif tag == "siglip2" and need_siglip2 and siglip2_target is not None:
                if feat_b_out is None:
                    continue
                D = int(siglip2_target.shape[-1])
                self._ensure_global_adapter(b, "siglip2", D, device=feat_b_out.device)
                key = f"b{b}_siglip2"
                student_g = self.global_adapters[key](feat_b_out)
                siglip2_loss_total = siglip2_loss_total + self._loss_global(student_g, siglip2_target)
                valid_siglip2_branches += 1

            elif tag == "depth" and need_depth and depth_target is not None:
                if feat_b_out is None:
                    continue
                self._ensure_depth_adapter(b, device=feat_b_out.device)
                key = f"b{b}_depth"
                student_depth = self.depth_adapters[key](feat_b_out)
                depth_loss_total = depth_loss_total + self._loss_depth(student_depth, depth_target)
                valid_depth_branches += 1

        if valid_clip_branches > 0:
            clip_loss_total = clip_loss_total / float(valid_clip_branches)
        else:
            clip_loss_total = z

        if valid_dino_branches > 0:
            dino_loss_total = dino_loss_total / float(valid_dino_branches)
        else:
            dino_loss_total = z

        if valid_siglip2_branches > 0:
            siglip2_loss_total = siglip2_loss_total / float(valid_siglip2_branches)
        else:
            siglip2_loss_total = z

        if valid_depth_branches > 0:
            depth_loss_total = depth_loss_total / float(valid_depth_branches)
        else:
            depth_loss_total = z

        total = (
            self.w_clip * clip_loss_total
            + self.w_dino * dino_loss_total
            + self.w_siglip2 * siglip2_loss_total
            + self.w_depth * depth_loss_total
        )

        return {
            "loss": total,
            "clip": clip_loss_total,
            "dino": dino_loss_total,
            "siglip2": siglip2_loss_total,
            "depth": depth_loss_total,
        }


# -------------------------
# Convenience: build from args-like config
# -------------------------

def build_distill_manager(
    *,
    branches: int,
    vit_dim: int,
    branch_map: Union[str, Dict[int, str]],
    w_clip: float = 0.0,
    w_dino: float = 0.0,
    w_siglip2: float = 0.0,
    w_depth: float = 0.0,
    target: str = "in",
    depth_target: str = "gt",
    mix_ratio: float = 0.5,
    token_grid: Tuple[int, int] = (14, 14),
    token_loss: str = "cos",
    global_loss: str = "cos",
    dino_source: str = "out",
    depth_grad_weight: float = 0.5,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    clip_input: int = 224,
    dino_model: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
    dino_input: int = 224,
    dino_backend: str = "auto",
    dino_remove_prefix_tokens: bool = True,
    dino_num_register_tokens: Optional[int] = None,
    dino_patch_size: Optional[int] = 16,
    dino_strict_token_check: bool = True,
    siglip2_model: str = "google/siglip2-base-patch16-224",
    siglip2_input: int = 224,
    depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf",
    depth_input: int = 518,
    device: str = "cuda",
) -> DistillManager:
    clip_cfg = ClipConfig(
        model_name=clip_model,
        pretrained=clip_pretrained,
        input_size=clip_input,
    )
    dino_cfg = DinoConfig(
        model_name=dino_model,
        input_size=dino_input,
        backend=dino_backend,
        remove_prefix_tokens=bool(dino_remove_prefix_tokens),
        num_register_tokens=dino_num_register_tokens,
        patch_size=dino_patch_size,
        strict_token_check=bool(dino_strict_token_check),
    )
    siglip2_cfg = Siglip2Config(
        model_name=siglip2_model,
        input_size=siglip2_input,
    )
    depth_cfg = DepthConfig(
        model_name=depth_model,
        input_size=depth_input,
    )

    mgr = DistillManager(
        branches=branches,
        branch_map=branch_map,
        student_channels=vit_dim,
        clip_cfg=clip_cfg if w_clip > 0 else None,
        dino_cfg=dino_cfg if w_dino > 0 else None,
        siglip2_cfg=siglip2_cfg if w_siglip2 > 0 else None,
        depth_cfg=depth_cfg if w_depth > 0 else None,
        weights={
            "clip": float(w_clip),
            "dino": float(w_dino),
            "siglip2": float(w_siglip2),
            "depth": float(w_depth),
        },
        target=target,
        depth_target=depth_target,
        mix_ratio=float(mix_ratio),
        token_grid=token_grid,
        token_loss=token_loss,
        global_loss=global_loss,
        dino_source=dino_source,
        depth_grad_weight=float(depth_grad_weight),
        device=device,
    ).to(device)
    return mgr


