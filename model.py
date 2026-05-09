import copy
import math
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Basic convolutional utilities
# =========================================================


class SepConv(nn.Module):
    """Depthwise separable convolution used inside GMOE."""

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel,
            in_channel,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channel,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(
            in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=bias
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


class BasicBlock(nn.Module):
    """HIN-style basic block used by GMOE."""

    def __init__(self, in_size, out_size, kernel_size=3, relu_slope=0.1):
        super().__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1 = SepConv(in_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2 = SepConv(out_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out = out + self.identity(x)
        return out


class GetGradient(nn.Module):
    """Gradient extractor used by GMOE."""

    def __init__(self, dim=3, mode="sobel"):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if mode == "sobel":
            kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

            kernel_y = (
                torch.tensor(kernel_y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            kernel_x = (
                torch.tensor(kernel_x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )

            self.register_buffer("kernel_y", kernel_y.repeat(self.dim, 1, 1, 1))
            self.register_buffer("kernel_x", kernel_x.repeat(self.dim, 1, 1, 1))
        elif mode == "laplacian":
            kernel_laplace = [[0.25, 1, 0.25], [1, -5, 1], [0.25, 1, 0.25]]
            kernel_laplace = (
                torch.tensor(kernel_laplace, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.register_buffer(
                "kernel_laplace", kernel_laplace.repeat(self.dim, 1, 1, 1)
            )
        else:
            raise ValueError(f"Unsupported gradient mode: {mode}")

    def forward(self, x):
        if self.mode == "sobel":
            grad_x = F.conv2d(x, self.kernel_x, padding=1, groups=self.dim)
            grad_y = F.conv2d(x, self.kernel_y, padding=1, groups=self.dim)
            grad_magnitude = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)
        else:
            grad_magnitude = F.conv2d(
                x, self.kernel_laplace, padding=1, groups=self.dim
            )
            grad_magnitude = torch.abs(grad_magnitude)
        return grad_magnitude


class GGDC(nn.Module):
    def __init__(self, dim=3):
        super(GGDC, self).__init__()

        self.dim = dim

        # Main branches
        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        # --- Branch: Sobel X ---
        # scale_sobel_x = torch.randn(size=(dim, 1, 1, 1)) * 1e-3
        # self.scale_sobel_x = nn.Parameter(torch.FloatTensor(scale_sobel_x))
        # sobel_x_bias = torch.randn(dim) * 1e-3
        # sobel_x_bias = torch.reshape(sobel_x_bias, (dim,))
        # self.sobel_x_bias = nn.Parameter(torch.FloatTensor(sobel_x_bias))
        self.scale_sobel_x = nn.Parameter(torch.ones(dim, 1, 1, 1))
        self.sobel_x_bias = nn.Parameter(torch.zeros(dim))

        self.mask_sobel_x = torch.zeros((dim, 1, 3, 3), dtype=torch.float32)
        for i in range(dim):
            self.mask_sobel_x[i, 0, 0, 1] = 1.0
            self.mask_sobel_x[i, 0, 1, 0] = 2.0
            self.mask_sobel_x[i, 0, 2, 0] = 1.0
            self.mask_sobel_x[i, 0, 0, 2] = -1.0
            self.mask_sobel_x[i, 0, 1, 2] = -2.0
            self.mask_sobel_x[i, 0, 2, 2] = -1.0
        self.mask_sobel_x = nn.Parameter(data=self.mask_sobel_x, requires_grad=False)

        # --- Branch: Sobel Y ---
        # scale_sobel_y = torch.randn(size=(dim, 1, 1, 1)) * 1e-3
        # self.scale_sobel_y = nn.Parameter(torch.FloatTensor(scale_sobel_y))
        # sobel_y_bias = torch.randn(dim) * 1e-3
        # sobel_y_bias = torch.reshape(sobel_y_bias, (dim,))
        # self.sobel_y_bias = nn.Parameter(torch.FloatTensor(sobel_y_bias))
        self.scale_sobel_y = nn.Parameter(torch.ones(dim, 1, 1, 1))
        self.sobel_y_bias = nn.Parameter(torch.zeros(dim))

        self.mask_sobel_y = torch.zeros((dim, 1, 3, 3), dtype=torch.float32)
        for i in range(dim):
            self.mask_sobel_y[i, 0, 0, 0] = 1.0
            self.mask_sobel_y[i, 0, 0, 1] = 2.0
            self.mask_sobel_y[i, 0, 0, 2] = 1.0
            self.mask_sobel_y[i, 0, 2, 0] = -1.0
            self.mask_sobel_y[i, 0, 2, 1] = -2.0
            self.mask_sobel_y[i, 0, 2, 2] = -1.0
        self.mask_sobel_y = nn.Parameter(data=self.mask_sobel_y, requires_grad=False)

        # --- Branch: Laplacian ---
        # scale_laplacian = torch.randn(size=(dim, 1, 1, 1)) * 1e-3
        # self.scale_laplacian = nn.Parameter(torch.FloatTensor(scale_laplacian))
        # laplacian_bias = torch.randn(dim) * 1e-3
        # laplacian_bias = torch.reshape(laplacian_bias, (dim,))
        # self.laplacian_bias = nn.Parameter(torch.FloatTensor(laplacian_bias))
        self.scale_laplacian = nn.Parameter(torch.ones(dim, 1, 1, 1))
        self.laplacian_bias = nn.Parameter(torch.zeros(dim))

        self.mask_laplacian = torch.zeros((dim, 1, 3, 3), dtype=torch.float32)
        for i in range(dim):
            self.mask_laplacian[i, 0, 0, 0] = 1.0
            self.mask_laplacian[i, 0, 1, 0] = 1.0
            self.mask_laplacian[i, 0, 1, 2] = 1.0
            self.mask_laplacian[i, 0, 2, 1] = 1.0
            self.mask_laplacian[i, 0, 1, 1] = -4.0
        self.mask_laplacian = nn.Parameter(data=self.mask_laplacian, requires_grad=False)

    def forward(self, x):

        out = self.conv(x)
        out += F.conv2d(input=x, weight=self.scale_sobel_x * self.mask_sobel_x, bias=self.sobel_x_bias,
                        stride=1, padding=1, groups=self.dim)
        out += F.conv2d(input=x, weight=self.scale_sobel_y * self.mask_sobel_y, bias=self.sobel_y_bias,
                        stride=1, padding=1, groups=self.dim)
        out += F.conv2d(input=x, weight=self.scale_laplacian * self.mask_laplacian, bias=self.laplacian_bias,
                        stride=1, padding=1, groups=self.dim)

        return out

    def slim(self):
        conv_weight = self.conv.weight
        conv_bias = self.conv.bias

        sobel_x_weight = self.scale_sobel_x * self.mask_sobel_x
        sobel_y_weight = self.scale_sobel_y * self.mask_sobel_y
        laplacian_weight = self.scale_laplacian * self.mask_laplacian

        total_bias = conv_bias + self.sobel_x_bias + self.sobel_y_bias + self.laplacian_bias

        total_weight = conv_weight + sobel_x_weight + sobel_y_weight + laplacian_weight

        return total_weight, total_bias


class GGDCS(nn.Module):
    def __init__(self, dim=3):
        super(GGDCS, self).__init__()
        self.dim = dim
        self.out_channels = dim

        self.conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

    def forward(self, x):
        out = self.conv(x)
        return out


class ScaleMOE(nn.Module):

    def __init__(self, channels, num_experts=3):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_experts)

    def forward(self, x):
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        b = x.shape[0]
        x = self.pool(x).view(b, -1)
        weights = F.softmax(self.fc(x), dim=1).view(-1, 3, 1, 1, 1)
        weights = F.softmax(self.fc(x), dim=1).view(-1, 3, 1, 1, 1)

        weighted_outs = expert_outs * weights
        return weighted_outs.sum(dim=1)

class NonlinearMOE(nn.Module):

    def __init__(self, channels, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Identity(),
            nn.ReLU(),
            nn.GELU(),
            nn.Mish(),
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_experts)

    def forward(self, x):
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)
        b = x.shape[0]
        x = self.pool(x).view(b, -1)
        weights = F.softmax(self.fc(x), dim=1).view(-1, 4, 1, 1, 1)

        weighted_outs = expert_outs * weights
        return weighted_outs.sum(dim=1)


# ====================================================================================

class GMOE(nn.Module):
    """Structure-guided feature block. This is preserved in every unified stage."""

    def __init__(self, feature_channels=48):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.frdb1 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.moe = nn.Sequential(
            ScaleMOE(feature_channels),
            NonlinearMOE(feature_channels)
        )
        self.get_gradient = GGDC(feature_channels)
        self.conv_grad = nn.Sequential(
            SepConv(feature_channels, feature_channels, kernel_size=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        grad = self.get_gradient(x)
        grad = self.conv_grad(grad)
        x = self.frdb1(x)
        alpha = torch.sigmoid(self.alpha)
        x = alpha * grad * x + (1 - alpha) * x
        x = self.moe(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class GrayWorldRetinex(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        gray_mean = mean.mean(dim=1, keepdim=True)
        gain = gray_mean / (mean + self.eps)
        x = x * gain
        x_log = torch.log(x + self.eps)
        x_log = x_log - x_log.mean(dim=(2, 3), keepdim=True)
        x_out = torch.exp(x_log)
        x_min = x_out.amin(dim=(-2, -1), keepdim=True)
        x_max = x_out.amax(dim=(-2, -1), keepdim=True)
        x_out = (x_out - x_min) / (x_max - x_min + self.eps)
        return x_out


# =========================================================
# LayerNorm + fold helpers
# =========================================================


class LayerNorm2d(nn.Module):
    """LayerNorm for NCHW tensors, with optional affine parameters for folding."""

    def __init__(self, channels: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = (channels,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x


class PadDWConvCPE(nn.Module):
    """
    Depthwise positional convolution with explicit padding.
    Replicate padding preserves constants, which is important for exact LN absorption.
    """

    def __init__(self, dim: int, k: int = 3, pad_mode: str = "replicate"):
        super().__init__()
        if k % 2 != 1:
            raise ValueError(f"CPE kernel size must be odd, but got {k}")
        self.kernel_size = k
        self.pad = k // 2
        self.pad_mode = pad_mode
        self.conv = nn.Conv2d(dim, dim, k, padding=0, groups=dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode=self.pad_mode)
        return self.conv(x_pad)


@torch.no_grad()
def absorb_ln2d_affine_into_depthwise_cpe_residual(ln: LayerNorm2d, cpe: PadDWConvCPE):
    """
    Absorb LN affine parameters into the residual form: x + CPE(x).
    After absorption, LN can be replaced by non-affine LayerNorm2d without changing behavior.
    """
    if not ln.elementwise_affine:
        return

    conv = cpe.conv
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("cpe.conv must be nn.Conv2d")
    if conv.in_channels != conv.out_channels:
        raise ValueError("CPE must be channel-preserving")
    if conv.groups != conv.in_channels:
        raise ValueError("CPE must be depthwise")
    if conv.stride != (1, 1):
        raise ValueError("Only stride=1 is supported for exact residual folding")
    if conv.dilation != (1, 1):
        raise ValueError("Only dilation=1 is supported for exact residual folding")

    c = conv.in_channels
    gamma = ln.weight.detach().view(c)
    beta = ln.bias.detach().view(c)

    w = conv.weight.detach().clone()
    kh, kw = w.shape[-2], w.shape[-1]
    if kh % 2 != 1 or kw % 2 != 1:
        raise ValueError("CPE kernel must be odd-sized for exact folding")
    cy, cx = kh // 2, kw // 2

    if conv.bias is None:
        b0 = torch.zeros(c, device=w.device, dtype=w.dtype)
    else:
        b0 = conv.bias.detach().clone()

    w_new = w * gamma.view(c, 1, 1, 1)
    w_new[:, 0, cy, cx] += (gamma - 1.0)

    ksum = w[:, 0, :, :].sum(dim=(1, 2))
    conv_beta = beta * ksum
    b_new = b0 + beta + conv_beta

    conv.weight.copy_(w_new)
    if conv.bias is None:
        conv.bias = nn.Parameter(b_new)
    else:
        conv.bias.copy_(b_new)


# =========================================================
# Foldable 4D attention
# =========================================================


def _validate_neighbor_kernel_size(neighbor_kernel_size: int):
    if not isinstance(neighbor_kernel_size, int) or neighbor_kernel_size < 1:
        raise ValueError(
            f"neighbor_kernel_size must be a positive integer, but got {neighbor_kernel_size}"
        )
    if neighbor_kernel_size % 2 != 1:
        raise ValueError(
            f"neighbor_kernel_size must be odd, but got {neighbor_kernel_size}"
        )

class FoldableWindowAttn4D_Train(nn.Module):
    """
    Train-time multi-branch local window attention.
    neighborhood interaction is explicitly configurable through neighbor_kernel_size.

    Distill-related outputs:
      - branch_outs: per-branch output features after Wo(...)
      - branch_qs  : per-branch query-side features (for DINO relation distill)

    Important export rule:
      - branch_qs and branch_outs are both averaged by num_heads before returning,
        so exported distill features have more stable scale across stages with
        different head counts.
      - This averaging only affects exported distill tensors, not the actual main path.
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=16,
        branches=2,
        neighbor_kernel_size=5,
        cpe_kernel_size=3,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        _validate_neighbor_kernel_size(neighbor_kernel_size)
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim}) must be divisible by num_heads({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.branches = branches
        self.neighbor_kernel_size = neighbor_kernel_size
        self.neighbor_radius = neighbor_kernel_size // 2

        self.Wq = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Conv2d(dim, self.head_dim, 1, bias=False) for _ in range(num_heads)]
                )
                for _ in range(branches)
            ]
        )
        self.Wk = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Conv2d(dim, self.head_dim, 1, bias=False) for _ in range(num_heads)]
                )
                for _ in range(branches)
            ]
        )
        self.Wv = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Conv2d(dim, self.head_dim, 1, bias=False) for _ in range(num_heads)]
                )
                for _ in range(branches)
            ]
        )
        self.Wo = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Conv2d(self.head_dim, dim, 1, bias=False) for _ in range(num_heads)]
                )
                for _ in range(branches)
            ]
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.register_buffer("join_lambda", torch.tensor(0.0))
        self.pos_cpe = PadDWConvCPE(dim, k=cpe_kernel_size, pad_mode="replicate")

    def set_join_lambda(self, val: float):
        val = float(val)
        val = 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)
        self.join_lambda.fill_(val)

    def _normalize_branch_indices(self, branch_indices):
        """
        Normalize requested branch indices for this attention module.

        - None -> all branches
        - int  -> [int]
        - iterable[int] -> sorted unique valid indices
        - invalid / overflow indices are silently ignored
        """
        if branch_indices is None:
            return list(range(self.branches))

        if isinstance(branch_indices, int):
            branch_indices = [branch_indices]
        elif not isinstance(branch_indices, (list, tuple, set)):
            raise TypeError(
                f"branch_indices must be None, int, list, tuple or set, but got {type(branch_indices)}"
            )

        out = []
        seen = set()
        for idx in branch_indices:
            if isinstance(idx, bool):
                continue
            if not isinstance(idx, int):
                continue
            if 0 <= idx < self.branches and idx not in seen:
                seen.add(idx)
                out.append(idx)

        out.sort()
        return out

    def _rectification(self, lam: float) -> float:
        return math.sqrt(1.0 + (lam * lam) * (self.branches - 1))

    def _window_partition_4d(self, x):
        b, c, h, w = x.shape
        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        hp, wp = x.shape[2], x.shape[3]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x_win4d = x.view(-1, c, ws, ws)
        return x_win4d, hp, wp, pad_h, pad_w

    def _window_reverse_4d(self, x_win4d, hp, wp, b, c, pad_h, pad_w):
        ws = self.window_size
        x = x_win4d.view(b, hp // ws, wp // ws, c, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, hp, wp)
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, : hp - pad_h, : wp - pad_w]
        return x

    def _neighborhood_attn_win_cached_k(self, q_win4d, k_nb, v_win4d, scale: float):
        ws = self.window_size
        radius = self.neighbor_radius
        ksz = self.neighbor_kernel_size
        k2 = ksz * ksz

        bn, cq, _, _ = q_win4d.shape
        _, t, k2_check, ck = k_nb.shape
        if k2_check != k2:
            raise RuntimeError(f"k_nb k2 mismatch: {k2_check} vs {k2}")
        if ck != cq:
            raise RuntimeError(f"channel mismatch: k_nb C={ck} vs q C={cq}")

        v_cols = F.unfold(v_win4d, kernel_size=ksz, padding=radius, stride=1)
        cv = v_win4d.shape[1]
        v_nb = v_cols.view(bn, cv, k2, t).permute(0, 3, 2, 1).contiguous()

        q_flat = q_win4d.view(bn, cq, t).permute(0, 2, 1).contiguous()
        logits = torch.einsum("btc,btkc->btk", q_flat, k_nb) * scale
        attn = logits.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out_flat = torch.einsum("btk,btkd->btd", attn, v_nb)
        out = out_flat.permute(0, 2, 1).contiguous().view(bn, cv, ws, ws)
        return out

    def forward(self, x, return_branch_out: bool = False, branch_indices=None):
        """
        Args:
            x: input tensor
            return_branch_out:
                False -> normal forward
                True  -> additionally return selected branch_outs and branch_qs
            branch_indices:
                - None: return all valid branches
                - int / list[int] / tuple[int] / set[int]: only return these branches
                - invalid / overflow indices are ignored silently

        Returns:
            - if return_branch_out=False:
                out
            - if return_branch_out=True:
                out, branch_outs, branch_qs

        branch_outs / branch_qs format:
            fixed-length list with len == self.branches
            selected branches -> Tensor
            unselected branches -> None
        """
        x = x + self.pos_cpe(x)
        b, _, _, _ = x.shape

        lam = float(self.join_lambda.item())
        base_scale = 1.0 / math.sqrt(self.head_dim)
        scale_eff = base_scale * self._rectification(lam)

        x_win4d, hp, wp, ph, pw = self._window_partition_4d(x)
        ksz = self.neighbor_kernel_size
        radius = self.neighbor_radius
        k2 = ksz * ksz
        t = self.window_size * self.window_size

        k_cols = F.unfold(x_win4d, kernel_size=ksz, padding=radius, stride=1)
        bn = k_cols.shape[0]
        k_nb = k_cols.view(bn, self.dim, k2, t).permute(0, 3, 2, 1).contiguous()

        selected_branch_indices = []
        selected_branch_set = set()
        branch_outs = None
        branch_qs = None

        if return_branch_out:
            selected_branch_indices = self._normalize_branch_indices(branch_indices)
            selected_branch_set = set(selected_branch_indices)
            branch_outs = [None for _ in range(self.branches)]
            branch_qs = [None for _ in range(self.branches)]
            for b_idx in selected_branch_indices:
                branch_outs[b_idx] = torch.zeros_like(x)
                branch_qs[b_idx] = torch.zeros_like(x)

        total_out = torch.zeros_like(x)

        for h_idx in range(self.num_heads):
            wqk_list = []
            for b_idx in range(self.branches):
                wq = self.Wq[b_idx][h_idx].weight.squeeze()
                wk = self.Wk[b_idx][h_idx].weight.squeeze()
                wqk_list.append(wq.T @ wk)

            if self.branches == 2:
                w_mixed = [
                    wqk_list[0] + lam * wqk_list[1],
                    wqk_list[1] + lam * wqk_list[0],
                ]
            else:
                w_sum = wqk_list[0]
                for i in range(1, self.branches):
                    w_sum = w_sum + wqk_list[i]
                w_mixed = []
                for b_idx in range(self.branches):
                    others_sum = w_sum - wqk_list[b_idx]
                    w_mixed.append(wqk_list[b_idx] + lam * others_sum)

            head_out_acc = torch.zeros_like(x)

            for b_idx in range(self.branches):
                q = F.conv2d(x, w_mixed[b_idx].view(self.dim, self.dim, 1, 1))

                if return_branch_out and b_idx in selected_branch_set:
                    branch_qs[b_idx] = branch_qs[b_idx] + q

                q_win4d, _, _, _, _ = self._window_partition_4d(q)

                v = self.Wv[b_idx][h_idx](x)
                v_win4d, _, _, _, _ = self._window_partition_4d(v)

                out_win4d = self._neighborhood_attn_win_cached_k(
                    q_win4d=q_win4d,
                    k_nb=k_nb,
                    v_win4d=v_win4d,
                    scale=scale_eff,
                )
                y_4d = self._window_reverse_4d(
                    out_win4d, hp, wp, b, self.head_dim, ph, pw
                )

                out_branch = self.Wo[b_idx][h_idx](y_4d)
                head_out_acc = head_out_acc + out_branch

                if return_branch_out and b_idx in selected_branch_set:
                    branch_outs[b_idx] = branch_outs[b_idx] + out_branch

            total_out = total_out + head_out_acc

        out = self.proj_drop(total_out)

        if return_branch_out:
            if self.num_heads > 1:
                inv_heads = 1.0 / float(self.num_heads)
                for b_idx in selected_branch_indices:
                    branch_qs[b_idx] = branch_qs[b_idx] * inv_heads
                    branch_outs[b_idx] = branch_outs[b_idx] * inv_heads
            return out, branch_outs, branch_qs

        return out

    @torch.no_grad()
    def absorb_ln_affine(self, ln: LayerNorm2d):
        absorb_ln2d_affine_into_depthwise_cpe_residual(ln, self.pos_cpe)

    def fold(self):
        infer = FoldableWindowAttn4D_Infer(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            branches=self.branches,
            neighbor_kernel_size=self.neighbor_kernel_size,
            cpe_kernel_size=self.pos_cpe.kernel_size,
        )
        infer = infer.to(self.Wq[0][0].weight.device)
        infer.pos_cpe.load_state_dict(self.pos_cpe.state_dict())

        with torch.no_grad():
            for h_idx in range(self.num_heads):
                wqk_sum = None
                for b_idx in range(self.branches):
                    wq = self.Wq[b_idx][h_idx].weight.squeeze()
                    wk = self.Wk[b_idx][h_idx].weight.squeeze()
                    wqk_b = wq.T @ wk
                    wqk_sum = wqk_b if wqk_sum is None else (wqk_sum + wqk_b)
                infer.Wqk[h_idx].copy_(wqk_sum.reshape(self.dim, self.dim, 1, 1))

                wvo_sum = None
                for b_idx in range(self.branches):
                    wv = self.Wv[b_idx][h_idx].weight.squeeze()
                    wo = self.Wo[b_idx][h_idx].weight.squeeze()
                    wvo_b = wo @ wv
                    wvo_sum = wvo_b if wvo_sum is None else (wvo_sum + wvo_b)
                infer.Wvo[h_idx].weight.copy_(wvo_sum.reshape(self.dim, self.dim, 1, 1))

        return infer


class FoldableWindowAttn4D_Infer(nn.Module):
    """Inference-time folded local window attention."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        branches,
        neighbor_kernel_size=5,
        cpe_kernel_size=3,
    ):
        super().__init__()
        _validate_neighbor_kernel_size(neighbor_kernel_size)
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim}) must be divisible by num_heads({num_heads})")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.branches = branches
        self.neighbor_kernel_size = neighbor_kernel_size
        self.neighbor_radius = neighbor_kernel_size // 2

        self.pos_cpe = PadDWConvCPE(dim, k=cpe_kernel_size, pad_mode="replicate")
        self.register_buffer("join_lambda", torch.tensor(1.0))
        self.register_buffer("Wqk", torch.zeros(num_heads, dim, dim, 1, 1))
        self.Wvo = nn.ModuleList([nn.Conv2d(dim, dim, 1, bias=False) for _ in range(num_heads)])
        self.attn_drop = nn.Identity()
        self.proj_drop = nn.Identity()

    def set_join_lambda(self, val: float):
        self.join_lambda.fill_(1.0)

    def _window_partition_4d(self, x):
        return FoldableWindowAttn4D_Train._window_partition_4d(self, x)

    def _window_reverse_4d(self, x_win4d, hp, wp, b, c, pad_h, pad_w):
        return FoldableWindowAttn4D_Train._window_reverse_4d(
            self, x_win4d, hp, wp, b, c, pad_h, pad_w
        )

    def _neighborhood_attn_win_cached_k(self, q_win4d, k_nb, v_win4d, scale: float):
        return FoldableWindowAttn4D_Train._neighborhood_attn_win_cached_k(
            self, q_win4d, k_nb, v_win4d, scale
        )

    def forward(self, x):
        x = x + self.pos_cpe(x)
        b, _, _, _ = x.shape

        base_scale = 1.0 / math.sqrt(self.dim // self.num_heads)
        scale_eff = base_scale * math.sqrt(self.branches)

        x_win4d, hp, wp, ph, pw = self._window_partition_4d(x)
        ksz = self.neighbor_kernel_size
        radius = self.neighbor_radius
        k2 = ksz * ksz
        t = self.window_size * self.window_size

        k_cols = F.unfold(x_win4d, kernel_size=ksz, padding=radius, stride=1)
        bn = k_cols.shape[0]
        k_nb = k_cols.view(bn, self.dim, k2, t).permute(0, 3, 2, 1).contiguous()

        total_out = torch.zeros_like(x)
        for h_idx in range(self.num_heads):
            x_k = F.conv2d(x, self.Wqk[h_idx])
            q_win4d, _, _, _, _ = self._window_partition_4d(x_k)

            z = self.Wvo[h_idx](x)
            v_win4d, _, _, _, _ = self._window_partition_4d(z)

            out_win4d = self._neighborhood_attn_win_cached_k(
                q_win4d=q_win4d,
                k_nb=k_nb,
                v_win4d=v_win4d,
                scale=scale_eff,
            )
            out_4d = self._window_reverse_4d(out_win4d, hp, wp, b, self.dim, ph, pw)
            total_out += out_4d

        return total_out


# =========================================================
# Unified block and unified stage wrappers
# =========================================================

class FoldableBlock4D_Train(nn.Module):
    """
    Unified train-time block used by every stage:
        x = x + Attn(LN1(x))
        x = x + GMOE(LN2(x))

    When return_branch_out=True, this block returns:
      - branch_outs: branch output features from attention output path
      - branch_qs  : branch query-side features from attention query path
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=16,
        branches=2,
        neighbor_kernel_size=5,
        cpe_kernel_size=3,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"GMOE requires even dim, but got dim={dim}")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.branches = branches
        self.neighbor_kernel_size = neighbor_kernel_size

        self.ln1 = LayerNorm2d(dim, elementwise_affine=True)
        self.attn = FoldableWindowAttn4D_Train(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            branches=branches,
            neighbor_kernel_size=neighbor_kernel_size,
            cpe_kernel_size=cpe_kernel_size,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ln2 = LayerNorm2d(dim, elementwise_affine=False)
        self.gmoe = GMOE(dim)

    def forward(self, x, return_branch_out: bool = False, branch_indices=None):
        """
        Args:
            branch_indices:
                forwarded to attention module; invalid / overflow indices are ignored.

        Returns:
            - if return_branch_out=False:
                x
            - if return_branch_out=True:
                x, branch_outs, branch_qs
        """
        if return_branch_out:
            attn_out, branch_outs, branch_qs = self.attn(
                self.ln1(x),
                return_branch_out=True,
                branch_indices=branch_indices,
            )
            x = x + attn_out
            x = x + self.gmoe(self.ln2(x))
            return x, branch_outs, branch_qs

        x = x + self.attn(self.ln1(x))
        x = x + self.gmoe(self.ln2(x))
        return x

    def set_join_lambda(self, val: float):
        self.attn.set_join_lambda(val)

    def structural_reparameterize_absorb_ln(self):
        self.attn.absorb_ln_affine(self.ln1)
        self.ln1 = LayerNorm2d(self.dim, elementwise_affine=False)
        self.ln2 = LayerNorm2d(self.dim, elementwise_affine=False)

    def fold(self):
        device = next(self.parameters()).device
        infer = FoldableBlock4D_Infer(
            dim=self.dim,
            num_heads=self.num_heads,
            window_size=self.window_size,
            branches=self.branches,
            neighbor_kernel_size=self.neighbor_kernel_size,
            cpe_kernel_size=self.attn.pos_cpe.kernel_size,
        ).to(device)
        infer.attn = self.attn.fold()
        infer.gmoe.load_state_dict(self.gmoe.state_dict())
        infer.ln1 = self.ln1
        infer.ln2 = self.ln2
        return infer


class FoldableBlock4D_Infer(nn.Module):
    """Unified inference-time block after branch folding."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        branches,
        neighbor_kernel_size=5,
        cpe_kernel_size=3,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"GMOE requires even dim, but got dim={dim}")

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.branches = branches
        self.neighbor_kernel_size = neighbor_kernel_size

        self.ln1 = LayerNorm2d(dim, elementwise_affine=False)
        self.attn = FoldableWindowAttn4D_Infer(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            branches=branches,
            neighbor_kernel_size=neighbor_kernel_size,
            cpe_kernel_size=cpe_kernel_size,
        )
        self.ln2 = LayerNorm2d(dim, elementwise_affine=False)
        self.gmoe = GMOE(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.gmoe(self.ln2(x))
        return x


class FoldableStage_Train(nn.Module):
    """
    Unified train-time stage wrapper.

    - For normal encoder/decoder stages: set vit_dim == in_dim.
    - For bottleneck stage: set vit_dim > in_dim to enable channel expansion.
    - Branch outputs and branch query features are preserved per stage for distillation.
    """

    def __init__(
        self,
        in_dim: int,
        vit_dim: Optional[int] = None,
        depth: int = 1,
        num_heads: int = 2,
        window_size: int = 16,
        branches: int = 2,
        neighbor_kernel_size: int = 5,
        cpe_kernel_size: int = 3,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        proj_act: str = "gelu",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.vit_dim = int(vit_dim) if vit_dim is not None else int(in_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.branches = int(branches)
        self.neighbor_kernel_size = int(neighbor_kernel_size)

        if self.vit_dim % self.num_heads != 0:
            raise ValueError(
                f"vit_dim({self.vit_dim}) must be divisible by num_heads({self.num_heads})"
            )
        if self.vit_dim % 2 != 0:
            raise ValueError(f"vit_dim must be even for GMOE, but got {self.vit_dim}")

        if proj_act == "gelu":
            act = nn.GELU()
        elif proj_act == "relu":
            act = nn.ReLU(inplace=True)
        elif proj_act == "none":
            act = nn.Identity()
        else:
            raise ValueError(
                f"proj_act must be one of ['gelu', 'relu', 'none'], but got {proj_act}"
            )

        if self.vit_dim != self.in_dim:
            self.proj_up = nn.Sequential(
                nn.Conv2d(self.in_dim, self.vit_dim, kernel_size=1, bias=False),
                act,
            )
            self.proj_down = nn.Conv2d(self.vit_dim, self.in_dim, kernel_size=1, bias=False)
        else:
            self.proj_up = nn.Identity()
            self.proj_down = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                FoldableBlock4D_Train(
                    dim=self.vit_dim,
                    num_heads=self.num_heads,
                    window_size=self.window_size,
                    branches=self.branches,
                    neighbor_kernel_size=self.neighbor_kernel_size,
                    cpe_kernel_size=cpe_kernel_size,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                )
                for _ in range(self.depth)
            ]
        )

        self.is_folded = False

    def set_join_lambda(self, lam: float):
        for blk in self.blocks:
            blk.set_join_lambda(lam)

    def structural_reparameterize_absorb_ln(self):
        for blk in self.blocks:
            blk.structural_reparameterize_absorb_ln()

    def _normalize_branch_indices(self, branch_indices):
        """
        Normalize requested branch indices for this stage.

        - None -> all branches
        - int  -> [int]
        - iterable[int] -> sorted unique valid indices
        - invalid / overflow indices are silently ignored
        """
        if branch_indices is None:
            return list(range(self.branches))

        if isinstance(branch_indices, int):
            branch_indices = [branch_indices]
        elif not isinstance(branch_indices, (list, tuple, set)):
            raise TypeError(
                f"branch_indices must be None, int, list, tuple or set, but got {type(branch_indices)}"
            )

        out = []
        seen = set()
        for idx in branch_indices:
            if isinstance(idx, bool):
                continue
            if not isinstance(idx, int):
                continue
            if 0 <= idx < self.branches and idx not in seen:
                seen.add(idx)
                out.append(idx)

        out.sort()
        return out

    def forward(
            self,
            x,
            return_branch_out: bool = False,
            collect: str = "last",
            branch_indices=None,
    ):
        """
        Args:
            return_branch_out:
                whether to return selected branch outputs and branch query features
            collect:
                - 'last': only keep outputs from the last block
                - 'all' : keep outputs from every block
            branch_indices:
                - None -> all branches in this stage
                - int / iterable[int] -> selected branches only
                - invalid / overflow indices are ignored silently

        Returns:
            - if return_branch_out=False:
                y
            - if return_branch_out=True and collect='last':
                y, branch_outs_last, branch_qs_last
            - if return_branch_out=True and collect='all':
                y, branch_outs_all, branch_qs_all
        """
        res = x
        x = self.proj_up(x)

        if not return_branch_out:
            for blk in self.blocks:
                x = blk(x)
            x = self.proj_down(x)
            return x + res

        if collect not in ("last", "all"):
            raise ValueError(f"collect must be 'last' or 'all', but got {collect}")

        selected_branch_indices = self._normalize_branch_indices(branch_indices)

        # No valid branch requested for this stage: run normal forward and return None.
        if len(selected_branch_indices) == 0:
            for blk in self.blocks:
                x = blk(x)
            x = self.proj_down(x)
            y = x + res
            return y, None, None

        branch_outs_last = None
        branch_qs_last = None
        branch_outs_all = [] if collect == "all" else None
        branch_qs_all = [] if collect == "all" else None

        for blk in self.blocks:
            x, branch_outs, branch_qs = blk(
                x,
                return_branch_out=True,
                branch_indices=selected_branch_indices,
            )
            if collect == "all":
                branch_outs_all.append(branch_outs)
                branch_qs_all.append(branch_qs)
            else:
                branch_outs_last = branch_outs
                branch_qs_last = branch_qs

        x = self.proj_down(x)
        y = x + res

        if collect == "all":
            return y, branch_outs_all, branch_qs_all
        return y, branch_outs_last, branch_qs_last

    def fold(self):
        infer_blocks = nn.ModuleList([blk.fold() for blk in self.blocks])
        return FoldableStage_Infer(
            infer_blocks=infer_blocks,
            proj_up=self.proj_up,
            proj_down=self.proj_down,
            in_dim=self.in_dim,
            vit_dim=self.vit_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            window_size=self.window_size,
            branches=self.branches,
            neighbor_kernel_size=self.neighbor_kernel_size,
        )


class FoldableStage_Infer(nn.Module):
    """Unified inference-time stage wrapper."""

    def __init__(
        self,
        infer_blocks: nn.ModuleList,
        proj_up: nn.Module,
        proj_down: nn.Module,
        in_dim: int,
        vit_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        branches: int,
        neighbor_kernel_size: int,
    ):
        super().__init__()
        self.blocks = infer_blocks
        self.proj_up = proj_up
        self.proj_down = proj_down

        # Keep stage metadata after folding so train/val side helper code
        # still sees a structurally consistent stage object.
        self.in_dim = int(in_dim)
        self.vit_dim = int(vit_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.branches = int(branches)
        self.neighbor_kernel_size = int(neighbor_kernel_size)

        self.is_folded = True

    def set_join_lambda(self, lam: float):
        # Folded inference stage ignores curriculum lambda by design.
        return None

    def forward(
            self,
            x,
            return_branch_out: bool = False,
            collect: str = "last",
            branch_indices=None,
            **kwargs,
    ):
        res = x
        x = self.proj_up(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.proj_down(x)
        y = x + res
        if return_branch_out:
            return y, None, None
        return y


# =========================================================
# Model definition
# =========================================================


def _default_stage_configs(feature_channels: int) -> Dict[str, Dict]:
    """
    Default per-stage configuration.
    All 5 main stages are unified to Attn + GMOE, but each stage keeps its own explicit settings.
    """
    c = feature_channels
    return {
        "enc1": {
            "vit_dim": c,
            "depth": 1,
            "num_heads": 1,
            "window_size": 8,
            "branches": 4,
            "neighbor_kernel_size": 5,
            "cpe_kernel_size": 3,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "proj_act": "gelu",
        },
        "enc2": {
            "vit_dim": c * 2,
            "depth": 1,
            "num_heads": 2,
            "window_size": 8,
            "branches": 4,
            "neighbor_kernel_size": 3,
            "cpe_kernel_size": 3,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "proj_act": "gelu",
        },
        "bottleneck": {
            "vit_dim": c * 8,
            "depth": 1,
            "num_heads": 2,
            "window_size": 16,
            "branches": 4,
            "neighbor_kernel_size": 5,
            "cpe_kernel_size": 3,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "proj_act": "gelu",
        },
        "dec1": {
            "vit_dim": c * 2,
            "depth": 1,
            "num_heads": 2,
            "window_size": 8,
            "branches": 4,
            "neighbor_kernel_size": 3,
            "cpe_kernel_size": 3,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "proj_act": "gelu",
        },
        "dec2": {
            "vit_dim": c,
            "depth": 1,
            "num_heads": 1,
            "window_size": 8,
            "branches": 4,
            "neighbor_kernel_size": 5,
            "cpe_kernel_size": 3,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "proj_act": "gelu",
        },
    }


class myModel(nn.Module):
    """
    Unified U-Net style model.

    Main changes compared with the original version:
    1. The old BasicLayer is fully removed.
    2. All 5 major stages are now built from the same Attn + GMOE block family.
    3. neighborhood is an explicit per-stage configuration item.
    4. Fold / branch-output interfaces are available for all stages, not only bottleneck.
    5. Structured distill outputs now include both:
       - branch_outs : per-stage branch output features
       - branch_qs   : per-stage branch query-side features for DINO distill
    6. Additional distill metadata and payload validation are added for safer integration.
    """

    STAGE_NAMES = ("enc1", "enc2", "bottleneck", "dec1", "dec2")

    def __init__(
        self,
        in_channels=3,
        feature_channels=32,
        use_white_balance=False,
        stage_configs: Optional[Dict[str, Dict]] = None,
    ):
        super().__init__()
        self.use_white_balance = use_white_balance
        self.feature_channels = feature_channels

        if self.use_white_balance:
            self.wb = GrayWorldRetinex()
            self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

        default_cfg = _default_stage_configs(feature_channels)
        self.stage_configs = copy.deepcopy(default_cfg)
        if stage_configs is not None:
            for stage_name, cfg in stage_configs.items():
                if stage_name not in self.stage_configs:
                    raise KeyError(
                        f"Unknown stage name '{stage_name}'. Supported stages: {list(self.stage_configs.keys())}"
                    )
                self.stage_configs[stage_name].update(cfg)

        self.first = nn.Conv2d(
            in_channels, feature_channels, kernel_size=3, stride=1, padding=1
        )

        self.encoder1 = FoldableStage_Train(
            in_dim=feature_channels,
            **self.stage_configs["enc1"],
        )
        self.down1 = Downsample(feature_channels)

        self.encoder2 = FoldableStage_Train(
            in_dim=feature_channels * 2,
            **self.stage_configs["enc2"],
        )
        self.down2 = Downsample(feature_channels * 2)

        self.bottleneck = FoldableStage_Train(
            in_dim=feature_channels * 4,
            **self.stage_configs["bottleneck"],
        )

        self.up1 = Upsample(feature_channels * 4)
        self.decoder1 = FoldableStage_Train(
            in_dim=feature_channels * 2,
            **self.stage_configs["dec1"],
        )

        self.up2 = Upsample(feature_channels * 2)
        self.decoder2 = FoldableStage_Train(
            in_dim=feature_channels,
            **self.stage_configs["dec2"],
        )

        self.out = nn.Conv2d(
            feature_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def _stage_items(self):
        return {
            "enc1": self.encoder1,
            "enc2": self.encoder2,
            "bottleneck": self.bottleneck,
            "dec1": self.decoder1,
            "dec2": self.decoder2,
        }

    def get_stage_distill_meta(self) -> Dict[str, Dict[str, int]]:
        """
        Return distillation-related static metadata for each stage.
        This is intended for train-side config normalization and validation.
        """
        meta = {}
        stage_items = self._stage_items()

        # Relative feature stride w.r.t. model input
        stage_stride = {
            "enc1": 1,
            "enc2": 2,
            "bottleneck": 4,
            "dec1": 2,
            "dec2": 1,
        }

        for stage_name in self.STAGE_NAMES:
            stage = stage_items[stage_name]
            cfg = self.stage_configs[stage_name]

            branches = int(getattr(stage, "branches", cfg["branches"]))
            depth = int(getattr(stage, "depth", cfg["depth"]))
            vit_dim = int(getattr(stage, "vit_dim", cfg["vit_dim"]))
            num_heads = int(getattr(stage, "num_heads", cfg["num_heads"]))
            window_size = int(getattr(stage, "window_size", cfg["window_size"]))
            neighbor_kernel_size = int(
                getattr(stage, "neighbor_kernel_size", cfg["neighbor_kernel_size"])
            )

            meta[stage_name] = {
                "branches": branches,
                "depth": depth,
                "vit_dim": vit_dim,
                "num_heads": num_heads,
                "window_size": window_size,
                "neighbor_kernel_size": neighbor_kernel_size,
                "stride": int(stage_stride[stage_name]),
                "export_head_averaged": 1,   # branch_outs / branch_qs are head-averaged
                "prefer_dino_source_q": 1,   # branch_qs are the preferred DINO distill source
            }
        return meta

    def _normalize_distill_request(self, distill_request=None, return_branch_out: bool = False):
        """
        Normalize stage-wise distillation request.

        Supported input formats for each stage:
            - missing / None / False -> stage disabled
            - True                   -> all valid branches
            - int                    -> one branch
            - list/tuple/set[int]    -> selected branches
            - dict:
                {
                    "enabled": bool,              # optional
                    "branch_indices": [...],      # optional
                }

        Rules:
            - invalid / overflow indices are silently ignored
            - if distill_request is None and return_branch_out=True:
                default to all branches of all stages (backward compatibility)
            - if distill_request is None and return_branch_out=False:
                no stage is collected
        """
        stage_meta = self.get_stage_distill_meta()
        normalized = {name: None for name in self.STAGE_NAMES}

        if distill_request is None:
            if return_branch_out:
                for stage_name in self.STAGE_NAMES:
                    normalized[stage_name] = list(range(stage_meta[stage_name]["branches"]))
            return normalized

        if not isinstance(distill_request, dict):
            raise TypeError(
                f"distill_request must be None or dict, but got {type(distill_request)}"
            )

        for stage_name in self.STAGE_NAMES:
            spec = distill_request.get(stage_name, None)
            num_branches = int(stage_meta[stage_name]["branches"])

            if spec is None or spec is False:
                normalized[stage_name] = None
                continue

            if spec is True:
                normalized[stage_name] = list(range(num_branches))
                continue

            if isinstance(spec, dict):
                enabled = spec.get("enabled", True)
                if not enabled:
                    normalized[stage_name] = None
                    continue
                if "branch_indices" in spec:
                    spec = spec["branch_indices"]
                else:
                    spec = None

                if spec is None:
                    normalized[stage_name] = list(range(num_branches))
                    continue

            if isinstance(spec, int):
                spec = [spec]
            elif not isinstance(spec, (list, tuple, set)):
                raise TypeError(
                    f"distill_request[{stage_name}] must be bool/int/list/tuple/set/dict, but got {type(spec)}"
                )

            out = []
            seen = set()
            for idx in spec:
                if isinstance(idx, bool):
                    continue
                if not isinstance(idx, int):
                    continue
                if 0 <= idx < num_branches and idx not in seen:
                    seen.add(idx)
                    out.append(idx)

            out.sort()
            normalized[stage_name] = out if len(out) > 0 else None

        return normalized

    def _validate_stage_branch_payload(
        self,
        stage_name: str,
        payload,
        collect: str,
        payload_name: str,
    ):
        """
        Validate branch_outs / branch_qs payload structure so distill-side bugs
        are caught early and loudly.
        """
        if payload is None:
            return

        meta = self.get_stage_distill_meta()[stage_name]
        expected_branches = int(meta["branches"])
        expected_depth = int(meta["depth"])

        if collect == "last":
            if not isinstance(payload, list):
                raise TypeError(
                    f"{payload_name}[{stage_name}] must be a list or None in collect='last', "
                    f"but got {type(payload)}"
                )
            if len(payload) != expected_branches:
                raise RuntimeError(
                    f"{payload_name}[{stage_name}] length mismatch: got {len(payload)}, "
                    f"expected {expected_branches}"
                )
            for idx, item in enumerate(payload):
                if item is not None and not torch.is_tensor(item):
                    raise TypeError(
                        f"{payload_name}[{stage_name}][{idx}] must be Tensor or None, got {type(item)}"
                    )
            return

        if collect == "all":
            if not isinstance(payload, list):
                raise TypeError(
                    f"{payload_name}[{stage_name}] must be a list or None in collect='all', "
                    f"but got {type(payload)}"
                )
            if len(payload) != expected_depth:
                raise RuntimeError(
                    f"{payload_name}[{stage_name}] block-depth mismatch: got {len(payload)}, "
                    f"expected {expected_depth}"
                )
            for blk_idx, blk_payload in enumerate(payload):
                if blk_payload is None:
                    continue
                if not isinstance(blk_payload, list):
                    raise TypeError(
                        f"{payload_name}[{stage_name}][block={blk_idx}] must be list or None, "
                        f"got {type(blk_payload)}"
                    )
                if len(blk_payload) != expected_branches:
                    raise RuntimeError(
                        f"{payload_name}[{stage_name}][block={blk_idx}] length mismatch: "
                        f"got {len(blk_payload)}, expected {expected_branches}"
                    )
                for branch_idx, item in enumerate(blk_payload):
                    if item is not None and not torch.is_tensor(item):
                        raise TypeError(
                            f"{payload_name}[{stage_name}][block={blk_idx}][branch={branch_idx}] "
                            f"must be Tensor or None, got {type(item)}"
                        )
            return

        raise ValueError(f"collect must be 'last' or 'all', but got {collect}")

    def forward(
            self,
            x,
            return_branch_out: bool = False,
            return_dict: bool = False,
            branch_collect: str = "last",
            distill_request: Optional[Dict[str, Any]] = None,
    ):
        """
        Default:
            pred = model(x) -> Tensor

        Structured output:
            out = model(
                x,
                return_dict=True,
                distill_request={
                    "enc1": [0, 1],
                    "enc2": None,
                    "bottleneck": [0, 2, 3],
                    "dec1": False,
                    "dec2": [1],
                }
            )

            returns:
            {
                "pred": pred,
                "branch_outs": {
                    "enc1": list[Tensor|None] or None,
                    "enc2": list[Tensor|None] or None,
                    "bottleneck": list[Tensor|None] or None,
                    "dec1": list[Tensor|None] or None,
                    "dec2": list[Tensor|None] or None,
                },
                "branch_qs": {
                    "enc1": list[Tensor|None] or None,
                    "enc2": list[Tensor|None] or None,
                    "bottleneck": list[Tensor|None] or None,
                    "dec1": list[Tensor|None] or None,
                    "dec2": list[Tensor|None] or None,
                }
            }

        Notes:
            1. return_dict only controls output wrapping; it does NOT force branch collection.
            2. return_branch_out=True and distill_request=None:
               backward-compatible behavior -> collect all branches for all stages.
            3. distill_request takes precedence for fine-grained stage/branch selection.
            4. overflow branch indices are silently ignored.
            5. if model is fully folded, branch_outs / branch_qs are always returned as None per stage.
        """
        if branch_collect not in ("last", "all"):
            raise ValueError(
                f"branch_collect must be 'last' or 'all', but got {branch_collect}"
            )

        res = x
        if self.use_white_balance:
            alpha = torch.sigmoid(self.alpha)
            x = alpha * self.wb(x) + (1 - alpha) * x

        normalized_distill_request = self._normalize_distill_request(
            distill_request=distill_request,
            return_branch_out=return_branch_out,
        )

        want_structured_output = bool(return_dict or return_branch_out or (distill_request is not None))
        want_distill_collection = any(
            normalized_distill_request[stage_name] is not None
            for stage_name in self.STAGE_NAMES
        )
        fully_folded = self._is_fully_folded()

        branch_outs = None
        branch_qs = None
        if want_structured_output:
            branch_outs = {name: None for name in self.STAGE_NAMES}
            branch_qs = {name: None for name in self.STAGE_NAMES}

        x1 = self.first(x)
        if want_distill_collection and not fully_folded and normalized_distill_request["enc1"] is not None:
            x1, branch_outs["enc1"], branch_qs["enc1"] = self.encoder1(
                x1,
                return_branch_out=True,
                collect=branch_collect,
                branch_indices=normalized_distill_request["enc1"],
            )
            self._validate_stage_branch_payload("enc1", branch_outs["enc1"], branch_collect, "branch_outs")
            self._validate_stage_branch_payload("enc1", branch_qs["enc1"], branch_collect, "branch_qs")
        else:
            x1 = self.encoder1(x1)

        x2 = self.down1(x1)
        if want_distill_collection and not fully_folded and normalized_distill_request["enc2"] is not None:
            x2, branch_outs["enc2"], branch_qs["enc2"] = self.encoder2(
                x2,
                return_branch_out=True,
                collect=branch_collect,
                branch_indices=normalized_distill_request["enc2"],
            )
            self._validate_stage_branch_payload("enc2", branch_outs["enc2"], branch_collect, "branch_outs")
            self._validate_stage_branch_payload("enc2", branch_qs["enc2"], branch_collect, "branch_qs")
        else:
            x2 = self.encoder2(x2)

        x3 = self.down2(x2)
        if want_distill_collection and not fully_folded and normalized_distill_request["bottleneck"] is not None:
            x3, branch_outs["bottleneck"], branch_qs["bottleneck"] = self.bottleneck(
                x3,
                return_branch_out=True,
                collect=branch_collect,
                branch_indices=normalized_distill_request["bottleneck"],
            )
            self._validate_stage_branch_payload("bottleneck", branch_outs["bottleneck"], branch_collect, "branch_outs")
            self._validate_stage_branch_payload("bottleneck", branch_qs["bottleneck"], branch_collect, "branch_qs")
        else:
            x3 = self.bottleneck(x3)

        x = self.up1(x3) + x2
        if want_distill_collection and not fully_folded and normalized_distill_request["dec1"] is not None:
            x, branch_outs["dec1"], branch_qs["dec1"] = self.decoder1(
                x,
                return_branch_out=True,
                collect=branch_collect,
                branch_indices=normalized_distill_request["dec1"],
            )
            self._validate_stage_branch_payload("dec1", branch_outs["dec1"], branch_collect, "branch_outs")
            self._validate_stage_branch_payload("dec1", branch_qs["dec1"], branch_collect, "branch_qs")
        else:
            x = self.decoder1(x)

        x = self.up2(x) + x1
        if want_distill_collection and not fully_folded and normalized_distill_request["dec2"] is not None:
            x, branch_outs["dec2"], branch_qs["dec2"] = self.decoder2(
                x,
                return_branch_out=True,
                collect=branch_collect,
                branch_indices=normalized_distill_request["dec2"],
            )
            self._validate_stage_branch_payload("dec2", branch_outs["dec2"], branch_collect, "branch_outs")
            self._validate_stage_branch_payload("dec2", branch_qs["dec2"], branch_collect, "branch_qs")
        else:
            x = self.decoder2(x)

        pred = self.out(x) + res

        if want_structured_output:
            if fully_folded:
                branch_outs = {name: None for name in self.STAGE_NAMES}
                branch_qs = {name: None for name in self.STAGE_NAMES}
            return {
                "pred": pred,
                "branch_outs": branch_outs,
                "branch_qs": branch_qs,
            }

        return pred

    def _is_fully_folded(self) -> bool:
        return all(bool(getattr(stage, "is_folded", False)) for stage in self._stage_items().values())

    def set_join_lambda(self, lam: float):
        for stage in self._stage_items().values():
            if hasattr(stage, "set_join_lambda"):
                stage.set_join_lambda(lam)

    def structural_reparameterize_absorb_ln(self):
        for stage in self._stage_items().values():
            if hasattr(stage, "structural_reparameterize_absorb_ln"):
                stage.structural_reparameterize_absorb_ln()

    def fold_model(self, inplace: bool = True):
        """Fold every unified stage from train-time structure to inference-time structure."""
        if not inplace:
            new_m = copy.deepcopy(self)
            new_m.fold_model(inplace=True)
            return new_m

        if hasattr(self.encoder1, "fold"):
            self.encoder1 = self.encoder1.fold()
        if hasattr(self.encoder2, "fold"):
            self.encoder2 = self.encoder2.fold()
        if hasattr(self.bottleneck, "fold"):
            self.bottleneck = self.bottleneck.fold()
        if hasattr(self.decoder1, "fold"):
            self.decoder1 = self.decoder1.fold()
        if hasattr(self.decoder2, "fold"):
            self.decoder2 = self.decoder2.fold()
        return self

# =========================================================
# Utilities and tests
# =========================================================


def _count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def _profile_macs_thop(model, x):
    from thop import profile

    macs, _ = profile(model, inputs=(x,), verbose=False)
    return macs


def profile_model(feature_channels=32, use_white_balance=True, H=256, W=256, device=None, stage_configs=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    m_train = myModel(
        in_channels=3,
        feature_channels=feature_channels,
        use_white_balance=use_white_balance,
        stage_configs=stage_configs,
    ).to(device).eval()

    x = torch.randn(1, 3, H, W, device=device)

    with torch.no_grad():
        y = m_train(x)
    print("[Train] sanity output:", tuple(y.shape))

    p_total, p_tr = _count_params(m_train)
    print(f"[Train] Params total: {p_total / 1e6:.3f} M | trainable: {p_tr / 1e6:.3f} M")

    try:
        macs = _profile_macs_thop(m_train, x)
        print(f"[Train] MACs (paper-style GFLOPs): {macs / 1e9:.3f} G @ {H}x{W}")
    except Exception as e:
        print("[Train] THOP MACs failed:", repr(e))

    m_fold = myModel(
        in_channels=3,
        feature_channels=feature_channels,
        use_white_balance=use_white_balance,
        stage_configs=stage_configs,
    ).to(device).eval()
    m_fold.set_join_lambda(1.0)
    m_fold.structural_reparameterize_absorb_ln()
    m_fold.fold_model(inplace=True)

    with torch.no_grad():
        y2 = m_fold(x)
    print("[Fold] sanity output:", tuple(y2.shape))

    p_total2, p_tr2 = _count_params(m_fold)
    print(f"[Fold] Params total: {p_total2 / 1e6:.3f} M | trainable: {p_tr2 / 1e6:.3f} M")

    try:
        macs2 = _profile_macs_thop(m_fold, x)
        print(f"[Fold] MACs (paper-style GFLOPs): {macs2 / 1e9:.3f} G @ {H}x{W}")
    except Exception as e:
        print("[Fold] THOP MACs failed:", repr(e))


@torch.no_grad()
def benchmark_latency(model, x, iters=100, warmup=20):
    model.eval()
    device = x.device.type

    for _ in range(warmup):
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        for _ in range(iters):
            _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        ms = starter.elapsed_time(ender) / iters
        return ms

    start = time.perf_counter()
    for _ in range(iters):
        _ = model(x)
    end = time.perf_counter()
    return (end - start) * 1000.0 / iters


def compare_train_fold_latency(feature_channels=32, use_white_balance=True, H=256, W=256, stage_configs=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, H, W, device=device)

    m_train = myModel(
        in_channels=3,
        feature_channels=feature_channels,
        use_white_balance=use_white_balance,
        stage_configs=stage_configs,
    ).to(device).eval()

    m_fold = myModel(
        in_channels=3,
        feature_channels=feature_channels,
        use_white_balance=use_white_balance,
        stage_configs=stage_configs,
    ).to(device).eval()
    m_fold.set_join_lambda(1.0)
    m_fold.structural_reparameterize_absorb_ln()
    m_fold.fold_model(inplace=True)

    if device == "cuda":
        iters, warmup = 100, 20
    else:
        iters, warmup = 10, 3

    t1 = benchmark_latency(m_train, x, iters=iters, warmup=warmup)
    t2 = benchmark_latency(m_fold, x, iters=iters, warmup=warmup)

    print(f"[Latency][{device}] Train: {t1:.3f} ms/iter")
    print(f"[Latency][{device}] Fold : {t2:.3f} ms/iter")
    print(f"[Speedup] x{t1 / t2:.3f}")


@torch.no_grad()
def _selftest_branch_outputs(feature_channels=24, use_white_balance=True, H=128, W=128, stage_configs=None):
    print("\n====================")
    print("[SELFTEST] Branch outputs sanity check")
    print("====================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1, 3, H, W, device=device)

    m = myModel(
        in_channels=3,
        feature_channels=feature_channels,
        use_white_balance=use_white_balance,
        stage_configs=stage_configs,
    ).to(device).eval()

    # selective request sanity
    selective_req = {
        "enc1": [0, 1, 2, 3],          # enc1 only has 2 branches -> 2,3 ignored
        "enc2": None,                  # disabled
        "bottleneck": [0, 2, 3, 99],   # 99 ignored
        "dec1": [],                    # disabled after normalization
        "dec2": [1],                   # only branch 1
    }
    out_sel = m(
        x,
        return_dict=True,
        distill_request=selective_req,
        branch_collect="last",
    )
    assert isinstance(out_sel, dict), "Expected dict output for selective distill request"
    assert "pred" in out_sel and "branch_outs" in out_sel, "Missing keys in selective output"

    bo_sel = out_sel["branch_outs"]
    bq_sel = out_sel["branch_qs"]

    # enc1: two valid branches, both selected
    assert bo_sel["enc1"] is not None, "enc1 should be collected"
    assert len(bo_sel["enc1"]) == 2, "enc1 should keep fixed-length branch list"
    assert bo_sel["enc1"][0] is not None and bo_sel["enc1"][1] is not None, "enc1 branch 0/1 should be valid"
    assert bq_sel["enc1"] is not None, "enc1 branch_qs should be collected"
    assert len(bq_sel["enc1"]) == 2, "enc1 branch_qs should keep fixed-length branch list"

    # enc2: disabled
    assert bo_sel["enc2"] is None, "enc2 should be disabled"
    assert bq_sel["enc2"] is None, "enc2 branch_qs should be disabled"

    # bottleneck: branches 0/2/3 selected, branch 1 not selected
    assert bo_sel["bottleneck"] is not None, "bottleneck should be collected"
    assert len(bo_sel["bottleneck"]) == 4, "bottleneck should keep fixed-length branch list"
    assert bo_sel["bottleneck"][0] is not None, "bottleneck branch 0 should be valid"
    assert bo_sel["bottleneck"][1] is None, "bottleneck branch 1 should be unselected"
    assert bo_sel["bottleneck"][2] is not None, "bottleneck branch 2 should be valid"
    assert bo_sel["bottleneck"][3] is not None, "bottleneck branch 3 should be valid"

    assert bq_sel["bottleneck"] is not None, "bottleneck branch_qs should be collected"
    assert len(bq_sel["bottleneck"]) == 4, "bottleneck branch_qs should keep fixed-length branch list"
    assert bq_sel["bottleneck"][0] is not None, "bottleneck q branch 0 should be valid"
    assert bq_sel["bottleneck"][1] is None, "bottleneck q branch 1 should be unselected"
    assert bq_sel["bottleneck"][2] is not None, "bottleneck q branch 2 should be valid"
    assert bq_sel["bottleneck"][3] is not None, "bottleneck q branch 3 should be valid"

    # dec1: disabled
    assert bo_sel["dec1"] is None, "dec1 should be disabled"
    assert bq_sel["dec1"] is None, "dec1 branch_qs should be disabled"

    # dec2: only branch 1 selected
    assert bo_sel["dec2"] is not None, "dec2 should be collected"
    assert len(bo_sel["dec2"]) == 2, "dec2 should keep fixed-length branch list"
    assert bo_sel["dec2"][0] is None, "dec2 branch 0 should be unselected"
    assert bo_sel["dec2"][1] is not None, "dec2 branch 1 should be valid"

    assert bq_sel["dec2"] is not None, "dec2 branch_qs should be collected"
    assert len(bq_sel["dec2"]) == 2, "dec2 branch_qs should keep fixed-length branch list"
    assert bq_sel["dec2"][0] is None, "dec2 q branch 0 should be unselected"
    assert bq_sel["dec2"][1] is not None, "dec2 q branch 1 should be valid"

    print("[SELFTEST] selective distill request: OK")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"



    x = torch.randn(1, 3, 64, 64)
    m = myModel(in_channels=3, feature_channels=16, use_white_balance=True).eval()
    y = m(x)
    print("[MAIN] output:", y.shape)
    # Full model (paper default)



    profile_model(feature_channels=24, use_white_balance=True, H=256, W=256)
    _selftest_branch_outputs(feature_channels=24, use_white_balance=True, H=256, W=256)
    # compare_train_fold_latency(feature_channels=16, use_white_balance=True, H=64, W=64)

    # Example for manual stage-wise tuning:
    # custom_stage_configs = {
    #     "enc1": {"num_heads": 1, "branches": 2, "neighbor_kernel_size": 5, "window_size": 8},
    #     "enc2": {"num_heads": 2, "branches": 2, "neighbor_kernel_size": 3, "window_size": 8},
    #     "bottleneck": {"vit_dim": 192, "num_heads": 3, "branches": 4, "neighbor_kernel_size": 5, "window_size": 16},
    #     "dec1": {"num_heads": 2, "branches": 2, "neighbor_kernel_size": 3, "window_size": 8},
    #     "dec2": {"num_heads": 1, "branches": 2, "neighbor_kernel_size": 5, "window_size": 8},
    # }
    # m_custom = myModel(feature_channels=24, use_white_balance=True, stage_configs=custom_stage_configs)