"""Workaround for a ROCm LayerNorm backward bug in torch < 2.13 on RDNA (wave32) GPUs.

On torch <= 2.12 ROCm builds, the tiled gamma/beta backward kernel hard-codes 64-wide
waves. On RDNA cards (wave32, e.g. RX 9070 XT / Radeon AI PRO R9700, gfx1201) it never
writes the odd-indexed elements of the LayerNorm weight/bias gradients once the input
has >= 128 rows; they hold whatever was in that memory. Nothing is NaN, so training
"runs", but every LayerNorm in the fine-tuned blocks and the pooler gets half its
gradient replaced by garbage. Reported as pytorch/pytorch#199265; fixed upstream by
pytorch/pytorch#183864 (runtime wave-size dispatch), released in torch 2.13.0.

`apply()` therefore patches only when all of these hold: a HIP build of torch, a torch
version below 2.13, and at least one visible GPU with 32-wide waves. It swaps
F.layer_norm for the same math built from elementwise ops in fp32, so autograd never
calls the fused backward kernel (~15% step-time cost). Everywhere else, including CUDA,
CPU, MPS, CDNA (wave64) and torch >= 2.13, it does nothing.
Set FERAL_ROCM_LN_FIX=0 to disable, or =1 to force it on.
"""
import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

FIXED_IN = (2, 13)
_applied = False


def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    dims = tuple(range(-len(normalized_shape), 0))
    xf = x.float()
    mean = xf.mean(dims, keepdim=True)
    var = (xf - mean).pow(2).mean(dims, keepdim=True)
    y = (xf - mean) * torch.rsqrt(var + eps)
    if weight is not None:
        y = y * weight.float()
    if bias is not None:
        y = y + bias.float()
    return y.to(x.dtype)


def _torch_version():
    """(major, minor) of the running torch, e.g. '2.12.1+rocm7.2' -> (2, 12)."""
    parts = torch.__version__.split("+")[0].split(".")
    return int(parts[0]), int("".join(c for c in parts[1] if c.isdigit()) or 0)


def _warp_sizes():
    """Wave sizes of the visible GPUs (empty if none, or if the build doesn't report them)."""
    if not torch.cuda.is_available():
        return []
    sizes = []
    for i in range(torch.cuda.device_count()):
        ws = getattr(torch.cuda.get_device_properties(i), "warp_size", None)
        if ws is not None:
            sizes.append(int(ws))
    return sizes


def affected():
    """True if this process would hit pytorch#199265 (HIP build, torch < 2.13, a wave32 GPU)."""
    if torch.version.hip is None or _torch_version() >= FIXED_IN:
        return False
    return 32 in _warp_sizes()


def apply():
    """Patch F.layer_norm if (and only if) the running setup is affected. Idempotent.
    Called when a FeralModel is built, so importing feral never touches the GPU."""
    global _applied
    if _applied:
        return
    flag = os.environ.get("FERAL_ROCM_LN_FIX")
    if flag == "0" or (flag != "1" and not affected()):
        return
    F.layer_norm = _layer_norm  # nn.LayerNorm.forward looks this up through F at call time
    _applied = True
    logger.warning("ROCm LayerNorm workaround active (torch %s on a wave32 GPU, pytorch#199265). "
                   "Upgrade to torch >= 2.13 to drop it.", torch.__version__)
