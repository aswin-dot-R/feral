"""The ROCm LayerNorm replacement must match F.layer_norm, and apply only where pytorch#199265 bites."""
import importlib

import pytest
import torch
import torch.nn.functional as F

from feral import rocm_compat


@pytest.mark.parametrize("affine", [True, False])
def test_layer_norm_matches_reference_fp32(affine):
    torch.manual_seed(0)
    d = 64
    x = torch.randn(3, 17, d, requires_grad=True)
    w = torch.randn(d, requires_grad=True) if affine else None
    b = torch.randn(d, requires_grad=True) if affine else None
    leaves = [t for t in (x, w, b) if t is not None]

    ref = F.layer_norm(x, (d,), w, b, 1e-6)
    ref_grads = torch.autograd.grad(ref.square().sum(), leaves)
    out = rocm_compat._layer_norm(x, (d,), w, b, 1e-6)
    out_grads = torch.autograd.grad(out.square().sum(), leaves)

    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    for g, r in zip(out_grads, ref_grads):
        torch.testing.assert_close(g, r, rtol=1e-4, atol=1e-4)


def test_layer_norm_bf16_no_less_accurate_than_builtin():
    """In bf16 the replacement computes in fp32, so vs an fp32 ground truth it must be at
    least as accurate as the built-in kernel (it is usually more accurate)."""
    torch.manual_seed(0)
    d = 64
    x32, w32, b32 = torch.randn(3, 17, d), torch.randn(d), torch.randn(d)
    truth = F.layer_norm(x32, (d,), w32, b32, 1e-6)
    x, w, b = (t.bfloat16() for t in (x32, w32, b32))
    err_builtin = (F.layer_norm(x, (d,), w, b, 1e-6).float() - truth).abs().max()
    out = rocm_compat._layer_norm(x, (d,), w, b, 1e-6)
    assert out.dtype == torch.bfloat16
    assert (out.float() - truth).abs().max() <= err_builtin * 1.01 + 1e-6


@pytest.fixture
def fresh(monkeypatch):
    """Reloaded rocm_compat with F.layer_norm restored afterwards and no env override."""
    monkeypatch.setattr(F, "layer_norm", F.layer_norm)
    monkeypatch.delenv("FERAL_ROCM_LN_FIX", raising=False)
    return importlib.reload(rocm_compat)


def _setup(monkeypatch, mod, hip, version, warps):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(mod, "_warp_sizes", lambda: warps)


@pytest.mark.parametrize("hip, version, warps, expected", [
    ("7.2.0", "2.12.1+rocm7.2", [32], True),        # RDNA on an affected torch: patch
    ("7.2.0", "2.12.1+rocm7.2", [64], False),       # CDNA (wave64) never had the bug
    ("7.2.0", "2.12.1+rocm7.2", [64, 32], True),    # any wave32 card visible
    ("7.2.0", "2.13.0+rocm7.1", [32], False),       # fixed upstream in 2.13 (pytorch#183864)
    ("7.2.0", "2.14.1+rocm7.2", [32], False),
    ("7.2.0", "2.9.1+rocm6.4", [32], True),         # numeric, not lexicographic, version compare
    ("7.2.0", "2.12.1+rocm7.2", [], False),         # no GPU visible
    (None, "2.12.1+cu130", [32], False),            # CUDA build
])
def test_apply_only_when_affected(monkeypatch, fresh, hip, version, warps, expected):
    original = F.layer_norm
    _setup(monkeypatch, fresh, hip, version, warps)
    assert fresh.affected() is expected
    fresh.apply()
    assert (F.layer_norm is fresh._layer_norm) is expected
    if not expected:
        assert F.layer_norm is original


def test_env_override(monkeypatch, fresh):
    original = F.layer_norm
    _setup(monkeypatch, fresh, "7.2.0", "2.12.1+rocm7.2", [32])
    monkeypatch.setenv("FERAL_ROCM_LN_FIX", "0")
    fresh.apply()
    assert F.layer_norm is original                  # opt-out wins on an affected setup

    fresh = importlib.reload(rocm_compat)
    _setup(monkeypatch, fresh, None, "2.14.1+cu130", [])
    monkeypatch.setenv("FERAL_ROCM_LN_FIX", "1")
    fresh.apply()
    assert F.layer_norm is fresh._layer_norm         # force-on works anywhere
    assert torch.nn.LayerNorm(8)(torch.randn(2, 8)).shape == (2, 8)
