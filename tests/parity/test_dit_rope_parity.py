"""Phase 2 micro-parity — RoPE (T2) and AdaLN (T3) vs the PT reference.

These run before the block/full-DiT tests: the complex axial RoPE is the trickiest
translation (MLX has no view_as_complex), so it gets an isolated gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "refs" / "Lens"))

IMG_SHAPES = [(1, 32, 32)]   # latent 32x32 -> seq 1024 (matches 512px goldens)
TXT_LEN = 35


def _max_abs(a, b):
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def test_embed_rope_parity():
    import torch, mlx.core as mx
    from lens.transformer import LensEmbedRope as PTRope
    from lens_mlx.model.transformer import LensEmbedRope as MXRope

    pt = PTRope(theta=10000, axes_dim=[8, 28, 28], scale_rope=True)
    mxr = MXRope(theta=10000, axes_dim=[8, 28, 28], scale_rope=True)

    vid_pt, txt_pt = pt(IMG_SHAPES, [TXT_LEN], device=torch.device("cpu"))
    (vid_cos, vid_sin), (txt_cos, txt_sin) = mxr(IMG_SHAPES, [TXT_LEN])

    # PT freqs_cis are complex (cos + i sin); compare real/imag to our cos/sin.
    vid_pt = vid_pt.detach().numpy(); txt_pt = txt_pt.detach().numpy()
    assert _max_abs(np.real(vid_pt), np.array(vid_cos)) < 1e-5
    assert _max_abs(np.imag(vid_pt), np.array(vid_sin)) < 1e-5
    assert _max_abs(np.real(txt_pt), np.array(txt_cos)) < 1e-5
    assert _max_abs(np.imag(txt_pt), np.array(txt_sin)) < 1e-5


def test_apply_rotary_parity():
    import torch, mlx.core as mx
    from lens.transformer import apply_rotary_emb_lens as pt_apply, LensEmbedRope as PTRope
    from lens_mlx.model.transformer import apply_rotary_emb_lens as mx_apply, LensEmbedRope as MXRope

    B, S, H, D = 2, 1024, 24, 64
    x = np.random.randn(B, S, H, D).astype(np.float32)

    pt = PTRope(theta=10000, axes_dim=[8, 28, 28], scale_rope=True)
    vid_pt, _ = pt(IMG_SHAPES, [TXT_LEN], device=torch.device("cpu"))
    pt_out = pt_apply(torch.tensor(x), vid_pt[:S]).detach().numpy()

    mxr = MXRope(theta=10000, axes_dim=[8, 28, 28], scale_rope=True)
    (vid_cos, vid_sin), _ = mxr(IMG_SHAPES, [TXT_LEN])
    mx_out = np.array(mx_apply(mx.array(x), vid_cos[:S], vid_sin[:S]))

    assert _max_abs(pt_out, mx_out) < 1e-4, f"rope diverges: {_max_abs(pt_out, mx_out):.3e}"


def test_adaln_modulate_parity():
    import torch, mlx.core as mx
    from lens.transformer import LensTransformerBlock as PTBlock
    from lens_mlx.model.transformer import LensTransformerBlock as MXBlock

    B, S, dim = 2, 16, 1536
    x = np.random.randn(B, S, dim).astype(np.float32)
    mod = np.random.randn(B, 3 * dim).astype(np.float32)

    pt_x, pt_gate = PTBlock._modulate(torch.tensor(x), torch.tensor(mod))
    mx_x, mx_gate = MXBlock._modulate(mx.array(x), mx.array(mod))
    assert _max_abs(pt_x.detach().numpy(), np.array(mx_x)) < 1e-5
    assert _max_abs(pt_gate.detach().numpy(), np.array(mx_gate)) < 1e-5
