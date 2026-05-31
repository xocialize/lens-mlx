"""Decisive check — do the DEQUANTIZED quantized weights match (HF MXFP4 vs MLX)?

embed (non-quantized) already matched at cosine 1.0. Here we compare quantized layers
(attn q_proj, MoE experts, attention sinks) between PT Mxfp4Config(dequantize=True) and
the mlx dequantize_model output. If these diverge, the bug is the MXFP4 dequant
interpretation; if they match, the bug is in the forward compute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "refs" / "Lens"))


def stats(name, a, b):
    a = a.astype("float64"); b = b.astype("float64")
    if a.shape != b.shape:
        print(f"  {name:28s} SHAPE MISMATCH pt={a.shape} mx={b.shape}")
        # try transpose
        if a.T.shape == b.shape:
            a = a.T; print(f"    (transposed pt -> {a.shape})")
        else:
            return
    af, bf = a.flatten(), b.flatten()
    cos = float(af @ bf / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-9))
    rel = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-9))
    print(f"  {name:28s} cos={cos:.6f} rel={rel:.3e} pt|max|={np.abs(a).max():.4g} mx|max|={np.abs(b).max():.4g}")


def main():
    import torch, mlx.core as mx
    from transformers import Mxfp4Config
    from lens import LensGptOssEncoder as PTEnc
    from mlx_lm.utils import load_model

    pt = PTEnc.from_pretrained(
        str(ROOT / "weights" / "Lens" / "text_encoder"),
        dtype=torch.bfloat16, quantization_config=Mxfp4Config(dequantize=True),
    )
    mxm, _ = load_model(ROOT / "weights" / "Lens-encoder-mlx-bf16")
    pl0 = pt.model.layers[0]
    ml0 = mxm.model.layers[0]

    def pnp(t): return t.detach().float().numpy()
    def mnp(t): return np.array(t.astype(mx.float32))

    print("=== layer 0 attention weights ===")
    for nm in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        stats(f"self_attn.{nm}.weight", pnp(getattr(pl0.self_attn, nm).weight),
              mnp(getattr(ml0.self_attn, nm).weight))
    if hasattr(pl0.self_attn, "sinks") and hasattr(ml0.self_attn, "sinks"):
        stats("self_attn.sinks", pnp(pl0.self_attn.sinks), mnp(ml0.self_attn.sinks))

    print("=== layer 0 MoE expert weights (PT names vs MLX names) ===")
    # PT (HF gpt-oss dequant): mlp.experts.gate_up_proj [E,H,2I], down_proj [E,I,H]
    pe = pl0.mlp.experts
    me = ml0.mlp.experts
    print("  PT expert attrs:", [a for a in dir(pe) if not a.startswith('_') and 'proj' in a])
    print("  MX expert attrs:", [a for a in dir(me) if not a.startswith('_') and 'proj' in a])
    for nm in ["gate_up_proj", "down_proj"]:
        if hasattr(pe, nm):
            t = getattr(pe, nm)
            if hasattr(t, "shape"):
                print(f"    PT experts.{nm}: {tuple(t.shape)} {t.dtype}")
    for nm in ["gate_proj", "up_proj", "down_proj"]:
        if hasattr(me, nm):
            w = getattr(me, nm).weight if hasattr(getattr(me, nm), "weight") else getattr(me, nm)
            if hasattr(w, "shape"):
                print(f"    MX experts.{nm}: {tuple(w.shape)}")


if __name__ == "__main__":
    main()
