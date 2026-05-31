"""Pin the encoder bug — compare expert weight VALUES (PT dequant vs MLX) and
isolate layer-0 attention-out vs MLP-out so we know which sub-block diverges.
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "refs" / "Lens"))


def cos(a, b):
    a = a.astype("float64").flatten(); b = b.astype("float64").flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    import torch, mlx.core as mx
    from transformers import Mxfp4Config
    from lens import LensGptOssEncoder as PTEnc
    from mlx_lm.utils import load_model

    pt = PTEnc.from_pretrained(
        str(ROOT / "weights" / "Lens" / "text_encoder"),
        dtype=torch.bfloat16, quantization_config=Mxfp4Config(dequantize=True))
    mxm, _ = load_model(ROOT / "weights" / "Lens-encoder-mlx-bf16")
    pe = pt.model.layers[0].mlp.experts
    me = mxm.model.layers[0].mlp.experts

    gup = pe.gate_up_proj.detach().float().numpy()   # [E, H, 2I]
    dwn = pe.down_proj.detach().float().numpy()       # [E, ?, ?]
    mg = np.array(me.gate_proj.weight.astype(mx.float32))  # [E, I, H]?
    mu = np.array(me.up_proj.weight.astype(mx.float32))
    md = np.array(me.down_proj.weight.astype(mx.float32))
    print(f"PT gate_up {gup.shape} down {dwn.shape} | MX gate {mg.shape} up {mu.shape} down {md.shape}")

    # Candidate PT->MX mappings for gate/up. HF interleaves on last axis (2I).
    cands = {
        "gate=gup[:,:,0::2]^T": gup[:, :, 0::2].transpose(0, 2, 1),
        "gate=gup[:,:,1::2]^T": gup[:, :, 1::2].transpose(0, 2, 1),
        "gate=gup[:,:,:I]^T":   gup[:, :, :gup.shape[2]//2].transpose(0, 2, 1),
        "gate=gup[:,:,I:]^T":   gup[:, :, gup.shape[2]//2:].transpose(0, 2, 1),
        "gate=gup[:,:,0::2]":   gup[:, :, 0::2],
    }
    print("\n[gate_proj] best PT-slice vs MX gate_proj:")
    for name, arr in cands.items():
        if arr.shape == mg.shape:
            print(f"  {name:24s} cos={cos(arr, mg):.6f}")
    print("[up_proj] PT-slice vs MX up_proj:")
    for name, arr in cands.items():
        if arr.shape == mu.shape:
            print(f"  {name.replace('gate','up'):24s} cos={cos(arr, mu):.6f}")
    # down: try direct + transpose
    print("[down_proj] PT vs MX:")
    for name, arr in {"down": dwn, "down^T": dwn.transpose(0, 2, 1)}.items():
        if arr.shape == md.shape:
            print(f"  {name:24s} cos={cos(arr, md):.6f}")


if __name__ == "__main__":
    main()
