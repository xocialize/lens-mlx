"""Measure int4 DiT parity vs the bf16/fp32 golden across quant scopes."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[2]
G = np.load(ROOT / "goldens" / "lens_goldens.npz")
DIT = ROOT / "weights" / "Lens" / "transformer"

from lens_mlx.model.transformer import LensTransformer2DModel
from lens_mlx.utils.weights import load_dit_weights, quantize_dit


def dit_inputs():
    hidden = mx.array(G["dit_in_hidden"].astype("float32"))
    timestep = mx.array(G["dit_in_timestep"].astype("float32"))
    enc = [mx.array(np.concatenate([G[f"text_feat_{i}"].astype("float32"),
                                    np.zeros_like(G[f"text_feat_{i}"].astype("float32"))], 0))
           for i in range(4)]
    mask = mx.array(np.concatenate([G["text_mask"].astype("int32"),
                                    np.zeros_like(G["text_mask"].astype("int32"))], 0))
    return hidden, enc, mask, timestep


def run(model):
    h, enc, mask, t = dit_inputs()
    out = model(hidden_states=h, encoder_hidden_states=enc,
                encoder_hidden_states_mask=mask, timestep=t, img_shapes=[(1, 32, 32)])
    mx.eval(out)
    return np.array(out).astype("float64")


def metrics(out):
    ref = G["dit_out_noise"].astype("float64")
    ma = float(np.abs(out - ref).max())
    cos = float((out.flatten() @ ref.flatten()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
    return ma, cos


def nbytes(model):
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


configs = [
    ("bf16 (baseline)", None),
    ("int4 all Linears", dict(group_size=64, bits=4, keep_hi_precision=())),
    ("int4 keep in/out/time", dict(group_size=64, bits=4,
        keep_hi_precision=("img_in", "txt_in", "proj_out", "time_text_embed", "norm_out"))),
    ("int8 all Linears", dict(group_size=64, bits=8, keep_hi_precision=())),
]

print(f"{'config':26s}{'GB':>7s}{'max_abs':>12s}{'cosine':>11s}")
for name, q in configs:
    m = LensTransformer2DModel()
    load_dit_weights(m, DIT, dtype=mx.bfloat16)
    if q:
        quantize_dit(m, **q)
    out = run(m)
    ma, cos = metrics(out)
    print(f"{name:26s}{nbytes(m)/1e9:7.2f}{ma:12.3e}{cos:11.6f}")
