"""Phase 2 gate — full Lens DiT forward parity vs the PT golden.

Feeds the captured DiT input (`dit_in_hidden`, `dit_in_timestep`) plus the
reconstructed CFG-batched encoder features (positive golden features + zero negative,
matching negative_prompt="") through the MLX DiT and compares to `dit_out_noise`.

Gate: max_abs < 1e-2 (skill full-DiT threshold). fp32 both sides.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "goldens" / "lens_goldens.npz"
DIT = ROOT / "weights" / "Lens" / "transformer"

pytestmark = pytest.mark.skipif(
    not GOLDENS.exists() or not DIT.exists(),
    reason="needs goldens + transformer weights",
)


def test_full_dit_parity():
    import mlx.core as mx
    from lens_mlx.model.transformer import LensTransformer2DModel
    from lens_mlx.utils.weights import load_dit_weights

    g = np.load(GOLDENS)
    hidden = mx.array(g["dit_in_hidden"].astype("float32"))          # [2,1024,128]
    timestep = mx.array(g["dit_in_timestep"].astype("float32"))       # [2]
    # Reconstruct DiT encoder inputs: positive golden feats + zero negative branch.
    enc = []
    for i in range(4):
        pos = g[f"text_feat_{i}"].astype("float32")                   # [1,35,2880]
        neg = np.zeros_like(pos)
        enc.append(mx.array(np.concatenate([pos, neg], axis=0)))      # [2,35,2880]
    pos_mask = g["text_mask"].astype("int32")                         # [1,35] (all 1)
    mask = mx.array(np.concatenate([pos_mask, np.zeros_like(pos_mask)], axis=0))  # [2,35]
    img_shapes = [(1, 32, 32)]

    model = LensTransformer2DModel()
    load_dit_weights(model, DIT, dtype=mx.float32)

    out = model(
        hidden_states=hidden, encoder_hidden_states=enc,
        encoder_hidden_states_mask=mask, timestep=timestep, img_shapes=img_shapes,
    )
    mx.eval(out)
    out = np.array(out).astype("float64")
    ref = g["dit_out_noise"].astype("float64")

    max_abs = float(np.abs(out - ref).max())
    rel = max_abs / (np.abs(ref).max() + 1e-9)
    cosv = float((out.flatten() @ ref.flatten()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
    print(f"\n[full DiT] max_abs={max_abs:.3e}  rel={rel:.3e}  cosine={cosv:.6f}  "
          f"out|max|={np.abs(out).max():.3f} ref|max|={np.abs(ref).max():.3f}")
    assert max_abs < 1e-2, f"DiT diverges: max_abs={max_abs:.3e}"
