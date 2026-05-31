"""Weight loading for the Lens DiT (PT safetensors -> MLX module).

The DiT is pure Linear + RMSNorm (no Conv), so layouts are identical PT<->MLX and no
transpose is needed. The only key renames: upstream wraps the AdaLN modulation in
`nn.Sequential(SiLU, Linear)` (param key `img_mod.1.*`), which we flatten to a single
Linear (`img_mod.*`). Same for `txt_mod`.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict

import mlx.core as mx


def sanitize_dit_key(k: str) -> str:
    return k.replace(".img_mod.1.", ".img_mod.").replace(".txt_mod.1.", ".txt_mod.")


def load_dit_state(weights_dir, dtype=mx.float32) -> Dict[str, mx.array]:
    """Load + sanitize the transformer safetensors into a flat {key: array} dict."""
    files = sorted(glob.glob(str(Path(weights_dir) / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors under {weights_dir}")
    state: Dict[str, mx.array] = {}
    for f in files:
        for k, v in mx.load(f).items():
            state[sanitize_dit_key(k)] = v.astype(dtype)
    return state


def load_dit_weights(model, weights_dir, dtype=mx.float32, strict: bool = True):
    """Load sanitized DiT weights into an instantiated LensTransformer2DModel."""
    state = load_dit_state(weights_dir, dtype=dtype)
    mx.eval(state)
    model.load_weights(list(state.items()), strict=strict)
    return model
