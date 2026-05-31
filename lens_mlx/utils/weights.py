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
import mlx.nn as nn


def quantize_dit(model, group_size: int = 64, bits: int = 4, keep_hi_precision=()):
    """Quantize DiT Linears in place (group_size/bits). `keep_hi_precision` is a tuple
    of substrings; any Linear whose path contains one is left at bf16/fp32 (e.g. the
    in/out projections and time embed, which are small and precision-sensitive).
    """
    def predicate(path: str, module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        return not any(s in path for s in keep_hi_precision)

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    return model


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


# ---------------------------------------------------------------------------
# FLUX.2 VAE (lifted from mflux: mflux.models.flux2.model.flux2_vae.Flux2VAE)
# ---------------------------------------------------------------------------

def load_vae(weights_dir, dtype=mx.float32):
    """Instantiate mflux's Flux2VAE and load the Lens diffusers VAE safetensors.

    Mapping (derived empirically): keys are diffusers-identical except
      - `to_out.0.` -> `to_out.` (mid-block attention out-projection ModuleList),
      - drop `bn.num_batches_tracked` (mflux's Flux2BatchNormStats has no counter),
      - 4D Conv weights transpose PT [O,I,kH,kW] -> MLX [O,kH,kW,I].
    The `bn` running stats (the T1 latent de-norm) ride along in the same file.
    """
    from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE

    vae = Flux2VAE()
    state = {}
    for f in sorted(glob.glob(str(Path(weights_dir) / "*.safetensors"))):
        for k, v in mx.load(f).items():
            if k.endswith("num_batches_tracked"):
                continue
            k = k.replace(".to_out.0.", ".to_out.")
            if v.ndim == 4:  # conv weight: PT (O,I,kH,kW) -> MLX (O,kH,kW,I)
                v = v.transpose(0, 2, 3, 1)
            state[k] = v.astype(dtype)
    mx.eval(state)
    vae.load_weights(list(state.items()), strict=True)
    return vae
