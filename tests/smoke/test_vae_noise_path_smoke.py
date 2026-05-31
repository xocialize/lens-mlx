"""Checkerboard / noise-path smoke (skill pitfall #7, trap T4).

Decode random Gaussian through the FULL post-DiT chain (bn de-norm -> unpatchify ->
VAE.decode) at a production-ish resolution and assert the image has no dominant
period-2 (stride-2) structure — the signature of a patchify/tile/pixel-shuffle axis
bug that small-scale layer parity can miss. Runs at 1024px (latent 64x64), larger
than the 512px golden.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
VAE_DIR = ROOT / "weights" / "Lens" / "vae"

pytestmark = pytest.mark.skipif(not VAE_DIR.exists(), reason="needs vae weights")


def _nyquist_ratio(plane: np.ndarray) -> float:
    """Energy at the period-2 (Nyquist) frequency vs the mid-band, averaged over rows.

    A stride-2 checkerboard concentrates huge energy at the Nyquist bin -> ratio >> 1.
    """
    spec = np.abs(np.fft.rfft(plane - plane.mean(axis=1, keepdims=True), axis=1))
    nyq = spec[:, -1].mean()
    mid = spec[:, 1:-1].mean() + 1e-9
    return float(nyq / mid)


def test_noise_path_no_checkerboard():
    import mlx.core as mx
    from lens_mlx.utils.weights import load_vae

    vae = load_vae(VAE_DIR)
    latent_h = latent_w = 64                       # 1024px output
    rng = np.random.default_rng(0)
    packed = mx.array(rng.standard_normal((1, 128, latent_h, latent_w)).astype("float32"))
    img = vae.decode_packed_latents(packed)
    mx.eval(img)
    img = np.array(img).astype("float64")
    # to NCHW if needed
    if img.shape[1] != 3 and img.shape[-1] == 3:
        img = np.transpose(img, (0, 3, 1, 2))

    assert np.isfinite(img).all(), "non-finite values in decoded image"
    # Check both axes on the luminance plane.
    lum = img[0].mean(axis=0)                       # [H, W]
    r_row = _nyquist_ratio(lum)
    r_col = _nyquist_ratio(lum.T)
    print(f"\n[noise-path] shape={img.shape} nyquist_ratio row={r_row:.2f} col={r_col:.2f}")
    assert r_row < 5.0 and r_col < 5.0, f"period-2 structure (checkerboard): row={r_row:.2f} col={r_col:.2f}"
