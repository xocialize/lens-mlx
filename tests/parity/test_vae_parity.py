"""Phase 3 gate — FLUX.2 VAE decode parity vs the PT golden.

Packs the golden `final_latent` (Lens rearrange + patchify -> [b, 128, 32, 32]) and
decodes via mflux's Flux2VAE.decode_packed_latents (T1 bn de-norm + T4 unpatchify +
decode), comparing to the golden `decoded_image` by PSNR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "goldens" / "lens_goldens.npz"
VAE_DIR = ROOT / "weights" / "Lens" / "vae"

pytestmark = pytest.mark.skipif(
    not GOLDENS.exists() or not VAE_DIR.exists(), reason="needs goldens + vae weights",
)


def pack_latents_for_decode(latents, latent_h, latent_w):
    """[b, h*w, c*4] -> packed [b, c*4, h, w] (Lens _decode: rearrange then patchify)."""
    import mlx.core as mx
    b = latents.shape[0]
    c = latents.shape[2] // 4
    # rearrange "b (h w) (c p1 p2) -> b c (h p1) (w p2)"
    x = latents.reshape(b, latent_h, latent_w, c, 2, 2).transpose(0, 3, 1, 4, 2, 5)
    x = x.reshape(b, c, latent_h * 2, latent_w * 2)
    # _patchify_latents: [b,c,H,W] -> [b,c*4,H//2,W//2]
    H, W = latent_h * 2, latent_w * 2
    x = x.reshape(b, c, H // 2, 2, W // 2, 2).transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, H // 2, W // 2)


def test_vae_decode_parity():
    import mlx.core as mx
    from lens_mlx.utils.weights import load_vae

    g = np.load(GOLDENS)
    latent = mx.array(g["final_latent"].astype("float32"))   # [1,1024,128]
    vae = load_vae(VAE_DIR)

    packed = pack_latents_for_decode(latent, 32, 32)          # [1,128,32,32]
    img = vae.decode_packed_latents(packed)
    mx.eval(img)
    img = np.array(img).astype("float64")

    ref = g["decoded_image"].astype("float64")                # [1,3,512,512] NCHW
    # mflux decoder may emit NHWC; align to NCHW for comparison.
    if img.shape != ref.shape:
        if img.shape == (ref.shape[0], ref.shape[2], ref.shape[3], ref.shape[1]):
            img = np.transpose(img, (0, 3, 1, 2))
    assert img.shape == ref.shape, f"shape mismatch img={img.shape} ref={ref.shape}"

    mse = float(np.mean((img - ref) ** 2))
    peak = float(max(np.abs(ref).max(), np.abs(img).max()))
    psnr = 10 * np.log10(peak ** 2 / (mse + 1e-12))
    max_abs = float(np.abs(img - ref).max())
    print(f"\n[VAE decode] PSNR={psnr:.2f} dB  MSE={mse:.3e}  max_abs={max_abs:.3e}  "
          f"img|max|={np.abs(img).max():.3f} ref|max|={np.abs(ref).max():.3f}")
    assert psnr >= 40.0, f"VAE decode PSNR too low: {psnr:.2f} dB"
