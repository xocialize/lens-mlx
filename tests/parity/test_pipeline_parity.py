"""Phase 3 gate — denoise loop (scheduler + norm-rescaled CFG) + full e2e vs golden.

MLX RNG is not torch-seed-compatible, so we inject the golden initial latent
(`dit_in_hidden[:1]`, which is the seed-42 noise the reference used) and the golden
encoder features, then run the MLX denoise loop and compare to `final_latent`, and the
full decode to `decoded_image`. This isolates scheduler/CFG/loop correctness.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
GOLDENS = ROOT / "goldens" / "lens_goldens.npz"
DIT = ROOT / "weights" / "Lens" / "transformer"
VAE_DIR = ROOT / "weights" / "Lens" / "vae"

pytestmark = pytest.mark.skipif(
    not GOLDENS.exists() or not DIT.exists(), reason="needs goldens + transformer weights",
)

STEPS = 4          # matches capture_goldens.py
GUIDANCE = 4.0


def _inputs(g):
    import mlx.core as mx
    latents = mx.array(g["dit_in_hidden"][:1].astype("float32"))   # initial noise [1,1024,128]
    enc = []
    for i in range(4):
        pos = g[f"text_feat_{i}"].astype("float32")
        enc.append(mx.array(np.concatenate([pos, np.zeros_like(pos)], axis=0)))
    pos_mask = g["text_mask"].astype("int32")
    mask = mx.array(np.concatenate([pos_mask, np.zeros_like(pos_mask)], axis=0))
    return latents, enc, mask


def test_denoise_loop_parity():
    import mlx.core as mx
    from lens_mlx.model.transformer import LensTransformer2DModel
    from lens_mlx.utils.weights import load_dit_weights
    from lens_mlx.pipeline_mlx import denoise

    g = np.load(GOLDENS)
    latents, enc, mask = _inputs(g)
    model = LensTransformer2DModel()
    load_dit_weights(model, DIT)

    final = denoise(model, latents, enc, mask, [(1, 32, 32)], STEPS, GUIDANCE)
    mx.eval(final)
    out = np.array(final).astype("float64")
    ref = g["final_latent"].astype("float64")
    max_abs = float(np.abs(out - ref).max())
    cosv = float((out.flatten() @ ref.flatten()) / (np.linalg.norm(out) * np.linalg.norm(ref) + 1e-9))
    print(f"\n[denoise {STEPS}-step] max_abs={max_abs:.3e}  cosine={cosv:.6f}  "
          f"out|max|={np.abs(out).max():.3f} ref|max|={np.abs(ref).max():.3f}")
    # Gate on trajectory direction (cosine); abs error accumulates over the 4 Euler
    # steps so it's a loose diagnostic, not the gate. The e2e image PSNR is the
    # quality gate (test_e2e_image_parity).
    assert cosv >= 0.999, f"final latent cosine too low: {cosv:.6f}"
    assert max_abs < 0.3, f"final latent max_abs grossly high: {max_abs:.3e}"


@pytest.mark.skipif(not VAE_DIR.exists(), reason="needs vae weights")
def test_e2e_image_parity():
    import mlx.core as mx
    from lens_mlx.model.transformer import LensTransformer2DModel
    from lens_mlx.utils.weights import load_dit_weights, load_vae
    from lens_mlx.pipeline_mlx import denoise
    from tests.parity.test_vae_parity import pack_latents_for_decode

    g = np.load(GOLDENS)
    latents, enc, mask = _inputs(g)
    model = LensTransformer2DModel()
    load_dit_weights(model, DIT)
    vae = load_vae(VAE_DIR)

    final = denoise(model, latents, enc, mask, [(1, 32, 32)], STEPS, GUIDANCE)
    img = vae.decode_packed_latents(pack_latents_for_decode(final, 32, 32))
    mx.eval(img)
    img = np.array(img).astype("float64")
    ref = g["decoded_image"].astype("float64")
    if img.shape != ref.shape and img.shape == (ref.shape[0], ref.shape[2], ref.shape[3], ref.shape[1]):
        img = np.transpose(img, (0, 3, 1, 2))
    mse = float(np.mean((img - ref) ** 2))
    peak = float(max(np.abs(ref).max(), np.abs(img).max()))
    psnr = 10 * np.log10(peak ** 2 / (mse + 1e-12))
    print(f"\n[e2e image] PSNR={psnr:.2f} dB  MSE={mse:.3e}  max_abs={np.abs(img-ref).max():.3e}")
    assert psnr >= 30.0, f"e2e image PSNR too low: {psnr:.2f} dB"
