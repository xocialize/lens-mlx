# Phase 3 — VAE + scheduler + e2e

**Status: GREEN — port functionally complete.** The full pipeline (encoder → DiT →
scheduler/CFG → VAE) reproduces the PT reference image, and `LensPipeline.from_pretrained`
generates real images from raw text: **1024×1024, 20 steps, ~33 s, 38.8 GB peak** (DiT
bf16) on Apple Silicon. See `assets/sample_lake.png`.

## Results

| gate | metric | threshold |
|---|---|---|
| VAE decode (T1 bn de-norm + T4 unpatchify) | PSNR **57.65 dB** | ≥ 40 ✅ |
| denoise loop (scheduler + norm-rescaled CFG) | cosine **0.999805** | ≥ 0.999 ✅ |
| full e2e image | PSNR **45.26 dB** | ≥ 30 ✅ |
| checkerboard noise-path (T4, 1024px) | nyquist ratio 0.35 / 0.30 | < 5 ✅ |

## How it went

- **VAE lifted from mflux, no re-port.** `mflux.models.flux2...Flux2VAE.decode_packed_latents`
  already does the exact Lens decode: `latents = packed*bn_std + bn_mean` (T1, eps 1e-4) →
  `_unpatchify_latents` (T4, identical permute) → post_quant_conv → decoder. Loaded the Lens
  diffusers VAE weights into it: **246 exact key matches**, only `to_out.0.`→`to_out.` (4 keys),
  drop `bn.num_batches_tracked`, and the standard 4D Conv transpose [O,I,kH,kW]→[O,kH,kW,I] (63).
- **Scheduler** (`lens_mlx/scheduler.py`): exponential time-shift(mu) + sigmas=linspace(1,1/N,N).
  Sanity: sigmas[0]=1 → shift=1 → timestep 1000 → DiT sees 1.0 (== golden dit_in_timestep). T6
  `compute_empirical_mu` ported verbatim.
- **Norm-rescaled CFG** (F4, NOT vanilla): `comb = uncond + g*(cond-uncond)`, then per-token
  `*= ||cond||/||comb||`. In `pipeline_mlx.lens_cfg` / `denoise`.
- **RNG caveat:** MLX RNG isn't torch-seed-compatible, so e2e parity injects the golden initial
  latent (`dit_in_hidden[:1]` = the seed-42 noise) + golden encoder features to isolate the
  loop. A real `generate()` (Phase 3e) uses fresh MLX noise (no golden to diff — visual check).
- The integrated 4-step latent `max_abs` (0.18) accumulates per-step DiT diffs; cosine 0.9998 +
  the 45 dB image are the real gates, not abs latent error.

## Artifacts
`lens_mlx/scheduler.py`, `pipeline_mlx.py` (`lens_cfg`, `denoise`), `utils/weights.py`
(`load_vae`), `tests/parity/test_vae_parity.py`, `test_pipeline_parity.py`,
`tests/smoke/test_vae_noise_path_smoke.py`.

## Skill note (F10)
mflux is a strong VAE oracle to *lift* (not re-port): its FLUX.2 VAE matched the Lens diffusers
VAE at 246/250 keys with only conv-transpose + a ModuleList-index rename, and its
`decode_packed_latents` already encoded the model-specific bn de-norm + unpatchify. Always check
whether an mlx-ecosystem repo already ships the component before porting it.
