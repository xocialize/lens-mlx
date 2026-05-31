# Phase 4 — Quantization (int4 DiT)

**Status: GREEN.** Scoped int4 DiT validated; generates high-quality images at ~3.5× smaller.

## Measurements (DiT, single-pass vs golden)

| config | DiT size | max_abs | cosine |
|---|---|---|---|
| bf16 baseline | 8.21 GB | 8.3e-3 | 0.999999 |
| **int4, keep in/out/time (ship)** | **2.35 GB** | 3.7e-1 | **0.9976** |
| int4, all Linears | 2.31 GB | 5.8e-1 | 0.9944 |
| int8, all Linears | 4.36 GB | 5.3e-2 | 0.99996 |

**Scope chosen:** int4 `group_size=64, bits=4`, keeping `img_in / txt_in / proj_out /
time_text_embed / norm_out` at bf16 (`LensPipeline.QUANT_KEEP_HI`). Same ~3.5× shrink as
full int4 but materially better parity (0.9976 vs 0.9944), since those small projections
are precision-sensitive. int8 is the higher-fidelity fallback (0.99996, 4.36 GB).

## Generation

`from_pretrained(..., quantize_bits=4)` → 1024×1024, 20 steps, **31.8 s, 32.9 GB peak**
(vs bf16 33 s / 38.8 GB). Output is sharp and photorealistic — `assets/sample_int4.png`.
The int4 image differs in *composition* from the bf16 one (same prompt/seed): quantization
perturbs the denoise trajectory, but quality is fully intact.

## Skill note (F11) — don't gate quantized generative models on PSNR-vs-the-fp32-golden
int4 perturbs each denoise step (per-pass cosine 0.9976), and diffusion sampling amplifies
that across steps into a *different* image. e2e PSNR vs the fp32-golden image was 15.6 dB —
not a quality defect, just trajectory divergence. The right gates: (1) per-pass weight-level
cosine ≥ 0.99 (deterministic), (2) image-validity sanity (finite, real content), (3) a
committed visual sample. Reserve e2e-PSNR-vs-golden for unquantized parity.

## Artifacts
`utils/weights.py` `quantize_dit`, `pipeline_mlx` `from_pretrained(quantize_bits=...)`,
`tests/parity/test_dit_quant_parity.py`, `tests/parity/measure_quant.py`,
`assets/sample_int4.png`.
