# Phase 1 — Text encoder (GPT-OSS feature capture)

**Status: GREEN.** MLX `LensGptOssEncoder` wraps mlx-lm `gpt_oss`, captures hidden states
at `[5,11,17,23]` with the T5 chat template + offset-97 slice, and matches the PT golden.

## Result

Per-layer cosine vs the bf16 PT golden (`goldens/lens_goldens.npz`):

| layer | bf16 (correctness gate) | MXFP4 (production) |
|---|---|---|
| L5  | 0.99986 | 0.99989 |
| L11 | 0.99955 | 0.99950 |
| L17 | 0.99874 | 0.99861 |
| L23 | 0.99834 | 0.99774 |

Gate: bf16 worst-layer cosine ≥ 0.998 (a structural bug gives ~0.94; the residual is bf16/
4-bit accumulation + minor YaRN-ramp differences over 24 layers). The MXFP4 production path is
essentially as good as bf16 — the quant gap is negligible.

## The bug we found (and why the rigorous gate mattered)

Initial MXFP4-vs-bf16 parity sat at cosine ~0.95 — temptingly dismissable as "quant noise."
The bf16-vs-bf16 diagnostic (matched precision) held at **0.94**, proving a *forward* bug:

- divergence uniform from layer 0; MLX activation magnitude ~half the reference;
- all weights matched exactly (attn + experts, cos=1.0) → pure compute difference;
- **root cause:** GPT-OSS-20B uses **YaRN rope** (`attention_scaling`/mscale ≈ 1.3466, factor 32,
  original_max_position 4096), which is an HF `GptOssConfig` *class default*. The checkpoint's
  `text_encoder/config.json` does NOT serialize `rope_scaling`/`rope_theta`, so mlx-lm fell back
  to plain rope (mscale 1.0). The missing mscale skews cos/sin at every layer/position.

**Fix:** `LensGptOssEncoder.from_pretrained` injects `GPT_OSS_YARN_ROPE` (via mlx-lm
`load_model(..., model_config=...)`) when the on-disk config omits `rope_scaling`. Validated by
deleting `rope_scaling` from the on-disk configs and confirming parity still holds via the loader.

General lesson: never trust `config.json` for rope — compare the **resolved** rope
on both sides (`pt.model.rotary_emb.attention_scaling` + `inv_freq` vs the mlx rope object).

## Artifacts

- `lens_mlx/model/text_encoder.py` — the capture wrapper (isomorphic to upstream).
- `lens_mlx/pipeline_mlx.py` — `build_chat_inputs` (T5) + `compute_empirical_mu` (T6) ported verbatim.
- `tests/parity/test_encoder_parity.py` — bf16 gate + MXFP4 informational.
- `tests/parity/make_bf16_encoder.py` — `mlx_lm.convert(dequantize=True)` → dense bf16 mlx encoder.
- `tests/parity/diag_*.py` — layer-by-layer / weight / expert / rope diagnostics (kept for reuse).

## Scratch weights (gitignored)
`weights/Lens-encoder-mlx-bf16/` (39 GB dense bf16) — diagnostic only; production uses the MXFP4
`weights/Lens/text_encoder/`.
