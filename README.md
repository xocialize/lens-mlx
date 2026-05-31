# lens-mlx

Apple MLX port of [`microsoft/Lens`](https://github.com/microsoft/Lens) — a 3.8B
GPT-OSS-conditioned text-to-image DiT with a FLUX.2 VAE decoder, for inference on
Apple Silicon.

> **Status:** Phase 0 (capture & scaffold) ✅ · Phase 1 (text encoder) ✅ — encoder
> parity green (bf16 worst-layer cosine 0.9983; a missing-YaRN-rope bug found & fixed).
> DiT and pipeline are next (Phases 2–3). See `docs/phase0-findings.md` (config
> reconciliation + traps T0–T8), `docs/phase1-findings.md` (encoder + the YaRN bug),
> and `Lens-MLX-Port-Handoff.md` (full plan).

## Pipeline

```
GPT-OSS-20B multi-layer text features ([5,11,17,23])
  → Lens DiT (48-layer double-stream flow-matching)
  → FLUX.2 VAE decode
```

## Scope (v1)

- **Variant:** Lens (RL-tuned, 20-step) only. Turbo / Base deferred.
- **Precision:** bf16 first; quantization (int4 DiT) deferred to a later pass.
- **Strategy:** validate DiT parity against `mflux`'s FLUX.2 VAE + flow-match
  scaffolding, then this standalone fork → `xocialize-code/lens-mlx` Swift mirror.

## Layout

```
lens_mlx/
├── model/transformer.py     # LensTransformer2DModel  (Phase 2 — the bulk)
├── model/text_encoder.py    # gpt_oss capture wrapper  (Phase 1)
├── pipeline_mlx.py          # from_pretrained, denoise, CFG, decode (Phase 3)
├── resolution.py            # ported verbatim from upstream (pure Python)
└── utils/weights.py         # split-safetensors load, HF fetch
recipes/convert_lens.py      # per-component conversion recipe
tests/{parity,smoke}/        # PT is a [parity] dev-only extra
refs/Lens/                   # reference oracle (depth-1 clone)
refs/configs/                # checkpoint configs (reconciled in Phase 0)
```

## Dev setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"     # mlx, mlx-lm, mflux + parity (torch/transformers/diffusers)
pytest tests/smoke
```

## License

Lens code/weights MIT; GPT-OSS-20B Apache-2.0. The **FLUX.2 VAE** weights license is
unverified — `from_pretrained` pulls the VAE from its original source rather than
re-hosting a bundled copy. See handoff §8.
