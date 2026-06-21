# lens-mlx

Apple MLX port of [`microsoft/Lens`](https://github.com/microsoft/Lens) — a 3.8B
GPT-OSS-conditioned text-to-image DiT with a FLUX.2 VAE decoder, for inference on
Apple Silicon.

> **Status:** Phases 0–3 ✅ — **the port is functionally complete and generates images.**
> Parity vs the PT reference: encoder cosine 0.998 · DiT cosine 0.999999 · VAE 57.65 dB ·
> full e2e image PSNR 45.26 dB. 14/14 tests green. End-to-end `generate()` produces a
> 1024×1024 image in ~33 s (DiT bf16, 20 steps, 38.8 GB peak) on Apple Silicon.
> Published to mlx-community — [collection](https://huggingface.co/collections/mlx-community/lens-38b-mlx-6a1c6846ca63123d871450f1):
> [**Lens-3.8B-bf16**](https://huggingface.co/mlx-community/Lens-3.8B-bf16) ·
> [**-4bit**](https://huggingface.co/mlx-community/Lens-3.8B-4bit) (2.35 GB) ·
> [**-8bit**](https://huggingface.co/mlx-community/Lens-3.8B-8bit) (4.39 GB), plus the distilled
> **Lens-Turbo** (4-step / cfg 1.0): [Turbo-bf16](https://huggingface.co/mlx-community/Lens-Turbo-3.8B-bf16) ·
> [-4bit](https://huggingface.co/mlx-community/Lens-Turbo-3.8B-4bit) · [-8bit](https://huggingface.co/mlx-community/Lens-Turbo-3.8B-8bit).
> Load a converted repo via `LensPipeline.from_pretrained(base, dit_repo="mlx-community/Lens-3.8B-4bit")`
> (Turbo: `dit_repo="mlx-community/Lens-Turbo-3.8B-bf16", num_inference_steps=4, guidance_scale=1.0`).
> Convert a variant with `recipes/convert_lens.py --src <transformer/> --bits {4,8}`.
> See per-phase docs under `docs/`.

bf16 · int4 (same prompt/seed — int4 perturbs the trajectory into a different, equally sharp image):

<p float="left">
  <img src="assets/sample_lake.png" width="45%" />
  <img src="assets/sample_int4.png" width="45%" />
</p>

```python
import mlx.core as mx
from lens_mlx.pipeline_mlx import LensPipeline

pipe = LensPipeline.from_pretrained("weights/Lens", dit_dtype=mx.bfloat16)
img = pipe("A serene lake below snow-capped mountains, golden hour.",
           height=1024, width=1024, num_inference_steps=20, seed=42)
img.save("out.png")
```

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
  scaffolding, then this standalone fork → `xocialize/lens-mlx` Swift mirror.

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
