# lens-mlx

Apple MLX port of [`microsoft/Lens`](https://github.com/microsoft/Lens) — a 3.8B
GPT-OSS-conditioned text-to-image DiT with a FLUX.2 VAE decoder, for inference on
Apple Silicon.

> **Status:** Phases 0–3 ✅ — **the port is functionally complete and generates images.**
> Parity vs the PT reference: encoder cosine 0.998 · DiT cosine 0.999999 · VAE 57.65 dB ·
> full e2e image PSNR 45.26 dB. 16/16 tests green (6 parity + 2 smoke files; see `tests/`).
> End-to-end `generate()` produces a 1024×1024 image in ~33 s (DiT bf16, 20 steps,
> 38.8 GB peak) on Apple Silicon.
> Published to mlx-community — [collection](https://huggingface.co/collections/mlx-community/lens-38b-mlx-6a1c6846ca63123d871450f1):
> [**Lens-3.8B-bf16**](https://huggingface.co/mlx-community/Lens-3.8B-bf16) ·
> [**-4bit**](https://huggingface.co/mlx-community/Lens-3.8B-4bit) (2.35 GB) ·
> [**-8bit**](https://huggingface.co/mlx-community/Lens-3.8B-8bit) (4.39 GB). Load a converted
> repo via `LensPipeline.from_pretrained(base, dit_repo="mlx-community/Lens-3.8B-4bit")`.
> See per-phase docs under `docs/`. Next: Swift mirror.

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

`__call__` signature: `pipe(prompt, height=1024, width=1024, num_inference_steps=20,
guidance_scale=4.0, seed=0)`. For a no-frills CLI smoke run:

```bash
python scripts/generate.py "A serene lake..." 20 1024 0 out.png
#                           prompt          steps size bits(0=bf16) outfile
```

## Pipeline

```
GPT-OSS-20B multi-layer text features ([5,11,17,23])
  → Lens DiT (48-layer double-stream flow-matching)
  → FLUX.2 VAE decode
```

## Scope (v1)

- **Variant:** Lens (RL-tuned, 20-step) only. Turbo / Base deferred.
- **Precision:** bf16 first; int4/int8 DiT quantization landed (see collection above).
- **Strategy:** validate DiT parity against `mflux`'s FLUX.2 VAE + flow-match
  scaffolding, then this standalone fork → `xocialize-code/lens-mlx` Swift mirror.

## Layout

```
lens_mlx/
├── model/transformer.py     # LensTransformer2DModel  (Phase 2 — the bulk)
├── model/text_encoder.py    # gpt_oss capture wrapper  (Phase 1)
├── pipeline_mlx.py          # from_pretrained, denoise, CFG, decode (Phase 3)
├── scheduler.py             # flow-matching scheduler
├── resolution.py            # ported verbatim from upstream (pure Python)
└── utils/weights.py         # split-safetensors load, HF fetch, DiT quantize
recipes/convert_lens.py      # per-component conversion recipe (bf16 / 4-bit / 8-bit)
scripts/generate.py          # full text→image smoke generate (visual check)
tests/{parity,smoke}/        # PT is a [parity] dev-only extra
refs/Lens/                   # reference oracle (depth-1 clone)
refs/configs/                # checkpoint configs (reconciled in Phase 0)
```

`from_pretrained(repo_dir, dit_repo=None, dit_dtype=mx.float32, ...)` — pass `dit_repo` to
load an already-converted (bf16/quantized) DiT repo; `dit_dtype` is then ignored for the DiT.

## Dev setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate   # repo targets py3.12; pyproject allows >=3.10
uv pip install -e ".[dev]"     # mlx, mlx-lm + parity (torch/transformers/diffusers) + mflux
pytest tests/smoke
```

Optional extras (per `pyproject.toml`): `[mflux]` (FLUX.2 VAE + flow-match scaffolding),
`[parity]` (torch/transformers/diffusers/einops/accelerate — DEV-ONLY; end users on MLX
never need torch), `[dev]` (= parity + mflux + pytest + ruff).

## License

Lens code/weights MIT; GPT-OSS-20B Apache-2.0. The **FLUX.2 VAE** weights license is
unverified — `from_pretrained` pulls the VAE from its original source rather than
re-hosting a bundled copy. See handoff §8.
