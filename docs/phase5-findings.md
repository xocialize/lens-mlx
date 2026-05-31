# Phase 5 — Publish to mlx-community

**Status: naming reserved + bf16 published.**

## Reserved repos (mlx-community)

Naming matches the house pattern (`Lance-3B-bf16`): **`<Name>-<size>-<quant>`**, using
microsoft's official "3.8B" branding.

| repo | status |
|---|---|
| [`mlx-community/Lens-3.8B-bf16`](https://huggingface.co/mlx-community/Lens-3.8B-bf16) | **published** — DiT bf16 (8.2 GB) + card + sample |
| [`mlx-community/Lens-3.8B-4bit`](https://huggingface.co/mlx-community/Lens-3.8B-4bit) | reserved (placeholder card) — int4 weights next |
| [`mlx-community/Lens-3.8B-8bit`](https://huggingface.co/mlx-community/Lens-3.8B-8bit) | reserved (placeholder card) |

## What ships in the repo (§8-safe)

**DiT only** (MIT). The model card directs the loader to pull the GPT-OSS-20B encoder
(Apache-2.0; reuse `mlx-community/gpt-oss-20b-MXFP4-*`) and the FLUX.2 VAE (license
unverified — **not re-hosted**) from their own sources.

## Conversion recipe — `recipes/convert_lens.py`

```
python recipes/convert_lens.py --dtype bf16 --out build/Lens-3.8B-bf16
python recipes/convert_lens.py --dtype bf16 --bits 4 --out build/Lens-3.8B-4bit
```

- Sanitized mlx keys, sharded at ~5 GB, `model.safetensors.index.json` for multi-shard.
- **Every tensor `mx.eval`'d before save** (lazy tensors serialize as zeros — the silent killer).
- Post-save verification: reload the saved repo and re-run DiT parity → cosine 0.999999
  (confirms materialization). Always do this before upload.

## Skill note (F12)
mlx-community image-model naming is **per-quant**: `<Name>[-<size>]-<quant>` with suffixes
`-bf16 / -4bit / -8bit` (e.g. `Qwen-Image-2512-4bit`, `Lance-3B-bf16`). Reserve the family
names early (create_repo + placeholder card) — generic names get taken. Verify materialization
by reloading + re-parity before pushing 8 GB.

## Remaining
- Push int4 (+ int8) weights to the reserved repos.
- Publish the `xocialize-code/lens-mlx` code repo (the model card references it) + Swift mirror.
