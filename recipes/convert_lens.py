"""Convert the Lens DiT to an mlx-community-ready repo (DiT-only, §8-safe).

Saves sanitized, MATERIALIZED mlx safetensors (sharded) + config.json. The DiT is
MIT (clean to host); the GPT-OSS encoder (Apache-2.0, reuse mlx-community) and the
FLUX.2 VAE (license unverified — do NOT re-host) are pulled from source by the loader.

Usage:
    python recipes/convert_lens.py --dtype bf16 --out build/Lens-3.8B-bf16
    python recipes/convert_lens.py --dtype bf16 --bits 4 --out build/Lens-3.8B-4bit
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "weights" / "Lens" / "transformer"
SHARD_BYTES = 5_000_000_000  # ~5GB shards (mlx-community convention)

DTYPES = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}


def _save_sharded(out: Path, state: dict, extra_metadata: dict):
    items = list(state.items())
    shards, cur, cur_bytes = [], {}, 0
    for k, v in items:
        if cur and cur_bytes + v.nbytes > SHARD_BYTES:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[k] = v; cur_bytes += v.nbytes
    if cur:
        shards.append(cur)
    n = len(shards)
    index = {"metadata": {"total_size": sum(v.nbytes for v in state.values()), **extra_metadata},
             "weight_map": {}}
    for i, shard in enumerate(shards, 1):
        name = "model.safetensors" if n == 1 else f"model-{i:05d}-of-{n:05d}.safetensors"
        mx.save_safetensors(str(out / name), shard)
        for k in shard:
            index["weight_map"][k] = name
    if n > 1:
        (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--bits", type=int, default=None, help="quantize DiT (e.g. 4, 8)")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from lens_mlx.model.transformer import LensTransformer2DModel
    from lens_mlx.utils.weights import load_dit_weights, quantize_dit
    from lens_mlx.pipeline_mlx import LensPipeline

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    model = LensTransformer2DModel()
    load_dit_weights(model, SRC, dtype=DTYPES[args.dtype])
    quant_meta = {}
    if args.bits is not None:
        quantize_dit(model, group_size=args.group_size, bits=args.bits,
                     keep_hi_precision=LensPipeline.QUANT_KEEP_HI)
        quant_meta = {"quantization": json.dumps(
            {"group_size": args.group_size, "bits": args.bits, "keep_hi_precision": LensPipeline.QUANT_KEEP_HI})}

    state = dict(tree_flatten(model.parameters()))
    mx.eval(state)  # MATERIALIZE before save — lazy tensors serialize as zeros (the silent killer).
    _save_sharded(out, state, {"format": "mlx", "model_type": "lens_transformer_2d", **quant_meta})

    cfg = json.loads((SRC / "config.json").read_text())
    cfg["mlx_format"] = True
    if args.bits is not None:
        cfg["quantization"] = {"group_size": args.group_size, "bits": args.bits,
                               "keep_hi_precision": list(LensPipeline.QUANT_KEEP_HI)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    total = sum(v.nbytes for v in state.values())
    print(f"[convert] {args.dtype}{'' if args.bits is None else f' int{args.bits}'} "
          f"-> {out} ({total/1e9:.2f} GB, {len(state)} tensors)")


if __name__ == "__main__":
    main()
