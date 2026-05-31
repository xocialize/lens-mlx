"""Phase 1 diagnostic — produce a matched-precision bf16 MLX encoder.

The shippable MLX encoder runs true MXFP4; our only Apple-runnable PT reference is
bf16-dequantized. To gate forward *correctness* (not the quant gap) we need bf16 on
BOTH sides.

Cleanest route: mlx_lm.convert(dequantize=True) loads the HF MXFP4 gpt-oss (mlx-lm's
sanitize handles the MXFP4 blocks), runs `dequantize_model` in MLX, and saves a dense
bf16 mlx model. No PyTorch round-trip (HF's MXFP4 save_pretrained drops dequantized
expert weights), no expert-layout guesswork.

  weights/Lens/text_encoder (+ tokenizer copied in)  --convert(dequantize)-->
  weights/Lens-encoder-mlx-bf16   (dense bf16 mlx gpt_oss)
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "weights" / "Lens" / "text_encoder"
TOK = ROOT / "weights" / "Lens" / "tokenizer"
MLX_BF16 = ROOT / "weights" / "Lens-encoder-mlx-bf16"


def main() -> None:
    # convert's loader needs a tokenizer next to the weights.
    for f in TOK.iterdir():
        dst = SRC / f.name
        if not dst.exists():
            shutil.copy(f, dst)

    if MLX_BF16.exists():
        shutil.rmtree(MLX_BF16)

    from mlx_lm import convert
    print(f"[bf16] convert(dequantize=True) {SRC} -> {MLX_BF16}")
    convert(str(SRC), mlx_path=str(MLX_BF16), dequantize=True, dtype="bfloat16")
    print("[bf16] done.")


if __name__ == "__main__":
    main()
