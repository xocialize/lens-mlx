"""Diagnostic — locate where the MLX encoder forward diverges from PT (bf16 both sides).

Runs the SAME input_ids through the PT reference (all 24 layers via
set_selected_layers(range(24))) and the MLX wrapper, comparing cosine at every layer
to find the divergence onset. Also checks embedding + a couple of raw weights to
separate 'weights differ' from 'compute differs'.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "refs" / "Lens"))
PROMPT = "A scenic landscape with a serene lake and snow-capped mountains."


def cos(a, b):
    a = a.astype("float64").flatten(); b = b.astype("float64").flatten()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    import torch
    import mlx.core as mx
    from transformers import AutoTokenizer, Mxfp4Config
    from lens import LensGptOssEncoder as PTEnc
    from lens_mlx.model.text_encoder import LensGptOssEncoder as MXEnc
    from lens_mlx.pipeline_mlx import build_chat_inputs

    tok = AutoTokenizer.from_pretrained(str(ROOT / "weights" / "Lens" / "tokenizer"))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    input_ids, attn = build_chat_inputs(tok, [PROMPT])
    print("input_ids:", input_ids.shape)

    # --- PT reference, all 24 layers ---
    pt = PTEnc.from_pretrained(
        str(ROOT / "weights" / "Lens" / "text_encoder"),
        dtype=torch.bfloat16, quantization_config=Mxfp4Config(dequantize=True),
    )
    pt.set_selected_layers(list(range(24)))
    pt_layers = pt(input_ids=torch.tensor(input_ids), attention_mask=torch.tensor(attn))
    pt_layers = [h.detach().float().numpy() for h in pt_layers]
    pt_embed = pt.model.embed_tokens.weight.detach().float().numpy()

    # --- MLX, all 24 layers ---
    mxenc = MXEnc.from_pretrained(str(ROOT / "weights" / "Lens-encoder-mlx-bf16"),
                                  selected_layers=list(range(24)))
    mx_embed = np.array(mxenc.model.model.embed_tokens.weight.astype(mx.float32)).astype("float64")
    mx_layers = mxenc(mx.array(input_ids))
    mx_layers = [np.array(h.astype(mx.float32)) for h in mx_layers]

    print(f"\nembed_tokens.weight cosine (PT-dequant vs MLX-dequant): {cos(pt_embed, mx_embed):.6f}"
          f"  | shapes {pt_embed.shape} {mx_embed.shape}")
    print("\nlayer  cosine    pt_absmax   mx_absmax")
    for i in range(24):
        c = cos(pt_layers[i], mx_layers[i])
        print(f" {i:2d}   {c:.6f}   {np.abs(pt_layers[i]).max():9.1f}  {np.abs(mx_layers[i]).max():9.1f}")


if __name__ == "__main__":
    main()
