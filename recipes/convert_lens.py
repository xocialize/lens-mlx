"""mlx-forge-style conversion recipe for Lens (per-component).

Convention follows ~/DEV_INT/longcat-avatar-mlx/recipes/convert_longcat_avatar.py:
a self-contained script (NOT an external mlx-forge dependency).

Per-component split safetensors so each loads / quantizes independently:
  - transformer/  (the DiT — 3.8B; bf16 first, int4 later)
  - vae/          (FLUX.2 — pulled from source; see §8 license)
  - text_encoder/ (GPT-OSS-20B MXFP4 — reuse existing mlx-community weights)

THE SILENT KILLER (skill rule): MLX is lazy. Call mx.eval(weight) on EVERY tensor
immediately before mx.save_safetensors — unevaluated tensors serialize as zeros
with no error.

Transpose conventions: PyTorch Conv (O,I,*K) -> MLX (O,*K,I); Linear/Embedding identical.
"""

from __future__ import annotations

# TODO(phase-3/5): classify_key / sanitize_key / transform / convert per component.

raise NotImplementedError("Phase 3/5: implement Lens per-component conversion recipe")
