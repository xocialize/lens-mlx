"""Lens denoising transformer (DiT) — MLX port.

Isomorphic to `refs/Lens/lens/transformer.py`. Class/method names and forward
call-order are kept identical to upstream; only PyTorch ops are swapped for MLX.

Phase 2 work (each gets a parity test before the block test):
  - apply_rotary_emb_lens  -> T2: reimplement torch.view_as_complex as real
                                  interleaved rotation. scale_rope=True path.
  - LensJointAttention     -> stacked QKV view(B,S,3,H,Dh).unbind(2) (NOT
                                  interleaved); QK-RMSNorm eps=1e-5; concat
                                  [img,txt] -> mx.fast.scaled_dot_product_attention.
  - LensTransformerBlock   -> T3 AdaLN x*(1+scale)+shift, gate on residual;
                                  6-way mod per stream; GateMLP SwiGLU hidden=dim/3*8.
  - LensTransformer2DModel -> time_proj scale=1000; multi-layer txt_norm[i] + txt_in;
                                  returns proj_out(norm_out(h, temb)).

Epsilons: QK-norm 1e-5, block norms 1e-6, txt_norm 1e-5, norm_out 1e-6.
"""

from __future__ import annotations

# TODO(phase-2): import mlx.core as mx, mlx.nn as nn and translate the modules.
# Upstream classes to mirror (same names):
#   get_timestep_embedding, apply_rotary_emb_lens, GateMLP,
#   LensTimestepProjEmbeddings, LensEmbedRope, LensJointAttention,
#   LensTransformerBlock, LensTransformer2DModel

raise NotImplementedError("Phase 2: port LensTransformer2DModel from refs/Lens/lens/transformer.py")
