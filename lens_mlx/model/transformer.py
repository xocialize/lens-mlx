"""Lens denoising transformer (DiT) — MLX port.

Isomorphic to `refs/Lens/lens/transformer.py`. Class / method names and the forward
call order match upstream; only PyTorch ops are swapped for MLX. The complex axial
RoPE (T2) is reimplemented as a real interleaved rotation (MLX has no
`view_as_complex`); `LensEmbedRope` emits (cos, sin) instead of a complex tensor.

Epsilons (confirmed from the reference): QK-norm 1e-5, block norms 1e-6,
txt_norm 1e-5, norm_out 1e-6.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Embeddings & RoPE
# ---------------------------------------------------------------------------


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1.0,
    scale: float = 1.0,
    max_period: int = 10000,
) -> mx.array:
    """Sinusoidal timestep embeddings (DDPM-style). Mirrors the reference helper."""
    assert timesteps.ndim == 1, "Timesteps should be 1-D"
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(0, half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = timesteps[:, None].astype(mx.float32) * emb[None, :]
    emb = scale * emb
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


def apply_rotary_emb_lens(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Apply complex-valued RoPE (Lens variant) as a real interleaved rotation.

    Args:
        x:   [B, S, H, D] query or key tensor.
        cos: [S, D/2] cos(freqs); sin: [S, D/2]. (= real/imag of the upstream
             ``freqs_cis`` complex tensor.)

    Upstream pairs (x[...,2i], x[...,2i+1]) as (real, imag), multiplies by
    ``freqs_cis = cos + i*sin``, then re-flattens. Complex product:
        out_r = x_r*cos - x_i*sin ;  out_i = x_r*sin + x_i*cos.
    """
    x_r = x[..., 0::2]
    x_i = x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos
    out = mx.stack([out_r, out_i], axis=-1)
    return out.reshape(x.shape).astype(x.dtype)


class GateMLP(nn.Module):
    """SwiGLU MLP used by the transformer blocks."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, time_embed_dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def __call__(self, sample: mx.array) -> mx.array:
        return self.linear_2(nn.silu(self.linear_1(sample)))


class LensTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        # time_proj = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def __call__(self, timestep: mx.array, hidden_states: mx.array) -> mx.array:
        proj = get_timestep_embedding(
            timestep, 256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000
        )
        return self.timestep_embedder(proj.astype(hidden_states.dtype))


class LensEmbedRope:
    """Frame/H/W axial RoPE shared between image and text streams (emits cos/sin).

    Plain class (NOT nn.Module): it has only computed rope tables, no learnable
    params. As an nn.Module, MLX would collect pos_freqs/neg_freqs as parameters and
    expect them in the checkpoint (upstream stores them as non-persistent buffers).
    """

    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False) -> None:
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope
        pos_index = mx.arange(4096)
        neg_index = mx.arange(4096)[::-1] * -1 - 1
        # angles, concatenated across axes -> [4096, sum(axes_dim)/2]
        self.pos_freqs = mx.concatenate(
            [self._rope_params(pos_index, d, theta) for d in axes_dim], axis=1
        )
        self.neg_freqs = mx.concatenate(
            [self._rope_params(neg_index, d, theta) for d in axes_dim], axis=1
        )

    @staticmethod
    def _rope_params(index: mx.array, dim: int, theta: int = 10000) -> mx.array:
        assert dim % 2 == 0
        freqs = mx.outer(
            index.astype(mx.float32),
            1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim)),
        )
        return freqs  # angle; cos/sin taken at apply time (upstream: polar(1, freqs))

    def __call__(
        self,
        video_fhw: Union[List[Tuple[int, int, int]], Tuple[int, int, int]],
        txt_seq_lens: Union[List[int], int],
    ) -> Tuple[Tuple[mx.array, mx.array], Tuple[mx.array, mx.array]]:
        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]
        if not isinstance(txt_seq_lens, list):
            txt_seq_lens = [txt_seq_lens]
        assert len(video_fhw) == 1, "video_fhw must have length 1"

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            video_freq = self._compute_video_freqs(frame, height, width, idx=0)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)
            vid_freqs.append(video_freq)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid = mx.concatenate(vid_freqs, axis=0)
        return (mx.cos(vid), mx.sin(vid)), (mx.cos(txt_freqs), mx.sin(txt_freqs))

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> mx.array:
        seq_lens = frame * height * width
        splits = [d // 2 for d in self.axes_dim]
        bounds = [sum(splits[:i]) for i in range(len(splits) + 1)]
        fp = [self.pos_freqs[:, bounds[i]:bounds[i + 1]] for i in range(len(splits))]
        fn = [self.neg_freqs[:, bounds[i]:bounds[i + 1]] for i in range(len(splits))]

        freqs_frame = mx.broadcast_to(
            fp[0][idx:idx + frame].reshape(frame, 1, 1, -1), (frame, height, width, splits[0])
        )
        if self.scale_rope:
            freqs_height = mx.broadcast_to(
                mx.concatenate([fn[1][-(height - height // 2):], fp[1][:height // 2]], axis=0)
                .reshape(1, height, 1, -1), (frame, height, width, splits[1])
            )
            freqs_width = mx.broadcast_to(
                mx.concatenate([fn[2][-(width - width // 2):], fp[2][:width // 2]], axis=0)
                .reshape(1, 1, width, -1), (frame, height, width, splits[2])
            )
        else:
            freqs_height = mx.broadcast_to(
                fp[1][:height].reshape(1, height, 1, -1), (frame, height, width, splits[1])
            )
            freqs_width = mx.broadcast_to(
                fp[2][:width].reshape(1, 1, width, -1), (frame, height, width, splits[2])
            )
        freqs = mx.concatenate([freqs_frame, freqs_height, freqs_width], axis=-1).reshape(seq_lens, -1)
        return freqs


# ---------------------------------------------------------------------------
# Attention (joint image + text, plain SDPA)
# ---------------------------------------------------------------------------


class LensJointAttention(nn.Module):
    """Joint image+text attention with fused QKV and SDPA backend."""

    def __init__(
        self,
        query_dim: int,
        added_kv_proj_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        out_dim: Optional[int] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = self.inner_dim // dim_head
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.norm_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps)
        self.norm_added_q = nn.RMSNorm(dim_head, eps=eps)
        self.norm_added_k = nn.RMSNorm(dim_head, eps=eps)

        self.img_qkv = nn.Linear(query_dim, 3 * self.inner_dim, bias=True)
        self.txt_qkv = nn.Linear(added_kv_proj_dim, 3 * self.inner_dim, bias=True)

        # upstream: to_out = ModuleList([Linear, Identity]); index 0 is the Linear.
        self.to_out = [nn.Linear(self.inner_dim, self.out_dim, bias=True)]
        self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        bsz, seq_img, _ = hidden_states.shape
        seq_txt = encoder_hidden_states.shape[1]
        H, Dh = self.heads, self.dim_head

        img_qkv = self.img_qkv(hidden_states).reshape(bsz, seq_img, 3, H, Dh)
        txt_qkv = self.txt_qkv(encoder_hidden_states).reshape(bsz, seq_txt, 3, H, Dh)
        img_q, img_k, img_v = img_qkv[:, :, 0], img_qkv[:, :, 1], img_qkv[:, :, 2]
        txt_q, txt_k, txt_v = txt_qkv[:, :, 0], txt_qkv[:, :, 1], txt_qkv[:, :, 2]

        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)
        txt_q = self.norm_added_q(txt_q)
        txt_k = self.norm_added_k(txt_k)

        (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
        img_q = apply_rotary_emb_lens(img_q, img_cos[:seq_img], img_sin[:seq_img])
        img_k = apply_rotary_emb_lens(img_k, img_cos[:seq_img], img_sin[:seq_img])
        if seq_txt > 0:
            txt_q = apply_rotary_emb_lens(txt_q, txt_cos[:seq_txt], txt_sin[:seq_txt])
            txt_k = apply_rotary_emb_lens(txt_k, txt_cos[:seq_txt], txt_sin[:seq_txt])

        # Joint sequence, [B, H, S, D] for SDPA.
        q = mx.concatenate([img_q, txt_q], axis=1).transpose(0, 2, 1, 3)
        k = mx.concatenate([img_k, txt_k], axis=1).transpose(0, 2, 1, 3)
        v = mx.concatenate([img_v, txt_v], axis=1).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(Dh)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=attention_mask)
        out = out.transpose(0, 2, 1, 3).reshape(bsz, seq_img + seq_txt, -1)

        img_out = self.to_out[0](out[:, :seq_img, :])
        txt_out = self.to_add_out(out[:, seq_img:, :])
        return img_out, txt_out


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class LensTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        rms_norm: bool = False,
        gate_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.attn = LensJointAttention(
            query_dim=dim, added_kv_proj_dim=dim, dim_head=attention_head_dim,
            heads=num_attention_heads, out_dim=dim, eps=eps,
        )
        # rms_norm=True for Lens: RMSNorm with a learnable weight, then AdaLN modulation.
        def norm_cls(d):
            return nn.RMSNorm(d, eps=eps) if rms_norm else nn.LayerNorm(d, affine=False, eps=eps)

        hidden = int(dim / 3 * 8)
        self.img_mod = nn.Linear(dim, 6 * dim, bias=True)   # upstream Sequential(SiLU, Linear) -> .1
        self.img_norm1 = norm_cls(dim)
        self.img_norm2 = norm_cls(dim)
        self.img_mlp = GateMLP(dim, hidden)

        self.txt_mod = nn.Linear(dim, 6 * dim, bias=True)
        self.txt_norm1 = norm_cls(dim)
        self.txt_norm2 = norm_cls(dim)
        self.txt_mlp = GateMLP(dim, hidden)

    @staticmethod
    def _modulate(x: mx.array, mod_params: mx.array) -> Tuple[mx.array, mx.array]:
        shift, scale, gate = mx.split(mod_params, 3, axis=-1)
        return x * (1 + scale[:, None]) + shift[:, None], gate[:, None]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        image_rotary_emb,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        img_mod1, img_mod2 = mx.split(self.img_mod(nn.silu(temb)), 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(self.txt_mod(nn.silu(temb)), 2, axis=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated, encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb, attention_mask=attention_mask,
        )

        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        return encoder_hidden_states, hidden_states


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class AdaLayerNormContinuous(nn.Module):
    """norm_out: SiLU -> Linear(dim, 2*dim) -> affine-less LayerNorm modulation."""

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.linear = nn.Linear(cond_dim, 2 * dim, bias=True)
        self.norm = nn.LayerNorm(dim, affine=False, eps=eps)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        emb = self.linear(nn.silu(conditioning))
        scale, shift = mx.split(emb, 2, axis=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class LensTransformer2DModel(nn.Module):
    """The Lens text-to-image DiT."""

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 128,
        out_channels: Optional[int] = 32,
        num_layers: int = 48,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        inner_dim: int = 1536,
        enc_hidden_dim: int = 2880,
        axes_dims_rope: Tuple[int, int, int] = (8, 28, 28),
        gate_mlp: bool = True,
        rms_norm: bool = True,
        multi_layer_encoder_feature: bool = True,
        selected_layer_index: Tuple[int, ...] = (5, 11, 17, 23),
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.multi_layer_encoder_feature = multi_layer_encoder_feature
        self.selected_layer_index = list(selected_layer_index)

        self.pos_embed = LensEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        self.time_text_embed = LensTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        if multi_layer_encoder_feature:
            self.txt_norm = [nn.RMSNorm(enc_hidden_dim, eps=1e-5) for _ in self.selected_layer_index]
            self.txt_in = nn.Linear(enc_hidden_dim * len(self.selected_layer_index), self.inner_dim)
        else:
            self.txt_norm = nn.RMSNorm(enc_hidden_dim, eps=1e-5)
            self.txt_in = nn.Linear(enc_hidden_dim, self.inner_dim)

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.transformer_blocks = [
            LensTransformerBlock(
                dim=self.inner_dim, num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim, rms_norm=rms_norm, gate_mlp=gate_mlp,
            )
            for _ in range(num_layers)
        ]
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states,
        encoder_hidden_states_mask: mx.array,
        timestep: mx.array,
        img_shapes: List[Tuple[int, int, int]],
    ) -> mx.array:
        bsz, img_len, _ = hidden_states.shape
        if self.multi_layer_encoder_feature:
            text_seq_len = encoder_hidden_states[0].shape[1]
            normed = [self.txt_norm[i](encoder_hidden_states[i]) for i in range(len(self.txt_norm))]
            encoder_hidden_states = mx.concatenate(normed, axis=-1)
        else:
            text_seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)

        attention_mask = self._build_joint_attention_mask(encoder_hidden_states_mask, img_len)

        hidden_states = self.img_in(hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        temb = self.time_text_embed(timestep.astype(hidden_states.dtype), hidden_states)
        image_rotary_emb = self.pos_embed(img_shapes, [text_seq_len])

        for block in self.transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                temb=temb, image_rotary_emb=image_rotary_emb, attention_mask=attention_mask,
            )

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)

    @staticmethod
    def _build_joint_attention_mask(text_mask: mx.array, img_len: int) -> mx.array:
        """Additive joint mask [B, 1, 1, img_len + S_txt]; -inf on padded text positions."""
        bsz = text_mask.shape[0]
        img_ones = mx.ones((bsz, img_len), dtype=mx.bool_)
        joint = mx.concatenate([img_ones, text_mask.astype(mx.bool_)], axis=1)
        additive = mx.where(joint, 0.0, -mx.inf).astype(mx.float32)
        return additive[:, None, None, :]
