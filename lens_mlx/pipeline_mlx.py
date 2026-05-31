"""Lens text-to-image pipeline — MLX port.

Isomorphic to `refs/Lens/lens/pipeline.py::LensPipeline`. The module-level helpers
(chat template, empirical-mu) are ported verbatim now — they are pure Python and are
needed by the Phase 1 encoder parity path. The `LensPipeline` class itself (denoise
loop, CFG, VAE decode) is assembled in Phase 3.

Phase-3 TODOs on the class:
  - from_pretrained(repo_id)  : split safetensors (HF hub), LENS_MLX_WEIGHTS_DIR override.
  - scheduler                 : exponential time-shift(mu) + sigmas=linspace(1,1/N,N).
  - denoise loop              : CFG batch cat([cond,uncond]); NORM-RESCALED guidance
                                noise_pred = comb*(||cond||/||comb||) (NOT vanilla).
  - _decode                   : T1 VAE bn de-norm in patchified space, T4 unpatchify, decode.
"""

from __future__ import annotations

from typing import List, Sequence

import mlx.core as mx

from .scheduler import FlowMatchEulerDiscreteScheduler

# Chat template constants used by the Lens text encoder (verbatim from upstream).
_CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
_CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
DEFAULT_TXT_OFFSET = 97


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Empirical ``mu`` for ``FlowMatchEulerDiscreteScheduler`` dynamic shift.

    Ported verbatim from upstream pipeline.py (T6). Constants are calibrated for the
    Lens inference schedule.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def build_chat_inputs(tokenizer, prompts: Sequence[str], max_sequence_length: int = 512):
    """Render the fixed GPT-OSS harmony chat template and tokenize (T5).

    Mirrors ``LensPipeline._build_chat_inputs``. Returns ``(input_ids, attention_mask)``
    as numpy arrays. The encoder does NOT see the raw prompt — it sees this template,
    and the pipeline later slices off the first ``DEFAULT_TXT_OFFSET`` tokens.
    """
    rendered: List[str] = []
    for prompt in prompts:
        conversation = [
            {"role": "system", "content": _CHAT_SYSTEM, "thinking": None},
            {"role": "user", "content": prompt, "thinking": None},
            {"role": "assistant", "thinking": _CHAT_ASSISTANT_THINKING, "content": ""},
        ]
        text = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        text = text.split("<|return|>")[0]
        rendered.append(text)

    encoded = tokenizer(
        rendered,
        padding=True,
        truncation=True,
        max_length=max_sequence_length,
        return_tensors="np",
        add_special_tokens=True,
    )
    return encoded["input_ids"], encoded["attention_mask"]


def lens_cfg(noise: mx.array, guidance_scale: float) -> mx.array:
    """Norm-rescaled classifier-free guidance (T-extra, NOT vanilla CFG).

    Upstream: comb = uncond + g*(cond-uncond); rescale per-token by ||cond||/||comb||.
    """
    cond, uncond = mx.split(noise, 2, axis=0)
    comb = uncond + guidance_scale * (cond - uncond)
    cond_norm = mx.linalg.norm(cond, axis=-1, keepdims=True)
    comb_norm = mx.linalg.norm(comb, axis=-1, keepdims=True)
    scale = mx.where(comb_norm > 0, cond_norm / mx.maximum(comb_norm, 1e-12), mx.ones_like(comb_norm))
    return comb * scale


def denoise(
    transformer,
    latents: mx.array,
    encoder_features: List[mx.array],
    encoder_mask: mx.array,
    img_shapes,
    num_inference_steps: int,
    guidance_scale: float = 4.0,
    scheduler: FlowMatchEulerDiscreteScheduler = None,
) -> mx.array:
    """Lens flow-match denoising loop with CFG batching + norm-rescaled guidance.

    `latents` is the single (uncond/cond-shared) image latent [B, S, C]; it is
    repeated for the joint CFG batch each step. `encoder_features`/`encoder_mask`
    are already CFG-batched ([cond; uncond]).
    """
    seq_len = latents.shape[1]
    mu = compute_empirical_mu(seq_len, num_inference_steps)
    N = num_inference_steps
    sigmas = [1.0 - i * (1.0 - 1.0 / N) / (N - 1) for i in range(N)] if N > 1 else [1.0]
    scheduler = scheduler or FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(sigmas, mu=mu)

    for t in scheduler.timesteps:
        hidden_states = mx.concatenate([latents, latents], axis=0)
        timestep = mx.full((hidden_states.shape[0],), t / 1000.0)
        noise = transformer(
            hidden_states=hidden_states, encoder_hidden_states=encoder_features,
            encoder_hidden_states_mask=encoder_mask, timestep=timestep, img_shapes=img_shapes,
        )
        noise_pred = lens_cfg(noise, guidance_scale)
        latents = scheduler.step(noise_pred, latents)
        mx.eval(latents)
    return latents


def _pack_latents_for_decode(latents: mx.array, latent_h: int, latent_w: int) -> mx.array:
    """[b, h*w, c*4] -> packed [b, c*4, h, w] (Lens _decode: rearrange then patchify)."""
    b = latents.shape[0]
    c = latents.shape[2] // 4
    x = latents.reshape(b, latent_h, latent_w, c, 2, 2).transpose(0, 3, 1, 4, 2, 5)
    x = x.reshape(b, c, latent_h * 2, latent_w * 2)
    H, W = latent_h * 2, latent_w * 2
    x = x.reshape(b, c, H // 2, 2, W // 2, 2).transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(b, c * 4, H // 2, W // 2)


class LensPipeline:
    """Lens text-to-image pipeline — MLX. Mirrors refs/Lens/lens/pipeline.py."""

    vae_scale_factor = 16
    latent_channels = 128
    txt_offset = DEFAULT_TXT_OFFSET

    def __init__(self, transformer, vae, text_encoder, tokenizer):
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    # Default int4 quant scope: keep the small, precision-sensitive in/out/time
    # projections at bf16 (better parity, negligible size cost — see Phase 4).
    QUANT_KEEP_HI = ("img_in", "txt_in", "proj_out", "time_text_embed", "norm_out")

    @classmethod
    def from_pretrained(cls, repo_dir, dit_repo=None, dit_dtype=mx.float32,
                        quantize_bits=None, quant_group_size=64):
        """Assemble the pipeline.

        `repo_dir`: a base Lens checkpoint dir providing the tokenizer, GPT-OSS encoder,
        and FLUX.2 VAE (e.g. a local `microsoft/Lens` snapshot).
        `dit_repo`: optional CONVERTED mlx DiT repo (e.g. `mlx-community/Lens-3.8B-4bit`).
        If given, the DiT is loaded from it (already bf16/quantized) and `dit_dtype` /
        `quantize_bits` are ignored. Otherwise the DiT is loaded from `repo_dir/transformer`
        (PT) and optionally quantized on the fly.
        """
        from pathlib import Path
        from transformers import AutoTokenizer
        from .model.transformer import LensTransformer2DModel
        from .model.text_encoder import LensGptOssEncoder
        from .utils.weights import load_dit_weights, load_dit_repo, load_vae, quantize_dit

        repo = Path(repo_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(repo / "tokenizer"))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        text_encoder = LensGptOssEncoder.from_pretrained(repo / "text_encoder")

        if dit_repo is not None:
            transformer = load_dit_repo(dit_repo)
        else:
            transformer = LensTransformer2DModel()
            load_dit_weights(transformer, repo / "transformer", dtype=dit_dtype)
            if quantize_bits is not None:
                quantize_dit(transformer, group_size=quant_group_size, bits=quantize_bits,
                             keep_hi_precision=cls.QUANT_KEEP_HI)

        vae = load_vae(repo / "vae")
        return cls(transformer, vae, text_encoder, tokenizer)

    def _encode(self, prompt: str, max_sequence_length: int = 512):
        """Chat-template encode + offset slice -> (features list, mask), batch 1."""
        input_ids, attn = build_chat_inputs(self.tokenizer, [prompt], max_sequence_length)
        layers = self.text_encoder(mx.array(input_ids))  # 4 x [1, S, 2880]
        off = self.txt_offset
        if input_ids.shape[1] > off:
            feats = [f[:, off:, :] for f in layers]
            mask = mx.array(attn[:, off:].astype("int32"))
        else:
            feats = [f[:, :0, :] for f in layers]
            mask = mx.zeros((1, 0), dtype=mx.int32)
        return feats, mask

    def __call__(
        self,
        prompt: str,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 20,
        guidance_scale: float = 4.0,
        seed: int = 0,
    ):
        from PIL import Image
        import numpy as np

        if height % self.vae_scale_factor or width % self.vae_scale_factor:
            raise ValueError(f"height/width must be divisible by {self.vae_scale_factor}")
        latent_h, latent_w = height // self.vae_scale_factor, width // self.vae_scale_factor

        # 1. Encode prompt (positive) + empty negative (zeros), CFG-batch [cond; uncond].
        pos, pos_mask = self._encode(prompt)
        enc = [mx.concatenate([p, mx.zeros_like(p)], axis=0) for p in pos]
        mask = mx.concatenate([pos_mask, mx.zeros_like(pos_mask)], axis=0)

        # 2. Initial latents (MLX RNG — not torch-seed-compatible).
        mx.random.seed(seed)
        latents = mx.random.normal((1, latent_h * latent_w, self.latent_channels))

        # 3. Denoise.
        latents = denoise(
            self.transformer, latents, enc, mask, [(1, latent_h, latent_w)],
            num_inference_steps, guidance_scale,
        )

        # 4. Decode (T1 bn de-norm + T4 unpatchify + VAE).
        img = self.vae.decode_packed_latents(_pack_latents_for_decode(latents, latent_h, latent_w))
        mx.eval(img)
        img = np.array(img.astype(mx.float32))
        if img.shape[1] != 3 and img.shape[-1] == 3:
            img = np.transpose(img, (0, 3, 1, 2))
        img = np.clip(img, -1.0, 1.0)
        img = ((img + 1.0) * 127.5).astype("uint8")[0].transpose(1, 2, 0)
        return Image.fromarray(img)
