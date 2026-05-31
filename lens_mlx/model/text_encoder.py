"""GPT-OSS text-feature encoder for Lens — MLX port.

Isomorphic to `refs/Lens/lens/text_encoder.py::LensGptOssEncoder`. Where the
upstream subclasses `transformers.GptOssForCausalLM` and overrides `forward` to
return hidden states at a layer subset, we wrap the **mlx-lm** `gpt_oss` model and
replicate the same masked decoder loop:

  - run the full 24-layer causal stack with alternating sliding(128)/full
    attention (T8 — causal, not bidirectional), reusing mlx-lm's mask builders;
  - capture post-layer hidden states at `selected_layers` (default [5,11,17,23],
    T0 — 0-indexed, confirmed in the checkpoint);
  - early-exit after the last selected layer (23 = final); skip final RMSNorm + LM head.

Tokenization + the T5 chat template live in `pipeline_mlx.py` (as upstream keeps
them in `pipeline.py`); this module is the feature-capture forward only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask
from mlx_lm.utils import load_model

DEFAULT_SELECTED_LAYERS = (5, 11, 17, 23)

# GPT-OSS-20B uses YaRN rope. This is a `GptOssConfig` class default in HF
# transformers, so the Lens checkpoint's text_encoder/config.json does NOT serialize
# it. mlx-lm reads the json, sees no rope_scaling, and silently falls back to plain
# rope (mscale=1.0) — which diverges from the reference at EVERY layer/position
# (the YaRN attention_scaling is ~1.3466). We inject it when the config omits it.
# See CLAUDE.md skill-feedback F9.
GPT_OSS_YARN_ROPE = {
    "rope_type": "yarn",
    "factor": 32.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "original_max_position_embeddings": 4096,
    "rope_theta": 150000,
    "truncate": False,
}


class LensGptOssEncoder:
    """mlx-lm `gpt_oss` wrapper that exposes selected hidden states."""

    def __init__(self, model, selected_layers: Sequence[int] = DEFAULT_SELECTED_LAYERS) -> None:
        self.model = model  # mlx_lm gpt_oss Model
        self.set_selected_layers(selected_layers)

    @classmethod
    def from_pretrained(cls, path, selected_layers: Sequence[int] = DEFAULT_SELECTED_LAYERS):
        """Load an mlx gpt_oss checkpoint dir (MXFP4 production or bf16 diagnostic).

        Injects the GPT-OSS YaRN rope params when the on-disk config omits them
        (it usually does — they're an HF class default). Without this the rope is
        plain and the encoder diverges from the reference (F9).
        """
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        model_config = None
        if not cfg.get("rope_scaling"):
            model_config = {"rope_theta": 150000, "rope_scaling": dict(GPT_OSS_YARN_ROPE)}
        model, _ = load_model(path, model_config=model_config or {})
        return cls(model, selected_layers)

    def set_selected_layers(self, layer_indices: Sequence[int]) -> None:
        layers = [int(i) for i in layer_indices]
        if not layers:
            raise ValueError("layer_indices must be non-empty")
        if len(set(layers)) != len(layers):
            raise ValueError(f"layer_indices must be unique; got {layers}")
        n = len(self.model.model.layers)
        if min(layers) < 0 or max(layers) >= n:
            raise ValueError(f"layer_indices out of range; got {layers}, model has {n} layers")
        self._selected_layers = layers
        self._max_layer = max(layers)

    def __call__(self, input_ids: mx.array) -> List[mx.array]:
        """Return per-layer hidden states at the selected layers.

        Args:
            input_ids: int array [B, S]. (Single unpadded sequence for parity;
                padding-mask support rides with the pipeline batching path.)
        """
        moe = self.model.model
        x = moe.embed_tokens(input_ids)

        # Mirror GptOssMoeModel.__call__: one full + one sliding causal mask,
        # selected per layer_type. No KV cache (full prompt, single forward).
        full_mask = create_attention_mask(x, None)
        swa_mask = create_attention_mask(x, None, window_size=moe.window_size)

        index_lookup = {idx: pos for pos, idx in enumerate(self._selected_layers)}
        captured: List = [None] * len(self._selected_layers)
        for i, (layer, layer_type) in enumerate(zip(moe.layers, moe.layer_types)):
            mask = full_mask if layer_type == "full_attention" else swa_mask
            x = layer(x, mask, None)
            if i in index_lookup:
                captured[index_lookup[i]] = x
            if i == self._max_layer:
                break

        for pos, layer_idx in enumerate(self._selected_layers):
            if captured[pos] is None:
                raise RuntimeError(f"Failed to capture hidden state for layer {layer_idx}")
        mx.eval(captured)
        return captured

    # Backwards-compatible alias matching upstream.
    def encode_layers(self, input_ids: mx.array) -> List[mx.array]:
        return self(input_ids)
