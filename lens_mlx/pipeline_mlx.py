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


class LensPipeline:
    """Lens text-to-image pipeline (assembled in Phase 3)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Phase 3: assemble LensPipeline (VAE + scheduler + e2e)")
