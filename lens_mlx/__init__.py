"""Apple MLX port of microsoft/Lens — GPT-OSS-conditioned text-to-image DiT.

Public surface mirrors the upstream `lens` package so a reader can diff the MLX
port against the PyTorch reference (`refs/Lens/lens/`) and see only op
substitutions. See docs/phase0-findings.md for the config reconciliation and the
named traps (T0–T8) each module must honor.
"""

from __future__ import annotations

__version__ = "0.1.0.dev0"

# Pure-Python (no torch) — ported verbatim from upstream.
from .resolution import (  # noqa: F401
    RESOLUTION_BUCKETS,
    SUPPORTED_ASPECT_RATIOS,
    SUPPORTED_BASE_RESOLUTIONS,
    resolve_resolution,
)

__all__ = [
    "RESOLUTION_BUCKETS",
    "SUPPORTED_ASPECT_RATIOS",
    "SUPPORTED_BASE_RESOLUTIONS",
    "resolve_resolution",
]
