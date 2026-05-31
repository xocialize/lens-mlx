"""Smoke generate — full text->image via LensPipeline (no goldens; visual check)."""

import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lens_mlx.pipeline_mlx import LensPipeline

PROMPT = sys.argv[1] if len(sys.argv) > 1 else \
    "A scenic landscape with a serene lake and snow-capped mountains, golden hour."
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
H = W = int(sys.argv[3]) if len(sys.argv) > 3 else 1024

print(f"[gen] loading pipeline (DiT bf16) ...")
t0 = time.time()
pipe = LensPipeline.from_pretrained(ROOT / "weights" / "Lens", dit_dtype=mx.bfloat16)
print(f"[gen] loaded in {time.time()-t0:.1f}s; generating {H}x{W} x{STEPS} steps")
print(f"[gen] prompt: {PROMPT!r}")
t0 = time.time()
img = pipe(PROMPT, height=H, width=W, num_inference_steps=STEPS, guidance_scale=4.0, seed=42)
print(f"[gen] done in {time.time()-t0:.1f}s; peak mem {mx.get_peak_memory()/1e9:.1f} GB")
out = ROOT / "outputs"
out.mkdir(exist_ok=True)
path = out / "gen_smoke.png"
img.save(path)
print(f"[gen] saved {path}")
