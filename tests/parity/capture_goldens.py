"""Phase 0e — capture PyTorch golden tensors for the Lens parity oracle.

Runs the upstream `refs/Lens` reference at a FIXED seed/prompt/resolution on CPU
(fp32 DiT + VAE; bf16-dequantized GPT-OSS encoder) and dumps the tensors every
later phase parity-tests against:

  goldens/lens_goldens.npz   (+ lens_goldens.manifest.json)
    text_feat_{0..3}     : per-layer encoder features [5,11,17,23]  (Phase 1)
    text_mask            : encoder feature mask
    dit_in_hidden        : DiT hidden_states input  (first denoise step)  (Phase 2)
    dit_in_timestep      : DiT timestep input       (first denoise step)
    dit_out_noise        : DiT output               (first denoise step)
    final_latent         : latent after the denoise loop                  (Phase 3)
    decoded_image        : VAE-decoded image [B,C,H,W] in [-1,1]          (Phase 3)

Run (after weights land under weights/Lens):
    LENS_WEIGHTS=weights/Lens .venv/bin/python tests/parity/capture_goldens.py

Deliberately small (512x512, 4 steps) so the CPU reference is tractable; this is
the layer-parity oracle, not a production render. A production-scale noise-path
smoke (pitfall #7) is added separately in tests/smoke.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
REPO = os.environ.get("LENS_WEIGHTS", str(ROOT / "weights" / "Lens"))
REFS = ROOT / "refs" / "Lens"
OUT_DIR = ROOT / "goldens"
sys.path.insert(0, str(REFS))

# Fixed oracle config.
PROMPT = "A scenic landscape with a serene lake and snow-capped mountains."
NEGATIVE = ""
HEIGHT = WIDTH = 512        # divisible by vae_scale_factor=16
STEPS = 4
CFG = 4.0
SEED = 42
DEVICE = "cpu"


def main() -> None:
    from transformers import Mxfp4Config
    from lens import LensGptOssEncoder, LensPipeline

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"[goldens] loading encoder (bf16 dequant) from {REPO}")
    text_encoder = LensGptOssEncoder.from_pretrained(
        REPO, subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=Mxfp4Config(dequantize=True),
    )
    print("[goldens] loading pipeline (fp32 DiT + VAE)")
    pipe = LensPipeline.from_pretrained(
        REPO, text_encoder=text_encoder, torch_dtype=torch.float32,
    )
    pipe.to(DEVICE)

    captured: dict[str, np.ndarray] = {}

    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().to(torch.float32).cpu().numpy()

    # --- Phase 1 oracle: encoder per-layer features + mask (no RNG) ---
    feats, mask = pipe._get_text_embeddings([PROMPT], max_sequence_length=512, device=DEVICE)
    for i, f in enumerate(feats):
        captured[f"text_feat_{i}"] = to_np(f)
    captured["text_mask"] = mask.detach().cpu().numpy().astype(np.int32)
    print(f"[goldens] text feats: {[captured[f'text_feat_{i}'].shape for i in range(len(feats))]}")

    # --- Phase 2 oracle: capture first-step DiT input/output via hooks ---
    dit_io: dict[str, torch.Tensor] = {}

    def pre_hook(_module, args, kwargs):
        if "dit_in_hidden" in dit_io:
            return
        dit_io["dit_in_hidden"] = kwargs["hidden_states"].clone()
        dit_io["dit_in_timestep"] = kwargs["timestep"].clone()

    def fwd_hook(_module, args, kwargs, output):
        if "dit_out_noise" in dit_io:
            return
        dit_io["dit_out_noise"] = output.clone()

    h1 = pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = pipe.transformer.register_forward_hook(fwd_hook, with_kwargs=True)

    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    print(f"[goldens] denoising {STEPS} steps @ {HEIGHT}x{WIDTH} ...")
    out = pipe(
        prompt=[PROMPT], negative_prompt=NEGATIVE,
        height=HEIGHT, width=WIDTH,
        num_inference_steps=STEPS, guidance_scale=CFG,
        num_images_per_prompt=1, generator=generator,
        output_type="latent",
    )
    h1.remove(); h2.remove()

    for k, v in dit_io.items():
        captured[k] = to_np(v)
    final_latent = out.images if not hasattr(out, "images") else out.images
    captured["final_latent"] = to_np(final_latent)

    # --- Phase 3 oracle: decode the final latent ---
    latent_h, latent_w = HEIGHT // pipe.vae_scale_factor, WIDTH // pipe.vae_scale_factor
    decoded = pipe._decode(final_latent.to(pipe.vae.dtype), latent_h, latent_w)
    captured["decoded_image"] = to_np(decoded)
    print(f"[goldens] dit_in={captured['dit_in_hidden'].shape} "
          f"dit_out={captured['dit_out_noise'].shape} "
          f"latent={captured['final_latent'].shape} image={captured['decoded_image'].shape}")

    OUT_DIR.mkdir(exist_ok=True)
    np.savez(OUT_DIR / "lens_goldens.npz", **captured)
    manifest = {
        "config": dict(prompt=PROMPT, negative=NEGATIVE, height=HEIGHT, width=WIDTH,
                       steps=STEPS, cfg=CFG, seed=SEED, device=DEVICE,
                       dit_dtype="float32", encoder_dtype="bfloat16"),
        "tensors": {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in captured.items()},
    }
    (OUT_DIR / "lens_goldens.manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[goldens] saved -> {OUT_DIR / 'lens_goldens.npz'}")


if __name__ == "__main__":
    main()
