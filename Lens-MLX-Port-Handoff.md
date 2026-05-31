# Lens → MLX Port Plan & Handoff

**Target:** `microsoft/Lens` (+ `Lens-Turbo`, `Lens-Base`) — 3.8B text-to-image DiT, MIT.
**Convention:** Tier 3 (multi-component diffusion pipeline) → `lens-mlx` fork + `mlx-forge` recipe + `mlx-arsenal`, mirroring the LTX-2 / Qwen-Image efforts. Swift consumer mirror under `xocialize-code/lens-mlx`.
**Reference oracle:** `github.com/microsoft/Lens` @ depth-1 clone (`lens/{transformer,text_encoder,pipeline,resolution,reasoner}.py`, `inference.py`). Total reference surface is ~1.8k LOC — small and clean.
**Status as of this doc:** No MLX port exists anywhere (not mlx-community, not mflux). Nvidia-only ports in flight (ComfyUI PR #14077, sdnext #4864).

---

## 1. Verdict / effort

Roughly **40–50% of a full LTX-2-scale port.** Three of five components are mostly off-the-shelf in MLX; the net-new authoring is the DiT plus four bespoke numerical details (complex axial RoPE, the empirical-mu shift, the VAE `bn` latent de-norm, and the fixed chat-template encoder path). The standout risk is *not* block count — it's that a naive port will produce plausible-but-wrong output that passes small-scale parity. Every one of the four bespoke details is a Step-5 parity gate, not a Step-6 discovery.

---

## 2. Architecture map (exact, from the reference)

Pipeline: **GPT-OSS multi-layer text features → Lens DiT (flow-matching) → FLUX.2 VAE decode.**

| Component | Reference class | Params | MLX status |
|---|---|---|---|
| Text encoder | `LensGptOssEncoder(GptOssForCausalLM)` | ~20B (GPT-OSS-20B) | **Reuse** mlx-lm `gpt_oss` + capture wrapper |
| Denoiser | `LensTransformer2DModel` | ~3.8B | **Author from scratch** (the work) |
| VAE | `AutoencoderKLFlux2` | small | **Reuse/port** — mflux has FLUX.2 VAE |
| Scheduler | `FlowMatchEulerDiscreteScheduler` + bespoke `mu` | — | **Assemble** from mlx-arsenal + port `mu` |
| Tokenizer | GPT-OSS (harmony) + fixed chat template | — | **Reuse** (rides with mlx-lm encoder) |

### DiT config (registered defaults — **confirm against checkpoint `transformer/config.json`**)

```
patch_size=2  in_channels=128  out_channels=32
num_layers=48  num_attention_heads=24  attention_head_dim=64  inner_dim=1536
enc_hidden_dim=2880               # = GPT-OSS hidden size
axes_dims_rope=(8,28,28)          # sums to head_dim 64
gate_mlp=True  rms_norm=True
multi_layer_encoder_feature=True
selected_layer_index=(5,11,17,23) # 0-indexed — NOT (4,12,18,24); see trap T0
```

`in_channels=128` = 32 VAE latent channels × 4 (the 2×2 patchify). `vae_scale_factor=16`; with the extra 2×2 patchify the DiT sees a 32× spatially-compressed grid.

### Block structure (`LensTransformerBlock`, ×48 — all double-stream, no single-stream tier)

Per block, both image and text streams carry their own params:
- **6-way AdaLN** per stream: `SiLU → Linear(dim, 6·dim)`, chunked into two `(shift, scale, gate)` triples (attn + mlp). Modulation is `x*(1 + scale) + shift`, gate applied to the residual.
- **Joint attention** (`LensJointAttention`): per-stream **fused QKV** `Linear(dim, 3·inner)`, reshaped `view(B,S,3,H,Dh).unbind(2)` → **stacked** layout (3 is its own axis *before* heads — replicate exactly, this is not the interleaved case). **QK-RMSNorm** on each of img/txt q/k (`eps=1e-5`). Complex axial RoPE on both streams. Concat `[img, txt]` → single SDPA → split back.
- **SwiGLU MLP** (`GateMLP`): hidden = `dim/3*8` = 4096. `w2(silu(w1 x) * w3 x)`.
- `norm_out = AdaLayerNormContinuous(eps=1e-6)`, `proj_out → Linear(inner, patch²·out_channels)`.

**Epsilon inventory (easy to mix up — skill flags this):** QK-norm `1e-5`, block norms `1e-6`, `txt_norm` `1e-5`, `norm_out` `1e-6`, VAE `batch_norm_eps` (read from VAE config).

---

## 3. Reuse inventory (what you do NOT write)

- **GPT-OSS-20B**: fully supported in mlx-lm; mlx-community has native MXFP4 (`gpt-oss-20b-MXFP4-Q4/Q8`). Reuse the architecture + weights wholesale.
- **FLUX.2 VAE**: `mflux` already ships it (`flux2-klein-4b/9b` support). Use as oracle; ideally lift directly rather than re-port.
- **Flow-match primitives**: `mlx_arsenal.diffusion` → `FlowMatchEulerDiscreteScheduler`, `euler_step`, `classifier_free_guidance`, `get_sampling_sigmas`, `dynamic_shift_schedule`. Assembly, not authoring.
- **Patchify**: `mlx_arsenal.spatial.pixel_shuffle` / `patchify` — but verify axis order against trap T4.

---

## 4. The port surface (what IS net-new)

1. `lens_mlx/model/transformer.py` — the DiT. ~550 LOC PyTorch → isomorphic MLX. The bulk.
2. `lens_mlx/model/text_encoder.py` — thin subclass/wrapper over mlx-lm `gpt_oss` that captures hidden states at `[5,11,17,23]` and early-exits after layer 23 (skips final norm + LM head). ~80 LOC equivalent.
3. `lens_mlx/pipeline_mlx.py` — `from_pretrained`, chat-template prompt wrapping, denoise loop, CFG, VAE `bn` de-norm + decode.
4. `mlx-forge` recipe — per-component classify/sanitize/transform/convert; split safetensors per component; **materialize every tensor (`mx.eval`) before save** (the Tier-3 silent killer).
5. `tests/parity/` + `tests/smoke/` — see §6.

---

## 5. Phased plan (each phase ends on a parity gate; never advance on a red gate)

**Phase 0 — Reference capture & config lock.**
Pull the real `transformer/config.json`, `vae/config.json`, and `model_index.json` from the HF checkpoint; reconcile against the registered defaults in §2 (esp. `num_layers`, `selected_layer_index`, `batch_norm_eps`). Dump golden tensors from the PT reference at a fixed seed/prompt: encoder per-layer outputs, DiT input/output, final latent, decoded image. These are the parity oracle for every later phase.

**Phase 1 — Text encoder.** Wrap mlx-lm `gpt_oss` to capture `[5,11,17,23]`. Reproduce the **exact chat-template prefix** (trap T5). Gate: per-layer hidden-state `max_abs < 1e-3` (fp16) vs PT, on the same tokenized prompt.

**Phase 2 — DiT, layer-by-layer.** Translate `GateMLP` → `LensJointAttention` → `LensTransformerBlock` → top model. Gate each in isolation against Phase-0 goldens: single block `< 5e-3`, full DiT pass `< 1e-2`. RoPE (T2) and AdaLN (T3) get their own micro-parity tests before the block test.

**Phase 3 — VAE + scheduler + e2e.** Wire FLUX.2 VAE (with the `bn` de-norm, T1), the empirical-`mu` flow-match schedule (T6), and CFG. Run the noise-path smoke test (decode random Gaussian through `bn`-inverse → unpatchify → VAE.decode) — any periodic pattern = a spatial-op bug regardless of green layer parity (skill pitfall #7). Gate: full-pipeline golden image PSNR vs PT reference.

**Phase 4 — Turbo + quantize.** Add the Lens-Turbo 4-step path. Quantize DiT Linears (`group_size=64, bits=4`), keep VAE/encoder feature path at bf16. Re-parity at relaxed int4 thresholds (`max_abs < 5e-2`).

**Phase 5 — Publish + Swift mirror.** mlx-community repos per quant-suffix grammar; `lens-mlx` fork README; then `xocialize-code/lens-mlx` Swift mirror (Xcode workspace, `Package.swift`, `xcodebuild` build — not SwiftPM CLI).

---

## 6. Named traps (the landmines, with exact references)

- **T0 — Layer-index off-by-one.** Code says `selected_layer_index=(5,11,17,23)` (0-indexed); the press said "4,12,18,24." The code is the oracle. Wrong indices → subtly wrong conditioning that *passes* shape checks.
- **T1 — VAE `bn` latent de-norm.** Before decode, Lens reverses a BatchNorm-style latent normalization in **patchified** space: `shift=-running_mean`, `scale=1/sqrt(running_var+eps)`, then `x = x/scale - shift`, then unpatchify, then `vae.decode`. Skip or misplace this and you get washed-out/wrong-scale output. This *is* the "BN inverse" in skill pitfall #7.
- **T2 — Complex axial RoPE.** Reference uses `torch.view_as_complex` / `torch.polar` with a pos/neg-frequency split and `scale_rope=True` (h/w centered around 0). MLX has no `view_as_complex` — reimplement as real interleaved rotation. Single trickiest translation; give it a dedicated parity test against `apply_rotary_emb_lens`.
- **T3 — AdaLN form.** `x*(1+scale)+shift` with `unsqueeze(1)` broadcast; gate multiplies the residual branch. Additive-only or missing `1+` is the classic silent AdaLN bug.
- **T4 — Patchify axis order.** `view(b,c,h//2,2,w//2,2).permute(0,1,3,5,2,4)`. If the e2e image is periodic noise at stride 2, this (or `tile` vs `repeat`) is the culprit — the checkerboard trap.
- **T5 — Chat-template prompt wrapping.** The encoder does **not** see the raw prompt. Lens wraps it in a fixed GPT-OSS harmony chat template: a fixed `_CHAT_SYSTEM` instruction + a canned assistant "thinking" turn, with `DEFAULT_TXT_OFFSET=97`. Feed raw text and conditioning is wrong from token 0.
- **T6 — Bespoke `mu` schedule.** `compute_empirical_mu(image_seq_len, num_steps)` is a piecewise-linear fit (constants `a1,b1=8.738e-5,1.898`; `a2,b2=1.6927e-4,0.4567`; breakpoint at `image_seq_len>4300`, else interpolate `m_10..m_200`). This is *not* the stock FLUX mu. Port the formula verbatim.
- **T7 — Swift gpt-oss MXFP4 config parse.** Known `quantization.mode` type-mismatch loading gpt-oss MXFP4 in mlx-swift-examples (issue #386). Plan a config shim for Phase 5.
- **T8 — Encoder is causal.** GPT-OSS used as a *causal* feature extractor with alternating sliding-window/full attention (`config.layer_types[i]`). mlx-lm already does this; preserve it, don't substitute bidirectional.

---

## 7. Repo layout

```
lens-mlx/
├── README.md                 # Quick Start + HF repo links
├── pyproject.toml            # mlx, mlx-arsenal, mlx-lm; mlx-forge (dev); torch (parity extra)
├── lens_mlx/
│   ├── model/
│   │   ├── transformer.py     # LensTransformer2DModel  (isomorphic to upstream)
│   │   └── text_encoder.py    # gpt_oss capture wrapper
│   ├── pipeline_mlx.py        # from_pretrained, chat-template, denoise, CFG, decode
│   ├── resolution.py          # port of RESOLUTION_BUCKETS / resolve_resolution
│   └── utils/weights.py       # split-safetensors load, HF hub fetch, LENS_MLX_WEIGHTS_DIR override
├── recipes/lens.yaml          # mlx-forge per-component recipe
└── tests/{parity,smoke}/      # PT optional dep; goldens from Phase 0
```

Keep file/class/method names **identical to upstream** (skill hard rule — drift is what cost the LTX-2 port). A reader should diff `transformer.py` (PT) vs `transformer.py` (MLX) and see only PT↔MLX op substitutions.

---

## 8. License posture (worth a `LicensePolicy` check like DubKit)

- **Lens weights + code:** MIT — clean for mlx-community.
- **GPT-OSS-20B:** Apache-2.0 — clean.
- **FLUX.2 VAE:** the question mark. The Comfy-Org packaging labels the FLUX.2-dev VAE Apache-2.0, but FLUX.2-dev's umbrella terms are not uniformly permissive. **Verify the exact VAE-weights license before publishing a bundled conversion.** Safe fallback: ship the DiT + encoder-wrapper conversions and have `from_pretrained` pull the VAE from its original source at load time, rather than re-hosting it.

---

## 9. Open decisions (yours to make)

1. **mflux host vs standalone fork first?** mflux already has the FLUX.2 VAE + flow-match scaffolding and a CLI — fastest path to lock DiT parity. Recommendation: validate in mflux, then lift into the standalone `lens-mlx` fork for the house convention + Swift mirror. (Alternative: add Lens as an mflux `--base-model` the way z-image/fibo were, and skip the standalone repo — lower effort, less aligned with prior efforts.)
2. **Which variants to ship?** Lens (RL-tuned, 20-step) + Lens-Turbo (4-step) cover the useful cases; Lens-Base (50-step, no RL) is lower priority.
3. **Quant targets for mlx-community?** Suggest DiT bf16 + int4; encoder MXFP4 (reuse existing); VAE bf16.
