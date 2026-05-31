# CLAUDE.md — lens-mlx

Working notes for the MLX port of `microsoft/Lens` (GPT-OSS-conditioned text-to-image DiT).
Read this with `docs/phase0-findings.md` (config reconciliation + T0–T8) and
`Lens-MLX-Port-Handoff.md` (full plan).

## What this is

Apple-Silicon inference port. Pipeline: **GPT-OSS-20B multi-layer text features
([5,11,17,23]) → Lens DiT (48-layer double-stream flow-matching) → FLUX.2 VAE decode.**
Driven by the `mlx-porting` skill. Reference oracle lives in `refs/Lens/` (depth-1 clone).

## v1 scope (locked decisions)

- **Variant:** Lens (RL, 20-step) only. Turbo / Base deferred.
- **Precision:** bf16 first; quantization deferred to a later pass.
- **Strategy:** validate DiT parity against `mflux`'s FLUX.2 VAE + flow-match, then this
  standalone fork → `xocialize-code/lens-mlx` Swift mirror.
- **Convention:** vendored MLX ops + `recipes/convert_lens.py` script (mirrors
  `~/DEV_INT/longcat-avatar-mlx`), torch as a `[parity]` extra. NOT external
  mlx-arsenal/mlx-forge package deps.

## Environment / workflow

- venv at `.venv` (Python 3.12). **It's a `uv` venv — use `uv pip install`, not `pip`**
  (`python -m pip` is absent).
- Weights under `weights/Lens/` (gitignored, 28 GB). Reference clone `refs/Lens/` gitignored.
- Golden capture: `LENS_WEIGHTS=weights/Lens .venv/bin/python tests/parity/capture_goldens.py`
  → writes `goldens/lens_goldens.npz`. CPU fp32 DiT/VAE + bf16-dequant encoder, 512×512/4-step.
- Smoke: `python -m pytest tests/smoke`. Parity: `python -m pytest tests/parity` (needs goldens).

## Hard rules (from the skill)

- **Isomorphic to upstream.** Same file/class/method names as `refs/Lens/lens/*`. A reader
  diffs MLX vs PT and sees only op substitutions. No refactors until fp16 parity is locked.
- **The code/checkpoint is the oracle**, not docs/press (T0 proved this: `[5,11,17,23]`).
- **Never advance on a red parity gate.** Each phase ends on one.
- **Materialize every tensor (`mx.eval`) before `save_safetensors`** — lazy tensors save as zeros.

---

# Skill-feedback log — candidates for the `mlx-porting` skill

> Append here whenever this port surfaces something general enough to fold back into
> `~/.claude/skills/mlx-porting`. Mark **status**: `proposed` (verify before upstreaming),
> `confirmed` (validated this port), `done` (already merged into the skill). Note the
> target file (SKILL.md or references/*.md).

### F1 — `uv` venvs have no `pip`; parity install instructions assume `pip`
- **status:** confirmed
- **target:** `references/parity-testing.md`, `references/repo-layout.md`
- The skill says `pip install -e ".[parity]"`. On a `uv venv` there is no `pip`/`python -m pip`;
  it fails with "No module named pip". Use `uv pip install -e ".[parity]"`. Suggest the skill
  show both, or detect the venv manager.

### F2 — Add `accelerate` to the `[parity]` extra
- **status:** confirmed
- **target:** `references/parity-testing.md` (parity deps), `references/repo-layout.md` pyproject
- Loading a large PT reference (here 20B encoder) without `accelerate` forces
  `low_cpu_mem_usage=False` → slower, more peak RAM. The reference loaders strongly recommend it.
  Parity extras should include `accelerate` alongside torch/transformers/diffusers.

### F3 — Running a `gpt-oss` (MXFP4) text encoder as a PT reference on Apple/CPU
- **status:** confirmed
- **target:** `references/common-pitfalls.md` (new subsection: gpt-oss-conditioned ports)
- gpt-oss-20b ships MXFP4 on the hub; MXFP4 kernels need Hopper+ GPUs. To produce goldens on
  Apple/CPU, load with `transformers.Mxfp4Config(dequantize=True)` (→ bf16). Without it the
  loader keeps MXFP4 and fails off-GPU. The MLX side (mlx-lm) handles MXFP4 natively, so the
  parity comparison is bf16(PT-dequant) vs MLX — set thresholds accordingly.

### F4 — Read the pipeline's guidance math; CFG is often NOT vanilla
- **status:** confirmed
- **target:** `references/common-pitfalls.md` (new trap)
- Lens uses **norm-rescaled CFG**: `comb = uncond + g·(cond−uncond)`, then rescale per-token by
  `‖cond‖/‖comb‖`. A vanilla-CFG assumption silently shifts output magnitude. Generalize:
  always port the *exact* guidance combine step, including norm/rescale/clamp tricks, not a
  textbook CFG. (Companion: timestep scaling can be split across modules — Lens has `scale=1000`
  in the time-proj AND `timestep/1000` in the loop; they compose.)

### F5 — Golden capture from inside a denoise loop via forward hooks
- **status:** confirmed
- **target:** `references/parity-testing.md` (technique)
- To capture DiT input/output at a real denoising step (production magnitudes, not random
  weights), register `module.register_forward_pre_hook(fn, with_kwargs=True)` +
  `register_forward_hook(fn, with_kwargs=True)` and snapshot the first call only. Cleaner than
  refactoring the reference. Pattern is in `tests/parity/capture_goldens.py`.

### F6 — diffusers custom-class pipelines load fine for golden capture if the ref pkg is importable
- **status:** confirmed
- **target:** `references/parity-testing.md`
- A pipeline with custom component classes (`LensTransformer2DModel`, `LensGptOssEncoder` named
  in `model_index.json`) loads via the reference's own `from_pretrained` once the reference
  package is on `sys.path` (`sys.path.insert(0, refs/<Model>)`). No `trust_remote_code` dance
  needed for local golden capture.

### F8 — Parity thresholds must be relative when activation magnitudes are large
- **status:** confirmed
- **target:** `references/parity-testing.md` (threshold table)
- The skill's table gives **absolute** `max_abs` thresholds (1e-3 single layer, etc.). Lens
  encoder features run to absmax ~1.07e4 at the deepest captured layer (pre-final-norm GPT-OSS
  hidden states grow with depth: 234 → 920 → 2496 → 10690 across [5,11,17,23]). An absolute
  1e-3 gate is impossible there. Use **relative error** `max|Δ|/max|ref|` or cosine similarity,
  and state the regime: absolute thresholds assume O(1) activations. Capture intermediate
  magnitudes during golden dump to choose the gate.

### F9 — mlx-lm model defaults can silently drop rope scaling (YaRN) the HF *config class* injects
- **status:** confirmed — **this was the Phase 1 bug** (cost the most debugging)
- **target:** `references/common-pitfalls.md` (NEW trap, high priority) + `references/parity-testing.md`
- **Symptom:** bf16-vs-bf16 encoder parity stuck at cosine ~0.94 (NOT a quant gap); MLX
  activations ~half the reference magnitude, uniform from layer 0, cosine *rising* with depth
  (a giant shared outlier dim dominates cosine in mid layers and masks it).
- **Root cause:** GPT-OSS-20B uses YaRN rope (`attention_scaling`/mscale ≈ 1.3466, factor 32,
  original_max_position 4096). That's a `GptOssConfig` **class default** in HF transformers, so the
  checkpoint's `text_encoder/config.json` does **not** serialize `rope_scaling`/`rope_theta`.
  mlx-lm reads the json, sees nothing, and falls back to plain rope (mscale=1.0). The mscale
  multiplies cos/sin at every layer/position → uniform divergence.
- **The general trap:** never trust `config.json` for rope. Compare the **RESOLVED** rope on both
  sides: `pt_model.model.rotary_emb.attention_scaling` + `inv_freq[:8]` vs the mlx rope object's
  `mscale`/freqs. If the HF model object has rope params the json lacks, inject them into the mlx
  config (mlx-lm `load_model(path, model_config={...})`) before building the model.
- **Fix here:** `LensGptOssEncoder.from_pretrained` injects `GPT_OSS_YARN_ROPE` when the on-disk
  config omits `rope_scaling`. After fix: bf16 worst cosine 0.9983, MXFP4 0.9977 (quant gap is
  negligible — the 0.94 was ALL rope). Also revises F8: matched-precision parity is what exposed
  this; an absolute/relative-threshold debate was secondary.

### F7 — Confirmations of existing skill guidance (no action, evidence for the skill's claims)
- **status:** confirmed
- "Code is the oracle, not press" — T0 (`selected_layer_index=[5,11,17,23]`, not "4,12,18,24").
- DiT was pure Linear + RMSNorm (no Conv) → recipe is a near-straight key copy; the skill's
  "Linear/Embedding layout identical PT↔MLX" held, no transpose needed for the DiT.
- Reference defaults `device="cuda"` but threads `device=hidden_states.device`, so loading the
  model on CPU is sufficient — no monkeypatch. Worth noting: grep the reference for hardcoded
  `"cuda"` before a non-CUDA golden run.
