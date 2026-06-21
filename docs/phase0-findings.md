# Phase 0 — Reference capture & config reconciliation

Reference oracle: `refs/Lens` (depth-1 clone of `github.com/microsoft/Lens`).
Checkpoint: `microsoft/Lens` on HF (public, ungated). Configs saved under `refs/configs/`.

## Config reconciliation vs handoff §2 — **all match**

`transformer/config.json` is byte-for-byte the registered defaults:

| field | checkpoint | §2 default | |
|---|---|---|---|
| patch_size | 2 | 2 | ✓ |
| in_channels / out_channels | 128 / 32 | 128 / 32 | ✓ |
| num_layers | 48 | 48 | ✓ |
| num_attention_heads / attention_head_dim | 24 / 64 | 24 / 64 | ✓ |
| inner_dim | 1536 | 1536 | ✓ (= 24·64) |
| enc_hidden_dim | 2880 | 2880 | ✓ (= GPT-OSS hidden) |
| axes_dims_rope | [8,28,28] | (8,28,28) | ✓ (sums to head_dim 64) |
| gate_mlp / rms_norm / multi_layer_encoder_feature | True | True | ✓ |
| **selected_layer_index** | **[5,11,17,23]** | (5,11,17,23) | ✓ **T0 RESOLVED — 0-indexed; press "4,12,18,24" is wrong** |

## Component facts captured

**Text encoder (`text_encoder/config.json`)** — GPT-OSS-20B:
- `num_hidden_layers=24`, `hidden_size=2880`, `head_dim=64`, heads 64 / kv 8 (GQA 8:1).
- MoE: `num_local_experts=32`, `num_experts_per_tok=4`.
- `sliding_window=128`, alternating `layer_types` = sliding/full (T8 — keep causal, don't bidirectional).
- `quantization_config.quant_method=mxfp4` (modules_to_not_convert: self_attn, mlp.router, embed_tokens, lm_head). mlx-lm handles MXFP4. (T7 = Swift-only config-parse shim, Phase 5.)
- selected layers [5,11,17,23]; 23 = last layer. Encoder runs full 24-layer stack, captures post-layer hidden states, **skips final RMSNorm + LM head**.

**VAE (`vae/config.json`)** — `AutoencoderKLFlux2`, `_name_or_path=black-forest-labs/FLUX.2-dev`:
- `batch_norm_eps=0.0001` (the T1 bn de-norm eps), `latent_channels=32` (×4 patchify = 128 DiT in_ch).
- block_out_channels [128,256,512,512], 4 down/up blocks, mid_block_add_attention, quant+post_quant conv.
- VAE conv stack 8× + pipeline 2×2 patchify = `vae_scale_factor=16`.
- **§8 license: VAE is FLUX.2-dev — do NOT re-host; `from_pretrained` pulls from source until license verified.**

**Scheduler (`scheduler/scheduler_config.json`)** — `FlowMatchEulerDiscreteScheduler`:
- `use_dynamic_shifting=True`, `time_shift_type=exponential`, `num_train_timesteps=1000`.
- Pipeline passes an **explicit precomputed `mu`** (T6 `compute_empirical_mu`) → `base_shift`/`max_shift` unused.
- Assembly: exponential time-shift(mu) + `sigmas=np.linspace(1.0, 1.0/N, N)`.

## Details NOT emphasized in the handoff (found while reading the reference)

1. **Norm-rescaled CFG (pipeline.py:502-511)** — not vanilla CFG. After `comb = uncond + g·(cond−uncond)`,
   rescale by `‖cond‖/‖comb‖` per-token: `noise_pred = comb * (cond_norm/comb_norm)`. Must port verbatim.
2. **Timestep scaling** — `time_proj` uses `scale=1000` (transformer.py:89) AND the pipeline passes
   `timestep/1000` (pipeline.py:498). Port both; they compose.
3. **Block forward returns `(encoder_hidden_states, hidden_states)`** — txt then img (transformer.py:362).
4. **QKV is stacked** `view(B,S,3,H,Dh).unbind(2)` — 3-axis before heads, NOT interleaved (transformer.py:232).
5. **AdaLN** `x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)`, gate on residual (transformer.py:330) — T3 confirmed.
6. **`scale_rope=True`** (transformer.py:409) — neg/pos freq split, h/w centered around 0 (T2).
7. **Patchify** `view(b,c,h//2,2,w//2,2).permute(0,1,3,5,2,4)` (pipeline.py:354) — T4 axis order.
8. **Chat template (T5)** — system+user+assistant-thinking turns, `apply_chat_template(tokenize=False)`,
   split on `<|return|>`, then `txt_offset=97` slices off the template prefix tokens (pipeline.py:163-206).

## Epsilon inventory (confirmed from code)
QK-norm `1e-5` · block norms `1e-6` · `txt_norm` `1e-5` · `norm_out` `1e-6` · VAE `batch_norm_eps` `1e-4`.

## Goldens captured (`goldens/lens_goldens.npz`)
Fixed config: prompt "A scenic landscape…", 512×512, 4 steps, cfg 4.0, seed 42, CPU,
fp32 DiT/VAE, bf16-dequant encoder.

| tensor | shape | absmax | note |
|---|---|---|---|
| text_feat_0..3 | (1,35,2880) | 234 / 920 / 2496 / 1.07e4 | 35 toks after offset-97 slice; magnitude grows with depth |
| text_mask | (1,35) | 1 | all valid |
| dit_in_hidden | (2,1024,128) | 4.63 | CFG-batched; 32×32 latent grid |
| dit_in_timestep | (2,) | 1.0 | first sigma=1.0 → t/1000 |
| dit_out_noise | (2,1024,128) | 4.93 | |
| final_latent | (1,1024,128) | 3.09 | |
| decoded_image | (1,3,512,512) | 1.08 | [-1,1], pre-clamp |

**Parity-gate note:** encoder features reach O(1e4) → gate Phase 1 on **relative error /
cosine**, not absolute `max_abs`.

## Env
venv `.venv` (py3.12): mlx 0.31.2, mlx-lm, mflux, transformers 5.9.0, huggingface_hub 1.17.0.
Parity extra: torch 2.12.0, diffusers 0.37.1, einops, accelerate.
