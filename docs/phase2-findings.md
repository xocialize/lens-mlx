# Phase 2 — DiT (LensTransformer2DModel)

**Status: GREEN, first-run parity.** The full 48-layer DiT matches the PT golden.

## Result

| test | metric | gate |
|---|---|---|
| RoPE embed (T2) cos/sin vs PT freqs_cis | max_abs < 1e-5 | ✅ |
| apply_rotary_emb_lens (T2) | max_abs 4.x e-? < 1e-4 | ✅ |
| AdaLN _modulate (T3) | max_abs < 1e-5 | ✅ |
| **full DiT forward** | **max_abs 8.3e-3, rel 1.7e-3, cosine 0.999999** | < 1e-2 ✅ |

Full-DiT input was the captured `dit_in_hidden`/`dit_in_timestep` + reconstructed
CFG-batched encoder features (positive golden feats + zero negative, matching
negative_prompt=""); output compared to `dit_out_noise`.

## How it went

- **Isomorphic structure → zero key friction.** Instantiated model: 1264 params;
  checkpoint: 1264 tensors; **0 missing / 0 extra / 0 shape mismatches.** Only two key
  renames (`img_mod.1`→`img_mod`, `txt_mod.1`→`txt_mod`, the flattened
  `Sequential(SiLU, Linear)`). No transpose anywhere (pure Linear + RMSNorm).
- **T2 complex RoPE → real interleaved worked first try.** `view_as_complex(x)*freqs_cis`
  became `out_r = x_r·cos − x_i·sin ; out_i = x_r·sin + x_i·cos` on the even/odd interleave;
  `LensEmbedRope` emits (cos, sin) instead of a complex tensor. The `scale_rope` neg/pos
  centered split ported directly. Validated <1e-4 before the block test.
- **LensEmbedRope made a plain class** (not nn.Module) so MLX doesn't collect its computed
  rope tables as parameters (upstream uses non-persistent buffers). See note below.

## Artifacts

- `lens_mlx/model/transformer.py` — the DiT (isomorphic to upstream).
- `lens_mlx/utils/weights.py` — `load_dit_weights` / `sanitize_dit_key` (the 2 renames).
- `tests/parity/test_dit_rope_parity.py` — T2/T3 micro-parity.
- `tests/parity/test_dit_parity.py` — full-DiT gate.

## Skill note
The isomorphic-naming hard rule paid for itself: the DiT loaded with zero key surgery and
passed full parity on the first run. A non-Module class is the clean way to hold computed
rope/constant buffers in MLX without them being treated as (missing) checkpoint parameters.
