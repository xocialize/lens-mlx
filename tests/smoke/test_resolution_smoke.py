"""Smoke test for the verbatim-ported resolution buckets (no torch / no MLX needed)."""

from lens_mlx import resolve_resolution, SUPPORTED_ASPECT_RATIOS, RESOLUTION_BUCKETS


def test_square_buckets():
    assert resolve_resolution(1024, "1:1") == (1024, 1024)
    assert resolve_resolution(1440, "1:1") == (1440, 1440)


def test_all_buckets_divisible_by_16():
    # Required so they tile cleanly into FLUX.2 VAE latents (vae_scale_factor=16).
    for table in RESOLUTION_BUCKETS.values():
        for h, w in table.values():
            assert h % 16 == 0 and w % 16 == 0


def test_aspect_ratio_is_w_to_h():
    # "16:9" is landscape -> width > height.
    h, w = resolve_resolution(1024, "16:9")
    assert w > h
    h, w = resolve_resolution(1024, "9:16")
    assert h > w


def test_nine_aspect_ratios():
    assert len(SUPPORTED_ASPECT_RATIOS) == 9
