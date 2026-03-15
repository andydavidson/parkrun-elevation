"""
Tests for elevation.py

Fast tests use only synthetic numpy data — no network, no SRTM tiles.

Live test (needs real tiles, hits the network):
    pytest -m live
"""

import numpy as np
import pytest

from parkrun_elevation.elevation import (
    calc_elevation_stats,
    calc_max_gradient,
    interpolate_route,
    smooth_elevations,
    tile_name,
    tile_url,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def straight_route(n_points, spacing_m, lat_start=51.5, lng_start=-1.3):
    """Build a simple straight-line route heading east."""
    # ~111,320m per degree of longitude at the equator; adjust for latitude
    lng_per_m = 1 / (111_320 * np.cos(np.radians(lat_start)))
    lats = np.full(n_points, lat_start)
    lngs = lng_start + np.arange(n_points) * spacing_m * lng_per_m
    return lats, lngs


# ---------------------------------------------------------------------------
# tile_name — pure function
# ---------------------------------------------------------------------------

def test_tile_name_northern_eastern():
    assert tile_name(51.5, 0.3) == "N51E000"


def test_tile_name_northern_western():
    # floor(-1.3) == -2, so the tile south-west corner is at lng=-2
    assert tile_name(51.5, -1.3) == "N51W002"


def test_tile_name_exactly_on_boundary():
    assert tile_name(51.0, -1.0) == "N51W001"


def test_tile_name_southern_western():
    # Southern hemisphere: floor(-33.8) == -34
    assert tile_name(-33.8, 18.5) == "S34E018"


def test_tile_name_zero_zero():
    assert tile_name(0.0, 0.0) == "N00E000"


def test_tile_url_contains_tile_name():
    url = tile_url(51.5, -1.3)
    assert "N51W002" in url
    assert url.endswith(".hgt.gz")


# ---------------------------------------------------------------------------
# interpolate_route
# ---------------------------------------------------------------------------

def test_interpolate_flat_route_elevations_unchanged():
    """A flat route should produce constant interpolated elevations."""
    lats, lngs = straight_route(6, spacing_m=100)
    elevs = np.zeros(6)
    _, _, new_elevs, _ = interpolate_route(lats, lngs, elevs, interval_m=10)
    np.testing.assert_allclose(new_elevs, 0.0, atol=1e-6)


def test_interpolate_steady_climb_stays_linear():
    """Linear elevation gain should remain linear after interpolation.

    We check that successive elevation differences are approximately equal
    (constant gradient), rather than comparing against absolute expected values
    that depend on exact haversine spacing.
    """
    lats, lngs = straight_route(6, spacing_m=100)
    elevs = np.linspace(0, 50, 6)  # 0m → 50m over 500m
    _, _, new_elevs, _ = interpolate_route(lats, lngs, elevs, interval_m=10)
    diffs = np.diff(new_elevs)
    # All step-wise gains should be equal to within floating-point tolerance
    np.testing.assert_allclose(diffs, diffs[0], rtol=0.01)


def test_interpolate_output_spacing():
    """Output points should be approximately interval_m apart."""
    lats, lngs = straight_route(3, spacing_m=100)
    elevs = np.array([0.0, 10.0, 20.0])
    _, _, _, dists = interpolate_route(lats, lngs, elevs, interval_m=10)
    gaps = np.diff(dists)
    np.testing.assert_allclose(gaps, 10.0, atol=0.1)


def test_interpolate_increases_point_count():
    """100m spacing → 10m spacing should give ~10x more points."""
    lats, lngs = straight_route(6, spacing_m=100)
    elevs = np.zeros(6)
    _, _, new_elevs, _ = interpolate_route(lats, lngs, elevs, interval_m=10)
    # 5 × 100m segments → ~50 output points (arange stops before the end)
    assert 45 <= len(new_elevs) <= 55


# ---------------------------------------------------------------------------
# smooth_elevations
# ---------------------------------------------------------------------------

def test_smooth_flat_unchanged():
    elevs = np.full(100, 50.0)
    smoothed = smooth_elevations(elevs)
    # Flat signal should survive convolution (edge effects aside)
    np.testing.assert_allclose(smoothed[3:-3], 50.0, atol=1e-6)


def test_smooth_reduces_noise():
    """Smoothing should reduce high-frequency noise.

    We compare the std of residuals from a linear trend (the noise component),
    not the raw std which is dominated by the trend itself.
    """
    rng = np.random.default_rng(42)
    x = np.linspace(0, 20, 200)
    noisy = x + rng.normal(scale=2.0, size=200)
    smoothed = smooth_elevations(noisy, window_m=70, interval_m=10)

    # Detrend by subtracting the linear baseline before comparing noise levels
    noisy_residuals = noisy[7:-7] - x[7:-7]
    smoothed_residuals = smoothed[7:-7] - x[7:-7]
    assert smoothed_residuals.std() < noisy_residuals.std() * 0.5


def test_smooth_three_points():
    """Short array (fewer points than window) should not crash."""
    elevs = np.array([10.0, 20.0, 30.0])
    result = smooth_elevations(elevs)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# calc_elevation_stats
# ---------------------------------------------------------------------------

def test_stats_flat_route():
    elevs = np.full(100, 25.0)
    stats = calc_elevation_stats(elevs, threshold=3.0)
    assert stats["total_ascent_m"] == 0.0
    assert stats["total_descent_m"] == 0.0


def test_stats_steady_climb():
    """10m increments well above threshold should all register as ascent."""
    elevs = np.arange(0, 110, 10, dtype=float)  # 0, 10, 20 … 100 → 100m gain
    stats = calc_elevation_stats(elevs, threshold=3.0)
    assert stats["total_ascent_m"] == pytest.approx(100.0)
    assert stats["total_descent_m"] == 0.0


def test_stats_steady_descent():
    elevs = np.arange(100, -10, -10, dtype=float)  # 100 → 0 → -10 doesn't matter
    stats = calc_elevation_stats(elevs, threshold=3.0)
    assert stats["total_descent_m"] == pytest.approx(100.0)
    assert stats["total_ascent_m"] == 0.0


def test_stats_noise_below_threshold_ignored():
    """Oscillations of ±1m (2m peak-to-peak) should be ignored with a 3m threshold."""
    base = 50.0
    elevs = np.array([base + (2 * (i % 2) - 1) * 1.0 for i in range(50)])
    stats = calc_elevation_stats(elevs, threshold=3.0)
    assert stats["total_ascent_m"] == 0.0
    assert stats["total_descent_m"] == 0.0


def test_stats_threshold_boundary_registers():
    """A change of exactly threshold should register."""
    elevs = np.array([0.0, 3.0, 0.0])
    stats = calc_elevation_stats(elevs, threshold=3.0)
    assert stats["total_ascent_m"] == pytest.approx(3.0)
    assert stats["total_descent_m"] == pytest.approx(3.0)


def test_stats_threshold_just_below_does_not_register():
    elevs = np.array([0.0, 2.9, 0.0])
    stats = calc_elevation_stats(elevs, threshold=3.0)
    assert stats["total_ascent_m"] == 0.0
    assert stats["total_descent_m"] == 0.0


def test_stats_high_low():
    elevs = np.array([10.0, 25.0, 5.0, 30.0, 15.0])
    stats = calc_elevation_stats(elevs, threshold=1.0)
    assert stats["elevation_high_m"] == pytest.approx(30.0)
    assert stats["elevation_low_m"] == pytest.approx(5.0)


def test_stats_strava_threshold():
    """2m threshold (Strava streams) should catch smaller changes."""
    elevs = np.array([0.0, 2.0, 0.0])
    stats = calc_elevation_stats(elevs, threshold=2.0)
    assert stats["total_ascent_m"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# calc_max_gradient
# ---------------------------------------------------------------------------

def test_max_gradient_flat():
    elevs = np.full(200, 50.0)
    assert calc_max_gradient(elevs) == 0.0


def test_max_gradient_known_slope():
    """10m rise over 100m distance = 10% gradient."""
    # 200 points at 10m = 2km; one 100m window rising 10m
    elevs = np.zeros(200)
    elevs[50:60] = np.linspace(0, 10, 10)  # 10m rise over 10 points × 10m
    elevs[60:] = 10.0
    result = calc_max_gradient(elevs, interval_m=10, window_m=100)
    assert result == pytest.approx(10.0, abs=0.5)


def test_max_gradient_short_array():
    """Array shorter than window should return 0.0 without error."""
    elevs = np.array([0.0, 5.0, 10.0])
    result = calc_max_gradient(elevs, interval_m=10, window_m=100)
    assert result == 0.0


# ---------------------------------------------------------------------------
# Live smoke test — downloads a real SRTM tile
# ---------------------------------------------------------------------------

@pytest.mark.live
def test_live_srtm_winchester(tmp_path):
    """Download the N51W002 tile and sample a point near Winchester.

    Winchester city centre is at roughly (51.06, -1.31), elevation ~100m asl.
    We accept a generous ±50m tolerance given SRTM geoid/ellipsoid offsets.
    """
    from parkrun_elevation.elevation import get_elevations

    coords = [(51.06, -1.31)]
    elevs = get_elevations(coords, srtm_dir=tmp_path)
    assert len(elevs) == 1
    assert 20 < elevs[0] < 150, f"Unexpected elevation for Winchester: {elevs[0]}m"
