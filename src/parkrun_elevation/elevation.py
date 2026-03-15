"""
Elevation lookup and gain/loss computation.

Pipeline for a single route:
    get_elevations()        — SRTM tile lookup via rasterio
        ↓
    interpolate_route()     — resample to regular 10m intervals
        ↓
    smooth_elevations()     — 70m moving-average to remove DEM noise
        ↓
    calc_elevation_stats()  — dead-band gain/loss accumulation
    calc_max_gradient()     — max gradient over a 100m rolling window

SRTM tiles are downloaded on demand from OpenTopography and cached locally in
the srtm/ directory as decompressed .hgt files (~25 MB each). UK coverage
requires ~110 tiles.

Important: rasterio.sample() takes (longitude, latitude) — not (lat, lng).
"""

import gzip
import logging
import shutil
from pathlib import Path

import httpx
import numpy as np
import rasterio
from haversine import haversine

logger = logging.getLogger(__name__)

# AWS open data — SRTM 1-arc-second (30m), no authentication required
# Path format: /skadi/{NS}{lat}/{NS}{lat}{EW}{lng}.hgt.gz
_TILE_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/"


# ---------------------------------------------------------------------------
# Tile utilities — pure functions
# ---------------------------------------------------------------------------

def tile_name(lat: float, lng: float) -> str:
    """Return the SRTM tile name covering the given coordinates.

    Tile names are derived from the south-west corner of each 1°×1° cell,
    using the floor of each coordinate.

    Examples:
        (51.5,  0.3) → "N51E000"
        (51.5, -1.3) → "N51W002"   # floor(-1.3) == -2
        (51.0, -1.0) → "N51W001"
    """
    lat_floor = int(np.floor(lat))
    lng_floor = int(np.floor(lng))

    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lng_floor >= 0 else "W"

    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lng_floor):03d}"


def tile_path(lat: float, lng: float, srtm_dir: Path) -> Path:
    """Return the local .hgt path for the tile covering (lat, lng)."""
    return srtm_dir / f"{tile_name(lat, lng)}.hgt"


def tile_url(lat: float, lng: float) -> str:
    """Return the AWS S3 HTTP URL for the SRTM tile covering (lat, lng)."""
    name = tile_name(lat, lng)
    # Tiles are organised into subdirectories by the NS+lat component: e.g. N51/N51W002.hgt.gz
    ns_dir = name[:3]
    return f"{_TILE_BASE_URL}{ns_dir}/{name}.hgt.gz"


# ---------------------------------------------------------------------------
# Tile download
# ---------------------------------------------------------------------------

def ensure_tile(lat: float, lng: float, srtm_dir: Path) -> Path:
    """Return the local .hgt path for this tile, downloading if necessary.

    The compressed .hgt.gz is downloaded, decompressed to .hgt, and the
    archive is removed. Subsequent calls return immediately.
    """
    path = tile_path(lat, lng, srtm_dir)
    if path.exists():
        return path

    srtm_dir.mkdir(parents=True, exist_ok=True)
    url = tile_url(lat, lng)
    gz_path = path.with_suffix(".hgt.gz")

    logger.info("Downloading SRTM tile %s", path.name)
    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as response:
        response.raise_for_status()
        with gz_path.open("wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)

    logger.info("Decompressing %s", gz_path.name)
    with gzip.open(gz_path, "rb") as f_in, path.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    gz_path.unlink()

    return path


# ---------------------------------------------------------------------------
# Elevation lookup
# ---------------------------------------------------------------------------

def get_elevations(
    coords: list[tuple[float, float]],
    srtm_dir: Path,
) -> list[float]:
    """Return SRTM elevations (metres) for a list of (lat, lng) pairs.

    Coords are grouped by tile so each .hgt file is opened only once.
    rasterio.sample() requires (lng, lat) order — note the swap below.
    """
    # Group indices by tile name so we open each file once
    tile_indices: dict[str, list[int]] = {}
    for i, (lat, lng) in enumerate(coords):
        name = tile_name(lat, lng)
        tile_indices.setdefault(name, []).append(i)

    results = [0.0] * len(coords)

    for name, indices in tile_indices.items():
        # Use the coordinates of the first point in the group to locate the tile
        sample_lat, sample_lng = coords[indices[0]]
        tile = ensure_tile(sample_lat, sample_lng, srtm_dir)

        with rasterio.open(tile) as src:
            # rasterio.sample() takes (lng, lat) — not (lat, lng)
            xy = [(coords[i][1], coords[i][0]) for i in indices]
            for i, (val,) in zip(indices, src.sample(xy)):
                results[i] = float(val)

    return results


# ---------------------------------------------------------------------------
# Algorithm functions — pure numpy, no I/O
# ---------------------------------------------------------------------------

def interpolate_route(
    lats: np.ndarray,
    lngs: np.ndarray,
    elevations: np.ndarray,
    interval_m: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample a route to regular spacing using cumulative haversine distance.

    Returns (new_lats, new_lngs, new_elevs, distances_m) where distances_m is
    the cumulative distance at each resampled point.

    At 10m intervals a 5km parkrun produces ~500 points — optimal for both
    SRTM resolution and API batching.
    """
    lats = np.asarray(lats, dtype=float)
    lngs = np.asarray(lngs, dtype=float)
    elevations = np.asarray(elevations, dtype=float)

    cum_dist = np.zeros(len(lats))
    for i in range(1, len(lats)):
        cum_dist[i] = cum_dist[i - 1] + haversine(
            (lats[i - 1], lngs[i - 1]),
            (lats[i], lngs[i]),
            unit="m",
        )

    target_dists = np.arange(0, cum_dist[-1], interval_m)
    new_lats = np.interp(target_dists, cum_dist, lats)
    new_lngs = np.interp(target_dists, cum_dist, lngs)
    new_elevs = np.interp(target_dists, cum_dist, elevations)

    return new_lats, new_lngs, new_elevs, target_dists


def smooth_elevations(
    elevations: np.ndarray,
    window_m: float = 70.0,
    interval_m: float = 10.0,
) -> np.ndarray:
    """Apply a moving-average to remove high-frequency DEM noise.

    Default window is 70m (7 points at 10m spacing), which smooths SRTM
    artefacts without washing out genuine terrain features.
    """
    elevations = np.asarray(elevations, dtype=float)
    window = max(1, min(int(round(window_m / interval_m)), len(elevations)))
    kernel = np.ones(window) / window
    return np.convolve(elevations, kernel, mode="same")


def calc_elevation_stats(
    elevations: np.ndarray,
    threshold: float = 3.0,
) -> dict:
    """Compute gain, loss, high, and low using a dead-band threshold filter.

    Only registers a gain or loss when the elevation has changed by at least
    threshold metres from the last confirmed level. This suppresses noise that
    survives smoothing.

    Use threshold=3.0 for SRTM-sourced elevation, 2.0 for Strava streams.
    """
    elevations = np.asarray(elevations, dtype=float)
    ascent = descent = 0.0
    last_confirmed = elevations[0]

    for elev in elevations[1:]:
        diff = elev - last_confirmed
        if diff >= threshold:
            ascent += diff
            last_confirmed = elev
        elif diff <= -threshold:
            descent += abs(diff)
            last_confirmed = elev

    return {
        "total_ascent_m": round(ascent, 1),
        "total_descent_m": round(descent, 1),
        "elevation_high_m": round(float(np.max(elevations)), 1),
        "elevation_low_m": round(float(np.min(elevations)), 1),
    }


def calc_max_gradient(
    elevations: np.ndarray,
    interval_m: float = 10.0,
    window_m: float = 100.0,
) -> float:
    """Return the maximum gradient (%) over any window_m-length section."""
    elevations = np.asarray(elevations, dtype=float)
    window_pts = int(window_m / interval_m)
    max_grad = 0.0
    for i in range(len(elevations) - window_pts):
        elev_diff = abs(elevations[i + window_pts] - elevations[i])
        grad = elev_diff / window_m * 100
        max_grad = max(max_grad, grad)
    return round(max_grad, 1)


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def compute_elevation_profile(
    lats: list[float],
    lngs: list[float],
    srtm_dir: Path,
    threshold: float = 3.0,
) -> dict:
    """Compute the full elevation profile for a route.

    Runs: SRTM lookup → interpolate to 10m → smooth 70m → stats + gradients.
    Returns a dict suitable for merging into a cache record.
    """
    coords = list(zip(lats, lngs))
    raw_elevations = get_elevations(coords, srtm_dir)

    lats_arr = np.array(lats)
    lngs_arr = np.array(lngs)
    elevs_arr = np.array(raw_elevations)

    _, _, elevs_interp, dists = interpolate_route(lats_arr, lngs_arr, elevs_arr)
    elevs_smooth = smooth_elevations(elevs_interp)

    stats = calc_elevation_stats(elevs_smooth, threshold=threshold)

    avg_abs_gradient = round(
        float(np.mean(np.abs(np.diff(elevs_smooth) / 10.0 * 100))), 2
    )
    max_gradient = calc_max_gradient(elevs_smooth)

    return {
        "distance_m": round(float(dists[-1]), 1),
        "elevation_source": "srtm30m",
        "smoothing_window_m": 70,
        "elevation_threshold_m": threshold,
        "max_gradient_window_m": 100,
        "avg_abs_gradient_pct": avg_abs_gradient,
        "max_gradient_pct": max_gradient,
        **stats,
    }
