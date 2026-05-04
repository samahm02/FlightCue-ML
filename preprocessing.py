"""
Offline preprocessing pipeline for the FlightCue dataset.

Reads raw sensor recordings produced by SensorRecord (accelerometer + barometer),
resamples them to fixed-rate causal grids, extracts 154 features per sliding window,
assigns TAKEOFF / LANDING / OTHER labels by overlap with annotated event timestamps,
and writes three output CSVs:

    features_raw.csv      -- raw (un-imputed) feature table
    features_imputed.csv  -- NaNs replaced with 0, missingness indicator columns added
    features_scaled.csv   -- z-score scaled (only written with --write_scaled)

The feature extraction logic in this file is the authoritative reference for the
on-device Kotlin pipeline (FlightDetector / Features.kt). Any change here must be
mirrored there and verified by FeatureParityTest.

Key design decisions
--------------------
- Causal only: no look-ahead. All filters, EMA streams and window boundaries are
  computed using data available at the window end time.
- Observation mask: coverage is tracked from the resampler's own bin-fill record,
  not from NaN presence on the forward-filled output.
- Gravity alignment: EMA that skips NaN, so sensor gaps do not silently propagate
  a stale gravity estimate across large discontinuities.
- dhdt: preserves NaN for missing barometer segments; does not substitute zeros.
- dt_prev_end_s: grouped by (file_id, grid) separately, so base-grid and event-grid
  windows are never compared against each other.


Usage -- takeoff model dataset (25s base window)
-------------------------------------------------
    python3 preprocessing.py all_data/ \
        --win 25 --hop 25 --win_to 20 --hop_to 10 --win_ld 24 --hop_ld 12 \
        --takeoff_pre 10 --takeoff_post 60 \
        --landing_pre 60 --landing_post 20 \
        --drop_overlap --overlap_margin 8 \
        --coverage 0.8 --window_anchor right \
        --label_rule overlap \
        --pos_overlap_secs_min 4 --pos_overlap_frac_min 0.25 \
        --disable_hnm --nonflight_chunk_s 1800 \
        --out_dir all_data/preprocessed_25/

Usage -- landing model dataset (20s base window)
------------------------------------------------
    python3 preprocessing.py all_data/ \
        --win 20 --hop 20 --win_to 20 --hop_to 10 --win_ld 24 --hop_ld 12 \
        --takeoff_pre 10 --takeoff_post 60 \
        --landing_pre 60 --landing_post 20 \
        --drop_overlap --overlap_margin 8 \
        --coverage 0.8 --window_anchor right \
        --label_rule overlap \
        --pos_overlap_secs_min 4 --pos_overlap_frac_min 0.25 \
        --disable_hnm --nonflight_chunk_s 1800 \
        --out_dir all_data/preprocessed_20/

Input folder layout
-------------------
    all_data/
        flight/          <-- recordings that contain Takeoff / Landing markers
            *.txt
        nonflight/       <-- recordings with no event markers
            *.txt

Each .txt file uses the SensorRecord format:
    <ts_ms>:<ax>:<ay>:<az>      accelerometer line
    <ts_ms>:<pressure_hPa>      barometer line
    <marker_name>;<ts_ms>       event marker line (e.g. "Takeoff;1700000000000")
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from random import random as py_random, seed as pyseed
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# SPECTRAL BAND DEFINITIONS
# Must match Params.kt / FeatureMath.kt exactly.
# ============================================================================

# Frequency bands for accelerometer magnitude PSD (Hz).
ACCEL_BANDS_AIRCRAFT: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.3),
    (0.3, 1.0),
    (1.0, 3.0),
    (3.0, 8.0),
    (8.0, 15.0),
)

# Frequency bands for dynamic-component PSD (Hz).
DYN_BANDS_AIRCRAFT: Tuple[Tuple[float, float], ...] = (
    (0.0, 0.5),
    (0.5, 2.0),
    (2.0, 5.0),
    (5.0, 10.0),
    (10.0, 20.0),
)


# ============================================================================
# TIMESTAMP UNIT INFERENCE
# SensorRecord writes millisecond timestamps. Older recordings may differ.
# ============================================================================

UNIT_DIVISORS = {
    "s":  1.0,
    "ms": 1e3,
    "us": 1e6,
    "ns": 1e9,
}


def infer_time_unit_from_dense_stream(
    ts_array: Sequence[float],
    expected_hz_range: Tuple[float, float] = (0.5, 200.0),
) -> Tuple[Optional[str], float]:
    """
    Infer timestamp unit by examining the median inter-sample interval of a
    dense sensor stream and checking which unit produces a plausible sample rate.

    Returns (unit_string, divisor). Returns (None, 1.0) if inference fails.
    """
    ts = np.asarray(ts_array, dtype=float)
    if ts.size < 10:
        return None, 1.0

    dts = np.diff(ts)
    dts_pos = dts[dts > 0]
    if dts_pos.size < 5:
        return None, 1.0

    md = float(np.median(dts_pos))
    min_hz, max_hz = expected_hz_range

    for unit in ["s", "ms", "us", "ns"]:
        divisor = UNIT_DIVISORS[unit]
        md_s = md / divisor
        if md_s > 0:
            implied_hz = 1.0 / md_s
            if min_hz <= implied_hz <= max_hz:
                return unit, divisor

    return None, 1.0


def infer_time_unit_from_magnitude(ts_array: Sequence[float]) -> Tuple[str, float]:
    """
    Fallback unit inference from the order of magnitude of the timestamp values.
    Used when the stream is too sparse for rate-based inference.
    """
    ts = np.asarray(ts_array, dtype=float)
    valid_ts = ts[np.isfinite(ts)]
    if valid_ts.size == 0:
        return "ms", 1e3

    max_ts = float(np.max(np.abs(valid_ts)))

    if max_ts > 1e15:
        return "ns", 1e9
    elif max_ts > 1e12:
        if 1.4e12 < max_ts < 2.5e12:
            return "ms", 1e3
        return "ns", 1e9
    elif max_ts > 1e9:
        if 1.4e9 < max_ts < 2e9:
            return "s", 1.0
        return "ms", 1e3
    elif max_ts > 1e6:
        return "ms", 1e3
    else:
        return "s", 1.0


def validate_markers_within_sensor_span(
    markers_sec: Dict[str, float],
    sensor_t_start: float,
    sensor_t_end: float,
    file_path: str,
    tolerance_s: float = 60.0,
) -> Dict[str, float]:
    """
    Raise ValueError if any marker timestamp falls far outside the sensor data
    span, which would indicate a timestamp unit mismatch.
    """
    if not markers_sec:
        return markers_sec

    for name, t in markers_sec.items():
        if not np.isfinite(t):
            continue
        if t < (sensor_t_start - tolerance_s) or t > (sensor_t_end + tolerance_s):
            raise ValueError(
                f"Marker '{name}' at t={t:.1f}s is outside sensor span "
                f"[{sensor_t_start:.1f}, {sensor_t_end:.1f}]s in {file_path}. "
                f"Likely timestamp unit mismatch."
            )

    return markers_sec


# ============================================================================
# CAUSAL RESAMPLING WITH OBSERVATION MASK
# ============================================================================

@dataclass
class ResampleResult:
    """Output of causal_resample_with_mask."""
    grid_t: np.ndarray
    data: pd.DataFrame
    obs_mask: Dict[str, np.ndarray]  # True where a real sensor sample landed


def causal_resample_with_mask(
    df: pd.DataFrame,
    t_col: str,
    val_cols: Sequence[str],
    target_hz: float,
    big_gap_factor: float = 5.0,
    fill_method: str = "ffill",
) -> ResampleResult:
    """
    Resample irregular sensor data onto a uniform causal grid.

    Each grid bin is filled with the mean of all raw samples that fall into it.
    If a bin contains no raw sample, it is forward-filled (up to big_gap_factor *
    median_dt seconds), then left as NaN for longer gaps.

    obs_mask[col] is True for bins that received at least one real sample.
    Coverage must be computed from obs_mask, not from NaN presence on the
    forward-filled output.

    Parameters
    ----------
    df            : DataFrame with at least t_col and val_cols.
    t_col         : Name of the timestamp column (already in seconds).
    val_cols      : Sensor value columns to resample.
    target_hz     : Output sample rate in Hz.
    big_gap_factor: Forward-fill is cut off after this multiple of the median
                    inter-sample interval, leaving NaN for genuine sensor gaps.
    fill_method   : "ffill" (default) or "nan" (no fill).
    """
    empty_result = ResampleResult(
        grid_t=np.array([], dtype=float),
        data=pd.DataFrame(columns=[t_col] + list(val_cols)),
        obs_mask={c: np.array([], dtype=bool) for c in val_cols},
    )

    if df.empty:
        return empty_result

    sdf = df[[t_col] + list(val_cols)].copy().sort_values(t_col)
    sdf = sdf.groupby(t_col, as_index=False)[list(val_cols)].mean()

    t_raw = sdf[t_col].to_numpy(dtype=float)
    if t_raw.size == 0:
        return empty_result

    if target_hz <= 0:
        raise ValueError("target_hz must be > 0")

    step = 1.0 / float(target_hz)
    t0, tN = float(t_raw[0]), float(t_raw[-1])
    grid_t = np.arange(t0, tN + step / 2.0, step, dtype=float)
    n_bins = int(grid_t.size)

    if n_bins == 0:
        return empty_result

    md = _median_dt(t_raw)
    max_hold_s = (big_gap_factor * md) if (md is not None and md > 0) else 0.0

    out_data = pd.DataFrame({t_col: grid_t})
    obs_mask: Dict[str, np.ndarray] = {}

    def _bin_index(t: np.ndarray) -> np.ndarray:
        idx = np.floor((t - t0) / step).astype(np.int64)
        return np.clip(idx, 0, n_bins - 1)

    for c in val_cols:
        x_raw = sdf[c].to_numpy(dtype=float)
        finite = np.isfinite(t_raw) & np.isfinite(x_raw)

        bin_has_obs = np.zeros(n_bins, dtype=bool)

        if finite.sum() == 0:
            out_data[c] = np.full(n_bins, np.nan, dtype=float)
            obs_mask[c] = bin_has_obs
            continue

        t_f = t_raw[finite]
        x_f = x_raw[finite]
        b = _bin_index(t_f)

        sums = np.bincount(b, weights=x_f, minlength=n_bins).astype(float)
        cnts = np.bincount(b, weights=np.ones_like(x_f, dtype=float), minlength=n_bins).astype(float)

        mean = np.full(n_bins, np.nan, dtype=float)
        m = cnts > 0
        mean[m] = sums[m] / cnts[m]
        bin_has_obs = m.copy()
        obs_mask[c] = bin_has_obs

        if fill_method == "nan":
            out_data[c] = mean
            continue

        # Forward fill with gap cutoff.
        last_obs_t = np.full(n_bins, -np.inf, dtype=float)
        np.maximum.at(last_obs_t, b, t_f)

        vals = np.full(n_bins, np.nan, dtype=float)
        last_val = np.nan
        last_t = -np.inf

        if max_hold_s <= 0.0:
            for i in range(n_bins):
                mi = mean[i]
                if np.isfinite(mi):
                    last_val = float(mi)
                    last_t = float(last_obs_t[i])
                    vals[i] = last_val
                else:
                    vals[i] = np.nan
        else:
            for i in range(n_bins):
                mi = mean[i]
                if np.isfinite(mi):
                    last_val = float(mi)
                    last_t = float(last_obs_t[i])
                    vals[i] = last_val
                else:
                    if np.isfinite(last_val) and (grid_t[i] - last_t) <= max_hold_s:
                        vals[i] = last_val
                    else:
                        vals[i] = np.nan

        out_data[c] = vals

    return ResampleResult(grid_t=grid_t, data=out_data, obs_mask=obs_mask)


def coverage_from_obs_mask(obs_mask: np.ndarray) -> float:
    """Fraction of grid bins that received at least one real sensor sample."""
    if obs_mask.size == 0:
        return 0.0
    return float(np.mean(obs_mask.astype(float)))


def _median_dt(t: np.ndarray) -> Optional[float]:
    """Median positive inter-sample interval; None if fewer than 2 samples."""
    if t.size < 2:
        return None
    dt = np.diff(t)
    dt_pos = dt[dt > 0]
    if dt_pos.size == 0:
        return None
    return float(np.median(dt_pos))


# ============================================================================
# FILE PARSING
# ============================================================================

def parse_flight_file(path: str) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Parse a SensorRecord .txt file (or a compatible .csv).

    Returns
    -------
    markers_sec : dict mapping marker name to timestamp in seconds.
    accel_df    : DataFrame with columns [t, ax, ay, az], t in seconds.
    baro_df     : DataFrame with columns [t, p], t in seconds.
    """
    csv_result = _try_parse_csv_raw(path)
    if csv_result is not None:
        markers_raw, accel_df_raw, baro_df_raw = csv_result
    else:
        markers_raw, accel_df_raw, baro_df_raw = _parse_txt_raw(path)

    # Infer the timestamp unit from the dense accelerometer stream first,
    # then fall back to the barometer stream, then to magnitude-based heuristics.
    divisor = None
    unit = None

    if not accel_df_raw.empty and len(accel_df_raw) >= 10:
        unit, divisor = infer_time_unit_from_dense_stream(
            accel_df_raw["t"].to_numpy(dtype=float)
        )

    if unit is None and not baro_df_raw.empty and len(baro_df_raw) >= 10:
        unit, divisor = infer_time_unit_from_dense_stream(
            baro_df_raw["t"].to_numpy(dtype=float)
        )

    if unit is None:
        all_ts = []
        if not accel_df_raw.empty:
            all_ts.extend(accel_df_raw["t"].tolist())
        if not baro_df_raw.empty:
            all_ts.extend(baro_df_raw["t"].tolist())
        if markers_raw:
            all_ts.extend(markers_raw.values())
        unit, divisor = infer_time_unit_from_magnitude(all_ts) if all_ts else ("ms", 1e3)

    # Convert raw timestamps to seconds.
    if not accel_df_raw.empty:
        accel_df_raw["t"] = accel_df_raw["t"].astype(float) / divisor
        accel_df = accel_df_raw.sort_values("t").reset_index(drop=True)
    else:
        accel_df = pd.DataFrame(columns=["t", "ax", "ay", "az"])

    if not baro_df_raw.empty:
        baro_df_raw["t"] = baro_df_raw["t"].astype(float) / divisor
        baro_df = baro_df_raw.sort_values("t").reset_index(drop=True)
    else:
        baro_df = pd.DataFrame(columns=["t", "p"])

    markers_sec = {name: float(raw_t) / divisor for name, raw_t in markers_raw.items()}

    # Sanity check: markers should fall within (or near) the sensor data span.
    sensor_t_start = np.inf
    sensor_t_end = -np.inf
    if not accel_df.empty:
        sensor_t_start = min(sensor_t_start, float(accel_df["t"].iloc[0]))
        sensor_t_end = max(sensor_t_end, float(accel_df["t"].iloc[-1]))
    if not baro_df.empty:
        sensor_t_start = min(sensor_t_start, float(baro_df["t"].iloc[0]))
        sensor_t_end = max(sensor_t_end, float(baro_df["t"].iloc[-1]))

    if np.isfinite(sensor_t_start) and np.isfinite(sensor_t_end) and markers_sec:
        try:
            markers_sec = validate_markers_within_sensor_span(
                markers_sec, sensor_t_start, sensor_t_end, path
            )
        except ValueError as e:
            warnings.warn(str(e))
            markers_sec = {}

    return markers_sec, accel_df, baro_df


def _try_parse_csv_raw(
    path: str,
) -> Optional[Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]]:
    """
    Attempt to parse a CSV file. Returns None if the file is not a CSV or
    does not contain the expected columns.
    """
    if not path.lower().endswith(".csv"):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df is None or df.empty:
        return None

    cols = [c.strip().lower() for c in df.columns]
    col_map = {c.strip().lower(): c for c in df.columns}

    tcol = None
    for cand in ["t", "timestamp", "time"]:
        if cand in cols:
            tcol = col_map[cand]
            break
    if tcol is None:
        return None

    markers_raw: Dict[str, float] = {}
    name_col = None
    for cand in ["name", "marker", "event", "label"]:
        if cand in cols:
            name_col = col_map[cand]
            break
    if name_col is not None:
        sub = df[[name_col, tcol]].dropna()
        if not sub.empty:
            for nm, tt in zip(sub[name_col].astype(str).tolist(), sub[tcol].astype(float).tolist()):
                nm = nm.strip()
                if nm:
                    markers_raw[nm] = float(tt)

    accel_df = pd.DataFrame(columns=["t", "ax", "ay", "az"])
    if all(c in cols for c in ["ax", "ay", "az"]):
        accel_df = df[[tcol, col_map["ax"], col_map["ay"], col_map["az"]]].copy()
        accel_df.columns = ["t", "ax", "ay", "az"]
        accel_df = accel_df.dropna(subset=["t"]).copy()

    baro_df = pd.DataFrame(columns=["t", "p"])
    pcol = None
    for cand in ["p", "pressure", "baro", "barometer"]:
        if cand in cols:
            pcol = col_map[cand]
            break
    if pcol is not None:
        baro_df = df[[tcol, pcol]].copy()
        baro_df.columns = ["t", "p"]
        baro_df = baro_df.dropna(subset=["t"]).copy()

    if accel_df.empty and baro_df.empty and not markers_raw:
        return None

    return markers_raw, accel_df, baro_df


def _parse_txt_raw(
    path: str,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    """
    Parse a SensorRecord .txt file without converting timestamps.

    Line formats:
        <ts>:<ax>:<ay>:<az>    -- accelerometer (4 colon-separated fields)
        <ts>:<pressure>        -- barometer (2 colon-separated fields)
        <name>;<ts>            -- event marker (semicolon-separated)
    """
    markers_raw: Dict[str, float] = {}
    accel_ts, ax, ay, az = [], [], [], []
    baro_ts, pvals = [], []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.lower().startswith("mer data"):
                continue

            if ";" in line:
                parts = line.split(";")
                if len(parts) == 2:
                    name = parts[0].strip()
                    try:
                        ts = float(parts[1].strip())
                        markers_raw[name] = ts
                        continue
                    except ValueError:
                        pass

            if ":" in line:
                parts = line.split(":")
                try:
                    ts = float(parts[0].strip())
                except ValueError:
                    continue

                if len(parts) == 2:
                    try:
                        p = float(parts[1].strip())
                        baro_ts.append(ts)
                        pvals.append(p)
                        continue
                    except ValueError:
                        pass

                if len(parts) == 4:
                    try:
                        _ax = float(parts[1].strip())
                        _ay = float(parts[2].strip())
                        _az = float(parts[3].strip())
                        accel_ts.append(ts)
                        ax.append(_ax)
                        ay.append(_ay)
                        az.append(_az)
                        continue
                    except ValueError:
                        pass

    accel_df = (
        pd.DataFrame({"t": accel_ts, "ax": ax, "ay": ay, "az": az})
        if accel_ts
        else pd.DataFrame(columns=["t", "ax", "ay", "az"])
    )
    baro_df = (
        pd.DataFrame({"t": baro_ts, "p": pvals})
        if baro_ts
        else pd.DataFrame(columns=["t", "p"])
    )

    return markers_raw, accel_df, baro_df


# ============================================================================
# CAUSAL DERIVED STREAMS
# ============================================================================

def ema_series(x: np.ndarray, alpha: float) -> np.ndarray:
    """Standard EMA (no NaN handling). Use ema_series_skipnan for sensor data."""
    if x.size == 0:
        return x
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def ema_series_skipnan(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Causal EMA that skips NaN inputs.

    NaN inputs produce NaN outputs; the filter state is only updated on
    finite values. This ensures that sensor gaps do not silently propagate
    a stale EMA value across long discontinuities.

    This behaviour must match Kotlin's emaSkipNanOutNan / emaMeanSkipNanOutNan.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    y = np.full_like(x, np.nan, dtype=float)
    m = np.nan
    for i in range(x.size):
        xi = x[i]
        if not np.isfinite(xi):
            y[i] = np.nan
            continue
        m = xi if not np.isfinite(m) else alpha * xi + (1.0 - alpha) * m
        y[i] = m
    return y


def ema_mean(x: np.ndarray, hz: float, tau_s: float) -> np.ndarray:
    """EMA mean with time-constant tau_s seconds at sample rate hz."""
    alpha = 2.0 / (1.0 + tau_s * hz)
    return ema_series_skipnan(x, alpha)


def ema_var(x: np.ndarray, hz: float, tau_s: float) -> np.ndarray:
    """
    EMA variance with time-constant tau_s seconds.

    Computes an EMA of (x - mu)^2 where mu is the running EMA mean.
    NaN inputs are skipped; output is NaN for those positions.
    """
    alpha = 2.0 / (1.0 + tau_s * hz)
    x = np.asarray(x, dtype=float)

    m = np.full_like(x, np.nan, dtype=float)
    mu = np.nan
    for i in range(x.size):
        if np.isfinite(x[i]):
            mu = x[i] if not np.isfinite(mu) else alpha * x[i] + (1 - alpha) * mu
            m[i] = mu

    v = np.full_like(x, np.nan, dtype=float)
    var = 0.0
    for i in range(x.size):
        if np.isfinite(x[i]) and np.isfinite(m[i]):
            err = x[i] - m[i]
            var = alpha * (err * err) + (1 - alpha) * var
            v[i] = var

    return v


def ema_zscore_series(
    x: np.ndarray, hz: float, tau_s: float = 60.0, eps: float = 1e-6
) -> np.ndarray:
    """Causal EMA z-score normalisation. Not used in the exported feature set."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    alpha = 2.0 / (1.0 + tau_s * float(hz))
    z = np.full_like(x, np.nan, dtype=float)
    m = np.nan
    v = 0.0
    for i in range(x.size):
        xi = x[i]
        if not np.isfinite(xi):
            z[i] = np.nan
            continue
        if not np.isfinite(m):
            m = xi
            v = 0.0
        else:
            m = alpha * xi + (1.0 - alpha) * m
            err = xi - m
            v = alpha * (err * err) + (1.0 - alpha) * v
        z[i] = (xi - m) / math.sqrt(v + eps)
    return z


def compute_gravity_aligned_columns(
    rs_a: pd.DataFrame, accel_hz: float, tau_s: float = 0.6
) -> pd.DataFrame:
    """
    Compute gravity-aligned components avert and ahoriz.

    Gravity direction is estimated as the slow EMA of the raw accelerometer
    vector (tau = 0.6 s by default). The EMA uses ema_series_skipnan so that
    gaps in the resampled data do not silently extend the last-known gravity
    estimate across long sensor outages.

    avert  = component of the dynamic acceleration along the gravity axis.
    ahoriz = magnitude of the component perpendicular to gravity.

    Both are NaN where the gravity estimate or the raw axes are NaN.

    Must match Features.kt / deriveAccel.
    """
    if rs_a is None or rs_a.empty:
        return rs_a

    ax = rs_a["ax"].to_numpy(dtype=float)
    ay = rs_a["ay"].to_numpy(dtype=float)
    az = rs_a["az"].to_numpy(dtype=float)

    alpha = 2.0 / (1.0 + tau_s * float(accel_hz))

    gx = ema_series_skipnan(ax, alpha)
    gy = ema_series_skipnan(ay, alpha)
    gz = ema_series_skipnan(az, alpha)

    gnorm = np.sqrt(gx * gx + gy * gy + gz * gz)
    valid_g = np.isfinite(gnorm) & (gnorm > 1e-8)

    ux = np.full_like(gx, np.nan)
    uy = np.full_like(gy, np.nan)
    uz = np.full_like(gz, np.nan)
    ux[valid_g] = gx[valid_g] / gnorm[valid_g]
    uy[valid_g] = gy[valid_g] / gnorm[valid_g]
    uz[valid_g] = gz[valid_g] / gnorm[valid_g]

    valid = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az) & valid_g
    dx = np.full_like(ax, np.nan)
    dy = np.full_like(ay, np.nan)
    dz = np.full_like(az, np.nan)
    dx[valid] = ax[valid] - gx[valid]
    dy[valid] = ay[valid] - gy[valid]
    dz[valid] = az[valid] - gz[valid]

    dot = np.full_like(ax, np.nan)
    dot[valid] = dx[valid] * ux[valid] + dy[valid] * uy[valid] + dz[valid] * uz[valid]

    avert = dot
    ahoriz = np.full_like(ax, np.nan)
    ahoriz[valid] = np.sqrt(np.clip(
        dx[valid]**2 + dy[valid]**2 + dz[valid]**2 - dot[valid]**2,
        0.0, None,
    ))

    out = rs_a.copy()
    out["avert"] = avert
    out["ahoriz"] = ahoriz
    return out


def p_to_h(p: np.ndarray, p0: float) -> np.ndarray:
    """Convert pressure (hPa) to approximate altitude (m) using the ISA formula."""
    ratio = np.clip(p / p0, 1e-6, 100.0)
    return 44330.0 * (1.0 - np.power(ratio, 1.0 / 5.255))


def add_baro_dhdt_column(
    rs_b: pd.DataFrame, baro_hz: float, dhdt_tau_s: float = 2.0
) -> pd.DataFrame:
    """
    Compute the smoothed barometric climb rate dhdt (m/s).

    Steps:
    1. Convert pressure to a temporary altitude h_temp using the first valid
       pressure sample as a local reference. This reference is only used to
       compute the derivative and is not stored.
    2. Compute backward differences dh/dt, leaving NaN wherever either adjacent
       sample is NaN (gaps are not interpolated).
    3. Smooth with ema_series_skipnan (tau = dhdt_tau_s) to preserve NaN gaps.

    Must match Features.kt / deriveBaro.
    """
    if rs_b is None or rs_b.empty:
        return pd.DataFrame(columns=["t", "p", "dhdt"])

    p = rs_b["p"].to_numpy(dtype=float)
    valid_p = p[np.isfinite(p)]
    if valid_p.size == 0:
        out = rs_b.copy()
        out["dhdt"] = np.nan
        return out

    p0_temp = float(valid_p[0])
    h_temp = p_to_h(p, p0_temp)

    dt = 1.0 / float(baro_hz)

    dh = np.full_like(h_temp, np.nan)
    for i in range(1, h_temp.size):
        if np.isfinite(h_temp[i]) and np.isfinite(h_temp[i - 1]):
            dh[i] = (h_temp[i] - h_temp[i - 1]) / dt

    alpha = 2.0 / (1.0 + dhdt_tau_s * float(baro_hz))
    dh_s = ema_series_skipnan(dh, alpha)

    out = rs_b.copy()
    out["dhdt"] = dh_s
    return out


# ============================================================================
# FEATURE HELPERS - BASIC STATISTICS
# ============================================================================

def iqr(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def safe_std(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.std(x)) if x.size else float("nan")


def safe_skew(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean((x - mu) ** 3) / (sd ** 3))


def safe_kurtosis_excess(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan")
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd == 0.0 or not np.isfinite(sd):
        return float("nan")
    return float(np.mean((x - mu) ** 4) / (sd ** 4) - 3.0)


def peak_count(x: np.ndarray, k: float = 2.0) -> float:
    """Count samples more than k standard deviations from the mean."""
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    sd = float(np.std(x))
    if sd == 0.0 or not np.isfinite(sd):
        return 0.0
    return float(np.sum(np.abs(x - np.mean(x)) > k * sd))


def linear_slope(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if t.size < 2:
        return float("nan")
    if np.allclose(np.std(t), 0.0):
        return float("nan")
    try:
        return float(np.polyfit(t, y, 1)[0])
    except np.linalg.LinAlgError:
        return float("nan")


def zero_cross_rate_per_sec(x: np.ndarray, hz: float, thr: float = 0.02) -> float:
    """Zero-crossing rate (crossings per second), with a dead-zone of width thr."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    x = np.where(np.isfinite(x), x, 0.0)
    sig = np.where(np.abs(x) <= thr, 0.0, np.sign(x))
    zc = sum(
        1 for i in range(1, sig.size)
        if sig[i] != 0.0 and sig[i - 1] != 0.0 and sig[i] != sig[i - 1]
    )
    dur = sig.size / float(hz)
    return float(zc) / dur if dur > 0 else 0.0


def longest_run_seconds(mask: np.ndarray, hz: float) -> float:
    """Length of the longest contiguous True run in mask, in seconds."""
    if mask.size == 0:
        return 0.0
    max_len, cur = 0, 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            max_len = max(max_len, cur)
        else:
            cur = 0
    return float(max_len) / float(hz)


# ============================================================================
# FEATURE HELPERS - TREND
# ============================================================================

def half_diff(arr: np.ndarray) -> float:
    """Mean of the second half minus mean of the first half (finite values only)."""
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 4:
        return float("nan")
    h = n // 2
    return float(np.mean(arr[h:]) - np.mean(arr[:h]))


def third_diff(arr: np.ndarray) -> float:
    """Mean of the last third minus mean of the first third (finite values only)."""
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 6:
        return float("nan")
    a, b = n // 3, 2 * n // 3
    return float(np.mean(arr[b:]) - np.mean(arr[:a]))


# ============================================================================
# FEATURE HELPERS - RECENCY
# ============================================================================

def recent_vs_earlier_diff(arr: np.ndarray, recent_frac: float = 0.25) -> float:
    """Mean of the last recent_frac of the window minus the mean of the rest."""
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 4:
        return float("nan")
    recent_start = int(n * (1.0 - recent_frac))
    if recent_start < 1 or recent_start >= n:
        return float("nan")
    return float(np.mean(arr[recent_start:]) - np.mean(arr[:recent_start]))


def recent_slope(arr: np.ndarray, hz: float, recent_s: float = 2.0) -> float:
    """Linear slope over the last recent_s seconds of the window."""
    arr = np.asarray(arr, dtype=float)
    recent_samples = min(arr.size, max(3, int(recent_s * hz)))
    recent = arr[-recent_samples:]
    valid = np.isfinite(recent)
    if valid.sum() < 3:
        return float("nan")
    t = np.arange(recent_samples)[valid] / hz
    y = recent[valid]
    if np.std(t) < 1e-9:
        return float("nan")
    try:
        return float(np.polyfit(t, y, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return float("nan")


def recent_std(arr: np.ndarray, hz: float, recent_s: float = 2.0) -> float:
    arr = np.asarray(arr, dtype=float)
    recent_samples = min(arr.size, max(3, int(recent_s * hz)))
    recent = arr[-recent_samples:][np.isfinite(arr[-recent_samples:])]
    return float(np.std(recent)) if recent.size >= 2 else float("nan")


def recent_mean(arr: np.ndarray, hz: float, recent_s: float = 2.0) -> float:
    arr = np.asarray(arr, dtype=float)
    recent_samples = min(arr.size, max(1, int(recent_s * hz)))
    recent = arr[-recent_samples:][np.isfinite(arr[-recent_samples:])]
    return float(np.mean(recent)) if recent.size else float("nan")


def recent_max_abs(arr: np.ndarray, hz: float, recent_s: float = 2.0) -> float:
    arr = np.asarray(arr, dtype=float)
    recent_samples = min(arr.size, max(1, int(recent_s * hz)))
    recent = arr[-recent_samples:][np.isfinite(arr[-recent_samples:])]
    return float(np.max(np.abs(recent))) if recent.size else float("nan")


def ratio_recent_to_window(
    arr: np.ndarray, hz: float, recent_s: float = 2.0,
    stat: str = "std", eps: float = 1e-6,
) -> float:
    """Ratio of a statistic over the last recent_s seconds to the same stat over the full window."""
    arr = np.asarray(arr, dtype=float)
    arr_valid = arr[np.isfinite(arr)]
    if arr_valid.size < 4:
        return float("nan")
    recent_samples = min(arr.size, max(3, int(recent_s * hz)))
    recent_valid = arr[-recent_samples:][np.isfinite(arr[-recent_samples:])]
    if recent_valid.size < 2:
        return float("nan")
    if stat == "std":
        recent_val = float(np.std(recent_valid))
        window_val = float(np.std(arr_valid))
    elif stat == "mean":
        recent_val = float(np.mean(np.abs(recent_valid)))
        window_val = float(np.mean(np.abs(arr_valid)))
    elif stat == "rms":
        recent_val = float(np.sqrt(np.mean(recent_valid ** 2)))
        window_val = float(np.sqrt(np.mean(arr_valid ** 2)))
    else:
        return float("nan")
    if window_val < eps:
        return float("nan")
    return float(np.clip(recent_val / window_val, 0.0, 100.0))


# ============================================================================
# FEATURE HELPERS - SPECTRAL
# ============================================================================

def spectral_relative_power(
    x: np.ndarray,
    fs: float,
    bands: Sequence[Tuple[float, float]],
    do_psd: bool = True,
) -> Dict[str, float]:
    """
    Relative power in each frequency band using an exact n-point DFT (no zero-padding).

    The window is Hanning-tapered and mean-subtracted before the FFT.
    Power in each band is divided by the total power to give relative fractions.

    Requires at least 8 finite samples; returns NaN for all bands otherwise.
    Must match FeatureMath.kt / spectralRelativePower.
    """
    out: Dict[str, float] = {}
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if (not do_psd) or x.size < 8:
        for lo, hi in bands:
            out[f"pow_{lo:.1f}_{hi:.1f}"] = float("nan")
        return out

    y = x - float(np.mean(x))
    w = np.hanning(y.size)
    X = np.fft.rfft(y * w)
    freqs = np.fft.rfftfreq(y.size, d=1.0 / fs)
    psd = np.abs(X) ** 2
    total = float(np.sum(psd))
    if total <= 0.0 or not np.isfinite(total):
        for lo, hi in bands:
            out[f"pow_{lo:.1f}_{hi:.1f}"] = float("nan")
        return out
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        out[f"pow_{lo:.1f}_{hi:.1f}"] = float(np.sum(psd[mask]) / total)
    return out


def dominant_frequency(x: np.ndarray, fs: float, min_freq: float = 0.1) -> float:
    """Frequency bin with the highest power above min_freq Hz."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return float("nan")
    y = x - np.mean(x)
    X = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1.0 / fs)
    valid = freqs >= min_freq
    if not np.any(valid) or X[valid].max() == 0:
        return float("nan")
    return float(freqs[valid][np.argmax(X[valid])])


def spectral_centroid(x: np.ndarray, fs: float, min_freq: float = 0.1) -> float:
    """Power-weighted mean frequency above min_freq Hz."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return float("nan")
    y = x - np.mean(x)
    X = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1.0 / fs)
    valid = freqs >= min_freq
    Xv, fv = X[valid], freqs[valid]
    total = Xv.sum()
    return float((Xv * fv).sum() / total) if total > 0 else float("nan")


def spectral_bandwidth(x: np.ndarray, fs: float, min_freq: float = 0.1) -> float:
    """Power-weighted standard deviation of frequency above min_freq Hz."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return float("nan")
    y = x - np.mean(x)
    X = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    freqs = np.fft.rfftfreq(len(y), d=1.0 / fs)
    valid = freqs >= min_freq
    Xv, fv = X[valid], freqs[valid]
    total = Xv.sum()
    if total == 0:
        return float("nan")
    centroid = (Xv * fv).sum() / total
    variance = (Xv * (fv - centroid) ** 2).sum() / total
    return float(np.sqrt(variance))


# ============================================================================
# WINDOWING AND LABELLING
# ============================================================================

@dataclass(frozen=True)
class WindowSpec:
    """Describes one sliding window."""
    ws: float     # window start time (seconds)
    we: float     # window end time / anchor (seconds)
    win_s: float  # window duration (seconds)
    hop_s: float  # hop size used for this grid (seconds)
    grid: str     # grid name: "base", "to", or "ld"
    grid_id: int  # 0 = base, 1 = takeoff-event, 2 = landing-event


def window_indices_meta(
    start_t: float, end_t: float,
    win_s: float, hop_s: float,
    grid: str, grid_id: int,
) -> List[WindowSpec]:
    """Generate all non-overlapping (or overlapping) windows for one grid."""
    if end_t <= start_t or win_s <= 0.0 or hop_s <= 0.0:
        return []
    starts = np.arange(start_t, end_t - win_s + 1e-9, hop_s, dtype=float)
    return [
        WindowSpec(ws=float(s), we=float(s + win_s),
                   win_s=float(win_s), hop_s=float(hop_s),
                   grid=str(grid), grid_id=int(grid_id))
        for s in starts
    ]


def build_union_windows_meta(t_start: float, t_end: float, p: "Params") -> List[WindowSpec]:
    """
    Build the union of windows across all three grids (base, to, ld).

    Duplicate (ws, we, grid_id) triples are deduplicated. The union grid
    approach matches UnionGridScheduler.kt on the device.
    """
    wins: List[WindowSpec] = []
    wins += window_indices_meta(t_start, t_end, p.win, p.hop, "base", 0)
    if p.win_to and p.hop_to:
        wins += window_indices_meta(t_start, t_end, p.win_to, p.hop_to, "to", 1)
    if p.win_ld and p.hop_ld:
        wins += window_indices_meta(t_start, t_end, p.win_ld, p.hop_ld, "ld", 2)

    out: List[WindowSpec] = []
    seen = set()
    for w in wins:
        key = (round(w.ws, 6), round(w.we, 6), w.grid_id)
        if key not in seen:
            seen.add(key)
            out.append(WindowSpec(
                ws=float(key[0]), we=float(key[1]),
                win_s=w.win_s, hop_s=w.hop_s,
                grid=w.grid, grid_id=w.grid_id,
            ))
    return out


def _overlap_seconds(ws: float, we: float, a: float, b: float) -> float:
    return max(0.0, min(we, b) - max(ws, a))


def label_by_anchor(
    anchor_t: float, markers: Dict[str, float], p: "Params"
) -> Optional[str]:
    """
    Assign a label to a window by checking whether the anchor time falls
    inside the event tolerance zone.

    Returns None for ambiguous windows (when drop_overlap is enabled and
    the anchor is close to both a takeoff and a landing marker).
    """
    candidates: List[Tuple[str, float]] = []
    if "Takeoff" in markers:
        to_t = markers["Takeoff"]
        if (anchor_t >= to_t - p.takeoff_pre) and (anchor_t <= to_t + p.takeoff_post):
            candidates.append(("TAKEOFF", abs(anchor_t - to_t)))
    if "Landing" in markers:
        ld_t = markers["Landing"]
        if (anchor_t >= ld_t - p.landing_pre) and (anchor_t <= ld_t + p.landing_post):
            candidates.append(("LANDING", abs(anchor_t - ld_t)))

    if p.mode == "to":
        candidates = [c for c in candidates if c[0] == "TAKEOFF"]
    elif p.mode == "ld":
        candidates = [c for c in candidates if c[0] == "LANDING"]

    if not candidates:
        return "OTHER"
    if p.drop_overlap and len(candidates) >= 2:
        candidates.sort(key=lambda kv: kv[1])
        if candidates[0][1] <= p.overlap_margin and candidates[1][1] <= p.overlap_margin:
            return None
    candidates.sort(key=lambda kv: kv[1])
    return candidates[0][0]


def label_by_overlap(
    ws: float, we: float, wc: float, markers: Dict[str, float], p: "Params"
) -> Optional[str]:
    """
    Assign a label by computing the overlap between the window and the event
    tolerance zone.

    A window is positive if the overlap (in seconds) meets both:
      - absolute threshold: >= pos_overlap_secs_min
      - relative threshold: >= pos_overlap_frac_min * window_duration

    Returns None for ambiguous windows when drop_overlap is enabled.
    """
    W = max(1e-6, we - ws)
    thr = max(p.pos_overlap_secs_min, p.pos_overlap_frac_min * W)
    cand: List[Tuple[str, float, float]] = []

    if "Takeoff" in markers:
        to_t = markers["Takeoff"]
        ov = _overlap_seconds(ws, we, to_t - p.takeoff_pre, to_t + p.takeoff_post)
        if ov >= thr:
            cand.append(("TAKEOFF", ov, abs(wc - to_t)))
    if "Landing" in markers:
        ld_t = markers["Landing"]
        ov = _overlap_seconds(ws, we, ld_t - p.landing_pre, ld_t + p.landing_post)
        if ov >= thr:
            cand.append(("LANDING", ov, abs(wc - ld_t)))

    if p.mode == "to":
        cand = [c for c in cand if c[0] == "TAKEOFF"]
    elif p.mode == "ld":
        cand = [c for c in cand if c[0] == "LANDING"]

    if not cand:
        return "OTHER"
    if p.drop_overlap and len(cand) >= 2:
        cand_sorted = sorted(cand, key=lambda t: t[2])
        if cand_sorted[0][2] <= p.overlap_margin and cand_sorted[1][2] <= p.overlap_margin:
            return None
    cand.sort(key=lambda t: (-t[1], t[2]))
    return cand[0][0]


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features_window(
    t_a: np.ndarray,
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    accel_hz: float,
    t_b: np.ndarray,
    p: np.ndarray,
    dhdt: np.ndarray,
    baro_hz: float,
    do_psd: bool = True,
    amag_w: Optional[np.ndarray] = None,
    amag_ema10: Optional[np.ndarray] = None,
    amag_emaVar10: Optional[np.ndarray] = None,
    p_ema30: Optional[np.ndarray] = None,
    p_ema30_dt1: Optional[np.ndarray] = None,
    avert: Optional[np.ndarray] = None,
    ahoriz: Optional[np.ndarray] = None,
    dyn_w: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Extract all 154 features from one window of resampled sensor data.

    All continuous streams (amag_w, dyn_w, amag_ema10, etc.) must be pre-computed
    over the entire recording and sliced to the window before calling. This ensures
    causal EMA state is carried forward correctly across windows.

    Parameters
    ----------
    t_a, ax, ay, az : Resampled accelerometer grid (window slice).
    accel_hz        : Accelerometer sample rate in Hz.
    t_b, p, dhdt   : Resampled barometer grid (window slice).
    baro_hz         : Barometer sample rate in Hz.
    do_psd          : Whether to compute spectral features.
    amag_w          : Accelerometer magnitude (window slice).
    amag_ema10      : EMA mean of amag with tau=10s (window slice).
    amag_emaVar10   : EMA variance of amag with tau=10s (window slice).
    p_ema30         : EMA mean of pressure with tau=30s (window slice).
    p_ema30_dt1     : Discrete derivative of p_ema30 (window slice).
    avert           : Vertical acceleration component (window slice).
    ahoriz          : Horizontal acceleration component (window slice).
    dyn_w           : Dynamic acceleration = amag - EMA_gravity (window slice).
    """
    feats: Dict[str, float] = {}

    # -- Accelerometer features --
    # m3 gates: require all three axes to be finite simultaneously.
    m3 = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    ax_u, ay_u, az_u = ax[m3], ay[m3], az[m3]

    if ax_u.size:
        mag = np.sqrt(ax_u**2 + ay_u**2 + az_u**2)

        for name, arr in [("ax", ax_u), ("ay", ay_u), ("az", az_u), ("amag", mag)]:
            feats[f"{name}_mean"]   = float(np.mean(arr))
            feats[f"{name}_std"]    = safe_std(arr)
            feats[f"{name}_min"]    = float(np.min(arr))
            feats[f"{name}_max"]    = float(np.max(arr))
            feats[f"{name}_median"] = float(np.median(arr))
            feats[f"{name}_iqr"]    = iqr(arr)
            feats[f"{name}_rms"]    = float(np.sqrt(np.mean(arr**2)))
            feats[f"{name}_range"]  = float(np.max(arr) - np.min(arr))
            feats[f"{name}_skew"]   = safe_skew(arr)
            feats[f"{name}_kurt"]   = safe_kurtosis_excess(arr)

        feats["amag_halfdiff"]          = half_diff(mag)
        feats["amag_thirddiff"]         = third_diff(mag)
        feats["amag_recent_diff_25pct"] = recent_vs_earlier_diff(mag, recent_frac=0.25)
        feats["amag_recent_slope_2s"]   = recent_slope(mag, accel_hz, recent_s=2.0)
        feats["amag_recent_slope_3s"]   = recent_slope(mag, accel_hz, recent_s=3.0)
        feats["amag_recent_std_2s"]     = recent_std(mag, accel_hz, recent_s=2.0)
        feats["amag_recent_mean_2s"]    = recent_mean(mag, accel_hz, recent_s=2.0)
        feats["amag_recent_max_abs_2s"] = recent_max_abs(mag, accel_hz, recent_s=2.0)
        feats["amag_ratio_recent_std"]  = ratio_recent_to_window(mag, accel_hz, recent_s=2.0, stat="std")

        if mag.size >= 2:
            dt = 1.0 / float(accel_hz)
            jerk = np.diff(mag) / dt
            feats["jerk_mean"]           = float(np.mean(np.abs(jerk)))
            feats["jerk_std"]            = safe_std(jerk)
            feats["jerk_rms"]            = float(np.sqrt(np.mean(jerk**2)))
            feats["jerk_recent_std_2s"]  = recent_std(jerk, accel_hz, recent_s=2.0)
            feats["jerk_recent_max_abs_2s"] = recent_max_abs(jerk, accel_hz, recent_s=2.0)
        else:
            for k in ["jerk_mean", "jerk_std", "jerk_rms",
                      "jerk_recent_std_2s", "jerk_recent_max_abs_2s"]:
                feats[k] = float("nan")

        if dyn_w is None:
            raise ValueError(
                "dyn_w must be supplied; compute from the full recording before windowing."
            )

        # dyn_thirddiff uses dyn_u (m3-gated, pre-finite-filter) to match
        # the Kotlin implementation exactly.
        dyn_u = dyn_w[m3]
        dyn_f = dyn_u[np.isfinite(dyn_u)]

        if dyn_f.size:
            feats["dyn_mean"]               = float(np.mean(dyn_f))
            feats["dyn_std"]                = safe_std(dyn_f)
            feats["dyn_rms"]                = float(np.sqrt(np.mean(dyn_f**2)))
            feats["dyn_max"]                = float(np.max(dyn_f))
            feats["dyn_range"]              = float(np.max(dyn_f) - np.min(dyn_f))
            feats["dyn_skew"]               = safe_skew(dyn_f)
            feats["dyn_kurt"]               = safe_kurtosis_excess(dyn_f)
            feats["dyn_peakcount"]          = peak_count(dyn_f, k=2.0)
            feats["dyn_halfdiff"]           = half_diff(dyn_f)
            feats["dyn_thirddiff"]          = third_diff(dyn_u)   # note: dyn_u, not dyn_f
            feats["dyn_recent_diff_25pct"]  = recent_vs_earlier_diff(dyn_f, recent_frac=0.25)
            feats["dyn_recent_slope_2s"]    = recent_slope(dyn_f, accel_hz, recent_s=2.0)
            feats["dyn_recent_std_2s"]      = recent_std(dyn_f, accel_hz, recent_s=2.0)
            feats["dyn_recent_max_abs_2s"]  = recent_max_abs(dyn_f, accel_hz, recent_s=2.0)
            feats["dyn_ratio_recent_std"]   = ratio_recent_to_window(dyn_f, accel_hz, recent_s=2.0, stat="std")
        else:
            for s in ["mean", "std", "rms", "max", "range", "skew", "kurt", "peakcount",
                      "halfdiff", "thirddiff", "recent_diff_25pct", "recent_slope_2s",
                      "recent_std_2s", "recent_max_abs_2s", "ratio_recent_std"]:
                feats[f"dyn_{s}"] = float("nan")

        feats["amag_peakcount"] = peak_count(mag, k=2.0)

        # Spectral features require at least 16 samples (matches Kotlin has_spectral gate).
        if mag.size >= 16:
            feats.update(spectral_relative_power(mag, fs=accel_hz, bands=ACCEL_BANDS_AIRCRAFT, do_psd=do_psd))
            feats["amag_dominant_freq"]       = dominant_frequency(mag, accel_hz)
            feats["amag_spectral_centroid"]   = spectral_centroid(mag, accel_hz)
            feats["amag_spectral_bandwidth"]  = spectral_bandwidth(mag, accel_hz)
            if dyn_f.size >= 16:
                dyn_psd = spectral_relative_power(dyn_f, fs=accel_hz, bands=DYN_BANDS_AIRCRAFT, do_psd=do_psd)
                for k, v in dyn_psd.items():
                    feats[f"dyn_{k}"] = v
                feats["dyn_dominant_freq"]     = dominant_frequency(dyn_f, accel_hz)
                feats["dyn_spectral_centroid"] = spectral_centroid(dyn_f, accel_hz)
            else:
                for lo, hi in DYN_BANDS_AIRCRAFT:
                    feats[f"dyn_pow_{lo:.1f}_{hi:.1f}"] = float("nan")
                feats["dyn_dominant_freq"] = feats["dyn_spectral_centroid"] = float("nan")

            p_ground = feats.get("pow_8.0_15.0", 0) + feats.get("pow_3.0_8.0", 0)
            p_air    = feats.get("pow_0.0_0.3", 0)  + feats.get("pow_0.3_1.0", 0)
            feats["ground_vs_air_ratio"] = (
                float(p_ground / (p_air + 1e-9)) if (p_ground + p_air) > 1e-9 else float("nan")
            )
            p_vibration = feats.get("pow_3.0_8.0", 0)
            p_static    = feats.get("pow_0.0_0.3", 0)
            feats["vibration_vs_static_ratio"] = (
                float(p_vibration / (p_static + 1e-9)) if (p_vibration + p_static) > 1e-9 else float("nan")
            )
        else:
            for lo, hi in ACCEL_BANDS_AIRCRAFT:
                feats[f"pow_{lo:.1f}_{hi:.1f}"] = float("nan")
            for lo, hi in DYN_BANDS_AIRCRAFT:
                feats[f"dyn_pow_{lo:.1f}_{hi:.1f}"] = float("nan")
            for k in ["amag_dominant_freq", "amag_spectral_centroid", "amag_spectral_bandwidth",
                      "dyn_dominant_freq", "dyn_spectral_centroid",
                      "ground_vs_air_ratio", "vibration_vs_static_ratio"]:
                feats[k] = float("nan")
    else:
        # No valid accelerometer data in this window.
        for name in ["ax", "ay", "az", "amag"]:
            for s in ["mean", "std", "min", "max", "median", "iqr", "rms", "range", "skew", "kurt"]:
                feats[f"{name}_{s}"] = float("nan")
        for s in ["halfdiff", "thirddiff", "recent_diff_25pct", "recent_slope_2s",
                  "recent_slope_3s", "recent_std_2s", "recent_mean_2s",
                  "recent_max_abs_2s", "ratio_recent_std"]:
            feats[f"amag_{s}"] = float("nan")
        for k in ["jerk_mean", "jerk_std", "jerk_rms",
                  "jerk_recent_std_2s", "jerk_recent_max_abs_2s"]:
            feats[k] = float("nan")
        for s in ["mean", "std", "rms", "max", "range", "skew", "kurt", "peakcount",
                  "halfdiff", "thirddiff", "recent_diff_25pct", "recent_slope_2s",
                  "recent_std_2s", "recent_max_abs_2s", "ratio_recent_std"]:
            feats[f"dyn_{s}"] = float("nan")
        feats["amag_peakcount"] = float("nan")
        for lo, hi in ACCEL_BANDS_AIRCRAFT:
            feats[f"pow_{lo:.1f}_{hi:.1f}"] = float("nan")
        for lo, hi in DYN_BANDS_AIRCRAFT:
            feats[f"dyn_pow_{lo:.1f}_{hi:.1f}"] = float("nan")
        for k in ["amag_dominant_freq", "amag_spectral_centroid", "amag_spectral_bandwidth",
                  "dyn_dominant_freq", "dyn_spectral_centroid",
                  "ground_vs_air_ratio", "vibration_vs_static_ratio"]:
            feats[k] = float("nan")

    # -- Gravity-aligned components --
    for name, arr_opt in [("avert", avert), ("ahoriz", ahoriz)]:
        if arr_opt is not None and arr_opt.size:
            a = arr_opt[np.isfinite(arr_opt)]
            feats[f"{name}_mean"]           = float(np.mean(a)) if a.size else float("nan")
            feats[f"{name}_std"]            = safe_std(a)
            feats[f"{name}_rms"]            = float(np.sqrt(np.mean(a**2))) if a.size else float("nan")
            feats[f"{name}_peakcount"]      = peak_count(a, k=2.0)
            feats[f"{name}_recent_slope_2s"] = recent_slope(a, accel_hz, recent_s=2.0)
            feats[f"{name}_recent_std_2s"]  = recent_std(a, accel_hz, recent_s=2.0)
        else:
            for s in ["mean", "std", "rms", "peakcount", "recent_slope_2s", "recent_std_2s"]:
                feats[f"{name}_{s}"] = float("nan")

    avr = feats.get("avert_rms", float("nan"))
    ahr = feats.get("ahoriz_rms", float("nan"))
    feats["avert_to_ahoriz_ratio"] = (
        float(np.clip(avr / ahr, 0.0, 100.0))
        if np.isfinite(avr) and np.isfinite(ahr) and ahr > 1e-6
        else float("nan")
    )

    # -- Barometer features --
    p_f = p[np.isfinite(p)]
    if p_f.size:
        feats["p_mean"]   = float(np.mean(p_f))
        feats["p_std"]    = safe_std(p_f)
        feats["p_min"]    = float(np.min(p_f))
        feats["p_max"]    = float(np.max(p_f))
        feats["p_median"] = float(np.median(p_f))
        feats["p_iqr"]    = iqr(p_f)
        feats["p_range"]  = float(np.max(p_f) - np.min(p_f))
        feats["p_skew"]   = safe_skew(p_f)
        feats["p_kurt"]   = safe_kurtosis_excess(p_f)
        if p_f.size >= 2:
            dt_b = 1.0 / float(baro_hz)
            dp = np.diff(p_f) / dt_b
            dp = np.concatenate(([dp[0]], dp))
            feats["dpdt_mean"] = float(np.mean(dp))
            feats["dpdt_std"]  = safe_std(dp)
        else:
            feats["dpdt_mean"] = feats["dpdt_std"] = float("nan")
        feats["p_slope"]            = linear_slope(t_b, p)
        feats["p_recent_diff_25pct"] = recent_vs_earlier_diff(p_f, recent_frac=0.25)
        feats["p_recent_slope_3s"]  = recent_slope(p_f, baro_hz, recent_s=3.0)
        feats["p_recent_slope_5s"]  = recent_slope(p_f, baro_hz, recent_s=5.0)
        feats["p_recent_std_3s"]    = recent_std(p_f, baro_hz, recent_s=3.0)
    else:
        for k in ["p_mean", "p_std", "p_min", "p_max", "p_median", "p_iqr", "p_range",
                  "p_skew", "p_kurt", "dpdt_mean", "dpdt_std", "p_slope",
                  "p_recent_diff_25pct", "p_recent_slope_3s", "p_recent_slope_5s", "p_recent_std_3s"]:
            feats[k] = float("nan")

    # dhdt features -- NaN is preserved for missing barometer segments.
    if dhdt.size:
        dhdt_arr = np.asarray(dhdt, dtype=float)
        finite = np.isfinite(dhdt_arr)
        if finite.sum() >= 2:
            v = dhdt_arr[finite]
            feats["dhdt_mean"]            = float(np.mean(v))
            feats["dhdt_std"]             = safe_std(v)
            feats["dhdt_maxabs"]          = float(np.max(np.abs(v)))
            feats["plateau_frac"]         = float(np.mean(np.abs(v) < 0.3))
            feats["dhdt_zcr_ps"]          = zero_cross_rate_per_sec(v, baro_hz, thr=0.05)
            feats["runlen_climb_s"]       = longest_run_seconds(finite & (dhdt_arr > 0.2), baro_hz)
            feats["runlen_descent_s"]     = longest_run_seconds(finite & (dhdt_arr < -0.2), baro_hz)
            feats["dhdt_recent_mean_2s"]  = recent_mean(v, baro_hz, recent_s=2.0)
            feats["dhdt_recent_mean_3s"]  = recent_mean(v, baro_hz, recent_s=3.0)
            feats["dhdt_recent_slope_3s"] = recent_slope(v, baro_hz, recent_s=3.0)
            feats["dhdt_recent_std_3s"]   = recent_std(v, baro_hz, recent_s=3.0)
            feats["dhdt_recent_max_abs_2s"] = recent_max_abs(v, baro_hz, recent_s=2.0)
        else:
            for k in ["dhdt_mean", "dhdt_std", "dhdt_maxabs", "plateau_frac", "dhdt_zcr_ps",
                      "runlen_climb_s", "runlen_descent_s", "dhdt_recent_mean_2s",
                      "dhdt_recent_mean_3s", "dhdt_recent_slope_3s", "dhdt_recent_std_3s",
                      "dhdt_recent_max_abs_2s"]:
                feats[k] = float("nan")
    else:
        for k in ["dhdt_mean", "dhdt_std", "dhdt_maxabs", "plateau_frac", "dhdt_zcr_ps",
                  "runlen_climb_s", "runlen_descent_s", "dhdt_recent_mean_2s",
                  "dhdt_recent_mean_3s", "dhdt_recent_slope_3s", "dhdt_recent_std_3s",
                  "dhdt_recent_max_abs_2s"]:
            feats[k] = float("nan")

    # -- EMA-smoothed stream features --
    if amag_ema10 is not None and amag_ema10.size:
        a = amag_ema10[np.isfinite(amag_ema10)]
        feats["amag_ema10_mean"]      = float(np.mean(a)) if a.size else float("nan")
        feats["amag_ema10_std"]       = safe_std(a)
        feats["amag_ema10_range"]     = float(np.max(a) - np.min(a)) if a.size else float("nan")
        feats["amag_ema10_thirddiff"] = third_diff(a)
    else:
        for s in ["mean", "std", "range", "thirddiff"]:
            feats[f"amag_ema10_{s}"] = float("nan")

    if amag_emaVar10 is not None and amag_emaVar10.size:
        a = amag_emaVar10[np.isfinite(amag_emaVar10)]
        feats["amag_emaVar10_mean"] = float(np.mean(a)) if a.size else float("nan")
        feats["amag_emaVar10_iqr"]  = iqr(a)
    else:
        feats["amag_emaVar10_mean"] = feats["amag_emaVar10_iqr"] = float("nan")

    if p_ema30 is not None and p_ema30.size:
        a = p_ema30[np.isfinite(p_ema30)]
        feats["p_ema30_mean"]  = float(np.mean(a)) if a.size else float("nan")
        feats["p_ema30_std"]   = safe_std(a)
        feats["p_ema30_slope"] = linear_slope(t_b, p_ema30)
    else:
        feats["p_ema30_mean"] = feats["p_ema30_std"] = feats["p_ema30_slope"] = float("nan")

    if p_ema30_dt1 is not None and p_ema30_dt1.size:
        a = p_ema30_dt1[np.isfinite(p_ema30_dt1)]
        feats["p_ema30_dt1_mean"]    = float(np.mean(a)) if a.size else float("nan")
        feats["p_ema30_dt1_std"]     = safe_std(a)
        feats["p_ema30_dt1_posfrac"] = float(np.mean(a > 0.0)) if a.size else float("nan")
    else:
        feats["p_ema30_dt1_mean"] = feats["p_ema30_dt1_std"] = feats["p_ema30_dt1_posfrac"] = float("nan")

    # -- Band ratio features --
    def _safe_div(a: float, b: float, max_val: float = 100.0) -> float:
        if not (np.isfinite(a) and np.isfinite(b)) or b < 1e-9:
            return float("nan")
        return float(np.clip(a / b, 0.0, max_val))

    feats["amag_ratio_1_3__0.3_1"]  = _safe_div(
        feats.get("pow_1.0_3.0", float("nan")),
        feats.get("pow_0.3_1.0", float("nan")),
    )
    feats["dyn_ratio_2_5__0.5_2"] = _safe_div(
        feats.get("dyn_pow_2.0_5.0", float("nan")),
        feats.get("dyn_pow_0.5_2.0", float("nan")),
    )

    return feats


# ============================================================================
# PARAMETERS
# ============================================================================

@dataclass(frozen=True)
class Params:
    """All preprocessing parameters, collected in one immutable dataclass."""
    accel_hz: float
    baro_hz: float
    big_gap_factor: float
    fill_method: str
    coverage: float
    win: float
    hop: float
    win_to: float
    hop_to: float
    win_ld: float
    hop_ld: float
    window_anchor: str
    label_rule: str
    pos_overlap_secs_min: float
    pos_overlap_frac_min: float
    takeoff_pre: float
    takeoff_post: float
    landing_pre: float
    landing_post: float
    drop_overlap: bool
    overlap_margin: float
    mode: str
    no_psd: bool
    robust_per_flight: bool
    robust_tau_s: float
    hnm_seconds: Optional[float]
    hnm_far_keep: float
    hnm_scheme: str
    hnm_near_s: float
    hnm_mid_s: float
    hnm_keep_near: float
    hnm_keep_mid: float
    nonflight_label: str
    nonflight_chunk_s: float
    nonflight_keep_prob: float
    export_resampled: bool
    out_dir_resampled: str
    out_dir: str


# ============================================================================
# NAN HANDLING AND IMPUTATION
# ============================================================================

def add_missingness_indicators(
    df: pd.DataFrame, feature_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Add binary missingness indicator columns and impute remaining NaNs with 0.

    Indicators are added for the four sensor groups (accel, baro, spectral, dyn)
    and two coverage columns (accel_coverage, baro_coverage) are computed as the
    fraction of non-NaN feature values within each group.

    All inf values are converted to NaN before indicators are computed.
    NaN imputation (fill with 0) happens after indicators are recorded.

    Returns the updated DataFrame and the extended list of feature column names.
    """
    df = df.copy()

    # Convert inf to NaN before any indicator or imputation step.
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    new_cols = []
    extra: dict = {}
    feature_groups = {
        "accel":    ["ax_mean", "ay_mean", "az_mean", "amag_mean"],
        "baro":     ["p_mean"],
        "spectral": ["amag_dominant_freq", "pow_0.0_0.3"],
        "dyn":      ["dyn_mean"],
    }

    for group_name, indicator_features in feature_groups.items():
        valid_cols = [c for c in indicator_features if c in df.columns]
        if valid_cols:
            col_name = f"has_{group_name}"
            extra[col_name] = df[valid_cols].notna().any(axis=1).astype(float)
            new_cols.append(col_name)

    accel_feats = [c for c in feature_cols if c.startswith(("ax_", "ay_", "az_", "amag_")) and c in df.columns]
    if accel_feats:
        extra["accel_coverage"] = df[accel_feats].notna().mean(axis=1)
        new_cols.append("accel_coverage")
    baro_feats = [c for c in feature_cols if c.startswith(("p_", "dhdt_")) and c in df.columns]
    if baro_feats:
        extra["baro_coverage"] = df[baro_feats].notna().mean(axis=1)
        new_cols.append("baro_coverage")
    if extra:
        df = pd.concat([df, pd.DataFrame(extra, index=df.index)], axis=1)
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)
    return df, feature_cols + new_cols


def compute_scaler(
    df: pd.DataFrame,
    feature_cols: List[str],
    passthrough_cols: Optional[List[str]] = None,
) -> Dict:
    """
    Compute z-score scaler statistics (mean, std) from a feature DataFrame.

    Passthrough columns (e.g. indicator flags) are stored with mean=0, std=1
    so they are left unchanged when the scaler is applied.

    Note: this scaler is NOT used by the GRU or XGBoost models. The GRU fits
    its own StandardScaler from training sequences, and the scaling is embedded
    inside the ONNX graph. This scaler is written to disk for reference only.
    """
    passthrough_cols = passthrough_cols or []
    fit_cols = [c for c in feature_cols if c not in passthrough_cols]
    means = df[fit_cols].mean(axis=0, skipna=True).to_dict()
    stds  = df[fit_cols].std(axis=0, ddof=0, skipna=True).to_dict()
    for c in passthrough_cols:
        means[c] = 0.0
        stds[c]  = 1.0
    return {"mean": means, "std": stds, "type": "zscore", "passthrough": passthrough_cols}


def compute_scaler_by_grid(
    df: pd.DataFrame, feature_cols: List[str], grid_col: str = "grid"
) -> Dict:
    """Compute per-grid scaler statistics (for reference; not used by models)."""
    out = {}
    for g, sub in df.groupby(grid_col):
        out[str(g)] = {
            "mean": sub[feature_cols].mean(axis=0, skipna=True).to_dict(),
            "std":  sub[feature_cols].std(axis=0, ddof=0, skipna=True).to_dict(),
        }
    return out


def apply_scaler(
    df: pd.DataFrame, scaler: Dict, feature_cols: List[str]
) -> pd.DataFrame:
    """Apply a pre-computed z-score scaler to a feature DataFrame."""
    df_scaled = df.copy()
    m = scaler.get("mean", {})
    s = scaler.get("std", {})
    passthrough = set(scaler.get("passthrough", []))
    for c in feature_cols:
        if c in passthrough:
            continue
        mu = float(m.get(c, 0.0))
        sd = float(s.get(c, 1.0))
        if (not np.isfinite(sd)) or abs(sd) < 1e-12:
            df_scaled[c] = 0.0
        else:
            df_scaled[c] = (df_scaled[c] - mu) / sd
    return df_scaled


# ============================================================================
# PER-FILE PROCESSING
# ============================================================================

def _domain_from_path(fp: str) -> str:
    """Infer flight / nonflight domain from the file path."""
    p = fp.replace("\\", "/").lower()
    if "/nonflight/" in p:
        return "nonflight"
    if "/flight/" in p:
        return "flight"
    return "unknown"


def _infer_domain(domain_hint: str, markers: Dict[str, float]) -> str:
    """Use path hint if conclusive; fall back to presence of event markers."""
    if domain_hint in ("flight", "nonflight"):
        return domain_hint
    if "Takeoff" in markers or "Landing" in markers:
        return "flight"
    return "nonflight"


def _make_nonflight_segments(
    t_start: float, t_end: float, chunk_s: float
) -> List[Tuple[float, float, int]]:
    """
    Split a long nonflight recording into chunks of at most chunk_s seconds.
    Each chunk gets a unique integer index for use in file_id construction.
    """
    if chunk_s <= 0.0 or t_end <= t_start:
        return [(t_start, t_end, 0)]
    segs, k, cur = [], 0, t_start
    while cur < t_end - 1e-9:
        nxt = min(t_end, cur + chunk_s)
        segs.append((cur, nxt, k))
        k += 1
        cur = nxt
    return segs


def process_one_file(
    fp: str, p: Params, rng_seed: Optional[int] = None
) -> List[Dict[str, float]]:
    """
    Process a single recording file end-to-end:

    1. Parse raw sensor data and event markers.
    2. Resample accelerometer to p.accel_hz and barometer to p.baro_hz.
    3. Compute derived continuous streams (amag, dyn, avert, ahoriz, dhdt, EMAs).
    4. Generate union sliding windows.
    5. For each window: check coverage, assign label, extract features.
    6. Return a list of row dicts (one per accepted window).
    """
    if rng_seed is not None:
        pyseed(rng_seed)

    domain_hint = _domain_from_path(fp)
    markers, accel_df, baro_df = parse_flight_file(fp)
    domain = _infer_domain(domain_hint, markers)

    if accel_df.empty and baro_df.empty:
        return []

    rs_a_result = causal_resample_with_mask(
        accel_df, "t", ["ax", "ay", "az"], p.accel_hz, p.big_gap_factor, p.fill_method
    )
    rs_b_result = causal_resample_with_mask(
        baro_df, "t", ["p"], p.baro_hz, p.big_gap_factor, p.fill_method
    )

    rs_a = rs_a_result.data
    obs_mask_a = rs_a_result.obs_mask
    rs_b = rs_b_result.data
    obs_mask_b = rs_b_result.obs_mask

    # Compute derived accelerometer streams over the full recording.
    if not rs_a.empty:
        ax = rs_a["ax"].to_numpy(dtype=float)
        ay = rs_a["ay"].to_numpy(dtype=float)
        az = rs_a["az"].to_numpy(dtype=float)
        m3 = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
        amag = np.full_like(ax, np.nan)
        amag[m3] = np.sqrt(ax[m3]**2 + ay[m3]**2 + az[m3]**2)
        rs_a["amag"]        = amag
        rs_a["amag_ema10"]  = ema_mean(amag, p.accel_hz, tau_s=10.0)
        rs_a["amag_emaVar10"] = ema_var(amag, p.accel_hz, tau_s=10.0)
        rs_a = compute_gravity_aligned_columns(rs_a, p.accel_hz, tau_s=0.6)

        alpha_dyn = 2.0 / (1.0 + 2.0 * float(p.accel_hz))
        g_hat = ema_series_skipnan(amag, alpha_dyn)
        rs_a["dyn"] = amag - g_hat

        # Observation mask: a bin must have real data on all three axes.
        obs_mask_accel = (
            obs_mask_a.get("ax", np.zeros(len(rs_a), dtype=bool))
            & obs_mask_a.get("ay", np.zeros(len(rs_a), dtype=bool))
            & obs_mask_a.get("az", np.zeros(len(rs_a), dtype=bool))
        )
    else:
        obs_mask_accel = np.array([], dtype=bool)

    # Compute derived barometer streams over the full recording.
    if not rs_b.empty:
        rs_b["p_ema30"] = ema_mean(rs_b["p"].to_numpy(dtype=float), p.baro_hz, tau_s=30.0)
        pm = rs_b["p_ema30"].to_numpy(dtype=float)
        if np.isfinite(pm).sum() >= 2:
            dt = 1.0 / float(p.baro_hz)
            dpm = np.diff(pm) / dt
            # Pad first element to match Python/Kotlin convention: dpm[0] = dpm[1].
            rs_b["p_ema30_dt1"] = np.concatenate(([dpm[0]], dpm))
        else:
            rs_b["p_ema30_dt1"] = np.nan

        rs_b = add_baro_dhdt_column(rs_b, p.baro_hz, dhdt_tau_s=2.0)
        obs_mask_baro = obs_mask_b.get("p", np.zeros(len(rs_b), dtype=bool))
    else:
        obs_mask_baro = np.array([], dtype=bool)

    if p.export_resampled:
        os.makedirs(p.out_dir_resampled, exist_ok=True)
        base = os.path.splitext(os.path.basename(fp))[0]
        if not rs_a.empty:
            rs_a.to_csv(
                os.path.join(p.out_dir_resampled, f"{base}_accel_{p.accel_hz:g}Hz.csv"),
                index=False,
            )
        if not rs_b.empty:
            rs_b.to_csv(
                os.path.join(p.out_dir_resampled, f"{base}_baro_{p.baro_hz:g}Hz.csv"),
                index=False,
            )

    # Determine the common time span across both sensors.
    t_start = t_end = None
    if not rs_a.empty:
        t_start = float(rs_a["t"].iloc[0])
        t_end   = float(rs_a["t"].iloc[-1])
    if not rs_b.empty:
        t_s = float(rs_b["t"].iloc[0])
        t_e = float(rs_b["t"].iloc[-1])
        t_start = t_s if t_start is None else min(t_start, t_s)
        t_end   = t_e if t_end is None else max(t_end, t_e)
    if t_start is None or t_end is None or t_end <= t_start:
        return []

    base_file = os.path.basename(fp)
    file_root = os.path.splitext(base_file)[0]

    # Nonflight recordings are split into fixed-length chunks to avoid
    # very long streams that would dominate the class balance.
    if domain == "nonflight":
        segs = _make_nonflight_segments(float(t_start), float(t_end), float(p.nonflight_chunk_s))
    else:
        segs = [(float(t_start), float(t_end), 0)]

    # Unpack full-recording arrays for slicing by window.
    if not rs_a.empty:
        ta = rs_a["t"].to_numpy(dtype=float)
        ax = rs_a["ax"].to_numpy(dtype=float)
        ay = rs_a["ay"].to_numpy(dtype=float)
        az = rs_a["az"].to_numpy(dtype=float)
        amag_all       = rs_a["amag"].to_numpy(dtype=float)
        dyn_all        = rs_a["dyn"].to_numpy(dtype=float)
        amag_ema10_all    = rs_a["amag_ema10"].to_numpy(dtype=float)
        amag_emaVar10_all = rs_a["amag_emaVar10"].to_numpy(dtype=float)
        avert_all      = rs_a.get("avert", pd.Series([], dtype=float)).to_numpy(dtype=float)
        ahoriz_all     = rs_a.get("ahoriz", pd.Series([], dtype=float)).to_numpy(dtype=float)
    else:
        ta = ax = ay = az = np.array([], dtype=float)
        amag_all = dyn_all = amag_ema10_all = amag_emaVar10_all = np.array([], dtype=float)
        avert_all = ahoriz_all = np.array([], dtype=float)

    if not rs_b.empty:
        tb = rs_b["t"].to_numpy(dtype=float)
        pp = rs_b["p"].to_numpy(dtype=float)
        vv = rs_b["dhdt"].to_numpy(dtype=float)
        p_ema30_all    = rs_b.get("p_ema30",    pd.Series([], dtype=float)).to_numpy(dtype=float)
        p_ema30_dt1_all = rs_b.get("p_ema30_dt1", pd.Series([], dtype=float)).to_numpy(dtype=float)
    else:
        tb = pp = vv = np.array([], dtype=float)
        p_ema30_all = p_ema30_dt1_all = np.array([], dtype=float)

    to_t = markers.get("Takeoff", None)
    ld_t = markers.get("Landing", None)
    rows: List[Dict[str, float]] = []

    for (seg_start, seg_end, seg_idx) in segs:
        max_win = max(p.win, p.win_to or 0.0, p.win_ld or 0.0)
        if (seg_end - seg_start) < (max_win + 1e-6):
            continue

        wins = build_union_windows_meta(seg_start, seg_end, p)
        if not wins:
            continue

        # Nonflight chunks get a unique file_id so the GRU treats each chunk
        # as a separate stream (no sequence crosses chunk boundaries).
        if domain == "nonflight" and len(segs) > 1:
            file_id = f"{file_root}#chunk{seg_idx:04d}"
        else:
            file_id = file_root

        ws_arr = np.array([w.ws for w in wins], dtype=float)
        we_arr = np.array([w.we for w in wins], dtype=float)

        if ta.size:
            a0 = np.searchsorted(ta, ws_arr, side="left")
            a1 = np.searchsorted(ta, we_arr, side="right")
        else:
            a0 = a1 = None

        if tb.size:
            b0 = np.searchsorted(tb, ws_arr, side="left")
            b1 = np.searchsorted(tb, we_arr, side="right")
        else:
            b0 = b1 = None

        for j, w in enumerate(wins):
            ws, we = w.ws, w.we
            if (we - ws) <= 0.0:
                continue

            wc_center = 0.5 * (ws + we)
            wc = float(we) if p.window_anchor == "right" else float(wc_center)

            # Slice accelerometer window.
            if ta.size:
                i0, i1 = int(a0[j]), int(a1[j])
                t_a_w         = ta[i0:i1]
                ax_w, ay_w, az_w = ax[i0:i1], ay[i0:i1], az[i0:i1]
                amag_w        = amag_all[i0:i1]         if amag_all.size        else np.array([], dtype=float)
                dyn_w         = dyn_all[i0:i1]          if dyn_all.size         else np.array([], dtype=float)
                amag_ema10_w  = amag_ema10_all[i0:i1]   if amag_ema10_all.size  else np.array([], dtype=float)
                amag_emaVar10_w = amag_emaVar10_all[i0:i1] if amag_emaVar10_all.size else np.array([], dtype=float)
                avert_w       = avert_all[i0:i1]        if avert_all.size       else np.array([], dtype=float)
                ahoriz_w      = ahoriz_all[i0:i1]       if ahoriz_all.size      else np.array([], dtype=float)
                obs_mask_accel_w = obs_mask_accel[i0:i1] if obs_mask_accel.size else np.array([], dtype=bool)
            else:
                t_a_w = ax_w = ay_w = az_w = np.array([], dtype=float)
                amag_w = dyn_w = amag_ema10_w = amag_emaVar10_w = np.array([], dtype=float)
                avert_w = ahoriz_w = np.array([], dtype=float)
                obs_mask_accel_w = np.array([], dtype=bool)

            # Slice barometer window.
            if tb.size:
                i0, i1 = int(b0[j]), int(b1[j])
                t_b_w       = tb[i0:i1]
                p_w, v_w    = pp[i0:i1], vv[i0:i1]
                p_ema30_w   = p_ema30_all[i0:i1]    if p_ema30_all.size    else np.array([], dtype=float)
                p_ema30_dt1_w = p_ema30_dt1_all[i0:i1] if p_ema30_dt1_all.size else np.array([], dtype=float)
                obs_mask_baro_w = obs_mask_baro[i0:i1] if obs_mask_baro.size else np.array([], dtype=bool)
            else:
                t_b_w = p_w = v_w = np.array([], dtype=float)
                p_ema30_w = p_ema30_dt1_w = np.array([], dtype=float)
                obs_mask_baro_w = np.array([], dtype=bool)

            # Coverage check: both sensors must have enough real observations.
            cov_ok = True
            accel_cov = baro_cov = 0.0
            if obs_mask_accel_w.size:
                accel_cov = coverage_from_obs_mask(obs_mask_accel_w)
                cov_ok &= accel_cov >= p.coverage
            if obs_mask_baro_w.size:
                baro_cov = coverage_from_obs_mask(obs_mask_baro_w)
                cov_ok &= baro_cov >= p.coverage
            if obs_mask_accel_w.size == 0 and obs_mask_baro_w.size == 0:
                cov_ok = False
            if not cov_ok:
                continue

            # Assign label.
            if domain == "nonflight":
                label = p.nonflight_label
            else:
                if p.label_rule == "overlap":
                    label = label_by_overlap(ws, we, wc, markers, p)
                else:
                    label = label_by_anchor(wc, markers, p)
                if label is None:
                    continue  # ambiguous window; discard

            if domain == "nonflight" and p.nonflight_keep_prob < 1.0:
                if py_random() > p.nonflight_keep_prob:
                    continue

            feats = extract_features_window(
                t_a_w, ax_w, ay_w, az_w, p.accel_hz,
                t_b_w, p_w, v_w, p.baro_hz,
                do_psd=(not p.no_psd),
                amag_w=amag_w,
                amag_ema10=amag_ema10_w,
                amag_emaVar10=amag_emaVar10_w,
                p_ema30=p_ema30_w,
                p_ema30_dt1=p_ema30_dt1_w,
                avert=avert_w,
                ahoriz=ahoriz_w,
                dyn_w=dyn_w,
            )

            if domain == "flight":
                d_to = abs(wc - to_t) if to_t is not None else np.nan
                d_ld = abs(wc - ld_t) if ld_t is not None else np.nan
                d_ev = np.nanmin([d_to, d_ld]) if (np.isfinite(d_to) or np.isfinite(d_ld)) else np.nan
            else:
                d_to = d_ld = d_ev = np.nan

            row: Dict[str, float] = {
                "domain":   domain,
                "file":     base_file,
                "file_id":  file_id,
                "grid":     w.grid,
                "grid_id":  float(w.grid_id),
                "win_s":    float(w.win_s),
                "hop_s":    float(w.hop_s),
                "t_start":  float(ws),
                "t_end":    float(we),
                "t_center": float(wc_center),
                "t_anchor": float(wc),
                "label":    label,
                "t_takeoff": float(to_t) if (to_t is not None and np.isfinite(to_t)) else np.nan,
                "t_landing": float(ld_t) if (ld_t is not None and np.isfinite(ld_t)) else np.nan,
                "dist_to_takeoff": float(d_to) if np.isfinite(d_to) else np.nan,
                "dist_to_landing": float(d_ld) if np.isfinite(d_ld) else np.nan,
                "dist_to_event":   float(d_ev) if np.isfinite(d_ev) else np.nan,
                "accel_obs_coverage": float(accel_cov),
                "baro_obs_coverage":  float(baro_cov),
            }
            row.update(feats)
            rows.append(row)

    return rows


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Offline preprocessing pipeline for FlightCue. "
            "Reads SensorRecord .txt files, extracts sliding-window features, "
            "and writes features_raw.csv and features_imputed.csv."
        )
    )

    ap.add_argument("folder", help="Root folder containing flight/ and nonflight/ subdirectories.")
    ap.add_argument("--out_dir", default=None, help="Output directory. Defaults to <folder>/preprocessed/.")

    # Sensor rates (must match the on-device Params.kt).
    ap.add_argument("--accel_hz", type=float, default=20.0, help="Accelerometer target rate (Hz). Default: 20.")
    ap.add_argument("--baro_hz",  type=float, default=1.0,  help="Barometer target rate (Hz). Default: 1.")
    ap.add_argument("--big_gap_factor", type=float, default=5.0,
                    help="Forward-fill cutoff: stop filling after this multiple of the median inter-sample interval.")
    ap.add_argument("--fill_method", choices=["nan", "ffill"], default="ffill")
    ap.add_argument("--coverage", type=float, default=0.8,
                    help="Minimum fraction of grid bins with real sensor data required to accept a window.")

    # Window grid parameters.
    ap.add_argument("--win",    type=float, default=10.0,  help="Base grid window duration (s).")
    ap.add_argument("--hop",    type=float, default=2.0,   help="Base grid hop size (s).")
    ap.add_argument("--win_to", type=float, default=20.0,  help="Takeoff-event grid window duration (s).")
    ap.add_argument("--hop_to", type=float, default=10.0,  help="Takeoff-event grid hop size (s).")
    ap.add_argument("--win_ld", type=float, default=24.0,  help="Landing-event grid window duration (s).")
    ap.add_argument("--hop_ld", type=float, default=12.0,  help="Landing-event grid hop size (s).")
    ap.add_argument("--window_anchor", choices=["center", "right"], default="right",
                    help="Whether to assign the label at the window centre or its right edge.")

    # Labelling.
    ap.add_argument("--label_rule", choices=["anchor", "overlap"], default="anchor")
    ap.add_argument("--pos_overlap_secs_min", type=float, default=3.0,
                    help="Minimum overlap in seconds for a positive label (overlap rule).")
    ap.add_argument("--pos_overlap_frac_min", type=float, default=0.25,
                    help="Minimum overlap as a fraction of window duration (overlap rule).")
    ap.add_argument("--takeoff_pre",  type=float, default=5.0,
                    help="Seconds before the takeoff marker included in the positive zone.")
    ap.add_argument("--takeoff_post", type=float, default=60.0,
                    help="Seconds after the takeoff marker included in the positive zone.")
    ap.add_argument("--landing_pre",  type=float, default=60.0)
    ap.add_argument("--landing_post", type=float, default=10.0)
    ap.add_argument("--drop_overlap", action="store_true",
                    help="Discard windows that overlap both a takeoff and a landing zone.")
    ap.add_argument("--overlap_margin", type=float, default=8.0,
                    help="Maximum anchor-to-event distance (s) for a window to be considered ambiguous.")
    ap.add_argument("--mode", choices=["all", "to", "ld"], default="all",
                    help="Restrict positive labels to one event type (for debugging).")

    # Parallelism and features.
    ap.add_argument("--n_workers", type=int, default=0, help="Number of parallel worker processes. 0 = serial.")
    ap.add_argument("--no_psd", action="store_true", help="Disable spectral features.")
    ap.add_argument("--robust_per_flight", action="store_true",
                    help="Compute per-flight EMA z-score streams (not used in exported feature set).")
    ap.add_argument("--robust_tau_s", type=float, default=60.0)

    # Hard-negative mining (disabled in production runs).
    ap.add_argument("--hnm_seconds",  type=float, default=300.0)
    ap.add_argument("--hnm_far_keep", type=float, default=0.05)
    ap.add_argument("--hnm_keep_far", type=float, default=None)
    ap.add_argument("--hnm_scheme",   choices=["flat", "strat"], default="flat")
    ap.add_argument("--hnm_near_s",   type=float, default=60.0)
    ap.add_argument("--hnm_mid_s",    type=float, default=180.0)
    ap.add_argument("--hnm_keep_near", type=float, default=1.0)
    ap.add_argument("--hnm_keep_mid",  type=float, default=0.5)
    ap.add_argument("--disable_hnm", action="store_true",
                    help="Disable hard-negative mining (recommended for production).")

    # Nonflight handling.
    ap.add_argument("--seed",                type=int,   default=42)
    ap.add_argument("--nonflight_label",     default="NONFLIGHT")
    ap.add_argument("--nonflight_chunk_s",   type=float, default=900.0,
                    help="Split nonflight recordings into chunks of this duration (s).")
    ap.add_argument("--nonflight_keep_prob", type=float, default=1.0,
                    help="Randomly drop nonflight windows with this probability (1.0 = keep all).")

    # Output options.
    ap.add_argument("--export_resampled", action="store_true",
                    help="Write resampled sensor CSVs to <out_dir>/resampled/.")
    ap.add_argument("--write_scaled", action="store_true",
                    help="Also write features_scaled.csv (z-score applied).")

    args = ap.parse_args()
    pyseed(args.seed)

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise SystemExit(f"Folder not found: {folder}")

    out_dir = args.out_dir or os.path.join(folder, "preprocessed")
    if args.hnm_keep_far is not None:
        args.hnm_far_keep = args.hnm_keep_far

    artifact_basenames = {
        "features_raw.csv", "features_scaled.csv", "features_imputed.csv",
        "scaler.json", "params.json",
    }

    def is_artifact(path: str) -> bool:
        bn = os.path.basename(path)
        try:
            common = os.path.commonpath([os.path.abspath(path), os.path.abspath(out_dir)])
        except ValueError:
            common = ""
        return (common == os.path.abspath(out_dir)) or (bn in artifact_basenames)

    cand_dirs = [
        os.path.join(folder, sub)
        for sub in ["flight", "nonflight"]
        if os.path.isdir(os.path.join(folder, sub))
    ]
    scan_roots = cand_dirs if cand_dirs else [folder]

    files: List[str] = []
    for root in scan_roots:
        for ext in ["*.txt", "*.csv"]:
            for fp in glob.iglob(os.path.join(root, "**", ext), recursive=True):
                if not is_artifact(fp):
                    files.append(fp)
    files = sorted(set(files))
    if not files:
        raise SystemExit("No .txt/.csv files found.")

    os.makedirs(out_dir, exist_ok=True)
    out_features_raw      = os.path.join(out_dir, "features_raw.csv")
    out_features_imputed  = os.path.join(out_dir, "features_imputed.csv")
    out_features_scaled   = os.path.join(out_dir, "features_scaled.csv")
    out_scaler            = os.path.join(out_dir, "scaler.json")
    out_params            = os.path.join(out_dir, "params.json")
    out_dir_resampled     = os.path.join(out_dir, "resampled")

    params = Params(
        accel_hz=args.accel_hz, baro_hz=args.baro_hz,
        big_gap_factor=args.big_gap_factor, fill_method=args.fill_method,
        coverage=args.coverage, win=args.win, hop=args.hop,
        win_to=args.win_to, hop_to=args.hop_to,
        win_ld=args.win_ld, hop_ld=args.hop_ld,
        window_anchor=args.window_anchor, label_rule=args.label_rule,
        pos_overlap_secs_min=args.pos_overlap_secs_min,
        pos_overlap_frac_min=args.pos_overlap_frac_min,
        takeoff_pre=args.takeoff_pre, takeoff_post=args.takeoff_post,
        landing_pre=args.landing_pre, landing_post=args.landing_post,
        drop_overlap=args.drop_overlap, overlap_margin=args.overlap_margin,
        mode=args.mode, no_psd=args.no_psd,
        robust_per_flight=args.robust_per_flight, robust_tau_s=args.robust_tau_s,
        hnm_seconds=None if args.disable_hnm else args.hnm_seconds,
        hnm_far_keep=max(0.0, min(1.0, float(args.hnm_far_keep))),
        hnm_scheme=args.hnm_scheme,
        hnm_near_s=float(args.hnm_near_s), hnm_mid_s=float(args.hnm_mid_s),
        hnm_keep_near=max(0.0, min(1.0, float(args.hnm_keep_near))),
        hnm_keep_mid=max(0.0, min(1.0, float(args.hnm_keep_mid))),
        nonflight_label=str(args.nonflight_label),
        nonflight_chunk_s=float(args.nonflight_chunk_s),
        nonflight_keep_prob=max(0.0, min(1.0, float(args.nonflight_keep_prob))),
        export_resampled=args.export_resampled,
        out_dir_resampled=out_dir_resampled,
        out_dir=out_dir,
    )

    all_rows: List[Dict[str, float]] = []
    t0 = time.perf_counter()

    if args.n_workers and args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
            futs = {
                ex.submit(process_one_file, fp, params, args.seed + i): i
                for i, fp in enumerate(files, 1)
            }
            for i, fut in enumerate(as_completed(futs), 1):
                try:
                    rows = fut.result()
                    all_rows.extend(rows)
                    print(f"[{i}/{len(files)}] +{len(rows)} windows (parallel)")
                except Exception as e:
                    print(f"[WARN] Parallel job failed: {e}")
    else:
        for i, fp in enumerate(files, 1):
            t_file = time.perf_counter()
            try:
                rows = process_one_file(fp, params, args.seed + i)
                all_rows.extend(rows)
                dt = time.perf_counter() - t_file
                print(f"[{i}/{len(files)}] {os.path.basename(fp)} -> {len(rows)} windows  ({dt:.2f}s)")
            except Exception as e:
                print(f"[WARN] Error processing {fp}: {e}")

    if not all_rows:
        raise SystemExit("No windows generated. Check --coverage, --win, and data paths.")

    df = pd.DataFrame(all_rows)

    # Sort by (file_id, grid, t_end) and compute per-grid temporal distance
    # between consecutive windows for use as a GRU sequence feature.
    df = df.sort_values(["file_id", "grid", "t_end", "t_start"]).reset_index(drop=True)
    df["dt_prev_end_s"]     = df.groupby(["file_id", "grid"])["t_end"].diff().astype(float).clip(lower=0.0)
    df["log_dt_prev_end_s"] = np.log1p(df["dt_prev_end_s"].fillna(0.0))
    df["log_win_s"]         = np.log1p(df["win_s"].astype(float))

    meta_cols = [
        "domain", "file", "file_id", "grid",
        "t_start", "t_end", "t_center", "t_anchor",
        "t_takeoff", "t_landing", "label",
        "dist_to_takeoff", "dist_to_landing", "dist_to_event",
    ]

    feat_cols = [c for c in df.columns if c not in meta_cols]

    df_raw = df[meta_cols + feat_cols].copy()
    df_raw.to_csv(out_features_raw, index=False)
    print(f"Wrote raw features: {out_features_raw}")

    df_imputed, feat_cols_with_indicators = add_missingness_indicators(df.copy(), feat_cols)
    df_imputed = df_imputed[meta_cols + feat_cols_with_indicators]
    df_imputed.to_csv(out_features_imputed, index=False)
    print(f"Wrote imputed features: {out_features_imputed}")

    # Report features with a high NaN rate before imputation (diagnostic).
    nan_pct = (df_raw[feat_cols].isna().sum() / len(df_raw) * 100).round(2)
    high_nan = nan_pct[nan_pct > 5].sort_values(ascending=False)
    if len(high_nan) > 0:
        print(f"\nFeatures with >5% NaN before imputation:")
        for feat, pct in high_nan.head(10).items():
            print(f"   {feat}: {pct:.1f}%")

    with open(out_params, "w", encoding="utf-8") as f:
        json.dump(params.__dict__, f, ensure_ascii=False, indent=2)
    print(f"Wrote params: {out_params}")

    passthrough = ["grid_id", "has_accel", "has_baro", "has_spectral", "has_dyn",
                   "accel_coverage", "baro_coverage"]
    numeric_feats = [
        c for c in feat_cols_with_indicators
        if pd.api.types.is_numeric_dtype(df_imputed[c])
    ]
    scaler = compute_scaler(
        df_imputed, numeric_feats,
        passthrough_cols=[c for c in passthrough if c in numeric_feats],
    )
    scaler["by_grid"] = compute_scaler_by_grid(df_imputed, numeric_feats, grid_col="grid")

    with open(out_scaler, "w", encoding="utf-8") as f:
        json.dump(scaler, f, ensure_ascii=False, indent=2)
    print(f"Wrote scaler: {out_scaler}")

    if args.write_scaled:
        df_scaled = apply_scaler(df_imputed, scaler, numeric_feats)
        df_scaled.to_csv(out_features_scaled, index=False)
        print(f"Wrote scaled features: {out_features_scaled}")

    print("\nLabel distribution:")
    total = len(df)
    for k in ["TAKEOFF", "LANDING", "OTHER", params.nonflight_label]:
        v = df["label"].value_counts().get(k, 0)
        print(f"  {k:10s}: {v:7d}  ({100.0 * v / total:5.2f}%)")
    print(f"  Total     : {total}")

    print("\nObservation-mask coverage:")
    print(f"  Accel: mean={df['accel_obs_coverage'].mean():.3f}  min={df['accel_obs_coverage'].min():.3f}")
    print(f"  Baro:  mean={df['baro_obs_coverage'].mean():.3f}  min={df['baro_obs_coverage'].min():.3f}")
    print(f"\nDone in {time.perf_counter() - t0:.1f}s. Output: {out_dir}")


if __name__ == "__main__":
    main()