"""
Generates a golden NPZ file for the end-to-end feature parity test in the
Android app (FeatureParityTest.kt).

This script mirrors the EXACT execution path of FlightDetector.tick() and
runStream() on the Python side, so the Kotlin JVM test can run the same
pipeline and compare every output value against these golden values.

Pipeline (identical to Kotlin):
    raw sensor data
      → causal_resample     (Resample.accel / Resample.baro)
      → derive signals      (Features.deriveAccel / Features.deriveBaro)
      → UnionGridScheduler  (same origin-based window emission as Kotlin)
      → window slicing
      → feature extraction
      → metadata injection
      → Scaler.scaleWindow

What this test verifies end-to-end:
    - Causal resampling (bin-mean, gap detection, forward-fill)
    - EMA / gravity alignment / dyn / dhdt / p_ema30
    - Window scheduling: Kotlin must emit the same (grid_id, t_end) pairs
    - All feature computations including spectral (exact n-point DFT)
    - has_spectral gate (requires >= 16 accel samples)
    - Z-score scaling with NaN imputation

Usage:
    python3 generate_feature_parity_golden.py path/to/recording.txt \\
        --model_dir mobile_export_final/takeoff_gru \\
        --out parity_golden.npz \\
        --n_windows 50

Outputs:
    parity_golden.npz              -- golden feature arrays
    parity_golden_feature_names.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional

import numpy as np

from preprosessing import (
    parse_flight_file,
    causal_resample_with_mask,
    compute_gravity_aligned_columns,
    add_baro_dhdt_column,
    ema_mean,
    ema_var,
    ema_series_skipnan,
    extract_features_window,
)


# ---------------------------------------------------------------------------
# Constants — must match Params.kt exactly
# ---------------------------------------------------------------------------

ACCEL_HZ       = 20.0   # accelerometer target sample rate
BARO_HZ        = 1.0    # barometer target sample rate
BIG_GAP_FACTOR = 5.0    # gap larger than this × expected interval triggers forward-fill
GRAV_TAU_S     = 0.6    # EMA time constant for gravity estimate
DYN_TAU_S      = 2.0    # EMA time constant for dynamic acceleration
DHDT_TAU_S     = 2.0    # EMA time constant for barometric rate of change
DO_PSD         = True   # whether to compute spectral features
COVERAGE_THR   = 0.80   # minimum fraction of observed (non-filled) samples to accept a window

# Grid definitions matching FlightDetector.kt (takeoff model).
# BASE win/hop must match toProfile.winLenSec / toProfile.hopSec.
BASE_WIN_S = 25.0
BASE_HOP_S = 25.0

GRIDS = [
    {"id": "BASE",     "win": BASE_WIN_S, "hop": BASE_HOP_S, "grid_id": 0},
    {"id": "TO_EVENT", "win": 20.0,       "hop": 10.0,       "grid_id": 1},
    {"id": "LD_EVENT", "win": 24.0,       "hop": 12.0,       "grid_id": 2},
]


# ---------------------------------------------------------------------------
# Python equivalent of Kotlin's UnionGridScheduler
# Mirrors UnionGridScheduler.kt exactly:
#   set_origin → next_end[grid] = round6(origin + win)
#   poll       → returns the grid with the smallest next_end <= now,
#                then advances it by one hop
# ---------------------------------------------------------------------------

def round6(x: float) -> float:
    """Round to 6 decimal places — matches Kotlin's scheduler rounding."""
    return round(x * 1_000_000) / 1_000_000


class UnionGridScheduler:
    """
    Emits windows from multiple overlapping grids in chronological order.

    Each grid fires its first window at origin + win_s, then every hop_s
    thereafter. When poll() is called with the current time, it returns the
    grid whose next window end is soonest and <= now, then advances that
    grid's pointer by one hop.
    """

    def __init__(self, grids: List[Dict]):
        self.grids    = grids
        self.next_end: Dict[str, float] = {}

    def set_origin(self, origin: float) -> None:
        for g in self.grids:
            self.next_end[g["id"]] = round6(origin + g["win"])

    def poll(self, now: float, eps: float = 1e-6) -> Optional[Dict]:
        """Return the next due window slot, or None if nothing is due yet."""
        best_id  = None
        best_end = math.inf
        for gid, end in self.next_end.items():
            if end < best_end:
                best_end = end
                best_id  = gid
        if best_id is None or best_end > now + eps:
            return None
        g = next(g for g in self.grids if g["id"] == best_id)
        self.next_end[best_id] = round6(best_end + g["hop"])
        return {
            "id":      best_id,
            "end":     best_end,
            "win":     g["win"],
            "hop":     g["hop"],
            "grid_id": g["grid_id"],
        }


def run_scheduler(origin: float, t_max: float, n_max: int) -> List[Dict]:
    """
    Drive the scheduler from origin to t_max and collect window slots.

    Advances time in 1-second steps; the scheduler catches up internally
    when multiple grids are due within the same step. Stops when n_max
    slots have been collected.
    """
    sched = UnionGridScheduler(GRIDS)
    sched.set_origin(origin)

    slots: List[Dict] = []
    t     = origin + min(g["win"] for g in GRIDS)
    step  = 1.0

    while t <= t_max + 1.0 and len(slots) < n_max:
        while True:
            w = sched.poll(t)
            if w is None:
                break
            slots.append(w)
            if len(slots) >= n_max:
                break
        t += step

    return slots


# ---------------------------------------------------------------------------
# Load model assets
# ---------------------------------------------------------------------------

def load_feature_names(model_dir: str) -> List[str]:
    """Load the ordered feature name list from features.json."""
    path = os.path.join(model_dir, "features.json")
    if not os.path.exists(path):
        sys.exit(f"features.json not found in {model_dir}")
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("features", "names"):
        if key in data:
            return data[key]
    sys.exit("Cannot parse features.json — expected a JSON list or a dict with 'features' key.")


def load_scaler(model_dir: str, feat_names: List[str]) -> Dict:
    """Load scaler mean and scale arrays from scaler.npz."""
    path = os.path.join(model_dir, "scaler.npz")
    if not os.path.exists(path):
        sys.exit(f"scaler.npz not found in {model_dir}")
    npz       = np.load(path)
    mean_arr  = npz["mean"].astype(float)
    scale_arr = npz["scale"].astype(float)
    if len(mean_arr) != len(feat_names):
        sys.exit(f"scaler has {len(mean_arr)} entries but features.json has {len(feat_names)}")
    return {"mean_arr": mean_arr, "scale_arr": scale_arr}


# ---------------------------------------------------------------------------
# Resample + derive signals
# Mirrors Features.deriveAccel() and Features.deriveBaro() in Kotlin
# ---------------------------------------------------------------------------

def run_pipeline(accel_df, baro_df):
    """
    Resample raw sensor data onto uniform grids and compute derived signals.

    Accelerometer signals derived:
        amag, amag_ema10, amag_emaVar10, avert, ahoriz, dyn

    Barometer signals derived:
        p_ema30, p_ema30_dt1, dhdt

    Returns (rs_a, rs_b, obs_accel, obs_baro) where obs_* are boolean arrays
    marking which samples were directly observed (not forward-filled).
    """
    rs_a_res = causal_resample_with_mask(
        accel_df, "t", ["ax", "ay", "az"], ACCEL_HZ, BIG_GAP_FACTOR, "ffill"
    )
    rs_b_res = causal_resample_with_mask(
        baro_df, "t", ["p"], BARO_HZ, BIG_GAP_FACTOR, "ffill"
    )
    rs_a, obs_a = rs_a_res.data, rs_a_res.obs_mask
    rs_b, obs_b = rs_b_res.data, rs_b_res.obs_mask

    if not rs_a.empty:
        ax = rs_a["ax"].to_numpy(float)
        ay = rs_a["ay"].to_numpy(float)
        az = rs_a["az"].to_numpy(float)
        m3   = np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
        amag = np.full_like(ax, np.nan)
        amag[m3] = np.sqrt(ax[m3]**2 + ay[m3]**2 + az[m3]**2)

        rs_a["amag"]          = amag
        rs_a["amag_ema10"]    = ema_mean(amag, ACCEL_HZ, tau_s=10.0)
        rs_a["amag_emaVar10"] = ema_var(amag,  ACCEL_HZ, tau_s=10.0)
        rs_a = compute_gravity_aligned_columns(rs_a, ACCEL_HZ, tau_s=GRAV_TAU_S)

        # Dynamic acceleration = amag minus a slow EMA of gravity.
        alpha_dyn   = 2.0 / (1.0 + DYN_TAU_S * ACCEL_HZ)
        g_hat       = ema_series_skipnan(amag, alpha_dyn)
        rs_a["dyn"] = amag - g_hat

        obs_accel = (
            obs_a.get("ax", np.zeros(len(rs_a), bool)) &
            obs_a.get("ay", np.zeros(len(rs_a), bool)) &
            obs_a.get("az", np.zeros(len(rs_a), bool))
        )
    else:
        obs_accel = np.array([], bool)

    if not rs_b.empty:
        p_arr           = rs_b["p"].to_numpy(float)
        rs_b["p_ema30"] = ema_mean(p_arr, BARO_HZ, tau_s=30.0)
        pm              = rs_b["p_ema30"].to_numpy(float)

        # First derivative of the smoothed pressure (Pa/s).
        if np.isfinite(pm).sum() >= 2:
            dpm = np.diff(pm) / (1.0 / BARO_HZ)
            rs_b["p_ema30_dt1"] = np.concatenate(([dpm[0]], dpm))
        else:
            rs_b["p_ema30_dt1"] = np.nan

        rs_b     = add_baro_dhdt_column(rs_b, BARO_HZ, dhdt_tau_s=DHDT_TAU_S)
        obs_baro = obs_b.get("p", np.zeros(len(rs_b), bool))
    else:
        obs_baro = np.array([], bool)

    return rs_a, rs_b, obs_accel, obs_baro


# ---------------------------------------------------------------------------
# Extract one window
# Mirrors FlightDetector.windowVector() exactly
# ---------------------------------------------------------------------------

def extract_one_window(
    slot:       Dict,
    rs_a,
    rs_b,
    obs_accel:  np.ndarray,
    obs_baro:   np.ndarray,
    last_end:   Dict[str, Optional[float]],
    feat_names: List[str],
    scaler:     Dict,
) -> Optional[Dict]:
    """
    Extract features for one window slot and apply z-score scaling.

    Returns None if the window fails the coverage gate (too many forward-filled
    samples), which mirrors FlightDetector.passesCoverage().

    The per-grid dt_prev_end_s tracking mirrors FlightDetector.runStream().
    """
    t_end   = slot["end"]
    win_s   = slot["win"]
    hop_s   = slot["hop"]
    grid_id = slot["grid_id"]
    gid     = slot["id"]

    # Snap window boundaries to the sensor grids, matching windowVector():
    #   weA = round(endSec * ACCEL_HZ) / ACCEL_HZ
    we_a = round(t_end * ACCEL_HZ) / ACCEL_HZ;  ws_a = we_a - win_s
    we_b = round(t_end * BARO_HZ)  / BARO_HZ;   ws_b = we_b - win_s

    # Slice the accelerometer resampled grid.
    if not rs_a.empty:
        ta = rs_a["t"].to_numpy(float)
        i0 = int(np.searchsorted(ta, ws_a - 1e-9))
        i1 = int(np.searchsorted(ta, we_a + 1e-9))
        t_a_w    = ta[i0:i1]
        ax_w     = rs_a["ax"].to_numpy(float)[i0:i1]
        ay_w     = rs_a["ay"].to_numpy(float)[i0:i1]
        az_w     = rs_a["az"].to_numpy(float)[i0:i1]
        amag_w   = rs_a["amag"].to_numpy(float)[i0:i1]
        dyn_w    = rs_a["dyn"].to_numpy(float)[i0:i1]
        ema10_w  = rs_a["amag_ema10"].to_numpy(float)[i0:i1]
        emaV10_w = rs_a["amag_emaVar10"].to_numpy(float)[i0:i1]
        avert_w  = rs_a.get("avert",  rs_a["ax"] * 0).to_numpy(float)[i0:i1]
        ahoriz_w = rs_a.get("ahoriz", rs_a["ax"] * 0).to_numpy(float)[i0:i1]
        obs_a_w  = obs_accel[i0:i1] if obs_accel.size else np.array([], bool)
    else:
        t_a_w = ax_w = ay_w = az_w = amag_w = dyn_w = np.array([], float)
        ema10_w = emaV10_w = avert_w = ahoriz_w = np.array([], float)
        obs_a_w = np.array([], bool)

    # Slice the barometer resampled grid.
    if not rs_b.empty:
        tb = rs_b["t"].to_numpy(float)
        j0 = int(np.searchsorted(tb, ws_b - 1e-9))
        j1 = int(np.searchsorted(tb, we_b + 1e-9))
        t_b_w         = tb[j0:j1]
        p_w           = rs_b["p"].to_numpy(float)[j0:j1]
        v_w           = rs_b["dhdt"].to_numpy(float)[j0:j1]
        p_ema30_w     = rs_b["p_ema30"].to_numpy(float)[j0:j1]
        p_ema30_dt1_w = rs_b["p_ema30_dt1"].to_numpy(float)[j0:j1]
        obs_b_w       = obs_baro[j0:j1] if obs_baro.size else np.array([], bool)
    else:
        t_b_w = p_w = v_w = p_ema30_w = p_ema30_dt1_w = np.array([], float)
        obs_b_w = np.array([], bool)

    # Coverage gate — mirrors FlightDetector.passesCoverage().
    # A window is rejected if too large a fraction of its samples were
    # forward-filled rather than directly observed.
    accel_cov = float(np.mean(obs_a_w)) if obs_a_w.size else 0.0
    baro_cov  = float(np.mean(obs_b_w)) if obs_b_w.size else 0.0
    if obs_a_w.size > 0 and accel_cov < COVERAGE_THR:
        return None
    if obs_b_w.size > 0 and baro_cov < COVERAGE_THR:
        return None
    if not obs_a_w.size and not obs_b_w.size:
        return None

    feats = extract_features_window(
        t_a_w, ax_w, ay_w, az_w, ACCEL_HZ,
        t_b_w, p_w,  v_w,         BARO_HZ,
        do_psd        = DO_PSD,
        amag_w        = amag_w,
        amag_ema10    = ema10_w,
        amag_emaVar10 = emaV10_w,
        p_ema30       = p_ema30_w,
        p_ema30_dt1   = p_ema30_dt1_w,
        avert         = avert_w,
        ahoriz        = ahoriz_w,
        dyn_w         = dyn_w,
    )

    # Per-grid dt_prev_end_s tracking — mirrors FlightDetector.runStream().
    prev    = last_end.get(gid)
    dt_prev = max(0.0, t_end - prev) if prev is not None else 0.0
    last_end[gid] = t_end

    # Metadata injection — mirrors FlightDetector.windowVector() exactly.
    feats["accel_coverage"]    = accel_cov
    feats["baro_coverage"]     = baro_cov
    feats["has_accel"]         = 1.0 if ax_w.size > 0 else 0.0
    feats["has_baro"]          = 1.0 if p_w.size  > 0 else 0.0
    feats["has_dyn"]           = 1.0 if (dyn_w.size > 0 and np.any(np.isfinite(dyn_w))) else 0.0
    feats["has_spectral"]      = 1.0 if (DO_PSD and ax_w.size >= 16) else 0.0
    feats["grid_id"]           = float(grid_id)
    feats["win_s"]             = float(win_s)
    feats["hop_s"]             = float(hop_s)
    feats["dt_prev_end_s"]     = dt_prev
    feats["log_win_s"]         = math.log1p(win_s)
    feats["log_dt_prev_end_s"] = math.log1p(dt_prev)

    raw_vec = np.array([feats.get(nm, math.nan) for nm in feat_names], dtype=np.float64)

    # Z-score scaling — mirrors Scaler.scaleWindow() exactly.
    # NaN features are replaced by the training mean (i.e. scaled to 0).
    # Features with zero or near-zero scale are clamped to 0.
    mean_arr  = scaler["mean_arr"]
    scale_arr = scaler["scale_arr"]
    scaled    = np.zeros(len(feat_names), dtype=np.float64)
    for i in range(len(feat_names)):
        x  = raw_vec[i]
        mu = float(mean_arr[i])
        sd = float(scale_arr[i])
        if not math.isfinite(x):
            x = mu
        if not math.isfinite(sd) or sd < 1e-8:
            scaled[i] = 0.0
        else:
            z = (x - mu) / sd
            scaled[i] = z if math.isfinite(z) else 0.0

    return {
        "t_end":         t_end,
        "win_s":         win_s,
        "hop_s":         hop_s,
        "grid_id":       float(grid_id),
        "grid_name":     gid,
        "dt_prev_end_s": dt_prev,
        "raw":           raw_vec,
        "scaled":        scaled,
        "dyn_n":         int(np.sum(np.isfinite(dyn_w))),  # finite dyn sample count
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate golden feature arrays for the Android feature parity test. "
            "Runs the full Python pipeline and saves expected output values to an NPZ "
            "file which FeatureParityTest.kt compares against Kotlin output."
        )
    )
    ap.add_argument("recording",
                    help="Path to a raw recording .txt file.")
    ap.add_argument("--model_dir", required=True,
                    help="Directory containing features.json and scaler.npz.")
    ap.add_argument("--out",       default="parity_golden.npz",
                    help="Output NPZ path.")
    ap.add_argument("--n_windows", type=int, default=50,
                    help="Maximum number of windows to include in the golden file.")
    args = ap.parse_args()

    feat_names = load_feature_names(args.model_dir)
    print(f"Features: {len(feat_names)}")

    scaler = load_scaler(args.model_dir, feat_names)
    print("Scaler loaded OK")

    print(f"Parsing {args.recording}")
    _, accel_df, baro_df = parse_flight_file(args.recording)
    print(f"  Accel: {len(accel_df)} rows   Baro: {len(baro_df)} rows")
    if accel_df.empty and baro_df.empty:
        sys.exit("No sensor data found in recording.")

    # Zero-reference timestamps — matches what RecordingParser.kt does.
    t0 = float(accel_df["t"].iloc[0]) if not accel_df.empty else float(baro_df["t"].iloc[0])
    if not accel_df.empty:
        accel_df["t"] = accel_df["t"] - t0
    if not baro_df.empty:
        baro_df["t"]  = baro_df["t"]  - t0

    print("Resampling and deriving signals...")
    rs_a, rs_b, obs_accel, obs_baro = run_pipeline(accel_df, baro_df)
    print(f"  Accel grid: {len(rs_a)} samples   Baro grid: {len(rs_b)} samples")

    # Origin is the first resampled timestamp.
    # The Kotlin test sets UnionGridScheduler origin = aR.t[0].
    origin = float(rs_a["t"].iloc[0]) if not rs_a.empty else float(rs_b["t"].iloc[0])
    t_max  = max(
        float(rs_a["t"].iloc[-1]) if not rs_a.empty else -math.inf,
        float(rs_b["t"].iloc[-1]) if not rs_b.empty else -math.inf,
    )
    print(f"  Origin: {origin:.4f}s   t_max: {t_max:.1f}s   span: {t_max - origin:.1f}s")

    # Collect window slots from the scheduler, then extract features.
    print(f"Running UnionGridScheduler (up to {args.n_windows} windows)...")
    slots = run_scheduler(origin, t_max, n_max=args.n_windows * 4)
    print(f"  Scheduler produced {len(slots)} raw slots")

    last_end: Dict[str, Optional[float]] = {}
    results = []
    for slot in slots:
        r = extract_one_window(
            slot, rs_a, rs_b, obs_accel, obs_baro,
            last_end, feat_names, scaler,
        )
        if r is not None:
            results.append(r)
            if len(results) >= args.n_windows:
                break

    print(f"  Extracted {len(results)} windows after coverage filtering")
    if not results:
        sys.exit("No windows passed the coverage gate. Use a longer recording.")

    n, nf = len(results), len(feat_names)

    # Build output arrays.
    features_raw    = np.full((n, nf), np.nan, dtype=np.float64)
    features_scaled = np.zeros((n, nf),         dtype=np.float64)
    t_end_arr   = np.zeros(n, np.float64)
    win_s_arr   = np.zeros(n, np.float64)
    hop_s_arr   = np.zeros(n, np.float64)
    grid_id_arr = np.zeros(n, np.float64)
    dt_prev_arr = np.zeros(n, np.float64)
    dyn_n_arr   = np.zeros(n, np.int32)

    for i, r in enumerate(results):
        features_raw[i]    = r["raw"]
        features_scaled[i] = r["scaled"]
        t_end_arr[i]   = r["t_end"]
        win_s_arr[i]   = r["win_s"]
        hop_s_arr[i]   = r["hop_s"]
        grid_id_arr[i] = r["grid_id"]
        dt_prev_arr[i] = r["dt_prev_end_s"]
        dyn_n_arr[i]   = r["dyn_n"]

    raw_at = accel_df["t"].to_numpy(np.float64)  if not accel_df.empty else np.array([], np.float64)
    raw_ax = accel_df["ax"].to_numpy(np.float64) if not accel_df.empty else np.array([], np.float64)
    raw_ay = accel_df["ay"].to_numpy(np.float64) if not accel_df.empty else np.array([], np.float64)
    raw_az = accel_df["az"].to_numpy(np.float64) if not accel_df.empty else np.array([], np.float64)
    raw_bt = baro_df["t"].to_numpy(np.float64)   if not baro_df.empty  else np.array([], np.float64)
    raw_bp = baro_df["p"].to_numpy(np.float64)   if not baro_df.empty  else np.array([], np.float64)

    np.savez_compressed(
        args.out,
        # Raw sensor data (pre-resample) — Kotlin resamples these itself.
        raw_accel_t  = raw_at,
        raw_accel_ax = raw_ax,
        raw_accel_ay = raw_ay,
        raw_accel_az = raw_az,
        raw_baro_t   = raw_bt,
        raw_baro_p   = raw_bp,
        # Scheduler parameters — Kotlin reads these to initialise UnionGridScheduler.
        scheduler_origin_sec = np.array([origin],     dtype=np.float64),
        base_win_s           = np.array([BASE_WIN_S], dtype=np.float64),
        base_hop_s           = np.array([BASE_HOP_S], dtype=np.float64),
        # Window sequence in scheduler emission order.
        window_t_end         = t_end_arr,
        window_win_s         = win_s_arr,
        window_hop_s         = hop_s_arr,
        window_grid_id       = grid_id_arr,
        window_dt_prev_end_s = dt_prev_arr,
        window_dyn_n         = dyn_n_arr,
        # Golden feature values — the Kotlin test compares against these.
        features_raw    = features_raw.flatten(),
        features_scaled = features_scaled.flatten(),
        n_windows  = np.array([n],  dtype=np.int32),
        n_features = np.array([nf], dtype=np.int32),
    )

    names_path = args.out.replace(".npz", "_feature_names.json")
    with open(names_path, "w") as f:
        json.dump(feat_names, f, indent=2)

    grid_counts: Dict[str, int] = {}
    for r in results:
        grid_counts[r["grid_name"]] = grid_counts.get(r["grid_name"], 0) + 1

    print(f"\nDone.")
    print(f"  Output:       {args.out}")
    print(f"  Feature names: {names_path}")
    print(f"  Windows: {n}   Features: {nf}   NaN rate: {np.isnan(features_raw).mean()*100:.1f}%")
    print(f"  Grid breakdown: {grid_counts}")
    print(f"  Scheduler origin: {origin:.4f}s")
    print(f"\nCopy to app/src/test/resources/:")
    print(f"  {args.out}")
    print(f"  {names_path}")
    print(f"  {args.model_dir}/scaler.npz  ->  scaler.npz")


if __name__ == "__main__":
    main()