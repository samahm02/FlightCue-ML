"""
GRU training, streaming-style evaluation, and optional ONNX export for the
FlightCue flight event detection system.

The model is a unidirectional GRU with a two-output linear head (TAKEOFF /
LANDING). It consumes sequences of sliding-window feature vectors produced by
preprocess_all_data_causal_fixed_v5_final.py and is evaluated using the same
streaming trigger logic as the on-device FlightDetector.

Architecture
------------
- Input  : (batch, seq_len, n_features) float32
- GRU    : hidden_dim, num_layers, causal (bidirectional=False)
- Head   : Dropout -> Linear(hidden_dim, 64) -> ReLU -> Dropout -> Linear(64, 2)
- Output : 2 logits (TAKEOFF head = index 0, LANDING head = index 1)

For Android export, ScaledGRUProb wraps the trained model with embedded
StandardScaler statistics and a sigmoid, producing a single probability output.
The scaling is baked into the ONNX graph so the on-device runtime receives
raw feature vectors and outputs a probability directly.


Usage -- takeoff model
----------------------
    python3 gru.py \
        --features all_data/preprocessed_v4_25/features_imputed.csv \
        --target_event takeoff \
        --seed 41 \
        --seq_len 25 \
        --label_shift 0 \
        --hidden_dim 256 \
        --num_layers 2 \
        --dropout 0.15 \
        --lr 5e-5 \
        --epochs 60 \
        --batch_size 64 \
        --patience 15 \
        --oversample_factor 0 \
        --primary_oversample 0 \
        --scaler_fit_mode unique_windows \
        --use_file_id_as_stream \
        --trigger_k 2 \
        --hit_window_s 90.0 \
        --fp_budget_per_hour 0.5 \
        --export_dir mobile_export_final/takeoff_gru \
        --export_event TAKEOFF

Usage -- landing model
----------------------
    python3 gru.py \
        --features all_data/preprocessed_v4_20/features_imputed.csv \
        --target_event landing \
        --seed 41 \
        --seq_len 25 \
        --label_shift 0 \
        --hidden_dim 192 \
        --num_layers 2 \
        --dropout 0.10 \
        --lr 3e-4 \
        --epochs 60 \
        --batch_size 64 \
        --patience 15 \
        --oversample_factor 2 \
        --primary_oversample 1 \
        --scaler_fit_mode unique_windows \
        --use_file_id_as_stream \
        --trigger_k 2 \
        --hit_window_s 90.0 \
        --fp_budget_per_hour 0.5 \
        --export_dir mobile_export_final/landing_gru \
        --export_event LANDING

Export output
-------------
    model.onnx       -- ONNX model with embedded scaler weights and sigmoid
    profile.json     -- full training configuration and test metrics
    features.json    -- ordered feature name list (154 features)
    scaler.npz       -- StandardScaler mean and scale arrays
    golden.npz       -- golden input/output samples for parity verification
    checkpoint.pt    -- raw PyTorch state dict (no scaler embedded)


"""

import argparse
import os
import time
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, DefaultDict
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Columns that are never used as model features (metadata / leakage)
# ---------------------------------------------------------------------------
META_COLS = {
    "file", "label", "window_id",
    "t_start", "t_end", "t_center", "t_anchor",
    "t_event",
    "t_takeoff", "t_landing",
    "dist_to_takeoff", "dist_to_landing", "dist_to_event",
    "win_start", "win_end",
    "win_len", "win_hop",
    "_hnm_keep_p", "sample_weight",
    "landing_proximity", "near_landing_30s", "near_landing_60s",
    "takeoff_proximity", "near_takeoff_30s", "near_takeoff_60s",
    "__row_idx",
    "group_id", "is_nonflight_stream",
    "domain", "file_id", "stream_id",
    "accel_obs_coverage", "baro_obs_coverage",
}

# GRU head indices
EVENT_TO_HEAD = {"TAKEOFF": 0, "LANDING": 1}

# Timing and grid metadata features that are always forced into the feature set.
# These must be present for the on-device sequence to match the training sequence.
ALWAYS_INCLUDE_FEATURES = {
    "win_s", "hop_s",
    "log_win_s", "log_hop_s",
    "dt_prev_end_s", "log_dt_prev_end_s",
    "dt_prev_anchor_s", "log_dt_prev_anchor_s",
    "grid_id",
}

# Prefix for binary missingness indicator columns added during loading.
MASK_PREFIX = "m__"


# ---------------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------------

def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy scalars and NaN/Inf to JSON-safe types."""
    if isinstance(obj, bool):
        return obj
    if obj is None:
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def write_json_strict(path: Path, data: Any) -> None:
    """Write data to a JSON file, converting NaN/Inf to null."""
    path.write_text(
        json.dumps(sanitize_for_json(data), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

def require_module(mod: str, hint: str) -> None:
    try:
        importlib.import_module(mod)
    except Exception as e:
        raise RuntimeError(f"Missing dependency '{mod}'. Install with:\n  {hint}") from e


def preflight_export_deps(verify_onnx: bool) -> None:
    require_module("onnx", "pip install onnx")
    if verify_onnx:
        require_module("onnxruntime", "pip install onnxruntime")


# ---------------------------------------------------------------------------
# Time column selection
# ---------------------------------------------------------------------------

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_time_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Select timing columns from the feature CSV.

    detect_time_col: the time at which detection can fire (right edge of window).
                     Prefers t_end, then t_anchor, then t_center.
    label_time_col:  the time used to assign the ground-truth label.
                     Prefers t_anchor, then t_center.
    order_col:       same as detect_time_col; used for deterministic sorting.

    Returns (order_col, detect_time_col, label_time_col).
    """
    detect_time_col = _pick_first_existing(df, ["t_end", "t_anchor", "t_center"])
    if detect_time_col is None:
        raise RuntimeError("CSV missing all of: t_end, t_anchor, t_center")

    label_time_col = _pick_first_existing(df, ["t_anchor", "t_center"])
    if label_time_col is None:
        label_time_col = detect_time_col

    return detect_time_col, detect_time_col, label_time_col


def deterministic_sort_cols(
    df: pd.DataFrame, primary_time_col: str, id_col: str
) -> List[str]:
    """
    Build a sort-column list for stable, deterministic ordering within a group.

    Primary sort is by id_col, then primary_time_col, then any available
    secondary columns, then __row_idx as a final tiebreaker.
    """
    cols = [id_col, primary_time_col]
    for c in ["t_start", "t_end", "win_s", "hop_s", "__row_idx"]:
        if c in df.columns and c not in cols:
            cols.append(c)
    if "__row_idx" not in cols:
        cols.append("__row_idx")
    return cols


def _median_finite(x: np.ndarray) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size > 0 else None


def estimate_sequence_delay_stats(sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Report statistics on (t_detect - t_label) across all sequences.

    This is a diagnostic to confirm that the detection time (window right edge)
    is consistently ahead of the label anchor time, as expected when
    window_anchor=right was used during preprocessing.
    """
    if not sequences:
        return {"n": 0, "mean_s": None, "median_s": None,
                "p90_s": None, "p95_s": None, "min_s": None, "max_s": None}

    d = np.array(
        [float(s["t_detect"]) - float(s["t_label"]) for s in sequences], dtype=float
    )
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0, "mean_s": None, "median_s": None,
                "p90_s": None, "p95_s": None, "min_s": None, "max_s": None}

    return {
        "n": int(d.size),
        "mean_s":   float(np.mean(d)),
        "median_s": float(np.median(d)),
        "p90_s":    float(np.percentile(d, 90)),
        "p95_s":    float(np.percentile(d, 95)),
        "min_s":    float(np.min(d)),
        "max_s":    float(np.max(d)),
    }


# ---------------------------------------------------------------------------
# Stream and group ID assignment
# ---------------------------------------------------------------------------

def add_stream_and_group_ids(
    df: pd.DataFrame,
    detect_time_col: str,
    nonflight_chunk_s: float,
    use_file_id_as_stream: bool,
    nonflight_label: str = "NONFLIGHT",
) -> pd.DataFrame:
    """
    Assign stream_id and group_id to every row.

    stream_id: identifies a contiguous sensor stream. If use_file_id_as_stream
               is True (recommended), this comes from the file_id column which
               already encodes nonflight chunk boundaries from preprocessing.

    group_id:  the splitting unit used by train/val/test assignment. For flight
               recordings this equals stream_id. For nonflight recordings a
               time-based chunk index may be appended to keep chunks independent.

    is_nonflight_stream: True for rows that belong to nonflight recordings.
    """
    df = df.copy()
    df["file"] = df["file"].astype(str)

    if use_file_id_as_stream and "file_id" in df.columns:
        df["stream_id"] = df["file_id"].astype(str)
    else:
        df["stream_id"] = df["file"].astype(str)

    if "domain" in df.columns:
        dom = df["domain"].astype(str).str.lower().fillna("")
        df["is_nonflight_stream"] = (dom == "nonflight")
    else:
        labels_u = df["label"].astype(str).str.upper()
        has_event_row = labels_u.isin(["TAKEOFF", "LANDING"])
        has_event_stream = has_event_row.groupby(df["stream_id"]).transform("any")
        df["is_nonflight_stream"] = (~has_event_stream) | (labels_u == str(nonflight_label).upper())

    df["group_id"] = df["stream_id"]

    chunk_s = float(nonflight_chunk_s or 0.0)
    if chunk_s > 0.0 and not (use_file_id_as_stream and "file_id" in df.columns):
        t = pd.to_numeric(df[detect_time_col], errors="coerce")
        t0 = t.groupby(df["stream_id"]).transform("min")
        dt = (t - t0).fillna(0.0)
        chunk_idx = np.floor(dt / chunk_s).astype("int64").clip(lower=0)
        chunk_str = pd.Series(chunk_idx, index=df.index).astype(str).str.zfill(4)
        m = df["is_nonflight_stream"].to_numpy(dtype=bool)
        df.loc[m, "group_id"] = df.loc[m, "stream_id"] + "__nf" + chunk_str.loc[m]

    return df


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def is_feature_column(df: pd.DataFrame, col: str) -> bool:
    """Return True if col should be used as a model feature."""
    if col in META_COLS:
        return False
    if col.startswith("_hnm_"):
        return False
    if not pd.api.types.is_numeric_dtype(df[col]):
        return False
    return True


def add_missingness_masks(
    df: pd.DataFrame,
    feature_cols: List[str],
    mode: str,
    prefix: str = MASK_PREFIX,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Optionally add binary missingness indicator columns.

    mode="none"  : no indicators added.
    mode="auto"  : add an indicator for each column that has any non-finite value.
    mode="all"   : add an indicator for every feature column.

    Returns the updated DataFrame, full feature column list, and mask column list.
    """
    mode = (mode or "none").lower().strip()
    if mode not in ("none", "auto", "all"):
        raise ValueError("missingness_masks must be one of: none, auto, all")

    if mode == "none":
        return df, list(feature_cols), []

    df = df.copy()
    mask_cols: List[str] = []

    for c in feature_cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        m = ~np.isfinite(v)
        if mode == "all" or bool(np.any(m)):
            mc = f"{prefix}{c}"
            df[mc] = m.astype(np.float32)
            mask_cols.append(mc)

    return df, list(feature_cols) + mask_cols, mask_cols


def load_features(csv_path: str) -> Tuple[pd.DataFrame, List[str], str, str, str]:
    """
    Load a features_imputed.csv produced by the preprocessor.

    Returns
    -------
    df             : full feature DataFrame with __row_idx added.
    feature_cols   : sorted list of numeric feature column names.
    order_col      : column used to sort windows within a group.
    detect_time_col: column used as the detection timestamp (t_end).
    label_time_col : column used as the label reference timestamp (t_anchor).
    """
    df = pd.read_csv(csv_path)
    df["__row_idx"] = np.arange(len(df), dtype=np.int64)

    if "file" not in df.columns or "label" not in df.columns:
        raise RuntimeError("CSV must contain columns: file, label")

    df["file"] = df["file"].astype(str)

    order_col, detect_time_col, label_time_col = choose_time_columns(df)

    feature_cols: List[str] = [col for col in df.columns if is_feature_column(df, col)]

    # Force timing and grid metadata features into the feature set.
    ensured = []
    for c in sorted(ALWAYS_INCLUDE_FEATURES):
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]) and c not in feature_cols:
            feature_cols.append(c)
            ensured.append(c)

    feature_cols = sorted(feature_cols)

    print(
        f"Loaded {len(df)} windows from {df['file'].nunique()} files "
        f"with {len(feature_cols)} numeric features"
    )
    print(f"[TimingCols] order_col={order_col}  "
          f"detect_time_col={detect_time_col}  label_time_col={label_time_col}")

    if ensured:
        print(f"[UnionFix] Forced-in timing features: {ensured}")

    for col_name, arg_name in [("win_s", "win_s"), ("hop_s", "hop_s")]:
        if col_name in df.columns:
            v = _median_finite(pd.to_numeric(df[col_name], errors="coerce").to_numpy())
            if v is not None:
                print(f"[WindowTiming] median {col_name}={v:.3f}s")

    if len(feature_cols) == 0:
        raise RuntimeError("No numeric feature columns found.")

    return df, feature_cols, order_col, detect_time_col, label_time_col


# ---------------------------------------------------------------------------
# Sequence creation
# ---------------------------------------------------------------------------

def create_sequences_with_label_pos(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int,
    label_idx: int,
    order_col: str,
    detect_time_col: str,
    label_time_col: str,
) -> List[Dict[str, Any]]:
    """
    Build GRU input sequences from the window-level feature DataFrame.

    Each sequence is a sliding window of seq_len consecutive windows from the
    same group_id. The label is taken from position label_idx within the
    sequence, where label_idx = seq_len // 2 + label_shift.

    A sequence is discarded if:
    - Any detect timestamp in the run is NaN.
    - The largest gap between consecutive windows exceeds 2.5x the median
    gap for that group, which usually means a stream discontinuity.

    Each returned dict contains:
        X             : (seq_len, n_features) float32 array
        y_takeoff     : takeoff label (0 or 1)
        y_landing     : landing label (0 or 1)
        flight_id     : source file name
        group_id      : stream chunk identifier
        is_nonflight  : whether this sequence comes from a nonflight recording
        t_detect      : detection timestamp (t_end of the last window in the sequence)
        t_label       : label reference timestamp (t_anchor at label_idx)
        accel_coverage: minimum accelerometer coverage fraction across the sequence
        baro_coverage : minimum barometer coverage fraction across the sequence

    """
    sequences: List[Dict[str, Any]] = []

    if "group_id" not in df.columns:
        df = df.copy()
        df["group_id"] = df["file"].astype(str)

    df_sorted = df.sort_values(
        deterministic_sort_cols(df, order_col, id_col="group_id"),
        kind="mergesort",
    ).reset_index(drop=True)

    for gid, gdf in df_sorted.groupby("group_id", sort=False):
        if len(gdf) < seq_len:
            continue

        file_id  = str(gdf["file"].iloc[0])
        group_id = str(gid)
        is_nf = bool(gdf["is_nonflight_stream"].iloc[0]) if "is_nonflight_stream" in gdf.columns else False

        X_g    = gdf[feature_cols].values.astype(np.float32)
        labels = gdf["label"].astype(str).str.upper().values
        y_to_g = (labels == "TAKEOFF").astype(int)
        y_ld_g = (labels == "LANDING").astype(int)

        t_detect_vec = pd.to_numeric(gdf[detect_time_col], errors="coerce").to_numpy(dtype=float)
        t_label_vec  = pd.to_numeric(gdf[label_time_col],  errors="coerce").to_numpy(dtype=float)

        accel_cov = (gdf["accel_obs_coverage"].to_numpy(dtype=float)
                     if "accel_obs_coverage" in gdf.columns else np.ones(len(gdf)))
        baro_cov  = (gdf["baro_obs_coverage"].to_numpy(dtype=float)
                     if "baro_obs_coverage" in gdf.columns else np.ones(len(gdf)))

        n = len(gdf)

        if n >= 2:
            diffs = np.diff(t_detect_vec)
            finite_diffs = diffs[np.isfinite(diffs)]
            median_hop = float(np.median(finite_diffs)) if finite_diffs.size > 0 else 60.0
        else:
            median_hop = 60.0

        max_allowed_gap = 2.5 * median_hop

        for start in range(0, n - seq_len + 1):
            end          = start + seq_len
            label_global = start + label_idx

            t_seq = t_detect_vec[start:end]
            if np.any(np.isnan(t_seq)):
                continue

            dt_seq  = np.diff(t_seq)
            max_gap = float(np.max(dt_seq)) if dt_seq.size > 0 else 0.0
            if max_gap > max_allowed_gap:
                continue

            sequences.append({
                "X":             X_g[start:end],
                "y_takeoff":     int(y_to_g[label_global]),
                "y_landing":     int(y_ld_g[label_global]),
                "flight_id":     file_id,
                "group_id":      group_id,
                "is_nonflight":  is_nf,
                "label_pos":     int(label_global),
                "t_detect":      float(t_detect_vec[end - 1]),
                "t_label":       float(t_label_vec[label_global]),
                "accel_coverage": float(np.min(accel_cov[start:end])),
                "baro_coverage":  float(np.min(baro_cov[start:end])),
            })

    return sequences


# ---------------------------------------------------------------------------
# Event segment extraction
# ---------------------------------------------------------------------------

def _flight_event_time(flight_df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the median of a time column within one flight, or None if absent."""
    if col not in flight_df.columns:
        return None
    v = pd.to_numeric(flight_df[col], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size > 0 else None


def build_event_segments_for_flights(
    df: pd.DataFrame,
    flights_subset: set,
    order_col: str,
    label_time_col: str,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Extract contiguous positive label segments for each flight in flights_subset.

    For each event type (TAKEOFF, LANDING) and each qualifying flight, returns
    a list of segment dicts with:
        start_idx : row index within the flight where the segment begins.
        end_idx   : row index where the segment ends (inclusive).
        t_ref     : reference event time (from t_takeoff / t_landing column,
                    or first label_time_col in segment as fallback).

    These segments are used by compute_event_eval_from_triggers to determine
    whether a streaming trigger counts as a hit.
    """
    df_sorted = df.sort_values(
        deterministic_sort_cols(df, order_col, id_col="file"),
        kind="mergesort",
    ).reset_index(drop=True)

    out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        "TAKEOFF": defaultdict(list),
        "LANDING": defaultdict(list),
    }

    for flight_id, flight_df in df_sorted.groupby("file", sort=False):
        fid = str(flight_id)
        if fid not in flights_subset:
            continue

        labels  = flight_df["label"].astype(str).str.upper().values
        t_label = pd.to_numeric(flight_df[label_time_col], errors="coerce").to_numpy(dtype=float)
        t_to    = _flight_event_time(flight_df, "t_takeoff")
        t_ld    = _flight_event_time(flight_df, "t_landing")

        for ev in ["TAKEOFF", "LANDING"]:
            idx = np.where((labels == ev).astype(int) == 1)[0]
            if idx.size == 0:
                continue

            segments = []
            s = prev = int(idx[0])
            for j in idx[1:]:
                j = int(j)
                if j == prev + 1:
                    prev = j
                else:
                    segments.append((s, prev))
                    s = prev = j
            segments.append((s, prev))

            for (a, b) in segments:
                t_ref = (t_to if ev == "TAKEOFF" else t_ld)
                if t_ref is None or not np.isfinite(float(t_ref)):
                    t_ref = float(t_label[a]) if np.isfinite(t_label[a]) else None

                out[ev][fid].append({
                    "start_idx": int(a),
                    "end_idx":   int(b),
                    "t_ref":     t_ref,
                })

    out["TAKEOFF"] = dict(out["TAKEOFF"])
    out["LANDING"] = dict(out["LANDING"])
    return out


# ---------------------------------------------------------------------------
# Streaming trigger logic
# ---------------------------------------------------------------------------

def _duration_hours(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return 0.0
    return max(float(np.max(t) - np.min(t)), 0.0) / 3600.0


def _sanitize_prob(p: float) -> float:
    return float(p) if np.isfinite(p) else -np.inf


def _collapse_duplicate_times(
    t: np.ndarray, p: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse rows with identical timestamps, keeping the max probability."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if t.size <= 1 or not np.any(np.diff(t) == 0):
        return t, p
    uniq_t, inv = np.unique(t, return_inverse=True)
    p2 = np.full_like(uniq_t, -np.inf, dtype=float)
    np.maximum.at(p2, inv, p)
    return uniq_t, p2


def detect_trigger_times(
    t: np.ndarray,
    p: np.ndarray,
    thr: float,
    trigger_k: int,
    cooldown_s: float,
) -> List[float]:
    """
    Simulate the on-device streaming trigger logic.

    A trigger fires when trigger_k consecutive windows all have probability >= thr.
    After a trigger, a cooldown_s period must elapse before a new trigger can fire.
    This matches the Kotlin FlightDetector trigger state machine exactly.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    p = p[order]
    t, p = _collapse_duplicate_times(t, p)

    k           = max(1, int(trigger_k))
    cooldown_s  = float(max(0.0, cooldown_s))
    consec      = 0
    last_trigger = -np.inf
    triggers: List[float] = []

    for tj, pj_raw in zip(t, p):
        pj = _sanitize_prob(pj_raw)

        if (tj - last_trigger) < cooldown_s:
            consec = 0
            continue

        if pj >= thr:
            consec += 1
            if consec >= k:
                triggers.append(float(tj))
                last_trigger = float(tj)
                consec = 0
        else:
            consec = 0

    return triggers


def count_stream_triggers_by_group(
    seqs: List[Dict[str, Any]],
    probs: np.ndarray,
    thr: float,
    trigger_k: int,
    cooldown_s: float,
) -> Dict[str, Any]:
    """
    Count streaming triggers across all groups and compute FP/hour.

    Used to evaluate the false positive rate on nonflight data during
    threshold selection on the validation set.
    """
    probs = np.asarray(probs, dtype=float)
    by_gid: DefaultDict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(seqs):
        by_gid[str(s["group_id"])].append(i)

    total_triggers = 0
    total_hours    = 0.0

    for _, idxs in by_gid.items():
        t = np.array([float(seqs[i]["t_detect"]) for i in idxs], dtype=float)
        p = np.array([float(probs[i]) for i in idxs], dtype=float)
        t_c, _ = _collapse_duplicate_times(np.sort(t), np.zeros(t.size))
        total_hours += _duration_hours(t_c)
        total_triggers += len(detect_trigger_times(t, p, thr=float(thr),
                                                   trigger_k=int(trigger_k),
                                                   cooldown_s=float(cooldown_s)))

    fp_per_hour = (total_triggers / total_hours) if total_hours > 0 else None
    return {
        "triggers":    int(total_triggers),
        "hours":       float(total_hours),
        "fp_per_hour": float(fp_per_hour) if fp_per_hour is not None else None,
        "n_groups":    int(len(by_gid)),
    }


def group_hours_from_sequences(seqs: List[Dict[str, Any]]) -> Dict[str, float]:
    """Return a dict mapping group_id to its total duration in hours."""
    by_gid: DefaultDict[str, List[float]] = defaultdict(list)
    for s in seqs:
        by_gid[str(s["group_id"])].append(float(s["t_detect"]))

    out: Dict[str, float] = {}
    for gid, times in by_gid.items():
        t = np.sort(np.asarray(times, dtype=float))
        t_c, _ = _collapse_duplicate_times(t, np.zeros_like(t))
        out[gid] = float(_duration_hours(t_c))
    return out


# ---------------------------------------------------------------------------
# Event-level evaluation
# ---------------------------------------------------------------------------

def _percentile_safe(x: np.ndarray, q: float) -> Optional[float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size > 0 else None


def compute_event_eval_from_triggers(
    test_seqs: List[Dict[str, Any]],
    prob_event: np.ndarray,
    thr: float,
    trigger_k: int,
    cooldown_s: float,
    segments_by_flight: Dict[str, List[Dict[str, Any]]],
    hit_window_s: float,
) -> Dict[str, Any]:
    """
    Evaluate event detection using the streaming trigger logic.

    For each event segment in segments_by_flight, determines whether the model
    triggers within hit_window_s seconds of the reference event time t_ref, and
    computes the detection latency (t_trigger - t_ref).

    Returns a dict with:
        event_n                 : total number of events with a valid t_ref.
        event_hit_after_rate    : fraction of events where a trigger fired within
                                  [t_ref, t_ref + hit_window_s].
        event_hit_sym_rate      : fraction within +/-hit_window_s (diagnostic).
        delay_post_s            : latency distribution for post-event triggers.
        event_details           : per-event trigger outcome list.
    """
    prob_event  = np.asarray(prob_event, dtype=float)
    hit_window_s = float(hit_window_s)

    by_flight: DefaultDict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(test_seqs):
        by_flight[str(s["flight_id"])].append(i)

    positives         = 0
    triggered_anywhere = 0
    triggered_after   = 0
    hits_sym          = 0
    hits_after        = 0
    delays_post:      List[float] = []
    delays_closest:   List[float] = []
    delays_hit_sym:   List[float] = []
    delays_hit_after: List[float] = []
    event_details:    List[Dict[str, Any]] = []

    for fid, segs in segments_by_flight.items():
        if fid not in by_flight:
            continue

        idxs       = by_flight[fid]
        t          = np.array([float(test_seqs[i]["t_detect"]) for i in idxs], dtype=float)
        p          = np.array([float(prob_event[i]) for i in idxs], dtype=float)
        trig_times = detect_trigger_times(t, p, thr=float(thr),
                                          trigger_k=int(trigger_k),
                                          cooldown_s=float(cooldown_s))

        for seg_idx, seg in enumerate(segs):
            t_ref = seg.get("t_ref", None)
            if t_ref is None or not np.isfinite(float(t_ref)):
                continue
            t_ref = float(t_ref)
            positives += 1
            hit_after          = False
            first_trigger_delay = None

            if len(trig_times) == 0:
                event_details.append({
                    "flight_id": fid, "seg_idx": seg_idx, "t_ref": t_ref,
                    "triggered": False, "hit_after": False,
                    "delay": None,
                    "max_prob": float(np.max(p)) if p.size > 0 else 0.0,
                })
                continue

            triggered_anywhere += 1
            tt = np.asarray(trig_times, dtype=float)

            j = int(np.argmin(np.abs(tt - t_ref)))
            delays_closest.append(float(tt[j] - t_ref))

            after = tt[tt >= t_ref]
            if after.size > 0:
                triggered_after    += 1
                first_trigger_delay = float(after[0] - t_ref)
                delays_post.append(first_trigger_delay)

            inwin_sym = tt[(tt >= t_ref - hit_window_s) & (tt <= t_ref + hit_window_s)]
            if inwin_sym.size > 0:
                hits_sym += 1
                delays_hit_sym.append(float(inwin_sym[0] - t_ref))

            inwin_after = tt[(tt >= t_ref) & (tt <= t_ref + hit_window_s)]
            if inwin_after.size > 0:
                hits_after += 1
                hit_after   = True
                delays_hit_after.append(float(inwin_after[0] - t_ref))

            event_details.append({
                "flight_id": fid, "seg_idx": seg_idx, "t_ref": t_ref,
                "triggered": True, "hit_after": hit_after,
                "delay": first_trigger_delay,
                "max_prob": float(np.max(p)) if p.size > 0 else 0.0,
            })

    def _stats(arr: np.ndarray) -> Dict[str, Any]:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        return {
            "n":      int(arr.size),
            "mean":   float(np.mean(arr))   if arr.size > 0 else None,
            "median": float(np.median(arr)) if arr.size > 0 else None,
            "p90":    _percentile_safe(arr, 90.0),
            "p95":    _percentile_safe(arr, 95.0),
            "min":    float(np.min(arr))    if arr.size > 0 else None,
            "max":    float(np.max(arr))    if arr.size > 0 else None,
        }

    def _rate(num: int, den: int) -> Optional[float]:
        return float(num) / float(den) if den > 0 else None

    return {
        "event_n":                       int(positives),
        "event_triggered_anywhere_n":    int(triggered_anywhere),
        "event_triggered_anywhere_rate": _rate(triggered_anywhere, positives),
        "event_triggered_after_n":       int(triggered_after),
        "event_triggered_after_rate":    _rate(triggered_after, positives),
        "event_hits_after_n":            int(hits_after),
        "event_hit_after_rate":          _rate(hits_after, positives),
        "event_hits_sym_n":              int(hits_sym),
        "event_hit_sym_rate":            _rate(hits_sym, positives),
        "hit_window_s":                  float(hit_window_s),
        "delay_post_s":                  _stats(np.asarray(delays_post,     dtype=float)),
        "delay_hit_after_s":             _stats(np.asarray(delays_hit_after, dtype=float)),
        "delay_hit_sym_s":               _stats(np.asarray(delays_hit_sym,  dtype=float)),
        "delay_closest_s":               _stats(np.asarray(delays_closest,  dtype=float)),
        "event_details":                 event_details,
    }


def print_event_latency_block(event_name: str, stats: Dict[str, Any]) -> None:
    """Print a formatted latency summary block for one event type."""
    print(f"\n{event_name.upper()} EVENT EVAL (TEST, app-realistic triggers)")
    print("-" * 70)
    print(f"Positives (valid t_ref):    {stats.get('event_n', 0)}")
    print(f"Triggered anywhere:         {stats.get('event_triggered_anywhere_n', 0)}"
          f"  (rate={stats.get('event_triggered_anywhere_rate')})")
    print(f"Triggered after marker:     {stats.get('event_triggered_after_n', 0)}"
          f"  (rate={stats.get('event_triggered_after_rate')})")
    hw = stats.get("hit_window_s")
    print(f"Hits after (0..+{hw}s):     {stats.get('event_hits_after_n', 0)}"
          f"  (hit_rate={stats.get('event_hit_after_rate')})")
    print(f"Hits symmetric (+/-{hw}s):  {stats.get('event_hits_sym_n', 0)}"
          f"  (hit_rate={stats.get('event_hit_sym_rate')})")

    def _print_delay(title: str, d: Dict[str, Any]) -> None:
        print(f"\n{title}")
        n = int(d.get("n", 0) or 0)
        print(f"  n={n}")
        if n == 0:
            print("  mean/median/p90/p95/min/max: N/A")
            return
        for k, label in [("mean", "mean"), ("median", "median"),
                          ("p90", "p90"), ("p95", "p95")]:
            v = d.get(k)
            print(f"  {label:7s}: {v:.2f} s" if v is not None else f"  {label:7s}: N/A")
        mn, mx = d.get("min"), d.get("max")
        if mn is not None and mx is not None:
            print(f"  min/max: {mn:.2f} / {mx:.2f} s")

    _print_delay("Post-event recognition delay (PRIMARY): first trigger >= t_ref",
                 stats.get("delay_post_s", {}))
    _print_delay(f"Post-hit delay: first trigger in [t_ref, t_ref+{hw}s]",
                 stats.get("delay_hit_after_s", {}))
    _print_delay("Closest-trigger delay (diagnostic; signed)",
                 stats.get("delay_closest_s", {}))
    _print_delay(f"Symmetric hit delay (diagnostic; signed; within +/-{hw}s)",
                 stats.get("delay_hit_sym_s", {}))


def bootstrap_event_hit_rate(
    event_details: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap 95% confidence interval for the event hit rate."""
    np.random.seed(seed)
    hit_rates = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(len(event_details), size=len(event_details), replace=True)
        hits   = sum(event_details[i]["hit_after"] for i in sample)
        hit_rates.append(hits / len(event_details))
    hr = np.array(hit_rates)
    return float(np.mean(hr)), float(np.percentile(hr, 2.5)), float(np.percentile(hr, 97.5))


# ---------------------------------------------------------------------------
# Scaler utilities
# ---------------------------------------------------------------------------

def _nan_safe_fit_data(X: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 before fitting the StandardScaler."""
    return np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def fit_scaler_from_train_sequences(train_seqs: List[Dict[str, Any]]) -> StandardScaler:
    """Fit StandardScaler from concatenated training sequences."""
    all_X = np.concatenate([s["X"] for s in train_seqs], axis=0)
    scaler = StandardScaler()
    scaler.fit(_nan_safe_fit_data(all_X))
    return scaler


def fit_scaler_from_train_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_groups: set,
    order_col: str,
) -> StandardScaler:
    """
    Fit StandardScaler from the unique training windows (before sequencing).

    This is the preferred mode (--scaler_fit_mode unique_windows) because it
    avoids over-weighting windows that appear in many sequences.
    """
    df_tr = df[df["group_id"].astype(str).isin({str(x) for x in train_groups})].copy()
    df_tr = df_tr.sort_values(
        deterministic_sort_cols(df_tr, order_col, id_col="group_id"),
        kind="mergesort",
    )
    X = df_tr[feature_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(_nan_safe_fit_data(X))
    return scaler


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class RawSequenceDataset(Dataset):
    """Dataset returning raw (unscaled) sequences for val/test inference."""

    def __init__(self, sequences: List[Dict[str, Any]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.sequences[idx]
        X = np.ascontiguousarray(s["X"], dtype=np.float32)
        return {
            "X_raw":    torch.from_numpy(X),
            "y_takeoff": torch.tensor(s["y_takeoff"], dtype=torch.float32),
            "y_landing": torch.tensor(s["y_landing"], dtype=torch.float32),
        }


class OversampledDataset(Dataset):
    """
    Training dataset with per-epoch random oversampling.

    Negative examples (windows far from any event) are kept as-is.
    Positive examples from the primary target event are repeated
    primary_oversample times; other positive examples are repeated
    oversample_factor times. Oversampling is randomised each time
    __getitem__ is called on a positive slot, so each epoch sees a
    different random draw from the positive pool.

    This prevents memorization of specific positive windows.
    """

    def __init__(
        self,
        sequences: List[Dict[str, Any]],
        oversample_factor: int = 1,
        primary_oversample: int = 1,
        target_event: str = "both",
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.neg = [s for s in sequences if (s["y_takeoff"] == 0 and s["y_landing"] == 0)]

        if target_event == "takeoff":
            self.primary_pos = [s for s in sequences if s["y_takeoff"] == 1]
            self.other_pos   = [s for s in sequences if s["y_landing"] == 1 and s["y_takeoff"] == 0]
        elif target_event == "landing":
            self.primary_pos = [s for s in sequences if s["y_landing"] == 1]
            self.other_pos   = [s for s in sequences if s["y_takeoff"] == 1 and s["y_landing"] == 0]
        else:
            self.primary_pos = [s for s in sequences if (s["y_takeoff"] == 1 or s["y_landing"] == 1)]
            self.other_pos   = []

        self.oversample_factor  = max(1, int(oversample_factor))
        self.primary_oversample = max(1, int(primary_oversample))
        self.n_neg     = len(self.neg)
        self.n_other   = len(self.other_pos)   * self.oversample_factor
        self.n_primary = len(self.primary_pos) * self.primary_oversample
        self.total     = self.n_neg + self.n_other + self.n_primary

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.n_neg:
            s = self.neg[idx]
        elif idx < self.n_neg + self.n_other:
            s = self.rng.choice(self.other_pos)
        else:
            s = self.rng.choice(self.primary_pos)
        X = np.ascontiguousarray(s["X"], dtype=np.float32)
        return {
            "X_raw":    torch.from_numpy(X),
            "y_takeoff": torch.tensor(s["y_takeoff"], dtype=torch.float32),
            "y_landing": torch.tensor(s["y_landing"], dtype=torch.float32),
        }


def scale_batch_torch(
    X_raw: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply z-score scaling in PyTorch, replacing any remaining NaN/Inf with 0."""
    X = torch.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X = (X - mean) / torch.clamp(scale, min=float(eps))
    return torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples and class imbalance.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    alpha : float
        Weighting factor in [0, 1] for positive examples.
    gamma : float
        Focusing parameter. Higher values focus more on misclassified examples.
    reduction : str
        'mean' or 'sum'.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha     = float(alpha)
        self.gamma     = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p   = torch.sigmoid(inputs)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * bce
        if self.reduction == "mean":
            return focal_loss.mean()
        if self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class FlightGRU(nn.Module):
    """
    Unidirectional multi-layer GRU with a two-output classification head.

    Output index 0 = TAKEOFF logit, index 1 = LANDING logit.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.head(h_n[-1])


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

class ModelWithTemperature(nn.Module):
    """
    Post-hoc temperature scaling wrapper.

    Optimizes a scalar temperature T on the validation set by minimizing NLL.
    The calibrated model outputs logits / T, which are then passed through
    sigmoid. A temperature > 1 makes the model less confident; < 1 makes it
    more confident.

    Only the temperature parameter is optimized; the base model is frozen.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) / self.temperature

    def get_temperature(self) -> float:
        return float(self.temperature.item())


def calibrate_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    mean_t: torch.Tensor,
    scale_t: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.01,
) -> ModelWithTemperature:
    """Optimise temperature scaling on the validation set using L-BFGS."""
    model.eval()
    temp_model = ModelWithTemperature(model).to(device)

    for param in temp_model.model.parameters():
        param.requires_grad = False

    optimizer    = torch.optim.LBFGS([temp_model.temperature], lr=lr, max_iter=max_iter)
    nll          = nn.BCEWithLogitsLoss()
    all_logits   = []
    all_labels_to = []
    all_labels_ld = []

    with torch.no_grad():
        for batch in val_loader:
            X = scale_batch_torch(batch["X_raw"].to(device, non_blocking=True), mean_t, scale_t)
            all_logits.append(model(X))
            all_labels_to.append(batch["y_takeoff"].to(device))
            all_labels_ld.append(batch["y_landing"].to(device))

    all_logits    = torch.cat(all_logits)
    all_labels_to = torch.cat(all_labels_to)
    all_labels_ld = torch.cat(all_labels_ld)

    def eval_loss():
        optimizer.zero_grad()
        scaled = all_logits / temp_model.temperature
        loss = 0.5 * (nll(scaled[:, 0], all_labels_to) + nll(scaled[:, 1], all_labels_ld))
        loss.backward()
        return loss

    initial_temp = temp_model.get_temperature()
    optimizer.step(eval_loss)
    final_temp   = temp_model.get_temperature()
    print(f"\n[TemperatureScaling] {initial_temp:.4f} -> {final_temp:.4f}")

    for param in temp_model.model.parameters():
        param.requires_grad = True
    temp_model.eval()
    return temp_model


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def active_events_from_target(target_event: str) -> List[str]:
    te = (target_event or "").lower().strip()
    if te == "takeoff":
        return ["TAKEOFF"]
    if te == "landing":
        return ["LANDING"]
    return ["TAKEOFF", "LANDING"]


def train_epoch_event_focused(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion_to: nn.Module,
    criterion_ld: nn.Module,
    device: torch.device,
    target_event: str,
    main_loss_weight: float,
    aux_loss_weight: float,
    mean_t: torch.Tensor,
    scale_t: torch.Tensor,
) -> float:
    """
    Run one training epoch.

    When target_event is 'takeoff' or 'landing', the primary head receives
    main_loss_weight and the auxiliary head receives aux_loss_weight.
    For target_event='both', both heads are weighted equally.
    """
    model.train()
    total_loss = 0.0
    n_samples  = 0
    te = (target_event or "").lower().strip()

    for batch in loader:
        X    = scale_batch_torch(batch["X_raw"].to(device, non_blocking=True), mean_t, scale_t)
        y_to = batch["y_takeoff"].to(device)
        y_ld = batch["y_landing"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits   = model(X)
        loss_to  = criterion_to(logits[:, 0], y_to)
        loss_ld  = criterion_ld(logits[:, 1], y_ld)

        if te == "takeoff":
            loss = main_loss_weight * loss_to + aux_loss_weight * loss_ld
        elif te == "landing":
            loss = main_loss_weight * loss_ld + aux_loss_weight * loss_to
        else:
            loss = 0.5 * (loss_to + loss_ld)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        bs          = X.size(0)
        total_loss  += float(loss.item()) * bs
        n_samples   += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Optional[DataLoader],
    device: torch.device,
    mean_t: torch.Tensor,
    scale_t: torch.Tensor,
) -> Dict[str, np.ndarray]:
    """
    Run inference over a DataLoader and collect output probabilities.

    Works with both FlightGRU and ModelWithTemperature because both return
    logits before sigmoid; sigmoid is applied here via torch.sigmoid.
    """
    model.eval()
    empty = {
        "prob_takeoff": np.array([]), "prob_landing": np.array([]),
        "y_takeoff":    np.array([]), "y_landing":   np.array([]),
    }
    if loader is None:
        return empty

    all_prob_to, all_prob_ld, all_y_to, all_y_ld = [], [], [], []

    for batch in loader:
        X      = scale_batch_torch(batch["X_raw"].to(device, non_blocking=True), mean_t, scale_t)
        probs  = torch.sigmoid(model(X))
        all_prob_to.append(probs[:, 0].cpu().numpy())
        all_prob_ld.append(probs[:, 1].cpu().numpy())
        all_y_to.append(batch["y_takeoff"].cpu().numpy())
        all_y_ld.append(batch["y_landing"].cpu().numpy())

    if not all_prob_to:
        return empty

    return {
        "prob_takeoff": np.concatenate(all_prob_to),
        "prob_landing": np.concatenate(all_prob_ld),
        "y_takeoff":   np.concatenate(all_y_to),
        "y_landing":   np.concatenate(all_y_ld),
    }


def find_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, beta: float = 2.0
) -> float:
    """Select the threshold that maximises F-beta on the PR curve."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.sum() == 0:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    b2 = beta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = (1 + b2) * (precision * recall) / (b2 * precision + recall + 1e-10)
    best_idx = min(int(np.nanargmax(f_beta)), len(thresholds) - 1)
    return float(thresholds[best_idx])


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "EVENT",
    beta: float = 2.0,
    fixed_threshold: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Compute classification metrics at a given (or optimised) threshold."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.size == 0:
        return {"pr_auc": 0.0, "roc_auc": 0.5, "threshold": float(fixed_threshold or 0.5),
                "TP": 0, "FN": 0, "FP": 0, "TN": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "fbeta": 0.0}

    pr_auc = float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else 0.0
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        roc_auc = 0.5

    thr    = float(fixed_threshold) if fixed_threshold is not None else find_best_threshold(y_true, y_prob, beta=beta)
    y_pred = (y_prob >= thr).astype(int)
    cm     = confusion_matrix(y_true, y_pred, labels=[1, 0])
    TP = FN = FP = TN = 0
    if cm.shape == (2, 2):
        TP, FN = int(cm[0, 0]), int(cm[0, 1])
        FP, TN = int(cm[1, 0]), int(cm[1, 1])

    prec  = TP / (TP + FP) if (TP + FP) else 0.0
    rec   = TP / (TP + FN) if (TP + FN) else 0.0
    f1    = (2 * prec * rec / (prec + rec + 1e-12)) if (prec + rec) else 0.0
    b2    = beta * beta
    fbeta = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-12) if (prec + rec) else 0.0

    if verbose:
        print(f"\n{'=' * 50}\n{name}\n{'=' * 50}")
        print(f"PR-AUC:  {pr_auc:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Thr(F{beta:.1f}): {thr:.6f}")
        print(f"Confusion [TP, FN; FP, TN]: [[{TP}, {FN}], [{FP}, {TN}]]")
        print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  F{beta:.1f}: {fbeta:.4f}")

    return {
        "pr_auc": pr_auc, "roc_auc": roc_auc, "threshold": thr,
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "precision": float(prec), "recall": float(rec),
        "f1": float(f1), "fbeta": float(fbeta),
    }


# ---------------------------------------------------------------------------
# Threshold selection with FP/hour budget
# ---------------------------------------------------------------------------

def _quantile_candidates(p: np.ndarray, thr_grid: int) -> np.ndarray:
    """Build a dense threshold candidate grid from quantiles and fixed anchors."""
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    n    = int(max(32, min(int(thr_grid), 4096)))
    cand = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n)))
    anchors = np.array([0.0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3,
                        0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 1.0])
    cand = np.unique(np.clip(np.concatenate([cand, anchors]), 0.0, 1.0))
    cand.sort()
    return cand


def select_threshold_fp_budget_with_objective(
    *,
    event_name: str,
    val_nf_seqs: List[Dict[str, Any]],
    val_nf_probs: np.ndarray,
    val_f_seqs: List[Dict[str, Any]],
    val_f_probs: np.ndarray,
    val_f_y: np.ndarray,
    df_all: pd.DataFrame,
    order_col: str,
    label_time_col: str,
    trigger_k: int,
    cooldown_s: float,
    hit_window_s: float,
    fp_budget_per_hour: float,
    thr_grid: int,
    objective: str,
    beta: float,
    min_nf_hours_for_budget: float,
) -> Tuple[float, Dict[str, Any], bool, Dict[str, Any]]:
    """
    Select the highest threshold that keeps FP/hour <= fp_budget_per_hour on the
    nonflight validation set, while maximising the chosen objective
    (event_hit_after or fbeta_flight) on the flight validation set.

    Falls back to F-beta thresholding if insufficient nonflight hours are available.

    Returns (threshold, nf_stats, fallback_used, objective_stats).
    """
    objective = (objective or "event_hit_after").lower().strip()
    if objective not in ("event_hit_after", "fbeta_flight"):
        raise ValueError("thr_objective must be one of: event_hit_after, fbeta_flight")

    nf_hours = count_stream_triggers_by_group(
        val_nf_seqs, val_nf_probs, thr=1.0,
        trigger_k=trigger_k, cooldown_s=cooldown_s,
    ).get("hours", 0.0)
    budget = float(fp_budget_per_hour)

    if (not np.isfinite(nf_hours)) or nf_hours < float(min_nf_hours_for_budget):
        thr_fb = find_best_threshold(val_f_y, val_f_probs, beta=float(beta))
        nf_stats = (
            count_stream_triggers_by_group(val_nf_seqs, val_nf_probs, thr=float(thr_fb),
                                           trigger_k=trigger_k, cooldown_s=cooldown_s)
            if len(val_nf_seqs) > 0
            else {"triggers": 0, "hours": float(nf_hours), "fp_per_hour": None, "n_groups": 0}
        )
        return float(thr_fb), nf_stats, True, {
            "objective": "fallback_fbeta", "score": None,
            "note": "Insufficient nonflight hours for fp_budget tuning",
        }

    cand = _quantile_candidates(np.asarray(val_nf_probs, dtype=float), thr_grid=int(thr_grid))

    segs_event = None
    val_flights = {str(s["flight_id"]) for s in val_f_seqs if not s.get("is_nonflight", False)}
    if objective == "event_hit_after" and len(val_flights) > 0:
        segments   = build_event_segments_for_flights(
            df=df_all, flights_subset=val_flights,
            order_col=order_col, label_time_col=label_time_col,
        )
        segs_event = segments.get(event_name.upper(), {})

    best = best_nf_stats = best_obj_stats = None

    for thr in cand:
        nf_stats = count_stream_triggers_by_group(
            seqs=val_nf_seqs, probs=val_nf_probs, thr=float(thr),
            trigger_k=int(trigger_k), cooldown_s=float(cooldown_s),
        )
        fp_h = nf_stats.get("fp_per_hour", None)
        if fp_h is None or fp_h > budget:
            continue

        if objective == "fbeta_flight":
            m     = compute_metrics(val_f_y, val_f_probs, beta=float(beta),
                                    fixed_threshold=float(thr), verbose=False)
            score = float(m.get("fbeta", 0.0) or 0.0)
            obj_stats = {"objective": "fbeta_flight", "score": score, "metrics": m}
        else:
            if not segs_event:
                m     = compute_metrics(val_f_y, val_f_probs, beta=float(beta),
                                        fixed_threshold=float(thr), verbose=False)
                score = float(m.get("fbeta", 0.0) or 0.0)
                obj_stats = {"objective": "event_hit_after_fallback_fbeta", "score": score,
                             "metrics": m, "note": "No flight segments on val; used Fbeta"}
            else:
                ev    = compute_event_eval_from_triggers(
                    test_seqs=val_f_seqs, prob_event=val_f_probs, thr=float(thr),
                    trigger_k=int(trigger_k), cooldown_s=float(cooldown_s),
                    segments_by_flight=segs_event, hit_window_s=float(hit_window_s),
                )
                score     = float(ev.get("event_hit_after_rate") or 0.0)
                obj_stats = {"objective": "event_hit_after", "score": score, "event_eval": ev}

        if best is None or score > best[0] + 1e-12:
            best          = (score, float(thr))
            best_nf_stats = nf_stats
            best_obj_stats = obj_stats
        elif abs(score - best[0]) <= 1e-12 and float(thr) > best[1]:
            best          = (score, float(thr))
            best_nf_stats = nf_stats
            best_obj_stats = obj_stats

    if best is None:
        nf_stats = count_stream_triggers_by_group(
            seqs=val_nf_seqs, probs=val_nf_probs, thr=1.0,
            trigger_k=int(trigger_k), cooldown_s=float(cooldown_s),
        )
        return 1.0, nf_stats, True, {
            "objective": objective, "score": None, "note": "Budget not met; forced thr=1.0",
        }

    return float(best[1]), best_nf_stats, False, best_obj_stats


# ---------------------------------------------------------------------------
# Train/val/test split
# ---------------------------------------------------------------------------

def split_groups_with_nonflight_focus(
    *,
    flight_groups: np.ndarray,
    nonflight_groups: np.ndarray,
    sequences: List[Dict[str, Any]],
    seed: int,
    flight_val_frac: float,
    flight_test_frac: float,
    nf_val_frac: float,
    nf_test_frac: float,
    min_nf_val_groups: int,
    min_nf_val_hours: float,
    min_nf_test_groups: int,
) -> Tuple[set, set, set]:
    """
    Deterministic train / val / test split by physical file.

    Flight groups are sorted alphabetically then rotated by a seed-derived
    offset, giving a deterministic but seed-dependent assignment.

    Nonflight groups are sorted by (duration DESC, group_id ASC), ensuring
    that the longest recordings are distributed across splits first and that
    tiebreaking is fully deterministic.

    No randomisation (no shuffle) is used; the only source of variation is
    the seed-based rotation for flight groups.
    """
    rng = np.random.default_rng(int(seed))

    fg = sorted(list(flight_groups))
    n  = len(fg)

    if n == 0:
        tr_f = va_f = te_f = set()
    else:
        rotation = int(rng.integers(0, n)) if n > 1 else 0
        fg       = fg[rotation:] + fg[:rotation]
        n_test   = max(1, int(round(float(flight_test_frac) * n))) if n >= 3 else 1
        n_val    = max(1, int(round(float(flight_val_frac)  * n))) if n >= 3 else 1
        te_f = set(fg[:n_test])
        va_f = set(fg[n_test:n_test + n_val])
        tr_f = set(fg[n_test + n_val:])

    nfg = list(nonflight_groups)

    if not nfg:
        tr_nf = va_nf = te_nf = set()
    else:
        dur_map = group_hours_from_sequences(
            [s for s in sequences if str(s["group_id"]) in set(nfg)]
        )
        nfg = sorted(nfg, key=lambda g: (-float(dur_map.get(str(g), 0.0)), str(g)))
        n_total      = len(nfg)
        n_test_target = max(int(round(float(nf_test_frac) * n_total)),
                            int(min_nf_test_groups)) if n_total >= 2 else 1
        te_nf = set(nfg[:max(1, min(n_total, n_test_target))])
        rem   = [g for g in nfg if g not in te_nf]

        n_val_target = max(int(round(float(nf_val_frac) * n_total)),
                           int(min_nf_val_groups)) if n_total >= 2 else 1
        va_nf    = set()
        va_hours = 0.0
        for g in rem:
            if (len(va_nf) < n_val_target
                    or len(va_nf) < int(min_nf_val_groups)
                    or va_hours < float(min_nf_val_hours)):
                va_nf.add(g)
                va_hours += float(dur_map.get(str(g), 0.0))
            else:
                break

        rem2  = [g for g in rem if g not in va_nf]
        tr_nf = set(rem2)

        if not tr_nf and n_total >= 3 and len(va_nf) > 1:
            g_move = min(va_nf)
            va_nf.remove(g_move)
            tr_nf.add(g_move)

    return set.union(tr_f, tr_nf), set.union(va_f, va_nf), set.union(te_f, te_nf)


# ---------------------------------------------------------------------------
# K-fold split (implemented but not used for final thesis models)
# ---------------------------------------------------------------------------

def kfold_split_groups(
    *,
    flight_groups: np.ndarray,
    nonflight_groups: np.ndarray,
    sequences: List[Dict[str, Any]],
    n_folds: int,
    seed: int,
) -> List[Tuple[set, set, set]]:
    """
    Generate n_folds deterministic (train, val, test) group splits.

    Flight groups are rotated by a seed-based offset then divided into
    equal-sized folds. Each fold uses one slice as test, one fifth of the
    remaining as val, and the rest as train.

    Returns a list of (train_groups, val_groups, test_groups) tuples.
    """
    rng = np.random.default_rng(int(seed))

    fg = sorted(list(flight_groups))
    if len(fg) > 1:
        rotation = int(rng.integers(0, len(fg)))
        fg = fg[rotation:] + fg[:rotation]
    fg = np.array(fg, dtype=str)

    nfg = list(nonflight_groups)
    if nfg:
        dur_map = group_hours_from_sequences(
            [s for s in sequences if str(s["group_id"]) in set(nfg)]
        )
        nfg = sorted(nfg, key=lambda g: (-float(dur_map.get(str(g), 0.0)), str(g)))
    nfg = np.array(nfg, dtype=str)

    n_folds = int(n_folds)
    folds   = []

    for fold_idx in range(n_folds):
        if len(fg) >= n_folds:
            fold_size  = len(fg) // n_folds
            ts         = fold_idx * fold_size
            te         = ts + fold_size if fold_idx < n_folds - 1 else len(fg)
            fg_test    = set(fg[ts:te])
            fg_tv      = np.concatenate([fg[:ts], fg[te:]])
        else:
            fg_test = set([fg[fold_idx]]) if fold_idx < len(fg) else set()
            fg_tv   = np.array([f for f in fg if f not in fg_test], dtype=str)

        if len(fg_tv) > 1:
            n_val    = max(1, len(fg_tv) // 5)
            fg_val   = set(fg_tv[:n_val])
            fg_train = set(fg_tv[n_val:])
        else:
            fg_val   = set()
            fg_train = set(fg_tv)

        if len(nfg) >= n_folds:
            nf_fold_size = len(nfg) // n_folds
            nf_ts        = fold_idx * nf_fold_size
            nf_te        = nf_ts + nf_fold_size if fold_idx < n_folds - 1 else len(nfg)
            nfg_test     = set(nfg[nf_ts:nf_te])
            nfg_tv       = np.concatenate([nfg[:nf_ts], nfg[nf_te:]])
        else:
            nfg_test = set([nfg[fold_idx]]) if fold_idx < len(nfg) else set()
            nfg_tv   = np.array([f for f in nfg if f not in nfg_test], dtype=str)

        if len(nfg_tv) > 1:
            n_val_nf = max(1, len(nfg_tv) // 5)
            nfg_val  = set(nfg_tv[:n_val_nf])
            nfg_train = set(nfg_tv[n_val_nf:])
        else:
            nfg_val   = set()
            nfg_train = set(nfg_tv)

        folds.append((
            set.union(fg_train, nfg_train),
            set.union(fg_val,   nfg_val),
            set.union(fg_test,  nfg_test),
        ))

    return folds


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

class ScaledGRUProb(nn.Module):
    """
    Export wrapper: embeds StandardScaler statistics and sigmoid into the graph.

    Input  : (1, seq_len, n_features) float32 raw feature tensor.
    Output : (1, 1) float32 probability for the exported event head.

    This is the model loaded by OnnxDetector on Android. The scaler mean and
    scale are stored as constant buffers so no separate scaler file is needed
    at inference time.
    """

    def __init__(
        self,
        model: nn.Module,
        mean: np.ndarray,
        scale: np.ndarray,
        head_index: int,
        eps: float = 1e-8,
    ):
        super().__init__()
        if isinstance(model, ModelWithTemperature):
            self.model       = model.model.eval()
            self.temperature = float(model.get_temperature())
        else:
            self.model       = model.eval()
            self.temperature = 1.0

        self.head_index = int(head_index)
        self.register_buffer("mean",  torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / torch.clamp(self.scale, min=self.eps)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        logit = self.model(x)[:, self.head_index:self.head_index + 1] / self.temperature
        return torch.sigmoid(logit)


def read_actual_opset(onnx_path: Path) -> int:
    import onnx
    m   = onnx.load(str(onnx_path))
    ops = [int(o.version) for o in m.opset_import]
    return max(ops) if ops else -1


def export_onnx(
    wrapper: nn.Module, seq_len: int, n_features: int, onnx_path: Path, opset: int
) -> None:
    """Export a single-file ONNX model with embedded weights."""
    wrapper = wrapper.eval()
    dummy   = torch.zeros((1, seq_len, n_features), dtype=torch.float32)
    kwargs  = dict(input_names=["x"], output_names=["p"],
                   opset_version=int(opset), do_constant_folding=True,
                   dynamic_axes=None, verbose=False)
    try:
        kwargs["training"] = torch.onnx.TrainingMode.EVAL
    except Exception:
        pass
    try:
        torch.onnx.export(wrapper, dummy, str(onnx_path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(wrapper, dummy, str(onnx_path), **kwargs)


def sample_golden_from_sequences(
    seqs: List[Dict[str, Any]], n: int, seed: int
) -> np.ndarray:
    """Sample n sequences for ONNX parity verification."""
    n   = min(n, len(seqs))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(seqs), size=n, replace=False) if n > 0 else np.array([], dtype=int)
    return np.stack([seqs[i]["X"] for i in idx], axis=0).astype(np.float32)


def ort_run_static_batch1(session: Any, X: np.ndarray) -> np.ndarray:
    """Run ONNX Runtime inference one sample at a time (static batch size 1)."""
    inp  = session.get_inputs()[0].name
    out  = session.get_outputs()[0].name
    outs = [session.run([out], {inp: X[i:i + 1]})[0].astype(np.float32)
            for i in range(X.shape[0])]
    return np.concatenate(outs, axis=0)


def verify_onnx_parity(
    onnx_path: Path, x_golden: np.ndarray, p_ref: np.ndarray
) -> float:
    """Return the max absolute difference between PyTorch and ONNX outputs."""
    import onnxruntime as ort
    sess  = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    p_ort = ort_run_static_batch1(sess, x_golden)
    return float(np.max(np.abs(p_ort - p_ref)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Train a GRU-based flight event detector and optionally export "
            "to ONNX for on-device inference."
        )
    )

    # Input data
    ap.add_argument("--features", required=True,
                    help="Path to features_imputed.csv from the preprocessor.")

    # Sequence configuration
    ap.add_argument("--seq_len",     type=int,   default=25,
                    help="Number of consecutive windows per GRU input sequence.")
    ap.add_argument("--label_shift", type=int,   default=0,
                    help="Offset of the label position from the sequence centre.")

    # Model architecture
    ap.add_argument("--hidden_dim",  type=int,   default=192)
    ap.add_argument("--num_layers",  type=int,   default=2)
    ap.add_argument("--dropout",     type=float, default=0.2)

    # Training
    ap.add_argument("--epochs",      type=int,   default=80)
    ap.add_argument("--batch_size",  type=int,   default=64)
    ap.add_argument("--num_workers", type=int,   default=0)
    ap.add_argument("--prefetch_factor",     type=int,   default=2)
    ap.add_argument("--persistent_workers",  action="store_true")
    ap.add_argument("--torch_threads",       type=int,   default=0)
    ap.add_argument("--torch_interop_threads", type=int, default=0)
    ap.add_argument("--train_stride", type=int,  default=1,
                    help="Only use every Nth training sequence (thin the training set).")
    ap.add_argument("--lr",          type=float, default=1e-3)
    ap.add_argument("--patience",    type=int,   default=15,
                    help="Early stopping patience in epochs.")
    ap.add_argument("--fbeta",       type=float, default=2.0)
    ap.add_argument("--seed",        type=int,   default=41)
    ap.add_argument("--max_pos_weight", type=float, default=10.0,
                    help="Cap on BCE pos_weight to prevent instability.")

    # Loss function
    ap.add_argument("--use_focal_loss", action="store_true",
                    help="Use Focal Loss instead of weighted BCE.")
    ap.add_argument("--focal_alpha", type=float, default=0.25)
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    # Scaler
    ap.add_argument("--scaler_fit_mode",
                    choices=["sequence", "unique_windows"], default="unique_windows",
                    help="Whether to fit the scaler from all sequence rows or unique windows only.")

    # Event focus and oversampling
    ap.add_argument("--target_event",
                    choices=["takeoff", "landing", "both"], default="both")
    ap.add_argument("--primary_oversample", type=int,   default=1)
    ap.add_argument("--oversample_factor",  type=int,   default=1)
    ap.add_argument("--main_loss_weight",   type=float, default=1.0)
    ap.add_argument("--aux_loss_weight",    type=float, default=0.0)

    # Evaluation
    ap.add_argument("--trigger_k",    type=int,   default=1)
    ap.add_argument("--hit_window_s", type=float, default=10.0)
    ap.add_argument("--cooldown_s",   type=float, default=60.0)

    # Temperature scaling
    ap.add_argument("--use_temperature_scaling", action="store_true")

    # K-fold (not used for final models)
    ap.add_argument("--use_kfold", action="store_true")
    ap.add_argument("--n_folds",   type=int, default=5)

    # Metrics output
    ap.add_argument("--metrics_out", type=str, default=None,
                    help="Optional path to write per-event metrics as JSON.")

    # Stream / nonflight settings
    ap.add_argument("--use_file_id_as_stream", action="store_true",
                    help="Use file_id (includes chunk suffix) as stream boundary.")
    ap.add_argument("--nonflight_chunk_s", type=float, default=0.0)
    ap.add_argument("--nonflight_label",   type=str,   default="NONFLIGHT")
    ap.add_argument("--missingness_masks",
                    choices=["none", "auto", "all"], default="auto")

    # Split fractions
    ap.add_argument("--flight_val_frac",  type=float, default=0.2)
    ap.add_argument("--flight_test_frac", type=float, default=0.3)
    ap.add_argument("--nf_val_frac",      type=float, default=0.5)
    ap.add_argument("--nf_test_frac",     type=float, default=0.2)
    ap.add_argument("--min_nf_val_groups",  type=int,   default=2)
    ap.add_argument("--min_nf_val_hours",   type=float, default=2.0)
    ap.add_argument("--min_nf_test_groups", type=int,   default=1)

    # Threshold selection
    ap.add_argument("--thr_mode",
                    choices=["fbeta", "fp_budget_nonflight"], default="fp_budget_nonflight")
    ap.add_argument("--fp_budget_per_hour",     type=float, default=0.05,
                    help="Maximum allowed FP/hour on nonflight val data.")
    ap.add_argument("--thr_grid",               type=int,   default=512)
    ap.add_argument("--thr_objective",
                    choices=["event_hit_after", "fbeta_flight"], default="event_hit_after")
    ap.add_argument("--min_nf_hours_for_budget", type=float, default=2.0)

    # Export
    ap.add_argument("--export_dir",   type=str, default=None,
                    help="Directory for ONNX export artifacts.")
    ap.add_argument("--export_event",
                    choices=["TAKEOFF", "LANDING"], default="TAKEOFF")
    ap.add_argument("--opset",        type=int,   default=18)
    ap.add_argument("--emit_golden",  type=int,   default=32,
                    help="Number of golden samples to save for parity verification.")
    ap.add_argument("--verify_onnx",  action="store_true")
    ap.add_argument("--parity_warn_if_gt", type=float, default=0.02)

    args = ap.parse_args()

    if int(args.torch_threads or 0) > 0:
        torch.set_num_threads(int(args.torch_threads))
        os.environ["OMP_NUM_THREADS"] = str(int(args.torch_threads))
        os.environ["MKL_NUM_THREADS"] = str(int(args.torch_threads))
    if int(args.torch_interop_threads or 0) > 0:
        torch.set_num_interop_threads(int(args.torch_interop_threads))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_num_workers = int(args.num_workers)
    dl_prefetch    = int(args.prefetch_factor) if dl_num_workers > 0 else 2

    print(f"Device: {device}")
    print(f"Target event: {args.target_event}")
    print(f"Loss: {'Focal Loss (alpha={}, gamma={})'.format(args.focal_alpha, args.focal_gamma) if args.use_focal_loss else 'BCE Loss'}")
    print(f"Temperature scaling: {'yes' if args.use_temperature_scaling else 'no'}")
    print(f"K-fold CV: {'yes (n={})'.format(args.n_folds) if args.use_kfold else 'no'}")

    active_events = active_events_from_target(args.target_event)

    center    = args.seq_len // 2
    label_idx = center + args.label_shift
    if not (0 <= label_idx < args.seq_len):
        raise ValueError("label_idx out of range for seq_len/label_shift combination.")

    print(f"seq_len={args.seq_len}  label_idx={label_idx}  "
          f"future_windows={args.seq_len - 1 - label_idx}")

    df, feature_cols, order_col, detect_time_col, label_time_col = load_features(args.features)

    if detect_time_col == "t_end" and label_time_col == "t_center":
        warnings.warn(
            "TIMING MISMATCH: detect_time uses t_end but label_time uses t_center. "
            "Ensure preprocessing used --window_anchor right."
        )

    df, feature_cols, mask_cols = add_missingness_masks(
        df, feature_cols, mode=args.missingness_masks
    )
    if args.missingness_masks != "none":
        print(f"[Missingness] mode={args.missingness_masks}  "
              f"added={len(mask_cols)}  total={len(feature_cols)}")

    df = add_stream_and_group_ids(
        df,
        detect_time_col=detect_time_col,
        nonflight_chunk_s=args.nonflight_chunk_s,
        use_file_id_as_stream=args.use_file_id_as_stream,
        nonflight_label=args.nonflight_label,
    )

    nf_groups = int(df.loc[df["is_nonflight_stream"], "group_id"].nunique())
    print(f"[SplitUnits] files={df['file'].nunique()}  "
          f"streams={df['stream_id'].nunique()}  groups={df['group_id'].nunique()}  "
          f"nonflight_groups={nf_groups}")

    print("Creating sequences...")
    sequences = create_sequences_with_label_pos(
        df, feature_cols, args.seq_len, label_idx,
        order_col=order_col,
        detect_time_col=detect_time_col,
        label_time_col=label_time_col,
    )
    if not sequences:
        raise RuntimeError("No sequences created. Check seq_len vs per-group window counts.")

    min_accel = min(s["accel_coverage"] for s in sequences)
    min_baro  = min(s["baro_coverage"]  for s in sequences)
    print(f"[Coverage] min_accel={min_accel:.3f}  min_baro={min_baro:.3f}")
    if min_accel < 0.5 or min_baro < 0.5:
        warnings.warn(f"Low coverage (accel={min_accel:.3f}, baro={min_baro:.3f}). "
                      f"Check --coverage in preprocessor.")

    timing_stats = estimate_sequence_delay_stats(sequences)
    print("[FixedTiming] t_detect - t_label (s):")
    if timing_stats["n"] > 0:
        print(f"  n={timing_stats['n']}  mean={timing_stats['mean_s']:.2f}  "
              f"median={timing_stats['median_s']:.2f}  "
              f"p90={timing_stats['p90_s']:.2f}  p95={timing_stats['p95_s']:.2f}")

    flight_files    = sorted(df.loc[~df["is_nonflight_stream"], "file"].unique())
    nonflight_files = sorted(df.loc[ df["is_nonflight_stream"], "file"].unique())
    print(f"[Split] flight files={len(flight_files)}  nonflight files={len(nonflight_files)}")

    if args.use_kfold:
        # K-fold path -- not used for thesis final models; raises to alert if called.
        raise NotImplementedError(
            "--use_kfold is implemented in kfold_split_groups() but the main loop "
            "is intentionally left as a stub. Use the single-split path for training."
        )

    # Single train/val/test split
    train_groups, val_groups, test_groups = split_groups_with_nonflight_focus(
        flight_groups=flight_files,
        nonflight_groups=nonflight_files,
        sequences=sequences,
        seed=int(args.seed),
        flight_val_frac=float(args.flight_val_frac),
        flight_test_frac=float(args.flight_test_frac),
        nf_val_frac=float(args.nf_val_frac),
        nf_test_frac=float(args.nf_test_frac),
        min_nf_val_groups=int(args.min_nf_val_groups),
        min_nf_val_hours=float(args.min_nf_val_hours),
        min_nf_test_groups=int(args.min_nf_test_groups),
    )

    file_to_groups  = df.groupby("file")["group_id"].apply(lambda x: set(x.astype(str))).to_dict()
    train_group_ids = set().union(*[file_to_groups.get(f, set()) for f in train_groups])
    val_group_ids   = set().union(*[file_to_groups.get(f, set()) for f in val_groups])
    test_group_ids  = set().union(*[file_to_groups.get(f, set()) for f in test_groups])

    train_seqs = [s for s in sequences if s["group_id"] in train_group_ids]
    if int(args.train_stride) > 1:
        train_seqs = train_seqs[::int(args.train_stride)]
    val_seqs  = [s for s in sequences if s["group_id"] in val_group_ids]
    test_seqs = [s for s in sequences if s["group_id"] in test_group_ids]

    print(f"\nSplit (deterministic, no leakage):")
    print(f"  Train: {len(train_seqs)} seqs  ({len(train_groups)} files)")
    print(f"  Val:   {len(val_seqs)} seqs  ({len(val_groups)} files)")
    print(f"  Test:  {len(test_seqs)} seqs  ({len(test_groups)} files)")

    # Leakage check
    tr_files   = set(df[df["group_id"].isin(train_group_ids)]["file"].unique())
    val_files  = set(df[df["group_id"].isin(val_group_ids)]["file"].unique())
    test_files = set(df[df["group_id"].isin(test_group_ids)]["file"].unique())
    assert not (tr_files & val_files),   "LEAKAGE: files in both train and val"
    assert not (tr_files & test_files),  "LEAKAGE: files in both train and test"
    assert not (val_files & test_files), "LEAKAGE: files in both val and test"
    print("[LeakageCheck] OK: no files appear in multiple splits")

    test_to = sum(s["y_takeoff"] for s in test_seqs)
    test_ld = sum(s["y_landing"] for s in test_seqs)
    print(f"[TestEvents] TAKEOFF={test_to}  LANDING={test_ld}")

    val_is_nonflight  = np.array([s["is_nonflight"] for s in val_seqs],  dtype=bool)
    test_is_nonflight = np.array([s["is_nonflight"] for s in test_seqs], dtype=bool)

    val_seqs_nf  = [s for s in val_seqs  if s["is_nonflight"]]
    val_seqs_f   = [s for s in val_seqs  if not s["is_nonflight"]]
    test_seqs_nf = [s for s in test_seqs if s["is_nonflight"]]

    for split_name, nf_seqs in [("VAL", val_seqs_nf), ("TEST", test_seqs_nf)]:
        if nf_seqs:
            h = count_stream_triggers_by_group(
                nf_seqs, np.zeros(len(nf_seqs)), thr=1.0,
                trigger_k=int(args.trigger_k), cooldown_s=float(args.cooldown_s),
            ).get("hours", 0.0)
        else:
            h = 0.0
        print(f"[NONFLIGHT {split_name}] groups={len({s['group_id'] for s in nf_seqs})}  hours={h:.3f}")

    # Scaler
    print(f"\n[Scaler] fit_mode={args.scaler_fit_mode}")
    if args.scaler_fit_mode == "sequence":
        scaler = fit_scaler_from_train_sequences(train_seqs)
    else:
        scaler = fit_scaler_from_train_windows(df, feature_cols, train_group_ids, order_col)

    mean_t  = torch.tensor(scaler.mean_.astype(np.float32),  device=device)
    scale_t = torch.tensor(scaler.scale_.astype(np.float32), device=device)

    # Positive class weights for BCE
    n_total   = len(train_seqs)
    n_to      = sum(s["y_takeoff"] for s in train_seqs)
    n_ld      = sum(s["y_landing"] for s in train_seqs)
    pos_wt_to = min((n_total - n_to) / max(n_to, 1), args.max_pos_weight)
    pos_wt_ld = min((n_total - n_ld) / max(n_ld, 1), args.max_pos_weight)
    print(f"Pos weights (capped at {args.max_pos_weight}): TO={pos_wt_to:.1f}  LD={pos_wt_ld:.1f}")

    common_dl = dict(
        num_workers=dl_num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.persistent_workers and dl_num_workers > 0),
    )
    if dl_num_workers > 0:
        common_dl["prefetch_factor"] = dl_prefetch

    train_loader = DataLoader(
        OversampledDataset(
            train_seqs,
            oversample_factor=args.oversample_factor,
            primary_oversample=args.primary_oversample,
            target_event=args.target_event,
            seed=args.seed,
        ),
        batch_size=args.batch_size, shuffle=True, **common_dl,
    )
    val_loader  = DataLoader(RawSequenceDataset(val_seqs),
                             batch_size=args.batch_size, shuffle=False, **common_dl)
    test_loader = DataLoader(RawSequenceDataset(test_seqs),
                             batch_size=args.batch_size, shuffle=False, **common_dl)

    model = FlightGRU(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.use_focal_loss:
        crit_to = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        crit_ld = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"[Loss] Focal (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        crit_to = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_wt_to], device=device))
        crit_ld = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_wt_ld], device=device))
        print(f"[Loss] BCE (pos_weight TO={pos_wt_to:.1f}, LD={pos_wt_ld:.1f})")

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    print("\n" + "=" * 60)
    print("TRAINING (early stopping on flight-only validation PR-AUC)")
    print("=" * 60)

    best_score = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch_event_focused(
            model=model, loader=train_loader, optimizer=opt,
            criterion_to=crit_to, criterion_ld=crit_ld,
            device=device, mean_t=mean_t, scale_t=scale_t,
            target_event=args.target_event,
            main_loss_weight=float(args.main_loss_weight),
            aux_loss_weight=float(args.aux_loss_weight),
        )
        sched.step()

        val_res   = evaluate(model, val_loader, device, mean_t, scale_t)
        is_flight = np.array([not s["is_nonflight"] for s in val_seqs], dtype=bool)

        pr_to = (float(average_precision_score(
                    val_res["y_takeoff"][is_flight], val_res["prob_takeoff"][is_flight]))
                 if (is_flight.any() and val_res["y_takeoff"][is_flight].sum() > 0)
                 else 0.0)
        pr_ld = (float(average_precision_score(
                    val_res["y_landing"][is_flight], val_res["prob_landing"][is_flight]))
                 if (is_flight.any() and val_res["y_landing"][is_flight].sum() > 0)
                 else 0.0)

        te = args.target_event.lower().strip()
        primary_val = pr_to if te == "takeoff" else (pr_ld if te == "landing" else 0.5 * (pr_to + pr_ld))

        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={train_loss:.4f}  "
                  f"val PR-AUC TO={pr_to:.3f}  LD={pr_ld:.3f}  primary={primary_val:.3f}")

        if primary_val > best_score:
            best_score = primary_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nTraining complete. Best val PR-AUC ({args.target_event}): {best_score:.4f}")

    calibration_temperature = 1.0
    if args.use_temperature_scaling:
        print("\n" + "=" * 60 + "\nTEMPERATURE SCALING\n" + "=" * 60)
        temp_model              = calibrate_temperature(
            model=model, val_loader=val_loader, device=device,
            mean_t=mean_t, scale_t=scale_t,
        )
        calibration_temperature = temp_model.get_temperature()
        model                   = temp_model
        print(f"[Calibration] temperature={calibration_temperature:.4f}")

    # Threshold selection on validation set
    print("\n" + "=" * 60 + "\nTHRESHOLD SELECTION (validation)\n" + "=" * 60)
    val_res          = evaluate(model, val_loader, device, mean_t, scale_t)
    thresholds:      Dict[str, float] = {}
    thr_warn:        Dict[str, bool]  = {}
    thr_nf_stats:    Dict[str, Any]   = {}
    thr_obj_stats:   Dict[str, Any]   = {}
    val_is_flight    = ~val_is_nonflight

    def _get_val_arrays(event: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if event == "TAKEOFF":
            return (val_res["y_takeoff"][val_is_flight],  val_res["prob_takeoff"][val_is_flight],
                    val_res["y_takeoff"][val_is_nonflight], val_res["prob_takeoff"][val_is_nonflight])
        return (val_res["y_landing"][val_is_flight],   val_res["prob_landing"][val_is_flight],
                val_res["y_landing"][val_is_nonflight],  val_res["prob_landing"][val_is_nonflight])

    for ev in active_events:
        y_f, p_f, y_nf, p_nf = _get_val_arrays(ev)
        if args.thr_mode == "fp_budget_nonflight" and val_seqs_nf and p_nf.size > 0 and val_seqs_f:
            thr, nf_stats, warn, obj_stats = select_threshold_fp_budget_with_objective(
                event_name=ev,
                val_nf_seqs=val_seqs_nf,    val_nf_probs=p_nf,
                val_f_seqs=val_seqs_f,      val_f_probs=p_f,
                val_f_y=y_f,                df_all=df,
                order_col=order_col,        label_time_col=label_time_col,
                trigger_k=int(args.trigger_k), cooldown_s=float(args.cooldown_s),
                hit_window_s=float(args.hit_window_s),
                fp_budget_per_hour=float(args.fp_budget_per_hour),
                thr_grid=int(args.thr_grid), objective=str(args.thr_objective),
                beta=float(args.fbeta),
                min_nf_hours_for_budget=float(args.min_nf_hours_for_budget),
            )
            thresholds[ev]    = float(thr)
            thr_warn[ev]      = bool(warn)
            thr_nf_stats[ev]  = nf_stats
            thr_obj_stats[ev] = obj_stats
        else:
            thresholds[ev]    = find_best_threshold(y_f, p_f, beta=args.fbeta)
            thr_warn[ev]      = args.thr_mode == "fp_budget_nonflight"
            thr_nf_stats[ev]  = None
            thr_obj_stats[ev] = {"objective": "fbeta"}

    for ev in active_events:
        print(f"  {ev} threshold={thresholds[ev]:.6f}")

    val_metrics: Dict[str, Any] = {}
    for ev in active_events:
        y_f, p_f, _, _ = _get_val_arrays(ev)
        val_metrics[ev] = compute_metrics(y_f, p_f,
                                          name=f"{ev} (val, flight-only)",
                                          beta=args.fbeta,
                                          fixed_threshold=float(thresholds[ev]),
                                          verbose=True)

    # Final evaluation on test set
    print("\n" + "=" * 60 + "\nFINAL EVALUATION (TEST)\n" + "=" * 60)
    test_res     = evaluate(model, test_loader, device, mean_t, scale_t)
    test_metrics: Dict[str, Any] = {}

    for ev in active_events:
        prob_ev = test_res["prob_takeoff"] if ev == "TAKEOFF" else test_res["prob_landing"]
        y_ev    = test_res["y_takeoff"]    if ev == "TAKEOFF" else test_res["y_landing"]
        test_metrics[ev] = compute_metrics(y_ev, prob_ev,
                                           name=f"{ev} (test)",
                                           beta=args.fbeta,
                                           fixed_threshold=float(thresholds[ev]),
                                           verbose=True)

    # Nonflight FP/hour on test set
    test_nf_fp: Dict[str, Any] = {}
    if test_seqs_nf:
        nf_idx_arr = np.array([i for i, s in enumerate(test_seqs) if s["is_nonflight"]], dtype=int)
        if nf_idx_arr.size > 0:
            for ev in active_events:
                p_nf = (test_res["prob_takeoff"] if ev == "TAKEOFF" else test_res["prob_landing"])[nf_idx_arr]
                test_nf_fp[ev] = count_stream_triggers_by_group(
                    test_seqs_nf, p_nf, float(thresholds[ev]),
                    int(args.trigger_k), float(args.cooldown_s),
                )
            print("\n[NONFLIGHT FP/h] (TEST)")
            for ev in active_events:
                d = test_nf_fp.get(ev, {})
                print(f"  {ev}: fp_h={d.get('fp_per_hour')}  triggers={d.get('triggers')}")

    # Event-level hit/latency metrics
    test_flights  = {str(s["flight_id"]) for s in test_seqs if not s["is_nonflight"]}
    segments_test = build_event_segments_for_flights(
        df=df, flights_subset=test_flights,
        order_col=order_col, label_time_col=label_time_col,
    )
    event_eval_test: Dict[str, Any] = {}

    for ev in active_events:
        prob_ev = test_res["prob_takeoff"] if ev == "TAKEOFF" else test_res["prob_landing"]
        segs    = segments_test.get(ev, {})
        event_eval_test[ev] = compute_event_eval_from_triggers(
            test_seqs=test_seqs, prob_event=prob_ev,
            thr=float(thresholds[ev]), trigger_k=int(args.trigger_k),
            cooldown_s=float(args.cooldown_s),
            segments_by_flight=segs, hit_window_s=float(args.hit_window_s),
        )
        print_event_latency_block(ev, event_eval_test[ev])
        details = event_eval_test[ev].get("event_details", [])
        if len(details) >= 3:
            mean_hr, ci_lo, ci_hi = bootstrap_event_hit_rate(details, seed=args.seed)
            print(f"[Bootstrap CI] hit_rate={mean_hr:.3f}  95% CI=[{ci_lo:.3f}, {ci_hi:.3f}]")

    # Write metrics JSON
    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "model_type":              "gru_union_final_v3",
                "event":                   ev,
                "seed":                    int(args.seed),
                "use_temperature_scaling": bool(args.use_temperature_scaling),
                "calibration_temperature": float(calibration_temperature),
                "threshold":               float(thresholds[ev]),
                "metrics_val":             val_metrics.get(ev),
                "metrics_test":            test_metrics.get(ev),
                "event_eval_test":         event_eval_test.get(ev),
            }
            for ev in active_events
        ]
        write_json_strict(out_path, rows)
        print(f"\n[METRICS] Wrote to {out_path}")

    # ONNX export
    if args.export_dir:
        preflight_export_deps(args.verify_onnx)
        export_dir = Path(args.export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70 + "\nEXPORTING FOR ANDROID\n" + "=" * 70)

        head_index = EVENT_TO_HEAD[args.export_event]
        print(f"Export event: {args.export_event}  (head {head_index})")

        write_json_strict(export_dir / "features.json", feature_cols)
        np.savez(export_dir / "scaler.npz",
                 mean=scaler.mean_.astype(np.float32),
                 scale=scaler.scale_.astype(np.float32))
        base_model = model.model if isinstance(model, ModelWithTemperature) else model
        torch.save(base_model.cpu().state_dict(), export_dir / "checkpoint.pt")

        golden_X  = sample_golden_from_sequences(test_seqs, n=args.emit_golden, seed=args.seed)
        golden_y_to = np.array([test_seqs[i]["y_takeoff"] for i in range(len(golden_X))], dtype=np.float32)
        golden_y_ld = np.array([test_seqs[i]["y_landing"]  for i in range(len(golden_X))], dtype=np.float32)

        wrapper    = ScaledGRUProb(model.cpu().eval(), scaler.mean_, scaler.scale_, head_index).eval()
        onnx_path  = export_dir / "model.onnx"
        export_onnx(wrapper, args.seq_len, len(feature_cols), onnx_path, args.opset)

        with torch.no_grad():
            golden_out = wrapper(torch.from_numpy(golden_X).float()).cpu().numpy()

        parity_diff   = None
        parity_status = "unknown"
        opset_actual  = args.opset

        if args.verify_onnx:
            try:
                parity_diff   = verify_onnx_parity(onnx_path, golden_X, golden_out)
                parity_status = "ok" if parity_diff < args.parity_warn_if_gt else "warning"
                opset_actual  = read_actual_opset(onnx_path)
                print(f"ONNX parity max_abs_diff={parity_diff:.8f}  status={parity_status}")
            except Exception as e:
                print(f"Warning: ONNX parity check failed: {e}")

        np.savez(export_dir / "golden.npz",
                 X=golden_X.astype(np.float32),
                 y_takeoff=golden_y_to,
                 y_landing=golden_y_ld,
                 p_out=golden_out.astype(np.float32))
        print(f"Saved golden.npz ({len(golden_X)} samples)")

        shipped_thr = float(thresholds[args.export_event])

        if "win_s" in df.columns:
            median_win_s = _median_finite(pd.to_numeric(df["win_s"], errors="coerce").to_numpy())
        else:
            median_win_s = 25.0 if args.target_event == "takeoff" else 20.0

        if "hop_s" in df.columns:
            median_hop_s = _median_finite(pd.to_numeric(df["hop_s"], errors="coerce").to_numpy())
        else:
            median_hop_s = 12.0

        thresholds_from_val = {
            "takeoff_thr": float(thresholds.get("TAKEOFF", 0.0)),
            "landing_thr": float(thresholds.get("LANDING", 0.0)),
        }

        def _metrics_block(ev: str, src: Dict[str, Any]) -> Dict[str, Any]:
            m = src.get(ev, {})
            return {
                "pr_auc":    float(m.get("pr_auc",    0.0)),
                "roc_auc":   float(m.get("roc_auc",   0.5)),
                "threshold": float(m.get("threshold", 0.0)),
                "TP":        int(m.get("TP", 0)),
                "FN":        int(m.get("FN", 0)),
                "FP":        int(m.get("FP", 0)),
                "TN":        int(m.get("TN", 0)),
                "precision": float(m.get("precision", 0.0)),
                "recall":    float(m.get("recall",    0.0)),
                "f1":        float(m.get("f1",        0.0)),
                "fbeta":     float(m.get("fbeta",     0.0)),
            }

        metrics_pytorch = {
            "val":  {ev.lower(): _metrics_block(ev, val_metrics)  for ev in ["TAKEOFF", "LANDING"]},
            "test": {ev.lower(): _metrics_block(ev, test_metrics) for ev in ["TAKEOFF", "LANDING"]},
        }

        profile = {
            "model_type":   "gru",
            "export_event": args.export_event,
            "target_event": args.target_event,
            "head_index":   int(head_index),
            "threshold":    shipped_thr,
            "trigger_k":    int(args.trigger_k),
            "scaler_fit_mode": args.scaler_fit_mode,
            "seq_len":      int(args.seq_len),
            "label_shift":  int(args.label_shift),
            "label_idx":    int(label_idx),
            "n_features":   int(len(feature_cols)),
            "timing_columns": {
                "order_col":      order_col,
                "detect_time_col": detect_time_col,
                "label_time_col": label_time_col,
            },
            "timing": {
                "hop_s":          float(median_hop_s),
                "win_len_s":      float(median_win_s),
                "future_windows": int(args.seq_len - 1 - label_idx),
            },
            "event_focused_training": {
                "primary_oversample": int(args.primary_oversample),
                "oversample_factor":  int(args.oversample_factor),
                "main_loss_weight":   float(args.main_loss_weight),
                "aux_loss_weight":    float(args.aux_loss_weight),
                "early_stop_metric":  "val_pr_auc_primary",
            },
            "thresholds_from_val":       thresholds_from_val,
            "use_temperature_scaling":   bool(args.use_temperature_scaling),
            "calibration_temperature":   float(calibration_temperature),
            "metrics_pytorch":           metrics_pytorch,
            "onnx": {
                "input_name":        "x",
                "output_name":       "p",
                "output_is_probability": True,
                "opset_requested":   int(args.opset),
                "opset_actual":      int(opset_actual),
                "static_batch":      1,
            },
            "onnx_parity": {
                "golden_rows":  int(len(golden_X)),
                "max_abs_diff": parity_diff,
                "warn_if_gt":   args.parity_warn_if_gt,
                "status":       parity_status,
            },
        }

        write_json_strict(export_dir / "profile.json", profile)

        print(f"\nExport complete.")
        print(f"  Directory:  {export_dir}")
        print(f"  Event:      {args.export_event}")
        print(f"  Threshold:  {shipped_thr:.4f}")
        print(f"  Test PR-AUC: {test_metrics[args.export_event]['pr_auc']:.4f}")
        print(f"  Files: model.onnx  profile.json  features.json  "
              f"scaler.npz  golden.npz  checkpoint.pt")


if __name__ == "__main__":
    main()