"""
XGBoost baseline detector for TAKEOFF and LANDING events.

This script is a baseline comparison against the GRU model. It is NOT deployed
to the Android app.

How it works
------------
1. Load features_imputed.csv from the preprocessor.
2. Split by physical file using the SAME deterministic algorithm as the GRU
   script, so all three models are evaluated on identical test flights when
   the same seed is used.
3. Select a threshold using group-aware cross-validation on train+val.
4. Train a final model on ALL of train+val with NO early stopping.
5. Evaluate on the held-out test set using the same streaming trigger logic
   as the GRU (trigger_k consecutive windows >= threshold).
6. Optionally repeat for multiple seeds to estimate variance.

The final model is NOT exported. Only metrics are written to disk.

TAKEOFF and LANDING use different preprocessed datasets (25s and 20s base
windows respectively) and must be run as separate commands.

Usage -- takeoff
----------------
    python3 xgb.py \
        --features all_data/preprocessed_25/features_raw.csv \
        --events TAKEOFF \
        --skip_tuning \
        --fixed_model_params '{"n_estimators":300,"learning_rate":0.103,"max_depth":4,
                                "subsample":0.91,"colsample_bytree":0.65,
                                "reg_alpha":0.0046,"reg_lambda":0.0089,
                                "min_child_weight":3.04}' \
        --fixed_wcfg '{"near_s":60.0,"mid_s":166.0,"w_near":1.0,
                        "w_mid":0.677,"w_far":0.268,"pos_boost":"auto"}' \
        --use_dist_weights --dist_source event \
        --seeds 41 42 --fbeta 2.0 \
        --trigger_k 2 --cooldown_s 60 --hit_window_s 90 \
        --metrics_out xgb_takeoff_metrics.json

Usage -- landing
----------------
    python3 xgb.py \
        --features all_data/preprocessed_20/features_raw.csv \
        --events LANDING \
        --skip_tuning \
        --fixed_model_params '{"n_estimators":300,"learning_rate":0.103,"max_depth":4,
                                "subsample":0.91,"colsample_bytree":0.65,
                                "reg_alpha":0.0046,"reg_lambda":0.0089,
                                "min_child_weight":3.04}' \
        --fixed_wcfg '{"near_s":60.0,"mid_s":166.0,"w_near":1.0,
                        "w_mid":0.677,"w_far":0.268,"pos_boost":"auto"}' \
        --use_dist_weights --dist_source event \
        --seeds 41 42 --fbeta 2.0 \
        --trigger_k 2 --cooldown_s 60 --hit_window_s 90 \
        --metrics_out xgb_landing_metrics.json
"""

import argparse
import json
import warnings
import inspect
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.impute import SimpleImputer

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_SGK = True
except Exception:
    HAVE_SGK = False

try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

try:
    import optuna
    HAVE_OPTUNA = True
except Exception:
    HAVE_OPTUNA = False


# ---------------------------------------------------------------------------
# Known-good hyperparameters for warm-starting Optuna on the takeoff event.
# These were found during initial tuning and can be enqueued as trial 0.
# ---------------------------------------------------------------------------
TAKEOFF_BASELINE = dict(
    n_estimators=300,
    learning_rate=0.11761469533691034,
    max_depth=4,
    subsample=0.8949539804664459,
    colsample_bytree=0.6337777284559556,
    reg_alpha=0.0041580740217060964,
    reg_lambda=0.002080708693094624,
    min_child_weight=2.9759632769780757,
)
TAKEOFF_BASELINE_WEIGHTS = dict(
    near_s=60.0, mid_s=166.0, w_near=1.0,
    w_mid=0.6771542177142438, w_far=0.26785313677518174, pos_boost="auto",
)


# ---------------------------------------------------------------------------
# JSON utilities
# ---------------------------------------------------------------------------

def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy scalars and NaN/Inf to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return str(obj)


def write_json_strict(path: Path, data: Any) -> None:
    """Write data to a JSON file. NaN and Inf are converted to null."""
    path.write_text(
        json.dumps(sanitize_for_json(data), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Columns that should never be used as model features
# ---------------------------------------------------------------------------

META_COLS = {
    "file", "label", "window_id",
    "t_start", "t_end", "t_center", "t_anchor",
    "t_event", "t_takeoff", "t_landing",
    "dist_to_takeoff", "dist_to_landing", "dist_to_event",
    "win_start", "win_end", "win_len", "win_hop",
    "_hnm_keep_p", "sample_weight",
    "landing_proximity", "near_landing_30s", "near_landing_60s",
    "takeoff_proximity", "near_takeoff_30s", "near_takeoff_60s",
    "__row_idx",
    "domain", "file_id", "stream_id", "group_id", "is_nonflight",
}

META_PREFIXES = ("win_", "_hnm_")


# ---------------------------------------------------------------------------
# Timing column selection
# ---------------------------------------------------------------------------

def _pick_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_time_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Return (order_col, detect_time_col, label_time_col)."""
    detect_time_col = _pick_first_existing(df, ["t_end", "t_anchor", "t_center"])
    if detect_time_col is None:
        raise RuntimeError("CSV missing all of: t_end, t_anchor, t_center")
    label_time_col = _pick_first_existing(df, ["t_anchor", "t_center"]) or detect_time_col
    return detect_time_col, detect_time_col, label_time_col


def deterministic_sort_cols(df: pd.DataFrame, primary_time_col: str) -> List[str]:
    """Return a sort-column list for stable ordering within a flight."""
    cols = ["file", primary_time_col]
    if "window_id" in df.columns:
        cols.append("window_id")
    elif "t_start" in df.columns:
        cols.append("t_start")
    cols.append("__row_idx")
    return cols


# ---------------------------------------------------------------------------
# Nonflight identification
# ---------------------------------------------------------------------------

def identify_nonflight_windows(
    df: pd.DataFrame, nonflight_label: str = "NONFLIGHT"
) -> pd.Series:
    """Return a boolean Series that is True for nonflight windows."""
    if "domain" in df.columns:
        return df["domain"].astype(str).str.lower().fillna("") == "nonflight"
    if "file" not in df.columns or "label" not in df.columns:
        return pd.Series(np.zeros(len(df), dtype=bool), index=df.index)
    labels_upper       = df["label"].astype(str).str.upper()
    has_event_per_file = labels_upper.isin(["TAKEOFF", "LANDING"]).groupby(df["file"]).transform("any")
    return (~has_event_per_file) | (labels_upper == str(nonflight_label).upper())


# ---------------------------------------------------------------------------
# Deterministic train/val/test split
# ---------------------------------------------------------------------------

def group_hours_from_df(
    df: pd.DataFrame, file_col: str, time_col: str
) -> Dict[str, float]:
    """
    Compute recording duration in hours for each file in df.

    Pass only the FLIGHT subset here. Nonflight files will be absent from the
    returned dict and get duration 0.0 when looked up, so they sort purely
    alphabetically. This matches GRU behaviour when --use_file_id_as_stream
    is active, where nonflight chunk group_ids never match raw filenames and
    therefore also default to duration 0.
    """
    out: Dict[str, float] = {}
    for fid, sub in df.groupby(file_col):
        t = pd.to_numeric(sub[time_col], errors="coerce").to_numpy(dtype=float)
        t = t[np.isfinite(t)]
        out[str(fid)] = (
            max(float(np.max(t) - np.min(t)), 0.0) / 3600.0
            if t.size >= 2 else 0.0
        )
    return out


def split_groups_deterministic(
    *,
    flight_groups: List[str],
    nonflight_groups: List[str],
    dur_map: Dict[str, float],
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

    This is a direct port of split_groups_with_nonflight_focus from the GRU
    script. Using the same seed and fractions produces the same test set in
    both scripts, which is required for a fair three-way comparison.

    Flight files: sorted alphabetically then rotated by a seed-derived offset.
    Nonflight files: sorted by (duration DESC, filename ASC). Because dur_map
    only contains flight files, all nonflight files get duration 0 and end up
    sorted purely alphabetically — matching the GRU.

    Returns (train_files, val_files, test_files) as sets of filename strings.
    """
    rng = np.random.default_rng(int(seed))

    # Flight split: alphabetical sort + seed rotation.
    fg = sorted(flight_groups)
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

    # Nonflight split: longest recordings first, ties broken alphabetically.
    nfg = sorted(nonflight_groups,
                 key=lambda g: (-float(dur_map.get(str(g), 0.0)), str(g)))

    if not nfg:
        tr_nf = va_nf = te_nf = set()
    else:
        n_total       = len(nfg)
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

        # If nothing ended up in train, move the alphabetically smallest val
        # group over — deterministic tiebreaker.
        if not tr_nf and n_total >= 3 and len(va_nf) > 1:
            g_move = min(va_nf)
            va_nf.remove(g_move)
            tr_nf.add(g_move)

    return (
        set.union(tr_f, tr_nf),
        set.union(va_f, va_nf),
        set.union(te_f, te_nf),
    )


# ---------------------------------------------------------------------------
# Streaming trigger logic
# ---------------------------------------------------------------------------

def _duration_hours(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    return max(float(np.max(t) - np.min(t)), 0.0) / 3600.0 if t.size >= 2 else 0.0


def _collapse_duplicate_times(
    t: np.ndarray, p: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Where two windows share a timestamp, keep the higher probability."""
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    if t.size <= 1 or not np.any(np.diff(t) == 0):
        return t, p
    uniq_t, inv = np.unique(t, return_inverse=True)
    p2 = np.full_like(uniq_t, -np.inf, dtype=float)
    np.maximum.at(p2, inv, p)
    return uniq_t, p2


def detect_trigger_times(
    t: np.ndarray, p: np.ndarray,
    thr: float, trigger_k: int, cooldown_s: float,
) -> List[float]:
    """
    Simulate the on-device trigger state machine.
    Fires when trigger_k consecutive windows all score >= thr.
    A cooldown_s period must pass before a new trigger can fire.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(t, kind="mergesort")
    t, p  = t[order], p[order]
    t, p  = _collapse_duplicate_times(t, p)

    k          = max(1, int(trigger_k))
    cooldown_s = max(0.0, float(cooldown_s))
    consec     = 0
    last_fire  = -np.inf
    triggers: List[float] = []

    for tj, pj in zip(t, p):
        pj = float(pj) if np.isfinite(pj) else -np.inf
        if (tj - last_fire) < cooldown_s:
            consec = 0
            continue
        if pj >= thr:
            consec += 1
            if consec >= k:
                triggers.append(float(tj))
                last_fire = float(tj)
                consec    = 0
        else:
            consec = 0
    return triggers


def count_nonflight_triggers(
    df_nf: pd.DataFrame, prob: np.ndarray,
    thr: float, trigger_k: int, cooldown_s: float,
    detect_time_col: str,
) -> Dict[str, Any]:
    """Count streaming triggers on nonflight data and compute FP/hour."""
    if len(df_nf) == 0 or len(prob) == 0:
        return {"triggers": 0, "hours": 0.0, "fp_per_hour": None, "n_groups": 0}
    if len(prob) != len(df_nf):
        raise RuntimeError(f"Length mismatch: prob={len(prob)} vs df_nf={len(df_nf)}")

    t_detect = pd.to_numeric(df_nf[detect_time_col], errors="coerce").to_numpy(dtype=float)
    by_file: Dict[str, List[int]] = defaultdict(list)
    for i, fid in enumerate(df_nf["file"].astype(str).to_numpy()):
        by_file[fid].append(i)

    total_triggers = 0
    total_hours    = 0.0

    for fid, idxs in by_file.items():
        t = t_detect[idxs]
        p = prob[idxs]
        t_c, _ = _collapse_duplicate_times(
            np.sort(t[np.isfinite(t)]), np.zeros(np.isfinite(t).sum())
        )
        total_hours    += _duration_hours(t_c)
        total_triggers += len(detect_trigger_times(t, p, thr, trigger_k, cooldown_s))

    fp_per_hour = (total_triggers / total_hours) if total_hours > 0 else None
    return {
        "triggers":    int(total_triggers),
        "hours":       float(total_hours),
        "fp_per_hour": float(fp_per_hour) if fp_per_hour is not None else None,
        "n_groups":    int(len(by_file)),
    }


# ---------------------------------------------------------------------------
# Distance helpers for sample weighting
# ---------------------------------------------------------------------------

def _abs_series(s: pd.Series) -> pd.Series:
    try:
        return s.astype(float).abs()
    except Exception:
        return pd.to_numeric(s, errors="coerce").abs()


def pick_distance(
    df: pd.DataFrame, event_label: str, dist_source: str
) -> pd.Series:
    """Return the absolute distance-to-event series for sample weighting."""
    n    = len(df)
    far  = pd.Series(np.full(n, 1e9, dtype=float), index=df.index)
    ev   = (event_label or "").upper().strip()
    src  = (dist_source or "auto").lower().strip()
    have_to = "dist_to_takeoff" in df.columns
    have_ld = "dist_to_landing" in df.columns
    have_ev = "dist_to_event"   in df.columns

    if src == "event":
        if ev == "TAKEOFF" and have_to:
            return _abs_series(df["dist_to_takeoff"]).fillna(1e9)
        if ev == "LANDING" and have_ld:
            return _abs_series(df["dist_to_landing"]).fillna(1e9)
        src = "auto"

    if src == "min" and have_to and have_ld:
        return pd.Series(
            np.minimum(_abs_series(df["dist_to_takeoff"]).to_numpy(),
                       _abs_series(df["dist_to_landing"]).to_numpy()),
            index=df.index).fillna(1e9)

    if src in ("auto", "min"):
        if ev == "TAKEOFF" and have_to:
            return _abs_series(df["dist_to_takeoff"]).fillna(1e9)
        if ev == "LANDING" and have_ld:
            return _abs_series(df["dist_to_landing"]).fillna(1e9)
        if have_to and have_ld:
            return pd.Series(
                np.minimum(_abs_series(df["dist_to_takeoff"]).to_numpy(),
                           _abs_series(df["dist_to_landing"]).to_numpy()),
                index=df.index).fillna(1e9)
        if have_ev:
            return _abs_series(df["dist_to_event"]).fillna(1e9)

    return far


def make_xy(
    df: pd.DataFrame, event_label: str, dist_source: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Build (X, y, groups, distances) for one event type."""
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    if "file" not in df.columns:
        df = df.copy()
        df["file"] = np.arange(len(df)).astype(str)

    num_cols = df.select_dtypes(include=[np.number]).columns
    cols = [c for c in num_cols
            if c not in META_COLS and not any(c.startswith(p) for p in META_PREFIXES)]
    if not cols:
        raise ValueError("No numeric feature columns found after removing meta columns.")

    ev = (event_label or "").upper().strip()
    return (
        df[cols],
        (df["label"].astype(str).str.upper() == ev).astype(int),
        df["file"].astype(str),
        pick_distance(df, ev, dist_source),
    )


def drop_zero_variance(X: pd.DataFrame) -> pd.DataFrame:
    """Remove constant columns before passing to XGBoost."""
    Xn  = X.select_dtypes(include=[np.number])
    var = Xn.var(axis=0, ddof=0)
    return Xn[var.index[var > 0.0]]


def compute_scale_pos_weight(y: np.ndarray) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / max(pos, 1.0)


def to_float32(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float32, copy=False)


def _safe_auc(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """Compute PR-AUC and ROC-AUC, returning NaN on failure."""
    pr = roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, prob))
    except Exception:
        pass
    try:
        if len(np.unique(y_true)) >= 2:
            roc = float(roc_auc_score(y_true, prob))
    except Exception:
        pass
    return pr, roc


def _check_split(y_trval: pd.Series, y_test: pd.Series, event_label: str) -> None:
    """Raise if either split has no positives or only one class."""
    if int(y_trval.sum()) == 0:
        raise SystemExit(f"[{event_label}] No positives in train+val. Try a different --seed.")
    if int(y_test.sum()) == 0:
        raise SystemExit(f"[{event_label}] No positives in test. Try a different --seed.")
    if len(np.unique(y_trval.to_numpy())) < 2:
        raise SystemExit(f"[{event_label}] Train+val has only one class.")
    if len(np.unique(y_test.to_numpy())) < 2:
        raise SystemExit(f"[{event_label}] Test has only one class.")


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def best_threshold_by_fbeta(
    y_true: np.ndarray, prob: np.ndarray, beta: float = 1.0
) -> float:
    """Find the threshold on the PR curve that maximises F-beta."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    p, r, thr = precision_recall_curve(y_true, prob)
    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (1 + b2) * (p * r) / np.maximum(b2 * p + r, 1e-12)
    best_idx = min(int(np.nanargmax(f)), len(thr) - 1)
    return float(thr[max(0, best_idx - 1)])


def best_threshold_for_recall(
    y_true: np.ndarray, prob: np.ndarray, target_recall: float
) -> float:
    """Find the lowest threshold that achieves at least target_recall."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    order     = np.argsort(prob)[::-1]
    cum_tp    = np.cumsum(y_true[order] == 1)
    total_pos = max(int((y_true == 1).sum()), 1)
    idx       = np.where(cum_tp / total_pos >= target_recall)[0]
    return float(prob[order[idx[-1]]]) if len(idx) > 0 else 0.0


# ---------------------------------------------------------------------------
# Distance-aware sample weighting
# ---------------------------------------------------------------------------

def compute_sample_weights(
    y, dist,
    near_s: float = 60.0, mid_s: float = 180.0,
    w_near: float = 1.0, w_mid: float = 0.6, w_far: float = 0.25,
    pos_boost="auto",
) -> np.ndarray:
    """
    Weight samples by distance zone; up-weight positives to balance total mass.
    All distances are treated as absolute values.
    """
    y = np.asarray(y)
    d = np.abs(np.asarray(dist, dtype=float))
    w = np.where(d <= float(near_s), float(w_near),
                 np.where(d <= float(mid_s), float(w_mid), float(w_far))).astype(float)
    pos_mask = (y == 1)
    pos_w    = (w[~pos_mask].sum() / max(1, int(pos_mask.sum()))
                if pos_boost == "auto" else float(pos_boost))
    w[pos_mask] = pos_w
    return np.maximum(w, 1e-8)


# ---------------------------------------------------------------------------
# XGBoost training helpers
# ---------------------------------------------------------------------------

def _xgb_supports_eval_sample_weight() -> bool:
    if not HAVE_XGB:
        return False
    try:
        return "eval_sample_weight" in inspect.signature(xgb.XGBClassifier.fit).parameters
    except Exception:
        return False


XGB_HAS_EVAL_SW = _xgb_supports_eval_sample_weight()


def fit_xgb_fold(
    X_tr, y_tr, X_va, y_va, params,
    sample_weight_tr=None, sample_weight_va=None,
    early_rounds: int = 50, seed: int = 42,
):
    """
    Train XGBoost on one CV fold with early stopping.
    The imputer is always fit only on the training fold to prevent leakage.
    """
    imp = SimpleImputer(strategy="median")
    Xtr = to_float32(imp.fit_transform(X_tr))
    Xva = to_float32(imp.transform(X_va.reindex(columns=X_tr.columns)))
    ytr = y_tr.to_numpy()
    yva = y_va.to_numpy()

    if len(np.unique(ytr)) < 2:
        raise ValueError("Train fold has only one class.")

    spw = 1.0 if sample_weight_tr is not None else compute_scale_pos_weight(ytr)
    xgb_params = {
        **params,
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "tree_method": "hist", "n_jobs": (os.cpu_count() or 1),
        "random_state": seed, "scale_pos_weight": spw, "max_delta_step": 1,
    }

    model  = xgb.XGBClassifier(**xgb_params)
    fit_kw = dict(X=Xtr, y=ytr, eval_set=[(Xva, yva)], verbose=False)
    if sample_weight_tr is not None:
        fit_kw["sample_weight"] = sample_weight_tr
    if sample_weight_va is not None and XGB_HAS_EVAL_SW:
        fit_kw["eval_sample_weight"] = [sample_weight_va]

    # Try three API shapes to handle differences across xgboost versions.
    for attempt in range(3):
        try:
            if attempt == 0:
                model.fit(**fit_kw, early_stopping_rounds=early_rounds)
            elif attempt == 1:
                fit_kw.pop("eval_sample_weight", None)
                model.fit(**fit_kw, callbacks=[xgb.callback.EarlyStopping(
                    rounds=early_rounds, save_best=True)])
            else:
                model.fit(**fit_kw)
            break
        except TypeError:
            continue

    return model, model.predict_proba(Xva)[:, 1], imp


def fit_xgb_full_no_early(
    X_trval, y_trval, d_trval, params, weight_cfg,
    use_dist_weights: bool, seed: int,
):
    """
    Train the final XGBoost model on all train+val data with no early stopping.
    The number of trees is fixed from params, so no holdout is needed.
    """
    if len(np.unique(y_trval.to_numpy())) < 2:
        raise SystemExit("Train+val has only one class; cannot train.")

    X_clean      = drop_zero_variance(X_trval)
    feature_cols = list(X_clean.columns)
    imp          = SimpleImputer(strategy="median")
    X_full       = to_float32(imp.fit_transform(X_clean))

    sw = (compute_sample_weights(y_trval.to_numpy(), d_trval.to_numpy(), **weight_cfg)
          if use_dist_weights and weight_cfg is not None else None)

    spw = 1.0 if sw is not None else compute_scale_pos_weight(y_trval.to_numpy())
    xgb_params = {
        **params,
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "tree_method": "hist", "n_jobs": (os.cpu_count() or 1),
        "random_state": seed, "scale_pos_weight": spw, "max_delta_step": 1,
    }

    model = xgb.XGBClassifier(**xgb_params)
    if sw is not None:
        model.fit(X_full, y_trval.to_numpy(), sample_weight=sw, verbose=False)
    else:
        model.fit(X_full, y_trval.to_numpy(), verbose=False)

    return model, imp, feature_cols


def eval_full_model_on_test(model, imp, feature_cols, X_test, y_test):
    """Run the trained model on the test set and return (pr_auc, roc_auc, prob)."""
    X_te_imp = to_float32(imp.transform(X_test.reindex(columns=feature_cols)))
    prob     = model.predict_proba(X_te_imp)[:, 1]
    pr, roc  = _safe_auc(y_test.to_numpy(), prob)
    return pr, roc, prob


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def _get_group_cv(n_splits: int, seed: int):
    if HAVE_SGK:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return GroupKFold(n_splits=n_splits)


def evaluate_xgb_cv(
    X, y, groups, dists, params, n_splits, seed,
    fbeta=1.0, target_recall=None, use_dist_weights=False, weight_cfg=None,
) -> Dict[str, Any]:
    """
    Group-aware cross-validation for threshold selection.
    Out-of-fold predictions are pooled and used to pick a single threshold.
    """
    splitter = _get_group_cv(n_splits, seed)
    pr_aucs: List[float] = []
    roc_aucs: List[float] = []
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    n_bad = 0

    for fold_idx, (tr, va) in enumerate(splitter.split(X, y, groups)):
        X_tr = drop_zero_variance(X.iloc[tr])
        X_va = X.iloc[va].reindex(columns=X_tr.columns)
        y_tr = y.iloc[tr]
        y_va = y.iloc[va]

        sw_tr = sw_va = None
        if use_dist_weights and weight_cfg is not None:
            sw_tr = compute_sample_weights(y_tr.to_numpy(), dists.iloc[tr].to_numpy(), **weight_cfg)
            sw_va = compute_sample_weights(y_va.to_numpy(), dists.iloc[va].to_numpy(), **weight_cfg)

        try:
            _, prob, _ = fit_xgb_fold(
                X_tr, y_tr, X_va, y_va, params,
                sample_weight_tr=sw_tr, sample_weight_va=sw_va,
                early_rounds=50, seed=seed + fold_idx,
            )
        except Exception:
            n_bad += 1
            continue

        pr, roc = _safe_auc(y_va.to_numpy(), prob)
        pr_aucs.append(pr)
        roc_aucs.append(roc)
        all_y.append(y_va.to_numpy())
        all_p.append(prob)

    if not all_y:
        raise SystemExit("All CV folds failed. Check that each file has enough windows.")

    oof_y = np.concatenate(all_y)
    oof_p = np.concatenate(all_p)

    if target_recall is not None:
        thr, thr_tag = best_threshold_for_recall(oof_y, oof_p, target_recall), f"recall>={target_recall:.2f}"
    else:
        thr, thr_tag = best_threshold_by_fbeta(oof_y, oof_p, beta=fbeta), f"F{fbeta:.1f}"

    return dict(
        params=params,
        pr_auc_mean=float(np.nanmean(pr_aucs)),
        pr_auc_std=float(np.nanstd(pr_aucs)),
        roc_auc_mean=float(np.nanmean(roc_aucs)),
        roc_auc_std=float(np.nanstd(roc_aucs)),
        thr=float(thr),
        thr_tag=thr_tag,
        bad_folds=int(n_bad),
    )


# ---------------------------------------------------------------------------
# Optuna hyperparameter search
# ---------------------------------------------------------------------------

def make_pruner(name: str):
    if not name or name.lower() in ("", "none"):
        return None
    name = name.lower()
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    if name == "successivehalving":
        return optuna.pruners.SuccessiveHalvingPruner()
    return None


def xgb_wide_space(trial) -> Dict[str, Any]:
    """Search space for all events except takeoff narrow mode."""
    return dict(
        n_estimators=trial.suggest_categorical("n_estimators", [300, 400, 600]),
        learning_rate=trial.suggest_float("learning_rate", 0.03, 0.18),
        max_depth=trial.suggest_categorical("max_depth", [3, 4, 5]),
        subsample=trial.suggest_float("subsample", 0.70, 1.00),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.60, 1.00),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 0.02, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 0.05, log=True),
        min_child_weight=trial.suggest_float("min_child_weight", 1.0, 5.0),
    )


def xgb_takeoff_narrow_space(trial) -> Dict[str, Any]:
    """Narrower search space centred on the known-good takeoff baseline."""
    return dict(
        n_estimators=trial.suggest_categorical("n_estimators", [300, 400]),
        learning_rate=trial.suggest_float("learning_rate", 0.09, 0.14),
        max_depth=trial.suggest_categorical("max_depth", [4]),
        subsample=trial.suggest_float("subsample", 0.85, 0.98),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.60, 0.72),
        reg_alpha=trial.suggest_float("reg_alpha", 0.003, 0.006),
        reg_lambda=trial.suggest_float("reg_lambda", 0.001, 0.01),
        min_child_weight=trial.suggest_float("min_child_weight", 2.5, 3.5),
    )


def run_optuna(
    event_label, X_trval, y_trval, g_trval, d_trval,
    n_splits, seed, fbeta, target_recall,
    use_dist_weights, tune_dist_weights, near_s_fixed,
    enqueue_takeoff_best, narrow_takeoff_space, trials, pruner_name,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, str]:
    """Run Optuna TPE search and return (best_params, best_weight_cfg, threshold, tag)."""
    if not HAVE_OPTUNA:
        raise RuntimeError("Optuna not installed. Run: pip install optuna")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=make_pruner(pruner_name),
    )

    if event_label == "TAKEOFF" and enqueue_takeoff_best:
        study.enqueue_trial({
            **TAKEOFF_BASELINE,
            "mid_s": TAKEOFF_BASELINE_WEIGHTS["mid_s"],
            "w_mid": TAKEOFF_BASELINE_WEIGHTS["w_mid"],
            "w_far": TAKEOFF_BASELINE_WEIGHTS["w_far"],
        })

    def objective(trial) -> float:
        params = (xgb_takeoff_narrow_space(trial)
                  if event_label == "TAKEOFF" and narrow_takeoff_space
                  else xgb_wide_space(trial))

        weight_cfg = None
        if use_dist_weights:
            if tune_dist_weights:
                mid_s = trial.suggest_float("mid_s", max(near_s_fixed + 10.0, 80.0), 260.0)
                w_far = trial.suggest_float("w_far", 0.05, 0.40)
                w_mid = trial.suggest_float("w_mid", min(0.95, max(0.20, w_far + 0.05)), 1.00)
                weight_cfg = dict(near_s=near_s_fixed, mid_s=mid_s, w_near=1.0,
                                  w_mid=w_mid, w_far=w_far, pos_boost="auto")
            else:
                weight_cfg = dict(near_s=near_s_fixed, mid_s=180.0, w_near=1.0,
                                  w_mid=0.6, w_far=0.25, pos_boost="auto")

        res = evaluate_xgb_cv(
            X_trval, y_trval, g_trval, d_trval, params,
            n_splits=n_splits, seed=seed, fbeta=fbeta,
            target_recall=target_recall,
            use_dist_weights=use_dist_weights, weight_cfg=weight_cfg,
        )
        trial.set_user_attr("thr",     res["thr"])
        trial.set_user_attr("thr_tag", res["thr_tag"])
        return res["pr_auc_mean"]

    print(f"[Optuna] Starting: trials={trials}  pruner={pruner_name or 'none'}")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best        = study.best_trial
    best_params = {k: v for k, v in best.params.items()
                   if k not in ("mid_s", "w_mid", "w_far")}
    mid_s, w_mid, w_far = (
        (float(best.params.get("mid_s", 180.0)),
         float(best.params.get("w_mid", 0.6)),
         float(best.params.get("w_far", 0.25)))
        if use_dist_weights and tune_dist_weights else (180.0, 0.6, 0.25)
    )
    best_weight_cfg = dict(near_s=near_s_fixed, mid_s=mid_s, w_near=1.0,
                           w_mid=w_mid, w_far=w_far, pos_boost="auto")

    print(f"[Optuna] Best CV PR-AUC: {study.best_value:.4f}")
    return (
        best_params,
        best_weight_cfg,
        float(best.user_attrs.get("thr",     0.5)),
        str(best.user_attrs.get("thr_tag", f"F{fbeta:.1f}")),
    )


# ---------------------------------------------------------------------------
# Event-level hit / latency evaluation
# ---------------------------------------------------------------------------

def _percentile_safe(x: np.ndarray, q: float) -> Optional[float]:
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if x.size > 0 else None


def _flight_event_time(flight_df: pd.DataFrame, col: str) -> Optional[float]:
    """Return the median of a timestamp column for one flight, or None."""
    if col not in flight_df.columns:
        return None
    v = pd.to_numeric(flight_df[col], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size > 0 else None


def compute_event_hit_latency_xgb(
    df_test, prob, event_label, thr, trigger_k, hit_window_s,
    order_col, detect_time_col, label_time_col,
) -> Dict[str, Any]:
    """
    Measure event detection hit rate and latency on the test set.

    For each contiguous positive segment in a flight, scans for trigger_k
    consecutive windows >= thr and checks whether the detection time falls
    within hit_window_s of the reference event time (t_takeoff / t_landing).
    Latency = detection_time - reference_time (positive = late).
    """
    ev  = (event_label or "").upper().strip()
    df  = df_test.copy()

    if len(prob) != len(df):
        raise RuntimeError("Length mismatch between prob and df_test.")

    df["__prob"] = prob.astype(float)
    if "__row_idx" not in df.columns:
        df["__row_idx"] = np.arange(len(df), dtype=np.int64)

    df = df.sort_values(deterministic_sort_cols(df, order_col),
                        kind="mergesort").reset_index(drop=True)

    labels   = df["label"].astype(str).str.upper().to_numpy()
    y        = (labels == ev).astype(int)
    p        = pd.to_numeric(df["__prob"],        errors="coerce").to_numpy(dtype=float)
    t_detect = pd.to_numeric(df[detect_time_col], errors="coerce").to_numpy(dtype=float)
    t_label  = pd.to_numeric(df[label_time_col],  errors="coerce").to_numpy(dtype=float)

    idx_by_fid: Dict[str, List[int]] = defaultdict(list)
    for i, fid in enumerate(df["file"].astype(str).to_numpy()):
        idx_by_fid[fid].append(i)

    event_n_total = event_n_with_ref = event_triggered = event_hit = 0
    latencies: List[float] = []

    for fid, idxs in idx_by_fid.items():
        idxs = np.array(idxs, dtype=int)
        y_f  = y[idxs]
        if y_f.sum() == 0:
            continue

        # Find contiguous positive label segments.
        pos_idx = np.where(y_f == 1)[0]
        segs: List[Tuple[int, int]] = []
        s = prev = int(pos_idx[0])
        for j in pos_idx[1:]:
            j = int(j)
            if j == prev + 1:
                prev = j
            else:
                segs.append((s, prev))
                s = prev = j
        segs.append((s, prev))

        flight_df = df.iloc[idxs]
        t_ref_f   = _flight_event_time(
            flight_df, "t_takeoff" if ev == "TAKEOFF" else "t_landing"
        )

        for a, b in segs:
            event_n_total += 1

            # Use per-flight event time; fall back to label timestamp of segment start.
            t_ref = t_ref_f
            if t_ref is None or not np.isfinite(float(t_ref)):
                t0    = float(t_label[idxs[a]])
                t_ref = t0 if np.isfinite(t0) else None

            has_ref = t_ref is not None and np.isfinite(float(t_ref))
            if has_ref:
                event_n_with_ref += 1

            consec    = 0
            trig_time = None
            for rel in range(a, b + 1):
                ii = idxs[rel]
                if np.isfinite(p[ii]) and p[ii] >= float(thr):
                    consec += 1
                    if consec >= max(1, int(trigger_k)):
                        if np.isfinite(t_detect[ii]):
                            trig_time = float(t_detect[ii])
                        break
                else:
                    consec = 0

            if trig_time is None:
                continue

            event_triggered += 1

            if has_ref:
                latency = trig_time - float(t_ref)
                if np.isfinite(latency):
                    latencies.append(latency)
                if abs(latency) <= float(hit_window_s):
                    event_hit += 1

    lat = np.asarray(latencies, dtype=float)
    return {
        "event_n_total":     int(event_n_total),
        "event_n_with_ref":  int(event_n_with_ref),
        "event_triggered_n": int(event_triggered),
        "event_hit_n":       int(event_hit),
        "event_hit_rate":    (float(event_hit) / float(event_n_with_ref)
                              if event_n_with_ref > 0 else None),
        "latency_median_s":  float(np.median(lat)) if lat.size > 0 else None,
        "latency_p95_s":     _percentile_safe(lat, 95.0),
    }


# ---------------------------------------------------------------------------
# Final training and test evaluation
# ---------------------------------------------------------------------------

def train_full_and_eval_test_fullfit(
    X_trval, y_trval, d_trval, X_test, y_test,
    best_params, best_weight_cfg, use_dist_weights, chosen_thr, seed, event_label,
) -> Tuple[Dict[str, Any], Any, SimpleImputer, List[str], np.ndarray]:
    """Train on all train+val and evaluate on test. Returns test probabilities."""
    print(f"[XGB][{event_label}] Full-fit training "
          f"({len(X_trval)} samples, {int(y_trval.sum())} positives)...")

    model, imp, feature_cols = fit_xgb_full_no_early(
        X_trval, y_trval, d_trval, best_params, best_weight_cfg,
        use_dist_weights=use_dist_weights, seed=seed,
    )
    pr_auc, roc_auc, prob = eval_full_model_on_test(model, imp, feature_cols, X_test, y_test)

    y_pred = (prob >= chosen_thr).astype(int)
    cm     = confusion_matrix(y_test, y_pred, labels=[1, 0]).tolist()
    TP, FN, FP, TN = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    print(f"[XGB][{event_label}] @{chosen_thr:.4f}  TP={TP}  FN={FN}  FP={FP}  TN={TN}")
    print(f"[XGB][{event_label}] PR-AUC={pr_auc:.3f}  ROC-AUC={roc_auc:.3f}")

    return (
        dict(
            pr_auc=float(pr_auc)  if np.isfinite(pr_auc)  else float("nan"),
            roc_auc=float(roc_auc) if np.isfinite(roc_auc) else float("nan"),
            threshold_used=float(chosen_thr),
        ),
        model, imp, feature_cols, prob,
    )


# ---------------------------------------------------------------------------
# Per-seed pipeline
# ---------------------------------------------------------------------------

def run_pipeline_for_seed(
    df_all, event_label, seed, args,
    fixed_model_params, fixed_wcfg,
    order_col, detect_time_col, label_time_col,
) -> Optional[Dict[str, Any]]:
    """Run train / threshold-select / test for one event and seed."""
    if not HAVE_XGB:
        raise SystemExit("xgboost is not installed. Run: pip install xgboost")

    event_label = (event_label or "").upper().strip()
    print(f"\n{'=' * 55}\nEVENT: {event_label}  SEED: {seed}\n{'=' * 55}")

    X_all, y_all, g_all, d_all = make_xy(df_all, event_label, args.dist_source)
    is_nonflight = identify_nonflight_windows(df_all, args.nonflight_label)

    files_all       = g_all.to_numpy()
    is_nf_arr       = is_nonflight.to_numpy()
    flight_files    = sorted(set(files_all[~is_nf_arr]))
    nonflight_files = sorted(set(files_all[ is_nf_arr]))

    # Compute durations for flight files only. Nonflight files are absent from
    # the map and get duration 0.0 when looked up, so they sort purely
    # alphabetically — identical to GRU behaviour with --use_file_id_as_stream.
    dur_map = group_hours_from_df(df_all[~is_nonflight], "file", detect_time_col)

    train_files, val_files, test_files = split_groups_deterministic(
        flight_groups=flight_files,
        nonflight_groups=nonflight_files,
        dur_map=dur_map,
        seed=int(seed),
        flight_val_frac=float(args.flight_val_frac),
        flight_test_frac=float(args.flight_test_frac),
        nf_val_frac=float(args.nf_val_frac),
        nf_test_frac=float(args.nf_test_frac),
        min_nf_val_groups=int(args.min_nf_val_groups),
        min_nf_val_hours=float(args.min_nf_val_hours),
        min_nf_test_groups=int(args.min_nf_test_groups),
    )

    trval_files  = train_files | val_files
    is_test_row  = np.array([f in test_files  for f in files_all])
    is_trval_row = np.array([f in trval_files for f in files_all])
    trval_idx    = np.where(is_trval_row)[0]
    test_idx     = np.where(is_test_row)[0]

    X_trval    = X_all.iloc[trval_idx]
    y_trval    = y_all.iloc[trval_idx]
    g_trval    = g_all.iloc[trval_idx]
    d_trval    = d_all.iloc[trval_idx]
    X_test     = X_all.iloc[test_idx]
    y_test     = y_all.iloc[test_idx]
    is_nf_test = is_nonflight.iloc[test_idx]

    # No file should appear in both train+val and test.
    assert not (set(pd.unique(g_trval)) & set(pd.unique(g_all.iloc[test_idx]))), \
        "LEAKAGE: files appear in both train+val and test"

    _check_split(y_trval, y_test, event_label)

    print(f"[Data] train+val={len(X_trval)} ({int(y_trval.sum())} pos)  "
          f"test={len(X_test)} ({int(y_test.sum())} pos)  "
          f"nonflight_test={int(is_nf_test.sum())}")

    # Select threshold via CV or use a fixed one.
    cv_res = None
    if (not args.skip_tuning) and fixed_model_params is None:
        if not HAVE_OPTUNA:
            raise RuntimeError("Optuna required. Use --skip_tuning with --fixed_model_params.")
        best_params, best_wcfg, thr, thr_tag = run_optuna(
            event_label, X_trval, y_trval, g_trval, d_trval,
            args.cv_splits, seed, args.fbeta, args.target_recall,
            args.use_dist_weights, args.tune_dist_weights, args.near_s,
            args.enqueue_takeoff_best, args.narrow_takeoff_space,
            args.optuna_trials, args.optuna_pruner,
        )
    else:
        if fixed_model_params is None:
            raise RuntimeError("--skip_tuning requires --fixed_model_params.")
        best_params = fixed_model_params
        best_wcfg   = None
        if args.use_dist_weights:
            best_wcfg = fixed_wcfg or dict(near_s=args.near_s, mid_s=180.0, w_near=1.0,
                                           w_mid=0.6, w_far=0.25, pos_boost="auto")
            best_wcfg.setdefault("pos_boost", "auto")
        if args.fixed_threshold is not None:
            thr, thr_tag = float(args.fixed_threshold), "fixed"
        else:
            cv_res = evaluate_xgb_cv(
                X_trval, y_trval, g_trval, d_trval, best_params,
                n_splits=args.cv_splits, seed=seed, fbeta=args.fbeta,
                target_recall=args.target_recall,
                use_dist_weights=args.use_dist_weights, weight_cfg=best_wcfg,
            )
            thr, thr_tag = cv_res["thr"], cv_res["thr_tag"]

    if cv_res is not None:
        if cv_res.get("bad_folds", 0):
            print(f"[CV] {cv_res['bad_folds']} degenerate folds skipped.")
        print(f"[CV] PR-AUC: mean={cv_res['pr_auc_mean']:.3f}  std={cv_res['pr_auc_std']:.3f}")

    print(f"[Threshold] {thr:.4f}  ({thr_tag})")

    test_report, model, imp, feature_cols, prob = train_full_and_eval_test_fullfit(
        X_trval, y_trval, d_trval, X_test, y_test,
        best_params=best_params, best_weight_cfg=best_wcfg,
        use_dist_weights=args.use_dist_weights,
        chosen_thr=thr, seed=seed, event_label=event_label,
    )

    df_test = df_all.iloc[test_idx].copy()
    if "__row_idx" not in df_test.columns:
        df_test["__row_idx"] = np.arange(len(df_test), dtype=np.int64)

    df_test_flight    = df_test[~is_nf_test]
    df_test_nonflight = df_test[ is_nf_test]
    prob_flight       = prob[~is_nf_test.to_numpy()]
    prob_nonflight    = prob[ is_nf_test.to_numpy()]

    ev_metrics = compute_event_hit_latency_xgb(
        df_test=df_test_flight, prob=prob_flight,
        event_label=event_label, thr=float(thr),
        trigger_k=int(args.trigger_k), hit_window_s=float(args.hit_window_s),
        order_col=order_col, detect_time_col=detect_time_col,
        label_time_col=label_time_col,
    )

    nf_fp_metrics: Dict[str, Any] = {}
    if len(df_test_nonflight) > 0:
        nf_fp_metrics = count_nonflight_triggers(
            df_nf=df_test_nonflight, prob=prob_nonflight,
            thr=float(thr), trigger_k=int(args.trigger_k),
            cooldown_s=float(args.cooldown_s), detect_time_col=detect_time_col,
        )
        print(f"\n[Nonflight FP/h] ({event_label})  "
              f"fp_per_hour={nf_fp_metrics.get('fp_per_hour')}  "
              f"triggers={nf_fp_metrics.get('triggers')}  "
              f"hours={nf_fp_metrics.get('hours', 0.0):.3f}")

    out: Dict[str, Any] = {}
    out.update(test_report)
    out.update(ev_metrics)
    out.update(dict(
        model_type="xgb",
        event=event_label,
        seed=int(seed),
        n_train=int(len(X_trval)),
        n_test=int(len(X_test)),
        n_test_flight=int((~is_nf_test).sum()),
        n_test_nonflight=int(is_nf_test.sum()),
        n_pos_train=int(y_trval.sum()),
        n_pos_test=int(y_test.sum()),
        thr_tag=str(thr_tag),
        trigger_k=int(args.trigger_k),
        cooldown_s=float(args.cooldown_s),
        hit_window_s=float(args.hit_window_s),
        timing_columns=dict(
            order_col=order_col,
            detect_time_col=detect_time_col,
            label_time_col=label_time_col,
        ),
        nonflight_fp_test=nf_fp_metrics,
    ))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "XGBoost baseline for flight event detection. "
            "Uses the same deterministic train/val/test split as the GRU script "
            "so that all three models are compared on the same test flights."
        )
    )

    # Input
    ap.add_argument("--features", required=True,
                    help="Path to features_imputed.csv from the preprocessor.")
    ap.add_argument("--events", nargs="+", default=["TAKEOFF", "LANDING"])
    ap.add_argument("--cv_splits", type=int, default=5)

    # Seeds
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Run for multiple seeds. Overrides --seed.")

    # Split fractions
    ap.add_argument("--flight_val_frac",    type=float, default=0.2)
    ap.add_argument("--flight_test_frac",   type=float, default=0.3)
    ap.add_argument("--nf_val_frac",        type=float, default=0.5)
    ap.add_argument("--nf_test_frac",       type=float, default=0.2)
    ap.add_argument("--min_nf_val_groups",  type=int,   default=2)
    ap.add_argument("--min_nf_val_hours",   type=float, default=2.0)
    ap.add_argument("--min_nf_test_groups", type=int,   default=1)

    # Threshold selection
    ap.add_argument("--fbeta",         type=float, default=2.0)
    ap.add_argument("--target_recall", type=float, default=None)

    # Distance weighting
    ap.add_argument("--use_dist_weights",  action="store_true")
    ap.add_argument("--tune_dist_weights", action="store_true")
    ap.add_argument("--near_s",      type=float, default=60.0)
    ap.add_argument("--dist_source", default="auto", choices=["auto", "event", "min"])

    # Evaluation
    ap.add_argument("--trigger_k",       type=int,   default=1)
    ap.add_argument("--cooldown_s",      type=float, default=60.0)
    ap.add_argument("--hit_window_s",    type=float, default=10.0)
    ap.add_argument("--nonflight_label", type=str,   default="NONFLIGHT")

    # Optuna
    ap.add_argument("--optuna_trials",        type=int,  default=60)
    ap.add_argument("--optuna_pruner",        type=str,  default="median",
                    choices=["none", "median", "hyperband", "successivehalving"])
    ap.add_argument("--enqueue_takeoff_best", action="store_true")
    ap.add_argument("--narrow_takeoff_space", action="store_true")

    # Fixed params (skip Optuna)
    ap.add_argument("--skip_tuning",        action="store_true")
    ap.add_argument("--fixed_model_params", type=str,   default=None,
                    help="JSON dict of XGBoost parameters.")
    ap.add_argument("--fixed_wcfg",         type=str,   default=None,
                    help="JSON dict of distance-weight config.")
    ap.add_argument("--fixed_threshold",    type=float, default=None)

    # Output
    ap.add_argument("--metrics_out", type=str, default=None,
                    help="Path to write metrics as a JSON file.")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore")

    if not HAVE_XGB:
        raise SystemExit("xgboost is not installed. Run: pip install xgboost")

    df_all = pd.read_csv(args.features)
    if "__row_idx" not in df_all.columns:
        df_all["__row_idx"] = np.arange(len(df_all), dtype=np.int64)

    order_col, detect_time_col, label_time_col = choose_time_columns(df_all)
    print(f"[TimingCols] order={order_col}  detect={detect_time_col}  label={label_time_col}")

    seeds              = args.seeds if args.seeds is not None else [args.seed]
    fixed_model_params = json.loads(args.fixed_model_params) if args.fixed_model_params else None
    fixed_wcfg         = json.loads(args.fixed_wcfg)         if args.fixed_wcfg         else None

    all_results: List[Dict[str, Any]] = []

    for ev in args.events:
        seed_results: List[Dict[str, Any]] = []
        for sd in seeds:
            res = run_pipeline_for_seed(
                df_all, ev, sd, args, fixed_model_params, fixed_wcfg,
                order_col=order_col,
                detect_time_col=detect_time_col,
                label_time_col=label_time_col,
            )
            if res is not None:
                seed_results.append(res)
                all_results.append(res)

        if len(seed_results) > 1:
            pr_vals  = np.array([r["pr_auc"]  for r in seed_results], dtype=float)
            roc_vals = np.array([r["roc_auc"] for r in seed_results], dtype=float)
            print(f"\n[{ev}] Seed summary across {len(seed_results)} seeds:")
            print(f"  PR-AUC  mean={np.nanmean(pr_vals):.3f}  std={np.nanstd(pr_vals):.3f}")
            print(f"  ROC-AUC mean={np.nanmean(roc_vals):.3f}  std={np.nanstd(roc_vals):.3f}")
            for r in seed_results:
                nf = r.get("nonflight_fp_test", {})
                print(f"  seed={r['seed']}  PR-AUC={r['pr_auc']:.3f}  "
                      f"thr={r['threshold_used']:.4f}  "
                      f"hit_rate={r.get('event_hit_rate')}  "
                      f"nf_fp_h={nf.get('fp_per_hour')}")

    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_strict(out_path, all_results)
        print(f"\n[Metrics] Saved {len(all_results)} records to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
