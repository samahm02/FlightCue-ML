"""
Random Forest baseline detector for TAKEOFF and LANDING events.

This script is a baseline comparison against the GRU model. It is NOT deployed
to the Android app.

How it works
------------
1. Load features_imputed.csv from the preprocessor.
2. Split by physical file using the SAME deterministic algorithm as the GRU
   and XGBoost scripts, so all three models are evaluated on identical test
   flights when the same seed is used.
3. Run a random hyperparameter search using group-aware cross-validation
   to find the best RF configuration (Stage A).
4. Optionally search over distance-weight configurations as well (Stage B).
5. Collect out-of-fold predictions from the best configuration and use
   them to select a threshold via F-beta.
6. Train a final model on ALL of train+val and evaluate on the test set.
7. Report event-level hit rate and latency using the same trigger logic
   as the GRU and XGBoost evaluations.

The final model is NOT exported. Per-run metrics are written to disk.

TAKEOFF and LANDING use different preprocessed datasets (25s and 20s base
windows respectively) and must be run as separate commands.

Usage -- takeoff
----------------
    python3 rf.py \
        --features all_data/preprocessed_25/features_raw.csv \
        --out_dir rf_runs/takeoff/ \
        --events TAKEOFF \
        --balancing undersample --neg_ratio 8 \
        --n_candidates 40 --beta 2.0 \
        --trigger_k 1 --hit_window_s 90.0 \
        --seed 42 --metrics_out rf_takeoff_metrics.json

Usage -- landing
----------------
    python3 rf.py \
        --features all_data/preprocessed_20/features_raw.csv \
        --out_dir rf_runs/landing/ \
        --events LANDING \
        --balancing undersample --neg_ratio 8 \
        --n_candidates 40 --beta 2.0 \
        --trigger_k 1 --hit_window_s 90.0 \
        --seed 42 --metrics_out rf_landing_metrics.json
"""

import argparse
import os
import json
import pprint
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAVE_SGK = True
except Exception:
    HAVE_SGK = False


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
}

META_PREFIXES = ("win_", "_hnm_")

BALANCING_CHOICES = ["undersample", "class_weight", "dist_weight", "none"]


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
    all three scripts, which is required for a fair three-way comparison.

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
        # group over — deterministic tiebreaker, identical to GRU and XGBoost.
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
# Data preparation
# ---------------------------------------------------------------------------

def _ensure_file_col(df: pd.DataFrame) -> pd.DataFrame:
    """Add a synthetic 'file' column if the CSV does not contain one."""
    if "file" in df.columns:
        return df
    df = df.copy()
    df["file"] = np.arange(len(df)).astype(str)
    return df


def _pick_distance(df: pd.DataFrame, event_label: str) -> pd.Series:
    """
    Return the absolute distance-to-event series for sample weighting.
    Prefers dist_to_event, then the event-specific column, then falls back
    to a constant large value meaning 'far from any event'.
    """
    n   = len(df)
    far = pd.Series(np.full(n, 1e9), index=df.index, dtype=float)

    if "dist_to_event" in df.columns:
        return pd.to_numeric(df["dist_to_event"], errors="coerce").astype(float).abs().fillna(1e9)

    ev  = str(event_label).upper().strip()
    col = "dist_to_takeoff" if ev == "TAKEOFF" else "dist_to_landing"
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").astype(float).abs().fillna(1e9)

    return far


def make_xy(
    df: pd.DataFrame, event_label: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Build (X, y, groups, distances) for one event type.

    X      : numeric feature DataFrame with meta columns removed.
    y      : binary label (1 = event, 0 = other).
    groups : file name per row, used for group-aware CV splitting.
    dists  : absolute distance to the event, used for distance weighting.
    """
    if "label" not in df.columns:
        raise SystemExit("Missing required column: label")

    df = _ensure_file_col(df)
    X  = df.select_dtypes(include=[np.number]).copy()
    X  = X.drop(columns=[c for c in X.columns if c in META_COLS], errors="ignore")
    for p in META_PREFIXES:
        X = X.drop(columns=[c for c in X.columns if c.startswith(p)], errors="ignore")

    if X.shape[1] == 0:
        raise SystemExit("No numeric feature columns left after removing meta columns.")

    ev = str(event_label).upper().strip()
    return (
        X,
        (df["label"].astype(str).str.upper() == ev).astype(int),
        df["file"].astype(str),
        _pick_distance(df, ev),
    )


def undersample_train_indices(
    y_idx: np.ndarray, y: pd.Series, neg_ratio: int, rng: np.random.Generator
) -> np.ndarray:
    """
    Keep approximately neg_ratio * n_positives negatives from the training set.
    Positives are always kept; negatives are randomly subsampled.
    """
    if not neg_ratio or neg_ratio <= 0:
        return y_idx
    y_arr   = y.iloc[y_idx].to_numpy()
    pos_idx = y_idx[y_arr == 1]
    neg_idx = y_idx[y_arr == 0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return y_idx
    k        = min(len(neg_idx), int(neg_ratio) * len(pos_idx))
    keep_neg = rng.choice(neg_idx, size=k, replace=False)
    keep     = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep)
    return keep


def compute_sample_weights(
    y,
    dist,
    near_s: float = 60.0,
    mid_s: float = 180.0,
    w_mid: float = 0.6,
    w_far: float = 0.25,
) -> np.ndarray:
    """
    Assign sample weights based on proximity to an event.
    Negatives are down-weighted the further they are from the event.
    Positives are up-weighted so their total mass equals the negatives'.
    All distances are treated as absolute values.
    """
    y = np.asarray(y, dtype=int)
    d = np.where(np.isfinite(np.abs(np.asarray(dist, dtype=float))),
                 np.abs(np.asarray(dist, dtype=float)), 1e9)
    w = np.where(d <= float(near_s), 1.0,
                 np.where(d <= float(mid_s), float(w_mid), float(w_far))).astype(float)
    pos     = (y == 1)
    pos_cnt = max(1, int(pos.sum()))
    w[pos]  = w[~pos].sum() / pos_cnt
    return np.maximum(w, 1e-12)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def fbeta_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float, beta: float
) -> float:
    yp   = (y_prob >= thr).astype(int)
    tp   = int(((yp == 1) & (y_true == 1)).sum())
    fp   = int(((yp == 1) & (y_true == 0)).sum())
    fn   = int(((yp == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    b2   = beta * beta
    denom = b2 * prec + rec
    return (1 + b2) * prec * rec / denom if denom > 0 else 0.0


def choose_threshold_by_fbeta(
    y_true: np.ndarray, y_prob: np.ndarray, beta: float
) -> Dict[str, float]:
    """Scan every PR-curve threshold and return the one that maximises F-beta."""
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {"threshold": 0.5, "fbeta": 0.0}
    _, _, thr_grid = precision_recall_curve(y_true, y_prob)
    thr_grid  = np.concatenate(([0.0], thr_grid, [1.0]))
    best_thr  = 0.5
    best_f    = -1.0
    for t in thr_grid:
        f = fbeta_at_threshold(y_true, y_prob, float(t), beta)
        if f > best_f:
            best_f, best_thr = f, float(t)
    return {"threshold": float(best_thr), "fbeta": float(best_f)}


def sanity_checks_split(
    files_trval: List[str], files_test: List[str],
    y_trval: pd.Series, y_test: pd.Series,
) -> None:
    """Raise if there is file leakage or if either split has no positives."""
    inter = set(files_trval) & set(files_test)
    if inter:
        raise SystemExit(f"Leakage: files in both train/val and test: {sorted(inter)[:5]}")
    if int(y_trval.sum()) == 0:
        raise SystemExit("No positives in train+val. Try a different --seed.")
    if int(y_test.sum()) == 0:
        raise SystemExit("No positives in test. Try a different --seed.")
    if len(np.unique(y_trval.to_numpy())) < 2:
        raise SystemExit("Train+val has only one class.")
    if len(np.unique(y_test.to_numpy())) < 2:
        raise SystemExit("Test set has only one class.")


def get_group_cv_splits(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int, seed: int
):
    """Return a CV splitter that keeps all windows from the same file together."""
    if HAVE_SGK:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X, y, groups)
    return GroupKFold(n_splits=n_splits).split(X, y, groups)


def safe_make_rf(params: Dict[str, Any]) -> RandomForestClassifier:
    """
    Instantiate a RandomForestClassifier.
    Falls back to 'gini' criterion if the sklearn version does not support 'log_loss'.
    """
    try:
        return RandomForestClassifier(**params)
    except TypeError:
        p = params.copy()
        if p.get("criterion") == "log_loss":
            p["criterion"] = "gini"
        return RandomForestClassifier(**p)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Compute PR-AUC and ROC-AUC, returning NaN on failure."""
    pr = roc = float("nan")
    try:
        pr = float(average_precision_score(y_true, y_prob))
    except Exception:
        pass
    try:
        if len(np.unique(y_true)) >= 2:
            roc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        pass
    return pr, roc


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def sample_param_candidates(
    n: int, rng: np.random.Generator, balancing: str
) -> List[Dict[str, Any]]:
    """
    Draw n random RF hyperparameter configurations.
    class_weight is only searched when balancing='class_weight'.
    """
    cands: List[Dict[str, Any]] = []
    for _ in range(n):
        params: Dict[str, Any] = dict(
            n_estimators=int(rng.choice([200, 300, 400, 600, 800, 1000])),
            max_depth=rng.choice([None, 10, 16, 20, 30]),
            min_samples_split=int(rng.choice([2, 4, 8])),
            min_samples_leaf=int(rng.choice([1, 2, 3, 5])),
            max_features=rng.choice(["sqrt", "log2", 0.5, None]),
            criterion=rng.choice(["gini", "entropy", "log_loss"]),
            bootstrap=bool(rng.integers(0, 2)),
            random_state=int(rng.integers(0, 2**31 - 1)),
            n_jobs=-1,
        )
        if params["bootstrap"] and rng.random() < 0.85:
            params["max_samples"] = rng.choice([0.5, 0.65, 0.8, None])
        params["class_weight"] = (
            rng.choice([None, "balanced", "balanced_subsample"])
            if balancing == "class_weight"
            else None
        )
        cands.append(params)
    return cands


def sample_weight_configs(
    n: int,
    rng: np.random.Generator,
    near_base: float,
    mid_base: float,
    w_mid_base: float,
    w_far_base: float,
) -> List[Dict[str, Any]]:
    """
    Generate n distance-weight configurations by sampling around a baseline.
    The baseline is always included as the first entry.
    """
    base = dict(near_s=float(near_base), mid_s=float(mid_base),
                w_mid=float(w_mid_base), w_far=float(w_far_base))
    if n <= 1:
        return [base]
    cfgs = [base]
    for _ in range(n - 1):
        near_s = float(np.clip(rng.normal(near_base, max(1e-6, near_base * 0.05)),
                               5.0, 2.0 * near_base))
        mid_s  = float(np.clip(rng.normal(mid_base,  max(1e-6, mid_base  * 0.06)),
                               near_s + 10.0, 2.0 * mid_base))
        w_mid  = float(np.clip(rng.normal(w_mid_base, 0.10), 0.15, 0.95))
        w_far  = float(np.clip(rng.normal(w_far_base, 0.10), 0.05, 0.95))
        cfgs.append(dict(near_s=near_s, mid_s=mid_s, w_mid=w_mid, w_far=w_far))
    return cfgs


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def evaluate_candidate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    params: Dict[str, Any],
    n_splits: int,
    seed: int,
    balancing: str,
    neg_ratio: int,
    dists: Optional[pd.Series] = None,
    dw_cfg: Optional[Dict[str, float]] = None,
    collect_oof: bool = False,
) -> Dict[str, Any]:
    """
    Group-aware cross-validation for one RF hyperparameter configuration.

    The imputer is fit only on the training fold to prevent leakage.
    Out-of-fold probabilities are collected when collect_oof=True, which is
    used later to pick a threshold on the full train+val set.

    Balancing modes:
        undersample  : down-sample negatives to neg_ratio * n_positives.
        class_weight : pass class_weight from params to the RF.
        dist_weight  : weight samples by proximity to the event.
        none         : no balancing.
    """
    splits = list(get_group_cv_splits(X, y, groups, n_splits, seed))
    rng    = np.random.default_rng(seed)

    pr_aucs:  List[float] = []
    roc_aucs: List[float] = []
    oof_prob = np.full(y.shape[0], np.nan, dtype=float)
    oof_true = np.full(y.shape[0], np.nan, dtype=float)

    for fold_idx, (tr, va) in enumerate(splits):
        imp  = SimpleImputer(strategy="median")
        X_tr = X.iloc[tr]
        X_va = X.iloc[va]
        y_tr = y.iloc[tr].reset_index(drop=True)
        y_va = y.iloc[va].reset_index(drop=True)

        # Skip folds where train or val has only one class.
        if len(np.unique(y_tr.to_numpy())) < 2 or len(np.unique(y_va.to_numpy())) < 2:
            continue

        # Imputer fit on train only.
        X_tr_imp = imp.fit_transform(X_tr)
        X_va_imp = imp.transform(X_va)

        sw         = None
        params_fit = params.copy()

        if balancing == "dist_weight":
            if dists is None or dw_cfg is None:
                raise ValueError("dist_weight requires dists and dw_cfg.")
            sw = compute_sample_weights(y_tr.to_numpy(), dists.iloc[tr].to_numpy(),
                                        near_s=dw_cfg["near_s"], mid_s=dw_cfg["mid_s"],
                                        w_mid=dw_cfg["w_mid"], w_far=dw_cfg["w_far"])
            params_fit.pop("class_weight", None)
            X_tr_use, y_tr_use = X_tr_imp, y_tr

        elif balancing == "undersample":
            keep = undersample_train_indices(np.arange(len(y_tr)), y_tr, neg_ratio, rng)
            X_tr_use, y_tr_use = X_tr_imp[keep], y_tr.iloc[keep]
            params_fit.pop("class_weight", None)

        elif balancing == "class_weight":
            X_tr_use, y_tr_use = X_tr_imp, y_tr

        elif balancing == "none":
            X_tr_use, y_tr_use = X_tr_imp, y_tr
            params_fit.pop("class_weight", None)

        else:
            raise ValueError(f"Unknown balancing mode: {balancing}")

        clf = safe_make_rf(params_fit)
        clf.fit(X_tr_use, y_tr_use, **({"sample_weight": sw} if sw is not None else {}))

        yp = clf.predict_proba(X_va_imp)[:, 1].astype(float)
        pr, roc = _safe_auc(y_va.to_numpy(dtype=int), yp)
        pr_aucs.append(pr)
        roc_aucs.append(roc)

        if collect_oof:
            oof_prob[va] = yp
            oof_true[va] = y_va.to_numpy(dtype=int)

    out = dict(
        params=params,
        pr_auc_mean=float(np.nanmean(pr_aucs))  if pr_aucs else float("nan"),
        pr_auc_std=float(np.nanstd(pr_aucs))   if pr_aucs else float("nan"),
        roc_auc_mean=float(np.nanmean(roc_aucs)) if roc_aucs else float("nan"),
        roc_auc_std=float(np.nanstd(roc_aucs))  if roc_aucs else float("nan"),
    )
    if collect_oof:
        out["oof_prob"] = oof_prob
        out["oof_true"] = oof_true
    return out


def train_full_and_eval_test(
    X_trval: pd.DataFrame,
    y_trval: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_params: Dict[str, Any],
    balancing: str,
    neg_ratio: int,
    seed: int,
    beta: float,
    threshold: float,
    dists_trval: Optional[pd.Series] = None,
    dw_cfg: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Train the final RF on all train+val and evaluate on test.
    The imputer is fit on train+val only; test is transformed using those stats.
    Returns the test probabilities in 'yprob_test' for event-level metrics.
    """
    rng = np.random.default_rng(seed)
    imp = SimpleImputer(strategy="median")

    X_trval_imp = imp.fit_transform(X_trval)
    X_test_imp  = imp.transform(X_test)

    sw         = None
    params_fit = best_params.copy()

    if balancing == "dist_weight":
        if dists_trval is None or dw_cfg is None:
            raise ValueError("dist_weight requires dists_trval and dw_cfg.")
        sw = compute_sample_weights(y_trval.to_numpy(), dists_trval.to_numpy(),
                                    near_s=dw_cfg["near_s"], mid_s=dw_cfg["mid_s"],
                                    w_mid=dw_cfg["w_mid"], w_far=dw_cfg["w_far"])
        params_fit.pop("class_weight", None)
        X_tr_use, y_tr_use = X_trval_imp, y_trval

    elif balancing == "undersample":
        keep = undersample_train_indices(np.arange(len(y_trval)), y_trval, neg_ratio, rng)
        X_tr_use, y_tr_use = X_trval_imp[keep], y_trval.iloc[keep]
        params_fit.pop("class_weight", None)

    elif balancing == "class_weight":
        X_tr_use, y_tr_use = X_trval_imp, y_trval

    elif balancing == "none":
        X_tr_use, y_tr_use = X_trval_imp, y_trval
        params_fit.pop("class_weight", None)

    else:
        raise ValueError(f"Unknown balancing mode: {balancing}")

    clf = safe_make_rf(params_fit)
    clf.fit(X_tr_use, y_tr_use, **({"sample_weight": sw} if sw is not None else {}))

    yprob_test = clf.predict_proba(X_test_imp)[:, 1].astype(float)
    pr_auc, roc_auc = _safe_auc(y_test.to_numpy(dtype=int), yprob_test)
    thr   = float(threshold)
    fb    = float(fbeta_at_threshold(y_test.to_numpy(dtype=int), yprob_test, thr, beta))
    cm    = confusion_matrix(y_test, (yprob_test >= thr).astype(int), labels=[1, 0]).tolist()

    return dict(
        threshold=thr,
        fbeta=fb,
        pr_auc=float(pr_auc)  if np.isfinite(pr_auc)  else float("nan"),
        roc_auc=float(roc_auc) if np.isfinite(roc_auc) else float("nan"),
        confusion_matrix_labels=[1, 0],
        confusion_matrix_cm=cm,
        yprob_test=yprob_test,  # kept in memory for event-level metrics; not written to disk
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


def compute_event_hit_latency_window_model(
    df_test: pd.DataFrame,
    prob: np.ndarray,
    event_label: str,
    thr: float,
    trigger_k: int,
    hit_window_s: float,
    order_col: str,
    detect_time_col: str,
    label_time_col: str,
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
        raise RuntimeError("Length mismatch between prob array and df_test.")

    df["__prob"] = prob.astype(float)
    if "file" not in df.columns:
        df["file"] = np.arange(len(df)).astype(str)
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
        idxs  = np.array(idxs, dtype=int)
        y_f   = y[idxs]
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
# Main pipeline for one (event, seed) combination
# ---------------------------------------------------------------------------

def run_event_pipeline(
    df_all: pd.DataFrame,
    event_label: str,
    n_splits: int,
    n_candidates: int,
    balancing: str,
    neg_ratio: int,
    seed: int,
    beta: float,
    out_dir: str,
    near_s: float,
    mid_s: float,
    w_mid: float,
    w_far: float,
    search_weight_params: bool,
    n_weight_candidates: int,
    trigger_k: int,
    hit_window_s: float,
    order_col: str,
    detect_time_col: str,
    label_time_col: str,
    nonflight_label: str,
    flight_val_frac: float,
    flight_test_frac: float,
    nf_val_frac: float,
    nf_test_frac: float,
    min_nf_val_groups: int,
    min_nf_val_hours: float,
    min_nf_test_groups: int,
) -> Dict[str, Any]:
    """
    Run the full RF pipeline for one event type and seed.

    Stage A: random hyperparameter search over n_candidates configs using CV.
    Stage B: optional search over distance-weight configs (dist_weight only).
    Final:   train on all train+val, evaluate on test, compute event metrics.

    Per-run output files are written to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    X_all, y_all, g_all, d_all = make_xy(df_all, event_label)
    is_nonflight = identify_nonflight_windows(df_all, nonflight_label)

    files_all       = g_all.to_numpy()
    is_nf_arr       = is_nonflight.to_numpy()
    flight_files    = sorted(set(files_all[~is_nf_arr]))
    nonflight_files = sorted(set(files_all[ is_nf_arr]))

    # Compute durations for flight files only. Nonflight files are absent from
    # the map and get duration 0.0 when looked up, so they sort purely
    # alphabetically — identical to GRU and XGBoost behaviour.
    dur_map = group_hours_from_df(df_all[~is_nonflight], "file", detect_time_col)

    train_files, val_files, test_files = split_groups_deterministic(
        flight_groups=flight_files,
        nonflight_groups=nonflight_files,
        dur_map=dur_map,
        seed=int(seed),
        flight_val_frac=float(flight_val_frac),
        flight_test_frac=float(flight_test_frac),
        nf_val_frac=float(nf_val_frac),
        nf_test_frac=float(nf_test_frac),
        min_nf_val_groups=int(min_nf_val_groups),
        min_nf_val_hours=float(min_nf_val_hours),
        min_nf_test_groups=int(min_nf_test_groups),
    )

    trval_files  = train_files | val_files
    is_test_row  = np.array([f in test_files  for f in files_all])
    is_trval_row = np.array([f in trval_files for f in files_all])
    trval_idx    = np.where(is_trval_row)[0]
    test_idx     = np.where(is_test_row)[0]

    X_trval      = X_all.iloc[trval_idx]
    y_trval      = y_all.iloc[trval_idx]
    groups_trval = g_all.iloc[trval_idx]
    dists_trval  = d_all.iloc[trval_idx]
    X_test       = X_all.iloc[test_idx]
    y_test       = y_all.iloc[test_idx]

    files_trval = sorted(pd.unique(groups_trval))
    files_test  = sorted(pd.unique(g_all.iloc[test_idx]))

    # Save the split for reproducibility.
    with open(os.path.join(out_dir, f"{event_label}_splits.json"), "w", encoding="utf-8") as f:
        json.dump({"trainval_files": files_trval, "test_files": files_test},
                  f, ensure_ascii=False, indent=2)

    sanity_checks_split(files_trval, files_test, y_trval, y_test)

    rng         = np.random.default_rng(seed)
    candidates  = sample_param_candidates(n_candidates, rng, balancing)
    baseline_dw = dict(near_s=near_s, mid_s=mid_s, w_mid=w_mid, w_far=w_far)

    print(f"\n{'=' * 55}")
    print(f"RF  EVENT: {event_label}  SEED: {seed}")
    print(f"{'=' * 55}")
    print(f"[Data] train+val={len(X_trval)} ({int(y_trval.sum())} pos)  "
          f"test={len(X_test)} ({int(y_test.sum())} pos)")
    cv_type = "StratifiedGroupKFold" if HAVE_SGK else "GroupKFold"
    print(f"[CV] {cv_type}(n_splits={n_splits})")

    # Stage A: random hyperparameter search.
    rows: List[Dict[str, Any]] = []
    for i, params in enumerate(candidates, 1):
        res = evaluate_candidate_cv(
            X_trval, y_trval, groups_trval,
            params=params, n_splits=n_splits, seed=seed + i,
            balancing=balancing, neg_ratio=neg_ratio,
            dists=dists_trval if balancing == "dist_weight" else None,
            dw_cfg=baseline_dw if balancing == "dist_weight" else None,
            collect_oof=False,
        )
        rows.append({"candidate": i, **res})
        print(f"[StageA] {i}/{n_candidates}  PR-AUC={res['pr_auc_mean']:.3f}  "
              f"ROC-AUC={res['roc_auc_mean']:.3f}")

    # Save candidate results and pick the best.
    df_cand   = pd.DataFrame(rows)
    params_df = pd.json_normalize(df_cand["params"]).set_index(df_cand.index)
    df_out    = pd.concat([df_cand.drop(columns=["params"]), params_df], axis=1)
    cand_csv  = os.path.join(out_dir, f"{event_label}_cv_candidates.csv")
    df_out.to_csv(cand_csv, index=False)
    print(f"Wrote: {cand_csv}")

    df_sel        = df_out.sort_values(["pr_auc_mean", "roc_auc_mean"],
                                        ascending=[False, False], kind="mergesort")
    best_cand_idx = int(df_sel.iloc[0]["candidate"]) - 1
    best_params   = candidates[best_cand_idx]

    with open(os.path.join(out_dir, f"{event_label}_best_params.json"), "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"Best params:\n{pprint.pformat(best_params)}")

    # Collect OOF predictions for threshold selection.
    oof_res  = evaluate_candidate_cv(
        X_trval, y_trval, groups_trval,
        params=best_params, n_splits=n_splits, seed=seed + 777,
        balancing=balancing, neg_ratio=neg_ratio,
        dists=dists_trval if balancing == "dist_weight" else None,
        dw_cfg=baseline_dw if balancing == "dist_weight" else None,
        collect_oof=True,
    )
    mask    = np.isfinite(oof_res["oof_true"]) & np.isfinite(oof_res["oof_prob"])
    thr_obj = choose_threshold_by_fbeta(
        oof_res["oof_true"][mask].astype(int),
        oof_res["oof_prob"][mask].astype(float),
        beta,
    )

    with open(os.path.join(out_dir, f"{event_label}_threshold_stageA.json"), "w", encoding="utf-8") as f:
        json.dump({"beta": beta, **thr_obj}, f, ensure_ascii=False, indent=2)
    print(f"[Threshold] {thr_obj['threshold']:.4f}  (F{beta:.1f}={thr_obj['fbeta']:.3f})")

    # Stage B: optional distance-weight search.
    best_dw_cfg = baseline_dw
    if balancing == "dist_weight" and search_weight_params:
        print(f"[StageB] Searching {n_weight_candidates} distance-weight configs...")
        dw_cfgs  = sample_weight_configs(n_weight_candidates, rng, near_s, mid_s, w_mid, w_far)
        dw_rows: List[Dict[str, Any]] = []

        for j, cfg in enumerate(dw_cfgs, 1):
            res_w   = evaluate_candidate_cv(
                X_trval, y_trval, groups_trval,
                params=best_params, n_splits=n_splits, seed=seed + 900 + j,
                balancing="dist_weight", neg_ratio=neg_ratio,
                dists=dists_trval, dw_cfg=cfg, collect_oof=True,
            )
            mw    = np.isfinite(res_w["oof_true"]) & np.isfinite(res_w["oof_prob"])
            thr_w = choose_threshold_by_fbeta(
                res_w["oof_true"][mw].astype(int),
                res_w["oof_prob"][mw].astype(float), beta,
            )
            dw_rows.append({
                "idx": j,
                **{k: float(cfg[k]) for k in ["near_s", "mid_s", "w_mid", "w_far"]},
                "pr_auc_mean":  float(res_w["pr_auc_mean"]),
                "roc_auc_mean": float(res_w["roc_auc_mean"]),
                "threshold":    float(thr_w["threshold"]),
                "fbeta":        float(thr_w["fbeta"]),
            })
            print(f"[StageB] {j}/{len(dw_cfgs)}  {cfg}  "
                  f"PR-AUC={res_w['pr_auc_mean']:.3f}  thr={thr_w['threshold']:.4f}")

        dw_df  = pd.DataFrame(dw_rows)
        dw_csv = os.path.join(out_dir, f"{event_label}_dw_candidates.csv")
        dw_df.to_csv(dw_csv, index=False)
        print(f"Wrote: {dw_csv}")

        best_dw_row = dw_df.sort_values(
            ["pr_auc_mean", "roc_auc_mean", "fbeta"],
            ascending=[False, False, False], kind="mergesort",
        ).iloc[0].to_dict()

        best_dw_cfg = dict(
            near_s=float(best_dw_row["near_s"]),
            mid_s=float(best_dw_row["mid_s"]),
            w_mid=float(best_dw_row["w_mid"]),
            w_far=float(best_dw_row["w_far"]),
        )
        thr_obj = dict(threshold=float(best_dw_row["threshold"]),
                       fbeta=float(best_dw_row["fbeta"]))

        with open(os.path.join(out_dir, f"{event_label}_best_dw.json"), "w", encoding="utf-8") as f:
            json.dump({"best_dw_cfg": best_dw_cfg, "thr": thr_obj}, f, ensure_ascii=False, indent=2)
        print(f"[StageB] Best weights: {best_dw_cfg}  "
              f"thr={thr_obj['threshold']:.4f}  F{beta:.1f}={thr_obj['fbeta']:.3f}")

    # Final training and test evaluation.
    test_report = train_full_and_eval_test(
        X_trval, y_trval, X_test, y_test,
        best_params=best_params, balancing=balancing, neg_ratio=neg_ratio,
        seed=seed + 999, beta=beta, threshold=thr_obj["threshold"],
        dists_trval=dists_trval if balancing == "dist_weight" else None,
        dw_cfg=best_dw_cfg      if balancing == "dist_weight" else None,
    )

    # Write test report without the prob array.
    test_report_small = {k: v for k, v in test_report.items() if k != "yprob_test"}
    with open(os.path.join(out_dir, f"{event_label}_test_report.json"), "w", encoding="utf-8") as f:
        json.dump(test_report_small, f, ensure_ascii=False, indent=2)

    cm = test_report["confusion_matrix_cm"]
    print(f"\n[Test] PR-AUC={test_report['pr_auc']:.3f}  ROC-AUC={test_report['roc_auc']:.3f}")
    print(f"[Test] threshold={test_report['threshold']:.4f}  F{beta:.1f}={test_report['fbeta']:.3f}")
    print(f"[Test] Confusion [TP FN; FP TN] = {cm}")

    # Event-level hit and latency metrics on flight data only.
    df_test    = df_all.iloc[test_idx].copy()
    if "__row_idx" not in df_test.columns:
        df_test["__row_idx"] = np.arange(len(df_test), dtype=np.int64)

    ev_metrics = compute_event_hit_latency_window_model(
        df_test=df_test,
        prob=np.asarray(test_report["yprob_test"], dtype=float),
        event_label=event_label,
        thr=float(test_report["threshold"]),
        trigger_k=int(trigger_k),
        hit_window_s=float(hit_window_s),
        order_col=order_col,
        detect_time_col=detect_time_col,
        label_time_col=label_time_col,
    )

    return dict(
        model_type="rf",
        event=str(event_label).upper().strip(),
        seed=int(seed),
        pr_auc=float(test_report["pr_auc"])  if np.isfinite(test_report["pr_auc"])  else float("nan"),
        roc_auc=float(test_report["roc_auc"]) if np.isfinite(test_report["roc_auc"]) else float("nan"),
        threshold_used=float(test_report["threshold"]),
        fbeta=float(test_report["fbeta"]),
        n_train=int(len(X_trval)),
        n_test=int(len(X_test)),
        n_pos_train=int(y_trval.sum()),
        n_pos_test=int(y_test.sum()),
        trigger_k=int(trigger_k),
        hit_window_s=float(hit_window_s),
        timing_columns=dict(
            order_col=order_col,
            detect_time_col=detect_time_col,
            label_time_col=label_time_col,
        ),
        **ev_metrics,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Random Forest baseline for flight event detection. "
            "Uses the same deterministic train/val/test split as the GRU and XGBoost "
            "scripts so that all three models are compared on the same test flights."
        )
    )

    # Input / output
    ap.add_argument("--features", required=True,
                    help="Path to features_imputed.csv from the preprocessor.")
    ap.add_argument("--out_dir", default="rf_runs_v3",
                    help="Root output directory. Per-run subdirs are created automatically.")
    ap.add_argument("--events", nargs="+", default=["TAKEOFF", "LANDING"])

    # CV
    ap.add_argument("--cv_splits",    type=int,   default=5)
    ap.add_argument("--n_candidates", type=int,   default=40,
                    help="Number of random hyperparameter configurations to try.")
    ap.add_argument("--beta",         type=float, default=2.0,
                    help="F-beta value for threshold selection.")

    # Seeds
    ap.add_argument("--seed",  type=int,         default=42)
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

    # Balancing
    ap.add_argument("--balancing", choices=BALANCING_CHOICES, default="undersample")
    ap.add_argument("--neg_ratio", type=int, default=8,
                    help="Negatives per positive when --balancing undersample.")

    # Distance weights (only used when --balancing dist_weight)
    ap.add_argument("--near_s",               type=float, default=60.0)
    ap.add_argument("--mid_s",                type=float, default=180.0)
    ap.add_argument("--w_mid",                type=float, default=0.6)
    ap.add_argument("--w_far",                type=float, default=0.25)
    ap.add_argument("--search_weight_params", action="store_true",
                    help="Run Stage B distance-weight search (dist_weight mode only).")
    ap.add_argument("--n_weight_candidates",  type=int,   default=8)

    # Evaluation
    ap.add_argument("--trigger_k",       type=int,   default=1)
    ap.add_argument("--hit_window_s",    type=float, default=10.0)
    ap.add_argument("--nonflight_label", type=str,   default="NONFLIGHT")

    # Output
    ap.add_argument("--metrics_out", type=str, default=None,
                    help="Path to write all per-(event, seed) metrics as a JSON file.")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df_all = pd.read_csv(args.features)

    if "label" not in df_all.columns:
        raise SystemExit("Missing required column: label")
    if "file" not in df_all.columns:
        df_all["file"] = np.arange(len(df_all)).astype(str)
    if "__row_idx" not in df_all.columns:
        df_all["__row_idx"] = np.arange(len(df_all), dtype=np.int64)

    order_col, detect_time_col, label_time_col = choose_time_columns(df_all)
    print(f"[TimingCols] order={order_col}  detect={detect_time_col}  label={label_time_col}")
    print(f"[Balancing] {args.balancing}")

    if args.balancing != "undersample" and args.neg_ratio:
        print("[Warn] --neg_ratio has no effect unless --balancing undersample")
    if args.balancing != "dist_weight" and args.search_weight_params:
        print("[Warn] --search_weight_params has no effect unless --balancing dist_weight")

    seeds       = args.seeds if args.seeds is not None else [args.seed]
    all_results: List[Dict[str, Any]] = []

    for ev in args.events:
        ev_up = str(ev).upper().strip()
        for sd in seeds:
            ev_dir = os.path.join(args.out_dir, ev_up.lower(), f"seed_{sd}")
            os.makedirs(ev_dir, exist_ok=True)

            summary = run_event_pipeline(
                df_all=df_all,
                event_label=ev_up,
                n_splits=args.cv_splits,
                n_candidates=args.n_candidates,
                balancing=args.balancing,
                neg_ratio=args.neg_ratio,
                seed=int(sd),
                beta=args.beta,
                out_dir=ev_dir,
                near_s=args.near_s,
                mid_s=args.mid_s,
                w_mid=args.w_mid,
                w_far=args.w_far,
                search_weight_params=args.search_weight_params,
                n_weight_candidates=args.n_weight_candidates,
                trigger_k=args.trigger_k,
                hit_window_s=args.hit_window_s,
                order_col=order_col,
                detect_time_col=detect_time_col,
                label_time_col=label_time_col,
                nonflight_label=args.nonflight_label,
                flight_val_frac=args.flight_val_frac,
                flight_test_frac=args.flight_test_frac,
                nf_val_frac=args.nf_val_frac,
                nf_test_frac=args.nf_test_frac,
                min_nf_val_groups=args.min_nf_val_groups,
                min_nf_val_hours=args.min_nf_val_hours,
                min_nf_test_groups=args.min_nf_test_groups,
            )
            all_results.append(summary)

    if args.metrics_out is not None:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_strict(out_path, all_results)
        print(f"\n[Metrics] Saved {len(all_results)} records to {out_path}")

    print(f"\nDone. Results in: {args.out_dir}")


if __name__ == "__main__":
    main()
