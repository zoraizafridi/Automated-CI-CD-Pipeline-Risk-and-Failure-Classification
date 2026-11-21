import numpy as np
import pandas as pd

EVENTS_FALLBACK = ["push", "pull_request", "schedule", "workflow_dispatch"]
WINDOW = 10


def log1p_clip(X):
    return np.log1p(np.clip(X, 0, None))


def _num(d, key, lo=0, hi=None, integer=False):
    if key not in d or d[key] in (None, ""):
        raise ValueError(f"'{key}' is required.")
    try:
        v = float(d[key])
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number.")
    if not np.isfinite(v):
        raise ValueError(f"'{key}' must be a finite number.")
    if v < lo or (hi is not None and v > hi):
        upper = hi if hi is not None else "infinity"
        raise ValueError(f"'{key}' must be between {lo} and {upper}.")
    if integer and v != int(v):
        raise ValueError(f"'{key}' must be a whole number.")
    return int(v) if integer else v


def _outcome(d, key):
    v = str(d.get(key)).strip().lower()
    if v in ("1", "failure", "failed", "fail", "true"):
        return 1
    if v in ("0", "success", "passed", "pass", "false"):
        return 0
    raise ValueError(f"'{key}' must be 'success' or 'failure'.")


def _history(d, prefix):
    prior = _num(d, f"{prefix}_prior_runs", 0, integer=True)
    if prior == 0:
        return np.nan, np.nan, 0
    window = min(WINDOW, prior)
    fails = _num(d, f"{prefix}_failures_last10", 0, window, integer=True)
    prev = _outcome(d, f"{prefix}_prev")
    return float(prev), fails / window, prior


def make_features(d, features):
    additions = _num(d, "additions", 0)
    deletions = _num(d, "deletions", 0)
    files = _num(d, "files_modified", 0, integer=True)
    msg_len = _num(d, "msg_len", 0, integer=True)
    hour = _num(d, "hour", 0, 23, integer=True)
    dow = _num(d, "dow", 0, 6, integer=True)
    event = str(d.get("event") or "").strip()
    if not event:
        raise ValueError("'event' is required.")
    is_merge = bool(d.get("is_merge", False))
    branch_main = bool(d.get("branch_is_main", False))

    repo_prev, repo_rate10, repo_prior = _history(d, "repo")
    if bool(d.get("wf_has_history", False)):
        wf_prev, wf_rate10, _ = _history(d, "wf")
    else:
        wf_prev, wf_rate10 = np.nan, np.nan

    churn = additions + deletions
    row = {
        "additions": additions,
        "deletions": deletions,
        "files_modified": files,
        "msg_len": msg_len,
        "churn_total": churn,
        "churn_per_file": churn / max(files, 1),
        "num_parents": 2.0 if is_merge else 1.0,
        "is_merge": float(is_merge),
        "hour": hour,
        "dow": dow,
        "branch_is_main": float(branch_main),
        "repo_prev": repo_prev,
        "repo_rate10": repo_rate10,
        "repo_prior_runs": repo_prior,
        "wf_prev": wf_prev,
        "wf_rate10": wf_rate10,
        "event": event,
    }
    return pd.DataFrame([row])[list(features)]


def risk_level(bundle, proba):
    default = {"low_cut": float(bundle["threshold"]), "high_cut": float(bundle["threshold"])}
    bands = bundle.get("risk_bands") or default
    if proba >= bands["high_cut"]:
        return "High"
    return "Medium" if proba >= bands["low_cut"] else "Low"


def predict(bundle, d):
    X = make_features(d, bundle["features"])
    proba = float(bundle["pipeline"].predict_proba(X)[:, 1][0])
    thr = float(bundle["threshold"])
    level = risk_level(bundle, proba)
    rates = bundle.get("risk_rates") or {}
    return {
        "probability": proba,
        "threshold": thr,
        "predicted_failure": proba >= thr,
        "risk_level": level,
        "observed_failure_rate": rates.get(level),
        "overall_failure_rate": rates.get("overall"),
        "model": bundle["model_name"],
    }


def event_choices(bundle):
    try:
        oh = bundle["pipeline"].named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
        cats = [c for c in oh.categories_[0] if c != "unknown"]
        return cats or EVENTS_FALLBACK
    except Exception:
        return EVENTS_FALLBACK