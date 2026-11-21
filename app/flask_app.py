import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from flask import Flask, jsonify, render_template, request

from ci_utils import event_choices, predict

app = Flask(__name__)
bundle = joblib.load(ROOT / "models" / "ci_failure_model.joblib")
EVENTS = event_choices(bundle)
BOOL_FIELDS = ("is_merge", "branch_is_main", "wf_has_history")


class InputError(ValueError):
    pass


def clean_inputs(data):
    if not isinstance(data, dict):
        raise InputError("Send a JSON object with the run details.")
    out = dict(data)
    for key in BOOL_FIELDS:
        val = out.get(key, False)
        if not isinstance(val, bool):
            raise InputError(f"'{key}' must be true or false.")
        out[key] = val
    if str(out.get("event") or "").strip() not in EVENTS:
        raise InputError("'event' must be one of: " + ", ".join(EVENTS) + ".")
    out["event"] = str(out["event"]).strip()
    if not out["wf_has_history"]:
        out.update(wf_prior_runs=0, wf_prev="success", wf_failures_last10=0)
    elif "wf_prior_runs" in out:
        try:
            if float(out["wf_prior_runs"]) < 1:
                raise InputError("'wf_prior_runs' must be at least 1 when the workflow has history.")
        except (TypeError, ValueError):
            raise InputError("'wf_prior_runs' must be a number.")
    return out


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/meta")
def meta():
    return jsonify(events=EVENTS, metrics=bundle.get("test_metrics", {}), model_name=bundle["model_name"],
                   rates=bundle["risk_rates"], threshold=bundle["threshold"], n_features=len(bundle["features"]))


@app.get("/health")
def health():
    return jsonify(status="ok", model=bundle["model_name"], features=len(bundle["features"]))


@app.post("/predict")
def predict_route():
    data = request.get_json(silent=True)
    if data is None:
        return jsonify(error="Request body must be valid JSON."), 400
    try:
        result = predict(bundle, clean_inputs(data))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    result["risk_rates"] = {k: bundle["risk_rates"][k] for k in ("Low", "Medium", "High")}
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)