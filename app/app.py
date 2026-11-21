import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from ci_utils import event_choices, predict

st.set_page_config(page_title="CI/CD Risk and Failure Classification", layout="centered")

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LEVELS = ["Low", "Medium", "High"]


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "ci_failure_model.joblib")


bundle = load_bundle()

st.title("CI/CD Pipeline Risk and Failure Classification")
st.write(
    "Classifies a GitHub Actions run as likely to fail or succeed before it starts, "
    "and assigns a Low, Medium or High risk level."
)

with st.container(border=True):
    st.markdown("##### Change")
    c1, c2 = st.columns(2)
    additions = c1.number_input("Lines added", min_value=0, value=40, step=1)
    deletions = c2.number_input("Lines deleted", min_value=0, value=10, step=1)
    files_modified = c1.number_input("Files changed", min_value=0, value=3, step=1)
    msg_len = c2.number_input("Commit message length (characters)", min_value=0, value=60, step=1)
    is_merge = c1.checkbox("Merge commit")
    branch_is_main = c2.checkbox("Default branch (main or master)", value=True)

with st.container(border=True):
    st.markdown("##### Trigger")
    c3, c4, c5 = st.columns(3)
    event = c3.selectbox("Event", event_choices(bundle))
    hour = c4.number_input("Hour (UTC)", min_value=0, max_value=23, value=12, step=1)
    day = c5.selectbox("Day of week", DAYS)

with st.container(border=True):
    st.markdown("##### Repository history")
    repo_prior_runs = st.number_input("Earlier runs in the repository", min_value=0, value=200, step=1)
    repo_prev = st.radio("Outcome of the previous run", ["success", "failure"], horizontal=True)
    repo_fails = st.number_input(
        "Failures in the last 10 runs",
        min_value=0,
        max_value=10,
        value=1,
        step=1,
        help="If the repository has fewer than 10 earlier runs, count the failures among all of them.",
    )

with st.container(border=True):
    st.markdown("##### Workflow history")
    wf_has_history = st.checkbox("This workflow has earlier runs", value=True)
    wf_prior_runs, wf_prev, wf_fails = 0, "success", 0
    if wf_has_history:
        wf_prior_runs = st.number_input("Earlier runs of this workflow", min_value=1, value=100, step=1)
        wf_prev = st.radio("Outcome of the workflow's previous run", ["success", "failure"], horizontal=True)
        wf_fails = st.number_input(
            "Workflow failures in the last 10 runs", min_value=0, max_value=10, value=1, step=1
        )

if st.button("Classify run", type="primary"):
    inputs = {
        "additions": additions,
        "deletions": deletions,
        "files_modified": files_modified,
        "msg_len": msg_len,
        "is_merge": is_merge,
        "branch_is_main": branch_is_main,
        "event": event,
        "hour": hour,
        "dow": DAYS.index(day),
        "repo_prior_runs": repo_prior_runs,
        "repo_prev": repo_prev,
        "repo_failures_last10": repo_fails,
        "wf_has_history": wf_has_history,
        "wf_prior_runs": wf_prior_runs,
        "wf_prev": wf_prev,
        "wf_failures_last10": wf_fails,
    }
    try:
        out = predict(bundle, inputs)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    level = out["risk_level"]
    obs = out["observed_failure_rate"]
    overall = out["overall_failure_rate"]

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Risk level", level)
    m2.metric("Classification", "Likely to fail" if out["predicted_failure"] else "Likely to succeed")
    m3.metric("Model risk score", f"{out['probability']:.0%}")

    if obs is not None:
        text = f"In the held-out test data, runs at this risk level failed {obs:.1%} of the time"
        text += f". Across all test runs the failure rate was {overall:.1%}." if overall is not None else "."
        st.write(text)

        rates = bundle["risk_rates"]
        df = pd.DataFrame(
            {
                "Risk level": LEVELS,
                "Failure rate": [rates[x] for x in LEVELS],
                "Selected": [x == level for x in LEVELS],
            }
        )
        bars = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("Risk level:N", sort=LEVELS, axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y("Failure rate:Q", axis=alt.Axis(format="%", title="Observed failure rate")),
                color=alt.condition(alt.datum.Selected, alt.value("#1f4e79"), alt.value("#b8c4d0")),
                tooltip=["Risk level", alt.Tooltip("Failure rate:Q", format=".1%")],
            )
        )
        labels = bars.mark_text(dy=-6).encode(text=alt.Text("Failure rate:Q", format=".1%"))
        chart = (bars + labels).properties(height=260, title="Failure rate by risk level (test data)")
        st.altair_chart(chart, width="stretch")

    st.caption(
        f"The model risk score ranks runs by risk. It is not a literal chance of failure, because the model "
        f"was trained with class weights. A run is classified as likely to fail at a score of "
        f"{out['threshold']:.0%} or higher. Model: {out['model']}."
    )

with st.expander("About the model"):
    metrics = bundle.get("test_metrics")
    if metrics:
        st.write("Test set results for the selected model.")
        st.table(pd.Series({k: round(float(v), 3) for k, v in metrics.items()}).to_frame("Value"))
    st.write(
        "Trained on public GitHub Actions runs from the GADFPD dataset (CC BY 4.0) combined with runs "
        "collected through the GitHub API. Failures are rare, so some alerts are false alarms. "
        "Run history drives most of the signal, and results can drift as repositories change."
    )