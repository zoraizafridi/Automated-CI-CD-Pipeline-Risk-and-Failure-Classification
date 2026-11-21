import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from ci_utils import event_choices, predict

st.set_page_config(page_title="CI/CD Risk and Failure Classification", page_icon="🚦", layout="wide")

LEVEL_COLORS = {"Low": "#35e0a1", "Medium": "#ffc857", "High": "#ff5c7a"}

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&display=swap');
  .stApp{background:radial-gradient(760px 520px at 85% -2%,rgba(59,10,107,.9),transparent 68%),radial-gradient(360px 260px at 98% 12%,rgba(0,229,255,.14),transparent 70%),radial-gradient(640px 420px at -4% 100%,rgba(59,10,107,.55),transparent 70%),#050505}
  .block-container{max-width:1120px;padding-top:2.2rem}
  h1,.verdict{font-family:'Cormorant Garamond',Georgia,serif!important;font-weight:400!important}
  h1{font-size:3.6rem!important;line-height:1!important}
  div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:14px}
  div[data-testid="stVerticalBlockBorderWrapper"] h5{margin:0 0 .6rem 0;font-weight:600}
  .stButton>button[kind="primary"]{width:100%;border-radius:99px!important;padding:.7rem;font-weight:650;color:#050505;box-shadow:0 0 28px rgba(0,229,255,.28)}
  .verdict{font-size:2.5rem;letter-spacing:-.015em;line-height:1.15;margin:0}
  .lvl{font-weight:600;margin-top:2px}
  .meter{position:relative;height:10px;border-radius:99px;margin:28px 0 8px;
         background:linear-gradient(90deg,#35e0a1,#ffc857 55%,#ff5c7a)}
  .mark{position:absolute;top:-5px;width:4px;height:20px;border-radius:2px;background:#f3eefc;
        box-shadow:0 0 0 3px #0b0612;transform:translateX(-50%)}
  .thr{position:absolute;top:-3px;height:16px;border-left:2px dashed #f3eefc;opacity:.55}
  .kv{display:flex;gap:28px;margin:18px 0 6px}
  .kv span{display:block;color:#a99bc0;font-size:.8rem}
  .kv strong{font-size:1.35rem;font-weight:500}
  .empty strong{display:block;font-size:1.15rem;margin-bottom:4px}
  .empty{color:#a99bc0}
</style>
""",
    unsafe_allow_html=True,
)

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LEVELS = ["Low", "Medium", "High"]


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "ci_failure_model.joblib")


bundle = load_bundle()

st.title("Will this run break the build?")
st.write(
    "Classifies a GitHub Actions run as likely to fail or succeed before it starts, "
    "and assigns a Low, Medium or High risk level."
)

form_col, result_col = st.columns([3, 2], gap="large")

with form_col:
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

    go = st.button("Classify run", type="primary")

with result_col:
    if not go:
        st.markdown(
            '<div class="empty"><strong>No result yet</strong>Fill in the run and select Classify run. '
            "The verdict and risk score appear here.</div>",
            unsafe_allow_html=True,
        )
    else:
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
        color = LEVEL_COLORS[level]
        obs = out["observed_failure_rate"]
        overall = out["overall_failure_rate"]
        verdict = "Likely to fail" if out["predicted_failure"] else "Likely to succeed"
        mark = min(max(out["probability"] * 100, 1), 99)

        with st.container(border=True):
            st.markdown(
                f'''<p class="verdict">{verdict}</p>
<div class="lvl" style="color:{color}">{level} risk</div>
<div class="meter"><i class="thr" style="left:{out["threshold"] * 100:.1f}%"></i><i class="mark" style="left:{mark:.1f}%"></i></div>
<div class="kv"><div><span>Model risk score</span><strong>{out["probability"]:.0%}</strong></div>
<div><span>Flagged at</span><strong>{out["threshold"]:.0%}</strong></div></div>''',
                unsafe_allow_html=True,
            )

            if obs is not None:
                text = f"In held-out test data, runs at this risk level failed {obs:.1%} of the time."
                if overall is not None:
                    text += f" Across all test runs: {overall:.1%}."
                st.caption(text)

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
                    .mark_bar(cornerRadiusEnd=6, size=18)
                    .encode(
                        y=alt.Y("Risk level:N", sort=LEVELS, axis=alt.Axis(title=None, domain=False, ticks=False)),
                        x=alt.X("Failure rate:Q", axis=alt.Axis(format="%", title=None, grid=False)),
                        color=alt.condition(
                            alt.datum.Selected,
                            alt.Color(
                                "Risk level:N",
                                scale=alt.Scale(domain=LEVELS, range=[LEVEL_COLORS[x] for x in LEVELS]),
                                legend=None,
                            ),
                            alt.value("#2a1347"),
                        ),
                        tooltip=["Risk level", alt.Tooltip("Failure rate:Q", format=".1%")],
                    )
                )
                labels = bars.mark_text(align="left", dx=6, color="#f3eefc").encode(
                    text=alt.Text("Failure rate:Q", format=".1%")
                )
                chart = (bars + labels).properties(height=150, title="Failure rate by risk level (test data)")
                st.altair_chart(chart, width="stretch")

            st.caption(
                f"The score ranks runs by risk. It is not a literal chance of failure, because the model "
                f"was trained with class weights. Runs scoring {out['threshold']:.0%} or higher are flagged. "
                f"Model: {out['model']}."
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