import os

import gradio as gr
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

from ci_utils import event_choices, predict

MODEL_REPO = os.environ.get("MODEL_REPO", "zoraizafridi/ci-failure-risk")
bundle = joblib.load(hf_hub_download(MODEL_REPO, "ci_failure_model.joblib"))

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LEVELS = ["Low", "Medium", "High"]
COLORS = {"Low": "#35e0a1", "Medium": "#ffc857", "High": "#ff5c7a"}
EMPTY_HTML = ('<div class="empty"><strong>No result yet</strong>Fill in the run and select Classify run. '
              'The verdict and risk score appear here.</div>')

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"], radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#050505", body_text_color="#f3eefc",
    block_background_fill="#0b0612", block_border_color="#2a1347", block_label_text_color="#a99bc0",
    input_background_fill="#12091f", input_border_color="#2a1347",
    button_primary_background_fill="#00E5FF", button_primary_background_fill_hover="#5cf0ff",
    button_primary_text_color="#050505",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&display=swap');
.gradio-container{max-width:1120px!important;background:radial-gradient(760px 520px at 85% -2%,rgba(59,10,107,.9),transparent 68%),radial-gradient(360px 260px at 98% 12%,rgba(0,229,255,.14),transparent 70%),radial-gradient(640px 420px at -4% 100%,rgba(59,10,107,.55),transparent 70%),#050505!important}
h1,.verdict{font-family:'Cormorant Garamond',Georgia,serif!important;font-weight:400!important}
button.primary{border-radius:99px!important;box-shadow:0 0 28px rgba(0,229,255,.28)}
h1{font-size:3.4rem!important;line-height:1!important}
.verdict{font-size:2.5rem;letter-spacing:-.015em;line-height:1.15;margin:0}
.lvl{font-weight:600;margin-top:2px}
.meter{position:relative;height:10px;border-radius:99px;margin:28px 0 8px;background:linear-gradient(90deg,#35e0a1,#ffc857 55%,#ff5c7a)}
.mark{position:absolute;top:-5px;width:4px;height:20px;border-radius:2px;background:#f3eefc;box-shadow:0 0 0 3px #0b0612;transform:translateX(-50%)}
.thr{position:absolute;top:-3px;height:16px;border-left:2px dashed #f3eefc;opacity:.55}
.kv{display:flex;gap:28px;margin:18px 0 6px}.kv span{display:block;color:#a99bc0;font-size:.8rem}
.kv strong{font-size:1.35rem;font-weight:500}
.sub{color:#a99bc0;font-size:.9rem;margin:18px 0 4px}
.bar{display:grid;grid-template-columns:64px 1fr 52px;align-items:center;gap:10px;margin:9px 0;font-size:.88rem;color:#a99bc0}
.bar.sel{color:#f3eefc;font-weight:600}.bt{height:12px;border-radius:99px;background:#12091f;overflow:hidden}
.fill{height:100%;border-radius:99px;background:#2a1347}
.note{color:#a99bc0;font-size:.8rem;margin-top:14px}
.empty{color:#a99bc0}.empty strong{display:block;color:#f3eefc;font-size:1.15rem;margin-bottom:4px}
"""


def classify(additions, deletions, files_modified, msg_len, is_merge, branch_is_main, event, hour, day,
             repo_prior_runs, repo_prev, repo_fails, wf_has_history, wf_prior_runs, wf_prev, wf_fails):
    inputs = {
        "additions": additions, "deletions": deletions, "files_modified": files_modified,
        "msg_len": msg_len, "is_merge": bool(is_merge), "branch_is_main": bool(branch_is_main),
        "event": event, "hour": hour, "dow": DAYS.index(day),
        "repo_prior_runs": repo_prior_runs, "repo_prev": repo_prev, "repo_failures_last10": repo_fails,
        "wf_has_history": bool(wf_has_history),
        "wf_prior_runs": wf_prior_runs if wf_has_history else 0,
        "wf_prev": wf_prev if wf_has_history else "success",
        "wf_failures_last10": wf_fails if wf_has_history else 0,
    }
    try:
        out = predict(bundle, inputs)
    except ValueError as e:
        raise gr.Error(str(e))
    rates = bundle["risk_rates"]
    level, color = out["risk_level"], COLORS[out["risk_level"]]
    verdict = "Likely to fail" if out["predicted_failure"] else "Likely to succeed"
    mark = min(max(out["probability"] * 100, 1), 99)
    top = max(max(rates[x] for x in LEVELS), 0.01)
    bars = "".join(
        f'<div class="bar{" sel" if x == level else ""}"><span>{x}</span>'
        f'<div class="bt"><div class="fill" style="width:{rates[x] / top * 100:.0f}%;'
        f'{"background:" + COLORS[x] if x == level else ""}"></div></div><span>{rates[x]:.1%}</span></div>'
        for x in LEVELS)
    return (
        f'<p class="verdict">{verdict}</p><div class="lvl" style="color:{color}">{level} risk</div>'
        f'<div class="meter"><i class="thr" style="left:{out["threshold"] * 100:.1f}%"></i>'
        f'<i class="mark" style="left:{mark:.1f}%"></i></div>'
        f'<div class="kv"><div><span>Model risk score</span><strong>{out["probability"]:.0%}</strong></div>'
        f'<div><span>Flagged at</span><strong>{out["threshold"]:.0%}</strong></div></div>'
        f'<div class="sub">In held-out test data, runs at this risk level failed {out["observed_failure_rate"]:.1%} '
        f'of the time. Across all test runs: {out["overall_failure_rate"]:.1%}.</div>{bars}'
        f'<div class="note">The score ranks runs by risk. It is not a literal chance of failure, because the model '
        f'was trained with class weights. Runs scoring {out["threshold"]:.0%} or higher are flagged.</div>'
    )


with gr.Blocks(title="CI/CD Risk and Failure Classification") as demo:
    gr.Markdown("# Will this run break the build?\n"
                "Describe a GitHub Actions run before it starts. The model classifies it as likely to fail "
                "or succeed and gives it a Low, Medium or High risk level.")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("#### Change")
            additions = gr.Number(label="Lines added", value=40, minimum=0, precision=0)
            deletions = gr.Number(label="Lines deleted", value=10, minimum=0, precision=0)
            files_modified = gr.Number(label="Files changed", value=3, minimum=0, precision=0)
            msg_len = gr.Number(label="Commit message length (characters)", value=60, minimum=0, precision=0)
            is_merge = gr.Checkbox(label="Merge commit")
            branch_is_main = gr.Checkbox(label="Default branch (main or master)", value=True)
            gr.Markdown("#### Trigger")
            event = gr.Dropdown(event_choices(bundle), value="push", label="Event")
            hour = gr.Slider(0, 23, value=12, step=1, label="Hour (UTC)")
            day = gr.Dropdown(DAYS, value="Wednesday", label="Day of week")
        with gr.Column(scale=2):
            gr.Markdown("#### Repository history")
            repo_prior_runs = gr.Number(label="Earlier runs in the repository", value=200, minimum=0, precision=0)
            repo_prev = gr.Radio(["success", "failure"], value="success", label="Outcome of the previous run")
            repo_fails = gr.Slider(0, 10, value=1, step=1, label="Failures in the last 10 runs")
            gr.Markdown("#### Workflow history")
            wf_has_history = gr.Checkbox(label="This workflow has earlier runs", value=True)
            wf_prior_runs = gr.Number(label="Earlier runs of this workflow", value=100, minimum=1, precision=0)
            wf_prev = gr.Radio(["success", "failure"], value="success", label="Outcome of the workflow's previous run")
            wf_fails = gr.Slider(0, 10, value=1, step=1, label="Workflow failures in the last 10 runs")
            button = gr.Button("Classify run", variant="primary")
        with gr.Column(scale=3):
            result = gr.HTML(EMPTY_HTML)
    button.click(classify,
                 [additions, deletions, files_modified, msg_len, is_merge, branch_is_main, event, hour, day,
                  repo_prior_runs, repo_prev, repo_fails, wf_has_history, wf_prior_runs, wf_prev, wf_fails],
                 result)

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)