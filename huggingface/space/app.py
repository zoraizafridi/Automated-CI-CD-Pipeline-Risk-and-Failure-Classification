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
EMPTY = pd.DataFrame({"Risk level": LEVELS, "Failure rate (%)": [0.0, 0.0, 0.0]})


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
    chart = pd.DataFrame({"Risk level": LEVELS, "Failure rate (%)": [round(rates[x] * 100, 1) for x in LEVELS]})
    verdict = "Likely to fail" if out["predicted_failure"] else "Likely to succeed"
    text = (f"### Risk level: {out['risk_level']}\n"
            f"**{verdict}** (model risk score {out['probability']:.0%}, threshold {out['threshold']:.0%}).\n\n"
            f"In the held-out test data, runs at this risk level failed {out['observed_failure_rate']:.1%} of the time. "
            f"Across all test runs the failure rate was {out['overall_failure_rate']:.1%}.\n\n"
            "The risk score ranks runs by risk. It is not a literal chance of failure, because the model was trained with class weights.")
    return text, chart


with gr.Blocks(title="CI/CD Risk and Failure Classification") as demo:
    gr.Markdown("# CI/CD pipeline risk and failure classification\n"
                "Describe a GitHub Actions run before it starts. The model classifies it as likely to fail "
                "or succeed and gives it a Low, Medium or High risk level.")
    with gr.Row():
        with gr.Column():
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
        with gr.Column():
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
    result = gr.Markdown()
    chart = gr.BarPlot(EMPTY, x="Risk level", y="Failure rate (%)", title="Failure rate by risk level (test data)",
                       y_lim=[0, 100], height=280)
    button.click(classify,
                 [additions, deletions, files_modified, msg_len, is_merge, branch_is_main, event, hour, day,
                  repo_prior_runs, repo_prev, repo_fails, wf_has_history, wf_prior_runs, wf_prev, wf_fails],
                 [result, chart])

if __name__ == "__main__":
    demo.launch()