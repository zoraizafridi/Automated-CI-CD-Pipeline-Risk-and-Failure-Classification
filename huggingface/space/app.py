import os

import gradio as gr
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

from ci_utils import event_choices, predict

MODEL_REPO = os.environ.get("MODEL_REPO", "zoraizafridi/ci-failure-risk")
bundle = joblib.load(hf_hub_download(MODEL_REPO, "ci_failure_model.joblib"))

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
LEVELS = ["Low", "Medium", "High"]
COLOR = {"Low": "#35e0a1", "Medium": "#ffc857", "High": "#ff5c7a"}
EVENTS = event_choices(bundle)
RATES = bundle["risk_rates"]
METRICS = bundle.get("test_metrics", {})
THRESHOLD = float(bundle["threshold"])
MODEL = bundle["model_name"]
N_FEAT = len(bundle["features"])

NAMES = {"macro_F1": "Macro F1", "fail_F1": "Failure F1", "fail_prec": "Failure precision",
         "fail_rec": "Failure recall", "ROC_AUC": "ROC-AUC", "PR_AUC": "PR-AUC"}
MEANS = {
    "macro_F1": "F1 averaged over success and failure, so the rare failures count equally.",
    "fail_F1": "Balance of precision and recall on failing runs.",
    "fail_prec": "Of the runs flagged as likely to fail, the share that really failed.",
    "fail_rec": "Of the runs that really failed, the share the model flagged.",
    "ROC_AUC": "How often a failing run scores higher than a passing one.",
    "PR_AUC": "Precision against recall on the rare failure class.",
}

FIELDS = ["additions", "deletions", "files_modified", "msg_len", "is_merge", "branch_is_main", "event", "hour", "dow",
          "repo_prior_runs", "repo_failures_last10", "repo_prev", "wf_has_history", "wf_prior_runs",
          "wf_failures_last10", "wf_prev"]
PRESETS = {
    "safe": dict(additions=12, deletions=4, files_modified=1, msg_len=48, is_merge=False, branch_is_main=False,
                 event="push", hour=10, dow="Tuesday", repo_prior_runs=400, repo_failures_last10=0,
                 repo_prev="success", wf_has_history=True, wf_prior_runs=150, wf_failures_last10=0, wf_prev="success"),
    "risky": dict(additions=900, deletions=350, files_modified=34, msg_len=12, is_merge=True, branch_is_main=True,
                  event="push", hour=2, dow="Saturday", repo_prior_runs=60, repo_failures_last10=6,
                  repo_prev="failure", wf_has_history=True, wf_prior_runs=25, wf_failures_last10=5, wf_prev="failure"),
}


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.cyan, neutral_hue=gr.themes.colors.slate,
    font=["Segoe UI Variable", "Segoe UI", "system-ui", "-apple-system", "Roboto", "sans-serif"],
    font_mono=["ui-monospace", "Consolas", "monospace"], radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#050505", body_background_fill_dark="#050505",
    body_text_color="#f3eefc", body_text_color_dark="#f3eefc",
    body_text_color_subdued="#a99bc0", body_text_color_subdued_dark="#a99bc0",
    background_fill_primary="#050505", background_fill_primary_dark="#050505",
    background_fill_secondary="#0b0612", background_fill_secondary_dark="#0b0612",
    block_background_fill="transparent", block_background_fill_dark="transparent",
    block_border_color="#2a1347", block_border_color_dark="#2a1347",
    block_label_text_color="#a99bc0", block_label_text_color_dark="#a99bc0",
    border_color_primary="#2a1347", border_color_primary_dark="#2a1347",
    input_background_fill="#12091f", input_background_fill_dark="#12091f",
    input_background_fill_focus="#12091f", input_background_fill_focus_dark="#12091f",
    input_border_color="#2a1347", input_border_color_dark="#2a1347",
    input_border_color_focus="#00E5FF", input_border_color_focus_dark="#00E5FF",
    checkbox_background_color="#12091f", checkbox_background_color_dark="#12091f",
    checkbox_background_color_selected="#00E5FF", checkbox_background_color_selected_dark="#00E5FF",
    checkbox_border_color="#2a1347", checkbox_border_color_dark="#2a1347",
    checkbox_border_color_selected="#00E5FF", checkbox_border_color_selected_dark="#00E5FF",
    button_primary_background_fill="#00E5FF", button_primary_background_fill_dark="#00E5FF",
    button_primary_background_fill_hover="#00E5FF", button_primary_background_fill_hover_dark="#00E5FF",
    button_primary_text_color="#050505", button_primary_text_color_dark="#050505",
    button_primary_border_color="#00E5FF", button_primary_border_color_dark="#00E5FF",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&display=swap');
:root,.dark{--bg:#050505;--panel:#0b0612;--field:#12091f;--line:#2a1347;--amethyst:#3B0A6B;--ink:#f3eefc;--muted:#a99bc0;--accent:#00E5FF;--accent-ink:#050505;
--low:#35e0a1;--med:#ffc857;--high:#ff5c7a;--lit:#a66bff;--serif:"Cormorant Garamond","Playfair Display","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
--sans:"Segoe UI Variable","Segoe UI",system-ui,-apple-system,Roboto,sans-serif}

/* ---- chrome removal + page background ---- */
html,body{background:var(--bg)!important;color-scheme:dark;overflow-x:hidden}
body::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
background:radial-gradient(760px 560px at 84% 4%,rgba(59,10,107,.95),transparent 68%),radial-gradient(380px 280px at 97% 20%,rgba(0,229,255,.17),transparent 70%),radial-gradient(660px 440px at -4% 102%,rgba(59,10,107,.6),transparent 70%)}
footer,.built-with,.settings,button.show-api,.api-docs{display:none!important}
gradio-app,.gradio-container,.gradio-container .main,.gradio-container .wrap{background:transparent!important}
.gradio-container{position:relative;z-index:1;max-width:1120px!important;margin:0 auto!important;padding:0 24px 96px!important;color:var(--ink);font:15px/1.55 var(--sans);font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
.gradio-container,.gradio-container :where(label,span,p,input,select,textarea,button,li,td,th,div){font-family:var(--sans)}
.gradio-container :where(h1,h2,h3,em){font-family:var(--serif)}
.gradio-container{overflow:visible!important}
.gradio-container .main,.gradio-container .main.fillable{padding:0!important}
.gradio-container .contain{padding:0!important}
.gradio-container .gap{gap:0!important}
.gradio-container .form{background:transparent!important;border:0!important;box-shadow:none!important;border-radius:0!important;overflow:visible!important;gap:0}
.gradio-container .form>*{min-width:0!important}
.gradio-container .block{background:transparent!important;border:0!important;box-shadow:none!important;padding:0!important}
.gradio-container .html-container,.gradio-container .prose{padding:0!important;max-width:none!important;color:inherit}
.gradio-container :focus-visible{outline:2px solid var(--accent)!important;outline-offset:3px}

/* ---- top navigation (the five tabs) ---- */
.gradio-container .tabs{border:0!important;background:transparent!important;overflow:visible!important}
.gradio-container .tab-wrapper,.gradio-container .tab-container{position:sticky;top:0;z-index:50;height:60px!important;border:0!important;padding:0!important;background:transparent!important;overflow:visible!important}
.gradio-container .tab-container{display:flex;align-items:center;gap:4px;flex-wrap:nowrap}
.gradio-container .tab-container::after{content:"";position:absolute;z-index:-1;top:0;bottom:auto;height:100%;left:50%;width:100vw;transform:translateX(-50%);
background:color-mix(in srgb,#050505 88%,transparent);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border:0;border-bottom:1px solid var(--line)}
.gradio-container .tab-container::before{content:"CI RISK";display:flex;align-items:center;flex:none;height:60px;margin-right:32px;padding-left:40px;white-space:nowrap;
font-weight:700;letter-spacing:.14em;font-size:.82rem;color:var(--ink);position:static;border:0;
background:radial-gradient(circle at 5px 50%,#35e0a1 4px,transparent 4.6px) 0 50%/30px 14px no-repeat,radial-gradient(circle at 15px 50%,#ffc857 4px,transparent 4.6px) 0 50%/30px 14px no-repeat,radial-gradient(circle at 25px 50%,#ff5c7a 4px,transparent 4.6px) 0 50%/30px 14px no-repeat,linear-gradient(rgba(243,238,252,.35),rgba(243,238,252,.35)) 5px 50%/20px 2px no-repeat}
.gradio-container .tab-container button[role="tab"]{position:relative;height:60px;padding:0 14px!important;margin:0!important;background:transparent!important;border:0!important;border-radius:0!important;
color:var(--muted)!important;font-size:.7rem!important;letter-spacing:.18em;text-transform:uppercase;white-space:nowrap;font-weight:400!important;box-shadow:none!important}
.gradio-container .tab-container button[role="tab"]:hover,.gradio-container .tab-container button[role="tab"].selected,.gradio-container .tab-container button[role="tab"][aria-selected="true"]{color:var(--ink)!important}
.gradio-container .tab-container button[role="tab"].selected::after,.gradio-container .tab-container button[role="tab"][aria-selected="true"]::after{content:"";position:absolute;left:12px;right:12px;bottom:0;height:2px;background:var(--accent)}
.gradio-container .tab-container::-webkit-scrollbar{display:none}
.gradio-container .overflow-menu{display:none!important}
.gradio-container .tabitem{border:0!important;background:transparent!important;padding:48px 0 0!important}

/* ---- shared typography (.ci scope) ---- */
.ci h1{font-family:var(--serif)!important;font-size:clamp(3rem,7vw,5.6rem)!important;line-height:.98!important;letter-spacing:-.02em!important;font-weight:400!important;margin:0 0 22px!important;max-width:11ch;padding:0!important;color:var(--ink);border:0!important}
.ci h1 em{font-style:italic;color:var(--lit)}
.ci h2{font-family:var(--serif)!important;font-size:2.2rem!important;letter-spacing:-.01em;font-weight:500!important;margin:0 0 8px!important;padding:0!important;line-height:1.2;color:var(--ink);border:0!important}
.ci h3{font-family:var(--serif)!important;padding:0!important;color:var(--ink);border:0!important}
.ci p,.ci li{font-size:15px;line-height:1.55;margin:0;color:var(--muted)}
.ci .lead{font-size:1.1rem;color:var(--muted);max-width:44ch;margin:0 0 32px}
.ci a{text-decoration:none}
.ci .prose{max-width:68ch!important}.ci .prose p,.ci .prose li{color:var(--muted)}.ci .prose ul{padding-left:18px;margin:0}.ci .prose strong{color:var(--ink);font-weight:600}
.ci .prose p+p{margin-top:1em}.ci .prose li{margin:0 0 .5em}
.ci .prose h3{font-size:1.45rem;font-weight:500;margin:0 0 6px!important}

/* ---- buttons ---- */
.gradio-container button.primary,.gradio-container button.btn{display:inline-block;width:auto!important;flex:none!important;background:var(--accent)!important;color:var(--accent-ink)!important;border:1px solid var(--accent)!important;border-radius:99px!important;padding:13px 28px!important;font:600 15px var(--sans)!important;cursor:pointer;box-shadow:0 0 28px rgba(0,229,255,.28)!important;transition:box-shadow .2s;min-width:0!important;height:auto!important}
.gradio-container button.primary:hover,.gradio-container button.btn:hover{box-shadow:0 0 40px rgba(0,229,255,.5)!important}
.gradio-container button.link{background:none!important;border:0!important;box-shadow:none!important;padding:0!important;min-width:0!important;width:auto!important;flex:none!important;height:auto!important;color:var(--accent)!important;font:inherit!important;font-size:.9rem!important;text-decoration:underline;text-underline-offset:3px;cursor:pointer}
.cta-row{margin-top:-4px!important}

/* ---- overview ---- */
.ci .hero{position:relative;min-height:540px;padding-top:36px}
.ci .hero .copy{position:relative;z-index:1}
.ci .ribbons{overflow:visible;position:absolute;right:-60px;top:-30px;width:min(680px,62%);height:auto;pointer-events:none}
.ci .ribbons .draw{stroke-dasharray:1400;stroke-dashoffset:1400;animation:draw 2.4s .15s cubic-bezier(.2,.7,.2,1) forwards}
@keyframes draw{to{stroke-dashoffset:0}}
.hero{position:relative;min-height:540px;padding-top:36px!important;display:flex!important;flex-direction:column;align-items:flex-start;gap:0!important}
.block.ribwrap{position:absolute!important;right:-60px;top:-30px;width:min(680px,62%)!important;pointer-events:none;z-index:0}
.prose.ribwrap{position:static!important;width:100%!important;right:auto!important;top:auto!important}
.ribwrap .ci,.ribwrap .html-container{overflow:visible!important}
.herocopy,.hero button.btn{position:relative;z-index:1}
.herocopy{padding-bottom:32px!important}
.ci .ribbons{position:static;width:100%;right:auto;top:auto}
.ci .ladder{margin:40px 0 0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));column-gap:40px}
.ci .rung{border-top:1px solid var(--line);padding:22px 0 8px}
.ci .rung .lab{display:flex;align-items:center;gap:8px;color:var(--muted);font-size:.9rem}
.ci .rung .lab i{width:8px;height:8px;border-radius:50%;display:block}
.ci .rung b{display:block;margin:10px 0 14px;font-size:clamp(2.6rem,5vw,3.8rem);font-weight:300;letter-spacing:-.03em;line-height:1;color:var(--ink);font-variant-numeric:lining-nums tabular-nums}
.ci .rung .trk{height:4px;border-radius:4px;background:var(--line);overflow:hidden}
.ci .rung .trk i{display:block;height:100%;border-radius:4px;min-width:4px}
.ci .rung .cap{display:block;margin-top:10px;color:var(--muted);font-size:.85rem}
.ci .ladder-cap{color:var(--muted);margin:16px 0 0;max-width:62ch;font-size:.92rem}
.ci .facts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));column-gap:40px;margin-top:48px}
.ci .facts>div{padding:22px 0 0;border-top:1px solid var(--line)}
.ci .facts .n{color:var(--accent);font-size:.8rem;letter-spacing:.14em;display:block;margin-bottom:12px}
.ci .facts h3{font-size:1.45rem;font-weight:500;margin:0 0 6px!important}
.ci .facts p{margin:0;color:var(--muted)}

/* ---- classify: form groups ---- */
.grp{display:grid!important;grid-template-columns:170px minmax(0,1fr);gap:8px 28px!important;padding:26px 0!important;border-top:1px solid var(--line)!important;align-items:start}
.grp h3{margin:0!important;font-size:1.35rem;font-weight:500;line-height:1.2;color:var(--ink)}
.grp .gp{margin:2px 0 0;color:var(--muted);font-size:.87rem}
.grp .fields{display:flex;flex-direction:column;gap:16px}
.pair,.pair .form{display:grid!important;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px!important;background:transparent!important;border:0!important}
.pair>*{min-width:0!important}
.grp .fields .form{gap:16px}
.gradio-container label>span[data-testid="block-info"],.gradio-container span[data-testid="block-info"]{color:var(--muted)!important;font-size:.85rem!important;font-weight:400!important;background:none!important;margin-bottom:0!important;display:block}
.gradio-container input[type=number],.gradio-container input[type=text],.gradio-container .wrap-inner,.gradio-container select{border:1px solid var(--line)!important;border-radius:8px!important;background:var(--field)!important;color:var(--ink)!important;font-size:.95rem!important;box-shadow:none!important}
.gradio-container input[type=number],.gradio-container .wrap-inner{padding:10px 12px!important;margin-top:6px}
.gradio-container .wrap-inner{padding:0 12px!important}
.gradio-container .wrap-inner input{border:0!important;padding:10px 0!important;margin:0!important;background:transparent!important}
.gradio-container input[type=number]:focus,.gradio-container .wrap-inner:focus-within{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(0,229,255,.15)!important}
.gradio-container ul.options{background:var(--field)!important;border:1px solid var(--line)!important}
.gradio-container ul.options li{color:var(--ink)!important}
.gradio-container ul.options li.selected,.gradio-container ul.options li:hover{background:var(--amethyst)!important}
.gradio-container input[type=checkbox]{accent-color:var(--accent);width:17px;height:17px}
.gradio-container .checkbox-container label,.gradio-container label.checkbox,.gradio-container input[type=checkbox]+span{color:var(--ink)!important;font-size:.92rem!important}
.tg{padding-top:22px!important}
#wf_toggle{padding-top:0!important;margin-bottom:14px!important}
.gradio-container [disabled]{opacity:.45;cursor:not-allowed}
/* segmented radios */
.gradio-container fieldset{border:0!important;padding:0!important;margin:0!important}
.gradio-container .wrap:has(> label > input[type=radio]){display:inline-flex!important;flex-direction:row!important;flex-wrap:nowrap!important;gap:0!important;margin-top:6px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--field)}
.gradio-container label:has(input[type=radio]){margin:0!important;padding:9px 16px!important;cursor:pointer;background:transparent!important;border:0!important;border-radius:0!important;box-shadow:none!important;color:var(--muted)!important;font-size:.9rem!important}
.gradio-container label:has(input[type=radio]) input{position:absolute;opacity:0;pointer-events:none;width:0;height:0}
.gradio-container label:has(input[type=radio]:checked){background:var(--amethyst)!important;box-shadow:inset 0 0 0 1px var(--accent)!important;color:var(--ink)!important;font-weight:600}
.gradio-container label:has(input[type=radio]) span{color:inherit!important;font-size:.9rem!important}
.actions{border-top:1px solid var(--line)!important;padding-top:24px!important;display:flex!important;flex-direction:row!important;align-items:center;gap:14px!important;flex-wrap:wrap}
.actions>*{flex:none!important;min-width:0!important;width:auto!important}
.actions .presets{color:var(--muted);font-size:.9rem}
.ci .err{margin-bottom:18px;padding:12px 14px;border-left:3px solid var(--high);background:color-mix(in srgb,var(--high) 10%,var(--panel));border-radius:0 8px 8px 0;color:var(--ink)}

/* ---- classify: result panel ---- */
.layout{display:grid!important;grid-template-columns:minmax(0,1fr) 370px;gap:56px!important;align-items:start}
.layout>*{min-width:0!important}
#aside{position:sticky;top:84px;align-self:start;border:1px solid rgba(0,229,255,.22)!important;background:linear-gradient(165deg,rgba(59,10,107,.45),rgba(5,5,5,.75) 60%)!important;border-radius:16px!important;padding:26px!important;box-shadow:0 0 60px rgba(59,10,107,.35)!important}
.ci .empty{color:var(--muted)}.ci .empty strong{display:block;color:var(--ink);font-size:1.1rem;margin-bottom:4px;font-weight:700}
.ci .verdict,.ci .verdict span{font-family:var(--serif)!important}
.ci .verdict{font-size:2.5rem;line-height:1.05;font-weight:500;letter-spacing:-.01em;margin:0;display:flex;align-items:center;gap:12px;color:var(--ink)}
.ci .dot{width:12px;height:12px;border-radius:50%;flex:none;background:var(--c,var(--muted));display:block}
.ci .lvl{margin-top:4px;color:var(--c)}
.ci .meter{margin:30px 0 8px}
.ci .track{position:relative;height:8px;border-radius:99px;background:linear-gradient(90deg,var(--low),var(--med) 55%,var(--high))}
.ci .mark{position:absolute;top:-6px;width:4px;height:20px;border-radius:2px;background:var(--ink);box-shadow:0 0 0 3px var(--panel);transform:translateX(-50%);animation:slide .5s cubic-bezier(.2,.8,.2,1)}
@keyframes slide{from{left:0}}
.ci .thr{position:absolute;top:-4px;height:16px;border-left:2px dashed var(--ink);opacity:.5}
.ci .scale{display:flex;justify-content:space-between;color:var(--muted);font-size:.78rem;margin-top:8px}
.ci .kv{display:flex;gap:32px;margin:18px 0 0}.ci .kv span{display:block;color:var(--muted);font-size:.8rem}.ci .kv strong{font-size:1.4rem;font-weight:500;color:var(--ink)}
.ci .sub{color:var(--muted);font-size:.88rem;margin:22px 0 4px}
.ci .bar{display:grid;grid-template-columns:64px 1fr 52px;align-items:center;gap:10px;margin:9px 0;font-size:.88rem;color:var(--muted)}
.ci .bar.sel{color:var(--ink);font-weight:600}
.ci .bt{height:8px;border-radius:99px;background:var(--field);overflow:hidden}
.ci .fill{height:100%;border-radius:99px;background:var(--line);animation:grow .6s cubic-bezier(.2,.8,.2,1)}
@keyframes grow{from{width:0}}
.ci .bar.sel .fill{background:var(--c)}
.ci .note{color:var(--muted);font-size:.8rem;margin:16px 0 0}

/* ---- model / data / method ---- */
.ci table{border-collapse:collapse;width:100%;max-width:760px;margin:20px 0 40px;background:transparent;border:0}
.ci th,.ci td{text-align:left;padding:14px 16px 14px 0;border:0;border-bottom:1px solid var(--line);vertical-align:top;background:transparent;color:var(--ink)}
.ci tr{border:0!important;background:transparent!important}
.ci th{font-weight:500;font-size:.85rem;color:var(--muted)}
.ci td.d{color:var(--muted)}
.ci td.v{font-size:1.3rem;font-weight:500;white-space:nowrap;color:var(--ink)}
.ci .split{display:grid;grid-template-columns:1fr 1fr;gap:48px}
@media (max-width:900px){
 .layout{grid-template-columns:1fr;gap:32px!important}#aside{position:static}
 .grp{grid-template-columns:1fr}
 .ci .facts,.ci .split{grid-template-columns:1fr}
 .gradio-container .tab-container::before{display:none}
}
@media (max-width:760px){.ci .ribbons{opacity:.45;width:110%;right:-30%}.ci .facts{row-gap:20px}.ci .ladder{grid-template-columns:1fr;row-gap:6px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}.ci .ribbons .draw{animation:none;stroke-dashoffset:0}}
"""

RIBBONS = """<svg class="ribbons" viewBox="0 0 640 560" fill="none" aria-hidden="true"><defs><linearGradient id="rg" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#3B0A6B"/><stop offset=".55" stop-color="#a66bff"/><stop offset="1" stop-color="#00E5FF"/></linearGradient><filter id="rb" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="16"/></filter></defs><g filter="url(#rb)" opacity=".55" stroke="url(#rg)" stroke-width="22" stroke-linecap="round"><path d="M40 520C170 470 240 290 350 220S540 90 620 40"/><path d="M0 400C150 410 260 320 310 220S450 100 600 140"/></g><g stroke="url(#rg)" stroke-linecap="round"><path class="draw" stroke-width="3" d="M40 520C170 470 240 290 350 220S540 90 620 40"/><path class="draw" stroke-width="2" d="M0 400C150 410 260 320 310 220S450 100 600 140"/><path class="draw" stroke-width="1.5" d="M90 560C220 520 350 470 420 350S520 220 630 260"/><path class="draw" stroke-width="1" d="M0 300C120 340 230 260 290 150S410 30 480 0"/></g><g fill="#00E5FF"><circle cx="350" cy="220" r="5"/><circle cx="310" cy="220" r="3.5" fill="#a66bff"/><circle cx="420" cy="350" r="4"/><circle cx="600" cy="140" r="3.5"/></g></svg>"""

fail_rec = pct(METRICS.get("fail_rec", 0) or 0)

HERO_COPY = (
    '<div class="ci"><div class="copy">'
    '<h1>Will this run <em>break</em> the build?</h1>'
    '<p class="lead">Describe a GitHub Actions run before it starts. The model classifies it as likely to fail '
    'or succeed and rates the risk as Low, Medium or High.</p></div></div>'
)
HERO_RIBBONS = f'<div class="ci">{RIBBONS}</div>'
LADDER = "".join(
    f'<div class="rung"><span class="lab"><i style="background:{COLOR[l]}"></i>{l} risk</span><b>{pct(RATES[l], 1)}</b>'
    f'<div class="trk"><i style="width:{RATES[l] * 100}%;background:{COLOR[l]}"></i></div>'
    f'<span class="cap">of runs failed in test data</span></div>'
    for l in LEVELS
)
OVERVIEW_BOTTOM = (
    f'<div class="ci"><div class="ladder">{LADDER}</div>'
    f'<p class="ladder-cap">Observed in held-out test data, where {pct(RATES["overall"], 1)} of all runs failed. '
    'Bars show each rate out of 100%.</p>'
    '<div class="facts">'
    '<div><span class="n">01</span><h3>What you enter</h3><p>The size of the change, what triggers the run, and how the repository and workflow have behaved lately.</p></div>'
    '<div><span class="n">02</span><h3>What you get</h3><p>A verdict, a risk level and a score, shown against the threshold at which runs are flagged.</p></div>'
    f'<div><span class="n">03</span><h3>How far to trust it</h3><p>It ranks runs for triage. It catches roughly {fail_rec} of failures, so treat it as an early warning, not a guarantee.</p></div>'
    '</div></div>'
)

CLASSIFY_HEAD = (
    '<div class="ci"><h2>Classify a run</h2>'
    '<p class="lead" style="margin-bottom:32px">Fill in what you know about the run, then select Classify run.</p></div>'
)
EMPTY_HTML = '<div class="ci"><div class="empty"><strong>No result yet</strong>The verdict and risk score appear here.</div></div>'


def group_head(title, text):
    return f'<div class="ci"><h3>{title}</h3><p class="gp">{text}</p></div>'


ROWS = "".join(
    f'<tr><td>{NAMES.get(k, k)}</td><td class="v">{float(v):.3f}</td><td class="d">{MEANS.get(k, "")}</td></tr>'
    for k, v in METRICS.items()
)
MBARS = "".join(
    f'<div class="bar sel" style="--c:{COLOR[l]};grid-template-columns:70px 1fr 56px"><span>{l}</span>'
    f'<div class="bt"><div class="fill" style="width:{RATES[l] / RATES["High"] * 100}%"></div></div><span>{pct(RATES[l], 1)}</span></div>'
    for l in LEVELS
)
MODEL_HTML = (
    f'<div class="ci"><div class="prose"><h2>How well it works</h2>'
    f'<p>{MODEL}, scored once on a held-out test set that comes later in time than the training data.</p></div>'
    '<table><thead><tr><th>Measure</th><th>Result</th><th>What it means</th></tr></thead>'
    f'<tbody>{ROWS}</tbody></table>'
    '<div class="prose"><h2>Failure rate by risk level</h2>'
    f'<p>Share of test runs that failed at each level. Runs scoring {pct(THRESHOLD)} or higher are flagged as likely to fail.</p></div>'
    f'<div style="max-width:560px;margin-top:16px">{MBARS}</div></div>'
)
DATA_HTML = (
    '<div class="ci"><div class="prose"><h2>Where the data comes from</h2>'
    '<p>Two sources of GitHub Actions runs, joined end to end in time.</p></div>'
    '<div class="split" style="margin-top:28px">'
    '<div class="prose"><h3>GADFPD</h3><p>GitHub Actions CI/CD Failure Prediction Dataset by Faruna et al. (2026), '
    'Mendeley Data, DOI 10.17632/mggwn7rj9f.1. Licence CC BY 4.0. Covers runs from 8 August 2016 to 18 April 2026.</p></div>'
    '<div class="prose"><h3>Own collection</h3><p>23,818 workflow runs from 14 public repositories, 19 April to 20 November 2026, '
    'collected with the GitHub REST API. Repository and actor names are hashed and no commit message text is stored.</p></div></div>'
    '<div class="prose" style="margin-top:36px"><p>Only runs with a clear success or failure outcome and a first attempt are kept. '
    'Runs are sorted by time and split 70% train, 15% validation and 15% test, so the test set is always the most recent data.</p></div></div>'
)
METHOD_HTML = (
    '<div class="ci"><div class="prose"><h2>How it was built</h2>'
    '<p>Six algorithms were tuned: Logistic Regression, Decision Tree, KNN, Linear SVM, Random Forest and Histogram Gradient Boosting. '
    'Each used randomized search with 5-fold time-series cross-validation, and the winner was picked by validation macro-F1. '
    f'That was <strong>{MODEL}</strong>, trained with class weights.</p>'
    f'<p>It uses <strong>{N_FEAT} features</strong>: change size, merge flag, hour, weekday, branch, trigger event, and run history '
    '(previous outcome, failure rate over the last 10 runs, number of earlier runs) for both the repository and the workflow.</p>'
    '<h2 style="margin-top:40px!important">Limitations</h2><ul>'
    f'<li>Failures are fairly rare. At the chosen threshold the model finds about {fail_rec} of them, so it misses some failures and raises some false alarms.</li>'
    '<li>Most of the signal comes from run history. A brand new repository or workflow is judged mainly on change size and trigger.</li>'
    '<li>The failure rate differs between the two data sources, and repositories change over time, so the model should be retrained periodically.</li>'
    '<li>The displayed failure probability is calibrated on the time-ordered validation split. It is an estimate, not a guarantee, and may drift as repositories change.</li>'
    '</ul></div></div>'
)


def result_html(o):
    c = COLOR[o["risk_level"]]
    verdict = "Likely to fail" if o["predicted_failure"] else "Likely to succeed"
    mark = min(max(o["probability"] * 100, 1), 99)
    text = ""
    if o["observed_failure_rate"] is not None:
        text = f'In held-out test data, runs at this risk level failed {pct(o["observed_failure_rate"], 1)} of the time.'
        if o["overall_failure_rate"] is not None:
            text += f' Across all test runs: {pct(o["overall_failure_rate"], 1)}.'
    top = max(max(RATES[l] for l in LEVELS), 0.01)
    bars = "".join(
        f'<div class="bar{" sel" if l == o["risk_level"] else ""}" style="--c:{COLOR[l]}"><span>{l}</span>'
        f'<div class="bt"><div class="fill" style="width:{RATES[l] / top * 100}%"></div></div><span>{pct(RATES[l], 1)}</span></div>'
        for l in LEVELS
    )
    return (
        f'<div class="ci" style="--c:{c}"><p class="verdict"><i class="dot"></i><span>{verdict}</span></p>'
        f'<div class="lvl">{o["risk_level"]} risk</div>'
        f'<div class="meter"><div class="track"><i class="thr" style="left:{o["threshold"] * 100}%"></i>'
        f'<i class="mark" style="left:{mark}%"></i></div><div class="scale"><span>0%</span><span>100%</span></div></div>'
        f'<div class="kv"><div><span>Estimated failure probability</span><strong>{pct(o["probability"])}</strong></div>'
        f'<div><span>Flagged at</span><strong>{pct(o["threshold"])}</strong></div></div>'
        f'<p class="sub">{text}</p>{bars}'
        f'<p class="note">This probability is calibrated using a held-out, time-ordered validation split to correct the class-weighted model scores. '
        f'It is an estimate, not a guarantee. Runs at or above {pct(o["threshold"])} are flagged. Model: {o["model"]}.</p></div>'
    )


def classify(additions, deletions, files_modified, msg_len, is_merge, branch_is_main, event, hour, day,
             repo_prior_runs, repo_fails, repo_prev, wf_has_history, wf_prior_runs, wf_fails, wf_prev, current):
    has_wf = bool(wf_has_history)
    inputs = {
        "additions": additions, "deletions": deletions, "files_modified": files_modified,
        "msg_len": msg_len, "is_merge": bool(is_merge), "branch_is_main": bool(branch_is_main),
        "event": event, "hour": hour, "dow": DAYS.index(day),
        "repo_prior_runs": repo_prior_runs, "repo_prev": repo_prev, "repo_failures_last10": repo_fails,
        "wf_has_history": has_wf,
        "wf_prior_runs": wf_prior_runs if has_wf else 0,
        "wf_prev": wf_prev if has_wf else "success",
        "wf_failures_last10": wf_fails if has_wf else 0,
    }
    try:
        # predict() already applies the Platt calibrator, threshold and risk bands.
        out = predict(bundle, inputs)
    except (ValueError, KeyError, AttributeError) as e:
        return current, f'<div class="ci"><div class="err" role="alert">{e}</div></div>'
    return result_html(out), ""


def load_preset(name):
    p = PRESETS[name]
    return [p[k] for k in FIELDS]


def toggle_workflow(has):
    return [gr.update(interactive=bool(has)) for _ in range(3)]


with gr.Blocks(title="CI Risk · CI/CD Risk and Failure Classification") as demo:
    with gr.Tabs(elem_id="nav") as tabs:
        # ------------------------------------------------------------ Overview
        with gr.Tab("Overview", id="overview"):
            with gr.Column(elem_classes="hero"):
                gr.HTML(HERO_RIBBONS, elem_classes="ribwrap")
                gr.HTML(HERO_COPY, elem_classes="herocopy")
                to_classify = gr.Button("Classify a run", elem_classes="btn")
            gr.HTML(OVERVIEW_BOTTOM)

        # ------------------------------------------------------------ Classify
        with gr.Tab("Classify", id="classify"):
            gr.HTML(CLASSIFY_HEAD)
            with gr.Row(elem_classes="layout"):
                with gr.Column(elem_classes="ci-form"):
                    err = gr.HTML("")
                    with gr.Row(elem_classes="grp"):
                        gr.HTML(group_head("Change", "What the commit touches."))
                        with gr.Column(elem_classes="fields"):
                            with gr.Row(elem_classes="pair"):
                                additions = gr.Number(label="Lines added", value=40, minimum=0, precision=0)
                                deletions = gr.Number(label="Lines deleted", value=10, minimum=0, precision=0)
                            with gr.Row(elem_classes="pair"):
                                files_modified = gr.Number(label="Files changed", value=3, minimum=0, precision=0)
                                msg_len = gr.Number(label="Commit message length (characters)", value=60, minimum=0, precision=0)
                            with gr.Row(elem_classes="pair"):
                                is_merge = gr.Checkbox(label="Merge commit", elem_classes="tg")
                                branch_is_main = gr.Checkbox(label="Default branch (main or master)", value=True, elem_classes="tg")
                    with gr.Row(elem_classes="grp"):
                        gr.HTML(group_head("Trigger", "What starts the run, and when."))
                        with gr.Column(elem_classes="fields"):
                            with gr.Row(elem_classes="pair"):
                                event = gr.Dropdown(EVENTS, value="push", label="Event", filterable=False)
                                hour = gr.Number(label="Hour (UTC)", value=12, minimum=0, maximum=23, precision=0)
                            with gr.Row(elem_classes="pair"):
                                day = gr.Dropdown(DAYS, value="Wednesday", label="Day of week", filterable=False)
                    with gr.Row(elem_classes="grp"):
                        gr.HTML(group_head("Repository history", "How the repository behaved recently."))
                        with gr.Column(elem_classes="fields"):
                            with gr.Row(elem_classes="pair"):
                                repo_prior_runs = gr.Number(label="Earlier runs in the repository", value=200, minimum=0, precision=0)
                                repo_fails = gr.Number(label="Failures in the last 10 runs", value=1, minimum=0, maximum=10, precision=0)
                            repo_prev = gr.Radio(["success", "failure"], value="success", label="Outcome of the previous run")
                    with gr.Row(elem_classes="grp"):
                        gr.HTML(group_head("Workflow history", "The same, for this workflow only."))
                        with gr.Column(elem_classes="fields"):
                            wf_has_history = gr.Checkbox(label="This workflow has earlier runs", value=True, elem_id="wf_toggle")
                            with gr.Row(elem_classes="pair"):
                                wf_prior_runs = gr.Number(label="Earlier runs of this workflow", value=100, minimum=1, precision=0)
                                wf_fails = gr.Number(label="Workflow failures in the last 10 runs", value=1, minimum=0, maximum=10, precision=0)
                            wf_prev = gr.Radio(["success", "failure"], value="success", label="Outcome of the workflow's previous run")
                    with gr.Row(elem_classes="actions"):
                        button = gr.Button("Classify run", variant="primary")
                        gr.HTML('<span class="presets">Or load an example:</span>')
                        p_safe = gr.Button("small change", elem_classes="link")
                        gr.HTML('<span class="presets" style="margin-left:-10px">,</span>')
                        p_risky = gr.Button("large risky merge", elem_classes="link")
                with gr.Column(elem_id="aside"):
                    result = gr.HTML(EMPTY_HTML)

        # ------------------------------------------------------------ Model / Data / Method
        with gr.Tab("Model", id="model"):
            gr.HTML(MODEL_HTML)
        with gr.Tab("Data", id="data"):
            gr.HTML(DATA_HTML)
        with gr.Tab("Method", id="method"):
            gr.HTML(METHOD_HTML)

    form = [additions, deletions, files_modified, msg_len, is_merge, branch_is_main, event, hour, day,
            repo_prior_runs, repo_fails, repo_prev, wf_has_history, wf_prior_runs, wf_fails, wf_prev]
    # FIELDS order for presets must match `form` order
    preset_order = ["additions", "deletions", "files_modified", "msg_len", "is_merge", "branch_is_main", "event",
                    "hour", "dow", "repo_prior_runs", "repo_failures_last10", "repo_prev", "wf_has_history",
                    "wf_prior_runs", "wf_failures_last10", "wf_prev"]
    assert preset_order == FIELDS

    to_classify.click(lambda: gr.Tabs(selected="classify"), None, tabs)
    button.click(classify, form + [result], [result, err])
    p_safe.click(lambda: load_preset("safe"), None, form)
    p_risky.click(lambda: load_preset("risky"), None, form)
    wf_has_history.change(toggle_workflow, wf_has_history, [wf_prior_runs, wf_fails, wf_prev])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)