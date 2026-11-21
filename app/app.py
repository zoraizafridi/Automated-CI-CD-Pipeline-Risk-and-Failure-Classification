import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import streamlit as st
import streamlit.components.v1 as components

from ci_utils import event_choices, predict

st.set_page_config(
    page_title="CI Risk · CI/CD Risk and Failure Classification",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LEVELS = ["Low", "Medium", "High"]
COLOR = {"Low": "#35e0a1", "Medium": "#ffc857", "High": "#ff5c7a"}
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PAGES = ["Overview", "Classify", "Model", "Data", "Method"]

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

DEFAULTS = dict(additions=40, deletions=10, files_modified=3, msg_len=60, is_merge=False, branch_is_main=True,
                event="push", hour=12, dow="Wednesday", repo_prior_runs=200, repo_failures_last10=1,
                repo_prev="success", wf_has_history=True, wf_prior_runs=100, wf_failures_last10=1, wf_prev="success")
PRESETS = {
    "safe": dict(additions=12, deletions=4, files_modified=1, msg_len=48, is_merge=False, branch_is_main=False,
                 hour=10, dow="Tuesday", repo_prior_runs=400, repo_failures_last10=0, repo_prev="success",
                 wf_has_history=True, wf_prior_runs=150, wf_failures_last10=0, wf_prev="success"),
    "risky": dict(additions=900, deletions=350, files_modified=34, msg_len=12, is_merge=True, branch_is_main=True,
                  hour=2, dow="Saturday", repo_prior_runs=60, repo_failures_last10=6, repo_prev="failure",
                  wf_has_history=True, wf_prior_runs=25, wf_failures_last10=5, wf_prev="failure"),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&display=swap');
:root{--bg:#050505;--panel:#0b0612;--field:#12091f;--line:#2a1347;--amethyst:#3B0A6B;--ink:#f3eefc;--muted:#a99bc0;--accent:#00E5FF;--accent-ink:#050505;
--low:#35e0a1;--med:#ffc857;--high:#ff5c7a;--lit:#a66bff;--serif:"Cormorant Garamond","Playfair Display","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
--sans:"Segoe UI Variable","Segoe UI",system-ui,-apple-system,Roboto,sans-serif}

/* ---- chrome removal + page background ---- */
html,body{background:var(--bg)!important;overflow-x:hidden}
header[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],#MainMenu,footer,[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stSidebarCollapsedControl"]{display:none!important}
.stApp{background:var(--bg)!important;color:var(--ink);font-size:15px;line-height:1.55;-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
.stApp::before{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
background:radial-gradient(760px 560px at 84% 4%,rgba(59,10,107,.95),transparent 68%),radial-gradient(380px 280px at 97% 20%,rgba(0,229,255,.17),transparent 70%),radial-gradient(660px 440px at -4% 102%,rgba(59,10,107,.6),transparent 70%)}
[data-testid="stMain"],[data-testid="stMainBlockContainer"]{position:relative;z-index:1;background:transparent!important}
.stApp :where(p,label,span,div,button,input,li,td,th,a,h1,h2,h3,h4,h5,textarea,select):not([data-testid="stIconMaterial"]){font-family:var(--sans)}
.stApp :where(h1,h2,h3,em,.serif):not([data-testid="stIconMaterial"]){font-family:var(--serif)!important}
.block-container,[data-testid="stMainBlockContainer"]{max-width:1120px!important;padding:0 24px 96px!important}
[data-testid="stVerticalBlock"]{gap:0}
[data-testid="stElementContainer"]:has(iframe[height="0"]){display:none}
a{color:inherit}
:focus-visible{outline:2px solid var(--accent)!important;outline-offset:3px}

/* ---- top navigation (the five st.tabs) ---- */
[data-testid="stTabs"],[data-testid="stTabs"]>div{overflow:visible!important}
[role="tablist"]{position:sticky;top:0;z-index:50;height:60px;gap:4px;align-items:center;padding:0!important;overflow:visible!important;background:transparent!important;border:0!important;box-shadow:none!important;flex-wrap:nowrap}
[role="tablist"]::after{content:"";position:absolute!important;z-index:-1;top:0!important;bottom:auto!important;height:100%!important;left:50%;width:100vw!important;transform:translateX(-50%);
background:color-mix(in srgb,#050505 88%,transparent);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
[role="tablist"]::before{content:"CI RISK";display:flex;align-items:center;flex:none;height:60px;margin-right:32px;padding-left:40px;white-space:nowrap;
font-weight:700;letter-spacing:.14em;font-size:.82rem;color:var(--ink);cursor:pointer;
background:radial-gradient(circle at 5px 50%,#35e0a1 4px,transparent 4.6px) 0 50%/30px 14px no-repeat,radial-gradient(circle at 15px 50%,#ffc857 4px,transparent 4.6px) 0 50%/30px 14px no-repeat,radial-gradient(circle at 25px 50%,#ff5c7a 4px,transparent 4.6px) 0 50%/30px 14px no-repeat,linear-gradient(rgba(243,238,252,.35),rgba(243,238,252,.35)) 5px 50%/20px 2px no-repeat}
[data-testid="stTab"]{position:relative;height:60px!important;padding:0 14px!important;margin:0!important;background:transparent!important;color:var(--muted)!important;border:0!important;display:flex;align-items:center;flex:none;cursor:pointer}
[data-testid="stTab"] p,[data-testid="stTab"] div{font-size:.7rem!important;letter-spacing:.18em;text-transform:uppercase;white-space:nowrap;color:inherit!important;margin:0!important;font-weight:400!important}
[data-testid="stTab"]:hover,[data-testid="stTab"][aria-selected="true"]{color:var(--ink)!important}
[data-testid="stTab"] .react-aria-SelectionIndicator,[role="tablist"] .react-aria-SelectionIndicator{position:absolute!important;left:12px!important;right:12px!important;bottom:0!important;top:auto!important;width:auto!important;height:2px!important;background:var(--accent)!important;border:0!important}
[role="tabpanel"]{padding:48px 0 0!important}
[data-testid="stTabsScrollRight"],[data-testid="stTabsScrollLeft"]{display:none!important}
[data-testid="stHeaderActionElements"]{display:none!important}
[data-testid="stHeadingWithActionElements"]{padding:0!important;margin:0!important}
.ci h1 span,.ci h2 span,.ci h3 span{font:inherit!important;letter-spacing:inherit!important;color:inherit!important}

/* ---- shared typography (.ci scope) ---- */
.ci h1{font-family:var(--serif)!important;font-size:clamp(3rem,7vw,5.6rem)!important;line-height:.98!important;letter-spacing:-.02em!important;font-weight:400!important;margin:0 0 22px!important;max-width:11ch;padding:0!important;color:var(--ink)}
.ci h1 em{font-style:italic;color:var(--lit)}
.ci h2{font-family:var(--serif)!important;font-size:2.2rem!important;letter-spacing:-.01em;font-weight:500!important;margin:0 0 8px!important;padding:0!important;line-height:1.2;color:var(--ink)}
.ci h3{font-family:var(--serif)!important;padding:0!important;color:var(--ink)}
.ci p,.ci li{font-size:15px;line-height:1.55;margin:0;color:var(--muted)}
.ci .lead{font-size:1.1rem;color:var(--muted);max-width:44ch;margin:0 0 32px}
.ci a{text-decoration:none}
.ci .btn{display:inline-block;background:var(--accent);color:var(--accent-ink)!important;border:1px solid var(--accent);border-radius:99px;padding:13px 28px;font-weight:600;cursor:pointer;box-shadow:0 0 28px rgba(0,229,255,.28);transition:box-shadow .2s}
.ci .btn:hover{box-shadow:0 0 40px rgba(0,229,255,.5)}
.ci .prose{max-width:68ch}.ci .prose p,.ci .prose li{color:var(--muted)}.ci .prose ul{padding-left:18px;margin:0}.ci .prose strong{color:var(--ink);font-weight:600}
.ci .prose p+p{margin-top:1em}.ci .prose li{margin:0 0 .5em}
.ci .prose h3{font-size:1.45rem;font-weight:500;margin:0 0 6px!important}

/* ---- overview ---- */
.ci .hero{position:relative;min-height:540px;padding-top:36px}
.ci .hero .copy{position:relative;z-index:1}
.ci .ribbons{overflow:visible;position:absolute;right:-60px;top:-30px;width:min(680px,62%);height:auto;pointer-events:none}
.ci .ribbons .draw{stroke-dasharray:1400;stroke-dashoffset:1400;animation:draw 2.4s .15s cubic-bezier(.2,.7,.2,1) forwards}
@keyframes draw{to{stroke-dashoffset:0}}
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
[class*="st-key-grp_"]{border-top:1px solid var(--line);padding:26px 0}
[class*="st-key-grp_"] [data-testid="stHorizontalBlock"]{gap:28px}
[class*="st-key-grp_"] h3{margin:0!important;font-size:1.35rem;font-weight:500;line-height:1.2;color:var(--ink)}
[class*="st-key-grp_"] .gp{margin:2px 0 0;color:var(--muted);font-size:.87rem}
[class*="st-key-grp_"] [data-testid="stHorizontalBlock"] [data-testid="stHorizontalBlock"]{gap:16px}
[data-testid="stColumn"]{min-width:0}
[data-testid="stWidgetLabel"],[data-testid="stWidgetLabel"] p,label[data-testid="stWidgetLabel"] *{color:var(--muted)!important;font-size:.85rem!important}
[data-testid="stWidgetLabel"]{margin-bottom:0!important}
[data-testid="stElementContainer"]{margin-bottom:0}
[class*="st-key-grp_"] [data-testid="stElementContainer"]{margin-bottom:16px}
[data-testid="stNumberInput"] [data-baseweb="input"],[data-testid="stNumberInput"] [data-baseweb="base-input"],[data-baseweb="select"]>div{background:var(--field)!important;border-radius:8px!important}
[data-testid="stNumberInput"] [data-baseweb="input"],[data-baseweb="select"]>div{border:1px solid var(--line)!important;margin-top:6px}
[data-testid="stNumberInput"] [data-baseweb="base-input"]{border:0!important}
[data-testid="stNumberInput"] input,[data-baseweb="select"] div{color:var(--ink)!important;font-size:.95rem}
[data-testid="stNumberInput"] input{padding:10px 12px!important;background:transparent!important}
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,[data-baseweb="select"]>div:focus-within{border-color:var(--accent)!important;box-shadow:0 0 0 3px rgba(0,229,255,.15)!important}
[data-testid="stNumberInputStepUp"],[data-testid="stNumberInputStepDown"]{display:none!important}
[data-baseweb="select"] svg{fill:var(--muted)}
[data-baseweb="popover"] [data-baseweb="menu"],[data-baseweb="popover"] ul{background:var(--field)!important}
[data-baseweb="popover"] li{color:var(--ink)!important}
[data-testid="stCheckbox"] label{align-items:center;gap:10px}
[data-testid="stCheckbox"] p{color:var(--ink)!important;font-size:.92rem!important}
.st-key-tg_merge,.st-key-tg_main{padding-top:22px}
.st-key-wf_toggle{padding-top:0}
.st-key-wf_fields_off{opacity:.45}
.st-key-wf_fields,.st-key-wf_fields_off{transition:opacity .2s}
/* segmented radios */
[data-testid="stRadioGroup"]{display:inline-flex!important;flex-direction:row!important;flex-wrap:nowrap;gap:0!important;margin-top:6px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--field)}
[data-testid="stRadioGroup"]>div{margin:0!important;padding:0!important}
[data-testid="stRadioOption"]{display:block!important;margin:0!important;padding:9px 16px!important;cursor:pointer;background:transparent}
[data-testid="stRadioOption"]>div>div:first-child{display:none!important}
[data-testid="stRadioOption"] p{font-size:.9rem!important;color:var(--muted)!important;margin:0!important}
[data-testid="stRadioOption"][data-selected="true"]{background:var(--amethyst)!important;box-shadow:inset 0 0 0 1px var(--accent)}
[data-testid="stRadioOption"][data-selected="true"] p{color:var(--ink)!important;font-weight:600!important}
/* actions row */
.st-key-actions{border-top:1px solid var(--line);padding-top:24px;align-items:center;flex-wrap:wrap;gap:0 6px!important}
.st-key-actions>[data-testid="stElementContainer"]:nth-child(1){margin-right:14px!important}
.st-key-actions>[data-testid="stElementContainer"]:nth-child(4){margin-left:-6px!important}
.st-key-actions [data-testid="stElementContainer"]{margin:0;width:auto!important}
.st-key-actions p{color:var(--muted);font-size:.9rem}
button[kind="primary"],[data-testid="stBaseButton-primary"]{background:var(--accent)!important;color:var(--accent-ink)!important;border:1px solid var(--accent)!important;border-radius:99px!important;padding:13px 28px!important;height:auto!important;min-height:0!important;font-weight:600;box-shadow:0 0 28px rgba(0,229,255,.28);transition:box-shadow .2s}
button[kind="primary"]:hover,[data-testid="stBaseButton-primary"]:hover{box-shadow:0 0 40px rgba(0,229,255,.5)}
button[kind="primary"] p,[data-testid="stBaseButton-primary"] p{color:var(--accent-ink)!important;font-size:15px;font-weight:600}
[data-testid="stBaseButton-tertiary"],button[kind="tertiary"]{color:var(--accent)!important;padding:0!important;min-height:0!important;height:auto!important;background:none!important;border:0!important;text-decoration:underline;text-underline-offset:3px}
[data-testid="stBaseButton-tertiary"] p,button[kind="tertiary"] p{color:var(--accent)!important;font-size:.9rem}
/* error */
.ci .err{margin-bottom:18px;padding:12px 14px;border-left:3px solid var(--high);background:color-mix(in srgb,var(--high) 10%,var(--panel));border-radius:0 8px 8px 0;color:var(--ink)}

/* ---- classify: result panel ---- */
[data-testid="stColumn"]:has(.st-key-aside){position:sticky;top:84px;align-self:flex-start}
.st-key-aside{border:1px solid rgba(0,229,255,.22);background:linear-gradient(165deg,rgba(59,10,107,.45),rgba(5,5,5,.75) 60%);border-radius:16px;padding:26px;box-shadow:0 0 60px rgba(59,10,107,.35)}
.ci .empty{color:var(--muted)}.ci .empty strong{display:block;color:var(--ink);font-size:1.1rem;margin-bottom:4px;font-weight:700}
.ci .verdict,.ci .verdict span{font-family:var(--serif)!important;font-size:2.5rem;line-height:1.05;font-weight:500;letter-spacing:-.01em;margin:0;display:flex;align-items:center;gap:12px;color:var(--ink)}
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
.ci table{border-collapse:collapse;width:100%;max-width:760px;margin:20px 0 40px;background:transparent}
.ci th,.ci td{text-align:left;padding:14px 16px 14px 0;border:0;border-bottom:1px solid var(--line);vertical-align:top;background:transparent;color:var(--ink)}
.ci tr{border:0!important;background:transparent!important}
.ci td.d{color:var(--muted)}
.ci th{font-weight:500;font-size:.85rem;color:var(--muted)}
.ci td.v{font-size:1.3rem;font-weight:500;white-space:nowrap;color:var(--ink)}
.ci .split{display:grid;grid-template-columns:1fr 1fr;gap:48px}
@media (max-width:900px){
 [data-testid="stColumn"]:has(.st-key-aside){position:static}
 .ci .facts,.ci .split{grid-template-columns:1fr}
 [class*="st-key-grp_"] [data-testid="stHorizontalBlock"]{gap:8px}
 [role="tablist"]::before{display:none}
}
@media (max-width:760px){.ci .ribbons{opacity:.45;width:110%;right:-30%}.ci .facts{row-gap:20px}.ci .ladder{grid-template-columns:1fr;row-gap:6px}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}.ci .ribbons .draw{animation:none;stroke-dashoffset:0}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models" / "ci_failure_model.joblib")


bundle = load_bundle()
RATES = bundle["risk_rates"]
METRICS = bundle.get("test_metrics", {})
THRESHOLD = float(bundle["threshold"])
MODEL = bundle["model_name"]
N_FEAT = len(bundle["features"])
EVENTS = event_choices(bundle)


def pct(x, d=0):
    return f"{x * 100:.{d}f}%"


for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("result", None)


def load_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# bridge: lets the in-page links (Classify a run, brand) switch the top navigation tabs
components.html(
    """
<script>
(function(){
  const P = window.parent, D = P.document;
  const NAMES = ["overview","classify","model","data","method"];
  function go(i){
    const t = D.querySelectorAll('[data-testid="stTab"]');
    if (t[i]) t[i].click();
    const m = D.querySelector('[data-testid="stMain"]'); if (m) m.scrollTo(0,0); P.scrollTo(0,0);
  }
  if (!P.__ciNav){
    P.__ciNav = true;
    D.addEventListener("click", function(e){
      const a = e.target.closest && e.target.closest('a[href^="#"]');
      if (a){ const i = NAMES.indexOf(a.getAttribute("href").slice(1)); if (i >= 0){ e.preventDefault(); go(i); } return; }
      const tl = e.target.closest && e.target.closest('[role="tablist"]');
      if (tl && e.target === tl && e.clientX - tl.getBoundingClientRect().left < 140) go(0);
    }, true);
  }
  const h = NAMES.indexOf((P.location.hash || "").slice(1));
  if (h > 0 && !P.__ciHash){ P.__ciHash = true; setTimeout(function(){ go(h); }, 150); }
})();
</script>
""",
    height=0,
)

tabs = st.tabs(PAGES)

# ---------------------------------------------------------------- Overview
RIBBONS = """<svg class="ribbons" viewBox="0 0 640 560" fill="none" aria-hidden="true"><defs><linearGradient id="rg" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#3B0A6B"/><stop offset=".55" stop-color="#a66bff"/><stop offset="1" stop-color="#00E5FF"/></linearGradient><filter id="rb" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="16"/></filter></defs><g filter="url(#rb)" opacity=".55" stroke="url(#rg)" stroke-width="22" stroke-linecap="round"><path d="M40 520C170 470 240 290 350 220S540 90 620 40"/><path d="M0 400C150 410 260 320 310 220S450 100 600 140"/></g><g stroke="url(#rg)" stroke-linecap="round"><path class="draw" stroke-width="3" d="M40 520C170 470 240 290 350 220S540 90 620 40"/><path class="draw" stroke-width="2" d="M0 400C150 410 260 320 310 220S450 100 600 140"/><path class="draw" stroke-width="1.5" d="M90 560C220 520 350 470 420 350S520 220 630 260"/><path class="draw" stroke-width="1" d="M0 300C120 340 230 260 290 150S410 30 480 0"/></g><g fill="#00E5FF"><circle cx="350" cy="220" r="5"/><circle cx="310" cy="220" r="3.5" fill="#a66bff"/><circle cx="420" cy="350" r="4"/><circle cx="600" cy="140" r="3.5"/></g></svg>"""

ladder = "".join(
    f'<div class="rung"><span class="lab"><i style="background:{COLOR[l]}"></i>{l} risk</span><b>{pct(RATES[l], 1)}</b>'
    f'<div class="trk"><i style="width:{RATES[l] * 100}%;background:{COLOR[l]}"></i></div>'
    f'<span class="cap">of runs failed in test data</span></div>'
    for l in LEVELS
)
fail_rec = pct(METRICS.get("fail_rec", 0) or 0)

with tabs[0]:
    st.markdown(
        f'<div class="ci"><div class="hero">{RIBBONS}<div class="copy">'
        '<h1>Will this run <em>break</em> the build?</h1>'
        '<p class="lead">Describe a GitHub Actions run before it starts. The model classifies it as likely to fail '
        'or succeed and rates the risk as Low, Medium or High.</p>'
        '<a class="btn" href="#classify">Classify a run</a></div></div>'
        f'<div class="ladder">{ladder}</div>'
        f'<p class="ladder-cap">Observed in held-out test data, where {pct(RATES["overall"], 1)} of all runs failed. '
        'Bars show each rate out of 100%.</p>'
        '<div class="facts">'
        '<div><span class="n">01</span><h3>What you enter</h3><p>The size of the change, what triggers the run, and how the repository and workflow have behaved lately.</p></div>'
        '<div><span class="n">02</span><h3>What you get</h3><p>A verdict, a risk level and a score, shown against the threshold at which runs are flagged.</p></div>'
        f'<div><span class="n">03</span><h3>How far to trust it</h3><p>It ranks runs for triage. It catches roughly {fail_rec} of failures, so treat it as an early warning, not a guarantee.</p></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- Classify
def group_head(title, text):
    return f'<div class="ci"><h3>{title}</h3><p class="gp">{text}</p></div>'


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
        f'<div class="kv"><div><span>Model risk score</span><strong>{pct(o["probability"])}</strong></div>'
        f'<div><span>Flagged at</span><strong>{pct(o["threshold"])}</strong></div></div>'
        f'<p class="sub">{text}</p>{bars}'
        f'<p class="note">The score ranks runs by risk. It is not a literal chance of failure, because the model was trained '
        f'with class weights. Runs scoring {pct(o["threshold"])} or higher are flagged. Model: {o["model"]}.</p></div>'
    )


EMPTY = '<div class="ci"><div class="empty"><strong>No result yet</strong>The verdict and risk score appear here.</div></div>'

with tabs[1]:
    st.markdown(
        '<div class="ci"><h2>Classify a run</h2>'
        '<p class="lead" style="margin-bottom:32px">Fill in what you know about the run, then select Classify run.</p></div>',
        unsafe_allow_html=True,
    )
    form_col, aside_col = st.columns([646, 370], gap="large")

    with form_col:
        err_slot = st.empty()

        # Example presets remain outside the form because Streamlit does not
        # allow ordinary callback buttons inside an st.form block.
        with st.container(key="actions", horizontal=True, horizontal_alignment="left", vertical_alignment="center", gap="small"):
            st.markdown("Load an example:")
            st.button("small change", type="tertiary", on_click=load_preset, args=("safe",))
            st.markdown(",")
            st.button("large risky merge", type="tertiary", on_click=load_preset, args=("risky",))

        with st.form("ci_prediction_form"):
            with st.container(key="grp_change"):
                lc, rc = st.columns([170, 448])
                lc.markdown(group_head("Change", "What the commit touches."), unsafe_allow_html=True)
                with rc:
                    a, b = st.columns(2)
                    a.number_input("Lines added", min_value=0, step=1, key="additions")
                    b.number_input("Lines deleted", min_value=0, step=1, key="deletions")
                    a, b = st.columns(2)
                    a.number_input("Files changed", min_value=0, step=1, key="files_modified")
                    b.number_input("Commit message length (characters)", min_value=0, step=1, key="msg_len")
                    a, b = st.columns(2)
                    with a.container(key="tg_merge"):
                        st.checkbox("Merge commit", key="is_merge")
                    with b.container(key="tg_main"):
                        st.checkbox("Default branch (main or master)", key="branch_is_main")

            with st.container(key="grp_trigger"):
                lc, rc = st.columns([170, 448])
                lc.markdown(group_head("Trigger", "What starts the run, and when."), unsafe_allow_html=True)
                with rc:
                    a, b = st.columns(2)
                    a.selectbox("Event", EVENTS, key="event")
                    b.number_input("Hour (UTC)", min_value=0, max_value=23, step=1, key="hour")
                    a, b = st.columns(2)
                    a.selectbox("Day of week", DAYS, key="dow")

            with st.container(key="grp_repo"):
                lc, rc = st.columns([170, 448])
                lc.markdown(group_head("Repository history", "How the repository behaved recently."), unsafe_allow_html=True)
                with rc:
                    a, b = st.columns(2)
                    a.number_input("Earlier runs in the repository", min_value=0, step=1, key="repo_prior_runs")
                    b.number_input("Failures in the last 10 runs", min_value=0, max_value=10, step=1, key="repo_failures_last10")
                    st.radio("Outcome of the previous run", ["success", "failure"], horizontal=True, key="repo_prev")

            with st.container(key="grp_workflow"):
                lc, rc = st.columns([170, 448])
                lc.markdown(group_head("Workflow history", "The same, for this workflow only."), unsafe_allow_html=True)
                with rc:
                    with st.container(key="wf_toggle"):
                        st.checkbox("This workflow has earlier runs", key="wf_has_history")
                    has_wf = st.session_state["wf_has_history"]
                    with st.container(key="wf_fields" if has_wf else "wf_fields_off"):
                        a, b = st.columns(2)
                        a.number_input("Earlier runs of this workflow", min_value=1, step=1, key="wf_prior_runs", disabled=not has_wf)
                        b.number_input("Workflow failures in the last 10 runs", min_value=0, max_value=10, step=1,
                                       key="wf_failures_last10", disabled=not has_wf)
                        st.radio("Outcome of the workflow's previous run", ["success", "failure"], horizontal=True,
                                 key="wf_prev", disabled=not has_wf)

            go = st.form_submit_button("Classify run", type="primary")

    if go:
        s = st.session_state
        has_wf = s["wf_has_history"]
        inputs = {
            "additions": s["additions"], "deletions": s["deletions"], "files_modified": s["files_modified"],
            "msg_len": s["msg_len"], "is_merge": s["is_merge"], "branch_is_main": s["branch_is_main"],
            "event": s["event"], "hour": s["hour"], "dow": DAYS.index(s["dow"]),
            "repo_prior_runs": s["repo_prior_runs"], "repo_prev": s["repo_prev"],
            "repo_failures_last10": s["repo_failures_last10"], "wf_has_history": has_wf,
            "wf_prior_runs": s["wf_prior_runs"] if has_wf else 0,
            "wf_prev": s["wf_prev"] if has_wf else "success",
            "wf_failures_last10": s["wf_failures_last10"] if has_wf else 0,
        }
        try:
            s["result"] = {"ok": predict(bundle, inputs)}
        except ValueError as e:
            s["result"] = {"error": str(e)}

    res = st.session_state["result"]
    if res and "error" in res:
        err_slot.markdown(f'<div class="ci"><div class="err" role="alert">{res["error"]}</div></div>', unsafe_allow_html=True)

    with aside_col:
        with st.container(key="aside"):
            if res and "ok" in res:
                st.markdown(result_html(res["ok"]), unsafe_allow_html=True)
            else:
                st.markdown(EMPTY, unsafe_allow_html=True)

# ---------------------------------------------------------------- Model
rows = "".join(
    f'<tr><td>{NAMES.get(k, k)}</td><td class="v">{float(v):.3f}</td><td class="d">{MEANS.get(k, "")}</td></tr>'
    for k, v in METRICS.items()
)
mbars = "".join(
    f'<div class="bar sel" style="--c:{COLOR[l]};grid-template-columns:70px 1fr 56px"><span>{l}</span>'
    f'<div class="bt"><div class="fill" style="width:{RATES[l] / RATES["High"] * 100}%"></div></div><span>{pct(RATES[l], 1)}</span></div>'
    for l in LEVELS
)
with tabs[2]:
    st.markdown(
        f'<div class="ci"><div class="prose"><h2>How well it works</h2>'
        f'<p>{MODEL}, scored once on a held-out test set that comes later in time than the training data.</p></div>'
        '<table><thead><tr><th>Measure</th><th>Result</th><th>What it means</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
        '<div class="prose"><h2>Failure rate by risk level</h2>'
        f'<p>Share of test runs that failed at each level. Runs scoring {pct(THRESHOLD)} or higher are flagged as likely to fail.</p></div>'
        f'<div style="max-width:560px;margin-top:16px">{mbars}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- Data
with tabs[3]:
    st.markdown(
        '<div class="ci"><div class="prose"><h2>Where the data comes from</h2>'
        '<p>Two sources of GitHub Actions runs, joined end to end in time.</p></div>'
        '<div class="split" style="margin-top:28px">'
        '<div class="prose"><h3>GADFPD</h3><p>GitHub Actions CI/CD Failure Prediction Dataset by Faruna et al. (2026), '
        'Mendeley Data, DOI 10.17632/mggwn7rj9f.1. Licence CC BY 4.0. Covers runs from 8 August 2016 to 18 April 2026.</p></div>'
        '<div class="prose"><h3>Own collection</h3><p>23,818 workflow runs from 14 public repositories, 19 April to 20 November 2026, '
        'collected with the GitHub REST API. Repository and actor names are hashed and no commit message text is stored.</p></div></div>'
        '<div class="prose" style="margin-top:36px"><p>Only runs with a clear success or failure outcome and a first attempt are kept. '
        'Runs are sorted by time and split 70% train, 15% validation and 15% test, so the test set is always the most recent data.</p></div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------- Method
with tabs[4]:
    st.markdown(
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
        '<li>The score ranks runs. It is not a literal probability of failure, because of the class weights.</li>'
        '</ul></div></div>',
        unsafe_allow_html=True,
    )