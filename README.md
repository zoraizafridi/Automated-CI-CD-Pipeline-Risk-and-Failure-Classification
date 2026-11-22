---
library_name: sklearn
tags:
  - sklearn
  - tabular-classification
  - ci-cd
  - github-actions
metrics:
  - f1
  - roc_auc
---

# CI/CD run failure risk (scikit-learn pipeline)

Predicts, before a GitHub Actions run starts, whether it is likely to fail, and maps the score to a **Low / Medium / High** risk level.
Built for CT-378 Machine Learning (semester project). Live demo: see the Space linked from the project README.

## Intended use
A developer or release manager wants an early warning before spending CI time or deploying.
The class-weighted model output is calibrated with sigmoid (Platt) scaling on the time-ordered validation split before it is shown as an estimated failure probability. This is an estimate, not a guarantee; calibration can drift as repositories and workflows change.

## Files
- `ci_failure_model.joblib`: a dict with the fitted scikit-learn `Pipeline` (`pipeline`), the decision `threshold`, `features`, `risk_bands`, `risk_rates` and `test_metrics`.
- `ci_utils.py`: input validation, feature building and the `log1p_clip` function the pipeline needs to unpickle.

## How to use
```python
import joblib
from huggingface_hub import hf_hub_download

repo = "zoiiiizxk/ci-failure-risk"
bundle = joblib.load(hf_hub_download(repo, "ci_failure_model.joblib"))
# also download ci_utils.py next to your script, then:
from ci_utils import predict

inputs = {
    "additions": 40, "deletions": 10, "files_modified": 3, "msg_len": 60,
    "is_merge": False, "branch_is_main": True, "event": "push", "hour": 12, "dow": 2,
    "repo_prior_runs": 200, "repo_prev": "success", "repo_failures_last10": 1,
    "wf_has_history": True, "wf_prior_runs": 100, "wf_prev": "success", "wf_failures_last10": 1,
}
result = predict(bundle, inputs)  # applies Platt calibration, threshold and risk bands
print(f"Estimated failure probability: {result['probability']:.1%} ({result['risk_level']} risk)")
```
Requires `scikit-learn==1.9.1`, `numpy==2.5.4`, `pandas==3.0.6`, `joblib==1.6.0` (Python 3.12). Other versions may fail to unpickle.

## Training data
1. **GADFPD**, GitHub Actions CI/CD Failure Prediction Dataset, Faruna, J. et al. (2026), Mendeley Data V1, DOI 10.17632/mggwn7rj9f.1, licence CC BY 4.0, published 30 June 2026.
2. **Own collection**: 23,818 workflow runs from 14 public repositories, 2026-04-19 to 2026-11-20 (starting the day after GADFPD ends), collected with the GitHub REST API (`collect_binary_runs.py`). Repository and actor names are hashed; no commit message text is stored.

Only runs with a clear success or failure outcome and a first attempt are kept. Runs are sorted by time and split 70/15/15 into train, validation and test.

## Model
Selected by validation macro-F1 from six tuned algorithms (Logistic Regression, Decision Tree, KNN, Linear SVM, Random Forest, Histogram Gradient Boosting) using `RandomizedSearchCV` with `TimeSeriesSplit(5)`.
Selected model: Hist Gradient Boosting (class-weighted). 17 features: change size, merge flag, hour, weekday, branch, trigger event and run history (previous outcome, failure rate over the last 10 runs, number of earlier runs) per repository and per workflow.

## Evaluation (held-out, time-ordered test set, scored once)
| macro-F1 | failure F1 | failure precision | failure recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.690 | 0.449 | 0.540 | 0.384 | 0.845 | 0.504 |

The notebook regenerates the calibrated decision threshold, held-out test metrics and observed test failure rates by risk level whenever the model is retrained. The values in the model artifact are authoritative for the deployed version.

## Limitations
- Failures are fairly rare, so the selected threshold can produce both false alarms and missed failures. Check the regenerated held-out test metrics in the model artifact.
- Most of the signal comes from run history; a brand new repository or workflow is predicted mainly from change size and trigger.
- GADFPD covers 2016-08-08 to 2026-04-18 and the own runs cover 2026-04-19 to 2026-11-20. The failure rate differs between the two, and repositories change over time, so the model can drift and should be retrained periodically.
- Own-collected runs were sampled across days, so their "last 10 runs" history is not strictly consecutive.
- Two of the 16 targeted repositories are not in the data: kubernetes/kubernetes returned no runs, and golang/go returned only 7, which were removed.