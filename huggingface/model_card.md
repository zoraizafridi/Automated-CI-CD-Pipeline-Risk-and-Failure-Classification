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
It ranks runs for triage. It is **not** a literal probability of failure, because the model was trained with class weights.

## Files
- `ci_failure_model.joblib`: a dict with the fitted scikit-learn `Pipeline` (`pipeline`), the decision `threshold`, `features`, `risk_bands`, `risk_rates` and `test_metrics`.
- `ci_utils.py`: input validation, feature building and the `log1p_clip` function the pipeline needs to unpickle.

## How to use
```python
import joblib
from huggingface_hub import hf_hub_download

repo = "zoraizafridi/ci-failure-risk"
bundle = joblib.load(hf_hub_download(repo, "ci_failure_model.joblib"))
# also download ci_utils.py next to your script, then:
from ci_utils import predict
print(predict(bundle, {
    "additions": 40, "deletions": 10, "files_modified": 3, "msg_len": 60,
    "is_merge": False, "branch_is_main": True, "event": "push", "hour": 12, "dow": 2,
    "repo_prior_runs": 200, "repo_prev": "success", "repo_failures_last10": 1,
    "wf_has_history": True, "wf_prior_runs": 100, "wf_prev": "success", "wf_failures_last10": 1,
}))
```
Requires `scikit-learn==1.9.1`, `numpy==2.5.4`, `pandas==3.0.6`, `joblib==1.6.0` (Python 3.12). Other versions may fail to unpickle.

## Training data
1. **GADFPD**, GitHub Actions CI/CD Failure Prediction Dataset, Faruna, J. et al. (2026), Mendeley Data V1, DOI 10.17632/mggwn7rj9f.1, licence CC BY 4.0, published 30 June 2026.
2. **Own collection**: 8,107 workflow runs from 14 public repositories, 2026-10-20 to 2026-11-20, collected with the GitHub REST API (`collect_binary_runs.py`). Repository and actor names are hashed; no commit message text is stored.

Only runs with a clear success or failure outcome and a first attempt are kept. Runs are sorted by time and split 70/15/15 into train, validation and test.

## Model
Selected by validation macro-F1 from six tuned algorithms (Logistic Regression, Decision Tree, KNN, Linear SVM, Random Forest, Histogram Gradient Boosting) using `RandomizedSearchCV` with `TimeSeriesSplit(5)`.
Selected model: Logistic Regression. 17 features: change size, merge flag, hour, weekday, branch, trigger event and run history (previous outcome, failure rate over the last 10 runs, number of earlier runs) per repository and per workflow.

## Evaluation (held-out, time-ordered test set, scored once)
| macro-F1 | failure F1 | failure precision | failure recall | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.758 | 0.551 | 0.520 | 0.586 | 0.902 | 0.576 |

Observed test failure rate by risk level: Low 1.0%, Medium 6.6%, High 52.0% (overall 6.7%).

## Limitations
- Failures are rare, so some alerts are false alarms.
- Most of the signal comes from run history; a brand new repository or workflow is predicted mainly from change size and trigger.
- The two data sources cover different periods (GADFPD was published 30 June 2026; own runs are from 20 Oct to 20 Nov 2026), and repositories change over time, so the model can drift and should be retrained periodically.
- Own-collected runs were sampled across days, so their "last 10 runs" history is not strictly consecutive.
- Two of the 16 targeted repositories (kubernetes/kubernetes, golang/go) returned no runs and are not in the data.