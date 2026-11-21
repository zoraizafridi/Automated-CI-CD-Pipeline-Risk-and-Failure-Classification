# Automated CI/CD Pipeline Risk and Failure Classification

CT-378 Machine Learning semester project.

**Problem.** Predict, before a GitHub Actions run starts, whether it is likely to fail, and assign it a Low, Medium or High risk level. The user is a developer or release manager who wants an early warning before spending CI time or deploying.

**Target.** `label`: 1 if the run failed, 0 if it succeeded. **Task:** binary classification, time-ordered split (70/15/15), 17 features.

**Result.** The selected model is Logistic Regression (tuned with `RandomizedSearchCV` and `TimeSeriesSplit(5)`). On the held-out test set: macro-F1 0.758, failure F1 0.551, ROC-AUC 0.902, PR-AUC 0.576. Observed test failure rate by risk level: Low 1.0%, Medium 6.6%, High 52.0% (overall 6.7%).

## Deployments

| Platform | Link |
|---|---|
| Streamlit Community Cloud | **TODO: add public URL** (main file: `app/app.py`) |
| Hugging Face model + Space | **TODO: add Hub and Space URLs** (see `huggingface/`) |
| Flask (local) | http://127.0.0.1:5000 after `python app/flask_app.py` |

## Repository layout

```
app/app.py                  Streamlit app
app/flask_app.py            Flask backend with /predict, plus app/templates/index.html
huggingface/                Model card, Gradio Space and deploy script
ci_utils.py                 Input validation, feature building, prediction (shared by all apps)
collect_binary_runs.py      Collects our own data with the GitHub REST API
repos.txt                   Repositories targeted by the collection script
data/runs_own.jsonl         Own collected data (hashed identifiers)
notebooks/ci_failure_prediction.ipynb   Full ML lifecycle
models/ci_failure_model.joblib          Final pipeline (joblib)
requirements.txt            Pinned versions
```

## Setup

Python 3.12 is recommended (the pinned numpy and pandas need 3.11 or newer).

```bash
git clone https://github.com/zoraizafridi/Automated-CI-CD-Pipeline-Risk-and-Failure-Classification.git
cd Automated-CI-CD-Pipeline-Risk-and-Failure-Classification
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

Run the apps from the repository root:

```bash
streamlit run app/app.py        # Streamlit
python app/flask_app.py         # Flask, then open http://127.0.0.1:5000
```

Flask API: `POST /predict` with a JSON body (fields as in the form), returns the risk level, score and test failure rates. Invalid input returns HTTP 400 with `{"error": "..."}`. `GET /health` checks the model is loaded.

### Re-training

1. Download GADFPD (see below) and save the CSV as `data/final_research_dataset_MASTER.csv`. It is not committed (it is in `.gitignore`).
2. Run `notebooks/ci_failure_prediction.ipynb` top to bottom. It rewrites `models/ci_failure_model.joblib` and `requirements.txt`.
3. To collect more of our own data: set the `GITHUB_TOKEN` environment variable, then run
   `python collect_binary_runs.py repos.txt <start-date> <end-date> data/runs_own.jsonl`.
   The script skips run IDs it already has.

## Data

| | Source 1: GADFPD | Source 2: own collection |
|---|---|---|
| What | GitHub Actions CI/CD Failure Prediction Dataset | Workflow runs and commit statistics from the GitHub REST API |
| Reference | Faruna, J., et al. (2026). Mendeley Data, V1. DOI [10.17632/mggwn7rj9f.1](https://doi.org/10.17632/mggwn7rj9f.1) | GitHub REST API (`actions/runs` and `commits` endpoints) |
| Licence | CC BY 4.0 | Public repository metadata, collected within the GitHub API terms |
| Date | Published 30 June 2026 | Collected 2026-10-20 to 2026-11-20 |
| Size | See the notebook (`raw.shape` and the cleaning steps) | 8,107 runs, 14 repositories, 12.5% failures |

- **Personal identifiers removed.** Repository and actor names are replaced by a salted SHA-256 hash. No commit message text is stored, only its length.
- **Repositories.** `repos.txt` lists 16 repositories. The data covers 14, because `kubernetes/kubernetes` and `golang/go` returned no runs (most likely they do not use GitHub Actions for CI; this was not verified).
- **Sampling.** The collection script samples runs across days (up to 600 per repository), so the "last 10 runs" history of our own runs is built from sampled rather than strictly consecutive runs.
- **Time coverage.** The two sources cover different periods, so the time-ordered test set is later than most of the training data and may show drift.

## Limitations

Failures are rare (about 7 to 12% of runs), so some alerts are false alarms. The model risk score ranks runs by risk and is not a literal probability, because the model was trained with class weights. Most of the signal comes from run history, so a brand-new repository or workflow is predicted mainly from change size and trigger.

## References

- Faruna, J., et al. (2026). *A comprehensive GitHub Actions dataset for CI/CD pipeline failure forecasting.* Mendeley Data, V1. https://doi.org/10.17632/mggwn7rj9f.1
- GitHub REST API documentation, Actions workflow runs. https://docs.github.com/en/rest/actions/workflow-runs
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR* 12, 2825-2830.