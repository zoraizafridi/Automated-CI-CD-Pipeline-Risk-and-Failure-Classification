import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
user = sys.argv[1] if len(sys.argv) > 1 else "zoraizafridi"
model_repo, space_repo = f"{user}/ci-failure-risk", f"{user}/ci-cd-risk-classifier"
api = HfApi()

api.create_repo(model_repo, repo_type="model", exist_ok=True)
for src, dst in [(ROOT / "models/ci_failure_model.joblib", "ci_failure_model.joblib"),
                 (ROOT / "ci_utils.py", "ci_utils.py"),
                 (ROOT / "huggingface/model_card.md", "README.md"),
                 (ROOT / "huggingface/space/requirements.txt", "requirements.txt")]:
    api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=model_repo, repo_type="model")

api.create_repo(space_repo, repo_type="space", space_sdk="gradio", exist_ok=True)
api.add_space_variable(space_repo, "MODEL_REPO", model_repo)
for src, dst in [(ROOT / "huggingface/space/app.py", "app.py"),
                 (ROOT / "huggingface/space/requirements.txt", "requirements.txt"),
                 (ROOT / "huggingface/space/README.md", "README.md"),
                 (ROOT / "ci_utils.py", "ci_utils.py")]:
    api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=space_repo, repo_type="space")

print("Model:", f"https://huggingface.co/{model_repo}")
print("Space:", f"https://huggingface.co/spaces/{space_repo}")