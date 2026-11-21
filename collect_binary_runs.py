"""Collect GitHub Actions runs (success vs failure) with pre-run commit features.

Usage:
    python collect_binary_runs.py repos.txt 2026-10-20 2026-11-20 runs_own.jsonl [max_per_repo]

Needs the GITHUB_TOKEN environment variable (public_repo read access).
Resumes automatically: run the same command again after an interruption.
Repo and actor names are hashed; set HASH_SALT to use your own salt.
"""
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time

import requests

TOKEN = os.environ["GITHUB_TOKEN"]
SALT = os.environ.get("HASH_SALT", "ned-ct378")
API = "https://api.github.com"
S = requests.Session()
S.headers.update({"Authorization": f"Bearer {TOKEN}",
                  "Accept": "application/vnd.github+json"})
LOCKFILE = re.compile(
    r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Pipfile\.lock|"
    r"requirements.*\.txt|pom\.xml|build\.gradle|go\.sum|Cargo\.lock)$")


def hid(x):
    return hashlib.sha256((SALT + str(x)).encode()).hexdigest()[:12]


def wait_for_reset(r):
    reset = int(r.headers.get("X-RateLimit-Reset", time.time() + 60))
    secs = max(reset - time.time(), 5) + 2
    print(f"rate limit reached, sleeping {int(secs)}s", flush=True)
    time.sleep(secs)


def get(url, params=None):
    for attempt in range(8):
        try:
            r = S.get(url, params=params, timeout=60)
        except requests.RequestException:
            wait = 5 * (attempt + 1)
            print(f"network error, retrying in {wait}s", flush=True)
            time.sleep(wait)
            continue
        if r.status_code in (403, 429):
            wait_for_reset(r)
            continue
        if r.status_code >= 500:
            time.sleep(5 * (attempt + 1))
            continue
        if int(r.headers.get("X-RateLimit-Remaining", 99)) < 3:
            wait_for_reset(r)
        return r
    return None


_commits = {}


def commit_feats(repo, sha):
    key = (repo, sha)
    if key in _commits:
        return _commits[key]
    out = None
    r = get(f"{API}/repos/{repo}/commits/{sha}")
    if r is not None and r.status_code == 200:
        c = r.json()
        stats = c.get("stats", {})
        files = c.get("files", [])
        parents = c.get("parents", [])
        out = {
            "additions": stats.get("additions", 0),
            "deletions": stats.get("deletions", 0),
            "files_modified": len(files),
            "msg_len": len(c["commit"]["message"]),
            "is_merge": int(len(parents) > 1),
            "num_parents": len(parents),
            "touches_lockfile": int(any(LOCKFILE.search(f["filename"]) for f in files)),
        }
    _commits[key] = out
    return out


def runs_for_day(repo, day):
    page = 1
    while page <= 10:
        r = get(f"{API}/repos/{repo}/actions/runs",
                {"created": day, "status": "completed", "per_page": 100, "page": page})
        if r is None or r.status_code != 200:
            return
        runs = r.json().get("workflow_runs", [])
        for run in runs:
            yield run
        if len(runs) < 100:
            return
        page += 1


def main(repos_file, start, end, out, max_per_repo=600):
    repos = [l.strip() for l in open(repos_file, encoding="utf-8-sig")
             if l.strip() and not l.startswith("#")]
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["run_id"])
            except (ValueError, KeyError):
                pass
    d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    n_days = (d1 - d0).days + 1
    day_cap = max(1, -(-max_per_repo // n_days))   # spread the repo cap evenly over the window
    with open(out, "a", encoding="utf-8") as f:
        for repo in repos:
            n, day = 0, d0
            print(repo, flush=True)
            while day <= d1 and n < max_per_repo:
                day_n = 0
                for run in runs_for_day(repo, day.isoformat()):
                    if n >= max_per_repo or day_n >= day_cap:
                        break
                    if run["id"] in done or run.get("run_attempt", 1) != 1:
                        continue
                    if run["event"] not in ("push", "pull_request"):
                        continue
                    if run["conclusion"] not in ("success", "failure"):
                        continue
                    feats = commit_feats(repo, run["head_sha"])
                    if feats is None:
                        continue
                    rec = {
                        "run_id": run["id"],
                        "repo": hid(repo),
                        "workflow_id": run["workflow_id"],
                        "run_number": run["run_number"],
                        "event": run["event"],
                        "branch_is_main": int(run["head_branch"] in ("main", "master")),
                        "created_at": run["created_at"],
                        "commit_sha": run["head_sha"],
                        "actor": hid((run.get("actor") or {}).get("login", "")),
                        "label": int(run["conclusion"] == "failure"),
                        **feats,
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    done.add(run["id"])
                    n += 1
                    day_n += 1
                day += dt.timedelta(days=1)
            print(f"  {n} new runs", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
         int(sys.argv[5]) if len(sys.argv) > 5 else 600)