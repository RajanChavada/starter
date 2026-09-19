"""Wait for the run Dryft created for a pushed commit and print its report.

    DRYFT_TOKEN=... python3 agent/watch.py <commit-sha-prefix> [timeout-s]
"""

import json
import os
import sys
import time
import urllib.request

API = os.environ.get("DRYFT_API", "https://htn.dryft.ai")


def get(path: str):
    request = urllib.request.Request(
        API + path, headers={"Authorization": "Bearer " + os.environ["DRYFT_TOKEN"]}
    )
    return json.load(urllib.request.urlopen(request))


def find_run(prefix: str):
    for run in get("/api/v1/runs?limit=30")["items"]:
        if (run.get("commitSha") or "").startswith(prefix):
            return run
    return None


def report(detail: dict) -> None:
    result = detail.get("result") or {}
    print(json.dumps({k: detail.get(k) for k in ("id", "mode", "state", "errorCode", "errorMessage")}))
    print("score", result.get("score"))
    for case in result.get("cases") or result.get("workloads") or []:
        print(
            f"  {case.get('name') or case.get('caseId')}: {case.get('caseStatus') or case.get('status')} "
            f"tok/s={case.get('tokensPerSecond')} tpot={case.get('tpotMs')} ttft={case.get('ttftMs')} "
            f"mem={case.get('peakMemoryBytes') or case.get('peakMemoryGiB')}"
        )
    if not result.get("cases") and not result.get("workloads"):
        print(json.dumps(result, indent=1)[:4000])


def main() -> None:
    prefix = sys.argv[1]
    deadline = time.time() + (float(sys.argv[2]) if len(sys.argv) > 2 else 3000)
    run = None
    while run is None and time.time() < deadline:
        run = find_run(prefix)
        if run is None:
            time.sleep(15)
    if run is None:
        sys.exit("no run for that commit yet")
    while time.time() < deadline:
        detail = get(f"/api/v1/runs/{run['id']}")["run"]
        if detail["state"] in ("succeeded", "failed", "cancelled", "canceled", "error"):
            report(detail)
            return
        time.sleep(20)
    sys.exit("timed out waiting")


if __name__ == "__main__":
    main()
