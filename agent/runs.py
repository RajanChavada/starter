"""Print the latest Dryft runs for this repository, one line per workload.

    DRYFT_TOKEN=... python3 agent/runs.py [limit]
"""

import json
import os
import sys
import urllib.request

API = os.environ.get("DRYFT_API", "https://htn.dryft.ai")


def get(path: str) -> dict:
    request = urllib.request.Request(
        API + path, headers={"Authorization": "Bearer " + os.environ["DRYFT_TOKEN"]}
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    for run in get(f"/api/v1/runs?limit={limit}")["items"]:
        result = run.get("result") or {}
        if not result:
            result = (get(f"/api/v1/runs/{run['id']}")["run"].get("result")) or {}
        score = result.get("score")
        print(
            f"{run['commitSha'][:7]} {run['mode']:8} {run['state']:10} {run['id'][:8]}"
            f" score={score if score is None else round(score, 1)}"
            f" {run.get('errorCode') or ''}"
        )
        for shape in result.get("shapes", []):
            metrics = shape["modelMetrics"]
            print(
                f"  {shape['id']}: {shape['caseStatus']} tok/s={shape['tokensPerSecond']:.1f}"
                f" tpot={metrics['tpotMs']:.3f} ttft={metrics['ttftMs']:.1f}"
                f" spread={shape['stddevMs'] / shape['meanMs'] * 100:.1f}%"
            )


if __name__ == "__main__":
    main()
