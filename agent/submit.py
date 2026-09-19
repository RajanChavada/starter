"""Package engine/, submit, start a run, wait, and print the report plus log tail.

    DRYFT_TOKEN=... python3 agent/submit.py [public|official] [--no-wait]
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("DRYFT_API", "https://htn.dryft.ai")

from client import Dryft  # noqa: E402
from loop import ENGINE_DIR, report  # noqa: E402
from package import package  # noqa: E402


def logs(client: Dryft, run_id: str) -> None:
    after = -1
    while True:
        page = client.logs(run_id, after=after, limit=500)
        for item in page.get("items") or []:
            print("  |", item.get("line") if isinstance(item, dict) else item)
        if not page.get("hasMore"):
            break
        after = page.get("nextAfter", after)


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "public"
    wait = "--no-wait" not in sys.argv
    client = Dryft()
    archive = package(ENGINE_DIR)
    submission_id = client.submit(archive)
    run = client.start_run(submission_id, mode=mode)
    print(f"submission {submission_id}\nrun {run['id']} ({mode})", flush=True)
    if not wait:
        return
    started = time.time()
    detail = client.wait(run["id"], timeout=3000, interval=15)
    print(f"finished in {time.time() - started:.0f}s")
    report(detail)
    logs(client, run["id"])


if __name__ == "__main__":
    main()
