import json, os, sys, urllib.request

API = os.environ.get("DRYFT_API", "https://htn.dryft.ai")
TOK = os.environ["DRYFT_TOKEN"]


def get(path):
    r = urllib.request.Request(API + path, headers={"Authorization": "Bearer " + TOK})
    return json.load(urllib.request.urlopen(r))


def summarize(rid):
    d = get("/api/v1/runs/" + rid)
    d = d.get("run", d)
    sha = (d.get("commitSha") or "")[:7]
    res = d.get("result") or {}
    line = f"{rid[:8]} {sha} {d.get('state'):10s} err={d.get('errorCode')}"
    if res:
        line += f"  score={res.get('score'):.1f}" if res.get("score") else "  score=-"
        line += f" fail={res.get('failureCode')}"
    print(line)
    for s in res.get("shapes") or []:
        m = s.get("modelMetrics") or {}
        print(
            f"   {s['id']}: tps={s.get('tokensPerSecond'):8.1f} "
            f"tpot={m.get('tpotMs'):6.2f} ttft={m.get('ttftMs'):8.2f} "
            f"ms={s.get('metricMs'):8.1f} sd={s.get('stddevMs'):.2f} "
            f"correct={s.get('correct')} {s.get('caseStatus')}"
        )


def recent(n=15):
    d = get(f"/api/v1/runs?limit={n}")
    for r in d["items"]:
        res = r.get("result") or {}
        sc = res.get("score")
        print(
            f"{r['id'][:8]} {(r.get('commitSha') or '')[:7]} {r['state']:10s} "
            f"score={sc if sc is None else round(sc,1)} err={r.get('errorCode')} {r['createdAt'][:19]}"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "recent":
        recent()
    else:
        for rid in sys.argv[1].split(","):
            summarize(rid)
