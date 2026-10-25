"""End-to-end smoke test of a running deployment, through the public entry point only.

Checks that the pages are served (including the /ops route), the services are ready, a
question is answered with citations, the answer streams as Server-Sent Events in order,
feedback is stored, an agent run completes, and the ops views respond. Standard library
only, so it runs anywhere (locally, in CI, against a cloud URL).

Usage:
    docker compose up -d --build --wait
    python scripts/smoke.py                              # http://localhost:8080
    python scripts/smoke.py --base-url https://... --user-key ... --admin-key ...

Exits with status 1 on the first failed check. Costs no LLM tokens when the deployment
runs without RAG_LLM (offline answers and the rule-based agent).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple


class Client:
    def __init__(self, base: str, user_key: str = "", admin_key: str = ""):
        self.base = base.rstrip("/")
        self.user_key, self.admin_key = user_key, admin_key

    def request(self, method: str, path: str, body: Optional[dict] = None, admin: bool = False, timeout: float = 60) -> Tuple[int, Dict[str, str], bytes]:
        headers = {"accept": "*/*", "x-request-id": f"smoke-{int(time.time() * 1000)}"}
        key = self.admin_key if admin else self.user_key
        if key:
            headers["authorization"] = f"Bearer {key}"
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["content-type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, {k.lower(): v for k, v in response.headers.items()}, response.read()
        except urllib.error.HTTPError as error:
            return error.code, {k.lower(): v for k, v in error.headers.items()}, error.read()


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--user-key", default="")
    parser.add_argument("--admin-key", default="")
    args = parser.parse_args()
    c = Client(args.base_url, args.user_key, args.admin_key)

    status, headers, body = c.request("GET", "/")
    check("the app page is served", status == 200 and b"Diaphragm Material Assistant" in body, f"{status}")
    status, _, body = c.request("GET", "/ops")
    check("the /ops route serves the app (SPA fallback)", status == 200 and b'id="root"' in body, f"{status}")

    status, headers, body = c.request("GET", "/api/health")
    check("BFF is alive and sets a request ID", status == 200 and headers.get("x-request-id", "").startswith("smoke-"), headers.get("x-request-id", "-"))
    status, _, body = c.request("GET", "/api/ready")
    ready = json.loads(body) if status == 200 else {}
    check("the API is ready with an index", status == 200 and ready.get("upstream", {}).get("index_chunks", 0) > 0, json.dumps(ready)[:120])

    status, _, body = c.request("POST", "/api/ask", {"question": "How is springback compensated?"})
    answer = json.loads(body) if status == 200 else {}
    check("a question is answered with citations", status == 200 and answer.get("citations"), f"{status}: {answer.get('answer', '')[:80]}")

    started = time.perf_counter()
    status, headers, body = c.request("POST", "/api/ask/stream", {"question": "What reduces galling?"})
    events = [line.split(":", 1)[1].strip() for line in body.decode("utf-8").splitlines() if line.startswith("event:")]
    check("the answer streams as Server-Sent Events", status == 200 and headers.get("content-type", "").startswith("text/event-stream"), headers.get("content-type", "-"))
    check("events arrive in protocol order", events[:1] == ["sources"] and events[-1:] == ["final"], f"{events[:3]}…{events[-1:]} in {time.perf_counter() - started:.2f} s")

    status, _, body = c.request("POST", "/api/feedback", {"request_id": answer.get("request_id", "smoke"), "rating": "up"})
    check("feedback is stored", status == 201, f"{status}")

    status, _, body = c.request("POST", "/api/agent/runs", {"question": "What causes tearing near a tight radius?"}, timeout=180)
    run = json.loads(body) if status == 200 else {}
    check("an agent run finishes", status == 200 and run.get("status") in ("completed", "awaiting_approval"), f"{run.get('status')} with {run.get('planner')}")

    for view in ("summary", "feedback", "approvals", "agent-runs", "ingestion"):
        status, _, body = c.request("GET", f"/api/ops/{view}", admin=True)
        check(f"ops view {view} responds", status == 200, f"{status}")
    print("all checks passed")


if __name__ == "__main__":
    main()
