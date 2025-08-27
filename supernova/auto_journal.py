# supernova/auto_journal.py
"""
Auto-journal middleware that logs API calls and important events into SuperJournal.txt.

Attach this into FastAPI via event hooks or route decorators.
"""
from datetime import datetime
import pathlib, json
from fastapi import Request

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "SuperJournal.txt"

async def log_api_call(request: Request, response_body: dict | None = None):
    """Log an API call with metadata and optional response."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "path": request.url.path,
        "method": request.method,
        "client": request.client.host if request.client else None,
        "query_params": dict(request.query_params),
    }
    try:
        body = await request.json()
        entry["body"] = body
    except Exception:
        pass
    if response_body:
        entry["response"] = response_body

    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

"""
# Usage example (in api.py):

from fastapi import Request
from .auto_journal import log_api_call

@app.post("/advice", response_model=AdviceOut)
async def get_advice(req: AdviceRequest, request: Request):
    action, conf, details, rationale, risk_notes = advise(...)
    response = AdviceOut(...)
    await log_api_call(request, response.model_dump())
    return response
"""
