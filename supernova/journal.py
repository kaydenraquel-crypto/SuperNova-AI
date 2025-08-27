from datetime import datetime
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "SuperJournal.txt"

def log_event(event: str, meta: dict | None = None):
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.utcnow().isoformat()}Z] {event}\n")
        if meta:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def append_session(supervisor: str, subagents: list[str], actions: list[str], results: list[str], suggestions: list[str], meta: dict | None = None):
    """Append a structured entry to SuperJournal.txt following the template."""
    from uuid import uuid4
    ts = datetime.utcnow().isoformat() + "Z"
    session_id = f"SN-{uuid4().hex[:8].upper()}"
    entry_lines = []
    entry_lines.append("\n---\n")
    entry_lines.append("### [Timestamp UTC]")
    entry_lines.append(ts)
    entry_lines.append("")
    entry_lines.append(f"**Session ID:** {session_id}")
    entry_lines.append(f"**Supervisor:** {supervisor}")
    entry_lines.append("**Sub-Agents Used (≤5):**")
    if subagents:
        for a in subagents[:5]:
            entry_lines.append(f"- {a}")
    else:
        entry_lines.append("- None")
    entry_lines.append("")
    entry_lines.append("### Actions Performed")
    if actions:
        for i,a in enumerate(actions, start=1):
            entry_lines.append(f"{i}. {a}")
    else:
        entry_lines.append("- (none)")
    entry_lines.append("")
    entry_lines.append("### Results")
    if results:
        for r in results:
            entry_lines.append(f"- {r}")
    else:
        entry_lines.append("- (none)")
    entry_lines.append("")
    entry_lines.append("### Suggestions / Next Steps")
    if suggestions:
        for s in suggestions:
            entry_lines.append(f"- {s}")
    else:
        entry_lines.append("- (none)")
    if meta:
        entry_lines.append("")
        entry_lines.append("### Meta")
        entry_lines.append(json.dumps(meta, ensure_ascii=False, indent=2))
    entry_lines.append("\n")
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write("\n".join(entry_lines))
