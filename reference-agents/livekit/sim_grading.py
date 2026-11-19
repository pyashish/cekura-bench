"""Grade `lk agent simulate` runs with the benchmark's own Tool Call Accuracy scorer.

Only runs under a LiveKit simulation (on_simulation_end); production calls never
reach it. The scenario's userdata carries its `cekura_id` (written by
local-sim/convert_scenarios.py), which selects the expected tool calls from the
published dataset in local-sim/data. The agent's actual
function calls are scored with the dataset's `scoring/tool_call_accuracy.py`,
unchanged, so a local pass means the same thing as Cekura's Tool Call Accuracy
gate. A score below 5/5 fails the run (the judge's verdict is ANDed with it).

Every run appends one line to local-sim/results/sim-runs.jsonl for the report.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from livekit.agents import SimulationContext

logger = logging.getLogger("sim-grading")



def _local_sim() -> Path:
    """<repo>/local-sim, or SIM_LOCAL_DIR. Resolved on use, never at import: agent.py
    imports this module in production too, and inside the deployed image the file
    sits at /app/sim_grading.py, where there is no repo two levels up."""
    override = os.environ.get("SIM_LOCAL_DIR")
    if override:
        return Path(override)
    parents = Path(__file__).resolve().parents
    return (parents[2] if len(parents) > 2 else parents[0]) / "local-sim"


def _results_path() -> Path:
    return Path(os.environ.get("SIM_RESULTS_JSONL") or _local_sim() / "results" / "sim-runs.jsonl")


def _load_scorer():
    # The published file is a Cekura custom-code metric: it defines _evaluate(data)
    # and then calls it on a global `data`. Run it with an empty one, keep the function.
    scorer = _local_sim() / "data" / "tool_call_accuracy.py"
    ns: dict = {"data": {}}
    exec(compile(scorer.read_text(), str(scorer), "exec"), ns)
    return ns["_evaluate"]


def _expected_tool_calls(sid: str) -> list[dict] | None:
    """The scenario's expected tool calls from the published dataset in
    local-sim/data/*.jsonl, or None if the scenario is not there. They are looked up
    here rather than passed in userdata, which has to stay under LiveKit's 2048-byte
    room-attribute limit."""
    for path in sorted((_local_sim() / "data").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("id") == sid:
                    return row.get("expected_tool_calls") or []
    return None


def _transcript(session) -> tuple[list[dict], list[dict], int]:
    """Cekura-shaped transcript_json (Function Call rows), a readable chat log, and
    how many user turns came after the first end_call.

    Under simulation end_call does not hang up (agent.py skips the room delete), so
    the conversation can carry on past it. On a real call the line is dead at that
    point, so only tool calls made before the first end_call count, the same as
    what Cekura sees on the phone."""
    calls, chat = [], []
    ended, user_after_end = False, 0
    for item in session.history.items:
        kind = getattr(item, "type", None)
        if kind == "function_call":
            if not ended:
                calls.append({"role": "Function Call", "content": item.name,
                              "data": {"name": item.name, "arguments": item.arguments}})
            chat.append({"role": "tool_call", "name": item.name, "arguments": item.arguments,
                         "after_end_call": ended})
            if item.name == "end_call":
                ended = True
        elif kind == "function_call_output":
            chat.append({"role": "tool_output", "name": item.name, "output": str(item.output)[:500]})
            if item.name == "end_call" and str(item.output).startswith("Not ended"):
                ended = False  # end_call deferred by its precondition check; the call went on
        elif kind == "message":
            chat.append({"role": item.role, "text": item.text_content or "", "after_end_call": ended})
            if ended and item.role == "user":
                user_after_end += 1
    return calls, chat, user_after_end


def score(expected: list[dict], calls: list[dict]):
    entries = [{"tool_name": e["tool"], "new_entry": {"input": e.get("arguments") or {}},
                "optional": bool(e.get("optional"))} for e in expected]
    return _load_scorer()({"generated_mock_tool_entries": entries, "transcript_json": calls})


async def on_simulation_end(ctx: SimulationContext) -> None:
    ud = ctx.userdata() or {}
    sid = ud.get("cekura_id", "?")
    session = ctx.job_context.primary_session
    calls, chat, user_after_end = _transcript(session)
    tca, detail = (None, f"no expected tool calls found for {sid}")
    try:
        expected = ud["expected_tool_calls"] if "expected_tool_calls" in ud else _expected_tool_calls(sid)
    except Exception:
        logger.exception("could not load expected tool calls for %s", sid)
        expected = None
    if expected is not None:
        try:
            tca, detail = score(expected, calls)
        except Exception as exc:  # never let grading crash the job
            logger.exception("tool scoring failed for %s", sid)
            detail = f"scorer error: {exc}"
    verdict = ctx.simulator_verdict
    row = {
        "ts": time.time(),
        "run_id": ctx.simulation_run_id,
        "job_id": ctx.simulation_job_id,
        "cekura_id": sid,
        "variant": os.environ.get("SIM_VARIANT", "baseline"),
        "judge_pass": verdict.success,
        "judge_reason": verdict.reason,
        "tool_score": tca,
        "tool_detail": detail,
        "user_turns_after_end_call": user_after_end,
        "chat": chat,
    }
    try:
        results = _results_path()
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except OSError:  # a read-only image must not turn a graded run into a crash
        logger.exception("could not write the simulation result row for %s", sid)
    logger.info("SIM_GRADE %s judge=%s tool=%s", sid, verdict.success, tca)
    if tca is not None and tca < 5.0:  # scorer is 0-5; Cekura gates on 5/5
        ctx.fail(reason=f"Tool Call Accuracy {tca:.1f}/5: {detail[:900]}")
