"""Canonical "Ava" benchmark agent on LiveKit Agents (self-hosted), two personas.

One worker serves BOTH benchmark agent definitions; the persona is selected per
call via Cekura's dispatch config (`{"test_persona": "appointments" | "insurance"}`
in the Cekura agent's LiveKit Config JSON, delivered through dispatch job
metadata). Definitions live in ./definitions/<persona>/:
  system-prompt.txt, first-message.txt, tool-definitions.json, mock-tools.json

Constant stack (updated 2026-07-30 per "Cekura - LiveKit recommended setup"):
  LLM  LiveKit Inference, LLM_MODEL env (default openai/gpt-4.1), temp 0  |  STT  Deepgram nova-3 (en)
  TTS  Cartesia sonic-3 (voice 9626c31c-bec5-4cca-baa8-f8ba9e84c8bc)
  Turn detection: LiveKit Inference TurnDetector via TurnHandlingOptions
  Noise cancellation: Krisp BVC Telephony on SIP calls; ai-coustics QUAIL_L otherwise

Tools are built dynamically from each persona's tool-definitions.json (the fixed
cross-platform tool contract) via function_tool(raw_schema=...) and resolve
against that persona's mock-tools.json — the worker IS the fake backend.

Run (needs LIVEKIT_* + vendor creds in .env):  python agent.py dev
Cekura wiring: cekura[livekit] LiveKitTracer (scaffold: cekura-python
examples/livekit-appointment-agent). track_session correlates via dispatch job
metadata (scenario_id/run_id — no PSTN-style matching) and, when the Cekura agent
has mock tools configured, injects them in place of the same-named local tools
(disable with CEKURA_MOCK_TOOLS_ENABLED=false to force local mock_backend).
"""
import ast
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit import api, rtc
from livekit.protocol import agent_simulation as agent_simulation_proto
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    TurnHandlingOptions,
    inference,
    JobContext,
    cli,
    llm as agents_llm,
    function_tool,
    get_job_context,
    RunContext,
    StopResponse,
    room_io,
)
from livekit.plugins import noise_cancellation, openai

try:
    # Optional: ai-coustics enhancement for non-SIP audio. Not in this repo's
    # pyproject/uv.lock (only the deployed worker's env has it), so degrade to
    # Krisp BVC when it's absent instead of failing at import time.
    from livekit.plugins import ai_coustics
except ImportError:
    ai_coustics = None
from cekura.livekit import LiveKitTracer

from mock_backend import MockBackend

try:
    # Optional local-development hook: grades `lk agent simulate` runs against the
    # benchmark's tool-call scorer. It is not part of this repo or the deployed image.
    from sim_grading import on_simulation_end
except ImportError:
    on_simulation_end = None

logger = logging.getLogger("livekit-demo")

ROOT = Path(__file__).resolve().parent
DEFINITIONS = ROOT / "definitions"
# override=True: the repo .env is authoritative — the shell exports a different
# CEKURA_API_KEY (other org) which otherwise silently wins and 403s every tracer call
# Prefer .env; fall back to .env.local (common local override filename).
_env_file = ROOT / ".env" if (ROOT / ".env").exists() else ROOT / ".env.local"
load_dotenv(_env_file, override=True)


def _env(key, default=None):
    v = os.environ.get(key, default)
    return v.strip() if isinstance(v, str) else v


# Full LiveKit Inference model id for the platform path (e.g. "openai/gpt-4.1",
# "google/gemma-4-31b-it"). Azure fallback keeps its own deployment name below.
LLM_MODEL = _env("LLM_MODEL", "openai/gpt-4.1")
LLM_DEPLOYMENT = _env("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
LLM_TEMPERATURE = float(_env("LLM_TEMPERATURE", "0") or 0)
STT_MODEL = _env("STT_MODEL", "deepgram/nova-3")
TTS_MODEL = _env("TTS_MODEL", "cartesia/sonic-3")
VOICE_ID = _env("VOICE_ID", "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc")  # Cartesia Sonic-3 default voice
COMPONENT_METRICS_ENABLED = (_env("BENCHMARK_COMPONENT_METRICS", "true") or "").lower() in (
    "1",
    "true",
    "yes",
)
COMPONENT_METRICS_JSONL = _env("BENCHMARK_COMPONENT_METRICS_JSONL", "component-metrics.jsonl")

# --- Cekura observability/simulation tracer (no-op until an agent id is set).
# Appointments and insurance are SEPARATE Cekura agents, so the tracer is built
# per job once the persona is resolved: CEKURA_AGENT_ID_APPOINTMENT /
# CEKURA_AGENT_ID_INSURANCE, with CEKURA_AGENT_ID as the fallback for both. ---
CEKURA_AGENT_IDS = {
    "appointments": int(_env("CEKURA_AGENT_ID_APPOINTMENT") or _env("CEKURA_AGENT_ID") or 0),
    "insurance": int(_env("CEKURA_AGENT_ID_INSURANCE") or _env("CEKURA_AGENT_ID") or 0),
}


def _cekura_tracer(persona: str, agent_id_override: int | None = None):
    """Build the tracer for this persona's Cekura agent (disabled if no id)."""
    agent_id = agent_id_override or CEKURA_AGENT_IDS.get(persona, 0)
    tracer = LiveKitTracer(
        api_key=_env("CEKURA_API_KEY", ""),
        agent_id=agent_id,
        host=_env("CEKURA_HOST", "https://api.cekura.ai"),
        enabled=bool(agent_id),
    )
    return tracer, agent_id


def _metric_to_dict(metric) -> dict:
    if hasattr(metric, "model_dump"):
        return metric.model_dump(mode="json", exclude_none=True)
    if hasattr(metric, "dict"):
        return metric.dict(exclude_none=True)
    return dict(metric or {})


def _metric_component(metric_type: str) -> str:
    if metric_type.startswith("stt_"):
        return "stt"
    if metric_type.startswith("llm_") or metric_type.startswith("realtime_"):
        return "llm"
    if metric_type.startswith("tts_"):
        return "tts"
    return "other"


def _jsonl_path() -> Path | None:
    if not COMPONENT_METRICS_JSONL:
        return None
    path = Path(COMPONENT_METRICS_JSONL)
    return path if path.is_absolute() else ROOT / path


def _record_component_metric(payload: dict) -> None:
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    logger.info("BENCHMARK_COMPONENT_METRIC %s", line)
    path = _jsonl_path()
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("component metric jsonl write failed: %s", exc)


def _attach_component_metrics(stt, llm, tts, meta_fn) -> None:
    if not COMPONENT_METRICS_ENABLED:
        return

    def attach(component: str, emitter) -> None:
        on = getattr(emitter, "on", None)
        if not callable(on):
            logger.warning("component metrics unavailable for %s: no event emitter", component)
            return

        def on_metrics(metric):
            metric_obj = getattr(metric, "metrics", metric)
            raw = _metric_to_dict(metric_obj)
            metric_type = raw.get("type") or type(metric_obj).__name__
            try:
                meta = meta_fn()
            except Exception as exc:
                meta = {"meta_error": str(exc)}
            payload = {
                "schema": "benchmark_component_metric_v1",
                "platform": "livekit",
                "component": component or _metric_component(metric_type),
                "metric_type": metric_type,
                "observed_at": time.time(),
                "meta": meta,
                "metric": raw,
            }
            _record_component_metric(payload)

        try:
            on("metrics_collected", on_metrics)
        except Exception as exc:
            logger.warning("component metrics attach failed for %s: %s", component, exc)

    attach("stt", stt)
    attach("llm", llm)
    attach("tts", tts)


# --- persona selection & loading -------------------------------------------------
def _resolve_persona(ctx) -> str:
    """Pick the test persona. Call AFTER ctx.connect().

    Sources, in priority order:
    1. LiveKit simulation userdata (`lk agent simulate` scenarios set
       userdata.test_persona; read via ctx.simulation_context(), agents >= 1.6.6)
    2. Room metadata - Cekura delivers the agent's "LiveKit Config (JSON)"
       (e.g. {"test_persona": "insurance"}) as room metadata (cekura SDK exposes
       it as get_simulation_data()['additional_config']; room metadata is only
       populated after ctx.connect())
    3. Job/dispatch metadata (carries scenario_id/run_id/test_profile_data)
    4. TEST_PERSONA env, then the appointments default
    """

    def find(d):
        if isinstance(d, dict):
            v = d.get("test_persona")
            if isinstance(v, str):
                return v
            for x in d.values():
                r = find(x)
                if r:
                    return r
        elif isinstance(d, list):
            for x in d:
                r = find(x)
                if r:
                    return r
        return None

    persona = None

    # 1. LiveKit simulation scenarios (lk agent simulate) pass userdata
    sim_fn = getattr(ctx, "simulation_context", None)
    sim = sim_fn() if callable(sim_fn) else None
    if sim is not None:
        try:
            ud = sim.userdata() or {}
            if isinstance(ud, dict):
                persona = ud.get("test_persona")
        except Exception as exc:
            logger.debug("could not read simulation userdata: %s", exc)
        if persona:
            logger.info("persona from simulation userdata: %s", persona)

    for raw in (
        getattr(getattr(ctx, "room", None), "metadata", None),  # LiveKit Config (JSON)
        getattr(getattr(ctx, "job", None), "metadata", None),  # dispatch metadata
    ):
        if persona:
            break
        if raw and raw.strip():
            try:
                persona = find(json.loads(raw))
            except (ValueError, TypeError):
                persona = None
    persona = (persona or _env("TEST_PERSONA", "appointments")).strip().lower()
    if persona not in ("appointments", "insurance"):
        logger.warning("unknown test_persona %r; defaulting to appointments", persona)
        persona = "appointments"
    return persona


def _id_consumers(tool_defs: list[dict]) -> dict[str, list[str]]:
    """Map each *_id argument to the tools whose schema takes it."""
    out: dict[str, list[str]] = {}
    for t in tool_defs:
        for prop in (t.get("parameters") or {}).get("properties", {}):
            if prop.endswith("_id"):
                out.setdefault(prop, []).append(t["name"])
    return out


def _required_ids(tool_defs: list[dict]) -> dict[str, list[str]]:
    """Map each tool to the *_id arguments its schema marks required."""
    out: dict[str, list[str]] = {}
    for t in tool_defs:
        ids = [a for a in (t.get("parameters") or {}).get("required", []) if a.endswith("_id")]
        if ids:
            out[t["name"]] = ids
    return out


def _make_mock_tool(tdef: dict, backend: MockBackend):
    """Wrap one entry of tool-definitions.json (the fixed cross-platform tool
    contract) as a raw-schema function tool resolving against the mock backend.
    The fixture output is returned unchanged."""
    name = tdef["name"]

    async def handler(raw_arguments: dict) -> dict:
        return backend.resolve(name, raw_arguments)

    return function_tool(
        handler,
        raw_schema={
            "name": name,
            "description": tdef.get("description", ""),
            "parameters": tdef.get("parameters", {"type": "object", "properties": {}}),
        },
    )


# --- Call state that code can check without knowing the domain.
#     Nothing in this section names a tool, an argument value or a step of either
#     persona. Every rule is derived at start-up from the tool schemas (which
#     arguments end in _id, which of those are required, which are phone numbers)
#     and applied to the conversation so far, so it holds for any prompt and tool
#     set written to the same conventions. The system prompt, the tool schemas and
#     the tool results are passed through untouched.
#
#     It runs around the LLM call (BenchmarkAgent.llm_node), not inside the local
#     tool handlers: on Cekura calls track_session replaces those handlers with the
#     dashboard's mock tools. A tool call that is corrected or withheld here never
#     reaches the history in its uncorrected form, which matters because a rejected
#     call that the model then retries is itself a wrong call.
#     CALL_STATE_CHECKS=false switches the whole section off. ---
CALL_STATE_CHECKS = (_env("CALL_STATE_CHECKS", "true") or "").lower() in ("1", "true", "yes")


def _returned_ids(output) -> dict[str, str]:
    """Top-level *_id string values of one tool result. Results are stored as text,
    JSON or a Python repr depending on who ran the tool; anything else yields {}.
    Ids nested inside lists are not included: those are options to choose between,
    not state to carry forward."""
    value = output
    if isinstance(value, str):
        for parse in (json.loads, ast.literal_eval):
            try:
                value = parse(output)
                break
            except (ValueError, SyntaxError):
                continue
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if k.endswith("_id") and isinstance(v, str) and v}


def _with_id_notes(chat_ctx, consumers: dict[str, list[str]]):
    """Ids are state. When a tool returns an id that other tools take as an argument,
    the model tends to lose it a few turns later. This returns a copy of the context
    with a short system note after such a result naming the id and the tools that
    take it, plus what the repair step needs: the latest value per id and the raw
    text of every tool result so far."""
    items, latest, outputs, pending = [], {}, [], []

    def flush(after) -> None:
        if pending:
            items.append(agents_llm.ChatMessage(role="system", content=["; ".join(pending)],
                                                created_at=after.created_at))
            pending.clear()

    prev = None
    for item in chat_ctx.items:
        kind = getattr(item, "type", None)
        # keep a tool call and all of its results together; the note goes after them
        if kind not in ("function_call", "function_call_output") and prev is not None:
            flush(prev)
        items.append(item)
        prev = item
        if kind == "function_call_output" and not getattr(item, "is_error", False):
            outputs.append(str(item.output))
            for key, value in _returned_ids(item.output).items():
                latest[key] = value
                users = [t for t in consumers.get(key, []) if t != item.name]
                if users:
                    pending.append(f'pass {key} "{value}" in every later call to {", ".join(users)}')
    if prev is not None:
        flush(prev)
    return agents_llm.ChatContext(items), latest, outputs


def _repair_tool_arguments(name: str, raw: str, required_ids: dict[str, list[str]],
                           latest: dict[str, str], outputs: list[str]) -> str:
    """Fill an id argument the schema marks required when the model left it out or
    put in a value no tool returned, using the value a tool did return. Optional ids
    are left alone: whether one belongs in a call is the model's judgement.
    Returns `raw` untouched unless something had to change."""
    try:
        args = json.loads(raw or "{}")
    except ValueError:
        return raw
    if not isinstance(args, dict):
        return raw
    fixes: dict = {}
    for key in required_ids.get(name, ()):
        known = latest.get(key)
        if not known:
            continue  # no tool has returned one yet; nothing to carry
        value = args.get(key)
        returned = isinstance(value, str) and value and any(
            f'"{value}"' in o or f"'{value}'" in o for o in outputs)
        if not returned:
            fixes[key] = known
    if not fixes:
        return raw
    logger.info("tool arguments repaired: %s %s", name, fixes)
    return json.dumps({**args, **fixes})


# --- Phone numbers. On a phone line speech-to-text drops, doubles and swaps digits,
#     and a model given nine digits for a ten-digit field will sometimes invent the
#     tenth. So before a tool call that carries a phone number is made, two things
#     are checked that need no judgement:
#       1. the number has the length the schema asks for and is made of digits the
#          caller actually spoke; otherwise the caller is asked to say it again;
#       2. the agent has read that exact number back and the caller has spoken since
#          (so had the chance to correct it); otherwise it is read back now.
#     Either way the call is withheld for that turn. At most MAX_PHONE_HOLDS per call,
#     then the model's call goes through as written. ---
MAX_PHONE_HOLDS = int(_env("MAX_PHONE_HOLDS", "3") or 0)
PHONE_REASK = (" Sorry, I did not catch that phone number clearly. "
               "Could you say the full number again, one digit at a time?")
PHONE_READBACK = " Let me read that number back: {number}. Is that right?"
_DIGIT_WORDS = {"zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_DIGIT_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
_REPEATS = {"double": 2, "triple": 3}


def _phone_arguments(tool_defs: list[dict]) -> dict[str, dict[str, int]]:
    """Map each tool to its phone-number arguments and the digit count each needs:
    string arguments with "phone" in the name; the count is whatever the schema
    description states ("exactly 10 digits"), ten if it states none."""
    out: dict[str, dict[str, int]] = {}
    for t in tool_defs:
        for key, prop in (t.get("parameters") or {}).get("properties", {}).items():
            if "phone" in key.lower() and prop.get("type") == "string":
                stated = re.search(r"(\d+)\s+digits", prop.get("description") or "")
                out.setdefault(t["name"], {})[key] = int(stated.group(1)) if stated else 10
    return out


def _spoken_digits(text: str) -> str:
    """The digits in what someone said, in order, whether transcribed as numerals
    ("916-555-0194") or words ("nine one six, double five five, oh one nine four").
    Everything else is skipped, so a number split by filler words still joins up."""
    out, repeat = [], 1
    for token in re.findall(r"[a-z]+|\d", text.lower()):
        if token in _REPEATS:
            repeat = _REPEATS[token]
            continue
        digit = token if token.isdigit() else _DIGIT_WORDS.get(token)
        if digit is not None:
            out.append(digit * repeat)
        repeat = 1
    return "".join(out)


def _say_digits(digits: str) -> str:
    """Digits as words in spoken groups (3-3-4 for ten digits), for text-to-speech."""
    sizes = [3, 3, 4] if len(digits) == 10 else [3] * (len(digits) // 3 + 1)
    groups, at = [], 0
    for size in sizes:
        if digits[at:at + size]:
            groups.append(" ".join(_DIGIT_NAMES[int(d)] for d in digits[at:at + size]))
        at += size
    return ", ".join(groups)


def _heard_and_read_back(chat_ctx) -> tuple[str, list[str]]:
    """(every digit the caller has spoken, across turns; the digits of each agent
    message the caller has spoken after). A number given in two halves around a
    pause is still one run of caller digits."""
    messages = [i for i in chat_ctx.items
                if getattr(i, "type", None) == "message" and i.role in ("user", "assistant")]
    caller = "".join(_spoken_digits(m.text_content or "") for m in messages if m.role == "user")
    last_user = max((n for n, m in enumerate(messages) if m.role == "user"), default=-1)
    read_back = [_spoken_digits(m.text_content or "") for n, m in enumerate(messages)
                 if m.role == "assistant" and n < last_user]
    return caller, read_back


def _phone_problem(name: str, raw: str, phone_args: dict[str, dict[str, int]],
                   caller_digits: str, read_back: list[str]) -> tuple[str, str] | None:
    """What is wrong with the first phone argument of this tool call, if anything:
    ("unheard", value) when it is not a number of the required length that the
    caller said, ("unconfirmed", digits) when it has not been read back yet."""
    try:
        args = json.loads(raw or "{}")
    except ValueError:
        return None
    if not isinstance(args, dict):
        return None
    for key, length in phone_args.get(name, {}).items():
        value = args.get(key)
        if not isinstance(value, str):
            continue
        digits = "".join(c for c in value if c.isdigit())
        if len(digits) == length + 1 and digits.startswith("1"):
            digits = digits[1:]  # a country code is a formatting slip, not a mishearing
        if len(digits) != length or digits not in caller_digits:
            return ("unheard", value)
        if not any(digits in said for said in read_back):
            return ("unconfirmed", digits)
    return None


# --- native end-call. The framework ships an EndCallTool (livekit.agents.beta.tools),
#     but it is still beta, so we define our own: the hang-up is deleting the room
#     (disconnects everyone, ends the SIP/telephony leg). Defined in-code (not in the
#     per-persona tool-definitions.json — that's the cross-platform CUSTOM-tool
#     contract); parallels the Pipecat end_call.
#     closing_message is a required argument: a hang-up should never be silent,
#     and making the sign-off part of the tool call guarantees the caller hears
#     a final utterance before the line drops. The tool speaks it, then ends the
#     call. ---
def _unused_ids(items, consumers: dict[str, list[str]]) -> list[tuple[str, str, str, list[str]]]:
    """(key, value, producing tool, tools that take it) for each id a tool returned
    that no later call passed on, when another id of the same kind was passed on.
    The asymmetry is the signal: one of two parallel pieces of work was finished and
    the other was not. An id kind that is never passed on at all says nothing."""
    returned: list[tuple[str, str, str]] = []
    used: set[tuple[str, str]] = set()
    for item in items:
        kind = getattr(item, "type", None)
        if kind == "function_call":
            try:
                args = json.loads(getattr(item, "arguments", None) or "{}")
            except ValueError:
                args = {}
            if isinstance(args, dict):
                used |= {(k, v) for k, v in args.items() if k.endswith("_id") and isinstance(v, str)}
        elif kind == "function_call_output" and not getattr(item, "is_error", False):
            returned += [(k, v, item.name) for k, v in _returned_ids(item.output).items()]
    kinds_in_use = {k for k, _ in used}
    out = []
    for key, value, producer in returned:
        takers = [t for t in consumers.get(key, []) if t != producer]
        if takers and key in kinds_in_use and (key, value) not in used:
            out.append((key, value, producer, takers))
    return out


def _hangup_checks(session) -> list[tuple[str, str]]:
    """The checks this call owes before it may end, as (key, what to tell the model).

    Hanging up cannot be undone, and the commonest serious failure of an agent like
    this is to end the call with something its own instructions required left
    undone. So end_call is answered with a check instead of a hang-up, one check per
    attempt, each put once per call. The checks know nothing about the domain: they
    point the model at its instructions and add whatever plain facts the history
    shows. One question at a time, because given two the model answers the easier.
      "tools": no tool has run on this call at all. Either that is right (the caller
               left early, or asked for something that must be refused) or a required
               lookup or record was skipped.
      "close": anything still undone, and whether the caller is finished."""
    items = list(session.history.items)
    current = getattr(session, "current_agent", None)
    worked = any(getattr(i, "type", None) == "function_call" and i.name != "end_call" for i in items)
    checks = []
    if not worked:
        # one more plain fact when the schemas support it: the caller read out a number
        # as long as the phone number some tool takes, and that tool never ran
        phone_args = getattr(current, "_phone_args", None) or {}
        lengths = [n for args in phone_args.values() for n in args.values()]
        said_number = lengths and any(
            len(_spoken_digits(i.text_content or "")) >= min(lengths) for i in items
            if getattr(i, "type", None) == "message" and i.role == "user")
        fact = (f" The caller gave a phone number and {', '.join(sorted(phone_args))} "
                f"take{'s' if len(phone_args) == 1 else ''} one." if said_number else "")
        checks.append(("tools", "Not ended; the call is still open. No tool has been called on this "
                       f"call.{fact} Go through the steps your instructions give for a call like this "
                       "one. If any step is a tool call (a lookup, a check, or a record of how the call "
                       "ended), make that call now, using what the caller already told you; a caller "
                       "telling you what a lookup would find does not replace the lookup. If your "
                       "instructions require no tool call here (for example the caller left before "
                       "asking for anything, or asked for something you must refuse), call end_call "
                       "again."))
    consumers = getattr(current, "_id_consumers", None) or {}
    facts = "".join(f' {producer} returned {key} "{value}", which no later call has used, although '
                    f'another {key} on this call was ({", ".join(takers)} take it).'
                    for key, value, producer, takers in _unused_ids(items, consumers))
    if facts:
        facts += " If your instructions require something for it, do that now."
    checks.append(("close", f"Not ended; the call is still open. One more check before the line drops.{facts} "
                   "Has the caller said they need nothing more, or said goodbye? If not, ask whether "
                   "there is anything else, as a normal message, and wait for the answer. If they have, "
                   "call end_call again."))
    return checks


@function_tool
async def end_call(ctx: RunContext, closing_message: str) -> str:
    """Hang up the call, speaking `closing_message` as your final utterance first.

    ALWAYS end the call with this tool once the caller has confirmed they need
    nothing else, or when the caller says goodbye / has to go. Saying goodbye
    without calling this tool leaves the phone line open. Do not call it while
    the caller still has an open request.

    closing_message (REQUIRED): your complete, contextual final sign-off — it is
    spoken aloud to the caller and nothing after it will be heard. Author it for
    THIS call:
    - acknowledge where the call ended up (e.g. recap that they're all set after
      a booking, or acknowledge that they have to go / are deferring);
    - if the caller abandoned, deferred, or must leave, include an explicit
      callback / return-path offer (e.g. "No problem — call us back anytime and
      we can pick this up.");
    - close with a brief, warm goodbye.
    closing_message must NEVER contain a question — the caller cannot answer it.
    If you want to ask "is there anything else?", ask it as a normal message and
    do NOT call this tool yet. Call this tool at most ONCE per call; put the
    entire sign-off in this argument instead of saying it as a separate message."""
    session = ctx.session
    # Hang-up is held until each applicable check (_hangup_checks) has been put to
    # the model once; then end_call goes through whatever the model decided. Nothing
    # is spoken to the caller while it is held. end_call is a control tool, not part
    # of the scored tool contract.
    if not getattr(session, "_end_call_guard", False):
        asked = getattr(session, "_end_call_checks", None)
        if asked is None:
            asked = set()
            setattr(session, "_end_call_checks", asked)
        for key, check in _hangup_checks(session):
            if key not in asked:
                asked.add(key)
                logger.info("end_call held for the %r check", key)
                return check
    # The call ends once. In simulations the session outlives the hang-up (the
    # room is not deleted, see below), so the caller may keep exchanging
    # pleasantries and the model may invoke end_call again. Answer those trailing
    # turns with a brief fixed goodbye — repeating the full sign-off is noisy,
    # and going silent leaves the caller talking to a dead line.
    if getattr(session, "_end_call_guard", False):
        try:
            await session.say("Goodbye.")
        except Exception:
            pass
        raise StopResponse()
    setattr(session, "_end_call_guard", True)
    # let any in-flight agent speech finish, then speak the authored sign-off
    current = getattr(session, "current_speech", None)
    if current is not None:
        try:
            await current.wait_for_playout()
        except Exception:
            pass
    msg = (closing_message or "").strip()
    if msg:
        try:
            await session.say(msg)
        except Exception as exc:
            logger.warning("end_call: failed to speak closing message: %s", exc)
    job_ctx = get_job_context()
    # In `lk agent simulate` runs, deleting the room would tear down the
    # simulated caller mid-run; the simulation framework ends the conversation
    # itself. Record the hang-up intent and suppress the follow-up LLM reply
    # (StopResponse) — the sign-off has already been spoken, so a generated
    # response to the tool output would just duplicate it.
    sim_fn = getattr(job_ctx, "simulation_context", None)
    if callable(sim_fn) and sim_fn() is not None:
        logger.info("end_call invoked in simulation - skipping room deletion")
        raise StopResponse()
    # Real calls: return the tool result (observability tooling records it) and
    # delete the room, which disconnects all participants and ends the call.
    await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))
    return "The call has ended."


async def _wait_until_answered(ctx: JobContext, timeout: float = 45.0) -> rtc.RemoteParticipant:
    """Block until there is a remote participant who can actually hear us.

    Returns the first remote participant, so the caller can branch on its kind
    (e.g. SIP vs WebRTC for noise-cancellation selection).

    on_enter speaks the fixed first message the moment the session starts. For
    agent-initiated (OUTBOUND) SIP calls the agent is in the room before the
    callee picks up, so without this gate the greeting plays into ringing /
    early media. We wait for a remote participant and, if it is an outbound
    SIP participant, for sip.callStatus == "active" (answered).

    INBOUND calls must NOT be gated: they join the room with
    sip.callStatus == "ringing", and the SIP bridge holds the 200 OK until the
    SIP participant subscribes to remote audio (waitSubscribe in
    sip/pkg/sip/inbound.go) — i.e. until WE publish via session.start().
    Waiting for "active" here deadlocks until this timeout fires, adding ~45s
    of ring time for the caller and ~45s of leading silence in egress
    recordings.
    """
    participant = await ctx.wait_for_participant()
    if participant.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return participant
    if participant.attributes.get("sip.callDirection") != "outbound":
        # inbound (or direction attribute missing): the bridge answers once we
        # publish audio — start the session immediately instead of deadlocking.
        return participant

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = participant.attributes.get("sip.callStatus")
        if status is None or status == "active":
            return participant  # answered (or attribute not provided - don't block)
        if status == "hangup":
            logger.info("call ended before it was answered (callStatus=hangup)")
            return participant
        await asyncio.sleep(0.2)
    logger.warning("timed out after %.0fs waiting for SIP answer; continuing", timeout)
    return participant


def _load_persona(persona: str):
    d = DEFINITIONS / persona
    prompt = (d / "system-prompt.txt").read_text().rstrip("\n")
    first_message = (d / "first-message.txt").read_text().strip()
    backend = MockBackend(d / "mock-tools.json")
    tool_defs = _load_tool_defs(persona)
    tools = [_make_mock_tool(t, backend) for t in tool_defs]
    tools.append(end_call)
    return prompt, first_message, tools


def _load_tool_defs(persona: str) -> list[dict]:
    return json.loads((DEFINITIONS / persona / "tool-definitions.json").read_text())


# Seconds of mutual silence before the agent re-prompts (0 disables). A line that
# stays dead for ten seconds reads as a dropped call, so the re-prompt has to come
# before that; it also has to be long enough not to cut into a caller who pauses for
# a few seconds in the middle of a sentence or a number.
REPROMPT_AFTER_S = float(_env("REPROMPT_AFTER_S", "6") or 0)
# After a re-prompt the caller needs time to hear it and answer before another one.
REPROMPT_AGAIN_AFTER_S = max(REPROMPT_AFTER_S, 10.0)
# One re-prompt per silence by default: a caller who is slow to answer the first
# should not be asked twice.
MAX_REPROMPTS = int(_env("MAX_REPROMPTS", "1") or 1)


def _attach_silence_reprompt(session: AgentSession) -> None:
    """When nobody has spoken for REPROMPT_AFTER_S while the agent is waiting on the
    caller, re-ask instead of sitting in silence.

    A timer is armed whenever the agent goes back to listening or the caller stops
    talking, and cancelled by any caller speech or agent activity. Because the agent's
    own re-prompt ends in "listening" again, the timer re-arms after it, up to
    MAX_REPROMPTS in a row (the framework's user "away" event fires only once per
    silence). Caller speech resets the count. Never after end_call."""
    if not REPROMPT_AFTER_S:
        return
    state: dict = {"count": 0, "task": None}

    def cancel() -> None:
        task = state["task"]
        if task is not None and not task.done():
            task.cancel()
        state["task"] = None

    async def wait_then_reprompt() -> None:
        await asyncio.sleep(REPROMPT_AFTER_S if state["count"] == 0 else REPROMPT_AGAIN_AFTER_S)
        if (getattr(session, "_end_call_guard", False) or state["count"] >= MAX_REPROMPTS
                or session.agent_state != "listening" or session.user_state == "speaking"):
            return
        state["count"] += 1
        logger.info("caller silent for %.0fs; re-prompt %d", REPROMPT_AFTER_S, state["count"])
        instructions = (
            "The caller has gone quiet. In one short sentence, check they are still there "
            "and repeat your last question. Do not call any tools."
            if state["count"] == 1 else
            "The caller is still silent. Say briefly that you may be having trouble hearing "
            "them, and invite them to repeat their answer. Do not call any tools."
        )
        session.generate_reply(instructions=instructions, tool_choice="none")

    def arm() -> None:
        cancel()
        state["task"] = asyncio.create_task(wait_then_reprompt())

    @session.on("agent_state_changed")
    def _on_agent_state(ev) -> None:
        if ev.new_state == "listening" and session.user_state != "speaking":
            arm()
        else:
            cancel()

    @session.on("user_state_changed")
    def _on_user_state(ev) -> None:
        if ev.new_state == "speaking":
            state["count"] = 0
            cancel()
        elif ev.new_state == "listening" and session.agent_state == "listening":
            arm()


class BenchmarkAgent(Agent):
    def __init__(self, instructions: str, tools: list, first_message: str,
                 tool_defs: list[dict] | None = None):
        super().__init__(instructions=instructions, tools=tools)
        self._first_message = first_message
        self._id_consumers = _id_consumers(tool_defs or [])
        self._required_ids = _required_ids(tool_defs or [])
        self._phone_args = _phone_arguments(tool_defs or [])
        self._phone_holds = 0

    async def llm_node(self, chat_ctx, tools, model_settings):
        """The default LLM step, with the call-state checks around it (see "Call
        state that code can check" above)."""
        if not CALL_STATE_CHECKS:
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                yield chunk
            return
        ctx, latest, outputs = _with_id_notes(chat_ctx, self._id_consumers)
        caller_digits, read_back = _heard_and_read_back(chat_ctx)
        say = None  # set when a tool call was withheld over its phone number
        async for chunk in Agent.default.llm_node(self, ctx, tools, model_settings):
            delta = getattr(chunk, "delta", None)
            tool_calls = getattr(delta, "tool_calls", None) or []
            if tool_calls and say is not None:
                delta.tool_calls = tool_calls = []  # nothing after a withheld call runs this turn
            elif tool_calls and self._phone_holds < MAX_PHONE_HOLDS:
                for index, tool_call in enumerate(tool_calls):
                    problem = _phone_problem(tool_call.name, tool_call.arguments, self._phone_args,
                                             caller_digits, read_back)
                    if problem is None:
                        continue
                    # the calls before it still run; this one and any after it wait
                    delta.tool_calls = tool_calls = tool_calls[:index]
                    self._phone_holds += 1
                    say = (PHONE_REASK if problem[0] == "unheard"
                           else PHONE_READBACK.format(number=_say_digits(problem[1])))
                    logger.info("held %s: phone number %s (%d)", tool_call.name, problem[0], self._phone_holds)
                    break
            for tool_call in tool_calls:
                tool_call.arguments = _repair_tool_arguments(
                    tool_call.name, tool_call.arguments, self._required_ids, latest, outputs)
            yield chunk
        if say is not None:
            yield say

    async def on_enter(self):
        # Fixed first message: speak it deterministically via session.say() so it
        # plays immediately with zero LLM latency. Left interruptible (default).
        await self.session.say(self._first_message)


# One tool call per model response. In a tool set where one call returns an id that
# the next call requires, two such calls issued in the same response cannot both be
# right: the second is written before the id exists. In local runs every call that
# was missing a required id had been issued that way, and no sequential call was.
PARALLEL_TOOL_CALLS = (_env("PARALLEL_TOOL_CALLS", "false") or "").lower() in ("1", "true", "yes")


def _build_llm():
    # LLM-source-v2 (2026-07-03, Tarush mandate): platform provisioning wherever the
    # platform offers it -> LiveKit Inference gateway. temperature passed via
    # extra_kwargs (gateway forwards OpenAI-compatible params).
    if _env("LLM_SOURCE", "platform") == "platform":
        from livekit.agents import inference
        return inference.LLM(model=LLM_MODEL,
                             extra_kwargs={"temperature": LLM_TEMPERATURE,
                                           "parallel_tool_calls": PARALLEL_TOOL_CALLS})
    # fallback: BYO Azure (pre-mandate config, kept for rollback/A-B)
    return openai.LLM.with_azure(
        azure_deployment=LLM_DEPLOYMENT,
        azure_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        api_key=_env("AZURE_OPENAI_API_KEY") or _env("OPENAI_API_KEY"),
        api_version=_env("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        temperature=LLM_TEMPERATURE,
        parallel_tool_calls=PARALLEL_TOOL_CALLS,
    )


server = AgentServer()

@server.rtc_session(agent_name="cekura-benchmark-agent", on_simulation_end=on_simulation_end)
async def entrypoint(ctx: JobContext):
    # Connect FIRST: the persona comes from the Cekura agent's "LiveKit Config
    # (JSON)", which arrives as room metadata and is empty until ctx.connect().
    await ctx.connect()

    # Access job/dispatch metadata (set by Cekura in automated flows) and room
    # metadata (the Cekura agent's "LiveKit Config (JSON)") — log both verbatim
    # so every run records exactly what the worker was dispatched with.
    def _parse_meta(raw, label):
        if not (raw and raw.strip()):
            return {}
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            logger.warning("unparseable %s metadata: %r", label, raw[:200])
            return {}

    job_metadata = _parse_meta(getattr(ctx.job, "metadata", None), "job")
    room_metadata = _parse_meta(getattr(ctx.room, "metadata", None), "room")
    logger.info(
        "dispatch metadata received",
        extra={"job_metadata": job_metadata, "room_metadata": room_metadata},
    )

    persona = _resolve_persona(ctx)
    configured_agent_id = room_metadata.get("cekura_agent_id")
    try:
        configured_agent_id = int(configured_agent_id) if configured_agent_id else None
    except (TypeError, ValueError):
        logger.warning("invalid cekura_agent_id in room metadata: %r", configured_agent_id)
        configured_agent_id = None
    logger.info("resolved test persona: %s", persona)
    prompt, first_message, tools = _load_persona(persona)
    cekura, cekura_agent_id = _cekura_tracer(persona, configured_agent_id)
    logger.info("cekura agent id for persona %s: %s", persona, cekura_agent_id or "unset")

    # A LiveKit simulation (`lk agent simulate`) is NOT a Cekura call: the Cekura
    # tracer's track_session injects dashboard mock tools over our local ones —
    # including an end_call mock that returns text and never terminates the call,
    # which hangs simulation scenarios that end via end_call. Skip Cekura tracing
    # entirely under a simulation context so our own (sim-aware) tools run.
    _sim_fn = getattr(ctx, "simulation_context", None)
    in_simulation = callable(_sim_fn) and _sim_fn() is not None

    stt = inference.STT(
        model=STT_MODEL,
        language="en",
    )
    llm = _build_llm()
    tts = inference.TTS(
        model=TTS_MODEL,
        voice=VOICE_ID,
    )

    def metric_meta() -> dict:
        try:
            sim = cekura.get_simulation_data(ctx) if cekura_agent_id else None
        except Exception:
            sim = None
        return {
            "agent": "livekit-demo",
            "persona": persona,
            "cekura_agent_id": cekura_agent_id or None,
            "room": getattr(ctx.room, "name", None),
            "run_id": (sim or {}).get("run_id"),
            "scenario_id": (sim or {}).get("scenario_id"),
        }

    _attach_component_metrics(stt, llm, tts, metric_meta)

    session = AgentSession(
        stt=stt,
        llm=llm,
        tts=tts,
        # Medicare's normal consented handoff needs four consecutive calls
        # (permission -> qualification -> route -> handoff); multi-intent paths
        # need up to six.  LiveKit defaults max_tool_steps to 3, which forces a
        # tool-disabled reply before those paths are complete.
        max_tool_steps=8,
        # config revision 2026-07-30 (Cekura - LiveKit recommended setup):
        # LiveKit Inference turn detector — contextually aware end-of-utterance model,
        # served via LiveKit Inference (no local model download or VAD prewarm), plus
        # preemptive generation (LLM/TTS start on interim transcripts before EOU confirm).
        # Interruptions/endpointing stay at framework defaults (no benchmark-specific tuning).
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            preemptive_generation={"enabled": True},
        ),
    )
    agent = BenchmarkAgent(prompt, tools, first_message, tool_defs=_load_tool_defs(persona))
    # Silence is an audio concept; in a text simulation a slow simulated reply is not dead air.
    _sim = _sim_fn() if callable(_sim_fn) else None
    if _sim is None or _sim.simulation_mode != agent_simulation_proto.SimulationMode.SIMULATION_MODE_TEXT:
        _attach_silence_reprompt(session)

    if cekura_agent_id and not in_simulation:
        # Cekura voice calls: transcript/tool export keyed by dispatch metadata
        # (scenario_id/run_id) + dashboard mock-tool injection into `agent`
        await cekura.track_session(ctx, session, agent)
    elif in_simulation:
        logger.info("LiveKit simulation context — skipping Cekura tracing/mock injection")
    else:
        logger.warning("no Cekura agent id for persona %s — running without Cekura tracing", persona)

    if cekura_agent_id:
        sim = cekura.get_simulation_data(ctx)
        if sim:
            logger.info(f"Cekura simulation call: run_id={sim.get('run_id')} scenario_id={sim.get('scenario_id')}")

    # outbound: don't speak the first message until the callee has answered
    # (inbound passes through immediately — see _wait_until_answered docstring)
    # console mode runs in a local fake room with no remote participant to wait for
    remote = None if ctx.is_fake_job() else await _wait_until_answered(ctx)

    if cekura_agent_id and not in_simulation:
        # production/observe: audio recording (dual-channel egress) + call log.
        # Started AFTER the answer gate so recordings never open with
        # pre-answer ringing silence — Cekura's call matching breaks on the
        # ~45s of leading dead air that an early egress start produces.
        await cekura.observe_session(ctx, session)

    # Input audio cleanup on the caller's track before it reaches STT:
    #   SIP calls  -> Krisp BVC Telephony (background voice cancellation tuned
    #                 for narrowband/PSTN audio; requires LiveKit Cloud)
    #   WebRTC / simulate -> ai-coustics QUAIL_L when the plugin is installed,
    #                        else Krisp BVC (wideband model)
    if remote is not None and remote.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        nc = noise_cancellation.BVCTelephony()
    elif ai_coustics is not None:
        nc = ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_L)
    else:
        nc = noise_cancellation.BVC()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=nc,
            ),
        ),
    )
    # first message is spoken in BenchmarkAgent.on_enter (fixed greeting, zero LLM latency)


if __name__ == "__main__":
    cli.run_app(server)
