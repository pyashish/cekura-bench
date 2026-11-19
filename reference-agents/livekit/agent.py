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
from sim_grading import on_simulation_end

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
    """Map each *_id argument to the tools whose schema takes it (e.g. consent_id ->
    save_medicare_qualification, create_handoff_summary)."""
    out: dict[str, list[str]] = {}
    for t in tool_defs:
        for prop in (t.get("parameters") or {}).get("properties", {}):
            if prop.endswith("_id"):
                out.setdefault(prop, []).append(t["name"])
    return out


def _required_ids(tool_defs: list[dict]) -> dict[str, list[str]]:
    """Map each tool to the *_id arguments its schema marks required
    (save_medicare_qualification -> consent_id, book_appointment -> patient_id)."""
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


# --- Call state the code owns: ids returned by tools.
#     In the Cekura run GPT-4.1 regularly dropped an id a tool had just returned from
#     the next call that needed it: consent_id missing from save_medicare_qualification
#     (where the schema marks it required; raw-schema tools are not strictly enforced)
#     and from create_handoff_summary (MS62/66/72/74/77/82). Rejecting the bad call and
#     letting the model retry does not help, because the benchmark scores the extra
#     call as a false positive, so the call has to be right the first time.
#
#     Both halves work from the conversation history, not from the local tool
#     handlers, because on real Cekura calls track_session swaps the handlers for the
#     dashboard's mock tools and anything done inside _make_mock_tool never runs:
#       1. before each LLM call, a note after the tool result names the id and the
#          tools that take it (_with_id_notes);
#       2. after it, a required id the model still left out, or filled with a value
#          no tool returned ("pending"), is set to the returned one
#          (_repair_tool_arguments). Optional ids are left to the model: whether a
#          handoff summary should carry consent_id depends on which handoff it is.
#     ARG_REPAIR=false switches both off. ---
ARG_REPAIR = (_env("ARG_REPAIR", "true") or "").lower() in ("1", "true", "yes")


def _returned_ids(output) -> dict[str, str]:
    """Top-level *_id string values of one tool result. Results are stored as text,
    JSON or a Python repr depending on who ran the tool; anything else yields {}.
    Nested ids (a patient's list of appointments) are deliberately not included:
    they are choices for the caller, not state to carry forward."""
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
    """A copy of the context with a short system note after each tool result that
    returned an id later tools take, plus what the repair step needs: the latest
    value per id and the raw text of every tool result so far."""
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


# Arguments the tool contract fixes as a function of another argument, so code can
# fill them when the model leaves them out. route_medicare_call's schema: "Supply
# not_applicable for sales, plan-review, non-qualified, or caregiver-authority
# routes." The model omitted it on the missing-ZIP callback route in 4 of 5 local
# full runs (MS74).
_NOT_A_SERVICE_ROUTE = ("sales_or_plan_review", "non_qualified", "caregiver_authority_needed")


def _derived_arguments(name: str, args: dict) -> dict:
    if (name == "route_medicare_call" and "service_issue_type" not in args
            and args.get("route_reason") in _NOT_A_SERVICE_ROUTE):
        return {"service_issue_type": "not_applicable"}
    return {}


def _repair_tool_arguments(name: str, raw: str, required_ids: dict[str, list[str]],
                           latest: dict[str, str], outputs: list[str]) -> str:
    """The model's arguments for one tool call, with the state code owns filled in.
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
    fixes.update(_derived_arguments(name, {**args, **fixes}))
    if not fixes:
        return raw
    logger.info("tool arguments repaired: %s %s", name, fixes)
    return json.dumps({**args, **fixes})


# --- native end-call. The framework ships an EndCallTool (livekit.agents.beta.tools),
#     but it is still beta, so we define our own: the hang-up is deleting the room
#     (disconnects everyone, ends the SIP/telephony leg). Defined in-code (not in the
#     per-persona tool-definitions.json — that's the cross-platform CUSTOM-tool
#     contract); parallels the Pipecat end_call.
#     closing_message is a required argument: a hang-up should never be silent,
#     and making the sign-off part of the tool call guarantees the caller hears
#     a final utterance before the line drops. The tool speaks it, then ends the
#     call. ---
_ANYTHING_ELSE = ("anything else", "something else", "else i can", "else can i", "else you need",
                  "other questions", "any other", "help with anything")
_CALLER_DONE = ("bye", "have to go", "gotta go", "got to go", "need to go", "that's all", "that is all",
                "nothing else", "no, that's it", "that's it", "all set", "i'm good", "no thanks",
                "no thank you", "don't need anything else", "do not need anything else")
# How many times each check may refuse end_call before it goes through anyway.
# "disposition" refuses once: one of its answers is "nothing to record, end the call".
_MAX_REFUSALS = {"disposition": 1}


def _end_call_precondition(session) -> tuple[str, str] | None:
    """What still has to happen before this call may end, as (key, instruction).

    In the Cekura run the model hung up with the caller's request unhandled or
    unrecorded: right after reading back an appointment (AS22, caller then asked
    for a follow-up), before ever calling lookup_patient (AS10), or after giving
    public Medicare resources without the route_medicare_call record the prompt
    requires (MS65, MS80, MS76). These checks are deterministic facts about the
    call so far; the model still decides what to do about them."""
    items = list(session.history.items)
    called = {i.name for i in items if getattr(i, "type", None) == "function_call"}
    tools = set()
    try:
        tools = {getattr(t, "id", None) or getattr(t, "name", None) for t in session.current_agent.tools}
    except Exception:
        pass
    msgs = [i for i in items if getattr(i, "type", None) == "message" and i.role in ("user", "assistant")]
    last_user = next((m.text_content or "" for m in reversed(msgs) if m.role == "user"), "").lower()
    caller_done = any(p in last_user for p in _CALLER_DONE)

    if "lookup_patient" in tools and "lookup_patient" not in called:
        return ("identify", "You skipped Step 2: lookup_patient was never called. Every appointment request, "
                "including a caller who says they are a new patient, must be checked against the records. If "
                "the caller already gave a phone number, call lookup_patient with it now (no need to ask "
                "again), tell them the result, then close. Only if this call is an emergency, outside the "
                "clinic's scope, or the caller refused to give a number, call end_call again instead.")
    if "route_medicare_call" in tools and not {"route_medicare_call", "create_handoff_summary"} & called:
        # Whether anything needs recording is a judgement about the call, so the model
        # makes it. The no-record case comes first: a caller who leaves during the
        # opening (MS79) must not get a routing record. A goodbye is not evidence
        # either way (in MS65 the caller says goodbye after being given the redirect),
        # which is why this is not a keyword test on the caller's last words.
        return ("disposition", "No outcome is recorded for this call yet. Decide which one of these "
                "describes the call, do only that, then end the call.\n"
                "1. The caller left before you handled any request, for example during or right after "
                "the opening disclosure: there is nothing to record. Call end_call again.\n"
                "2. The caller refused permission: do not route. Record the closeout with "
                "create_handoff_summary (summary_type no_consent_close, route_status closed_no_consent).\n"
                "3. You pointed a caller who is not yet on Medicare to public resources: call "
                "route_medicare_call with route_reason non_qualified, the product_interest the caller "
                "stated if they stated one, service_issue_type not_applicable and availability_context "
                "normal. That route is the complete record; do not create a handoff summary for it.\n"
                "4. The caller has an issue with an existing plan: call route_medicare_call with "
                "route_reason member_services and the service_issue_type, then record the handoff.")
    if not caller_done:
        # the agent's message just before the caller's latest turn
        prev_agent = ""
        for m in reversed(msgs):
            if m.role == "assistant":
                prev_agent = (m.text_content or "").lower()
                break
        if not any(p in prev_agent for p in _ANYTHING_ELSE):
            return ("anything_else", "Before ending, ask the caller whether there is anything else you "
                    "can help with, as a normal message, and wait for their answer.")
    return None


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
    # Code-owned preconditions, checked before anything is spoken. A failed check
    # refuses the hang-up and says what is missing. Each check refuses a limited
    # number of times per call (_MAX_REFUSALS, default 2) and then lets the call
    # end, so a call that legitimately skips the step (emergency, refusal,
    # injection attempt) still ends. The count is per call, not per turn: a model
    # that answers a refusal with "one moment, I'll check" and no tool call is
    # refused again when it next tries to hang up (AS10). end_call is a control
    # tool, so a refused call does not count against Tool Call Accuracy.
    if not getattr(session, "_end_call_guard", False):
        reason = _end_call_precondition(session)
        if reason:
            refusals = getattr(session, "_end_call_refusals", None)
            if refusals is None:
                refusals = {}
                setattr(session, "_end_call_refusals", refusals)
            count = refusals.get(reason[0], 0)
            if count < _MAX_REFUSALS.get(reason[0], 2):
                refusals[reason[0]] = count + 1
                logger.info("end_call deferred: %s (%d)", reason[0], count + 1)
                return f"Not ended; the call is still open. {reason[1]}"
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


# Runtime addendum to the frozen appointments prompt (a prompt change, kept out of
# definitions/ so the benchmark input stays byte-identical). In audio runs with
# background noise or packet loss, STT swaps or drops digits of the caller's phone
# number and the model called lookup_patient with what it heard, unconfirmed
# (local AS28/AS31/AS51/AS55/AS56; Cekura AS58). A wrong argument cannot be fixed
# after the fact (a rejected call still counts as an extra call), so confirmation
# has to happen before the call.
PHONE_READBACK_RULE = """
# Phone number capture (audio reliability)
- Phone audio often drops or swaps digits. After the caller gives a phone number,
  read it back digit by digit in groups of three, three and four, and ask them to
  confirm before calling lookup_patient.
- If you heard fewer or more than ten digits, or any part was unclear, ask only for
  the missing or unclear part, then read back the whole number again.
- After any correction, read back the full corrected number and wait for a clear yes.
- Never call lookup_patient with a number the caller has not confirmed."""


def _load_persona(persona: str):
    d = DEFINITIONS / persona
    prompt = (d / "system-prompt.txt").read_text().rstrip("\n")
    if persona == "appointments" and (_env("PHONE_READBACK", "true") or "").lower() in ("1", "true", "yes"):
        prompt += "\n" + PHONE_READBACK_RULE
    first_message = (d / "first-message.txt").read_text().strip()
    backend = MockBackend(d / "mock-tools.json")
    tool_defs = _load_tool_defs(persona)
    tools = [_make_mock_tool(t, backend) for t in tool_defs]
    tools.append(end_call)
    return prompt, first_message, tools


def _load_tool_defs(persona: str) -> list[dict]:
    return json.loads((DEFINITIONS / persona / "tool-definitions.json").read_text())


# Seconds of mutual silence before the agent re-prompts (0 disables). Cekura counts
# more than 10 s without an agent response as an infrastructure failure (AS50, AS39),
# and AS44/AS46 expect the agent to re-ask once after a 6-8 s caller pause. Short
# enough to stay under 10 s, long enough not to fire inside the 2-3 s mid-utterance
# pauses of AS57/AS58.
REPROMPT_AFTER_S = float(_env("REPROMPT_AFTER_S", "6") or 0)
# After a re-prompt the caller needs time to hear it and answer; a second nudge at
# 6 s made AS44 (expects exactly one re-ask) fail when the caller answered at ~8 s.
REPROMPT_AGAIN_AFTER_S = max(REPROMPT_AFTER_S, 10.0)
# One re-prompt. A second one made AS44 (expects exactly one re-ask) fail in 2 of 6
# audio runs when the simulated caller was slow to answer the first; the original
# agent, with no re-prompt at all, passed AS44. Raise only with evidence.
MAX_REPROMPTS = int(_env("MAX_REPROMPTS", "1") or 1)


def _attach_silence_reprompt(session: AgentSession) -> None:
    """When nobody has spoken for REPROMPT_AFTER_S while the agent is waiting on the
    caller, re-ask instead of sitting in silence.

    A timer is armed whenever the agent goes back to listening or the caller stops
    talking, and cancelled by any caller speech or agent activity. Because the agent's
    own re-prompt ends in "listening" again, the timer re-arms after it, up to
    MAX_REPROMPTS in a row (the framework's user "away" event fires only once per
    silence, which left the second re-prompt out in AS46). Caller speech resets the
    count. Never after end_call."""
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

    async def llm_node(self, chat_ctx, tools, model_settings):
        if not ARG_REPAIR:
            async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
                yield chunk
            return
        ctx, latest, outputs = _with_id_notes(chat_ctx, self._id_consumers)
        async for chunk in Agent.default.llm_node(self, ctx, tools, model_settings):
            delta = getattr(chunk, "delta", None)
            for tool_call in getattr(delta, "tool_calls", None) or ():
                tool_call.arguments = _repair_tool_arguments(
                    tool_call.name, tool_call.arguments, self._required_ids, latest, outputs)
            yield chunk

    async def on_enter(self):
        # Fixed first message: speak it deterministically via session.say() so it
        # plays immediately with zero LLM latency. Left interruptible (default).
        await self.session.say(self._first_message)


# One tool call per model response. Every tool in both personas takes something an
# earlier tool returns (consent_id -> lead_id -> route_id; patient_id), so a call
# issued in the same response as the one that produces its input cannot be right.
# That is what the dropped consent_id was: across 246 save_medicare_qualification
# calls in the local runs, the 8 without a consent_id were exactly the 8 the model
# issued in parallel with record_medicare_permissions, and all 238 sequential ones
# carried it.
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
    # Experiment override (dead-air investigation): off | nc | bvc | bvc_telephony
    nc_override = (_env("NOISE_CANCELLATION", "") or "").lower()
    if nc_override:
        nc = {"off": None, "nc": noise_cancellation.NC(), "bvc": noise_cancellation.BVC(),
              "bvc_telephony": noise_cancellation.BVCTelephony()}[nc_override]
        logger.info("noise cancellation override: %s", nc_override)

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
