"""Offline checks for the end_call preconditions and id carry-forward in agent.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import agent  # noqa: E402


def msg(role, text):
    return SimpleNamespace(type="message", role=role, text_content=text)


def call(name):
    return SimpleNamespace(type="function_call", name=name)


def session(persona, items):
    _, _, tools = agent._load_persona(persona)
    return SimpleNamespace(history=SimpleNamespace(items=items),
                           current_agent=SimpleNamespace(tools=tools))


def test_appointments_requires_lookup():
    s = session("appointments", [msg("assistant", "Anything else I can help with?"), msg("user", "No, bye.")])
    assert agent._end_call_precondition(s)[0] == "identify"


def test_insurance_requires_disposition():
    s = session("insurance", [msg("assistant", "Anything else?"), msg("user", "That answers it, nothing else.")])
    assert agent._end_call_precondition(s)[0] == "disposition"


def test_goodbye_does_not_waive_the_disposition_check():
    """MS65: the caller says goodbye after being given the redirect, and the route
    record is still owed. Whether nothing is owed (MS79, left during the opening) is
    the first option the instruction offers the model."""
    s = session("insurance", [msg("assistant", "Your first step is Social Security."),
                              msg("user", "That answers my request. Goodbye.")])
    key, text = agent._end_call_precondition(s)
    assert key == "disposition"
    assert text.index("nothing to record") < text.index("non_qualified")
    assert agent._MAX_REFUSALS["disposition"] == 1


async def test_refusals_count_per_call_not_per_turn(monkeypatch):
    """AS10: refused for the missing lookup, the model said "one moment, I'll check"
    without calling the tool, the caller said goodbye, and the next end_call used to
    go through because the caller had spoken. The step is still missing, so it is
    refused again; the third attempt ends the call (the escape for an emergency)."""
    items = [msg("assistant", "Is there anything else I can help you with?"), msg("user", "Thank you.")]
    s = session("appointments", items)
    spoken = []

    async def say(text):
        spoken.append(text)

    s.say, s.current_speech = say, None
    ctx = SimpleNamespace(session=s)
    assert (await agent.end_call(ctx, closing_message="Bye")).startswith("Not ended")
    items += [msg("assistant", "One moment, I'll check our records."), msg("user", "Thanks, goodbye.")]
    assert (await agent.end_call(ctx, closing_message="Bye")).startswith("Not ended")
    assert spoken == []  # nothing is said to the caller while the hang-up is refused
    monkeypatch.setattr(agent, "get_job_context", lambda: SimpleNamespace(simulation_context=lambda: object()))
    with pytest.raises(agent.StopResponse):
        await agent.end_call(ctx, closing_message="Bye")
    assert spoken == ["Bye"]


def test_every_route_but_the_public_redirect_needs_a_handoff_summary():
    """MS76: member-services route made, sales handoff written, member-services handoff forgotten."""
    L = agent.agents_llm
    member = [_call("route_medicare_call", "r1", json.dumps({"route_reason": "member_services",
                                                              "service_issue_type": "claim"})),
              _output("route_medicare_call", "r1", str({"route_id": "route_3007", "route_status": "x"}))]
    sales = [_call("route_medicare_call", "r2", json.dumps({"route_reason": "sales_or_plan_review"})),
             _output("route_medicare_call", "r2", json.dumps({"route_id": "route_3017"})),
             _call("create_handoff_summary", "h2", json.dumps({"route_id": "route_3017",
                                                               "summary_type": "warm_transfer"}))]
    close = [L.ChatMessage(role="assistant", content=["Is there anything else I can help with?"]),
             L.ChatMessage(role="user", content=["No, that's all. Goodbye."])]
    key, text = agent._end_call_precondition(session("insurance", member + sales + close))
    assert key == "handoff" and "route_3007 (member_services)" in text and "route_3017" not in text

    done = [_call("create_handoff_summary", "h1", json.dumps({"route_id": "route_3007"}))]
    assert agent._end_call_precondition(session("insurance", member + sales + done + close)) is None

    redirect = [_call("route_medicare_call", "r3", json.dumps({"route_reason": "non_qualified"})),
                _output("route_medicare_call", "r3", json.dumps({"route_id": "route_3009"}))]
    assert agent._end_call_precondition(session("insurance", redirect + close)) is None


def test_must_ask_anything_else_unless_caller_is_done():
    items = [call("lookup_patient"), msg("assistant", "Your appointment is on July 8 at 9."),
             msg("user", "Great, thanks.")]
    assert agent._end_call_precondition(session("appointments", items))[0] == "anything_else"
    items[-1] = msg("user", "Great, thanks, that's all. Bye!")
    assert agent._end_call_precondition(session("appointments", items)) is None


def test_clean_close_passes():
    items = [call("lookup_patient"), msg("assistant", "Is there anything else I can help you with?"),
             msg("user", "No.")]
    assert agent._end_call_precondition(session("appointments", items)) is None


def _tool_defs(persona):
    return json.loads((ROOT / f"definitions/{persona}/tool-definitions.json").read_text())


def _history(*items):
    return agent.agents_llm.ChatContext(list(items))


def _call(name, call_id, arguments="{}"):
    return agent.agents_llm.FunctionCall(name=name, call_id=call_id, arguments=arguments)


def _output(name, call_id, output):
    return agent.agents_llm.FunctionCallOutput(name=name, call_id=call_id, output=output, is_error=False)


PERMISSION = {"consent_id": "perm_1004", "status": "recorded"}


def test_fixture_output_is_returned_unchanged():
    defs = _tool_defs("insurance")
    backend = agent.MockBackend(ROOT / "definitions/insurance/mock-tools.json")
    tdef = next(t for t in defs if t["name"] == "record_medicare_permissions")
    args = {"caller_name": "Daniel Rivera", "caller_relationship": "adult_child",
            "callback_phone": "6025550166", "contact_consent": True, "data_sharing_consent": True,
            "scope_of_appointment_product_types": ["medicare_advantage"], "consent_statement": "yes",
            "beneficiary_name": "Rosa Rivera"}
    assert backend.resolve(tdef["name"], args)["consent_id"] == "perm_1004"
    assert "id_carry_forward" not in backend.resolve(tdef["name"], args)


@pytest.mark.parametrize("stored", [json.dumps(PERMISSION), str(PERMISSION)])
def test_id_note_follows_the_tool_result_whatever_ran_the_tool(stored):
    """The note is built from history, so it also appears when Cekura's injected
    mock tools (not our handlers) produced the result, as JSON or as a repr."""
    ctx = _history(agent.agents_llm.ChatMessage(role="user", content=["yes, you can share it"]),
                   _call("record_medicare_permissions", "c1"),
                   _output("record_medicare_permissions", "c1", stored))
    out, latest, outputs = agent._with_id_notes(ctx, agent._id_consumers(_tool_defs("insurance")))
    note = out.items[-1]
    assert note.type == "message" and note.role == "system"
    assert 'consent_id "perm_1004"' in note.text_content
    assert "save_medicare_qualification" in note.text_content
    assert latest == {"consent_id": "perm_1004"} and outputs == [stored]
    assert len(ctx.items) == 3  # the session's own history is not touched


def test_id_note_waits_for_all_results_of_a_parallel_tool_group():
    ctx = _history(_call("record_medicare_permissions", "c1"), _call("route_medicare_call", "c2"),
                   _output("record_medicare_permissions", "c1", json.dumps(PERMISSION)),
                   _output("route_medicare_call", "c2", json.dumps({"route_id": "route_3007"})),
                   agent.agents_llm.ChatMessage(role="assistant", content=["Thanks."]))
    out, _, _ = agent._with_id_notes(ctx, agent._id_consumers(_tool_defs("insurance")))
    kinds = [(i.type, getattr(i, "role", None)) for i in out.items]
    assert kinds == [("function_call", None), ("function_call", None), ("function_call_output", None),
                     ("function_call_output", None), ("message", "system"), ("message", "assistant")]


def test_nested_ids_are_not_carried_forward():
    lookup = {"patient_id": "p_1001", "upcoming_appointments": [{"appointment_id": "appt_5001"}]}
    assert agent._returned_ids(json.dumps(lookup)) == {"patient_id": "p_1001"}
    assert agent._returned_ids("not a record") == {}


def _repair(name, args, latest, outputs, persona="insurance"):
    required = agent._required_ids(_tool_defs(persona))
    return json.loads(agent._repair_tool_arguments(name, json.dumps(args), required, latest, outputs))


def test_required_id_filled_when_missing_or_invented():
    latest, outputs = {"consent_id": "perm_1004"}, [json.dumps(PERMISSION)]
    base = {"caller_name": "Daniel Rivera", "product_interest": "medicare_advantage"}
    assert _repair("save_medicare_qualification", base, latest, outputs)["consent_id"] == "perm_1004"
    assert _repair("save_medicare_qualification", {**base, "consent_id": "pending"},
                   latest, outputs)["consent_id"] == "perm_1004"


def test_repair_leaves_valid_and_optional_ids_alone():
    outputs = [str({"patient_id": "p_1001"}), str({"patient_id": "p_1002"})]
    args = {"patient_id": "p_1001", "datetime": "2026-07-08T09:00:00", "reason": "checkup"}
    raw = json.dumps(args)
    # an id a tool did return is the model's choice, even if it is not the latest one
    assert agent._repair_tool_arguments("book_appointment", raw, agent._required_ids(_tool_defs("appointments")),
                                        {"patient_id": "p_1002"}, outputs) is raw
    # consent_id is optional on a handoff summary: which handoff it belongs to is the model's call
    summary = {"summary_type": "member_services_redirect", "route_status": "x", "destination": "y",
               "fields_collected": [], "compliance_notes": "z"}
    assert "consent_id" not in _repair("create_handoff_summary", summary, {"consent_id": "perm_1004"},
                                       [json.dumps(PERMISSION)])
    # nothing to carry before any tool has returned the id
    assert "consent_id" not in _repair("save_medicare_qualification", {"caller_name": "A"}, {}, [])
    assert agent._repair_tool_arguments("lookup_patient", "{not json", {}, {}, []) == "{not json"


def test_service_issue_type_follows_the_route_reason():
    route = {"route_reason": "sales_or_plan_review", "availability_context": "normal"}
    assert _repair("route_medicare_call", route, {}, [])["service_issue_type"] == "not_applicable"
    member = {"route_reason": "member_services"}
    assert "service_issue_type" not in _repair("route_medicare_call", member, {}, [])
    stated = {"route_reason": "non_qualified", "service_issue_type": "other"}
    assert _repair("route_medicare_call", stated, {}, [])["service_issue_type"] == "other"


def test_refusal_closeout_lists_no_collected_fields():
    """MS80: the model listed product interest and Part A/B on a no-consent close-out."""
    closeout = {"summary_type": "no_consent_close", "route_status": "closed_no_consent",
                "destination": "Medicare.gov", "compliance_notes": "refused",
                "fields_collected": ["product_interest: medicare_advantage"]}
    assert _repair("create_handoff_summary", closeout, {}, [])["fields_collected"] == []
    transfer = {**closeout, "summary_type": "warm_transfer"}
    assert _repair("create_handoff_summary", transfer, {}, [])["fields_collected"] == closeout["fields_collected"]


def test_phone_arguments_come_from_the_schema():
    assert agent._phone_arguments(_tool_defs("appointments")) == {"lookup_patient": ["phone"]}
    assert agent._phone_arguments(_tool_defs("insurance")) == {
        "record_medicare_permissions": ["callback_phone"], "save_medicare_qualification": ["callback_phone"]}


HEARD = "My callback number is nine one six five five zero one nine four."     # Cekura MS76, as transcribed
SAID = "My callback number is nine one six, five five five, zero one nine four."


@pytest.mark.parametrize("text,digits", [
    (SAID, "9165550194"), ("It's 916-555-0194.", "9165550194"),
    ("nine one six, uh, double five five, oh one nine four", "9165550194"),
    ("I also want to shop for next year.", ""),
])
def test_spoken_digits(text, digits):
    assert agent._spoken_digits(text) == digits


@pytest.mark.parametrize("heard,number,bad", [
    (HEARD, "916550194", True),     # what the agent sent on Cekura: one digit short
    (HEARD, "9165501940", True),    # what it more often does locally: pads to ten digits
    (HEARD, "9165550194", True),    # right by luck is still not what was heard
    (SAID, "9165550194", False), (SAID, "19165550194", False), (SAID, "916-555-0194", False),
])
def test_phone_argument_must_be_ten_digits_the_caller_said(heard, number, bad):
    phones = agent._phone_arguments(_tool_defs("insurance"))
    raw = json.dumps({"caller_name": "Grace Thompson", "callback_phone": number})
    said = agent._spoken_digits(heard)
    assert (agent._bad_phone("record_medicare_permissions", raw, phones, said) is not None) is bad
    assert agent._bad_phone("route_medicare_call", raw, phones, said) is None  # not a phone-taking tool


def test_number_given_in_two_turns_counts_as_said():
    ctx = _history(agent.agents_llm.ChatMessage(role="user", content=["It's two zero two five five five"]),
                   agent.agents_llm.ChatMessage(role="assistant", content=["Go on."]),
                   agent.agents_llm.ChatMessage(role="user", content=["zero one eight eight."]))
    phones = agent._phone_arguments(_tool_defs("appointments"))
    assert agent._bad_phone("lookup_patient", '{"phone": "2025550188"}', phones, agent._caller_digits(ctx)) is None


class _Stream:
    """Stands in for the default llm_node: yields the given chunks."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __call__(self, *_):
        async def gen():
            for c in self.chunks:
                yield c
        return gen()


def _tool_chunk(*calls):
    L = agent.agents_llm
    return L.ChatChunk(id="c", delta=L.ChoiceDelta(role="assistant", tool_calls=[
        L.FunctionToolCall(name=n, arguments=json.dumps(a), call_id=f"call_{i}") for i, (n, a) in enumerate(calls)]))


async def _drive(ava, monkeypatch, chunk, *caller_lines):
    monkeypatch.setattr(agent.Agent.default, "llm_node", _Stream([chunk]))
    ctx = _history(*(agent.agents_llm.ChatMessage(role="user", content=[line]) for line in caller_lines))
    return [c async for c in ava.llm_node(ctx, [], None)]


async def test_call_with_a_number_the_caller_did_not_say_is_held_and_asked_again(monkeypatch):
    prompt, first, tools = agent._load_persona("insurance")
    ava = agent.BenchmarkAgent(prompt, tools, first, tool_defs=_tool_defs("insurance"))
    route = ("route_medicare_call", {"route_reason": "member_services", "service_issue_type": "claim"})
    padded = ("record_medicare_permissions", {"caller_name": "Grace Thompson", "callback_phone": "9165501940"})
    save = ("save_medicare_qualification", {"caller_name": "Grace Thompson", "callback_phone": "9165501940"})

    out = await _drive(ava, monkeypatch, _tool_chunk(route, padded, save), HEARD)
    assert [t.name for t in out[0].delta.tool_calls] == ["route_medicare_call"]  # the valid call still runs
    assert out[-1] == agent.PHONE_REASK

    good = ("record_medicare_permissions", {"caller_name": "Grace Thompson", "callback_phone": "9165550194"})
    out = await _drive(ava, monkeypatch, _tool_chunk(good), HEARD, SAID)
    assert [t.name for t in out[0].delta.tool_calls] == ["record_medicare_permissions"] and len(out) == 1

    # after MAX_PHONE_REASKS the model's call goes through rather than looping on the caller
    await _drive(ava, monkeypatch, _tool_chunk(padded), HEARD)
    out = await _drive(ava, monkeypatch, _tool_chunk(padded), HEARD)
    assert ava._phone_reasks == agent.MAX_PHONE_REASKS
    assert [t.name for t in out[0].delta.tool_calls] == ["record_medicare_permissions"] and len(out) == 1


def test_disposition_instruction_names_every_field_the_prompt_requires():
    s = session("insurance", [msg("assistant", "Anything else?"), msg("user", "That answers it, nothing else.")])
    text = agent._end_call_precondition(s)[1]
    for field in ("non_qualified", "product_interest", "not_applicable", "normal"):
        assert field in text


class FakeSession:
    def __init__(self):
        self.handlers, self.replies = {}, []
        self.agent_state, self.user_state = "speaking", "listening"

    def on(self, name):
        def deco(fn):
            self.handlers[name] = fn
            return fn
        return deco

    def emit(self, name, **kw):
        if name == "agent_state_changed":
            self.agent_state = kw["new_state"]
        else:
            self.user_state = kw["new_state"]
        self.handlers[name](SimpleNamespace(**kw))

    def generate_reply(self, **kw):
        self.replies.append(kw)
        # the agent speaks the re-prompt, then listens again
        self.emit("agent_state_changed", new_state="speaking")
        self.emit("agent_state_changed", new_state="listening")


async def test_silence_reprompt_twice_then_stops(monkeypatch):
    import asyncio
    monkeypatch.setattr(agent, "REPROMPT_AFTER_S", 0.02)
    monkeypatch.setattr(agent, "REPROMPT_AGAIN_AFTER_S", 0.04)
    monkeypatch.setattr(agent, "MAX_REPROMPTS", 2)
    s = FakeSession()
    agent._attach_silence_reprompt(s)
    s.emit("agent_state_changed", new_state="listening")
    await asyncio.sleep(0.15)
    assert len(s.replies) == agent.MAX_REPROMPTS
    assert all(r["tool_choice"] == "none" for r in s.replies)


async def test_silence_reprompt_reset_by_speech_and_off_after_end_call(monkeypatch):
    import asyncio
    monkeypatch.setattr(agent, "REPROMPT_AFTER_S", 0.05)
    s = FakeSession()
    agent._attach_silence_reprompt(s)
    s.emit("agent_state_changed", new_state="listening")
    await asyncio.sleep(0.02)
    s.emit("user_state_changed", new_state="speaking")   # caller talks: timer cancelled
    await asyncio.sleep(0.08)
    assert s.replies == []
    s._end_call_guard = True
    s.emit("user_state_changed", new_state="listening")  # re-armed, but the call has ended
    await asyncio.sleep(0.08)
    assert s.replies == []


async def test_second_reprompt_waits_longer(monkeypatch):
    import asyncio
    monkeypatch.setattr(agent, "REPROMPT_AFTER_S", 0.02)
    monkeypatch.setattr(agent, "REPROMPT_AGAIN_AFTER_S", 0.2)
    monkeypatch.setattr(agent, "MAX_REPROMPTS", 2)
    s = FakeSession()
    agent._attach_silence_reprompt(s)
    s.emit("agent_state_changed", new_state="listening")
    await asyncio.sleep(0.1)  # past the first wait, well before the second
    assert len(s.replies) == 1


async def test_default_is_single_reprompt(monkeypatch):
    import asyncio
    monkeypatch.setattr(agent, "REPROMPT_AFTER_S", 0.02)
    monkeypatch.setattr(agent, "REPROMPT_AGAIN_AFTER_S", 0.02)
    s = FakeSession()
    agent._attach_silence_reprompt(s)
    s.emit("agent_state_changed", new_state="listening")
    await asyncio.sleep(0.15)
    assert len(s.replies) == 1
