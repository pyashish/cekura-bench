"""Offline checks for the call-state mechanisms in agent.py.

The mechanisms are meant to be generic, so most tests run against a made-up tool set
(a help desk) that shares nothing with the benchmark's personas except the schema
conventions the code relies on: ids end in _id, phone arguments have "phone" in the
name. One test at the end checks the same code against the real definitions.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import agent  # noqa: E402

L = agent.agents_llm


def _tool(name, required=(), **props):
    return {"name": name, "parameters": {"type": "object", "required": list(required),
                                         "properties": {k: {"type": "string", "description": v}
                                                        for k, v in props.items()}}}


HELPDESK = [
    _tool("find_customer", ["contact_phone"], contact_phone="The caller's number as exactly 10 digits."),
    _tool("open_case", ["customer_id"], customer_id="From find_customer.", summary="What is wrong."),
    _tool("add_note", ["case_id"], case_id="From open_case.", text="The note."),
    _tool("close_case", [], case_id="Supply only when a case was opened.", outcome="How it ended."),
]
CONSUMERS = agent._id_consumers(HELPDESK)
REQUIRED = agent._required_ids(HELPDESK)
PHONES = agent._phone_arguments(HELPDESK)


def user(text):
    return L.ChatMessage(role="user", content=[text])


def assistant(text):
    return L.ChatMessage(role="assistant", content=[text])


def call(name, call_id, **arguments):
    return L.FunctionCall(name=name, call_id=call_id, arguments=json.dumps(arguments))


def output(name, call_id, result, as_json=True):
    return L.FunctionCallOutput(name=name, call_id=call_id, is_error=False,
                                output=json.dumps(result) if as_json else str(result))


def history(*items):
    return L.ChatContext(list(items))


# ---------------------------------------------------------------- schema-derived tables
def test_tables_are_read_from_the_schemas():
    assert CONSUMERS == {"customer_id": ["open_case"], "case_id": ["add_note", "close_case"]}
    assert REQUIRED == {"open_case": ["customer_id"], "add_note": ["case_id"]}
    assert PHONES == {"find_customer": {"contact_phone": 10}}
    seven = [_tool("page", ["desk_phone"], desk_phone="Internal extension, 7 digits.")]
    assert agent._phone_arguments(seven) == {"page": {"desk_phone": 7}}


# ---------------------------------------------------------------- ids are carried forward
@pytest.mark.parametrize("as_json", [True, False])
def test_id_note_follows_the_tool_result_whoever_ran_the_tool(as_json):
    """Built from history, so it also appears when the platform's injected mock tools
    (not our handlers) produced the result, stored as JSON or as a Python repr."""
    ctx = history(user("my screen is broken"), call("open_case", "c1", customer_id="cu_7"),
                  output("open_case", "c1", {"case_id": "case_42", "ok": True}, as_json))
    out, latest, outputs = agent._with_id_notes(ctx, CONSUMERS)
    note = out.items[-1]
    assert note.type == "message" and note.role == "system"
    assert 'case_id "case_42"' in note.text_content and "add_note, close_case" in note.text_content
    assert latest == {"case_id": "case_42"} and len(outputs) == 1
    assert len(ctx.items) == 3  # the session's own history is not touched


def test_id_note_waits_for_all_results_of_a_tool_group():
    ctx = history(call("find_customer", "c1"), call("open_case", "c2"),
                  output("find_customer", "c1", {"customer_id": "cu_7"}),
                  output("open_case", "c2", {"case_id": "case_42"}), assistant("Thanks."))
    out, _, _ = agent._with_id_notes(ctx, CONSUMERS)
    kinds = [(i.type, getattr(i, "role", None)) for i in out.items]
    assert kinds == [("function_call", None), ("function_call", None), ("function_call_output", None),
                     ("function_call_output", None), ("message", "system"), ("message", "assistant")]


def test_nested_ids_are_options_not_state():
    result = {"customer_id": "cu_7", "open_cases": [{"case_id": "case_1"}, {"case_id": "case_2"}]}
    assert agent._returned_ids(json.dumps(result)) == {"customer_id": "cu_7"}
    assert agent._returned_ids("not a record") == {}


def _repair(name, args, latest, outputs):
    return json.loads(agent._repair_tool_arguments(name, json.dumps(args), REQUIRED, latest, outputs))


def test_required_id_is_filled_when_missing_or_invented():
    latest, outputs = {"case_id": "case_42"}, [json.dumps({"case_id": "case_42"})]
    assert _repair("add_note", {"text": "rebooted"}, latest, outputs)["case_id"] == "case_42"
    assert _repair("add_note", {"text": "rebooted", "case_id": "pending"}, latest, outputs)["case_id"] == "case_42"


def test_repair_leaves_valid_and_optional_ids_alone():
    outputs = [str({"case_id": "case_41"}), str({"case_id": "case_42"})]
    raw = json.dumps({"case_id": "case_41", "text": "x"})
    # an id a tool did return is the model's choice, even if it is not the latest one
    assert agent._repair_tool_arguments("add_note", raw, REQUIRED, {"case_id": "case_42"}, outputs) is raw
    # optional on this tool: whether it belongs in the call is the model's judgement
    assert "case_id" not in _repair("close_case", {"outcome": "no fault found"}, {"case_id": "case_42"}, outputs)
    # nothing to carry before any tool has returned the id
    assert "case_id" not in _repair("add_note", {"text": "x"}, {}, [])
    assert agent._repair_tool_arguments("add_note", "{not json", REQUIRED, {}, []) == "{not json"


# ---------------------------------------------------------------- phone numbers
SAID = "It's nine one six, five five five, zero one nine four."
HEARD = "It's nine one six five five zero one nine four."  # one digit lost in transcription


@pytest.mark.parametrize("text,digits", [
    (SAID, "9165550194"), ("It's 916-555-0194.", "9165550194"),
    ("nine one six, uh, double five five, oh one nine four", "9165550194"),
    ("I also want to ask about my bill.", ""),
])
def test_spoken_digits(text, digits):
    assert agent._spoken_digits(text) == digits


def test_say_digits_groups_for_speech():
    assert agent._say_digits("9165550194") == "nine one six, five five five, zero one nine four"
    assert agent._spoken_digits(agent._say_digits("9165550194")) == "9165550194"
    assert agent._spoken_digits(agent._say_digits("5550194")) == "5550194"


def _problem(number, *messages):
    caller, read_back = agent._heard_and_read_back(history(*messages))
    return agent._phone_problem("find_customer", json.dumps({"contact_phone": number}), PHONES, caller, read_back)


def test_phone_number_must_be_what_the_caller_said():
    assert _problem("916550194", user(HEARD)) == ("unheard", "916550194")      # one digit short
    assert _problem("9165501940", user(HEARD)) == ("unheard", "9165501940")    # padded to fit the schema
    assert _problem("9165550194", user(HEARD)) == ("unheard", "9165550194")    # right by luck is still unheard


def test_phone_number_must_have_been_read_back_and_answered():
    assert _problem("9165550194", user(SAID)) == ("unconfirmed", "9165550194")
    read = assistant("I have nine one six, five five five, zero one nine four. Is that right?")
    assert _problem("9165550194", user(SAID), read) == ("unconfirmed", "9165550194")  # caller has not answered
    assert _problem("9165550194", user(SAID), read, user("Yes.")) is None
    assert _problem("19165550194", user(SAID), read, user("Yes.")) is None            # country code tolerated
    assert _problem("916-555-0194", user(SAID), read, user("Yes.")) is None


def test_number_given_in_two_turns_counts_as_said():
    said = [user("It's nine one six five five five"), assistant("Go on."), user("zero one nine four.")]
    assert _problem("9165550194", *said) == ("unconfirmed", "9165550194")


def test_other_tools_and_arguments_are_not_phone_checked():
    caller, read_back = agent._heard_and_read_back(history(user(HEARD)))
    raw = json.dumps({"case_id": "916550194"})
    assert agent._phone_problem("add_note", raw, PHONES, caller, read_back) is None


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
    return L.ChatChunk(id="c", delta=L.ChoiceDelta(role="assistant", tool_calls=[
        L.FunctionToolCall(name=n, arguments=json.dumps(a), call_id=f"call_{i}") for i, (n, a) in enumerate(calls)]))


async def _drive(ava, monkeypatch, chunk, *messages):
    monkeypatch.setattr(agent.Agent.default, "llm_node", _Stream([chunk]))
    return [c async for c in ava.llm_node(history(*messages), [], None)]


def _helpdesk_agent():
    return agent.BenchmarkAgent("You staff a help desk.", [], "Hello.", tool_defs=HELPDESK)


async def test_unheard_number_is_asked_again_then_read_back_then_used(monkeypatch):
    ava = _helpdesk_agent()
    note = ("add_note", {"case_id": "case_1", "text": "caller rang"})
    padded = ("find_customer", {"contact_phone": "9165501940"})
    good = ("find_customer", {"contact_phone": "9165550194"})

    out = await _drive(ava, monkeypatch, _tool_chunk(note, padded, note), user(HEARD))
    assert [t.name for t in out[0].delta.tool_calls] == ["add_note"]  # calls before the bad one still run
    assert out[-1] == agent.PHONE_REASK

    out = await _drive(ava, monkeypatch, _tool_chunk(good), user(HEARD), assistant(agent.PHONE_REASK), user(SAID))
    assert out[0].delta.tool_calls == []
    assert out[-1] == agent.PHONE_READBACK.format(number="nine one six, five five five, zero one nine four")

    confirmed = [user(HEARD), assistant(agent.PHONE_REASK), user(SAID), assistant(out[-1]), user("Yes, that's it.")]
    out = await _drive(ava, monkeypatch, _tool_chunk(good), *confirmed)
    assert [t.name for t in out[0].delta.tool_calls] == ["find_customer"] and len(out) == 1


async def test_number_the_agent_already_read_back_is_not_held(monkeypatch):
    ava = _helpdesk_agent()
    said = [user(SAID), assistant("Got it, nine one six, five five five, zero one nine four. What is wrong?"),
            user("My screen is broken.")]
    out = await _drive(ava, monkeypatch, _tool_chunk(("find_customer", {"contact_phone": "9165550194"})), *said)
    assert [t.name for t in out[0].delta.tool_calls] == ["find_customer"] and len(out) == 1
    assert ava._phone_holds == 0


async def test_phone_holds_are_capped_per_call(monkeypatch):
    ava = _helpdesk_agent()
    def padded():  # a fresh chunk each time: the agent edits the one it is given
        return _tool_chunk(("find_customer", {"contact_phone": "9165501940"}))

    for _ in range(agent.MAX_PHONE_HOLDS):
        assert (await _drive(ava, monkeypatch, padded(), user(HEARD)))[-1] == agent.PHONE_REASK
    out = await _drive(ava, monkeypatch, padded(), user(HEARD))
    assert [t.name for t in out[0].delta.tool_calls] == ["find_customer"] and len(out) == 1


async def test_checks_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(agent, "CALL_STATE_CHECKS", False)
    out = await _drive(_helpdesk_agent(), monkeypatch,
                       _tool_chunk(("find_customer", {"contact_phone": "9165501940"})), user(HEARD))
    assert [t.name for t in out[0].delta.tool_calls] == ["find_customer"] and len(out) == 1


# ---------------------------------------------------------------- hanging up
def session(*items):
    return SimpleNamespace(history=SimpleNamespace(items=list(items)),
                           current_agent=SimpleNamespace(_id_consumers=CONSUMERS, _phone_args=PHONES))


def test_no_tool_check_mentions_a_number_the_caller_gave_for_a_tool_that_takes_one():
    gave = agent._hangup_checks(session(user("I'm new here."), assistant("Your number?"), user(SAID)))
    assert "The caller gave a phone number and find_customer takes one." in gave[0][1]
    none = agent._hangup_checks(session(user("I was born on 1990-03-15, bye.")))
    assert "phone number and" not in none[0][1]


def test_hangup_checks_state_only_what_the_history_shows():
    idle = agent._hangup_checks(session(user("hi"), assistant("How can I help?"), user("Never mind, bye.")))
    assert [k for k, _ in idle] == ["tools", "close"]
    assert "No tool has been called on this call." in idle[0][1]
    assert all(text.startswith("Not ended") for _, text in idle)

    done = [call("open_case", "c1", customer_id="cu_7"), output("open_case", "c1", {"case_id": "case_42"}),
            call("add_note", "c2", case_id="case_42", text="x"), output("add_note", "c2", {"ok": True})]
    (key, text), = agent._hangup_checks(session(*done))
    assert key == "close" and "case_42" not in text and "instructions require" not in text


def test_unused_id_is_reported_only_when_a_sibling_was_used():
    """Two cases opened, one written up: the other is worth mentioning. An id kind
    nobody passes on (the only case, never noted) says nothing."""
    two = [call("open_case", "c1", customer_id="cu_7"), output("open_case", "c1", {"case_id": "case_41"}),
           call("open_case", "c2", customer_id="cu_7"), output("open_case", "c2", {"case_id": "case_42"}),
           call("add_note", "c3", case_id="case_42", text="x"), output("add_note", "c3", {"ok": True})]
    assert agent._unused_ids(two, CONSUMERS) == [("case_id", "case_41", "open_case", ["add_note", "close_case"])]
    (_, text), = agent._hangup_checks(session(*two))
    assert 'open_case returned case_id "case_41"' in text and "add_note, close_case take it" in text
    assert "If your instructions require something for it, do that now." in text
    assert agent._unused_ids(two[:2], CONSUMERS) == []


async def test_each_hangup_check_is_put_once_then_the_call_ends(monkeypatch):
    spoken = []

    async def say(text):
        spoken.append(text)

    monkeypatch.setattr(agent, "get_job_context", lambda: SimpleNamespace(simulation_context=lambda: object()))

    idle = session(user("hi"), assistant("How can I help?"), user("Thanks, bye."))
    idle.say, idle.current_speech = say, None
    ctx = SimpleNamespace(session=idle)
    assert (await agent.end_call(ctx, closing_message="Bye")).startswith("Not ended")
    assert (await agent.end_call(ctx, closing_message="Bye")).startswith("Not ended")
    assert spoken == []  # nothing is said to the caller while the hang-up is held
    with pytest.raises(agent.StopResponse):
        await agent.end_call(ctx, closing_message="Bye")
    assert spoken == ["Bye"]

    busy = session(call("open_case", "c1", customer_id="cu_7"), output("open_case", "c1", {"case_id": "case_42"}),
                   call("add_note", "c2", case_id="case_42"), output("add_note", "c2", {"ok": True}))
    busy.say, busy.current_speech = say, None
    ctx = SimpleNamespace(session=busy)
    assert (await agent.end_call(ctx, closing_message="All set")).startswith("Not ended")
    with pytest.raises(agent.StopResponse):
        await agent.end_call(ctx, closing_message="All set")
    assert spoken == ["Bye", "All set"]


# ---------------------------------------------------------------- silence re-prompt
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
    monkeypatch.setattr(agent, "REPROMPT_AFTER_S", 0.02)
    monkeypatch.setattr(agent, "REPROMPT_AGAIN_AFTER_S", 0.2)
    monkeypatch.setattr(agent, "MAX_REPROMPTS", 2)
    s = FakeSession()
    agent._attach_silence_reprompt(s)
    s.emit("agent_state_changed", new_state="listening")
    await asyncio.sleep(0.1)  # past the first wait, well before the second
    assert len(s.replies) == 1


async def test_default_is_single_reprompt(monkeypatch):
    monkeypatch.setattr(agent, "REPROMPT_AFTER_S", 0.02)
    monkeypatch.setattr(agent, "REPROMPT_AGAIN_AFTER_S", 0.02)
    s = FakeSession()
    agent._attach_silence_reprompt(s)
    s.emit("agent_state_changed", new_state="listening")
    await asyncio.sleep(0.15)
    assert len(s.replies) == 1


# ---------------------------------------------------------------- the shipped definitions
@pytest.mark.parametrize("persona", ["appointments", "insurance"])
def test_definitions_are_loaded_untouched_and_yield_tables(persona):
    """The canonical prompt reaches the model byte for byte, fixture results come back
    unchanged, and the generic tables find something to work with in the real schemas."""
    d = ROOT / "definitions" / persona
    prompt, first, tools = agent._load_persona(persona)
    assert prompt == (d / "system-prompt.txt").read_text().rstrip("\n")
    assert first == (d / "first-message.txt").read_text().strip()
    defs = agent._load_tool_defs(persona)
    assert len(tools) == len(defs) + 1  # the contract's tools plus end_call
    assert agent._id_consumers(defs) and agent._required_ids(defs)
    assert all(n == 10 for args in agent._phone_arguments(defs).values() for n in args.values())
    assert agent._phone_arguments(defs)
