"""Fork patch #14 — deferred clarify on api_server surfaces (gateway/api_clarify_deferred.py).

Kept as a fork test file so a future ``hermes-full-update`` rebase flags it alongside the
integration anchors if upstream ever ships its own HTTP clarify lane (at which point this
patch should be dropped, not rebased).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway import api_clarify_deferred as acd


class _Source:
    def __init__(self, platform):
        self.platform = platform


class _Ctx:
    def __init__(self, platform, session_key):
        self.source = _Source(platform)
        self.session_key = session_key


class _Event:
    def __init__(self, text):
        self.text = text


SK = "agent:main:api_server:dm:clarify-test"


@pytest.fixture(autouse=True)
def _clean_registry():
    acd._pending.clear()
    acd._index.clear()
    yield
    acd._pending.clear()
    acd._index.clear()


def _defer(questions, platform="api_server", key=SK):
    return acd.try_defer(_Ctx(platform, key), questions)


def test_messenger_surface_never_defers():
    assert _defer([{"qid": "q0", "question": "Q?", "choices": ["a", "b"]}],
                  platform="telegram") is None
    assert not acd._pending


def test_defer_returns_outcome_and_rendered_block():
    reply = _defer([{"qid": "q0", "question": "Which env?",
                     "choices": ["staging (Recommended)", "prod"], "multi_select": False}])
    assert reply is not None
    assert reply["outcome"] == "deferred" and reply["answers"] == {}
    assert "Q: Which env?" in reply["notice"] and "1. staging (Recommended)" in reply["notice"]


def test_numeric_answer_is_coerced_fifo():
    _defer([{"qid": "q0", "question": "Which env?", "choices": ["staging (Recommended)", "prod"]},
            {"qid": "q1", "question": "Which svc?", "choices": ["api", "cron"], "multi_select": True}])
    ev = _Event("1")
    assert acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)
    assert ev.text.startswith('[clarify answered] Q: "Which env?" -> A: staging.')
    assert "still pending" in ev.text
    ev = _Event("1,2")
    assert acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)
    assert "A: api, cron" in ev.text


def test_invalid_selection_retries_then_resolves():
    _defer([{"qid": "q0", "question": "Pick?", "choices": ["x (Recommended)", "y"]}])
    ev = _Event("9")
    assert acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)
    assert "not a valid choice" in ev.text          # armed, retry notice instead of a swallowed turn
    ev = _Event("y")
    assert acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)
    assert "-> A: y." in ev.text


def test_prose_abandons_a_choices_prompt():
    _defer([{"qid": "q0", "question": "Pick?", "choices": ["x (Recommended)", "y"]}])
    ev = _Event("actually, let's change topic entirely")
    assert acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)
    assert ev.text.startswith("actually")           # unchanged...
    ev = _Event("2")                               # ...and a later bare selection no longer hijacked
    assert not acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)


def test_open_entry_accepts_prose_and_skip():
    _defer([{"qid": "q0", "question": "Preferences?", "choices": None}])
    ev = _Event("blue with green accents")
    assert acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)
    assert "-> A: blue with green accents." in ev.text
    _defer([{"qid": "q0", "question": "Name?", "choices": None}])
    ev = _Event("skip")
    assert acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)
    assert "(skipped)" in ev.text


def test_slash_commands_pass_through_and_leave_entry_armed():
    _defer([{"qid": "q0", "question": "Pick?", "choices": ["x (Recommended)", "y"]}])
    ev = _Event("/help")
    assert not acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)
    ev = _Event("x")
    assert acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)


def test_expired_entry_is_ignored():
    _defer([{"qid": "q0", "question": "Pick?", "choices": ["x (Recommended)", "y"]}])
    for entry in acd._pending.values():
        entry.created_at -= 10 ** 5
    ev = _Event("1")
    assert not acd.maybe_rewrite_api_answer(ev, _Source("api_server"), SK)
    assert ev.text == "1"


def test_other_platform_answers_never_rewritten():
    _defer([{"qid": "q0", "question": "Pick?", "choices": ["x (Recommended)", "y"]}])
    ev = _Event("1")
    assert not acd.maybe_rewrite_api_answer(ev, _Source("bluebubbles"), SK)
    assert ev.text == "1"


def test_tool_contract_carries_deferred_outcome_and_notice():
    from tools.clarify_tool import clarify_tool
    result = json.loads(clarify_tool(
        [{"question": "Ship now?", "choices": ["yes", "no"]}],
        callback=lambda questions: _defer(questions)))
    assert result["outcome"] == "deferred"
    assert result["responses"][0]["status"] == "unanswered"
    assert "Q: Ship now?" in result["notice"]
