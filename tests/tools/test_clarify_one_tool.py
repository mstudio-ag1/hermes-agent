"""Fork patch #15 — clarify_one (tools/clarify_one_tool.py).

Schema-shape guarantees (structurally single-question, no multi_select, 2-3
choices), the handler's validation battery, and delegation through the global
clarify core (qid/decoration/result wire shape ride the shared path).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.clarify_one_tool import CLARIFY_ONE_SCHEMA, clarify_one_tool


def _cb(seen):
    def callback(questions):
        seen.append(questions)
        answers = {}
        for q in questions:
            opts = q["choices_offered"]
            answers[q["qid"]] = opts[0] if opts else "custom answer"
        return {"answers": answers, "outcome": "submitted"}
    return callback


def test_schema_shape_is_structurally_single_question():
    props = CLARIFY_ONE_SCHEMA["parameters"]["properties"]
    assert set(props) == {"question", "choices"}
    assert CLARIFY_ONE_SCHEMA["parameters"]["required"] == ["question"]
    assert props["choices"]["minItems"] == 2 and props["choices"]["maxItems"] == 3
    assert "multi_select" not in json.dumps(CLARIFY_ONE_SCHEMA)
    assert CLARIFY_ONE_SCHEMA["parameters"]["properties"]["choices"]["items"]["type"] == "string"


def test_single_question_delegates_to_global_core():
    seen = []
    result = json.loads(clarify_one_tool("Which env?", ["staging", "prod"], callback=_cb(seen)))
    assert len(seen) == 1 and len(seen[0]) == 1 and seen[0][0]["qid"] == "q0"
    assert seen[0][0]["choices"][0] == "staging (Recommended)"  # decoration rides the core
    assert result["outcome"] == "submitted"
    assert result["responses"][0]["status"] == "answered"
    assert result["responses"][0]["user_response"] == "staging"


def test_open_ended_question_passes_without_choices():
    seen = []
    result = json.loads(clarify_one_tool("Any preferences?", None, callback=_cb(seen)))
    assert seen[0][0]["choices"] is None
    assert result["responses"][0]["status"] == "answered"
    assert result["responses"][0]["user_response"] == "custom answer"


def test_blank_and_oversize_questions_are_rejected():
    assert "non-empty" in clarify_one_tool("  ", ["a", "b"], callback=_cb([]))
    assert "500" in clarify_one_tool("q" * 501, ["a", "b"], callback=_cb([]))


def test_single_option_and_long_options_are_rejected():
    assert "not a decision" in clarify_one_tool("Pick?", ["only"], callback=_cb([]))
    assert "80 characters" in clarify_one_tool("Pick?", ["ok", "x" * 81], callback=_cb([]))


def test_missing_callback_reports_unavailable():
    assert "not available" in clarify_one_tool("Which env?", ["staging", "prod"])
