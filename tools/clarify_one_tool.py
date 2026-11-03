"""clarify_one: single-question clarify for the HTTP lane (fork patch #15, api_server/OWUI).

The global ``clarify`` schema teaches batch-first ("several INDEPENDENT questions
belong in ONE call") because CLI/TUI/desktop/messengers render one multi-question
form. That shape is wrong for the api_server lane (``gateway/api_clarify_deferred``):
the deferred flow delivers-and-ends-turn and answers exactly ONE entry per inbound
message, so a batched call strands N-1 questions as a text interview — and an OWUI
pipe never sees a card for the rest.

``clarify_one`` makes the lane's contract STRUCTURAL instead of prose-enforced:
one question per call by schema (no batch is even expressible), 2-3 choices of
<=80 chars to match the OWUI native ask_user card's clickable rows, no
multi_select (the card has no multi-pick; the HTTP flow answers one message at a
time). The description carries the delivery protocol; the deferred notice keeps
the per-call mechanics.

Reuses the global core verbatim: ``clarify_tool([q], callback)`` normalizes,
decorates "(Recommended)", redacts, observes via human_input_request, and shapes
the result — and the callback is the SAME ``agent.clarify_callback`` the gateway
binds (``run_turn_runner._clarify_callback_sync``), so ``api_clarify_deferred``'s
defer/coerce/guard machinery applies unchanged (it keys on platform + session,
never on tool name).

INTEGRATION (fork lines, shipped with this file):
    agent/inline_tool_executors.py, in _RAW_INLINE_TOOL_EXECUTORS after the
    "clarify" entry:
        "clarify_one": _callback_tool(
            "tools.clarify_one_tool", "clarify_one_tool", "clarify_callback",
            ("question", "question"), ("choices", "choices"),
        ),
    agent/tool_dispatch_helpers.py: add "clarify_one" to _NEVER_PARALLEL_TOOLS.
    agent/agent_runtime_helpers.py: add "clarify_one" to
        AGENT_RUNTIME_POST_HOOK_TOOL_NAMES.
    config: platform_toolsets.api_server: [hermes-api-server, clarify_one]
        (drop "clarify" from that lane only; every other platform keeps it).
    agent/prompt_builder.py api_server hint: name clarify_one, one-per-turn.

Registry discovery picks this file up automatically (registry.py
discover_builtin_tools AST-scans tools/ for registry.register calls).
"""

from tools.registry import registry, tool_error

MAX_QUESTION_CHARS = 500   # the OWUI ask_user card truncates at 500
MAX_CHOICE_CHARS = 80      # card option label cap (detail belongs in the question)

# No maxLength in the JSON schema (same discipline as CLARIFY_SCHEMA: grammar
# converters are hostile; the handler enforces the limits and errors loudly).
CLARIFY_ONE_SCHEMA = {
    "name": "clarify_one",
    "description": (
        "Ask the user EXACTLY ONE question when you need a decision or feedback. "
        "This surface delivers and answers one question per call: make the call, "
        "then in your final reply reproduce the numbered question block shown in "
        "the tool result's notice VERBATIM and END your turn — never hand-write a "
        "numbered question block that did not come from this turn's call. The "
        "user's next message arrives already coerced to their choice, as a "
        "[clarify answered] line. Chain follow-up questions as NEW calls, one per "
        "turn, after each answer. Put your recommended option FIRST (the display "
        "marks it '(Recommended)' and offers an 'Other' free-text row); omit "
        "choices for an open-ended question. Limits: question <= 500 characters; "
        "2-3 choices of <= 80 characters each (more or longer belong in the "
        "question text). Result: {responses: [one entry], outcome}; the response "
        "has status answered, skipped or unanswered (user_response is null unless "
        "answered); outcome says how the wait ended, with a notice saying why. "
        "Prefer deciding low-stakes questions yourself; dangerous-command "
        "confirmation is the terminal tool's job."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question (<= 500 characters). Any options belong in `choices`, never enumerated inside this text.",
            },
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 3,
                "description": "2-3 single-select options (<= 80 characters each, recommended first); omit for open-ended.",
            },
        },
        "required": ["question"],
    },
}


def clarify_one_tool(question=None, choices=None, callback=None) -> str:
    """One question -> the global clarify core with a one-entry array (identical
    normalization, decoration, redaction and result wire shape)."""
    if callback is None:
        from tools.clarify_tool import _UNAVAILABLE
        return tool_error(_UNAVAILABLE)
    from tools.clarify_tool import clarify_tool
    q = str(question or "").strip()
    if not q:
        return tool_error("question must be non-empty text.")
    if len(q) > MAX_QUESTION_CHARS:
        return tool_error(
            f"question is {len(q)} characters; the limit is {MAX_QUESTION_CHARS}. "
            "Tighten it, or move the detail into context around the call.")
    cleaned = None
    if choices is not None:
        if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
            return tool_error("choices must be an array of strings (2-3 options), or omitted for open-ended.")
        cleaned = [c.strip() for c in choices if c.strip()][:3]
        if len(cleaned) == 1:
            return tool_error(
                "choices carries one option — that is not a decision. Add a real "
                "alternative (2-3 total) or omit choices to ask open-ended.")
        over = [c for c in cleaned if len(c) > MAX_CHOICE_CHARS]
        if over:
            return tool_error(
                f"{len(over)} choice(s) exceed {MAX_CHOICE_CHARS} characters; the card "
                "truncates labels — shorten them or fold the detail into the question.")
    return clarify_tool([{"question": q, "choices": cleaned}], callback=callback)


def check_clarify_one_requirements() -> bool:
    """No external requirements -- availability is the toolset grant."""
    return True


registry.register(
    name="clarify_one",
    toolset="clarify_one",
    schema=CLARIFY_ONE_SCHEMA,
    handler=lambda args, **kw: clarify_one_tool(
        args.get("question"), args.get("choices"), callback=kw.get("callback")),
    check_fn=check_clarify_one_requirements,
    emoji="❓",
)
