"""Deferred clarify on HTTP surfaces: deliver-and-end-turn for api_server (fork patch #14).

The messaging clarify flow (TurnRunner._ask_clarify_question -> adapter card -> blocking
``clarify_gateway.wait_for_response``) is structurally impossible on the Responses /
chat-completions lanes: the agent thread would hold an open HTTP stream waiting for a reply
the OWUI client structurally cannot type (its input is locked while a turn streams,
``disabled={generating}`` in v0.11.4 MessageInput.svelte), and api_server's ``send()`` is a
documented no-op, so nothing could render the card anyway.

The HTTP-consistent shape is the one the OWUI compaction lane already uses: **deliver the
question, close the turn, resolve on the next message.** When a clarify call happens on an
api_server turn this module registers the questions as *deferred* (module-level, the same
process hosts both the turn and its follow-up), and the tool returns immediately with
``outcome="deferred"`` plus a notice telling the model to present the numbered question as
its final reply. The user answers in a normal (unlocked) message; the inbound hook rewrites
that message into an explicit "[clarify answered] ..." line before the turn dispatches, so
the model resumes with a structured answer instead of a bare ``2``.

INTEGRATION (fork lines, shipped in git with this module):
    gateway/run_turn_runner.py, in _clarify_callback_sync (after the two delivery imports,
    before the ``_finite_viewer_turn()`` headless branch):
        from gateway.api_clarify_deferred import try_defer
        _deferred = try_defer(self._ctx, questions)
        if _deferred is not None:
            return _deferred
    gateway/run_inbound.py, right after the ``_hm_pending_reply_intercepts`` block in the
    admitted-message flow (``if _reply is not None: return _reply``):
        if is_internal:
            from gateway.api_clarify_deferred import maybe_rewrite_api_answer
            maybe_rewrite_api_answer(event, source, _quick_key)
    tools/clarify_tool.py, CLARIFY_SCHEMA description: outcome enum mentions ``deferred``.
    config: ``platform_toolsets.api_server: [hermes-api-server, clarify]`` exposes the tool
    (the stock composite is ``_core_without(..., "clarify", ...)``).

COERCION reuses tools.clarify_gateway's typed-reply matcher (numeric pick, exact label,
comma/space multi-select, "(Recommended)"-insensitive, the literal word "skip") by building
an unregistered ``_ClarifyEntry`` — deferred entries deliberately never enter the shared
registry, so the messenger text-intercept can never consume them.

LIFETIME: a deferred entry answers at most one question per inbound message (FIFO), expires
on the shared clarify timeout (``clarify_gateway.get_clarify_timeout``; <=0 = unlimited),
passes slash commands and unmatched prose straight through (prose on a choices prompt
abandons the entry, mirroring the messenger cancel-and-route rule), and dies with the
process — a gateway restart in between leaves the user's reply to flow as ordinary text.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

API_PLATFORM = "api_server"
_DEFERRED_OUTCOME = "deferred"


@dataclass
class _DeferredEntry:
    clarify_id: str
    session_key: str
    qid: str
    question: str                 # display copy the user was shown (already redacted)
    choices: Optional[list[str]]  # decorated labels, 1-based numbered for the user
    multi_select: bool
    created_at: float


_lock = threading.RLock()
_pending: dict[str, _DeferredEntry] = {}        # clarify_id -> entry
_index: dict[str, list[str]] = {}               # session_key -> [clarify_id] FIFO


def _platform_value(source: Any) -> str:
    platform = getattr(source, "platform", None)
    return str(getattr(platform, "value", platform) or "")


def _render_block(entries: list[_DeferredEntry]) -> str:
    """The numbered question block the model must reproduce in its final reply."""
    from tools.clarify_tool import RECOMMENDED_LABEL, strip_recommended
    lines: list[str] = []
    for e in entries:
        lines.append(f"Q: {e.question}")
        if e.choices:
            for i, choice in enumerate(e.choices, 1):
                label = strip_recommended(str(choice))
                tag = f" {RECOMMENDED_LABEL}" if str(choice) != label else ""
                kind = "(multi-select: reply e.g. \"1,3\")" if e.multi_select else ""
                lines.append(f"  {i}. {label}{tag} {kind}".rstrip())
            lines.append("  (or type \"skip\" to skip this question)")
        lines.append("")
    return "\n".join(lines).strip()


def _purge_expired_locked(session_key: Optional[str] = None) -> None:
    try:
        from tools.clarify_gateway import get_clarify_timeout
        timeout = get_clarify_timeout()
    except Exception:
        timeout = 3600
    now = time.time()
    keys = [session_key] if session_key else list(_index)
    for key in keys:
        if not key:
            continue
        ids = _index.get(key) or []
        for cid in list(ids):
            entry = _pending.get(cid)
            if entry is None:
                ids.remove(cid)
                continue
            if timeout > 0 and now - entry.created_at > timeout:
                _pending.pop(cid, None)
                ids.remove(cid)
                logger.info("Deferred clarify expired (session=%s id=%s)", key, cid)
        if not ids:
            _index.pop(key, None)


def try_defer(ctx: Any, questions: list[dict]) -> Optional[dict]:
    """TurnRunner hook: answer a clarify call on an api_server turn without blocking.

    Returns the clarify_tool reply dict (``outcome="deferred"`` + notice carrying the exact
    numbered block) or None when this is not an api_server surface (messenger turns
    proceed to the normal card-and-wait flow untouched).
    """
    if _platform_value(getattr(ctx, "source", None)) != API_PLATFORM:
        return None
    session_key = getattr(ctx, "session_key", None) or ""
    if not session_key or not questions:
        return None
    entries: list[_DeferredEntry] = []
    with _lock:
        _purge_expired_locked(session_key)
        for q in questions:
            offered = q.get("choices")
            choices = [str(c) for c in offered] if offered else None
            entry = _DeferredEntry(
                clarify_id=uuid.uuid4().hex[:10], session_key=session_key,
                qid=str(q.get("qid") or f"q{len(entries)}"),
                question=str(q.get("question") or ""),
                choices=choices, multi_select=bool(q.get("multi_select")),
                created_at=time.time())
            _pending[entry.clarify_id] = entry
            _index.setdefault(session_key, []).append(entry.clarify_id)
            entries.append(entry)
    if not entries:
        return None
    logger.info("Clarify deferred for HTTP surface (session=%s ids=%s): question delivered "
                "as the turn's final reply, answer coerced from the next message",
                session_key, [e.clarify_id for e in entries])
    notice = (
        "The clarify UI is unavailable on this HTTP surface, so the user has NOT seen these "
        "questions yet. Reply to the user with the questions below reproduced VERBATIM "
        "(keep the numbering and labels; you may add one lead-in sentence), then END your "
        "turn: no further tool calls, do not answer the questions yourself. The user's next "
        "message arrives already coerced to their choice(s), as a [clarify answered] line.\n\n"
        + _render_block(entries)
    )
    return {"answers": {}, "outcome": _DEFERRED_OUTCOME, "notice": notice}


def _rewrite(entry: _DeferredEntry, text: str) -> Optional[str]:
    """Coerce the inbound reply against ``entry``; returns the replacement message text,
    the ORIGINAL text (pass-through), or None to leave the entry armed and rewrite the
    message into a retry instruction."""
    from tools import clarify_gateway as cg
    from tools.clarify_tool import strip_recommended
    probe = cg._ClarifyEntry(
        clarify_id=entry.clarify_id, session_key=entry.session_key, question=entry.question,
        choices=list(entry.choices) if entry.choices else None,
        multi_select=entry.multi_select, awaiting_text=not entry.choices)
    value, reason = cg._coerce_text_response_detailed(probe, text)
    if value is None and reason == "prose":
        # Free prose at a choices prompt abandons the entry (messenger cancel-and-route
        # parity): the user moved on, so the question must not swallow a later bare "2".
        with _lock:
            _pending.pop(entry.clarify_id, None)
            ids = _index.get(entry.session_key) or []
            if entry.clarify_id in ids:
                ids.remove(entry.clarify_id)
                if not ids:
                    _index.pop(entry.session_key, None)
        return text
    if value is None:
        return None  # invalid_selection: caller keeps the entry armed, rewrites as retry
    with _lock:
        _pending.pop(entry.clarify_id, None)
        ids = _index.get(entry.session_key) or []
        if entry.clarify_id in ids:
            ids.remove(entry.clarify_id)
            if not ids:
                _index.pop(entry.session_key, None)
    if value == cg.SKIPPED:
        answered = "(skipped)"
    else:
        import json as _json
        if entry.multi_select and isinstance(value, str):
            try:
                value = _json.loads(value)
            except ValueError:
                pass
        if isinstance(value, list):
            answered = ", ".join(str(strip_recommended(v)) for v in value)
        else:
            answered = str(strip_recommended(value))
    tail = " Continue the work using this answer; do not re-ask."
    remaining = len(_index.get(entry.session_key) or [])
    if remaining:
        tail += (f" {remaining} clarify question(s) are still pending — present the next one "
                 "in this reply's closing and end the turn.")
    return f'[clarify answered] Q: "{entry.question}" -> A: {answered}.{tail}'


def maybe_rewrite_api_answer(event: Any, source: Any, session_key: str) -> bool:
    """Inbound hook (admitted internal events only): if this api_server session has a
    deferred clarify awaiting an answer, coerce this message into an explicit answer line.

    Returns True when the message was rewritten (or intentionally held for retry). Slash
    commands, empty text and unrelated prose pass straight through; prose at a choices
    prompt abandons the pending entry (mirrors the messenger cancel-and-route rule)."""
    if _platform_value(source) != API_PLATFORM or not session_key:
        return False
    text = getattr(event, "text", None)
    if not isinstance(text, str) or not text.strip() or text.strip().startswith("/"):
        return False
    with _lock:
        _purge_expired_locked(session_key)
        ids = _index.get(session_key) or []
        entry = _pending.get(ids[0]) if ids else None
    if entry is None:
        return False
    rewritten = _rewrite(entry, text)
    if rewritten is None:
        options = ", ".join(f"{i}. {c}" for i, c in enumerate(entry.choices or [], 1))
        event.text = (f'[clarify retry] The user replied "{text.strip()[:200]}" to your pending '
                      f'question "{entry.question}" but that is not a valid choice. Valid options: '
                      f"{options}. Re-ask it clearly in your reply and end the turn; their next "
                      "message is the answer.")
        logger.info("Deferred clarify kept armed after invalid selection (session=%s id=%s)",
                    session_key, entry.clarify_id)
        return True
    if rewritten != text:
        event.text = rewritten
        logger.info("Coerced deferred clarify answer (session=%s id=%s)",
                    session_key, entry.clarify_id)
    return True
