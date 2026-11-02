"""Manual compaction over HTTP for Responses-API sessions.

Committed patch on the fork's main (2026-11-02; was untracked checkout drift
2026-10-14..11-01, which made it invisible to install-workspace
re-materialization - the gateway served 404 while the clone carried the file).
The integration lines below are in git with this module.

Adds POST /v1/responses/{response_id}/compress to the api_server platform so an
external client (the OWUI hermes pipe's ~compact trigger) can force ONE manual
compaction of a stored response's conversation history — the missing lane
documented in .chat/HERMES-COMPACTION-CAPABILITY.md (auto compaction already
rides the stream as hermes.status; manual /compress exists only on messenger
slashes and the desktop session.compress RPC, which needs the in-process
session["agent"] that API-created sessions never have).

INTEGRATION (2 lines in gateway/platforms/api_server.py, shipped in git):
    # near the other platform-module imports:
    from gateway.platforms import api_server_compaction as _compact
    # inside the routes list, before routes.extend(...):
    routes.append(("POST", "/v1/responses/{response_id}/compress",
                   functools.partial(_compact.handle_compact_response, self)))

WIRE CONTRACT (v1):
    POST /v1/responses/{response_id}/compress      (API key auth, like every route)
    -> 200 {"object": "hermes.compression",
            "status": "compressed" | "ineffective" | "nothing_to_do" | "lock_skipped",
            "before_tokens": int, "after_tokens": int, "messages_removed": int,
            "source_response_id": str, "compressed_response_id": str | null}
    Token figures are TRANSCRIPT-only (estimate_messages_tokens_rough on both
    sides; the agent-level before/after estimates are not comparable, see
    comment at the call site). "ineffective" = the LLM digest did not come out
    smaller than the turns it replaces: NOTHING is committed, the tip is
    unchanged (a worse tip would be a trap for the caller).
    The SOURCE entry is never mutated (clients mid-chain on it are untouched);
    compression writes a NEW store entry with the same envelope + compressed
    history, and any `conversations` name whose tip was the source is remapped
    to the new id. The pipe switches its chain tip to compressed_response_id.

OPEN RISKS to verify at apply time (read before restarting a gateway with it):
  1. _compress_context(force=True) agent-surface: compress_now drives
     agent._compress_context; a bare AIAgent(model=..., quiet_mode=True)
     carries no system-prompt cache or tool schemas (estimate falls back via
     getattr defaults - conservative token math, no crash expected). Watch the
     first real compression's before/after numbers against the stream footer.
  2. model_routes aliases: runtime resolution below replicates only
     {model, provider, base_url, api_key}; a route needing custom headers /
     OAuth (Codex) compresses on the GLOBAL default instead - acceptable for
     local routes (all three here are vLLM-style), loud in the response via
     the resolved "model" field.
  3. Compression is a real summarizer LLM call (local vLLM, minutes on big
     histories); the handler runs it off the event loop with the same
     hygiene-style inactivity budget. Keep the pipe's HTTP timeout generous.
  4. Integration lines ship in git with this module (api_server.py imports
     `_compact` and appends the route) — `hermes update` carries them on the
     fork's main; see the upstream/main..main history.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Same spirit as compression.hygiene_timeout_seconds: bound the wait WITHOUT
# forward progress, never the summarizer call itself (the agent's own
# auxiliary/compression timeouts own that).
COMPACT_WAIT_CEILING_SECONDS = 600


def build_compaction_agent(adapter: Any, model: str) -> Any:
    """Minimal AIAgent that can drive the summarizer for ONE compress pass.

    Mirrors the chat lane's model_routes precedence (alias -> {model, provider,
    base_url, api_key} wins over raw name) WITHOUT toolsets, memory, reasoning
    config or session-db: compaction only needs a working chat backend.
    """
    from run_agent import AIAgent  # the chat lane's own import (api_server.py:2417)

    kwargs: dict[str, Any] = {
        "quiet_mode": True, "verbose_logging": False,
        "platform": "api_server_compaction",
    }
    route = (getattr(adapter, "_model_routes", None) or {}).get(model)
    if isinstance(route, dict):
        # A model_routes alias: chat-lane precedence, alias wins.
        kwargs["model"] = route.get("model") or model
        for key in ("provider", "base_url", "api_key"):
            if route.get(key):
                kwargs[key] = route[key]
    else:
        # The platform SERVE name (e.g. "hermes-agent") is not an engine model
        # id: sending it verbatim 404s at vLLM and the compressor silently
        # degrades to a deterministic fold that GROWS the context. Resolve the
        # same runtime the chat lane builds (api_server.py:2431-2437).
        from gateway.run import _resolve_gateway_model, _resolve_runtime_agent_kwargs
        try:
            runtime = _resolve_runtime_agent_kwargs()
        except RuntimeError as exc:  # provider-auth resolution (chat lane same)
            raise RuntimeError(f"compaction agent: provider auth failed: {exc}") from exc
        kwargs["model"] = runtime.pop("model", None) or _resolve_gateway_model()
        runtime.pop("_fallback_notice", None)
        kwargs.update(runtime)
    return AIAgent(**kwargs)


def _compress_sync(adapter: Any, model: str, history: list[dict[str, Any]]) -> Any:
    from agent.conversation_compression_manual import compress_now, parse_compress_args

    agent = build_compaction_agent(adapter, model)
    # Full compress, no boundary args (v1 keeps the pipe contract minimal;
    # partial forms are a one-line follow-up: parse_compress_args(body_args)).
    result = compress_now(agent, history, parse_compress_args(""), task_id="http-compact")
    try:
        from agent.conversation_compression import (
            finalize_context_engine_compression_notification)
        finalize_context_engine_compression_notification(
            agent, committed=(result.status == "compressed"))
    except Exception:
        logger.exception("[api_server_compaction] finalize notification failed")
    return result


async def handle_compact_response(adapter_self: Any, request: Any) -> Any:
    """POST /v1/responses/{response_id}/compress — see module docstring."""
    from aiohttp import web
    from gateway.platforms.api_server import _error_response

    auth_err = adapter_self._check_auth(request)
    if auth_err:
        return auth_err
    rid = request.match_info["response_id"]
    store = adapter_self._current_response_store()
    stored = store.get(rid)
    if stored is None:
        return _error_response(f"Response not found: {rid}", 404, code="response_not_found")

    history = list(stored.get("conversation_history") or [])
    model = ((stored.get("response") or {}).get("model")) or getattr(
        adapter_self, "_model_name", "hermes-agent")

    from agent.conversation_compression_manual import MIN_MESSAGES

    base = {"object": "hermes.compression", "source_response_id": rid, "model": model}
    if len(history) < MIN_MESSAGES:
        return web.json_response({
            **base, "status": "nothing_to_do",
            "reason": f"manual compression needs >= {MIN_MESSAGES} messages, "
                      f"this response carries {len(history)}",
            "before_tokens": 0, "after_tokens": 0, "messages_removed": 0,
            "compressed_response_id": None})

    # TRANSCRIPT-only token math for the wire contract. result.before/after_
    # tokens are NOT comparable: the before-estimate runs before any system
    # prompt/tool cache exists on the throwaway agent, while the after-
    # estimate (post _compress_context rebuild) counts the materialized
    # system prompt + tool schemas (live: 13k -> 22k on a real LLM summary
    # that had in fact shrunk the transcript's share). Same function on both
    # sides is the only apples-to-apples check, and it is what the caller's
    # next request actually grows or shrinks.
    from agent.model_metadata import estimate_messages_tokens_rough
    before_msgtoks = estimate_messages_tokens_rough(history)

    t0 = time.monotonic()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_compress_sync, adapter_self, model, history),
            timeout=COMPACT_WAIT_CEILING_SECONDS)
    except asyncio.TimeoutError:
        logger.error("[api_server] manual compaction of %s exceeded %ss ceiling",
                     rid, COMPACT_WAIT_CEILING_SECONDS)
        return _error_response("Compression exceeded the wait ceiling", 504,
                               code="compression_timeout")
    except Exception as exc:  # summarizer/config failures must stay loud
        logger.exception("[api_server] manual compaction of %s failed", rid)
        return _error_response(f"Compression failed: {exc}", 502, code="compression_failed")

    status = getattr(result, "status", "failed")
    payload = {
        **base, "status": status,
        "before_tokens": before_msgtoks,
        "after_tokens": before_msgtoks,
        "messages_removed": int(getattr(result, "removed", 0) or 0),
        "compressed_response_id": None,
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }
    if status != "compressed":
        # "nothing_to_do" / "lock_skipped" / "preview": honest passthrough.
        after_msgs = list(getattr(result, "after_messages", history) or history)
        payload["before_tokens"] = before_msgtoks
        payload["after_tokens"] = estimate_messages_tokens_rough(after_msgs)
        return web.json_response(payload)

    after_msgs = list(result.after_messages)
    after_msgtoks = estimate_messages_tokens_rough(after_msgs)
    if after_msgtoks >= before_msgtoks:
        # THE ENTRY IS NEVER COMMITTED when the handoff did not shrink the
        # transcript: on lean histories the LLM digest can outweigh the
        # middle it replaces (measured live: 15,215 -> 17,789 tokens on the
        # NEXT prompt). A compress call that returns a WORSE tip is a trap;
        # refuse honestly instead. The source stays the tip either way.
        payload["status"] = "ineffective"
        payload["before_tokens"] = before_msgtoks
        payload["after_tokens"] = after_msgtoks
        payload["messages_removed"] = 0
        payload["reason"] = ("the summary did not come out smaller than the "
                             "turns it replaces; nothing was committed and "
                             "the chain tip is unchanged")
        return web.json_response(payload)

    new_id = "resp_compact_" + uuid.uuid4().hex[:24]
    entry = dict(stored)
    entry["conversation_history"] = after_msgs
    payload["before_tokens"] = before_msgtoks
    payload["after_tokens"] = after_msgtoks
    store.put(new_id, entry)
    # Tip remap: a conversation name pointing at the source follows the
    # compression; names mid-chain further back must NOT (they are rewinds).
    try:
        rows = store._conn.execute(
            "SELECT name FROM conversations WHERE response_id = ?", (rid,)).fetchall()
        for (name,) in rows:
            store.set_conversation(name, new_id)
    except Exception:
        logger.exception("[api_server] compression tip remap failed for %s", rid)
    payload["compressed_response_id"] = new_id
    headers = {}
    sid = stored.get("session_id")
    if sid:
        headers["X-Hermes-Session-Id"] = str(sid)
    return web.json_response(payload, headers=headers)
