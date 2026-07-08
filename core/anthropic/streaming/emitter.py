"""Anthropic SSE serialization helpers."""

import json
from typing import Any

from loguru import logger

ANTHROPIC_SSE_RESPONSE_HEADERS: dict[str, str] = {
    "X-Accel-Buffering": "no",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def map_stop_reason(openai_reason: str | None) -> str:
    """Map OpenAI ``finish_reason`` values to Anthropic ``stop_reason`` values."""
    return (
        STOP_REASON_MAP.get(openai_reason, "end_turn") if openai_reason else "end_turn"
    )


def format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Format one Anthropic-style SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


class AnthropicSseEmitter:
    """Serialize Anthropic SSE events and optionally log raw event bodies."""

    def __init__(self, *, log_raw_events: bool = False) -> None:
        self._log_raw_events = log_raw_events

    def event(self, event_type: str, data: dict[str, Any]) -> str:
        event = format_sse_event(event_type, data)
        if self._log_raw_events:
            logger.debug("SSE_EVENT: {} - {}", event_type, event.strip())
        return event


def anthropic_terminal_error_frame(message: str) -> str:
    """Serialize a terminal Anthropic SSE ``error`` event.

    Emitted by the API egress guard when a streaming response body raises after
    HTTP ``200`` + headers are already committed, so the client observes a
    parseable terminal event instead of an empty or truncated body (issue #1020).
    The frame shape matches the Anthropic mid-stream error protocol: a bare
    ``event: error`` carrying ``{type: error, error: {type, message}}``.
    """
    return format_sse_event(
        "error",
        {"type": "error", "error": {"type": "api_error", "message": message}},
    )
