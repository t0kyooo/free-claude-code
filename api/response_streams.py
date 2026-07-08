"""FastAPI streaming response wrappers for public API wire formats."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping

from fastapi.responses import StreamingResponse

from core.anthropic.streaming import (
    ANTHROPIC_SSE_RESPONSE_HEADERS,
    anthropic_terminal_error_frame,
)
from core.trace import trace_event

EGRESS_STREAM_INTERRUPTED_MESSAGE = (
    "The upstream response stream ended unexpectedly; the request could not be "
    "completed."
)


def _trace_egress_failure(exc: BaseException) -> None:
    trace_event(
        stage="egress",
        event="api.response.egress_error_frame_emitted",
        source="api",
        exc_type=type(exc).__name__,
    )


async def _stream_with_terminal_error_frame(
    body: AsyncIterator[str],
    *,
    emit_error_frame: Callable[[], str],
) -> AsyncGenerator[str]:
    """Yield ``body`` unchanged, guaranteeing a terminal SSE frame on failure.

    A pure passthrough on success: each upstream chunk is yielded verbatim. When
    the body raises after the streaming response has already committed HTTP 200
    and headers (so the error can no longer be surfaced as a non-200 response),
    emit exactly one protocol-specific terminal error frame via
    ``emit_error_frame`` so the client observes a parseable terminal event
    instead of a truncated or empty body (issue #1020), then re-raise so the
    server-side traceback and the request-correlated
    ``api.response.stream_interrupted`` trace still fire.

    ``GeneratorExit`` and ``asyncio.CancelledError`` (the client has
    disconnected or the task was cancelled) are re-raised without writing: the
    frame would either fail to flush or loop while nothing is reading.
    ``BaseExceptionGroup`` is matched before ``Exception`` (it subclasses
    ``BaseException``, not ``Exception``) because escaped exception groups have
    been observed from fan-out in tool-call assembly and midstream recovery.
    """
    try:
        async for chunk in body:
            yield chunk
    except GeneratorExit:
        raise
    except asyncio.CancelledError:
        raise
    except BaseExceptionGroup as exc:
        _trace_egress_failure(exc)
        yield emit_error_frame()
        raise
    except Exception as exc:
        _trace_egress_failure(exc)
        yield emit_error_frame()
        raise


def anthropic_sse_streaming_response(body: AsyncIterator[str]) -> StreamingResponse:
    """Return a streaming response for Anthropic-style SSE streams.

    Guarantees a terminal ``error`` SSE event if the body raises after the
    response has started (issue #1020).
    """
    return StreamingResponse(
        _stream_with_terminal_error_frame(
            body,
            emit_error_frame=lambda: anthropic_terminal_error_frame(
                EGRESS_STREAM_INTERRUPTED_MESSAGE
            ),
        ),
        media_type="text/event-stream",
        headers=ANTHROPIC_SSE_RESPONSE_HEADERS,
    )


def openai_responses_sse_streaming_response(
    body: AsyncIterator[str],
    *,
    headers: Mapping[str, str],
    emit_error_frame: Callable[[], str],
) -> StreamingResponse:
    """Return a streaming response for OpenAI Responses-style SSE.

    Guarantees a terminal ``response.failed`` SSE event if the body raises after
    the response has started (issue #1020). ``emit_error_frame`` is supplied by
    the Responses handler via the shared ``OpenAIResponsesAdapter`` so the frame
    shape matches normal upstream failures exactly.
    """
    return StreamingResponse(
        _stream_with_terminal_error_frame(body, emit_error_frame=emit_error_frame),
        media_type="text/event-stream",
        headers=dict(headers),
    )
