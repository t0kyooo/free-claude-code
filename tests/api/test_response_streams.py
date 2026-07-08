"""Tests for the egress terminal-error-frame guard (issue #1020).

The streaming response wrappers in :mod:`api.response_streams` wrap every
public SSE body in a guard that is a pure passthrough on success but, when the
body raises *after* HTTP ``200`` + headers are already committed, emits exactly
one protocol-specific terminal SSE frame and re-raises. This keeps the client
from observing an empty or truncated body (issue #1020).

Coverage here is at two levels:

* **Iterator level** — drain ``response.body_iterator`` directly and assert the
  concatenated text plus the re-raised exception.
* **ASGI end-to-end** — drive the real ``StreamingResponse.__call__`` with a
  capturing ``send`` to prove the terminal frame is flushed via ``send``
  before the re-raise closes the connection (no trailing empty
  ``more_body=False`` body).

Both wire formats (Anthropic Messages and OpenAI Responses) are exercised, plus
the ``BaseExceptionGroup``-before-``Exception`` ordering and the
``GeneratorExit`` / ``CancelledError`` no-frame paths that the guard must
re-raise unwritten.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Iterator, Mapping
from unittest.mock import MagicMock, patch

import pytest

from api.response_streams import (
    EGRESS_STREAM_INTERRUPTED_MESSAGE,
    anthropic_sse_streaming_response,
    openai_responses_sse_streaming_response,
)
from core.anthropic.stream_contracts import parse_sse_text
from core.anthropic.streaming import anthropic_terminal_error_frame
from core.openai_responses.adapter import OpenAIResponsesAdapter


@pytest.fixture
def trace_mock() -> Iterator[MagicMock]:
    """Patch the egress guard's trace sink so tests can assert the trace row."""
    with patch("api.response_streams.trace_event") as mock:
        yield mock


# ---------------------------------------------------------------------------
# Body / wrapper helpers
# ---------------------------------------------------------------------------

_ANTHROPIC_FRAME = anthropic_terminal_error_frame(EGRESS_STREAM_INTERRUPTED_MESSAGE)
# The Responses terminal frame is NOT byte-deterministic: ``egress_error_frame``
# builds a fresh ``ResponsesStreamAssembler`` per call, minting a new
# ``response_id`` (uuid) and ``created_at`` (``time.time()``). Tests therefore
# assert the frame *shape* (parsed event + status/model/error fields) rather
# than exact text.


async def _body_chunks(chunks: list[str]) -> AsyncGenerator[str]:
    for chunk in chunks:
        yield chunk


async def _body_then_raise(
    chunks: list[str], exc: BaseException
) -> AsyncGenerator[str]:
    for chunk in chunks:
        yield chunk
    raise exc


def _anthropic_wrapper(body: AsyncIterator[str]):
    return anthropic_sse_streaming_response(body)


def _responses_wrapper(body: AsyncIterator[str]):
    adapter = OpenAIResponsesAdapter()
    return openai_responses_sse_streaming_response(
        body,
        headers=adapter.sse_headers,
        emit_error_frame=lambda: adapter.egress_error_frame(
            EGRESS_STREAM_INTERRUPTED_MESSAGE
        ),
    )


def _egress_trace_calls(trace_mock: MagicMock) -> list[Mapping[str, object]]:
    return [
        call.kwargs
        for call in trace_mock.call_args_list
        if call.kwargs.get("event") == "api.response.egress_error_frame_emitted"
    ]


async def _drain(
    response,
) -> tuple[str, BaseException | None]:
    """Drain ``response.body_iterator``; capture concatenated text + any raise."""
    parts: list[str] = []
    raised: BaseException | None = None
    try:
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes | bytearray | memoryview):
                parts.append(bytes(chunk).decode("utf-8"))
            else:
                parts.append(str(chunk))
    except BaseException as exc:
        raised = exc
    return "".join(parts), raised


def _assert_anthropic_terminal(text: str) -> None:
    events = parse_sse_text(text)
    assert events, "no SSE events parsed from interrupted stream"
    assert events[-1].event == "error"
    assert events[-1].data == {
        "type": "error",
        "error": {"type": "api_error", "message": EGRESS_STREAM_INTERRUPTED_MESSAGE},
    }


def _assert_responses_terminal(text: str) -> None:
    events = parse_sse_text(text)
    assert events, "no SSE events parsed from interrupted stream"
    last = events[-1]
    assert last.event == "response.failed"
    assert last.data["type"] == "response.failed"
    response = last.data["response"]
    assert isinstance(response, dict)
    assert response["status"] == "failed"
    assert response["model"] == ""
    assert response["error"] == {
        "message": EGRESS_STREAM_INTERRUPTED_MESSAGE,
        "type": "api_error",
        "param": None,
        "code": None,
    }


# ---------------------------------------------------------------------------
# Passthrough: success is byte-identical, no frame, no trace
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_passthrough_yields_chunks_unchanged_no_frame(
    trace_mock: MagicMock,
) -> None:
    chunks = ["event: message_start\ndata: {}\n\n", "event: message_stop\ndata: {}\n\n"]
    response = _anthropic_wrapper(_body_chunks(chunks))

    text, raised = await _drain(response)

    assert raised is None
    assert text == "".join(chunks)
    assert "event: error" not in text
    assert trace_mock.call_count == 0


@pytest.mark.asyncio
async def test_responses_passthrough_yields_chunks_unchanged_no_frame(
    trace_mock: MagicMock,
) -> None:
    chunks = [
        'event: response.created\ndata: {"type":"response.created"}\n\n',
        'event: response.completed\ndata: {"type":"response.completed"}\n\n',
    ]
    response = _responses_wrapper(_body_chunks(chunks))

    text, raised = await _drain(response)

    assert raised is None
    assert text == "".join(chunks)
    assert "response.failed" not in text
    assert trace_mock.call_count == 0


# ---------------------------------------------------------------------------
# Pre-start raise: body raises before any chunk -> exactly the terminal frame
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_emits_terminal_frame_on_pre_start_raise(
    trace_mock: MagicMock,
) -> None:
    exc = RuntimeError("upstream failed before first byte")
    response = _anthropic_wrapper(_body_then_raise([], exc))

    text, raised = await _drain(response)

    assert text == _ANTHROPIC_FRAME
    _assert_anthropic_terminal(text)
    assert raised is exc
    calls = _egress_trace_calls(trace_mock)
    assert len(calls) == 1
    assert calls[0]["stage"] == "egress"
    assert calls[0]["source"] == "api"
    assert calls[0]["exc_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_responses_emits_terminal_frame_on_pre_start_raise(
    trace_mock: MagicMock,
) -> None:
    exc = RuntimeError("upstream failed before first byte")
    response = _responses_wrapper(_body_then_raise([], exc))

    text, raised = await _drain(response)

    events = parse_sse_text(text)
    assert [e.event for e in events] == ["response.failed"]
    _assert_responses_terminal(text)
    assert raised is exc
    calls = _egress_trace_calls(trace_mock)
    assert len(calls) == 1
    assert calls[0]["exc_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# Mid-stream raise: emitted chunks then the terminal frame, then re-raise
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_emits_frame_after_chunks_on_mid_stream_raise(
    trace_mock: MagicMock,
) -> None:
    chunks = [
        "event: message_start\ndata: {}\n\n",
        "event: content_block_start\ndata: {}\n\n",
    ]
    response = _anthropic_wrapper(_body_then_raise(chunks, ValueError("mid-stream")))

    text, raised = await _drain(response)

    assert "message_start" in text
    assert text.endswith(_ANTHROPIC_FRAME)
    _assert_anthropic_terminal(text)
    assert isinstance(raised, ValueError)
    calls = _egress_trace_calls(trace_mock)
    assert len(calls) == 1
    assert calls[0]["exc_type"] == "ValueError"


@pytest.mark.asyncio
async def test_responses_emits_frame_after_chunks_on_mid_stream_raise(
    trace_mock: MagicMock,
) -> None:
    chunks = ['event: response.created\ndata: {"type":"response.created"}\n\n']
    response = _responses_wrapper(_body_then_raise(chunks, ValueError("mid-stream")))

    text, raised = await _drain(response)

    assert "response.created" in text
    _assert_responses_terminal(text)
    assert isinstance(raised, ValueError)
    calls = _egress_trace_calls(trace_mock)
    assert len(calls) == 1
    assert calls[0]["exc_type"] == "ValueError"


# ---------------------------------------------------------------------------
# ExceptionGroup: matched before Exception (it subclasses BaseException), so
# the frame is emitted and the group is re-propagated, not swallowed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_emits_frame_on_exception_group_then_re_raises(
    trace_mock: MagicMock,
) -> None:
    inner = RuntimeError("tool-call assembly fan-out")
    exc = BaseExceptionGroup("escaped fan-out", [inner])
    response = _anthropic_wrapper(
        _body_then_raise(["event: message_start\ndata: {}\n\n"], exc)
    )

    text, raised = await _drain(response)

    _assert_anthropic_terminal(text)
    assert isinstance(raised, BaseExceptionGroup)
    assert list(raised.exceptions) == [inner]
    calls = _egress_trace_calls(trace_mock)
    assert len(calls) == 1
    assert calls[0]["exc_type"] == "ExceptionGroup"


@pytest.mark.asyncio
async def test_responses_emits_frame_on_exception_group_then_re_raises(
    trace_mock: MagicMock,
) -> None:
    inner = RuntimeError("tool-call assembly fan-out")
    exc = BaseExceptionGroup("escaped fan-out", [inner])
    response = _responses_wrapper(_body_then_raise([], exc))

    text, raised = await _drain(response)

    _assert_responses_terminal(text)
    assert isinstance(raised, BaseExceptionGroup)
    assert list(raised.exceptions) == [inner]
    calls = _egress_trace_calls(trace_mock)
    assert len(calls) == 1
    assert calls[0]["exc_type"] == "ExceptionGroup"


# ---------------------------------------------------------------------------
# Disconnect paths: GeneratorExit / CancelledError must re-raise UNWRITTEN
# (no terminal frame, no egress trace) so we don't flush into a dead socket.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_exit_emits_no_frame(trace_mock: MagicMock) -> None:
    """Closing the guard mid-iteration injects GeneratorExit at the yield.

    The body yields two chunks; after forwarding the first, the guard is
    suspended at its ``yield`` inside the ``try``. ``aclose()`` throws
    ``GeneratorExit`` there, the ``except GeneratorExit: raise`` clause
    re-raises it unwritten, and ``aclose()`` returns normally (successful
    close). No frame or egress trace is produced.
    """

    async def _body_two() -> AsyncGenerator[str]:
        yield "chunk1"
        yield "chunk2"

    body = _body_two()
    response = _anthropic_wrapper(body)

    first = await response.body_iterator.__anext__()
    assert first == "chunk1"

    await response.body_iterator.aclose()
    await body.aclose()

    assert trace_mock.call_count == 0


@pytest.mark.asyncio
async def test_cancelled_error_emits_no_frame(trace_mock: MagicMock) -> None:
    """Cancelling an iterating task injects CancelledError at ``await anext``.

    The body yields one chunk then blocks forever; the consumer task pulls
    that chunk and re-enters ``await anext(body)``, where cancellation lands.
    The ``except asyncio.CancelledError: raise`` clause re-raises unwritten,
    so no frame or egress trace is produced and the task is cancelled.
    """

    blocked = asyncio.Event()

    async def _body_one_then_block() -> AsyncGenerator[str]:
        yield "chunk1"
        blocked.set()
        await asyncio.Future()

    body = _body_one_then_block()
    response = _anthropic_wrapper(body)

    produced: asyncio.Queue[str] = asyncio.Queue()

    async def _consume() -> None:
        async for chunk in response.body_iterator:
            await produced.put(chunk)

    task = asyncio.create_task(_consume())
    await asyncio.wait_for(blocked.wait(), timeout=2)
    assert await asyncio.wait_for(produced.get(), timeout=2) == "chunk1"

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await body.aclose()

    assert trace_mock.call_count == 0


# ---------------------------------------------------------------------------
# ASGI end-to-end: the terminal frame is flushed through the real
# StreamingResponse before the re-raise closes the connection.
# ---------------------------------------------------------------------------


async def _drive_asgi(response) -> tuple[list[dict], BaseException | None]:
    sent: list[dict] = []

    async def send(message: Mapping[str, object]) -> None:
        sent.append(dict(message))

    async def receive() -> Mapping[str, object]:
        # Block until the streaming task is cancelled on disconnect.
        await asyncio.Future()
        return {"type": "http.disconnect"}

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}}
    raised: BaseException | None = None
    try:
        await response(scope, receive, send)
    except BaseException as exc:
        raised = exc
    return sent, raised


def _assert_frame_flushed_before_disconnect(
    sent: list[dict], raised: BaseException | None, leading_chunk: str
) -> str:
    """Assert the frame reached ``send`` before the raise closed the body.

    Guards issue #1020 at the ASGI layer: HTTP 200 + the streamed chunks + the
    terminal frame all reached ``send`` with ``more_body=True``, and the raise
    propagated out of ``StreamingResponse.__call__`` WITHOUT a final empty
    ``more_body=False`` body (which Starlette skips when the iterator raises out
    of the ``async for``). Returns the concatenated rendered body text so each
    protocol test can assert frame-shape (Anthropic exact, Responses parsed).
    """
    assert raised is not None, "interrupted stream did not re-raise to __call__"

    starts = [m for m in sent if m.get("type") == "http.response.start"]
    assert starts and starts[0]["status"] == 200

    bodies = [m for m in sent if m.get("type") == "http.response.body"]
    assert bodies, "no body chunks were sent before the connection closed"
    rendered = b"".join(bytes(by["body"]) for by in bodies).decode("utf-8")
    assert rendered.startswith(leading_chunk)
    # The terminal frame was the last thing flushed; Starlette did not send the
    # trailing empty ``more_body=False`` body, so the socket closes with the
    # parseable terminal event as its final chunk.
    assert bodies[-1].get("more_body") is True
    return rendered


@pytest.mark.asyncio
async def test_anthropic_frame_flushes_through_real_streaming_response(
    trace_mock: MagicMock,
) -> None:
    chunk = "event: message_start\ndata: {}\n\n"
    response = _anthropic_wrapper(_body_then_raise([chunk], RuntimeError("asgi boom")))

    sent, raised = await _drive_asgi(response)

    rendered = _assert_frame_flushed_before_disconnect(sent, raised, chunk)
    # Anthropic terminal frame is deterministic, so assert exact bytes.
    assert rendered == chunk + _ANTHROPIC_FRAME
    _assert_anthropic_terminal(rendered)


@pytest.mark.asyncio
async def test_responses_frame_flushes_through_real_streaming_response(
    trace_mock: MagicMock,
) -> None:
    chunk = 'event: response.created\ndata: {"type":"response.created"}\n\n'
    response = _responses_wrapper(_body_then_raise([chunk], RuntimeError("asgi boom")))

    sent, raised = await _drive_asgi(response)

    rendered = _assert_frame_flushed_before_disconnect(sent, raised, chunk)
    # Responses terminal frame mints a fresh id/created_at per call, so assert
    # shape (parsed event + status/model/error), not exact bytes.
    _assert_responses_terminal(rendered)
