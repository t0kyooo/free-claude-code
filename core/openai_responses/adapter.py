"""Facade for OpenAI Responses protocol adaptation."""

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from typing import Any, ClassVar

from .errors import ResponsesConversionError, openai_error_payload
from .events import OPENAI_RESPONSES_SSE_HEADERS
from .input import convert_request_to_anthropic_payload
from .stream import iter_responses_sse_from_anthropic
from .streaming.assembler import ResponsesStreamAssembler


class OpenAIResponsesAdapter:
    """Convert between OpenAI Responses and the proxy's Anthropic core path."""

    ConversionError: ClassVar[type[ResponsesConversionError]] = ResponsesConversionError
    sse_headers: ClassVar[dict[str, str]] = OPENAI_RESPONSES_SSE_HEADERS

    def to_anthropic_payload(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return convert_request_to_anthropic_payload(request)

    def iter_sse_from_anthropic(
        self,
        chunks: AsyncIterable[Any],
        request: Mapping[str, Any],
    ) -> AsyncIterator[str]:
        return iter_responses_sse_from_anthropic(chunks, request)

    def error_payload(self, *, message: str, error_type: str) -> dict[str, Any]:
        return openai_error_payload(message=message, error_type=error_type)

    def egress_error_frame(self, message: str) -> str:
        """Build a terminal ``response.failed`` SSE frame for an interrupted stream.

        Reuses the canonical ``ResponsesStreamAssembler.fail_response`` path so the
        response object shape matches normal upstream failures exactly (DRY). Emitted
        by the API egress guard when a streaming response body raises after HTTP 200
        + headers are already committed (issue #1020). The frame is a bare
        ``response.failed`` with no synthesized ``response.created``: a mid-stream
        failure has already emitted ``response.created``, and a bare
        ``response.failed`` remains a parseable terminal event for a pre-start
        failure.
        """
        assembler = ResponsesStreamAssembler({"model": ""})
        chunks = assembler.fail_response(
            {"error": {"type": "api_error", "message": message}}
        )
        return chunks[0]
