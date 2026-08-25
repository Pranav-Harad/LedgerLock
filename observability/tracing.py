"""Observability layer for LedgerLock (OpenTelemetry & Langfuse).

Provides unified span tracing across:
- Tier 1 Deterministic Engine
- Tier 2 Agentic LangGraph Matcher
- Tier 3 RAG TDS Validator
- Multi-Agent CrewAI Audit & Report Generation
- FastAPI Endpoint Invocations
"""

from __future__ import annotations

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

logger = logging.getLogger("ledgerlock.observability")

# Configure in-memory or console exporter
try:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    _in_memory_exporter = InMemorySpanExporter()
except ImportError:
    class InMemorySpanExporter:
        def __init__(self):
            self._spans = []
        def export(self, spans):
            self._spans.extend(spans)
        def get_finished_spans(self):
            return self._spans
    _in_memory_exporter = InMemorySpanExporter()

_resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "ledgerlock-reconciliation")})
_provider = TracerProvider(resource=_resource)
_provider.add_span_processor(SimpleSpanProcessor(_in_memory_exporter))

if os.getenv("OTEL_CONSOLE_EXPORT", "false").lower() == "true":
    _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace.set_tracer_provider(_provider)
tracer = trace.get_tracer("ledgerlock.reconciliation")


class LangfuseObserver:
    """Langfuse trace coordinator for LLM generation capture."""

    def __init__(self):
        self.public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        self.host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        self._client = None
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        if self.public_key and self.secret_key:
            try:
                from langfuse import Langfuse
                self._client = Langfuse(
                    public_key=self.public_key,
                    secret_key=self.secret_key,
                    host=self.host,
                )
            except Exception as e:
                logger.debug("Langfuse client initialization info: %s", e)

    def trace_generation(
        self,
        name: str,
        prompt: str,
        response: str,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Capture an LLM prompt-response generation event in Langfuse."""
        if self._client:
            try:
                self._client.generation(
                    name=name,
                    input=prompt,
                    output=response,
                    model=model or "default-reconciliation-llm",
                    metadata=metadata or {},
                )
            except Exception as e:
                logger.debug("Langfuse event capture info: %s", e)


langfuse_observer = LangfuseObserver()


@contextmanager
def trace_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """Context manager wrapping execution inside an OpenTelemetry span."""
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for k, v in attributes.items():
                if isinstance(v, (bool, str, int, float)):
                    span.set_attribute(k, v)
                else:
                    span.set_attribute(k, str(v))
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(trace.StatusCode.ERROR, str(exc))
            raise


def get_collected_spans() -> list:
    """Retrieve all collected spans from the current session."""
    return _in_memory_exporter.get_finished_spans()
