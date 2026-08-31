"""OpenTelemetry wiring for the surface-defect inspection service.

Why this exists
---------------
This service runs two models on every request -- the classical LBP+SVM baseline
and the ONNX EfficientNet -- and a single request duration cannot tell you which
one is costing the latency. They are traced as separate child spans, each
carrying its predicted label and confidence, so the two can be compared on live
traffic rather than only on the held-out test set.

That comparison is the point: the baseline scored 97.1% and the CNN 99.6% on a
static split, but agreement rate in production is a different question, and it
is answerable from these spans.

Configuration, in precedence order
----------------------------------
1. ``APPLICATIONINSIGHTS_CONNECTION_STRING`` set  -> export to Azure Monitor.
   This is what the Container App gets from Bicep; nothing else has to change.
2. ``OTEL_CONSOLE_EXPORT=1``                      -> print spans to stdout.
   For local development, and how the test suite asserts on span structure.
3. Neither                                        -> no-op.

The no-op case is deliberate and load-bearing. Tests, CI, a plain ``docker run``
and the Render deployment all run with no telemetry configured, and none of them
should acquire a hard dependency on an Azure SDK or fail because a collector is
unreachable. Instrumentation that can break the thing it observes is worse than
no instrumentation.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

# Resolved once by setup_telemetry(); until then every span is a no-op.
_tracer = None
_configured = False


def _console_exporter():
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    # Simple (not batched) so spans appear synchronously -- a batch processor
    # would make the tests race against a flush interval.
    return SimpleSpanProcessor(ConsoleSpanExporter())


def _azure_exporter(connection_string: str):
    from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    return BatchSpanProcessor(
        AzureMonitorTraceExporter(connection_string=connection_string)
    )


def setup_telemetry(app=None) -> bool:
    """Configure tracing. Returns True if an exporter was actually installed.

    Safe to call more than once and safe to call when nothing is configured;
    any failure here is logged and swallowed, because a broken exporter must
    never take the service down with it.
    """
    global _tracer, _configured
    if _configured:
        return _tracer is not None

    _configured = True
    connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "").strip()
    console = os.getenv("OTEL_CONSOLE_EXPORT", "").strip() == "1"

    if not connection_string and not console:
        logger.debug("Telemetry not configured; spans are no-ops.")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        resource = Resource.create(
            {
                "service.name": os.getenv("OTEL_SERVICE_NAME", "surface-defect-inspector"),
                "service.version": os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
                # Set by the CD pipeline to the deployed git SHA, so a latency
                # regression can be attributed to a specific commit.
                "service.instance.id": os.getenv("REVISION_SHA", "local"),
            }
        )
        provider = TracerProvider(resource=resource)

        if connection_string:
            provider.add_span_processor(_azure_exporter(connection_string))
        if console:
            provider.add_span_processor(_console_exporter())

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(__name__)

        if app is not None:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            # /health is polled by the Container Apps liveness and readiness
            # probes every 10-30s. Left in, it would be the overwhelming
            # majority of spans and the ingestion bill.
            FastAPIInstrumentor.instrument_app(
                app, tracer_provider=provider, excluded_urls="/health,/static/.*"
            )

        logger.info(
            "Telemetry configured (azure_monitor=%s, console=%s).",
            bool(connection_string),
            console,
        )
        return True
    except Exception:  # noqa: BLE001 -- observability must not break the service
        logger.exception("Telemetry setup failed; continuing without it.")
        _tracer = None
        return False


@contextmanager
def span(name: str, **attributes) -> Iterator[object]:
    """Trace a block of work, or do nothing at all if tracing is off.

    Used as::

        with span("rag.retrieve", k=4) as s:
            ...
            set_attributes(s, top_score=0.71)
    """
    if _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def set_attributes(current, **attributes) -> None:
    """Attach attributes to a span that may be a no-op (None)."""
    if current is None:
        return
    for key, value in attributes.items():
        if value is not None:
            current.set_attribute(key, value)


def is_configured() -> bool:
    """Whether an exporter is installed -- surfaced on /health."""
    return _tracer is not None
