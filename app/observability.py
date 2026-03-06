"""Langfuse observability setup.

Provides @observe decorator and score helpers.
When LANGFUSE_PUBLIC_KEY is not set, tracing is disabled (no-op).
"""
import logging

from app.config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

log = logging.getLogger(__name__)

_enabled = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

if _enabled:
    from langfuse import Langfuse, observe  # noqa: F401

    langfuse = Langfuse(
        public_key=LANGFUSE_PUBLIC_KEY,
        secret_key=LANGFUSE_SECRET_KEY,
    )
    log.info("Langfuse tracing enabled")
else:
    langfuse = None
    log.info("Langfuse tracing disabled (no keys configured)")

    def observe(*args, **kwargs):
        """No-op decorator when Langfuse is disabled."""
        if args and callable(args[0]):
            return args[0]
        def decorator(fn):
            return fn
        return decorator


def score_current_trace(name: str, value: float, comment: str = "") -> None:
    """Log a heuristic score on the current trace. No-op if tracing disabled."""
    if not langfuse:
        return
    try:
        langfuse.score_current_trace(
            name=name, value=value, data_type="NUMERIC", comment=comment,
        )
    except Exception as e:
        log.debug("Failed to log score %s: %s", name, e)


def flush() -> None:
    """Flush pending Langfuse events. Call on shutdown."""
    if langfuse:
        langfuse.flush()
