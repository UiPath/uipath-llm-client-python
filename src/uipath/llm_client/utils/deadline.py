"""Execution-deadline propagation for LLM calls.

Serverless agent runs are terminated by the control plane after a fixed
execution window (15 minutes today). A single LLM call plus its automatic
retries could previously outlast that window: the fixed 895s request timeout
left no room for retries, the retry loop was bounded by attempt count rather
than elapsed time, and the process was force-killed mid-call — no logs were
flushed and spans never recorded a timeout.

This module lets the host runtime declare the run's hard deadline once, at
startup::

    from uipath.llm_client import set_execution_deadline

    # serverless window minus a safety buffer for graceful shutdown
    set_execution_deadline(15 * 60 - 10)

Every request sent through the shared retryable transports then:

* rewrites the ``X-UiPath-LLMGateway-TimeoutSeconds`` request header downward
  so the server-side timeout never exceeds the caller's remaining budget —
  the gateway ends an in-flight attempt at the deadline with a 504,
* caps backoff sleeps to the remaining budget and stops the retry loop once
  the deadline has passed, and
* fails fast with ``UiPathExecutionDeadlineError`` when the deadline has
  already passed, instead of starting an attempt with no budget.

When no deadline is set, behaviour is exactly as before.

The deadline is stored in a ContextVar holding a ``time.monotonic()`` timestamp.
Set it in the main task before any request-issuing tasks are spawned.
"""

import math
import time
from contextvars import ContextVar, Token

from httpx import Request

from uipath.llm_client.utils.exceptions import UiPathExecutionDeadlineError

LLM_GATEWAY_TIMEOUT_SECONDS_HEADER = "X-UiPath-LLMGateway-TimeoutSeconds"

_EXECUTION_DEADLINE: ContextVar[float | None] = ContextVar("_execution_deadline", default=None)


def set_execution_deadline(seconds_from_now: float) -> Token[float | None]:
    """Declare that the current run must finish within *seconds_from_now* seconds.

    The caller owns the safety buffer: pass the execution window minus
    whatever time graceful shutdown needs (log/span flushing, state save).

    Returns:
        A token that can be passed to :func:`clear_execution_deadline` to
        restore the previous value.
    """
    return _EXECUTION_DEADLINE.set(time.monotonic() + seconds_from_now)


def clear_execution_deadline(token: Token[float | None] | None = None) -> None:
    """Remove the execution deadline for the current context.

    Args:
        token: When given (from :func:`set_execution_deadline`), restores the
            previous value instead of unconditionally clearing.
    """
    if token is not None:
        _EXECUTION_DEADLINE.reset(token)
    else:
        _EXECUTION_DEADLINE.set(None)


def get_execution_deadline() -> float | None:
    """The run's deadline as a ``time.monotonic()`` timestamp, or None."""
    return _EXECUTION_DEADLINE.get()


def remaining_time_budget() -> float | None:
    """Seconds left until the execution deadline.

    Returns None when no deadline is set. Never negative — a passed deadline
    reports 0.0.
    """
    deadline = _EXECUTION_DEADLINE.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def apply_execution_deadline(request: Request) -> None:
    """Limit a single request attempt to the remaining execution budget.

    Applied by the shared transports to every outgoing LLM request (with or
    without retries). No-op when no deadline is declared. Otherwise:

    * raises :class:`UiPathExecutionDeadlineError` when the deadline has
      already passed,
    * lowers the server-side ``X-UiPath-LLMGateway-TimeoutSeconds`` timeout
      limit to the remaining budget, so the gateway ends an in-flight attempt
      at the deadline with a 504.

    Called once per attempt so retries see a freshly shrunk budget.
    """
    remaining = remaining_time_budget()
    if remaining is None:
        return
    if remaining <= 0:
        raise UiPathExecutionDeadlineError()

    timeout_limit = math.ceil(remaining)
    try:
        configured_timeout_limit = int(request.headers[LLM_GATEWAY_TIMEOUT_SECONDS_HEADER])
    except (KeyError, ValueError):
        configured_timeout_limit = None
    if configured_timeout_limit is None or timeout_limit < configured_timeout_limit:
        request.headers[LLM_GATEWAY_TIMEOUT_SECONDS_HEADER] = str(timeout_limit)


__all__ = [
    "LLM_GATEWAY_TIMEOUT_SECONDS_HEADER",
    "set_execution_deadline",
    "clear_execution_deadline",
    "get_execution_deadline",
    "remaining_time_budget",
    "apply_execution_deadline",
]
