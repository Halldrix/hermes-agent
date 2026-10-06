"""A vanished client is a distinct stop kind from a deliberate user stop (#84207).

``interrupt(stop_kind=...)`` records provenance; ``interrupt_issuer`` surfaces
``client_disconnect`` for it; the turn explainer names it to the user.
"""

from __future__ import annotations

import threading

from agent.interrupt_control import (
    STOP_KIND_CLIENT_DISCONNECT,
    STOP_KIND_USER_STOP,
    interrupt_issuer,
)
from tools.interrupt import set_interrupt


def _bare_agent():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._interrupt_requested = False
    agent._interrupt_message = None
    agent._tool_interrupt_reason = None
    agent._interrupt_stop_kind = None
    agent._hard_interrupt_requested = threading.Event()
    agent._execution_thread_id = None
    agent._interrupt_thread_signal_pending = False
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent.quiet_mode = True
    return agent


def test_client_disconnect_names_its_own_issuer():
    """A disconnect stop is attributed to the vanished client, not the user."""
    agent = _bare_agent()
    try:
        agent.interrupt("SSE client disconnected", stop_kind=STOP_KIND_CLIENT_DISCONNECT)
        assert agent._interrupt_stop_kind == STOP_KIND_CLIENT_DISCONNECT
        assert interrupt_issuer(agent) == STOP_KIND_CLIENT_DISCONNECT
    finally:
        set_interrupt(False)


def test_plain_user_stop_has_no_issuer():
    """Without stop_kind, attribution is unchanged: a human stop names no issuer."""
    agent = _bare_agent()
    try:
        agent.interrupt("user pressed stop")
        assert agent._interrupt_stop_kind is None
        assert interrupt_issuer(agent) is None
    finally:
        set_interrupt(False)


def test_user_stop_kind_has_no_issuer():
    """An explicit user stop_kind is recorded but still attributes to the user."""
    agent = _bare_agent()
    try:
        agent.interrupt("user pressed stop", stop_kind=STOP_KIND_USER_STOP)
        assert agent._interrupt_stop_kind == STOP_KIND_USER_STOP
        assert interrupt_issuer(agent) is None
    finally:
        set_interrupt(False)


def test_stop_kind_cleared_with_interrupt():
    """clear_interrupt drops the provenance with the rest of the interrupt state."""
    agent = _bare_agent()
    try:
        agent.interrupt("SSE client disconnected", stop_kind=STOP_KIND_CLIENT_DISCONNECT)
        agent.clear_interrupt()
        assert agent._interrupt_stop_kind is None
        assert interrupt_issuer(agent) is None
    finally:
        set_interrupt(False)


def test_disconnect_explainer_names_reconnect_not_continue():
    """The disconnect exit reason explains itself: reconnect, not `continue`."""
    from run_agent import AIAgent

    out = AIAgent._format_turn_completion_explanation(
        "interrupted_during_api_call(client_disconnect)"
    )
    assert "No reply:" in out
    lower = out.lower()
    assert "disconnect" in lower
    assert "reconnect" in lower


def test_plain_interrupt_explainer_unchanged():
    """A reason-less interrupt keeps the generic mid-call explanation."""
    from run_agent import AIAgent

    out = AIAgent._format_turn_completion_explanation("interrupted_during_api_call")
    assert "Send `continue` to retry" in out
    assert "disconnect" not in out.lower()
