"""Machine notices — the gateway/plugins' mid-turn channel.

A machine notice is appended inside the newest tool result (an artifact the model reads as the
gateway speaking), never a user row and never the out-of-band user marker. This proves the real
agent machinery does that.
"""
from __future__ import annotations

import threading

import pytest

from agent.notice_inject import MACHINE_NOTICE_SUFFIX, inject_notice_after_newest_tool_result
from run_agent import AIAgent


def _bare_agent() -> AIAgent:
    agent = object.__new__(AIAgent)
    agent._pending_notice = None
    agent._pending_notice_lock = threading.Lock()
    agent._active_children = []
    agent._active_children_lock = threading.Lock()
    agent._tool_worker_threads = None
    agent._tool_worker_threads_lock = None
    agent._turn_user_intervened = False
    agent.quiet_mode = True
    agent.api_mode = "chat_completions"
    return agent


def test_notice_lands_inside_the_tool_result_not_a_user_row():
    agent = _bare_agent()
    assert agent.notice("process radar finished (exit 0)") is True
    notice = agent._drain_pending_notices()
    assert notice == "process radar finished (exit 0)"

    messages = [
        {"role": "user", "content": "working"},
        {"role": "assistant", "content": "running the tool", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "tool_call_id": "c1", "content": "build finished"},
    ]
    assert inject_notice_after_newest_tool_result(agent, messages, notice) is True
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool"], "a notice never introduces or moves a user row"
    tool = messages[2]
    assert MACHINE_NOTICE_SUFFIX in tool["content"]
    assert "process radar finished (exit 0)" in tool["content"]
    assert "[OUT-OF-BAND USER MESSAGE" not in tool["content"]


def test_notice_survives_a_whole_tool_batch_flush_cycle():
    """The notice is injected at the drain boundary (like the wrap-up notice): after a
    post-batch drain the notice joins the persistent tool row."""
    agent = _bare_agent()
    agent.notice("watch matched: threshold crossed")
    messages = [{"role": "tool", "tool_call_id": "c1", "content": "output"}]
    assert inject_notice_after_newest_tool_result(agent, messages, agent._drain_pending_notices()) is True
    assert "threshold crossed" in messages[0]["content"]


def test_notice_with_ticket_round_trips_through_drain_and_withdraw():
    agent = _bare_agent()
    agent.notice("delegator done", ticket="t1")
    # Not yet delivered into the turn: withdraw returns True (the transfer's settlement requeues it).
    assert agent.withdraw_notice("t1") is True

    agent = _bare_agent()
    agent.notice("delegator done", ticket="t2")
    notice = agent._drain_pending_notices()
    assert notice == "delegator done"
    assert agent.withdraw_notice("t2") is False   # drain consumed it


def test_empty_notice_is_refused():
    agent = _bare_agent()
    assert agent.notice("") is False
    assert agent.notice("   ") is False
