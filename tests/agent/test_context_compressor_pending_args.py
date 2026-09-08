"""Pending tool-call args survive the Pass-4 pressure pass (#105574)."""

import json
import logging
from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _make_compressor(**overrides):
    kwargs = dict(
        model="test/model",
        quiet_mode=True,
        protect_first_n=1,
        protect_last_n=2,
    )
    kwargs.update(overrides)
    with patch(
        "agent.context_compressor.get_model_context_length", return_value=100000
    ):
        return ContextCompressor(**kwargs)


def _delegate_call(call_id, goal_chars):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "delegate_task",
                "arguments": json.dumps(
                    {"goal": "G" * goal_chars, "tasks": ["t1", "t2"]}
                ),
            },
        }],
    }


def _tool_result(call_id, size):
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": "R" * size,
    }


def _goal_of(msg):
    args = msg["tool_calls"][0]["function"]["arguments"]
    return json.loads(args)["goal"]


class TestPendingArgsExemptFromPressure:
    def test_pending_call_survives_while_executed_call_shrinks(self):
        c = _make_compressor()
        old_args_len = len(json.dumps({"goal": "G" * 2000, "tasks": ["t1"]}))
        assert old_args_len > 500
        msgs = [
            {"role": "user", "content": "start " + "x" * 200},
            _delegate_call("call_old", 2000),
            _tool_result("call_old", 20000),
            _delegate_call("call_pending", 2000),
            {"role": "user", "content": "active ask"},
        ]
        pending_before = _goal_of(msgs[3])
        result, _ = c._prune_old_tool_results(
            msgs, protect_tail_count=4, protect_tail_tokens=100
        )
        pending_after = _goal_of(result[3])
        assert pending_after == pending_before
        assert len(pending_after) == 2000
        assert "...[truncated]" not in result[3]["tool_calls"][0]["function"]["arguments"]
        old_after = _goal_of(result[1])
        assert len(old_after) < 2000
        assert "...[truncated]" in result[1]["tool_calls"][0]["function"]["arguments"]

    def test_pressure_truncation_warns_with_tool_name(self, caplog):
        c = _make_compressor()
        msgs = [
            {"role": "user", "content": "start " + "x" * 200},
            _delegate_call("call_old", 2000),
            _tool_result("call_old", 20000),
            {"role": "user", "content": "active ask"},
        ]
        with caplog.at_level(logging.WARNING, logger="agent.context_compressor"):
            c._prune_old_tool_results(
                msgs, protect_tail_count=4, protect_tail_tokens=100
            )
        warnings = [
            r for r in caplog.records
            if "truncated tool-call args" in r.getMessage()
            and "delegate_task" in r.getMessage()
        ]
        assert warnings, "expected a warning naming delegate_task and the shrink"
