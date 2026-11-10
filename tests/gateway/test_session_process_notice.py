"""A background process completion can reach the agent in the turn that is running now.

``display.background_process_delivery: notice`` hands the admitted completion to the running claim
instead of leaving it queued for the next turn. The transfer is the durable one the client-facing
``prompt.steer_queued`` uses: the row leaves the queue, so the drain can never claim it too, and a
delivery that reached nobody puts it back in its FIFO place instead of stranding it. The text rides
the tool result (a machine notice), never a user row and never the out-of-band user marker.
Without a live claim to take it, the completion stays queued and runs as the next turn, as before.
"""
import asyncio

import pytest

from agent.prompt_builder import NOTICE_DISPLAY_KIND
from agent.turn_iteration_prep import _inject_notice_after_newest_tool_result
from hermes_state_runtime import admit_session_input, claim_session_input, list_session_admissions
from tests.gateway.test_session_end_delegations import _local


class Agent:
    """The running turn's agent: records notices and refuses them when asked to.

    Models the real contract: a drain consumes the buffered text (marks the ticket consumed), and
    ``withdraw_notice`` returns True only while the turn never took it."""

    def __init__(self, notice_ok=True):
        self.noticed, self.notice_ok, self._consumed = [], notice_ok, set()

    def notice(self, text, ticket=None):
        self.noticed.append((text, ticket))
        return self.notice_ok

    def withdraw_notice(self, ticket):
        return ticket not in self._consumed

    def _drain_pending_notices(self):
        if not self.noticed:
            return None
        self._consumed.update(t for _, t in self.noticed)
        return "\n".join(t for t, _ in self.noticed)

    def clear_interrupt(self, *args, **kwargs):
        pass


def _delivery(runner, monkeypatch, mode):
    """The knob, as the runner resolves it for the ambient profile."""
    monkeypatch.setattr(runner, '_load_background_process_delivery', lambda: mode)


async def _running_turn(authority, ref, notice_ok=True):
    """A claim the authority is executing, with its agent adopted (the notice seam)."""
    admit_session_input(authority.db, epoch=authority.epoch, principal_id='test-owner',
                        session_id=ref.session_id, request_id='human', payload={'text': 'working'})
    claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=ref.session_id)
    agent = Agent(notice_ok=notice_ok)
    authority.executing[ref.session_id] = claim['generation']
    authority.adopted[ref.session_id] = (claim['generation'], agent)
    return agent, claim


def _completion_event(runner, authority, ref, name='proc_radar'):
    """The event the process watcher admits for a finished background process."""
    from gateway.platforms.event import MessageEvent
    live = authority.sessions[ref.session_id]
    entry = runner.session_store.lookup_by_session_key(live.route)
    text = '[finished] the radar finished'
    event = MessageEvent(text=text, source=live.source, internal=True, metadata={
        'gateway_session_key': live.route, 'gateway_session_id': entry.session_id,
        'display_kind': 'process_complete', 'display_text': 'Finished: ' + name})
    evt = {'type': 'completion', 'session_id': name, 'started_at': 1700000000.0,
           'command': 'radar', 'exit_code': 0, 'output': 'the radar finished',
           'session_key': live.route, 'parent_session_id': entry.session_id}
    return event, evt


@pytest.mark.asyncio
async def test_notice_delivery_hands_the_completion_to_the_running_turn(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _delivery(runner, monkeypatch, 'notice')
        agent, _claim = await _running_turn(authority, ref)
        event, evt = _completion_event(runner, authority, ref)

        from gateway.session_automation import admit_process_notice
        receipt = await admit_process_notice(runner, authority,
                                              runner._adapter_for_source(event.source), event, evt)

        # The completion reached the running turn instead of waiting for the next one — via the
        # machine-notice seam, ticketed by its admission id.
        assert [text for text, _ticket in agent.noticed] == ['[finished] the radar finished']
        assert agent.noticed[0][1] == receipt.admission_id
        rows = [row for row in list_session_admissions(authority.db, session_id=ref.session_id,
                                                       pending_only=False)
                if row['admission_id'] == receipt.admission_id]
        assert rows[0]['status'] == 'terminal' and rows[0]['outcome'] == 'steering', rows[0]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_a_refused_notice_queues_the_completion_again_instead_of_stranding_it(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _delivery(runner, monkeypatch, 'notice')
        agent, _claim = await _running_turn(authority, ref, notice_ok=False)
        event, evt = _completion_event(runner, authority, ref)

        from gateway.session_automation import admit_process_notice
        await admit_process_notice(runner, authority,
                                    runner._adapter_for_source(event.source), event, evt)

        assert agent.noticed, 'the agent was asked'
        completion = [row for row in list_session_admissions(authority.db, session_id=ref.session_id,
                                                             pending_only=False)
                      if row['request_id'].startswith('["completion"')]
        assert completion[0]['status'] == 'queued' and completion[0]['outcome'] is None, completion[0]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_notice_delivery_without_a_running_turn_leaves_the_completion_queued(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _delivery(runner, monkeypatch, 'notice')
        event, evt = _completion_event(runner, authority, ref)

        from gateway.session_automation import admit_process_notice
        await admit_process_notice(runner, authority,
                                    runner._adapter_for_source(event.source), event, evt)

        completion = [row for row in list_session_admissions(authority.db, session_id=ref.session_id,
                                                             pending_only=False)
                      if row['request_id'].startswith('["completion"')]
        assert completion[0]['status'] == 'queued', completion[0]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_queue_delivery_leaves_the_completion_for_the_next_turn(tmp_path, monkeypatch):
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        _delivery(runner, monkeypatch, 'queue')
        agent, _claim = await _running_turn(authority, ref)
        event, evt = _completion_event(runner, authority, ref)

        from gateway.session_automation import admit_process_notice
        await admit_process_notice(runner, authority,
                                    runner._adapter_for_source(event.source), event, evt)

        assert agent.noticed == [], 'queue delivery must not hand the notice to the running turn'
        completion = [row for row in list_session_admissions(authority.db, session_id=ref.session_id,
                                                             pending_only=False)
                      if row['request_id'].startswith('["completion"')]
        assert completion[0]['status'] == 'queued', completion[0]
    finally:
        authority.db.close()


def test_the_delivery_knob_reads_queue_unless_it_says_notice():
    from gateway.session_process_notice import delivery_mode
    assert delivery_mode('') == 'queue'
    assert delivery_mode(None) == 'queue'
    assert delivery_mode('queue') == 'queue'
    assert delivery_mode('notice') == 'notice'
    assert delivery_mode('NOTICE') == 'notice'
    assert delivery_mode('steer') == 'queue'
    assert delivery_mode('nonsense') == 'queue'


def test_machine_notice_is_a_user_row_with_machine_provenance_after_tool():
    agent = Agent()
    agent.notice('process radar finished', ticket='t1')
    drained = agent._drain_pending_notices()
    assert drained == 'process radar finished'

    messages = [
        {'role': 'assistant', 'content': 'working', 'tool_calls': [{'id': 'c1'}]},
        {'role': 'tool', 'tool_call_id': 'c1', 'content': 'build finished'},
    ]
    _inject_notice_after_newest_tool_result(agent, messages, drained)
    assert [m['role'] for m in messages] == ['assistant', 'tool', 'user']
    row = messages[2]
    assert row['display_kind'] == NOTICE_DISPLAY_KIND
    assert 'not written by a human user' in row['content']
    assert 'OUT-OF-BAND USER MESSAGE' not in row['content']


def test_a_notice_withdrawn_before_injection_stays_unconsumed():
    agent = Agent()
    agent.notice('a finished', ticket='w1')
    assert agent.withdraw_notice('w1') is True
    agent = Agent()
    agent.notice('a finished', ticket='w2')
    drained = agent._drain_pending_notices()
    assert drained == 'a finished'
    assert agent.withdraw_notice('w2') is False  # already consumed by the drain
