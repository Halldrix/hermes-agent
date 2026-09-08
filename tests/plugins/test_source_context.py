"""Contract tests for execution-scoped original-message context (#103941).

Covers the public surface in ``plugins.source_context``: the getter resolves
only inside an authorized native registry dispatch with a live execution
lease; the renderer emits ordinal-only boundaries with rebased spans; and
every failure mode (missing/contradictory metadata, altered presentation,
foreign/stale worker context, internal events, model-forged fields, quote
text without a transport reference) refuses mutations while keeping
authorized reads available.
"""

from __future__ import annotations

import copy

import pytest

from plugins import source_context as sc
from plugins.source_context import (
    SourceFragment,
    ToolSourceContext,
    bind_execution,
    clear_execution,
    get_tool_source_context,
    invalidate_source_fragments,
    merge_append_fragments,
    note_single_source,
    render_source_fragments,
    scoped_tool_call,
    source_context_allows_mutation,
    validate_fragments,
    verify_source_quote,
)


def _frag(
    start: int,
    end: int,
    namespace: str = "weixin",
    message_id=None,
    reference: str = "ref-1",
    **kw,
) -> SourceFragment:
    return SourceFragment(
        namespace=namespace,
        message_id="mid-1" if message_id is None else message_id,
        reference=reference,
        start=start,
        end=end,
        **kw,
    )


_LEAKS: list = []


@pytest.fixture(autouse=True)
def _clean_leaks():
    yield
    while _LEAKS:
        try:
            clear_execution(_LEAKS.pop())
        except Exception:
            pass


def _bind(text="hello\nworld", frags=None, **kw):
    if frags is None:
        frags = (_frag(0, 5), _frag(6, 11, message_id="mid-2"))
    params = {"text": text, "fragments": frags, "session_key": "sess-1",
              "run_generation": 7}
    params.update(kw)
    token, eid = bind_execution(**params)
    _LEAKS.append(token)
    return token, eid


@pytest.fixture
def bound():
    _, eid = _bind()
    return eid


def test_getter_returns_none_outside_dispatch(bound):
    assert get_tool_source_context() is None


def test_getter_resolves_inside_scoped_call(bound):
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert ctx.execution_id == bound
        assert ctx.run_generation == 7
        assert ctx.session_key == "sess-1"
        assert ctx.complete is True
    assert get_tool_source_context() is None


def test_cleared_lease_yields_none():
    token, _ = _bind()
    _LEAKS.remove(token)
    clear_execution(token)
    with scoped_tool_call():
        assert get_tool_source_context() is None


def test_foreign_execution_id_yields_none(bound):
    forged = ToolSourceContext(
        execution_id="forged-by-model", run_generation=7, session_key="sess-1",
        scope="sess-1", fragments=(_frag(0, 5),), text_hash="x", complete=True,
    )
    token = sc._CALL_SCOPE.set(forged)
    try:
        assert get_tool_source_context() is None
    finally:
        sc._CALL_SCOPE.reset(token)


def test_copied_worker_context_unusable_after_reset(bound):
    with scoped_tool_call():
        snapshot = copy.copy(sc._CALL_SCOPE.get())
    assert snapshot is not None
    # Turn ends: the lease dies; the copied record must not resolve anymore.
    while _LEAKS:
        clear_execution(_LEAKS.pop())
    token = sc._CALL_SCOPE.set(snapshot)
    try:
        assert get_tool_source_context() is None
    finally:
        sc._CALL_SCOPE.reset(token)


def test_render_ordinal_boundaries_and_rebased_spans():
    text = "helloworld"
    frags = (_frag(0, 5), _frag(5, 10, message_id="mid-2"))
    rendered, spans = render_source_fragments(text, frags)
    assert rendered == "[1]hello[2]world"
    assert spans == [
        {"ordinal": 1, "start": 3, "end": 8},
        {"ordinal": 2, "start": 11, "end": 16},
    ]
    assert "mid-1" not in rendered and "weixin" not in rendered


def test_render_empty_fragments_passthrough():
    assert render_source_fragments("abc", ()) == ("abc", [])


def test_merge_append_rebases_offsets():
    merged, complete = merge_append_fragments(
        "hello", (_frag(0, 5),), "world", (_frag(0, 5, message_id="mid-2"),),
        "hello\nworld",
    )
    assert complete is True
    assert [(f.start, f.end) for f in merged] == [(0, 5), (6, 11)]


def test_merge_missing_metadata_flags_incomplete():
    merged, complete = merge_append_fragments("hello", (), "world",
                                              (_frag(0, 5),), "hello\nworld")
    assert complete is False
    assert [(f.start, f.end) for f in merged] == [(6, 11)]


def test_merge_contradictory_spans_fails_closed():
    merged, complete = merge_append_fragments(
        "hello", (_frag(0, 99),), "world", (_frag(0, 5),), "hello\nworld",
    )
    assert complete is False
    assert merged == ()


def test_merge_empty_side_needs_no_separator():
    merged, complete = merge_append_fragments(
        "", (), "world", (_frag(0, 5),), "world",
    )
    assert complete is True
    assert [(f.start, f.end) for f in merged] == [(0, 5)]


def test_validate_fragments_rejects_overlap_and_oob():
    assert validate_fragments("hello", (_frag(0, 3), _frag(2, 5))) is False
    assert validate_fragments("hi", (_frag(0, 9),)) is False
    assert validate_fragments("hello", (_frag(0, 5),)) is True


def test_altered_presentation_refuses_mutation(bound):
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert source_context_allows_mutation(ctx, text="hello\nworld") is True
        assert source_context_allows_mutation(ctx, text="hello\nWORLD") is False


def test_internal_events_never_authorize_mutation():
    _bind("hello", (_frag(0, 5),), internal=True)
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None  # authorized read scope still exists
        assert source_context_allows_mutation(ctx) is False


def test_read_scope_without_ids_never_grants_write():
    frags = (SourceFragment(namespace="weixin", start=0, end=5),)
    _bind("hello", frags)
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert source_context_allows_mutation(ctx) is False


def test_verify_quote_matches_bound_presentation(bound):
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert verify_source_quote(ctx, "hello\nworld", 2, "world") is True
        assert verify_source_quote(ctx, "hello\nworld", 2, "WORLD") is False
        assert verify_source_quote(ctx, "hello\nworld", 9, "world") is False


def test_verify_quote_requires_transport_reference():
    frags = (SourceFragment(namespace="weixin", start=0, end=5),)
    _bind("hello", frags)
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert verify_source_quote(ctx, "hello", 1, "hello") is False


def test_model_forged_record_rejected(bound):
    with scoped_tool_call():
        live = get_tool_source_context()
        assert live is not None
        forged = ToolSourceContext(
            execution_id=live.execution_id, run_generation=live.run_generation,
            session_key=live.session_key, scope=live.scope,
            fragments=live.fragments, text_hash="tampered", complete=True,
        )
        assert source_context_allows_mutation(forged) is False
        assert verify_source_quote(forged, "hello\nworld", 1, "hello") is False


class _Event:
    def __init__(self, text):
        self.text = text
        self.source_fragments = ()


def test_clear_does_not_touch_process_environ(bound):
    import os

    # P1-4 regression: turn-local clear must not mutate process-global
    # session authority.  os.environ may hold a legacy/CLI fallback; the
    # gateway moved to task-local ContextVars — a turn finishing must not
    # delete a sibling turn's mirror or the process fallback.
    os.environ["HERMES_SESSION_ID"] = "stale-from-prior-turn"
    token = _LEAKS[-1]
    assert "HERMES_SESSION_ID" in os.environ
    # simulate sibling turn B establishing the same mirror concurrently:
    # A clearing must not wipe it.  We hold B's value in a second var.
    os.environ["HERMES_SESSION_ID"] = "from-sibling-turn-B"
    _LEAKS.remove(token)
    clear_execution(token)
    # process env must be untouched — only ContextVar + lease revocation
    assert os.environ.get("HERMES_SESSION_ID") == "from-sibling-turn-B"
    # cleanup the proc env this test dirtied
    os.environ.pop("HERMES_SESSION_ID", None)


def test_partial_coalescer_provenance_must_not_authorize():
    # P1-1: hello without provenance + world with provenance -> no mutation auth
    hello_text, world_text = "hello", "world"
    hello_frags = ()  # text-bearing side with no provenance
    world_frags = (_frag(0, 5, message_id="m2"),)
    # simulate the coalescer rebase (hello + "\n" + world) without provenance flag
    from dataclasses import replace

    world_event = _Event(world_text)
    world_event.source_fragments = world_frags
    hello_event = _Event(hello_text)
    hello_event.source_fragments = hello_frags
    # use the helper that propagates incomplete: rebase keeps span but marks incomplete
    from plugins.source_context import rebase_event_fragments

    hello_event.text = world_text  # start as second only for helper test
    hello_event.text = hello_text
    hello_event.source_fragments = hello_frags
    merged_frags, merge_complete = __import__("plugins.source_context", fromlist=["merge_append_fragments"]).merge_append_fragments(
        hello_text, hello_frags, world_text, world_frags, hello_text + "\n" + world_text,
    )
    assert merge_complete is False
    # bind the merged presentation — must stay non-authorizing
    text = hello_text + "\n" + world_text
    merged = tuple(replace(f, complete=False) for f in merged_frags) if merged_frags else ()
    token, _ = _bind(text, merged)
    try:
        with scoped_tool_call():
            ctx = get_tool_source_context()
            assert ctx is not None
            assert ctx.complete is False
            assert source_context_allows_mutation(ctx) is False
            assert source_context_allows_mutation(ctx, text=text) is False
    finally:
        if token in _LEAKS:
            _LEAKS.remove(token)
        clear_execution(token)


def test_pre_gateway_dispatch_rewrite_must_invalidate():
    # P1-2: dataclasses.replace(event, text=rewritten) + invalidate
    import dataclasses

    from gateway.platforms.event import MessageEvent, MessageType

    event = MessageEvent(text="hello", message_type=MessageType.TEXT)
    note_single_source(event, namespace="wecom", message_id="m1", reference="r1")
    rewritten = dataclasses.replace(event, text="REWRITTEN PAYLOAD THAT IS LONG ENOUGH")
    invalidate_source_fragments(rewritten)
    token, _ = _bind(rewritten.text, tuple(rewritten.source_fragments), session_key="sess-1", run_generation=7)
    try:
        with scoped_tool_call():
            ctx = get_tool_source_context()
            assert ctx is not None
            assert ctx.complete is False
            assert source_context_allows_mutation(ctx) is False
    finally:
        if token in _LEAKS:
            _LEAKS.remove(token)
        clear_execution(token)


def test_auto_skill_prefix_shift_exposes_post_shift_context():
    # P1-3: shift (prefix injection) must be visible through the bound context
    from gateway.platforms.event import MessageEvent, MessageType
    from plugins.source_context import shift_event_fragments

    event = MessageEvent(text="user hello", message_type=MessageType.TEXT)
    note_single_source(event, namespace="wecom", message_id="m1", reference="r1")
    prefix = "[skill: my-skill] payload\n\n"
    pre = event.text or ""
    event.text = prefix + pre
    shift_event_fragments(event, len(event.text) - len(pre))
    token, _ = _bind(event.text, tuple(event.source_fragments), session_key="sess-1", run_generation=9)
    try:
        with scoped_tool_call():
            ctx = get_tool_source_context()
            assert ctx is not None
            assert ctx.complete is True
            assert ctx.text_hash != ""
            assert ctx.fragments[0].start == len(prefix)
    finally:
        if token in _LEAKS:
            _LEAKS.remove(token)
        clear_execution(token)


def test_abort_event_set_on_clear():
    token, eid = _bind()
    _LEAKS.remove(token)
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert ctx.abort_cancelled is not None
        assert ctx.abort_cancelled.is_set() is False
        cancel_evt = ctx.abort_cancelled
    clear_execution(token)
    assert cancel_evt.is_set() is True
    assert sc._lease_generation(eid) is None


def test_fragment_count_property(bound):
    with scoped_tool_call():
        ctx = get_tool_source_context()
        assert ctx is not None
        assert ctx.fragment_count == 2
    assert SourceFragment(namespace="x", start=0, end=0).complete is True


def test_clear_is_idempotent(bound):
    token = _LEAKS.pop()
    clear_execution(token)
    clear_execution(token)  # early-return + funnel clears must never raise
    with scoped_tool_call():
        assert get_tool_source_context() is None


def test_thread_inherited_scope_resolves_while_lease_live(bound):
    import threading
    from contextvars import copy_context

    seen: list = []
    with scoped_tool_call():
        worker_ctx = copy_context()

        def _reader():
            found = get_tool_source_context()
            seen.append(found.execution_id if found else None)

        worker = threading.Thread(target=lambda: worker_ctx.run(_reader))
        worker.start()
        worker.join()
    assert seen == [bound]


def test_note_single_source_and_invalidate():
    event = _Event("hello")
    note_single_source(event, namespace="wecom", message_id="m1", reference="r1")
    assert len(event.source_fragments) == 1
    frag = event.source_fragments[0]
    assert (frag.start, frag.end) == (0, 5)
    assert frag.message_id == "m1" and frag.reference == "r1"
    invalidate_source_fragments(event)
    assert event.source_fragments == ()


def test_merge_through_pending_event_shape():
    from gateway.platforms.base import merge_pending_message_event
    from gateway.platforms.event import MessageEvent, MessageType

    first = MessageEvent(text="hello", message_type=MessageType.TEXT)
    note_single_source(first, namespace="weixin", message_id="m1")
    second = MessageEvent(text="world", message_type=MessageType.TEXT)
    note_single_source(second, namespace="weixin", message_id="m2")
    pending: dict = {}
    merge_pending_message_event(pending, "key", first)
    merge_pending_message_event(pending, "key", second, merge_text=True)
    merged_event = pending["key"]
    assert merged_event.text == "hello\nworld"
    assert [(f.start, f.end) for f in merged_event.source_fragments] == [(0, 5), (6, 11)]
    assert [f.message_id for f in merged_event.source_fragments] == ["m1", "m2"]


def test_caption_merge_rebases_with_double_newline():
    from gateway.platforms.base import merge_pending_message_event
    from gateway.platforms.event import MessageEvent, MessageType

    photo = MessageEvent(text="caption-a", message_type=MessageType.PHOTO,
                         media_urls=["/tmp/a.jpg"], media_types=["image"])
    note_single_source(photo, namespace="weixin", message_id="m1")
    follow = MessageEvent(text="caption-b", message_type=MessageType.PHOTO,
                          media_urls=["/tmp/b.jpg"], media_types=["image"])
    note_single_source(follow, namespace="weixin", message_id="m2")
    pending: dict = {}
    merge_pending_message_event(pending, "key", photo)
    merge_pending_message_event(pending, "key", follow)
    merged_event = pending["key"]
    assert merged_event.text == "caption-a\n\ncaption-b"
    assert [(f.start, f.end) for f in merged_event.source_fragments] == [(0, 9), (11, 20)]


def test_shift_preserves_user_spans_after_prefix_injection():
    from plugins.source_context import shift_event_fragments

    event = _Event("hello")
    note_single_source(event, namespace="weixin", message_id="m1")
    event.text = "PREFIX\n\nhello"
    shift_event_fragments(event, len("PREFIX\n\n"))
    assert [(f.start, f.end) for f in event.source_fragments] == [(8, 13)]
