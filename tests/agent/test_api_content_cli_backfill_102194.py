"""Repro for NousResearch/hermes-agent#102194.

CLI path never persists the ``api_content`` sidecar: when the current-turn
user row reaches the DB BEFORE ``build_turn_context`` stamps the sidecar
(close-persist ``_persist_active_session_before_close`` on the staged
``_pending_cli_user_message``, an early gateway/cross-process flush, or a
retry of a failed turn-start persist), the prologue stamp only lands on the
in-memory dict and the marker-based flush skips the already-written row.
``set_latest_user_api_content`` is only invoked from the in-place-compaction
branch, so on the normal flow the row keeps ``api_content = NULL`` and the
next turn replays the clean content — the request prefix diverges exactly at
the first decorated message and the provider prompt cache is missed on the
first call of every turn.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.turn_context import compose_user_api_content
from hermes_state import SessionDB
from tests.agent.test_api_content_sidecar import _FakeAgent, _build


def _agent_with_preexisting_row(tmp_path, session_id: str, content: str):
    """FakeAgent whose current-turn user row is ALREADY in the DB (without
    sidecar) before the prologue runs — exactly what a close-persist on the
    staged CLI dict produces. The in-turn persist no-ops for that row, the
    same way the marker-based flush skips already-written messages."""
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(session_id, source="cli")
    db.append_message(session_id, "user", content=content)

    agent = _FakeAgent()
    agent.session_id = session_id
    agent._session_db = db
    agent._session_db_created = True
    # Marker-skip: the early persist sees the row as already durable and
    # writes nothing.
    agent._persist_session = lambda _messages, _history=None: None
    agent._ensure_db_session = lambda: None
    return agent, db


class TestPreExistingUserRowBackfill:
    def test_prologue_backfills_sidecar_onto_preexisting_row(self, tmp_path):
        """Row inserted BEFORE the stamp (close-persist racing the staged CLI
        dict) must still receive the sidecar once the prologue composes it.
        Without the fix the row stays NULL and the next turn replays the
        clean content, breaking the prompt-cache prefix."""
        sid = "sess-102194"
        agent, db = _agent_with_preexisting_row(tmp_path, sid, "hello")
        try:
            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "PLUGIN-CTX"}],
            ):
                ctx = _build(agent)

            msg = ctx.messages[ctx.current_turn_user_idx]
            expected = compose_user_api_content("hello", "", "PLUGIN-CTX")
            # The live dict always carries the stamp (in-memory turn works).
            assert msg["api_content"] == expected

            # The durable row must carry the same sidecar; otherwise the
            # next turn rehydrates clean bytes and misses the cache.
            rows = db.get_messages_as_conversation(sid)
            assert len(rows) == 1
            assert rows[0]["content"] == "hello"
            assert rows[0].get("api_content") == expected
        finally:
            db.close()

    def test_no_backfill_without_injections(self, tmp_path):
        """No injection → no sidecar → the pre-existing clean row is left
        exactly as-is (no spurious UPDATE)."""
        sid = "sess-102194-b"
        agent, db = _agent_with_preexisting_row(tmp_path, sid, "hello")
        try:
            with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
                ctx = _build(agent)
            msg = ctx.messages[ctx.current_turn_user_idx]
            assert "api_content" not in msg
            rows = db.get_messages_as_conversation(sid)
            assert "api_content" not in rows[0]
        finally:
            db.close()

    def test_no_backfill_when_content_mismatch(self, tmp_path):
        """If the newest user row no longer matches the live content (e.g. a
        persist-override rewrote the durable value), the backfill must NOT
        stamp the sidecar onto the wrong row."""
        sid = "sess-102194-c"
        agent, db = _agent_with_preexisting_row(tmp_path, sid, "clean override text")
        try:
            with patch(
                "hermes_cli.plugins.invoke_hook",
                return_value=[{"context": "PLUGIN-CTX"}],
            ):
                ctx = _build(agent)
            msg = ctx.messages[ctx.current_turn_user_idx]
            # Live dict is stamped (live content "hello" ≠ durable row).
            assert msg["api_content"] == "hello\n\nPLUGIN-CTX"
            rows = db.get_messages_as_conversation(sid)
            assert rows[0]["content"] == "clean override text"
            assert "api_content" not in rows[0]
        finally:
            db.close()
