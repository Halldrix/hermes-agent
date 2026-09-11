"""Prune refuses under a live writer instead of joining the WAL (Refs #103339).

Bulk `sessions prune` deletes against a store a running gateway may own. The
repair path and `doctor --fix` already gate on `_live_writer_holds_db`; prune
was the remaining unguarded bulk writer. The gate runs before SessionDB() is
opened (the probe trips on any open handle, including our own). These drive
the real CLI (`hermes sessions prune --yes`) against the isolated test home:
a refused prune leaves every row and byte intact, and an idle store still
prunes. Read-only flows (`--dry-run`) never reach the gate.
"""

import sys
import time

import pytest

from hermes_state import SessionDB


def _mk_ended(sid):
    db = SessionDB()
    db.create_session(sid, source="cli")
    db.set_session_title(sid, "doomed task")
    with db._lock:
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, started_at=?, message_count=1 WHERE id=?",
            (time.time() - 100 * 86400, time.time() - 100 * 86400, sid),
        )
        db._conn.commit()
    db.close()


def _run_prune(monkeypatch, argv_tail):
    import hermes_cli.main as main_mod

    monkeypatch.setattr(
        sys, "argv", ["hermes", "sessions", "prune", *argv_tail])
    return main_mod.main()


def test_prune_refused_under_live_writer_leaves_db_untouched(
        monkeypatch, capsys):
    _mk_ended("20260101_000000_aaaaaa")

    holder = SessionDB()
    holder._conn.execute("BEGIN IMMEDIATE")  # live writer mid-write, like a gateway
    try:
        from hermes_state import _default_db_path
        before = _default_db_path().read_bytes()
        with pytest.raises(SystemExit) as exc:
            _run_prune(monkeypatch, ["--title", "doomed", "--yes"])
    finally:
        holder._conn.execute("ROLLBACK")
        holder.close()

    assert exc.value.code == 1
    assert "Refused" in capsys.readouterr().out
    check = SessionDB()
    try:
        assert check.get_session("20260101_000000_aaaaaa") is not None
    finally:
        check.close()
    from hermes_state import _default_db_path
    assert _default_db_path().read_bytes() == before


def test_prune_proceeds_when_no_live_writer(monkeypatch):
    _mk_ended("20260101_000000_aaaaaa")

    rc = _run_prune(monkeypatch, ["--title", "doomed", "--yes"])

    assert rc in (None, 0)
    check = SessionDB()
    try:
        assert check.get_session("20260101_000000_aaaaaa") is None
    finally:
        check.close()
