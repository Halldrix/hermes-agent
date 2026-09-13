"""Read-only observational openers outside the #109725/#110026/#110186 family.

``main._session_db``, terminal breadcrumb resolution, the TUI exit summary,
and console ``sessions list/stats/export`` only query state: they must open
``SessionDB(read_only=True)`` so a nested inspection never mints a second
writable WAL handle while a gateway owns the store. Console mutating paths
(``rename``/``optimize``/``repair``) intentionally stay writable.

Each mode test pins the constructor kwargs, so reverting any opener to a
bare ``SessionDB()`` turns its test red (sabotage-checked).
"""

import time

import pytest

import hermes_state
from hermes_state import SessionDB as _RealSessionDB
from hermes_cli import console_engine as ce
from hermes_cli import main as cli_main
from hermes_cli import main_tui_launch as tui_launch
from hermes_cli import terminal_breadcrumbs as tb


class _RecordingDB:
    """Fake SessionDB recording constructor kwargs; stub read/write surface."""

    instances = []
    seed = {}

    def __init__(self, *args, **kwargs):
        type(self).instances.append(kwargs)
        self._sessions = dict(type(self).seed)

    def close(self):
        pass

    # -- read surface used by the observational paths ---------------------
    def get_session(self, session_id):
        return self._sessions.get(session_id)

    def get_session_title(self, session_id):
        session = self._sessions.get(session_id) or {}
        return session.get("title")

    def get_compression_tip(self, session_id):
        return session_id

    def resolve_session_by_title(self, title):
        return None

    def resolve_session_id(self, session_id):
        return session_id if session_id in self._sessions else None

    def search_sessions(self, **kwargs):
        return [{"id": sid} for sid in self._sessions]

    def session_count(self, **kwargs):
        return len(self._sessions)

    def message_count(self, **kwargs):
        return 0

    def list_sessions_rich(self, **kwargs):
        return [{"id": sid} for sid in self._sessions]

    def export_session(self, session_id):
        return {"id": session_id} if session_id in self._sessions else None

    def export_all(self, **kwargs):
        return [{"id": sid} for sid in self._sessions]

    def assert_export_safe(self, session_id, max_messages=None):
        return 0

    # -- write surface (console rename must reach it through a WRITER) ----
    def set_session_title(self, session_id, title):
        return True


@pytest.fixture
def recording_db(monkeypatch):
    _RecordingDB.instances.clear()
    _RecordingDB.seed = {}
    monkeypatch.setattr(hermes_state, "SessionDB", _RecordingDB)
    return _RecordingDB


def _last_kwargs():
    assert _RecordingDB.instances, "expected SessionDB to be constructed"
    return _RecordingDB.instances[-1]


def _ensure_real_store():
    """Production precondition for readers: a real store file exists.

    Called before the class is faked, so the missing-db guard passes and
    the mode assertion observes the real open path selection.
    """
    db = _RealSessionDB()
    try:
        db.create_session("seed-1", source="cli")
    finally:
        db.close()


# ------------------------------------------------------------------ modes

def test_main_session_db_opens_readonly(recording_db, _isolate_hermes_home):
    with cli_main._session_db():
        pass
    assert _last_kwargs().get("read_only") is True


def test_breadcrumb_resolve_opens_readonly(recording_db, _isolate_hermes_home, monkeypatch):
    _RecordingDB.seed = {"sid-1": {"id": "sid-1"}}
    monkeypatch.setattr(tb, "is_enabled", lambda: True)
    monkeypatch.setattr(
        tb, "read_breadcrumb", lambda: {"session_id": "sid-1", "ts": time.time()}
    )
    assert tb.resolve_breadcrumb_session() == "sid-1"
    assert _last_kwargs().get("read_only") is True


def test_tui_exit_summary_opens_readonly(
    recording_db, _isolate_hermes_home, monkeypatch, capsys
):
    # Seed one visible session through the fake before the summary runs.
    _RecordingDB.seed = {"sid-9": {"title": "hello", "message_count": 3}}
    tui_launch._print_tui_exit_summary("sid-9")
    out = capsys.readouterr().out
    assert "sid-9" in out
    assert _last_kwargs().get("read_only") is True


@pytest.mark.parametrize("line", ["sessions list", "sessions stats"])
def test_console_list_stats_open_readonly(
    recording_db, _isolate_hermes_home, line
):
    _ensure_real_store()
    engine = ce.HermesConsoleEngine()
    result = engine.execute(line)
    assert result.status == "ok"
    assert _last_kwargs().get("read_only") is True


def test_console_export_opens_readonly(recording_db, _isolate_hermes_home):
    _ensure_real_store()
    engine = ce.HermesConsoleEngine()
    result = engine.execute("sessions export - --source cli", confirmed=True)
    assert result.status == "ok"
    assert _last_kwargs().get("read_only") is True


def test_console_rename_stays_writable(recording_db, _isolate_hermes_home):
    _RecordingDB.seed = {"sid-1": {"id": "sid-1"}}
    text = ce._sessions_rename(None, ["sid-1", "New", "Title"])
    assert "renamed" in text
    assert "read_only" not in _last_kwargs()


def test_console_list_on_missing_db_fails_closed_without_minting(
    _isolate_hermes_home,
):
    """No database yet: friendly error, and no store file created."""
    from hermes_state import _default_db_path

    engine = ce.HermesConsoleEngine()
    result = engine.execute("sessions list")
    assert result.status == "error"
    assert "No session database" in result.output
    assert not _default_db_path().exists()


# ------------------------------------------------------- live-writer proof

def test_live_writer_survives_observational_readers(
    _isolate_hermes_home, monkeypatch, capsys
):
    """A writable holder stays usable after every converted reader runs."""
    from hermes_state import SessionDB

    writer = SessionDB()
    try:
        writer.create_session("live-1", source="cli")
        writer.append_message("live-1", "user", "hello")

        # 1. breadcrumb resolve (read-only after the fix).
        monkeypatch.setattr(tb, "is_enabled", lambda: True)
        monkeypatch.setattr(
            tb, "read_breadcrumb", lambda: {"session_id": "live-1", "ts": time.time()}
        )
        assert tb.resolve_breadcrumb_session() == "live-1"

        # 2. main MRU/title lookups (read-only after the fix).
        assert cli_main._resolve_last_session(source="cli") == "live-1"

        # 3. console list + stats (read-only after the fix).
        engine = ce.HermesConsoleEngine()
        assert engine.execute("sessions list").status == "ok"
        assert engine.execute("sessions stats").status == "ok"

        # 4. TUI exit summary (read-only after the fix).
        tui_launch._print_tui_exit_summary("live-1")
        assert "live-1" in capsys.readouterr().out

        # The live writer's generation is intact: keep writing and reading.
        writer.append_message("live-1", "assistant", "still here")
        writer.create_session("live-2", source="cli")
        assert writer.get_session("live-2")["id"] == "live-2"
        assert writer.get_session("live-1")["message_count"] == 2
    finally:
        writer.close()
