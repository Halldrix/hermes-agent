"""Regression: ``hermes doctor --fix`` must not checkpoint the live WAL under a running gateway.

Checkpoint-lock premise (#40177): a bare ``sqlite3.connect`` runs WAL recovery and
``PRAGMA wal_checkpoint(PASSIVE)`` joins the live WAL — that second-writer handling
on a gateway-held state.db is the corruption class #103339 tracks. The check must
skip with an actionable finding while a live writer holds the database.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

from hermes_cli.doctor_report import Finding
from hermes_cli.doctor_state import _state_db_wal


# NOTE: no ``requires_wal`` marker here on purpose. That gate exists for tests
# that depend on Hermes *choosing* WAL mode (declined on vulnerable SQLite
# builds). This test forces WAL explicitly through raw SQL and asserts only on
# the holder-guard skip, so the probe mechanics work on any build.
def test_wal_checkpoint_skipped_while_live_writer_holds_db(tmp_path):
    """A held database skips the checkpoint; nothing is checkpointed or fixed."""
    db = tmp_path / "state.db"
    setup = sqlite3.connect(str(db))
    try:
        setup.execute("CREATE TABLE t(x)")
        setup.execute("PRAGMA journal_mode=WAL")
        setup.execute("INSERT INTO t VALUES (1)")
        setup.commit()
    finally:
        setup.close()
    holder = sqlite3.connect(str(db))
    holder.execute("SELECT count(*) FROM t").fetchone()
    try:
        wal = Path(f"{db}-wal")
        assert wal.exists()
        # Push past the 50 MB fix threshold without 50 MB of real frames: the
        # guard runs before any WAL byte is parsed, so padding is never read.
        with open(wal, "ab") as handle:
            handle.truncate(51 * 1024 * 1024)
        finding = Finding()
        _state_db_wal(finding, True, db)
    finally:
        try:
            holder.close()
        except Exception:
            pass

    assert finding.fixed == 0
    assert any("gateway" in issue for issue in finding.issues)


def _make_padded_wal_db(tmp_path):
    """A WAL db past the 50 MB fix threshold with zero open connections.

    The writer subprocess exits via ``os._exit`` (no clean close), so its
    committed frames stay in the -wal instead of being auto-checkpointed
    away on last-close — the crash-recovery shape ``doctor --fix`` exists
    for. Pure-sqlite3 child: no Hermes imports, no HERMES_HOME involved.
    """
    db = tmp_path / "state.db"
    writer_code = (
        "import os, sqlite3, sys;"
        "c = sqlite3.connect(sys.argv[1]);"
        "c.execute('CREATE TABLE t(x)');"
        "c.execute('PRAGMA journal_mode=WAL');"
        "c.execute('INSERT INTO t VALUES (1)');"
        "c.commit();"
        "os._exit(0)"
    )
    proc = subprocess.run([sys.executable, "-c", writer_code, str(db)], timeout=60)
    assert proc.returncode == 0
    wal = Path(f"{db}-wal")
    assert wal.exists()
    # Push past the 50 MB fix threshold without 50 MB of real frames: the
    # guard runs before any WAL byte is parsed, so padding is never read.
    with open(wal, "ab") as handle:
        handle.truncate(51 * 1024 * 1024)
    return db


def test_wal_checkpoint_refuses_unprovable_operational_error(tmp_path, monkeypatch):
    """A non-lock OperationalError is unprovable, not quiet: skip, fix nothing."""
    import hermes_state_repair

    db = _make_padded_wal_db(tmp_path)

    def _raise_io_error(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    # Patched at the source: _state_db_wal late-imports this name at call time.
    monkeypatch.setattr(hermes_state_repair, "_connect_repair_durable", _raise_io_error)

    finding = Finding()
    _state_db_wal(finding, True, db)

    assert finding.fixed == 0
    assert any("gateway" in issue for issue in finding.issues)


def test_wal_checkpoint_refuses_writer_admitted_after_quiet_probe(tmp_path, monkeypatch):
    """Quiet preflight + writer entering before the checkpoint still refuses.

    Deterministic interleaving for the check-then-act race: the preflight
    proves quiet, then guard acquisition fails (a writer took exclusion
    first). The checkpoint must not run and nothing is reported fixed.
    """
    import contextlib

    import hermes_state_holders
    import hermes_state_repair

    db = _make_padded_wal_db(tmp_path)
    # Patched at the source modules: _state_db_wal late-imports both names
    # at call time, so module-attribute patches take effect.
    monkeypatch.setattr(hermes_state_holders, "live_writer_holds_db", lambda *_a, **_k: False)

    @contextlib.contextmanager
    def _contended_guard(_path):
        yield None, sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(hermes_state_repair, "_exclusive_repair_db_guard", _contended_guard)

    finding = Finding()
    _state_db_wal(finding, True, db)

    assert finding.fixed == 0
    assert any("gateway" in issue for issue in finding.issues)


def test_wal_checkpoint_runs_once_quiet_and_guarded(tmp_path):
    """A quiet database checkpoints through the held guard and reports fixed."""
    db = _make_padded_wal_db(tmp_path)

    finding = Finding()
    _state_db_wal(finding, True, db)

    assert finding.fixed == 1
