"""Cross-process writer coordination for state.db (see #103339).

SQLite owns row concurrency; the gate owns file-structure concurrency:
ordinary row writes from any number of processes proceed, while structural
work (repair surgery, second-connection checkpoints) refuses under a live
writer instead of corrupting the WAL.
"""
import multiprocessing as mp
import os
import time
import uuid
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_errors import StateDbWriterHeldError
from hermes_state_repair import repair_state_db_schema
from hermes_state_writergate import (
    OwnerToken,
    acquire_writer_gate,
    release_writer_gate,
    writer_gate_holder,
)


def _hold_gate_child(db_path_str, ready, release):
    """Second process: take the gate and hold it until released."""
    acquire_writer_gate(Path(db_path_str))
    ready.set()
    release.wait(60)


def _write_child(db_path_str, queue):
    """Second process: try one real SessionDB write, report the outcome."""
    try:
        db = SessionDB(db_path=Path(db_path_str))
        sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
        db.append_message(sid, role="user", content="second writer hello")
        db.close()
        queue.put("wrote")
    except Exception as exc:  # noqa: BLE001 — the refusal IS the assertion
        queue.put(f"{type(exc).__name__}: {exc}")


def _build_db_child(db_path_str, queue):
    """Second process: build a healthy db, then exit (gate dies with it)."""
    try:
        db = SessionDB(db_path=Path(db_path_str))
        sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
        db.append_message(sid, role="user", content="hello")
        db.close()
        queue.put("built")
    except Exception as exc:  # noqa: BLE001
        queue.put(f"{type(exc).__name__}: {exc}")


def _corrupt_duplicate_fts(db_path: Path) -> None:
    """Duplicate messages_fts row in sqlite_master (raw sqlite: no gate)."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute(
        "INSERT INTO sqlite_master (type, name, tbl_name, rootpage, sql) "
        "SELECT type, name, tbl_name, rootpage, sql FROM sqlite_master "
        "WHERE name='messages_fts'"
    )
    conn.commit()
    conn.close()


def _hold_surgery_child(db_path_str, ready, release):
    """Second process: hold the GLOBAL structural lock as repair (surgery)."""
    from hermes_state_writergate import OwnerToken, acquire_writer_gate

    acquire_writer_gate(
        Path(db_path_str), role="repair", owner=OwnerToken("surgery"), exclusive=True)
    ready.set()
    release.wait(60)


def _presence_file(db_path: Path, pid: int) -> Path:
    return db_path.with_name(f"{db_path.name}.writer.{pid}.lock")


def _spawn(fn, *args):
    # "spawn", not fork: a genuinely separate process shares neither memory
    # nor fds — exactly the gateway-vs-CLI shape. Fork presence safety is
    # pinned by test_forked_child_cannot_remove_parent_presence below.
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=fn, args=args)
    proc.start()
    return proc


def _snapshot(db_path: Path) -> bytes:
    return db_path.read_bytes()


def test_concurrent_row_writes_proceed_under_sqlite(tmp_path):
    """Row writes are SQLite's job: a second process announces presence and
    writes; both land (the pinned conformance cell proves exactly-once)."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.append_message(sid, role="user", content="holder hello")

    queue = mp.get_context("spawn").Queue()
    proc = _spawn(_write_child, str(db_path), queue)
    try:
        outcome = queue.get(timeout=60)
    finally:
        proc.join(timeout=60)
    assert outcome == "wrote", outcome
    db.append_message(sid, role="assistant", content="holder again")
    row = db._read_one("SELECT COUNT(*) FROM sessions")
    assert row is not None and row[0] == 2
    row = db._read_one("SELECT COUNT(*) FROM messages")
    assert row is not None and row[0] == 3
    db.close()


def test_same_process_second_handle_writes_freely(tmp_path):
    """Same-process writers (registry, threads, sub-agents) never self-lock."""
    db_path = tmp_path / "state.db"
    db1 = SessionDB(db_path=db_path)
    db2 = SessionDB(db_path=db_path)
    sid = db1.create_session(session_id=str(uuid.uuid4()), source="cli")
    db2.append_message(sid, role="user", content="via second handle")
    db1.append_message(sid, role="assistant", content="via first handle")
    row = db1._read_one("SELECT COUNT(*) FROM messages")
    assert row is not None and row[0] == 2
    db1.close()
    db2.close()


def test_read_only_unaffected_by_foreign_gate(tmp_path):
    """Reads never touch the gate: a foreign holder blocks writes, not reads."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        assert writer_gate_holder(db_path) is not None  # foreign holder visible
        ro = SessionDB(db_path=db_path, read_only=True)
        row = ro._read_one("SELECT COUNT(*) FROM sessions")
        ro.close()
        assert row is not None and row[0] == 1
    finally:
        release.set()
        holder.join(timeout=60)


def test_repair_refuses_live_gated_db_without_touching_file(tmp_path):
    """Repair under a live writer returns REFUSED and modifies nothing."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        before = _snapshot(db_path)
        report = repair_state_db_schema(db_path, backup=False)
        assert report.get("repaired") is False
        assert "REFUSED" in (report.get("error") or ""), report
        assert _snapshot(db_path) == before
        assert report.get("backup_path") is None
        assert list(tmp_path.glob("*.backup*")) == []
    finally:
        release.set()
        holder.join(timeout=60)
    # Gate free again: no longer refused (healthy db needs no repair either way).
    report = repair_state_db_schema(db_path, backup=False)
    assert "REFUSED" not in (report.get("error") or ""), report


def test_doctor_checkpoint_skipped_under_live_writer(tmp_path):
    """`doctor --fix` never checkpoints a WAL a live writer owns."""
    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _state_db_wal

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    wal_path = tmp_path / "state.db-wal"
    fd = os.open(str(wal_path), os.O_WRONLY | os.O_CREAT)
    try:
        os.ftruncate(fd, 60 * 1024 * 1024)  # sparse: big size, no disk cost
    finally:
        os.close(fd)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        f = Finding()
        _state_db_wal(f, True, db_path)
        assert f.fixed == 0
        assert any("gateway" in issue for issue in f.issues), f.issues
        assert wal_path.stat().st_size == 60 * 1024 * 1024
    finally:
        release.set()
        holder.join(timeout=60)


def test_close_releases_gate_for_other_processes(tmp_path):
    """A writer that closed does not pin the gate: later processes proceed."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.append_message(sid, role="user", content="hello")
    db.close()
    assert writer_gate_holder(db_path) is None
    queue = mp.get_context("spawn").Queue()
    proc = _spawn(_write_child, str(db_path), queue)
    try:
        outcome = queue.get(timeout=60)
    finally:
        proc.join(timeout=60)
    assert outcome == "wrote", outcome


def test_gate_free_when_nobody_holds(tmp_path):
    """Probe is silent on a free gate; same-process acquire is idempotent."""
    db_path = tmp_path / "state.db"
    assert writer_gate_holder(db_path) is None
    acquire_writer_gate(db_path)
    acquire_writer_gate(db_path)  # idempotent, no self-lock
    assert writer_gate_holder(db_path) is None  # ours reads as free


@pytest.mark.linux_only
def test_forked_child_cannot_remove_parent_presence(tmp_path):
    """After fork(), the child announces under its own pid and its close()
    must not disturb the parent's presence (unlink is own-pid-only)."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.append_message(sid, role="user", content="holder hello")
    parent_file = _presence_file(db_path, os.getpid())
    assert parent_file.exists()

    pid = os.fork()  # windows-footgun: ok — linux_only test, fork guarded by marker
    if pid == 0:  # child: own presence ok; close must spare the parent file
        try:
            from hermes_state_writergate import OwnerToken, acquire_writer_gate

            acquire_writer_gate(db_path, owner=OwnerToken("fork"))
            child_file = _presence_file(db_path, os.getpid())
            announced = child_file.exists()
            db.close()
            ok = announced and parent_file.exists() and not child_file.exists()
            os._exit(20 if ok else 10)
        except Exception:  # noqa: BLE001
            os._exit(30)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 20
    assert parent_file.exists()
    db.append_message(sid, role="assistant", content="parent still writes")
    db.close()


def test_stale_presence_litter_ignored_and_reaped(tmp_path):
    """A crashed writer's presence file (dead pid, old, flock-free) neither
    blocks repair nor survives it."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    stale = _presence_file(db_path, 1 << 30)
    stale.write_bytes(b'{"pid": 1073741824, "role": "writer"}')
    stamp = time.time() - 120.0
    os.utime(stale, (stamp, stamp))
    report = repair_state_db_schema(db_path, backup=False)
    assert "REFUSED" not in (report.get("error") or ""), report
    assert not stale.exists()


def test_row_write_refused_while_surgery_holds_global(tmp_path):
    """Surgery first, then writer: the writer's open (which now announces
    presence fail-closed and holds it across DDL) must refuse while the global
    is held (P1 — proof lifetime covers mutation). After surgery releases,
    the writer opens and writes cleanly."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    surgery = _spawn(_hold_surgery_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        with pytest.raises(StateDbWriterHeldError, match="structural"):
            SessionDB(db_path=db_path)
    finally:
        release.set()
        surgery.join(timeout=60)
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.close()


def test_open_refuses_init_repair_under_live_gate(tmp_path):
    """SessionDB() on a corrupt db held by a live writer raises instead of
    repairing: the INIT self-heal delegates to the gated repair (proven
    REFUSED by test_repair_refuses_live_gated_db_without_touching_file), so
    the open fails with the original error and the file is untouched."""
    import sqlite3

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)
    _corrupt_duplicate_fts(db_path)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        before = _snapshot(db_path)
        with pytest.raises(sqlite3.DatabaseError):
            SessionDB(db_path=db_path)
        assert _snapshot(db_path) == before
    finally:
        release.set()
        holder.join(timeout=60)


def test_doctor_repair_path_refuses_live_gate(tmp_path):
    """`doctor --fix` schema repair delegates to the gated repair: REFUSED
    under a live writer, recorded as failed-issue, file untouched."""
    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _repair_state_db

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)
    _corrupt_duplicate_fts(db_path)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        before = _snapshot(db_path)
        f = Finding()
        _repair_state_db(f, True, db_path, "schema")
        assert f.fixed == 0
        assert any("malformed" in issue for issue in f.issues), f.issues
        assert _snapshot(db_path) == before
    finally:
        release.set()
        holder.join(timeout=60)


def test_doctor_checkpoint_runs_when_gate_free(tmp_path):
    """Free gate: `doctor --fix` checkpoints the WAL and reports fixed."""
    import os

    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _state_db_wal

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    wal_path = tmp_path / "state.db-wal"
    fd = os.open(str(wal_path), os.O_WRONLY | os.O_CREAT)
    try:
        os.ftruncate(fd, 60 * 1024 * 1024)  # sparse
    finally:
        os.close(fd)

    f = Finding()
    _state_db_wal(f, True, db_path)
    assert f.fixed == 1, f.issues
    assert not any("gateway" in issue for issue in f.issues), f.issues


def _probe_child(db_path_str, queue):
    """Second process: report the foreign structural probe (None or holder)."""
    from hermes_state_writergate import writer_gate_holder

    try:
        queue.put(("holder", writer_gate_holder(Path(db_path_str))))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _foreign_holder(db_path):
    """writer_gate_holder() as seen by a genuinely separate process."""
    queue = mp.get_context("spawn").Queue()
    proc = _spawn(_probe_child, str(db_path), queue)
    try:
        kind, value = queue.get(timeout=60)
    finally:
        proc.join(timeout=60)
    assert kind == "holder", value
    return value


def test_second_handle_pins_presence_until_it_closes(tmp_path):
    """Lifetime invariant: with two writable handles, closing the first must
    not clear presence — only the second close may (blocker 2)."""
    db_path = tmp_path / "state.db"
    db1 = SessionDB(db_path=db_path)
    db1.create_session(session_id=str(uuid.uuid4()), source="cli")
    db2 = SessionDB(db_path=db_path)
    db2.create_session(session_id=str(uuid.uuid4()), source="cli")
    db1.close()
    assert _foreign_holder(db_path) is not None
    db2.close()
    assert _foreign_holder(db_path) is None


def test_structural_take_refuses_writer_announced_after_probe(tmp_path):
    """Deterministic probe->take interleaving: a writer announcing after our
    free probe still refuses the structural take (blocker 1)."""
    from hermes_state_writergate import OwnerToken, acquire_writer_gate

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    assert _foreign_holder(db_path) is None  # probe: free
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    holder = _spawn(_hold_gate_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)  # writer announces after the probe
        token = OwnerToken("late-take")
        with pytest.raises(StateDbWriterHeldError):
            acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    finally:
        release.set()
        holder.join(timeout=60)
    token = OwnerToken("late-take")
    acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    release_writer_gate(db_path, token)


def test_open_time_mutations_refuse_during_surgery(tmp_path):
    """Constructor DDL/generation writes refuse while a surgery holds the
    global lock, and open cleanly once it releases (blocker 3)."""
    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    surgery = _spawn(_hold_surgery_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        with pytest.raises(StateDbWriterHeldError, match="structural"):
            SessionDB(db_path=db_path)
    finally:
        release.set()
        surgery.join(timeout=60)
    db = SessionDB(db_path=db_path)
    db.close()


def test_row_write_refuses_when_presence_cannot_be_established(tmp_path, monkeypatch):
    """Fail-closed: inability to hold the presence authority record must
    refuse the row mutation and the callback is never reached (P1)."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=str(uuid.uuid4()), source="cli")
    # Fresh fault path so the next announce actually tries to open/lock.
    fault_path = tmp_path / "fault.db"
    fault_path.write_bytes(b"")
    reached = []

    def fake_open(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("hermes_state_writergate._open_gate_file", fake_open)

    def cb(conn):
        reached.append(1)
        conn.execute("INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                     (str(uuid.uuid4()), "x", 0))

    # Even SessionDB open itself now announces presence fail-closed.
    with pytest.raises(StateDbWriterHeldError, match="cannot establish presence|re-establish presence|cannot lock presence|cannot open"):
        SessionDB(db_path=fault_path)
    # And a direct _execute_write on the already-open handle also refuses
    # before the callback (clear this db's presence so the next announce
    # really hits the fault; otherwise the fast path skips it).
    from hermes_state_writergate import _held, _held_lock, _presence_path

    with _held_lock:
        _held.pop(str(_presence_path(db_path, os.getpid())), None)
    import pathlib as _pl

    _pl.Path(_presence_path(db_path, os.getpid())).unlink(missing_ok=True)
    with pytest.raises(StateDbWriterHeldError, match="cannot establish presence|re-establish presence|cannot lock presence|cannot open"):
        db._execute_write(cb)
    assert reached == []
    db.close()


def test_open_probes_free_then_structural_take_then_mutation_refuses(tmp_path):
    """Deterministic open-vs-surgery: open probes free, surgery takes global,
    then open-time mutation must still refuse before mutating (P1)."""
    from hermes_state_writergate import writer_gate_holder

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    # Phase 1: open probes free.
    assert writer_gate_holder(db_path) is None
    # Phase 2: surgery takes the global lock before open's DDL runs.
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    surgery = _spawn(_hold_surgery_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        # Phase 3: open-time mutation must refuse (presence announced before
        # connect, global held, so fail-closed; file untouched).
        before = _snapshot(db_path)
        with pytest.raises(StateDbWriterHeldError, match="structural"):
            SessionDB(db_path=db_path)
        assert _snapshot(db_path) == before
    finally:
        release.set()
        surgery.join(timeout=60)
    db = SessionDB(db_path=db_path)
    db.close()


@pytest.mark.linux_only
def test_forked_child_refused_while_parent_holds_structural(tmp_path):
    """Fork safety (P1): a parent holding exclusive=True must refuse a forked
    child's SessionDB write — the inherited global hold is foreign, not local."""
    from hermes_state_writergate import OwnerToken, acquire_writer_gate, release_writer_gate

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.close()
    # Parent takes the global structural lock (surgery).
    token = OwnerToken("parent-surgery")
    acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    try:
        pid = os.fork()  # windows-footgun: ok — linux_only test, fork guarded by marker
        if pid == 0:  # child: must see the inherited hold as foreign
            try:
                child_db = SessionDB(db_path=db_path)
                try:
                    child_db.create_session(session_id=str(uuid.uuid4()), source="cli")
                finally:
                    with __import__("contextlib").suppress(Exception):
                        child_db.close()
                os._exit(10)  # BUG: wrote under parent surgery
            except StateDbWriterHeldError:
                os._exit(20)  # OK: refused
            except Exception:  # noqa: BLE001
                os._exit(30)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 20
    finally:
        release_writer_gate(db_path, token)


def test_exclusive_refuses_with_live_same_process_writer(tmp_path):
    """Quietness includes self (P1): a live local SessionDB must refuse an
    unrelated structural take in the same process."""
    from hermes_state_writergate import OwnerToken, acquire_writer_gate, release_writer_gate

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=str(uuid.uuid4()), source="cli")
    token = OwnerToken("unrelated-surgery")
    with pytest.raises(StateDbWriterHeldError):
        acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    # And the reverse: after a (foreign-process) surgery holds, a local row
    # write refuses too. Covered by test_row_write_refused_while_surgery_holds_global
    # for the cross-process case; here assert the local writer still works
    # once no surgery holds.
    db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.close()


def test_reopen_refuses_without_opening_when_structural_holds(tmp_path):
    """Reopen admission precedes SQLite open (P1): close, structural take,
    then in-flight write must refuse with conn left None (no second WAL join)."""
    from hermes_state_writergate import OwnerToken, acquire_writer_gate, release_writer_gate

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.append_message(sid, role="user", content="before close")
    # Simulate the teardown race: close drops conn + presence...
    db.close()
    # ...then a structural op takes the global while we are conn-less...
    token = OwnerToken("surgery")
    acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    try:
        # ...then the in-flight writer resumes via the reopen path and must
        # refuse BEFORE opening SQLite (no new connection, no WAL handling).
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db._execute_write(lambda conn: conn.execute("SELECT 1").fetchone())
        assert db._conn is None
    finally:
        release_writer_gate(db_path, token)


def test_self_heal_repin_failure_does_not_reopen(tmp_path, monkeypatch):
    """Post-repair re-admission failure must propagate without reopening
    SQLite (P1 hermes_state.py:558: swallowed re-pin + reopen under surgery)."""
    import hermes_state as _hs_mod
    from hermes_state_errors import StateDbWriterHeldError as _Held

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)
    _corrupt_duplicate_fts(db_path)

    real_acquire = _hs_mod.acquire_writer_gate
    calls = {"n": 0}

    def fake_acquire(path, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_acquire(path, *a, **k)  # open admission ok
        raise _Held("foreign surgery took the global before re-pin")

    monkeypatch.setattr(_hs_mod, "acquire_writer_gate", fake_acquire)
    monkeypatch.setattr(
        "hermes_state_repair.repair_state_db_schema",
        lambda *a, **k: {"repaired": True, "strategy": "mocked", "backup_path": None, "error": None},
    )
    connects = {"n": 0}
    real_connect = _hs_mod.SessionDB._connect_and_init_with_lock_patience

    def counting_connect(self):
        connects["n"] += 1
        return real_connect(self)

    monkeypatch.setattr(
        _hs_mod.SessionDB, "_connect_and_init_with_lock_patience", counting_connect)
    with pytest.raises(_Held, match="foreign surgery"):
        SessionDB(db_path=db_path)
    # Initial connect only; the reopen after failed re-pin must never run.
    assert connects["n"] == 1, connects


def test_stale_reap_keeps_racing_replacement(tmp_path, monkeypatch):
    """P1-A: a replacement winning the pathname between verify and unlink
    must be detected by identity and kept live (deterministic forced order:
    the swap fires inside the reaper's own stat/unlink calls)."""
    import json
    from pathlib import Path

    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    dead_pid = 2**30 - 7
    stale = _wg._presence_path(db_path, dead_pid)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(json.dumps(
        {"pid": dead_pid, "start_ticks": 1, "acquired_at": 0.0, "role": "writer"}
    ).encode())
    swapped = {"n": 0}

    def _swap_in_replacement():
        # A live replacement takes the pathname RIGHT NOW (new inode).
        with __import__("contextlib").suppress(OSError):
            os.unlink(stale)
        stale.write_bytes(json.dumps(
            {"pid": os.getpid(), "start_ticks": _wg._proc_start_ticks(os.getpid()),
             "acquired_at": time.time(), "role": "writer"}
        ).encode())

    real_stat = Path.stat
    real_unlink = Path.unlink

    def swapping_stat(self, *a, **k):
        if str(self) == str(stale) and swapped["n"] == 0:
            swapped["n"] += 1
            _swap_in_replacement()
        return real_stat(self, *a, **k)

    def swapping_unlink(self, *a, **k):
        if str(self) == str(stale) and swapped["n"] == 0:
            swapped["n"] += 1
            _swap_in_replacement()
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "stat", swapping_stat)
    monkeypatch.setattr(Path, "unlink", swapping_unlink)
    live = _wg._live_writer_presences(db_path)
    assert swapped["n"] == 1  # the race actually fired, not vacuous
    assert any(str(stale) in d for d in live)  # replacement kept live
    assert stale.exists()  # and NOT deleted
    assert b"acquired_at" in stale.read_bytes()  # replacement content intact


@pytest.mark.linux_only
def test_recycled_pid_stale_record_reaped(tmp_path):
    """P1-B: a record naming a live pid with mismatched start_ticks (PID
    recycled onto an unrelated process) must reap, not block surgery."""
    import json

    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    me = os.getpid()
    ticks = _wg._proc_start_ticks(me)
    assert ticks is not None
    rec_path = _wg._presence_path(db_path, me)
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_bytes(json.dumps(
        {"pid": me, "start_ticks": int(ticks) + 1000000,
         "acquired_at": time.time(), "role": "writer"}
    ).encode())
    live = _wg._live_writer_presences(db_path, include_self=True)
    assert live == []  # reaped, not live
    assert not rec_path.exists()


@pytest.mark.linux_only
def test_orphaned_fork_descriptor_does_not_block_surgery(tmp_path):
    """P1-B: dead parent + live child inheriting the presence flock (#100108
    shape) must not block structural work; the orphan file is reaped."""
    import signal
    import time as _time

    import hermes_state_writergate as _wg
    from hermes_state_writergate import OwnerToken, acquire_writer_gate, release_writer_gate

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    c1 = os.fork()  # windows-footgun: ok — linux_only test, fork guarded by marker
    if c1 == 0:
        try:
            from hermes_state_writergate import _open_gate_file, _try_flock_nb, _write_gate_record
            mine = _wg._presence_path(db_path, os.getpid())
            mine.parent.mkdir(parents=True, exist_ok=True)
            fh = _open_gate_file(mine)
            assert _try_flock_nb(fh) is True
            _write_gate_record(fh, "writer")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
            c2 = os.fork()  # windows-footgun: ok — linux_only test, fork guarded by marker
            if c2 == 0:
                _time.sleep(30)
                os._exit(0)
            # C1 dies holding nothing released: kernel keeps the flock alive
            # via C2's inherited description.
            os._exit(0)
        except Exception:  # noqa: BLE001
            os._exit(99)
    _, status = os.waitpid(c1, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    # Find C2 = the live child still holding C1's flock.
    orphan = _wg._presence_path(db_path, c1)
    assert orphan.exists()
    token = OwnerToken("surgery-after-orphan")
    try:
        acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    finally:
        with __import__("contextlib").suppress(Exception):
            release_writer_gate(db_path, token)
    assert not orphan.exists()  # orphan file reaped
    # Reap C2 (the sleeper, C1's child now orphaned) via /proc ppid scan.
    c2pid = None
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            with open(f"/proc/{p}/stat", "rb") as fh:
                parts = fh.read().rsplit(b")", 1)[1].split()
            if int(parts[1]) == c1:  # ppid == dead C1 → likely C2
                c2pid = int(p)
                break
        except (OSError, ValueError, IndexError):
            continue
    if c2pid is not None:
        with __import__("contextlib").suppress(OSError):
            os.kill(c2pid, signal.SIGTERM)


@pytest.mark.linux_only
def test_overlapping_probes_coexist(tmp_path):
    """P1 (probe owns flock): two overlapping observation probes must not
    refuse each other — only a real exclusive holder is structural."""
    import threading

    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    lock_path = _wg._writer_lock_path(db_path)
    holding, release = threading.Event(), threading.Event()

    def _hold_observe():
        fh = _wg._open_gate_file(lock_path)
        assert _wg._try_flock_observe_nb(fh) is True
        holding.set()
        assert release.wait(timeout=60)
        _wg._unlock_handle(fh)
        fh.close()

    t = threading.Thread(target=_hold_observe)
    t.start()
    try:
        assert holding.wait(timeout=60)
        foreign, _role, _desc = _wg._probe_global(db_path)
        assert foreign is False  # fellow probe is not a surgery
    finally:
        release.set()
        t.join(timeout=60)


def test_probe_sees_real_structural_hold(tmp_path):
    """Observation still refuses under a real exclusive holder."""
    import hermes_state_writergate as _wg
    from hermes_state_writergate import OwnerToken, acquire_writer_gate, release_writer_gate

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    token = OwnerToken("surgery")
    acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    try:
        foreign, _role, _desc = _wg._probe_global(db_path)
        assert foreign is True
    finally:
        release_writer_gate(db_path, token)


def test_exclusive_refuses_fellow_exclusive_owner_same_process(tmp_path):
    """P1 (shared hold): a distinct same-process structural owner must not
    piggyback — only the identical token re-enters; anonymous takes raise."""
    from hermes_state_writergate import OwnerToken, acquire_writer_gate, release_writer_gate

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    with pytest.raises(ValueError, match="owner token"):
        acquire_writer_gate(db_path, role="repair", owner=None, exclusive=True)
    a = OwnerToken("repair-a")
    b = OwnerToken("checkpoint-b")
    acquire_writer_gate(db_path, role="repair", owner=a, exclusive=True)
    try:
        with pytest.raises(StateDbWriterHeldError, match="another structural operation"):
            acquire_writer_gate(db_path, role="repair", owner=b, exclusive=True)
        acquire_writer_gate(db_path, role="repair", owner=a, exclusive=True)  # same token: ok
    finally:
        release_writer_gate(db_path, a)
    acquire_writer_gate(db_path, role="repair", owner=b, exclusive=True)  # free after release
    release_writer_gate(db_path, b)


@pytest.mark.linux_only
def test_mutation_mutex_held_across_reap_unlink(tmp_path, monkeypatch):
    """P1 (final interval): the reaper must hold the presence-path mutation
    mutex while unlinking — proven by failing a fresh mutex take inside the
    unlink call itself."""
    import fcntl
    import json
    from pathlib import Path

    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    dead_pid = 2**30 - 7
    stale = _wg._presence_path(db_path, dead_pid)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(json.dumps(
        {"pid": dead_pid, "start_ticks": 1, "acquired_at": 0.0, "role": "writer"}
    ).encode())
    mtx = _wg._mutation_mutex_path(db_path)
    observed = {"held": None}
    real_unlink = Path.unlink

    def _checked_unlink(self, *a, **k):
        if str(self) == str(stale):
            probe = mtx.open("a+b")
            try:
                try:
                    fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    observed["held"] = True  # mutex held by the reaper: serialized
                else:
                    observed["held"] = False
                    fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
            finally:
                probe.close()
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", _checked_unlink)
    assert _wg._live_writer_presences(db_path) == []
    assert observed["held"] is True
    assert not stale.exists()


def test_mutation_mutex_contention_reports_live(tmp_path):
    """A reaper blocked on the mutation mutex fails closed (live, no unlink);
    the litter reaps once the mutex is free. Deterministic via join."""
    import json
    import threading

    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    dead_pid = 2**30 - 7
    stale = _wg._presence_path(db_path, dead_pid)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(json.dumps(
        {"pid": dead_pid, "start_ticks": 1, "acquired_at": 0.0, "role": "writer"}
    ).encode())
    out = {}
    with _wg._presence_mutation_serialized(db_path) as serialized:
        assert serialized
        t = threading.Thread(target=lambda: out.update(live=_wg._live_writer_presences(db_path)))
        t.start()
        t.join(timeout=60)
        assert not t.is_alive()
    assert any(str(stale) in d for d in out["live"])
    assert stale.exists()
    assert _wg._live_writer_presences(db_path) == []
    assert not stale.exists()


def test_read_reopen_refuses_without_opening_when_structural_holds(tmp_path):
    """P1 (read fallback): close, structural take, then an in-flight read
    must refuse BEFORE opening SQLite — same invariant as the write reopen."""
    from hermes_state_writergate import OwnerToken, acquire_writer_gate, release_writer_gate

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    sid = db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.append_message(sid, role="user", content="before close")
    db.close()
    token = OwnerToken("surgery")
    acquire_writer_gate(db_path, role="repair", owner=token, exclusive=True)
    try:
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db._read_one("SELECT COUNT(*) FROM sessions")
        assert db._conn is None
    finally:
        release_writer_gate(db_path, token)


def test_doctor_checkpoint_closes_conn_before_releasing_gate_on_error(tmp_path, monkeypatch):
    """P1 (doctor lifecycle): a failed checkpoint closes its connection
    BEFORE releasing the structural gate — close, then release, in order."""
    import sqlite3

    from hermes_cli.doctor_report import Finding
    from hermes_cli.doctor_state import _state_db_wal
    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)

    wal_path = tmp_path / "state.db-wal"
    fd = os.open(str(wal_path), os.O_WRONLY | os.O_CREAT)
    try:
        os.ftruncate(fd, 60 * 1024 * 1024)  # sparse
    finally:
        os.close(fd)

    events = []
    real_release = _wg.release_writer_gate

    def _spy_release(path, owner):
        events.append("release")
        return real_release(path, owner)

    monkeypatch.setattr(_wg, "release_writer_gate", _spy_release)

    class _FailConn:
        def execute(self, *a, **k):
            raise sqlite3.OperationalError("injected checkpoint failure")

        def close(self):
            events.append("close")

    def _fail_connect(*a, **k):
        events.append("connect")
        return _FailConn()

    monkeypatch.setattr(sqlite3, "connect", _fail_connect)
    f = Finding()
    _state_db_wal(f, True, db_path)  # error swallowed by warn_on_error; order is the assert
    assert events == ["connect", "close", "release"], events
    assert f.fixed == 0
    assert _wg.writer_gate_holder(db_path) is None


def _surgery_backdrop(tmp_path):
    """Build a db, open+close it in this process, then hold surgery from a
    spawned process. Yields the closed db; caller drives reads/writes that
    must refuse. Setup helper, not a test."""
    import contextlib

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)
    db = SessionDB(db_path=db_path)
    db.close()  # teardown race shape: conn-less handle, presence released
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    surgery = _spawn(_hold_surgery_child, str(db_path), ready, release)
    assert ready.wait(timeout=60)

    class _Backdrop:
        def __init__(self):
            self.db = db
            self._surgery = surgery
            self._release = release

        def __enter__(self):
            return db

        def __exit__(self, *exc):
            self._release.set()
            self._surgery.join(timeout=60)
            with contextlib.suppress(Exception):
                db.close()
            return False

    return _Backdrop()


def test_gate_refusal_propagates_from_fts_match_rows(tmp_path, monkeypatch):
    """Swallow-site guard: a gate refusal from _match_rows propagates instead
    of degrading to an FTS-syntax fallback (None). Fault at the seam: the
    guard, not SQL building, is under test."""
    from hermes_state_errors import StateDbWriterHeldError as _Held

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        monkeypatch.setattr(db, "_fts_match_sql", lambda *a, **k: ("SELECT 1", ()))

        def _raise(sql, params):
            raise _Held("injected gate refusal")

        monkeypatch.setattr(db, "_read_all", _raise)
        with pytest.raises(_Held, match="injected gate refusal"):
            db._match_rows("messages_fts", "hello", "rank", limit=10, offset=0)
    finally:
        db.close()


def test_gate_refusal_propagates_from_clear_session_activity_labels(tmp_path, monkeypatch):
    """Swallow-site guard: a gate refusal from clear_session_activity_labels
    propagates before any write is attempted (not swallowed as missing row).
    Fault at the seam: the read guard, not the gate, is under test."""
    from hermes_state_errors import StateDbWriterHeldError as _Held

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        def _raise(sql, params):
            raise _Held("injected gate refusal")

        monkeypatch.setattr(db, "_read_one", _raise)
        writes = []
        monkeypatch.setattr(db, "_write_sql", lambda *a, **k: writes.append(1))
        with pytest.raises(_Held, match="injected gate refusal"):
            db.clear_session_activity_labels("sid")
        assert writes == []  # refused before any write attempt
    finally:
        db.close()


def test_gate_refusal_propagates_from_topic_read_one(tmp_path):
    """Swallow-site guard: a gate refusal from _topic_read_one propagates
    instead of degrading to unmigrated-table None."""
    with _surgery_backdrop(tmp_path) as db:
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db._topic_read_one("SELECT 1", ())


def test_gate_refusal_propagates_from_write_sql_logged(tmp_path):
    """Swallow-site guard: a refused write via _write_sql_logged raises
    instead of being logged-and-done (a lost write must never be silent)."""
    with _surgery_backdrop(tmp_path) as db:
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db._write_sql_logged("test-op", "sid", "UPDATE sessions SET source=? WHERE id=?", ("x", "y"))


def test_mutation_mutex_file_is_a_persistent_sentinel(tmp_path):
    """The .writer.mtx.lock file is never unlinked (like the global lock):
    pins the intentional-persistence contract against future cleanups."""
    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=str(uuid.uuid4()), source="cli")
    db.close()
    mtx = _wg._mutation_mutex_path(db_path)
    # Run a reap cycle (dead-pid litter); the sentinel is created and stays.
    import json

    dead_pid = 2**30 - 7
    stale = _wg._presence_path(db_path, dead_pid)
    stale.write_bytes(json.dumps(
        {"pid": dead_pid, "start_ticks": 1, "acquired_at": 0.0, "role": "writer"}
    ).encode())
    assert _wg._live_writer_presences(db_path) == []
    assert mtx.exists()


def test_refused_announce_leaves_no_presence(tmp_path):
    """P1 (announce rollback): a refused row announce must roll back its own
    presence — post-refusal state, not just the raise. After surgery exits,
    the gate is quiet and a fresh writer proceeds."""
    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    queue = mp.get_context("spawn").Queue()
    builder = _spawn(_build_db_child, str(db_path), queue)
    try:
        assert queue.get(timeout=60) == "built"
    finally:
        builder.join(timeout=60)
    db = SessionDB(db_path=db_path)
    db.close()  # conn-less handle, presence released
    presence = _wg._presence_path(db_path, os.getpid())
    assert not presence.exists()
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    surgery = _spawn(_hold_surgery_child, str(db_path), ready, release)
    try:
        assert ready.wait(timeout=60)
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db._execute_write(lambda conn: conn.execute("SELECT 1").fetchone())
        assert db._conn is None
        assert not presence.exists(), "refused announce littered presence"
    finally:
        release.set()
        surgery.join(timeout=60)
    assert _wg.writer_gate_holder(db_path) is None
    db2 = SessionDB(db_path=db_path)
    db2.create_session(session_id=str(uuid.uuid4()), source="cli")
    db2.close()


def test_refused_anonymous_announce_keeps_other_leases(tmp_path):
    """Rollback consumes exactly one anonymous lease: with two live leases,
    one rollback keeps the presence; the second drops it. (A live local
    presence blocks foreign surgery by design, so the refused-second-lease
    shape is exercised at the rollback unit level.)"""
    import hermes_state_writergate as _wg
    from hermes_state_writergate import _held, _held_lock, _rollback_announce

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    _wg.acquire_writer_gate(db_path)  # lease 1
    _wg.acquire_writer_gate(db_path)  # lease 2
    presence = _wg._presence_path(db_path, os.getpid())
    assert presence.exists()
    with _held_lock:
        assert _held[str(presence)].anon_leases == 2
    _rollback_announce(db_path, None)  # refused lease 2 rolls back
    assert presence.exists(), "rollback ate another caller's live lease"
    with _held_lock:
        assert _held[str(presence)].anon_leases == 1
    _rollback_announce(db_path, None)  # last lease drops the hold
    assert not presence.exists()
    assert _wg.writer_gate_holder(db_path) is None


def test_release_close_marks_pending_when_mutex_contended(tmp_path):
    """P1 (release wedge): a closer colliding with the mutation mutex keeps
    its hold (flock held, owner restored) instead of an unlocked ghost, marks
    retry state, and __del__ drains it independently — close() never returns
    a live-pid record with no retry left. Deterministic via join, no timing."""
    import threading

    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=str(uuid.uuid4()), source="cli")
    presence = _wg._presence_path(db_path, os.getpid())
    assert presence.exists()
    entered, free = threading.Event(), threading.Event()

    def _hold_mutex():
        with _wg._presence_mutation_serialized(db_path) as serialized:
            assert serialized
            entered.set()
            assert free.wait(timeout=60)

    t = threading.Thread(target=_hold_mutex)
    t.start()
    try:
        assert entered.wait(timeout=60)
        db.close()  # mutex contended: hold re-registered, retry state marked
        # Retained, never a ghost: file present and an independent retry armed.
        assert presence.exists()
        assert db._gate_release_pending is True
    finally:
        free.set()
        t.join(timeout=60)
    db.__del__()  # independent retry without an explicit close
    assert db._gate_release_pending is False
    assert not presence.exists()
    assert _wg.writer_gate_holder(db_path) is None


@pytest.mark.linux_only
def test_orphan_reap_keeps_replacement_record(tmp_path, monkeypatch):
    """P1 (orphan record race): a replacement that wins the pathname between
    dead-classification and mutex cleanup differs in record content and must
    be kept live — identity alone is insufficient."""
    import fcntl
    import json

    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    dead_pid = 2**30 - 7
    orphan = _wg._presence_path(db_path, dead_pid)
    orphan.parent.mkdir(parents=True, exist_ok=True)
    dead_record = {"pid": dead_pid, "start_ticks": 1, "acquired_at": 0.0, "role": "writer"}
    orphan.write_bytes(json.dumps(dead_record).encode())
    # Orphan descriptor: an unrelated live fd holding the flock.
    keeper = orphan.open("r+b")
    fcntl.flock(keeper.fileno(), fcntl.LOCK_EX)
    real_serialized = _wg._presence_mutation_serialized
    swapped = {"n": 0}

    @__import__("contextlib").contextmanager
    def _swapping_serialized(path):
        with real_serialized(path) as ok:
            if ok and swapped["n"] == 0:
                swapped["n"] += 1
                # Replacement wins NOW: new inode, live record, flock held.
                with __import__("contextlib").suppress(OSError):
                    os.unlink(orphan)
                orphan.write_bytes(json.dumps(
                    {"pid": os.getpid(), "start_ticks": _wg._proc_start_ticks(os.getpid()),
                     "acquired_at": time.time(), "role": "writer"}
                ).encode())
            yield ok

    monkeypatch.setattr(_wg, "_presence_mutation_serialized", _swapping_serialized)
    try:
        live = _wg._live_writer_presences(db_path)
    finally:
        try:
            fcntl.flock(keeper.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        keeper.close()
    assert swapped["n"] == 1
    assert any(str(orphan) in d for d in live), "replacement record was reaped as orphan"
    assert orphan.exists()


def test_gate_refusal_propagates_from_telegram_bindings(tmp_path):
    with _surgery_backdrop(tmp_path) as db:
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db.list_telegram_topic_bindings_for_chat(chat_id="c")


def test_gate_refusal_propagates_from_handoff_reads(tmp_path):
    with _surgery_backdrop(tmp_path) as db:
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db.get_handoff_state("sid")
    with _surgery_backdrop(tmp_path) as db:
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db.list_pending_handoffs()


def test_gate_refusal_propagates_from_fts_stale_refresh(tmp_path):
    with _surgery_backdrop(tmp_path) as db:
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db._refresh_fts_stale_state()


def test_gate_refusal_propagates_from_unindexed_gap(tmp_path, monkeypatch):
    """P1 (deferred-gap swallow): a gate refusal inside the FTS-rebuild gap
    supplement must propagate, not return the indexed subset as complete.
    Precise fault injection: a real interleaving cannot be scheduled
    in-process (the main FTS query would refuse first)."""
    import sqlite3

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db._fts_enabled = True
    db._fts_stale = False
    monkeypatch.setattr(db, "_read_all", lambda sql, params: [{"id": 1}])
    monkeypatch.setattr(
        db, "fts_rebuild_status",
        lambda: {"pending": True, "total": 2, "indexed": 1, "percent": 50},
    )

    def _refuse(*args, **kwargs):
        raise StateDbWriterHeldError("writer gate held: structural surgery in progress")

    monkeypatch.setattr(db, "_search_unindexed_gap", _refuse)
    with pytest.raises(StateDbWriterHeldError):
        db.search_messages("hello", limit=10)

    def _ordinary_failure(*args, **kwargs):
        raise sqlite3.OperationalError("no such table: messages_fts")

    monkeypatch.setattr(db, "_search_unindexed_gap", _ordinary_failure)
    assert [m["id"] for m in db.search_messages("hello", limit=10)] == [1]


def test_gate_refusal_propagates_from_page_pragmas(tmp_path):
    with _surgery_backdrop(tmp_path) as db:
        with pytest.raises(StateDbWriterHeldError, match="structural|writer gate"):
            db._page_pragmas(("page_count",), "probe")


def test_gate_refusal_propagates_from_fts_backfill_retry(tmp_path, monkeypatch):
    """Retry-loop sites must not spin on a refusal: it propagates instead of
    degrading to will-retry True."""
    from hermes_state_errors import StateDbWriterHeldError as _Held

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        monkeypatch.setattr(db, "get_meta", lambda *a, **k: "1000")

        def _raise(_cb):
            raise _Held("injected gate refusal")

        monkeypatch.setattr(db, "_execute_write", _raise)
        with pytest.raises(_Held, match="injected gate refusal"):
            db._rebuild_step("t", [], fail_msg="x", finish=lambda: None)
        monkeypatch.setattr(db, "_read_ctx", lambda: (_ for _ in ()).throw(_Held("injected gate refusal")))
        with pytest.raises(_Held, match="injected gate refusal"):
            db._fts_teardown_trash_step()
    finally:
        db.close()


def test_gate_refusal_propagates_from_write_seams(tmp_path, monkeypatch):
    """Write-fallback sites (reclaim, end_session, compression lock, tip
    resolve, identity probe) propagate via the shared classifier."""
    from hermes_state_errors import StateDbWriterHeldError as _Held
    from hermes_state_errors import is_gate_refusal

    assert is_gate_refusal(_Held("x"))
    assert not is_gate_refusal(ValueError("x"))
    assert not is_gate_refusal(__import__("sqlite3").OperationalError("busy"))

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        def _raise(*a, **k):
            raise _Held("injected gate refusal")

        monkeypatch.setattr(db, "_execute_write", _raise)
        with pytest.raises(_Held, match="injected gate refusal"):
            db.reclaim_stale_running_handoffs("boom")
    finally:
        db.close()
    db2 = SessionDB(db_path=db_path)
    try:
        monkeypatch.setattr(db2, "_execute_write", _raise)
        with pytest.raises(_Held, match="injected gate refusal"):
            db2.promote_to_session_reset("sid")
    finally:
        db2.close()


def test_failed_init_closes_conn_before_releasing_presence(tmp_path, monkeypatch):
    """P3 (init-finally order): a failed construction must settle close-time
    WAL work BEFORE releasing writer presence, so no structural take can
    slip into the window. Deterministic fault injection at the seam."""
    import hermes_state_writergate as _wg

    events = []
    real_open = SessionDB._open_writer
    real_close_conn = SessionDB._close_connection_quietly
    real_release = _wg.release_writer_gate

    def _fail_after_open(self):
        real_open(self)
        raise RuntimeError("boom-after-open")

    def _rec_close(self, conn):
        events.append("close")
        return real_close_conn(conn)

    def _rec_release(db_path, owner):
        events.append("release")
        return real_release(db_path, owner)

    monkeypatch.setattr(SessionDB, "_open_writer", _fail_after_open)
    monkeypatch.setattr(SessionDB, "_close_connection_quietly", _rec_close)
    monkeypatch.setattr(_wg, "release_writer_gate", _rec_release)
    with pytest.raises(RuntimeError, match="boom-after-open"):
        SessionDB(db_path=tmp_path / "state.db")
    assert events == ["close", "release"]


def test_reopen_failure_closes_conn_before_releasing_presence(tmp_path, monkeypatch):
    """P3 (reopen-fail order): same close-before-release contract on the
    teardown/worker-races reopen path. Deterministic fault injection."""
    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.close()
    events = []
    real_close_conn = SessionDB._close_connection_quietly
    real_release = _wg.release_writer_gate

    def _rec_close(self, conn):
        events.append("close")
        return real_close_conn(conn)

    def _rec_release(path, owner):
        events.append("release")
        return real_release(path, owner)

    monkeypatch.setattr(db, "_open_writer_conn", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(SessionDB, "_close_connection_quietly", _rec_close)
    monkeypatch.setattr(_wg, "release_writer_gate", _rec_release)
    with pytest.raises(__import__("sqlite3").OperationalError):
        db._reopen_after_close_locked("test")
    assert events == ["close", "release"]
    db.close()


def test_gate_setup_failure_classifies_distinct_from_locked(tmp_path):
    """P4 (lease poll): permanent gate-setup failures must not classify as
    retryable "locked" — only live-holder contention polls."""
    from hermes_state import classify_persistence_error
    from hermes_state_errors import StateDbGateSetupError

    assert classify_persistence_error(
        StateDbWriterHeldError("writer gate held: structural surgery in progress")) == "locked"
    assert classify_persistence_error(
        StateDbGateSetupError("state.db writer gate: cannot establish presence")) == "gate_setup"


def test_turn_lease_raises_at_once_on_gate_setup_failure(tmp_path, monkeypatch):
    """P4 (lease poll): the turn lease must raise a permanent setup failure
    immediately instead of polling it for wait_seconds."""
    from hermes_state_errors import StateDbGateSetupError

    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        def _setup_failed(*a, **k):
            raise StateDbGateSetupError("state.db writer gate: cannot establish presence")

        monkeypatch.setattr(db, "try_acquire_session_turn_lease", _setup_failed)
        started = __import__("time").monotonic()
        with pytest.raises(StateDbGateSetupError):
            db.acquire_session_turn_lease("sid", "holder", wait_seconds=2.0)
        # Immediate: the raise precedes any poll sleep, so it must land well
        # before the budget a polling regression would exhaust.
        assert __import__("time").monotonic() - started < 2.0
    finally:
        db.close()


def test_gate_refusal_propagates_from_unarchive_lookup(tmp_path):
    """P5 (unarchive swallow): gate refusal during unarchive lookup must
    propagate, not hide the canonical session as absent. Real second-process
    surgery against a closed handle (teardown-race shape)."""
    import contextlib

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        sid_live = str(uuid.uuid4())
        db.create_session(session_id=sid_live, source="cli")
        db.end_session(sid_live, "ws_orphan_reap")
        db.set_session_archived(sid_live, True)
        assert db.unarchive_recoverable_session(sid_live) is True

        sid_test = str(uuid.uuid4())
        db.create_session(session_id=sid_test, source="cli")
        db.end_session(sid_test, "ws_orphan_reap")
        db.set_session_archived(sid_test, True)
    finally:
        db.close()
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    surgery = _spawn(_hold_surgery_child, str(db_path), ready, release)
    assert ready.wait(timeout=60)
    try:
        with pytest.raises(StateDbWriterHeldError):
            db.unarchive_recoverable_session(sid_test)
    finally:
        release.set()
        surgery.join(timeout=60)
        with contextlib.suppress(Exception):
            db.close()


def test_pending_release_drains_without_owner_retry(tmp_path, monkeypatch):
    """P1 (release wedge): a close() that exhausts the mutation-mutex budget
    must not strand the presence when the owner dies with no retry left.
    Real contention via a holder thread, then a REAL gc collection of the
    owner — the module drain (not __del__, not another close) cleans up.
    Event-driven: the drain's poll loop is patched to a barrier the test
    controls, so no timing is assumed."""
    import threading

    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    db.create_session(session_id=str(uuid.uuid4()), source="cli")
    presence = _wg._presence_path(db_path, os.getpid())
    assert presence.exists()
    entered, free, drained = threading.Event(), threading.Event(), threading.Event()

    def _hold_mutex():
        with _wg._presence_mutation_serialized(db_path) as serialized:
            assert serialized is True
            entered.set()
            assert free.wait(timeout=60)

    # Gate the drain's poll on events the test controls (no timing races).
    real_sleep = _wg.time.sleep

    def _sleep_and_signal(seconds):
        real_sleep(seconds)
        if seconds == _wg._DEFERRED_RELEASE_DRAIN_POLL_S:
            drained.set()

    monkeypatch.setattr(_wg.time, "sleep", _sleep_and_signal)

    t = threading.Thread(target=_hold_mutex)
    t.start()
    try:
        assert entered.wait(timeout=60)
        db.close()  # mutex contended: hold flagged deferred, drain armed
        assert db._gate_release_pending is True
        assert presence.exists()
        with _wg._held_lock:
            hold = _wg._held[str(presence)]
            assert hold.deferred_release is True
        # Kill EVERY retry the owner could ever provide: __del__ has already
        # run (close() left _conn=None), so the object's collection below is
        # the "no future close()" world the reviewer reproduced.
        del db
        import gc
        gc.collect()
        with _wg._held_lock:
            assert str(presence) in _wg._held  # flagged hold survives GC
    finally:
        free.set()
        t.join(timeout=60)
    # The module drain — no owner, no close(), no __del__ — finishes it.
    assert drained.wait(timeout=60)
    real_sleep(0.5)
    with _wg._held_lock:
        assert str(presence) not in _wg._held
    assert not presence.exists()
    assert _wg.writer_gate_holder(db_path) is None


def _wait_until(predicate, timeout_s: float = 30.0) -> bool:
    """Event-friendly poll (no timing assumption beyond the wall bound)."""
    import time as _time

    deadline = _time.monotonic() + timeout_s
    while _time.monotonic() < deadline:
        if predicate():
            return True
        _time.sleep(0.05)
    return predicate()


def test_rollback_announce_preserves_live_retry_path(tmp_path):
    """P1 (rollback wedge): a refused announce whose rollback release also
    collides with a contended mutation mutex must leave a flagged hold and
    an armed drain — never a live-pid presence with zero owners and no
    retry. Real mutex contention, deterministic via join/events."""
    import gc
    import threading

    import hermes_state_writergate as _wg
    from hermes_state_writergate import _held, _held_lock

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    # Build the announce to roll back: one named-owner share we can refuse.
    token = _wg.OwnerToken("victim")
    _wg.acquire_writer_gate(db_path, owner=token)
    presence = _wg._presence_path(db_path, os.getpid())
    assert presence.exists()
    entered, free = threading.Event(), threading.Event()

    def _hold_mutex():
        with _wg._presence_mutation_serialized(db_path) as serialized:
            assert serialized is True
            entered.set()
            assert free.wait(timeout=60)

    t = threading.Thread(target=_hold_mutex)
    t.start()
    try:
        assert entered.wait(timeout=60)
        # Rollback under contention: this is the exact _rollback_announce ->
        # release_writer_gate path the reviewer's P1 names.
        _wg._rollback_announce(db_path, token)
        with _held_lock:
            hold = _held.get(str(presence))
            assert hold is not None, "rollback dropped the pinned hold entirely"
            assert hold.deferred_release is True, "False release result was discarded"
            assert list(hold.owners), "hold must stay accurate (owner restored)"
        del token  # the owner dies: only the module drain can finish this
        gc.collect()
        with _held_lock:
            assert not list(hold.owners)  # weak owner died with the object
    finally:
        free.set()
        t.join(timeout=60)
    # The drain — no owner, no close() — completes the owed unlink.
    assert _wait_until(lambda: not presence.exists())
    with _held_lock:
        assert str(presence) not in _held
    assert _wg.writer_gate_holder(db_path) is None


@pytest.mark.linux_only
def test_gate_setup_failure_on_mutex_open_not_classified_locked(tmp_path, monkeypatch):
    """P1 (lease poll): a mutation-mutex SETUP failure (PermissionError on
    opening the MUTEX file, not flock contention, not the presence file)
    during a stale-presence reclaim must raise StateDbGateSetupError —
    classify_persistence_error then reports gate_setup, so the turn lease
    raises at once instead of polling the 1800s default. The probe paths
    keep their never-raises contract."""
    from hermes_state_errors import StateDbGateSetupError, classify_persistence_error

    import hermes_state_writergate as _wg

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    # Stale same-pid presence file, flock-held by a leftover descriptor (the
    # pid-recycled shape), so the reclaim branch runs inside _announce_presence.
    presence = _wg._presence_path(db_path, os.getpid())
    presence.parent.mkdir(parents=True, exist_ok=True)
    presence.write_bytes(b'{"pid": 1, "start_ticks": 1, "acquired_at": 0.0, "role": "writer"}')
    import fcntl

    stale_fd = presence.open("a+b")
    fcntl.flock(stale_fd.fileno(), fcntl.LOCK_EX)  # contended: reclaim branch
    mutex_path = _wg._mutation_mutex_path(db_path)

    real_open = _wg._open_gate_file

    def _denied_only_mutex(lock_path, *a, **k):
        if Path(str(lock_path)) == mutex_path:
            raise PermissionError(13, "Permission denied")
        return real_open(lock_path, *a, **k)

    monkeypatch.setattr(_wg, "_open_gate_file", _denied_only_mutex)
    monkeypatch.delitem(_wg._held, str(presence), raising=False)

    try:
        with pytest.raises(StateDbGateSetupError, match="mutation mutex"):
            _wg.acquire_writer_gate(db_path)
    finally:
        fcntl.flock(stale_fd.fileno(), fcntl.LOCK_UN)
        stale_fd.close()
    assert classify_persistence_error(
        StateDbGateSetupError("cannot open or lock the presence-path mutation mutex")
    ) == "gate_setup"
    # The never-raises probe contract is untouched: the same fault on the
    # probe path (unopenable presence file) still fails closed to "live",
    # never raises.
    monkeypatch.undo()
    monkeypatch.setattr(_wg, "_open_gate_file", _denied_only_mutex)
    assert _wg._live_writer_presences(db_path)  # fail-closed: reports live


def test_api_and_profiles_re_raise_gate_refusals(tmp_path, monkeypatch):
    """P1 (caller boundaries): a gate refusal from unarchive_recoverable_session
    must surface at BOTH production caller boundaries — the api_server
    heal path returns the store-busy 503 envelope (not an empty 200 list)
    and the tui_gateway canonical resolver propagates (not a silent
    canonical_session=None)."""
    import asyncio

    from aiohttp.test_utils import TestClient, TestServer
    from unittest.mock import MagicMock

    from hermes_state_errors import StateDbWriterHeldError

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    try:
        sid = str(uuid.uuid4())
        db.create_session(session_id=sid, source="cli")
        db.end_session(sid, "ws_orphan_reap")
        db.set_session_archived(sid, True)
    finally:
        db.close()

    def _refuse(*a, **k):
        raise StateDbWriterHeldError("writer gate held: structural surgery in progress")

    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = MagicMock()
    adapter._session_db.get_session_by_title = lambda title: {
        "id": sid, "title": title, "archived": True}
    adapter._session_db.unarchive_recoverable_session = _refuse
    adapter._session_db.list_sessions_rich = lambda **k: []

    from aiohttp import web

    app = web.Application()
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app["api_server_adapter"] = adapter

    async def _drive():
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/api/sessions", params={"title": "Bot Chat"})
            return resp.status, await resp.json()

    status, payload = asyncio.run(_drive())
    assert status == 503
    assert payload["error"]["code"] == "session_db_unavailable"

    # TUI gateway boundary, through the REAL server binding (the module's
    # bodies resolve Path & friends from server.py globals, so a direct
    # module call is not the production path): the refusal surfaces as a
    # profiles.list error envelope, never a silent canonical_session=None.
    import tui_gateway.server as srv

    home = tmp_path / ".hermes"
    profile_dir = home / "profiles" / "ops"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    prof_db = SessionDB(db_path=profile_dir / "state.db")
    bot = str(uuid.uuid4())
    prof_db.create_session(bot, "cli")
    prof_db.end_session(bot, "ws_orphan_reap")
    prof_db.set_session_archived(bot, True)
    with prof_db._lock:
        prof_db._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?", ("Bot Chat", bot))
    prof_db.close()

    fake_wdb = MagicMock()
    fake_wdb.unarchive_recoverable_session = _refuse
    # The resurrect path late-imports acquire/release_or_close from
    # hermes_state_registry inside the function: a module-attribute patch is
    # the seam production reads.
    import hermes_state_registry as _reg
    from unittest.mock import patch as _patch

    with _patch.object(_reg, "acquire", lambda path: fake_wdb), \
         _patch.object(_reg, "release_or_close", lambda wdb: None):
        envelope = srv._methods["profiles.list"](1, {"include_sessions": True})
    assert envelope.get("error", {}).get("code") == 5061, envelope


def test_deferred_drain_never_unlinks_a_replacement(tmp_path):
    """P1 (stale capture): the drain's scan snapshots (key, hold) before
    taking _held_lock; a release+re-announce may install a NEW hold at the
    same path in that window. The drain must drain only the CURRENT owner
    of the key — never unlink the replacement's live presence, never pop a
    hold it does not own."""
    import hermes_state_writergate as _wg
    from hermes_state_writergate import _held, _held_lock

    db_path = tmp_path / "state.db"
    db_path.write_bytes(b"")
    presence = _wg._presence_path(db_path, os.getpid())

    # Old hold: flagged (as a budget-exhausted release leaves it), then
    # orphaned exactly as a winning release does (popped, never re-registered).
    old_token = _wg.OwnerToken("old")
    _wg.acquire_writer_gate(db_path, owner=old_token)
    with _held_lock:
        old_hold = _held[str(presence)]
        old_hold.deferred_release = True

    # The winning release: drops old share, unlinks, pops — the live path.
    assert _wg.release_writer_gate(db_path, old_token) is True
    with _held_lock:
        assert str(presence) not in _held
    assert not presence.exists()

    # A fresh announce installs a NEW hold at the same key.
    new_token = _wg.OwnerToken("new")
    _wg.acquire_writer_gate(db_path, owner=new_token)
    with _held_lock:
        new_hold = _held[str(presence)]
        assert new_hold is not old_hold

    # The drain, still holding the stale (key, old_hold) capture, must no-op.
    _wg._drain_one_deferred_release(str(presence), old_hold)
    assert presence.exists(), "drain unlinked the replacement's live presence"
    with _held_lock:
        assert _held.get(str(presence)) is new_hold, "drain popped a hold it does not own"

    # And the new hold still works: releases cleanly.
    assert _wg.release_writer_gate(db_path, new_token) is True
    assert not presence.exists()
