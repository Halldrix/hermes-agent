"""PID-namespace identity for gateway runtime records (#123081).

A gateway running under ``systemd PrivatePIDs=`` (or any container-per-gateway
layout with a shared state dir) records PID 1, because ``os.getpid()`` is
namespace-relative. Every reader outside that namespace resolves PID 1 to the
host's init, which is alive, is not a gateway, and — in the PID-file path —
caused the identity files to be unlinked. Deleting ``gateway.lock`` while the
live gateway still holds an flock on the unlinked inode let a second gateway
create a fresh lock and win: the singleton guard, the only atomic arbiter
against two gateways on one home, was bypassed and both ran.

These tests pin the three consumers that read another process' recorded PID, and
the tri-state namespace identity they share. Namespace identity is injected as
data (``local_pid_namespace``), never derived from ``sys.platform``, so the
suite is deterministic on every host; one test reads the real ``/proc`` on Linux.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import pytest

from hermes_platform.host import pid_namespace as pns
from hermes_platform.host.pid_namespace import (
    LocalPidNamespace,
    _parse_pid_namespace_link,
    local_pid_namespace,
    pid_checkable_from,
    pid_namespace_id,
)

pytestmark = pytest.mark.platforms("linux")

_HOST_NS = "4026531834"
_OTHER_NS = "4026532999"

_LIVE = LocalPidNamespace(id=_HOST_NS, supported=True)
_UNKNOWN = LocalPidNamespace(id=None, supported=True)
_NONE = LocalPidNamespace(id=None, supported=False)


@pytest.fixture
def foreign_namespace(monkeypatch):
    """This process knows its own namespace, but the record claims a different one."""
    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _LIVE)


# ---------------------------------------------------------------------------
# The tri-state identity
# ---------------------------------------------------------------------------


def test_parse_pid_namespace_link_reads_the_kernel_inode():
    assert _parse_pid_namespace_link("pid:[4026531834]") == "4026531834"
    assert _parse_pid_namespace_link("pid:[4026531834]\n") == "4026531834"
    # Not a pid namespace link at all: a PID namespace inode is never empty.
    assert _parse_pid_namespace_link("pid:[]") is None
    assert _parse_pid_namespace_link("mnt:[4026531840]") is None
    assert _parse_pid_namespace_link("") is None


def test_local_namespace_is_stable_and_known_on_this_linux_host():
    """A definite answer is cached; the namespace cannot change under a live process."""
    first = local_pid_namespace()
    assert first.supported is True
    assert first.known is True
    assert first.id and first.id.isdigit()
    assert local_pid_namespace() is first


def test_real_readlink_agrees_with_the_public_resolver():
    """The cached identity is the kernel's own inode, not a guess."""
    import os as _os

    assert pid_namespace_id(_os.getpid()) == local_pid_namespace().id


def test_failed_lookup_is_not_cached_so_a_transient_proc_problem_recovers(monkeypatch):
    """A failed /proc read must not pin the process to "unknown" for its whole life.

    The resolver retries within a single call, so a transient failure is already
    invisible by the time it returns; what must not happen is caching that
    failure. With every read failing, each call re-reads rather than remembering.
    """
    reads = {"n": 0}

    def failing_readlink(path):
        reads["n"] += 1
        raise PermissionError("transient")

    pns._local_pid_namespace_cached.cache_clear()
    monkeypatch.setattr(pns.os, "readlink", failing_readlink)
    try:
        assert local_pid_namespace().known is False
        after_first = reads["n"]
        assert local_pid_namespace().known is False
        assert reads["n"] > after_first, "the failure was cached instead of retried"
    finally:
        pns._local_pid_namespace_cached.cache_clear()


def test_a_definite_answer_is_cached(monkeypatch):
    """The resolved namespace cannot change under a live process, so it is read once."""
    reads = {"n": 0}

    def counting_readlink(path):
        reads["n"] += 1
        return f"pid:[{_HOST_NS}]"

    pns._local_pid_namespace_cached.cache_clear()
    monkeypatch.setattr(pns.os, "readlink", counting_readlink)
    try:
        first = local_pid_namespace()
        assert first.id == _HOST_NS
        local_pid_namespace()
        assert reads["n"] == 1
    finally:
        pns._local_pid_namespace_cached.cache_clear()


# ---------------------------------------------------------------------------
# The predicate every consumer routes through
# ---------------------------------------------------------------------------


def test_pid_checkable_matches_only_inside_the_same_namespace(monkeypatch):
    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _LIVE)
    assert pid_checkable_from(_HOST_NS) is True
    assert pid_checkable_from(_OTHER_NS) is False


def test_pid_checkable_keeps_hostname_only_semantics_where_no_namespace_exists(monkeypatch):
    """macOS/Windows: one namespace, so a bare PID keeps its meaning."""
    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _NONE)
    assert pid_checkable_from(None) is True
    assert pid_checkable_from(_OTHER_NS) is True


def test_pid_checkable_fails_closed_when_our_own_lookup_failed(monkeypatch):
    """Absence of provenance cannot become provenance because the read failed."""
    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _UNKNOWN)
    assert pid_checkable_from(_HOST_NS) is False
    assert pid_checkable_from(None) is False


def test_pid_checkable_keeps_legacy_unstamped_records_probeable(monkeypatch):
    """The rollout boundary: an unstamped record keeps main's behavior.

    Refusing to probe it would make every pre-upgrade record permanently
    unverifiable, silently disabling unclean-death detection for every install
    that had not yet restarted on a stamping build. Protection arrives when the
    gateway next restarts.
    """
    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _LIVE)
    assert pid_checkable_from(None) is True


# ---------------------------------------------------------------------------
# Consumer 1: the runtime record carries the namespace that issued its PID
# ---------------------------------------------------------------------------


def test_pid_record_stamps_the_pid_namespace(tmp_path, monkeypatch):
    from gateway import status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = status._build_pid_record()
    assert record["pidns"] == local_pid_namespace().id

    # The stamp reaches the file the next process reads.
    status.write_pid_file()
    payload = json.loads((tmp_path / "gateway.pid").read_text())
    assert payload["pidns"] == local_pid_namespace().id


def test_pid_record_omits_pidns_where_no_namespace_exists(tmp_path, monkeypatch):
    from gateway import status

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # Patch the CONSUMING module: status.py imports the resolver by name, so a
    # patch on the defining module would pass silently.
    monkeypatch.setattr(status, "local_pid_namespace", lambda: _NONE)
    assert "pidns" not in status._build_pid_record()


# ---------------------------------------------------------------------------
# Consumer 2: the unlink that bypassed the singleton guard
# ---------------------------------------------------------------------------


def _write_pair(tmp_path, monkeypatch, record):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "gateway.pid").write_text(json.dumps(record))
    (tmp_path / "gateway.lock").write_text(json.dumps(record))


def _held_lock(home):
    """Hold a real flock on ``home/gateway.lock`` for the duration of the test."""
    from gateway import status

    lock = home / "gateway.lock"
    handle = open(lock, "a+", encoding="utf-8")
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": 1, "kind": "hermes-gateway"}))
    handle.flush()
    assert status._try_acquire_file_lock(handle)
    return handle


def test_foreign_namespace_record_does_not_lose_its_live_runtime_lock(tmp_path, monkeypatch, foreign_namespace):
    """The #123081 incident, end to end.

    The live gateway holds gateway.lock from inside PrivatePIDs=. A checker
    outside the namespace reads its recorded PID 1 as the host's init and calls
    it dead. Before the fix that unlinked gateway.pid AND gateway.lock; the live
    flock stayed on the deleted inode, the next acquire created a fresh file and
    succeeded, and two gateways ran against one Telegram token.
    """
    from gateway import status

    foreign_record = {
        "pid": 1, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(1), "pidns": _OTHER_NS,
        "hermes_home": str(tmp_path),
    }
    _write_pair(tmp_path, monkeypatch, foreign_record)

    ready, release = threading.Event(), threading.Event()

    def hold():
        handle = _held_lock(tmp_path)
        ready.set()
        release.wait(30)
        status._release_file_lock(handle)
        handle.close()

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert ready.wait(5)
    time.sleep(0.2)
    try:
        assert status.is_gateway_runtime_lock_active(tmp_path / "gateway.lock") is True
        assert status.get_running_pid() is None  # unverifiable, not "dead"

        # The fix: the identity files survive, so the flock still arbitrates.
        assert (tmp_path / "gateway.pid").exists(), "refused to unlink a live gateway's pid file"
        assert (tmp_path / "gateway.lock").exists(), "refused to unlink a live gateway's lock"
        assert status.acquire_gateway_runtime_lock() is False, "second gateway was admitted"
    finally:
        status.release_gateway_runtime_lock()
        release.set()
        holder.join(5)


def test_same_namespace_record_still_cleans_up_a_genuine_poison_file(tmp_path, monkeypatch):
    """#89315 keeps working: a record we CAN verify is unlinked as before."""
    from gateway import status

    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _LIVE)
    dead = 2 ** 22 + 12345
    _write_pair(tmp_path, monkeypatch, {
        "pid": dead, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(dead), "pidns": _HOST_NS,
        "hermes_home": str(tmp_path),
    })
    status.get_running_pid()
    assert not (tmp_path / "gateway.pid").exists()
    assert not (tmp_path / "gateway.lock").exists()


def test_unstamped_legacy_record_still_cleans_up(tmp_path, monkeypatch):
    """A pre-upgrade record (no pidns) on a host with no namespace concept is stale-checked as before."""
    from gateway import status

    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _NONE)
    dead = 2 ** 22 + 12345
    _write_pair(tmp_path, monkeypatch, {
        "pid": dead, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(dead), "hermes_home": str(tmp_path),
    })
    status.get_running_pid()
    assert not (tmp_path / "gateway.pid").exists()


# ---------------------------------------------------------------------------
# Consumer 3: the lifecycle ledger verdict and the scoped bot-token lock
# ---------------------------------------------------------------------------


def test_lifecycle_ledger_does_not_call_a_foreign_namespace_gateway_unclean(monkeypatch):
    """`phase=running` from another namespace is "cannot verify", not "exited UNCLEANLY"."""
    from gateway import lifecycle_ledger as ll

    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _LIVE)
    assert ll._pid_is_sentinel_owner(1, 12345.0, 12345.0, _OTHER_NS) is True
    # Same namespace: the real probe decides, unchanged.
    assert ll._pid_is_sentinel_owner(2 ** 22 + 12345, 12345.0, 12345.0, _HOST_NS) is False


def test_lifecycle_ledger_records_the_namespace_it_claimed_in(tmp_path, monkeypatch):
    from gateway import lifecycle_ledger as ll

    record = {"phase": "running", "pid": os.getpid(), "start_time": 1.0, "pidns": _HOST_NS}
    path = tmp_path / "state" / "gateway.lifecycle.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    # Our own live process in our own namespace is still a live owner.
    assert ll.detect_unclean_exit(tmp_path) is None


def test_scoped_lock_from_another_namespace_is_not_stolen(monkeypatch):
    """A second gateway must not take a live bot token because PID 1 is not a gateway cmdline."""
    from gateway import status

    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _LIVE)
    live_pid = 1  # the namespaced gateway's own number, which is the host's init out here
    foreign = {
        "pid": live_pid, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(live_pid), "pidns": _OTHER_NS,
    }
    # Host init's cmdline is not a gateway, so every pre-fix staleness signal fires.
    assert status._read_process_cmdline(live_pid) is not None
    assert status._scoped_lock_record_is_stale(foreign, live_pid) is False


def test_scoped_lock_from_our_namespace_still_reclaims_a_dead_owner(monkeypatch):
    from gateway import status

    monkeypatch.setattr(pns, "local_pid_namespace", lambda: _LIVE)
    dead = 2 ** 22 + 12345
    same_ns = {
        "pid": dead, "kind": "hermes-gateway", "argv": ["hermes", "gateway", "run"],
        "start_time": status._get_process_start_time(dead), "pidns": _HOST_NS,
    }
    assert status._scoped_lock_record_is_stale(same_ns, dead) is True
