"""Cross-process writer coordination for ``state.db`` (see #103339).

Design (v3): SQLite owns *row* concurrency; this gate owns *file-structure*
concurrency. Ordinary row writes (appends, claims, handoffs, leases) from any
number of processes proceed under SQLite's own WAL locking — exactly as on
base, where a pinned conformance cell proves 8 concurrent claimants stay
exactly-once. What this gate forbids is *structural* work under a live
writer — schema surgery, checkpoints from a second connection, VACUUM-class
operations — the evidenced corruption class: the second connection's WAL
handling unlinks/replaces the WAL inode the owning gateway holds.

Mechanism, two files beside the database:

- Presence: ``<state.db>.writer.<pid>.lock``. Each writing process holds its
  OWN file exclusive for as long as it may write (first write → close).
  Per-pid files never contend with each other, so any number of writers
  coexist; a probe enumerates them to learn that a writer lives. Crashed
  writers leave litter, which the probe reaps (see below) — no wedged state.
- Structural exclusion: ``<state.db>.writer.lock`` (global). A structural
  operation (today: schema repair) holds it exclusive for the whole surgery.
  Row announces refuse while it is held, and it is only grantable when no
  writer presence is live — closing the race in both directions.

Fail-closed everywhere: contention, an unopenable file, or an unreadable
holder record refuses the structural op, never allows it. Row announces
refuse only while a surgery holds the global lock.

Probe liveness rules for a presence file (all best-effort, never raising):

- Our own pid's file, registered locally → ours, skip.
- Exclusive flock still acquirable → holder gone. Unlink + ignore, UNLESS the
  file is brand new with no readable record (a racing announcer mid-create)
  or the record names a live pid (a mid-close race) — both read as LIVE.
- Exclusive flock contended → live holder; role comes from its record
  (``writer``; unreadable → ``unknown``, still a refusal).

Roles (``writer`` / ``repair``) live in the global lock's record so repair
can tell a fellow repairer (serialize via the repair lock — queuing is
correct) from a live writer (refuse).

Ownership is per SessionDB instance: ``SessionDB.close()`` releases its
share, and presence drops when the last in-process owner closes. A live
gateway (registry-owned handle, never closed mid-life) therefore announces
exactly while it can write; a CLI that wrote and closed does not pin
presence against a later repair. (``close()`` on a registry-shared instance
only decrements the registry refcount; the share releases on final registry
release, which runs the real ``close()``.)

``fork()`` needs no pid guard by construction: presence files are keyed by
pid, so the child naturally announces under its own pid, and release only
ever unlinks a file whose pid matches the releaser — a child can never
delete its parent's presence. A child that never touches the gate leaves no
trace; one that writes announces like any other second process.

Read-only paths never touch the gate (``sessions list/export``, backups,
health probes, ``SessionDB(read_only=True)``).

Windows tier: ``msvcrt.locking`` is exclusive-only. Takes are exclusive on
both platforms; observation probes are shared on POSIX (concurrent probes
never contend with each other) but exclusive on Windows, where a contended
observation retries briefly to separate probe transients from real surgery
holds. Residual Windows-only case (a probe paused indefinitely while
holding): fail-closed spurious refusal, retryable by the caller.

Known limitation: open-time writes that bypass ``_execute_write`` (the
``state_meta`` generation stamp, schema init DDL) announce nothing — gating
the open would refuse speculative writer handles that never write
(``sessions list`` while the gateway runs).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
import weakref
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from hermes_state_common import (
    _proc_start_ticks,
    _read_lock_holder_record,
    _rewrite_lock_file,
    is_advisory_lock_contention,
)
from hermes_state_errors import StateDbGateSetupError, StateDbWriterHeldError

logger = logging.getLogger("hermes_state")

_IS_WINDOWS = sys.platform == "win32"

#: Global structural lock: ``<state.db>.writer.lock`` sits beside the database.
_WRITER_LOCK_SUFFIX = ".writer.lock"
#: Presence-file infixes: ``<state.db>.writer.<pid>.lock``.
_WRITER_PRESENCE_INFIX = ".writer."
_PRESENCE_SUFFIX = ".lock"

#: Fresh-file window (seconds): a flock-free presence file younger than this
#: with no readable record is a racing announcer, not litter — read as live.
_PRESENCE_SETTLE_SECONDS = 60.0

#: Holder roles recorded in lock files.
_WRITER_ROLE = "writer"
_REPAIR_ROLE = "repair"
_UNKNOWN_ROLE = "unknown"


class _GateHold:
    """One held lock file plus its in-process owners and acquirer pid.

    ``structural_owner`` is only set on the global structural hold: the token
    of the operation that owns the surgery. Same-process re-entry is allowed
    solely for that identical token — a distinct structural owner (repair vs
    checkpoint) must serialize outside, never piggyback.

    ``anon_leases`` counts anonymous (owner=None) announces on a presence
    hold. Anonymous callers share an empty ``owners`` set by design, so the
    count — incremented under ``_held_lock`` for every anonymous announce —
    is what lets a refused announce roll back exactly its own lease without
    disturbing other live anonymous callers.
    """

    __slots__ = ("handle", "owners", "role", "acquired_pid", "structural_owner", "anon_leases",
                 "deferred_release", "db_path")

    def __init__(self, handle, role: str, db_path: Optional[Path] = None):
        self.handle = handle
        self.role = role
        self.owners = weakref.WeakSet()
        self.acquired_pid = os.getpid()
        self.structural_owner = None
        self.anon_leases = 0
        #: Sticky flag: a release exhausted the mutation-mutex budget with the
        #: hold still pinned, so the pathname cleanup is owed. The module-level
        #: drain completes it once no owner/lease is live — the cleanup can
        #: never be stranded by the owner dying without another retry (P1).
        self.deferred_release = False
        #: ``db_path`` for the deferred drain (presence-path mutations need it).
        self.db_path = db_path


class OwnerToken:
    """Weakref-able owner token for :func:`acquire_writer_gate` (a bare
    ``object()`` cannot be weak-referenced). SessionDB instances pass
    themselves; one-shot holders (repair surgery) mint one of these."""

    __slots__ = ("label", "__weakref__")

    def __init__(self, label: str = ""):
        self.label = label

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"OwnerToken({self.label!r})"


#: Lock path -> live hold. Guarded by ``_held_lock``.
_held: Dict[str, _GateHold] = {}
_held_lock = threading.Lock()


def _writer_lock_path(db_path: Path) -> Path:
    """Global structural lock file for *db_path*."""
    try:
        resolved = Path(db_path).expanduser().resolve()
    except OSError:
        resolved = Path(db_path).expanduser().absolute()
    return resolved.with_name(resolved.name + _WRITER_LOCK_SUFFIX)


def _presence_path(db_path: Path, pid: int) -> Path:
    """This process's presence file for *db_path* (never shared across pids,
    so a symlink and its target resolve to one gate like the registry)."""
    try:
        resolved = Path(db_path).expanduser().resolve()
    except OSError:
        resolved = Path(db_path).expanduser().absolute()
    return resolved.with_name(
        f"{resolved.name}{_WRITER_PRESENCE_INFIX}{pid}{_PRESENCE_SUFFIX}"
    )


#: Presence-path mutation mutex: ``<state.db>.writer.mtx.lock``. Persistent
#: sentinel beside the database (never unlinked, like the global lock).
#: Every presence-pathname mutation (stale reclaim unlink+create, litter
#: reap check+unlink, orphan break, release unlink) holds it exclusive, so a
#: replacement can never win the pathname between another actor's identity
#: check and unlink. Fresh creates need no mutex (they delete nothing).
#: Callers always take ``_held_lock`` first (mutex ward); the mutex is the
#: inner lock for cross-process serialization. Contention fails closed.
#: Release-path mutex retries: a closer colliding with a reaper's
#: microsecond mutex hold rides it out instead of leaving litter. Past the
#: budget the hold is re-registered (flock still held, owner restored),
#: flagged ``deferred_release``, and the module-level drain finishes the
#: unlinking as soon as the mutex frees — cleanup never depends on the
#: owner surviving to retry (never an unlinked-but-recorded ghost either).
_RELEASE_MUTEX_ATTEMPTS = 10
_RELEASE_MUTEX_RETRY_S = 0.02
_MUTATION_MUTEX_SUFFIX = ".writer.mtx.lock"

#: Deferred-release drain: a release that exhausts the mutation-mutex budget
#: leaves its hold flagged (share dropped, flock + presence file still held).
#: This drain completes the pathname cleanup once no owner/lease is live. It
#: works from the hold record alone, so the cleanup cannot be stranded when
#: the owner that owed it dies without another ``close()`` (P1) — a caller's
#: lifecycle retry is a second chance, never the only one.
_DEFERRED_RELEASE_DRAIN_POLL_S = 0.2
_DEFERRED_RELEASE_DRAIN_MAX_S = 900.0

_drain_lock = threading.Lock()
_drain_running = False
#: Bumped on every deferred-release flag set; the drain compares epochs to
#: decide whether a NEW hold needs a fresh worker after it exits.
_deferred_release_epoch = 0


def _mutation_mutex_path(db_path: Path) -> Path:
    """Mutex file serializing presence-pathname mutations for *db_path*."""
    try:
        resolved = Path(db_path).expanduser().resolve()
    except OSError:
        resolved = Path(db_path).expanduser().absolute()
    return resolved.with_name(resolved.name + _MUTATION_MUTEX_SUFFIX)


@contextlib.contextmanager
def _presence_mutation_serialized(db_path: Path):
    """Hold the presence-path mutation mutex across a pathname mutation.

    Yields True when serialized (caller may unlink/create). Yields False on
    real contention — another cooperating mutator holds the mutex. Yields
    None when the mutex itself cannot be opened or locked (an unexpected,
    non-contention failure): retrying never helps there, so the announce
    path classifies it as ``StateDbGateSetupError`` instead of a
    live-holder refusal. Failure callers must fail closed (report live /
    refuse the announce), never mutate. ``_held_lock`` must already be held
    (it is always the outer lock). Never raises.
    """
    try:
        handle = _open_gate_file(_mutation_mutex_path(db_path))
    except OSError:
        yield None
        return
    try:
        held = _try_flock_nb(handle)
        if held is not True:
            yield (False if held is False else None)
            return
        yield True
    finally:
        _unlock_handle(handle)
        try:
            handle.close()
        except OSError:
            pass


def _presence_pid(lock_path: Path, db_name: str) -> Optional[int]:
    """PID encoded in a presence filename, or None when it is not one."""
    name = lock_path.name
    prefix = db_name + _WRITER_PRESENCE_INFIX
    if not name.startswith(prefix) or not name.endswith(_PRESENCE_SUFFIX):
        return None
    try:
        return int(name[len(prefix):-len(_PRESENCE_SUFFIX)])
    except ValueError:
        return None


def _cmdline_of(pid: int) -> Optional[str]:
    """Best-effort ``pid -> cmdline`` for refusal messages; None when unknowable."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            return cmd[:200] if cmd else None
    except (OSError, ValueError):
        return None


def _read_gate_record(lock_path: Path) -> Optional[dict]:
    """Best-effort parse of the holder record; None when unreadable."""
    try:
        with lock_path.open("rb") as fh:
            record = _read_lock_holder_record(fh)
    except OSError:
        return None
    return record if isinstance(record, dict) else None


def _holder_role(record: Optional[dict]) -> str:
    if not record:
        return _UNKNOWN_ROLE
    role = record.get("role")
    return role if role in (_WRITER_ROLE, _REPAIR_ROLE) else _UNKNOWN_ROLE


def _describe_holder(lock_path: Path, record: Optional[dict] = None) -> str:
    """Human-readable holder for refusal messages; never raises."""
    if record is None:
        record = _read_gate_record(lock_path)
    if not record:
        return f"another process holds {lock_path} (holder record unreadable)"
    pid = record.get("pid")
    try:
        pid = int(pid) if pid is not None else 0
    except (TypeError, ValueError):
        pid = 0
    if pid <= 0:
        return f"another process holds {lock_path} (holder record has no pid)"
    cmd = _cmdline_of(pid)
    who = f"pid {pid}" + (f" ({cmd})" if cmd else "")
    return f"another process holds {lock_path} ({who})"


def _open_gate_file(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    return lock_path.open("a+b")


def _try_flock_nb(handle) -> Optional[bool]:
    """Non-blocking exclusive flock: True (held), False (contended), None
    (non-contention OSError — fail closed upstream)."""
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError) as exc:
        if not is_advisory_lock_contention(exc):
            logger.warning(
                "state.db writer gate: unexpected lock error (fail closed): %s", exc
            )
            return None
        return False


def _try_flock_observe_nb(handle) -> Optional[bool]:
    """Observation flock for probes: shared on POSIX, exclusive on Windows.

    POSIX ``LOCK_SH`` lets concurrent probes coexist — a probe never mistakes
    a fellow probe's transient observation for structural ownership; only a
    real exclusive holder (surgery) contends. ``msvcrt.locking`` has no shared
    mode, so Windows probes stay exclusive (see ``_probe_global``'s retry,
    which separates microsecond probe transients from real surgery holds).
    Same fail-closed contract as :func:`_try_flock_nb`.
    """
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError) as exc:
        if not is_advisory_lock_contention(exc):
            logger.warning(
                "state.db writer gate: unexpected lock error (fail closed): %s", exc
            )
            return None
        return False


def _unlock_handle(handle) -> None:
    """Release a disposable handle's flock (never a registered hold)."""
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _write_gate_record(handle, role: str) -> None:
    """Record this process as holder (best effort, under the flock)."""
    record = {
        "pid": os.getpid(),
        "start_ticks": _proc_start_ticks(os.getpid()),
        "acquired_at": time.time(),
        "role": role,
    }
    _rewrite_lock_file(handle, json.dumps(record, sort_keys=True).encode("utf-8"))


def _pid_is_alive(pid: int) -> Optional[bool]:
    """Windows-safe liveness: True/False, None when unknowable (fail closed).

    Uses ``psutil.pid_exists`` (core dependency); falls back to ``os.kill``
    only on POSIX. Never calls ``os.kill(pid, 0)`` on Windows (CPython maps
    signal 0 to CTRL_C_EVENT there).
    """
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return False
    try:
        import psutil as _psutil  # type: ignore[import-not-found]
    except ImportError:
        _psutil = None  # type: ignore[assignment]
    if _psutil is not None:
        try:
            return bool(_psutil.pid_exists(pid))
        except Exception:
            return None
    if _IS_WINDOWS:
        return None  # unknowable without psutil: fail closed
    try:
        os.kill(pid, 0)  # windows-footgun: ok — POSIX-only (guarded by _IS_WINDOWS above)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return None


def _file_age_seconds(path: Path) -> float:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return 0.0


def _path_identity_matches_handle(handle, lock_path: Path) -> Optional[bool]:
    """True when *lock_path* still names *handle*'s inode (dev+ino, both nonzero).

    False when the pathname names a different inode (a replacement won the
    path between our open and now). None when identity cannot be established
    (fstat/stat failure, zero inode on network FS). ENOENT counts as False —
    callers re-stat to tell gone (reaped) from replaced (live).
    """
    try:
        fd_stat = os.fstat(handle.fileno())
    except OSError:
        return None
    try:
        path_stat = lock_path.stat()
    except OSError:
        return False
    if not fd_stat.st_ino or not path_stat.st_ino:
        return None
    return (fd_stat.st_dev, fd_stat.st_ino) == (path_stat.st_dev, path_stat.st_ino)


def _locked_reap_presence(handle, lock_path: Path, db_path: Path) -> bool:
    """Unlink *lock_path* while the caller holds its flock; identity-safe.

    Returns True when the pathname is gone afterwards (reaped: do NOT report
    live). Returns False to treat as live (fail closed): the mutation mutex
    is contended, a replacement was detected, identity is unverifiable on
    POSIX, or the unlink failed. The mutex makes check+unlink atomic against
    cooperating mutators (reclaim/reap/release all serialize on it); the
    identity check additionally catches replacements that slipped in outside
    the mutex. On Windows a contested unlink fails via OS share semantics
    (failure → live), the safe equivalent.
    """
    with _presence_mutation_serialized(db_path) as serialized:
        if not serialized:
            return False
        same = _path_identity_matches_handle(handle, lock_path)
        if same is False:
            try:
                lock_path.stat()
            except OSError:
                return True  # already gone: reaped
            return False  # replacement holds the path: live
        if same is None and not _IS_WINDOWS:
            return False  # POSIX: unverifiable identity never reaps
        try:
            lock_path.unlink()
        except OSError:
            return False
        return True


def _unlink_orphan_presence(lock_path: Path, db_path: Path, expected_record) -> bool:
    """Best-effort removal of an orphan-held presence pathname (no flock held).

    For the #100108 shape: recorded holder dead, a fork-inherited descriptor
    holds the flock. Opens fresh (no flock), reaps only when the CURRENT
    record still equals ``expected_record`` (the dead classification made
    pre-mutex) AND the open probe matches the current pathname identity —
    a replacement that won the path in between differs in record content
    and is kept live. True means gone (do NOT report live), False means
    treat as live. Never raises.
    """
    with _presence_mutation_serialized(db_path) as serialized:
        if not serialized:
            return False
        try:
            probe = lock_path.open("r+b")
        except FileNotFoundError:
            return True
        except OSError:
            return False
        try:
            if _path_identity_matches_handle(probe, lock_path) is not True:
                return False
            if expected_record is not None and _read_gate_record(lock_path) != expected_record:
                return False  # replaced after classification: live
            try:
                lock_path.unlink()
            except OSError:
                return False
            return True
        finally:
            try:
                probe.close()
            except OSError:
                pass


def _presence_holder_provably_dead(record) -> bool:
    """POSIX-only break-glass: True only when the recorded holder is provably
    dead or PID-recycled (same contract as
    ``hermes_state_common._lock_holder_provably_dead``). Always False on
    Windows (fail closed) — it must never run there: CPython maps
    ``os.kill(pid, 0)`` to CTRL_C_EVENT.
    """
    if _IS_WINDOWS:
        return False
    try:
        from hermes_state_common import _lock_holder_provably_dead
        return bool(_lock_holder_provably_dead(record))
    except Exception:
        return False


def _live_writer_presences(db_path: Path, *, include_self: bool = False) -> List[str]:
    """Descriptions of live foreign writer presences; reaps crash litter.

    Never raises. A flock-free file is litter ONLY when it is old or names a
    provably-absent record; a fresh file with no record is a racing announcer
    (fail closed), as is a file whose record names a live pid (mid-close).

    With ``include_self`` (structural-take quietness check) the caller's own
    presence counts as live: same-process reentrancy applies only to the same
    structural operation (global re-entry), never to unrelated writer owners.
    """
    try:
        resolved = Path(db_path).expanduser().resolve()
    except OSError:
        resolved = Path(db_path).expanduser().absolute()
    parent, db_name = resolved.parent, resolved.name
    try:
        candidates = list(parent.glob(db_name + _WRITER_PRESENCE_INFIX + "*" + _PRESENCE_SUFFIX))
    except OSError:
        return [f"cannot scan {parent} for writer presence (fail closed)"]
    live: List[str] = []
    me = os.getpid()
    for lock_path in candidates:
        pid = _presence_pid(lock_path, db_name)
        if pid is None:
            continue
        # One mutex around open + flock + unlock + close: sibling threads'
        # identical sequences can never overlap, so probes never mistake each
        # other for a foreign holder (same-process flock descriptions
        # contend). Record reads and litter reaps stay outside (fail-closed).
        # Own files skip via registry AND pid (unless include_self): a forked
        # child inherits the registry copy but has a new pid, so it must still
        # see (never skip) its parent's presence file.
        with _held_lock:
            if not include_self and pid == me and str(lock_path) in _held:
                continue  # ours
            try:
                handle = lock_path.open("a+b")
            except OSError:
                live.append(f"unopenable presence file {lock_path} (fail closed)")
                continue
            try:
                held = _try_flock_nb(handle)
                if held is True:
                    _unlock_handle(handle)
            finally:
                try:
                    handle.close()
                except OSError:
                    pass
            if held is None:
                live.append(f"unprovable presence file {lock_path} (fail closed)")
                continue
            if held is False:
                # Contended: usually a live holder. POSIX orphan exception
                # (#100108 shape): recorded holder provably dead while a
                # fork-inherited descriptor holds the flock — break it like
                # _acquire_db_flock does, else one dead parent blocks surgery
                # until its unrelated child exits. Windows never takes this
                # branch (no fork; fail closed to live).
                if not _IS_WINDOWS:
                    _rec = _read_gate_record(lock_path)
                    if _rec is not None and _presence_holder_provably_dead(_rec):
                        # Already under _held_lock (mutex ward above): the
                        # helper takes only the cross-process mutex, never
                        # _held_lock (threading.Lock is not reentrant).
                        # _rec is the pre-mutex classification; the helper
                        # re-reads under the mutex and only unlinks when the
                        # current record still equals it (no replacement won
                        # the path in between).
                        if _unlink_orphan_presence(lock_path, db_path, _rec):
                            continue  # reaped: not live
                live.append(_describe_holder(lock_path, _read_gate_record(lock_path)))
                continue
        # Flock-free here: litter only when provably not a racer. A free
        # flock plus a record naming a dead pid cannot be a racing announcer
        # (announcers hold the flock while writing their record), so it is
        # reaped regardless of file age.
        record = _read_gate_record(lock_path)
        if record is None and _file_age_seconds(lock_path) < _PRESENCE_SETTLE_SECONDS:
            live.append(_describe_holder(lock_path, record))  # racing announcer
            continue
        if record is not None and _holder_role(record) == _UNKNOWN_ROLE:
            live.append(_describe_holder(lock_path, record))  # conservative
            continue
        if record is not None:
            rp = record.get("pid")
            try:
                rp = int(rp) if rp is not None else 0
            except (TypeError, ValueError):
                rp = 0
            if rp > 0:
                if not _IS_WINDOWS:
                    # POSIX: pid existence alone cannot prove liveness (PID
                    # reuse); compare start_ticks via the provably-dead
                    # contract. Unknown identity fails closed to live.
                    if not _presence_holder_provably_dead(record):
                        live.append(_describe_holder(lock_path, record))  # live or unknowable
                        continue
                else:
                    alive = _pid_is_alive(rp)
                    if alive is True:
                        live.append(_describe_holder(lock_path, record))  # mid-close race
                        continue
                    if alive is None:
                        live.append(_describe_holder(lock_path, record))  # unknowable: closed
                        continue
                # Dead pid with a record, flock already proven free above:
                # crash litter (a racing announcer holds the flock while
                # writing, so this cannot be one). Reap under the mutex with
                # the flock HELD across the unlink: a replacement winning the
                # pathname in between is detected by identity and kept live.
                with _held_lock:
                    try:
                        handle = lock_path.open("a+b")
                    except OSError:
                        continue
                    try:
                        if _try_flock_nb(handle) is not True:
                            # Lost the re-verify race (someone holds it now):
                            # no lock held here, so no unlock — just report.
                            live.append(_describe_holder(
                                lock_path, _read_gate_record(lock_path)))
                            continue
                        if not _locked_reap_presence(handle, lock_path, db_path):
                            live.append(_describe_holder(
                                lock_path, _read_gate_record(lock_path)))
                        _unlock_handle(handle)
                    finally:
                        try:
                            handle.close()
                        except OSError:
                            pass
                continue
        # Re-verify under the mutex before reaping (covers the residual
        # no-record old-litter path; dead-pid records were reaped above).
        # The flock stays HELD across the unlink (same identity-safe reap).
        with _held_lock:
            try:
                handle = lock_path.open("a+b")
            except OSError:
                continue
            try:
                if _try_flock_nb(handle) is not True:
                    live.append(_describe_holder(lock_path, _read_gate_record(lock_path)))
                    continue
                if _file_age_seconds(lock_path) < _PRESENCE_SETTLE_SECONDS:
                    live.append(_describe_holder(lock_path, _read_gate_record(lock_path)))
                    _unlock_handle(handle)
                    continue
                if not _locked_reap_presence(handle, lock_path, db_path):
                    live.append(_describe_holder(lock_path, _read_gate_record(lock_path)))
                    _unlock_handle(handle)
                    continue
                _unlock_handle(handle)
            finally:
                try:
                    handle.close()
                except OSError:
                    pass
    return live


def _probe_global(db_path: Path, *, settle_role: bool = False) -> Tuple[bool, str, str]:
    """Non-holding probe of the global structural lock:
    ``(foreign_held, role, description)``. Never raises (fail closed).

    Observation uses a SHARED flock on POSIX, so concurrent probes never
    contend with each other — only a real exclusive holder (surgery) refuses
    a row writer. On Windows (no shared ``msvcrt.locking`` mode) a contended
    observation is retried briefly: a fellow probe's microsecond hold clears,
    a real surgery's hold persists. Residual Windows-only case (a probe
    paused indefinitely while holding): fail-closed spurious refusal,
    retryable by the caller.
    """
    lock_path = _writer_lock_path(db_path)
    key = str(lock_path)
    attempts = 6 if settle_role else 1
    last: Tuple[bool, str, str] = (True, _UNKNOWN_ROLE, _describe_holder(lock_path))
    for _ in range(attempts):
        # Mutex Ward (see _live_writer_presences): open + flock + unlock +
        # close never overlap a sibling thread, so no false contention.
        # No self fast-path here: a global hold by this process must still
        # refuse row callers (unrelated owners are never invisible). Re-entry
        # for the same structural operation is handled in acquire_writer_gate
        # itself, not in the probe. Fork-inherited holds are dropped below.
        with _held_lock:
            hold = _held.get(key)
            if hold is not None and hold.acquired_pid != os.getpid():
                # Fork-inherited: drop locally without touching the fd (the
                # parent's description keeps the kernel lock; unlocking here
                # would release the parent).
                try:
                    hold.handle.close()
                except OSError:
                    pass
                _held.pop(key, None)
            try:
                handle = _open_gate_file(lock_path)
            except OSError as exc:
                return True, _UNKNOWN_ROLE, f"cannot open {lock_path} ({exc})"
            try:
                held = _try_flock_observe_nb(handle)
                if held is True:
                    _unlock_handle(handle)
                elif held is False and _IS_WINDOWS:
                    # No shared mode on Windows: a contended observation may
                    # be a fellow probe's microsecond hold, not a surgery.
                    # Retry briefly — transients clear, real holds persist.
                    for _ in range(10):
                        time.sleep(0.005)
                        try:
                            handle.seek(0)
                        except OSError:
                            break
                        held = _try_flock_observe_nb(handle)
                        if held is not False:
                            break
                    if held is True:
                        _unlock_handle(handle)
            finally:
                try:
                    handle.close()
                except OSError:
                    pass
            if held is not True:
                pass  # contended: report held below (no self fast-path)
        if held is True:
            return False, _UNKNOWN_ROLE, ""
        record = _read_gate_record(lock_path)
        role = _holder_role(record)
        if role != _UNKNOWN_ROLE or not settle_role:
            return True, role, _describe_holder(lock_path, record)
        last = (True, role, _describe_holder(lock_path, record))
        time.sleep(0.2)
    return last


def _refusal(db_path: Path, what: str) -> StateDbWriterHeldError:
    return StateDbWriterHeldError(
        f"state.db writer gate: {what}; refusing structural work on {db_path} so it cannot "
        "corrupt the live WAL (see #103339). Stop that process's gateway "
        "(`hermes gateway stop`) and retry."
    )


def acquire_writer_gate(db_path: Path, *, role: str = _WRITER_ROLE, owner=None,
                        exclusive: bool = False) -> None:
    """Take this process's share of *db_path*'s writer gate.

    - Row announces (``exclusive=False``, the ``_execute_write`` path):
      idempotent per process; records presence in this process's own
      presence file and refuses ONLY while a surgery holds the global
      structural lock (fail-closed in both directions). Any number of
      processes may announce concurrently — SQLite arbitrates their row
      writes, as on base.
    - Structural takes (``exclusive=True``, repair surgery): ATOMIC —
      takes the global lock, then scans foreign writer presences, and
      releases + refuses when any are live. Only an actually-quiet database
      is ever returned held. Same-process re-entry is allowed solely for the
      identical owner token (same logical operation); a distinct structural
      owner refuses and serializes outside (repairers queue via the repair
      lock). Requires a non-None ``owner``. This primitive is the single
      authority carrier: every structural caller (repair, checkpoints,
      future VACUUM/optimize/migration gates) takes through here instead of
      reimplementing probe-then-take.
    """
    if role not in (_WRITER_ROLE, _REPAIR_ROLE):
        role = _WRITER_ROLE
    if exclusive and owner is None:
        raise ValueError(
            "state.db writer gate: exclusive take requires an owner token "
            "(anonymous structural holds cannot be released or re-entered)."
        )
    if not exclusive:
        _announce_presence(db_path, owner=owner)
        foreign, _role, description = _probe_global(db_path)
        if foreign:
            # Our presence was registered BEFORE the probe (announce-first
            # closes the check-then-act gap); a refusal must roll it back or
            # the refused opener's file litters and blocks later surgery.
            _rollback_announce(db_path, owner)
            raise StateDbWriterHeldError(
                f"state.db writer gate: {description}; refusing to write {db_path} while a "
                "structural operation owns it (see #103339). Retry once it finishes."
            )
        return
    lock_path = _writer_lock_path(db_path)
    key = str(lock_path)
    # Mutex Ward: the whole check + open + flock + register sequence runs
    # under one hold so sibling threads can never interleave two takes of
    # the same file (same-process flock descriptions contend).
    # Fork safety: an inherited hold is foreign — drop it locally (never
    # unlock: the parent's description keeps the kernel lock) and take fresh.
    with _held_lock:
        hold = _held.get(key)
        if hold is not None and hold.acquired_pid != os.getpid():
            try:
                hold.handle.close()
            except OSError:
                pass
            _held.pop(key, None)
            hold = None
        if hold is not None:
            # Same-process re-entry is the SAME logical operation only: the
            # identical owner token. A distinct structural owner (checkpoint
            # vs repair) must serialize outside — piggybacking would run
            # surgery under a foreign hold.
            if hold.structural_owner is not owner:
                raise _refusal(
                    db_path,
                    f"another structural operation of this process holds {lock_path}",
                )
            hold.owners.add(owner)
            return
        try:
            handle = _open_gate_file(lock_path)
        except OSError as exc:
            raise StateDbWriterHeldError(
                f"state.db writer gate: cannot open {lock_path} ({exc}); refusing structural work on "
                f"{db_path} rather than risk a second-writer corruption (see #103339)."
            ) from exc
        # The mutex is held from the registry check through registration, so
        # no sibling thread can interleave a competing take of this file.
        # The holder record goes down BEFORE registration: a sibling probe
        # that lands mid-announce then reads a complete role instead of a
        # transient unknown (both fail closed; the former avoids spurious
        # refusals).
        held = _try_flock_nb(handle)
        if held is not True:
            try:
                handle.close()
            except OSError:
                pass
            raise _refusal(db_path, _describe_holder(lock_path))
        _write_gate_record(handle, role)
        hold = _GateHold(handle, role)
        hold.structural_owner = owner
        hold.owners.add(owner)
        _held[key] = hold
    # Atomicity: the global flock is held from here on, so verify foreign
    # writer presence BEFORE returning. A writer announcing later sees the
    # held global lock and refuses; a writer already present is seen here.
    # Either order closes fully — only an actually-quiet database is ever
    # returned held. Own presences count too (include_self): a live local
    # SessionDB must refuse a structural take from an unrelated owner in the
    # same process. (Fellow repairers never announce per-pid presence, so
    # they cannot trip this; they queue via the repair lock.)
    writers = _live_writer_presences(db_path, include_self=True)
    if writers:
        with _held_lock:
            _held.pop(key, None)
        _unlock_handle(handle)
        try:
            handle.close()
        except OSError:
            pass
        raise _refusal(db_path, "; ".join(writers))
    return


def structural_lock_held_by_other(db_path: Path) -> Optional[str]:
    """Global-structural-lock-only probe for open-time mutation sites.

    Unlike :func:`writer_gate_holder` this ignores mere writer presence:
    ordinary writers coexisting must NOT block opens — only a live surgery
    (repair/checkpoint/VACUUM-class holder) refuses constructor-time DDL
    and generation writes. Returns a holder description or None. Never
    raises (unprovable reads as held).
    """
    foreign, _role, description = _probe_global(db_path)
    return description if foreign else None


def refuse_if_structural_op_holds(db_path: Path, what: str) -> None:
    """Fail-closed entry guard for open-time mutation sites (schema DDL,
    generation stamp): constructor work must not interleave a live surgery.

    Only a held global structural lock refuses — coexisting writers never
    do. Raises :class:`StateDbWriterHeldError` with an actionable message
    (worded to avoid the write path's locked/busy retry match, so opens
    fail fast instead of waiting out a patience window).
    """
    holder = structural_lock_held_by_other(db_path)
    if holder is not None:
        raise StateDbWriterHeldError(
            f"state.db structural gate: {holder}; refusing {what} on {db_path} while a "
            "structural operation owns it (see #103339). Stop that process's gateway "
            "(`hermes gateway stop`) and retry once it finishes."
        )


def _announce_presence(db_path: Path, owner=None) -> None:
    """Record this process as a live writer (fail-closed).

    Warns and raises :class:`StateDbWriterHeldError` when presence itself
    cannot be recorded (exotic filesystem): proceeding without an authority
    record would let a later structural acquirer see an apparently quiet
    database. The caller must not run its mutation in that case.
    """
    lock_path = _presence_path(db_path, os.getpid())
    key = str(lock_path)
    # Mutex Ward (see _live_writer_presences): the whole check + open + flock
    # + register sequence runs under one hold. Without it, two sibling
    # threads announcing at once contend on the same per-pid file and the
    # loser would unlink the winner's live file as "stale".
    with _held_lock:
        if key in _held:
            hold = _held[key]
            # A live announce re-pins the presence: a deferred cleanup
            # flagged while the hold was orphaned is no longer owed.
            hold.deferred_release = False
            # Lifetime invariant: every announcing owner registers, so the
            # presence drops only when the LAST in-process owner closes.
            # Anonymous announces take a lease (see _GateHold.anon_leases)
            # so a refused announce can roll back exactly its own share.
            if owner is not None:
                hold.owners.add(owner)
            else:
                hold.anon_leases += 1
            return
        try:
            handle = _open_gate_file(lock_path)
        except OSError as exc:
            logger.warning(
                "state.db writer gate: cannot record presence in %s (%s); refusing the mutation.",
                lock_path, exc)
            raise StateDbGateSetupError(
                f"state.db writer gate: cannot establish presence in {lock_path} ({exc}); refusing "
                f"mutation on {db_path} so a later structural operation cannot miss this writer (see #103339)."
            ) from exc
        held = _try_flock_nb(handle)
        if held is not True:
            # None (unexpected lock error) is a permanent setup failure, not
            # contention — never classify it as a live holder downstream.
            flock_error = held is None
            # Same-pid stale file (crashed previous owner with a recycled pid):
            # reclaim once under the same hold. The unlink+create runs under
            # the presence-path mutation mutex, so a concurrent reaper cannot
            # slip a replacement between our unlink and recreate (or vice
            # versa); contention fails closed via the refusal below.
            with _presence_mutation_serialized(db_path) as serialized:
                if serialized is None:
                    try:
                        handle.close()
                    except OSError:
                        pass
                    raise StateDbGateSetupError(
                        f"state.db writer gate: cannot open or lock the presence-path mutation "
                        f"mutex for {lock_path}; refusing mutation on {db_path} — a permanent "
                        f"setup failure, not holder contention (see #103339)."
                    )
                if serialized is False:
                    try:
                        handle.close()
                    except OSError:
                        pass
                    if flock_error:
                        raise StateDbGateSetupError(
                            f"state.db writer gate: cannot lock presence file {lock_path} "
                            f"(unexpected lock error); refusing mutation on {db_path} (see #103339)."
                        )
                    raise StateDbWriterHeldError(
                        f"state.db writer gate: presence-path mutation for {lock_path} is contended; "
                        f"refusing mutation on {db_path} rather than risk a pathname race (see #103339)."
                    )
                try:
                    handle.close()
                except OSError:
                    pass
                try:
                    lock_path.unlink()
                    handle = _open_gate_file(lock_path)
                    held = _try_flock_nb(handle)
                except OSError as exc:
                    logger.warning(
                        "state.db writer gate: cannot record presence in %s (%s); refusing the mutation.",
                        lock_path, exc)
                    raise StateDbGateSetupError(
                        f"state.db writer gate: cannot re-establish presence in {lock_path} ({exc}); refusing "
                        f"mutation on {db_path} (see #103339)."
                    ) from exc
            if held is not True:
                logger.warning(
                    "state.db writer gate: cannot lock presence file %s; refusing the mutation.", lock_path)
                try:
                    handle.close()
                except OSError:
                    pass
                raise StateDbGateSetupError(
                    f"state.db writer gate: cannot lock presence file {lock_path}; refusing mutation on "
                    f"{db_path} (see #103339)."
                )
        hold = _held.get(key)
        if hold is not None:
            if owner is not None:
                hold.owners.add(owner)
            try:
                handle.close()
            except OSError:
                pass
            return
        # Record before registration (see exclusive take above).
        _write_gate_record(handle, _WRITER_ROLE)
        hold = _GateHold(handle, _WRITER_ROLE, db_path)
        if owner is not None:
            hold.owners.add(owner)
        else:
            hold.anon_leases = 1
        _held[key] = hold


def _rollback_announce(db_path: Path, owner) -> None:
    """Undo one :func:`_announce_presence` share; never raises.

    A refused row announce (structural probe reported a holder AFTER our
    presence was registered) must not leave litter: without rollback the
    refused opener's presence file stays live and blocks later surgery.
    Named owners roll back via :func:`release_writer_gate` (other live
    owners keep the hold). Anonymous announces consume exactly one lease;
    the hold drops only when its last lease and last owner are gone.

    A ``False`` from :func:`release_writer_gate` is never discarded: that
    hold already flagged ``deferred_release`` and armed the module drain
    (below), so the owed unlink survives an owner that never closes again
    instead of blocking structural takes until process exit.
    """
    if owner is not None:
        try:
            release_writer_gate(db_path, owner)
        except Exception:
            pass
        return
    try:
        lock_path = _presence_path(db_path, os.getpid())
    except Exception:
        return
    key = str(lock_path)
    with _held_lock:
        hold = _held.get(key)
        if hold is None:
            return
        if hold.anon_leases > 0:
            hold.anon_leases -= 1
        if len(list(hold.owners)) > 0 or hold.anon_leases > 0:
            return  # other live callers keep the presence
        del _held[key]
        if hold.acquired_pid != os.getpid():
            return  # fork-inherited: drop locally, never touch the fd
        handle = hold.handle
        try:
            db_name = Path(db_path).expanduser().resolve().name
        except OSError:
            db_name = Path(db_path).name
        if _presence_pid(lock_path, db_name) == os.getpid():
            with _presence_mutation_serialized(db_path) as ok:
                if ok:
                    with contextlib.suppress(OSError):
                        lock_path.unlink()
        _unlock_handle(handle)
        try:
            handle.close()
        except OSError:
            pass


def release_writer_gate(db_path: Path, owner) -> bool:
    """Drop *owner*'s share; the flock goes when the last owner closes.

    Never raises. ``owner=None`` is a no-op. Presence files are unlinked only
    when their pid matches the releaser (a forked child can never delete its
    parent's presence); the global structural file is never unlinked. A hold
    acquired by another pid (fork-inherited) is dropped locally without
    unlocking — the owner's description keeps the kernel lock.

    Returns True when nothing of *owner* remains held. Returns False when the
    presence-path cleanup was retained for an independent retry (the mutation
    mutex stayed contended for the whole budget): the hold is still accurate
    (flock held, owner restored, ``deferred_release`` flagged), the module
    drain completes the unlink with no owner dependence, and the caller's
    lifecycle retry (``SessionDB._gate_release_pending``) is a second chance
    rather than the single retry that could be stranded by the owner's death.
    """
    if owner is None:
        return True
    global_path = _writer_lock_path(db_path)
    try:
        db_name = Path(db_path).expanduser().resolve().name
    except OSError:
        db_name = Path(db_path).name
    fully_released = True
    for lock_path in (_presence_path(db_path, os.getpid()), global_path):
        key = str(lock_path)
        # Unlock + close + unlink under the same hold (Mutex Ward): a sibling
        # thread's open + flock can never land between our unlock and close
        # and mistake itself for a foreign holder.
        needs_retry = False
        for _attempt in range(_RELEASE_MUTEX_ATTEMPTS):
            needs_retry = False
            with _held_lock:
                hold = _held.get(key)
                if hold is None:
                    pass  # already gone: done
                else:
                    try:
                        hold.owners.discard(owner)
                    except (KeyError, TypeError):
                        pass
                    if len(list(hold.owners)) > 0 or hold.anon_leases > 0:
                        pass  # other live callers keep the hold: done
                    elif hold.acquired_pid != os.getpid():
                        del _held[key]  # fork-inherited: drop locally, never touch the fd
                    else:
                        handle = hold.handle
                        # Presence files are unlinked only when their pid matches the
                        # releaser (a forked child can never delete its parent's
                        # presence); the global structural file is never unlinked.
                        # The unlink runs under the presence-path mutation mutex
                        # while the flock is still held, so no replacement can win
                        # the pathname in between; the unlock follows.
                        if lock_path != global_path and _presence_pid(lock_path, db_name) == os.getpid():
                            with _presence_mutation_serialized(db_path) as ok:
                                if not ok:
                                    # Mutex contended: re-register this share
                                    # and retry outside the lock. The hold
                                    # (flock still held) stays accurate — the
                                    # caller is told (False return) to keep
                                    # retry lifecycle state — instead of an
                                    # unlocked ghost whose record names a
                                    # live pid forever.
                                    hold.owners.add(owner)
                                    needs_retry = True
                                else:
                                    with contextlib.suppress(OSError):
                                        lock_path.unlink()
                                    del _held[key]
                                    _unlock_handle(handle)
                                    try:
                                        handle.close()
                                    except OSError:
                                        pass
                        else:
                            del _held[key]
                            _unlock_handle(handle)
                            try:
                                handle.close()
                            except OSError:
                                pass
            if not needs_retry:
                break
            time.sleep(_RELEASE_MUTEX_RETRY_S)
        if needs_retry:
            # Budget exhausted with the hold still pinned: flag it and arm the
            # module drain, which finishes the pathname cleanup once no
            # owner/lease is live. Then report False so the caller keeps its
            # lifecycle retry state as a second chance (never fail silent).
            global _deferred_release_epoch
            with _held_lock:
                hold = _held.get(key)
                if hold is not None:
                    hold.deferred_release = True
                    _deferred_release_epoch += 1
            _ensure_deferred_release_drain()
            fully_released = False
    return fully_released


def _ensure_deferred_release_drain() -> None:
    """Start the single deferred-release drain worker (idempotent, never raises)."""
    global _drain_running
    with _drain_lock:
        if _drain_running:
            return
        try:
            t = threading.Thread(
                target=_deferred_release_drain,
                name="hermes-writergate-drain",
                daemon=True,
            )
            t.start()
        except RuntimeError:
            return  # interpreter teardown: process exit reaps the flock anyway
        _drain_running = True


def _deferred_release_drain() -> None:
    """Finish shadowed presence-pathname cleanups once no owner/lease is live.

    Works from the hold record (flock still held, ``owners`` /
    ``anon_leases`` as the liveness tally), never the dead owner's lifecycle
    state, so teardown cannot strand a live-pid presence when the owner died
    before another ``close()``. Exits when nothing is flagged; re-arms only
    when a NEW flag was raised after this run started (epoch), never into an
    endless retry of a permanently failed setup.
    """
    global _drain_running
    with _held_lock:
        seen_epoch = _deferred_release_epoch
    deadline = time.monotonic() + _DEFERRED_RELEASE_DRAIN_MAX_S
    try:
        while time.monotonic() < deadline:
            with _held_lock:
                items = [
                    (key, hold) for key, hold in _held.items()
                    if getattr(hold, "deferred_release", False)
                ]
            if not items:
                return  # nothing left to drain: single-shot worker
            for key, hold in items:
                _drain_one_deferred_release(key, hold)
            time.sleep(_DEFERRED_RELEASE_DRAIN_POLL_S)
    finally:
        with _drain_lock:
            _drain_running = False
        # Re-arm only for flags raised after this run began: the setter arms
        # the drain itself, so this only closes the scan-tail window.
        with _held_lock:
            new_flags = _deferred_release_epoch != seen_epoch
        if new_flags:
            _ensure_deferred_release_drain()


def _drain_one_deferred_release(key: str, hold: "_GateHold") -> None:
    """Attempt the deferred pathname cleanup for one hold (best effort, never raises).

    The whole check + unlink runs under ``_held_lock`` with the mutation
    mutex nested inside (the same ordering ``release_writer_gate`` uses), so
    a live announce that re-pins the presence between our scan and the unlink
    is impossible: announce takes the same outer lock.
    """
    try:
        with _held_lock:
            if hold.deferred_release is not True:
                return
            # Stale-capture guard: the scan captured (key, hold) before the
            # lock; a release that won the race since then popped the hold
            # (flag still True on the orphaned object) and a fresh announce
            # may have installed a NEW presence at the same path. Unlinking
            # then would erase a live writer's file. Only the CURRENT owner
            # of the key may be drained.
            if _held.get(key) is not hold:
                return
            if len(list(hold.owners)) > 0 or hold.anon_leases > 0:
                return  # re-pinned live; _announce_presence cleared the flag
            if hold.acquired_pid != os.getpid():
                del _held[key]  # fork-inherited: never touch the parent's fd
                return
            lock_path = Path(key)
            db_path = getattr(hold, "db_path", None)
            if db_path is None:
                # Cannot happen in-process: every presence hold registers its
                # db_path at announce time (see _GateHold.db_path). Dropping
                # the ghost keeps the invariant honest without dead code.
                del _held[key]
                return
            handle = hold.handle
            with _presence_mutation_serialized(db_path) as ok:
                if not ok:
                    return  # contended or setup-failed: retry on the next poll
                with contextlib.suppress(OSError):
                    lock_path.unlink()
            _held.pop(key, None)
        _unlock_handle(handle)
        try:
            handle.close()
        except OSError:
            pass
    except Exception:
        logger.debug("deferred writer-gate drain failed for %s", key, exc_info=True)


def writer_gate_holder(db_path: Path) -> Optional[str]:
    """Non-acquiring probe: None when no foreign surgery or writer presence
    is live (or only ours is); otherwise a human-readable description.

    Used by repair/doctor/checkpoint paths that must refuse without taking
    the gate.
    """
    foreign, _role, description = _probe_global(db_path)
    if foreign:
        return description
    writers = _live_writer_presences(db_path)
    if writers:
        return "; ".join(writers)
    return None


def writer_gate_holder_role(db_path: Path) -> Optional[str]:
    """Non-acquiring probe: None when this process may do structural work;
    otherwise the live holder's role (``writer`` / ``repair`` / ``unknown``).

    Repair uses this to tell a fellow repairer (serialize via the repair
    lock — queuing is correct) from a live writer (refuse).
    """
    foreign, role, _description = _probe_global(db_path, settle_role=True)
    if foreign:
        if role == _UNKNOWN_ROLE and not _live_writer_presences(db_path):
            # Contended global lock with no readable record and no writer
            # presence: fellow repairer mid-record-write (settle already
            # retried) — report repair so we queue rather than refuse.
            return _REPAIR_ROLE
        return role
    if _live_writer_presences(db_path):
        return _WRITER_ROLE
    return None
