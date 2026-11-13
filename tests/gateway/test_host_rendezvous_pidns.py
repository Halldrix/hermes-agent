"""PID-namespace identity on the host-rendezvous record (#123081, ported from #123187).

Sibling of ``test_host_rendezvous.py``; the namespace rows of the matrix live here.
"""

import dataclasses
import json
import os

import pytest

from gateway import host_rendezvous as hr

_LOCAL_NS = "4026532999"
_FOREIGN_NS = "4026532330"


@pytest.fixture
def host_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_LOCK_DIR", str(tmp_path / "locks"))
    yield tmp_path
    hr.release_host_lock(hr.ROLE_GATEWAY)
    hr.release_host_lock(hr.ROLE_SERVE)


def _force_local_namespace(monkeypatch, ns):
    import hermes_state_pidns

    monkeypatch.setattr(hermes_state_pidns, "_LOCAL_PID_NS", ns)


def _local_pid_namespace(ns):
    import hermes_state_pidns

    return hermes_state_pidns.LocalPidNamespace(ns, True)


def _published(host_dir, monkeypatch):
    _force_local_namespace(monkeypatch, _local_pid_namespace(_LOCAL_NS))
    record = hr.publish_record(hr.ROLE_GATEWAY, profiles=("default",), home=str(host_dir))
    assert record is not None
    return record


def _foreign_record_on_disk(record):
    foreign = dataclasses.replace(record, pidns=_FOREIGN_NS)
    assert foreign.pid == os.getpid()
    hr.ensure_host_state_dir()
    hr.record_path(hr.ROLE_GATEWAY).write_text(json.dumps(foreign.to_json()), encoding="utf-8")
    hr.token_path(hr.ROLE_GATEWAY).write_text("token", encoding="utf-8")
    return foreign


def test_published_host_record_stamps_the_publishing_namespace(host_dir, monkeypatch):
    record = _published(host_dir, monkeypatch)
    assert record.pidns == _LOCAL_NS
    roundtrip = hr.HostRecord.from_json(json.loads(json.dumps(record.to_json())))
    assert roundtrip is not None
    assert roundtrip.pidns == _LOCAL_NS


def test_foreign_namespace_host_record_is_never_an_owner(host_dir, monkeypatch):
    """A record stamped in another namespace names an unrelated process here (#123081)."""
    record = _published(host_dir, monkeypatch)
    foreign = dataclasses.replace(record, pidns=_FOREIGN_NS)
    assert hr.record_is_stale(foreign) is True
    assert hr.liveness_is_proven(foreign) is False


def test_unstamped_host_record_keeps_its_pre_namespace_semantics(host_dir, monkeypatch):
    """The rollout boundary: a record from a pre-stamp build stays exactly as legible as before."""
    record = _published(host_dir, monkeypatch)
    legacy = dataclasses.replace(record, pidns=None)
    assert hr.record_is_stale(legacy) is False


def test_unresolvable_local_namespace_keeps_the_legacy_rows(host_dir, monkeypatch):
    """Local lookup failed, supported platform: unstamped keeps probing, a stamp does not (#123081)."""
    import hermes_state_pidns

    _force_local_namespace(monkeypatch, None)  # a failed read is not cached; stay empty
    monkeypatch.setattr(
        hermes_state_pidns, "_resolve_local_pid_namespace",
        lambda: hermes_state_pidns.LocalPidNamespace(None, True),
    )
    record = hr.publish_record(hr.ROLE_GATEWAY, profiles=("default",), home=str(host_dir))
    assert record is not None
    assert record.pidns is None  # the writer never asserts a namespace it cannot prove
    assert hr.record_is_stale(record) is False
    stamped = dataclasses.replace(record, pidns=_FOREIGN_NS)
    assert hr.record_is_stale(stamped) is True


def test_a_missing_or_non_digit_pidns_stays_out_of_the_legacy_row(host_dir, monkeypatch):
    """A present-but-invalid stamp reads foreign; the legacy probe row is only for absent stamps."""
    record = _published(host_dir, monkeypatch)
    payload = json.loads(json.dumps(record.to_json()))
    for bogus in ("", "not-a-number"):
        payload["pidns"] = bogus
        parsed = hr.HostRecord.from_json(payload)
        assert parsed is not None
        assert hr.record_is_stale(parsed) is True


def test_a_numeric_pidns_is_judged_by_the_predicate(host_dir, monkeypatch):
    """An int equal to our stamp stays readable: coerced to digits, which the predicate judges."""
    record = _published(host_dir, monkeypatch)
    payload = json.loads(json.dumps(record.to_json()))
    payload["pidns"] = int(_LOCAL_NS)
    parsed = hr.HostRecord.from_json(payload)
    assert parsed is not None
    assert parsed.pidns == _LOCAL_NS
    assert hr.record_is_stale(parsed) is False


def test_retraction_of_a_foreign_record_follows_the_host_lock(host_dir, monkeypatch):
    """A foreign record is never retracted on pid evidence; winning the flock is the only proof of death."""
    record = _published(host_dir, monkeypatch)
    _foreign_record_on_disk(record)

    assert hr.claim_host_lock(hr.ROLE_GATEWAY)[0] is hr.HostLockOutcome.ACQUIRED
    try:
        assert hr.discard_dead_record(hr.ROLE_GATEWAY) is False
        assert hr.record_path(hr.ROLE_GATEWAY).exists(), "a held-lock foreign record was retracted"
        assert hr.token_path(hr.ROLE_GATEWAY).exists()
    finally:
        hr.release_host_lock(hr.ROLE_GATEWAY)

    real_hold = hr._hold_lock_briefly
    monkeypatch.setattr(hr, "_hold_lock_briefly", lambda _role: None)  # unprobeable: still refuse
    assert hr.discard_dead_record(hr.ROLE_GATEWAY) is False
    assert hr.record_path(hr.ROLE_GATEWAY).exists()

    monkeypatch.setattr(hr, "_hold_lock_briefly", real_hold)
    assert hr.discard_dead_record(hr.ROLE_GATEWAY) is True  # free: owner is gone
    assert not hr.record_path(hr.ROLE_GATEWAY).exists(), "a dead foreign record was left behind"
    assert not hr.token_path(hr.ROLE_GATEWAY).exists()


def test_exit_cleanup_never_clears_a_foreign_record(host_dir, monkeypatch):
    """``clear_record`` on the exit path refuses: at exit there is no flock fact to qualify it."""
    record = _published(host_dir, monkeypatch)
    _foreign_record_on_disk(record)

    hr.clear_record(hr.ROLE_GATEWAY)
    assert hr.record_path(hr.ROLE_GATEWAY).exists(), "a foreign record was cleared on exit"
    assert hr.token_path(hr.ROLE_GATEWAY).exists(), "a foreign token was cleared on exit"
