"""Audit log for dashboard-auth events.

Profile-aware location: ``$HERMES_HOME/logs/dashboard-auth.log``.
Format: one JSON object per line. Token-like kwargs are dropped before
serialisation so we never leak refresh tokens or JWTs to disk.
"""
from __future__ import annotations

import json
import pytest

from hermes_cli.dashboard_auth.audit import audit_log, AuditEvent


@pytest.fixture
def profile_home(tmp_path, monkeypatch):
    """Redirect $HERMES_HOME and ~ to a tmp dir for the duration of the test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Some code paths fall back to Path.home() — patch that too.
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    return home


def test_audit_writes_jsonlines(profile_home):
    audit_log(AuditEvent.LOGIN_START, provider="nous", ip="1.2.3.4")
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", user_id="u1",
        email="a@b.com", ip="1.2.3.4",
    )

    path = profile_home / "logs" / "dashboard-auth.log"
    assert path.exists(), f"audit log not created at {path}"
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2

    second = json.loads(lines[1])
    assert second["event"] == "login_success"
    assert second["provider"] == "nous"
    assert second["user_id"] == "u1"
    assert second["email"] == "a@b.com"
    assert "ts" in second  # ISO-8601 timestamp


def test_audit_redacts_token_like_fields(profile_home):
    audit_log(
        AuditEvent.LOGIN_SUCCESS,
        provider="nous", access_token="should-not-appear",
        refresh_token="also-not", code="not-this", state="nope",
    )
    raw = (profile_home / "logs" / "dashboard-auth.log").read_text()
    for forbidden in ("should-not-appear", "also-not", "not-this", "nope"):
        assert forbidden not in raw, f"token-like value leaked into audit log: {forbidden}"


def test_audit_honors_max_size_and_rotates(profile_home, monkeypatch):
    """Writing past the configured max bytes rotates instead of growing
    without bound (upstream #98338)."""
    path = profile_home / "logs" / "dashboard-auth.log"

    import hermes_cli.dashboard_auth.audit as audit_mod

    # Force a tiny cap so a few records trigger the RotatingFileHandler path.
    monkeypatch.setattr(audit_mod, "_rotation_policy", lambda: (500, 2))

    for i in range(60):
        audit_log(AuditEvent.LOGIN_SUCCESS, provider="nous", n=i)

    assert path.exists()
    # Rotation happened → a backup file exists (dashboard-auth.log.1).
    assert path.with_name("dashboard-auth.log.1").exists(), (
        "expected a rotated backup after exceeding maxBytes"
    )
    # Each line must still be one JSON object.
    lines = path.read_text().splitlines()
    assert lines, "no audit records written"
    assert all(l.startswith("{") and l.endswith("}") for l in lines)
    # The live file must not have grown without bound (60 records can't
    # balloon past a sane bound once rotation is active).
    assert path.stat().st_size < 50 * 1024, "live audit log grew unboundedly"




