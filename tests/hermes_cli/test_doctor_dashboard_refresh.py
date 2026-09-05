"""Dashboard-auth refresh health check for `hermes doctor` (Refs #98338, Defect 4).

The "Nous Portal auth (logged in)" row reports CLI-credential *presence* — a
3 req/s rejection storm once ran invisible beneath that green tick. The new
check reads the dashboard-auth audit log passively (never triggering a
refresh) and reports the recent REFRESH_FAILURE rate: aggregates only, so no
token material can leak into doctor output.

Run: scripts/run_tests.sh tests/hermes_cli/test_doctor_dashboard_refresh.py
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

import hermes_cli.doctor as doctor_mod
from hermes_cli.dashboard_auth import audit as audit_mod


def _line(event, *, reason="all_providers_rejected_rt", age_s=60):
    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=age_s)).isoformat()
    return json.dumps({"ts": ts, "event": event, "reason": reason}) + "\n"


def _now():
    return dt.datetime.now(dt.timezone.utc)


class TestSummarizeRefreshFailures:
    def test_counts_failures_by_reason_in_window(self):
        lines = [
            _line("refresh_failure", reason="all_providers_rejected_rt"),
            _line("refresh_failure", reason="all_providers_rejected_rt"),
            _line("refresh_failure", reason="rate_limited"),
            _line("refresh_success"),
        ]
        total, by_reason, latest = doctor_mod._summarize_refresh_failures(
            lines, window_s=3600.0, now=_now()
        )
        assert total == 3
        assert by_reason == {"all_providers_rejected_rt": 2, "rate_limited": 1}
        assert latest is not None

    def test_old_failures_outside_window_ignored(self):
        lines = [
            _line("refresh_failure", age_s=7200),
            _line("refresh_failure", age_s=60),
        ]
        total, _, _ = doctor_mod._summarize_refresh_failures(
            lines, window_s=3600.0, now=_now()
        )
        assert total == 1

    def test_malformed_lines_skipped(self):
        lines = [
            "not json\n",
            json.dumps({"event": "refresh_failure"}) + "\n",
            json.dumps({"ts": "garbage", "event": "refresh_failure"}) + "\n",
            json.dumps(["refresh_failure"]) + "\n",
        ]
        total, by_reason, latest = doctor_mod._summarize_refresh_failures(
            lines, window_s=3600.0, now=_now()
        )
        assert (total, by_reason, latest) == (0, {}, None)

    def test_empty_log_is_clean(self):
        assert doctor_mod._summarize_refresh_failures(
            [], window_s=3600.0, now=_now()
        ) == (0, {}, None)


class TestResolveLogPath:
    def test_points_at_dashboard_auth_log(self):
        assert audit_mod.resolve_log_path().name == "dashboard-auth.log"
        assert audit_mod.resolve_log_path().parent.name == "logs"


def _write_log(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


class TestDashboardAuthRefreshCheck:
    def test_storm_fails_with_issue(self, tmp_path, monkeypatch, capsys):
        log = tmp_path / "logs" / "dashboard-auth.log"
        _write_log(log, [_line("refresh_failure") for _ in range(25)])
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert len(finding.issues) == 1
        assert "25" in finding.issues[0]
        assert "✗" in capsys.readouterr().out

    def test_few_failures_warn_without_issue(self, tmp_path, monkeypatch, capsys):
        log = tmp_path / "logs" / "dashboard-auth.log"
        _write_log(log, [_line("refresh_failure") for _ in range(3)])
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert finding.issues == []
        assert "⚠" in capsys.readouterr().out

    def test_clean_log_passes(self, tmp_path, monkeypatch, capsys):
        log = tmp_path / "logs" / "dashboard-auth.log"
        _write_log(log, [_line("refresh_success") for _ in range(5)])
        monkeypatch.setattr(audit_mod, "resolve_log_path", lambda: log)
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert finding.issues == []
        assert "✓" in capsys.readouterr().out

    def test_missing_log_is_info_not_failure(
        self, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            audit_mod, "resolve_log_path", lambda: tmp_path / "nope.log"
        )
        finding = doctor_mod._check_dashboard_auth_refresh(False)
        assert finding.issues == []
        out = capsys.readouterr().out
        assert "✗" not in out and "⚠" not in out
