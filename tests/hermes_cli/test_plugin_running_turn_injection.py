"""Tests for plugin machine notices delivered into the running turn
(PluginContext.deliver_to_running_turn)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import hermes_yaml as yaml

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _context(name: str = "notify-plugin") -> tuple[PluginContext, PluginManager]:
    manager = PluginManager()
    manifest = PluginManifest(name=name, key=name, source="user")
    return PluginContext(manifest, manager), manager


def _write_plugin_config(tmp_path, monkeypatch, entry: dict) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"entries": {"notify-plugin": entry}}})
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))


class _Agent:
    def __init__(self):
        self.noticed = []

    def notice(self, content):
        self.noticed.append(content)
        return True


def test_denied_by_default_until_the_plugin_opts_in(tmp_path, monkeypatch):
    """No config entry = no capability: delivering into someone's live turn is their say-so."""
    context, manager = _context()
    manager._cli_ref = SimpleNamespace(_agent=_Agent())
    assert context.deliver_to_running_turn("go faster") is False


def test_cli_running_turn_receives_the_notice(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {"allow_running_turn_injection": True})
    context, manager = _context()
    agent = _Agent()
    manager._cli_ref = SimpleNamespace(_agent=agent)

    assert context.deliver_to_running_turn("build finished: 3 tests, 0 failures") is True
    assert agent.noticed == ["build finished: 3 tests, 0 failures"]


def test_the_running_turn_target_is_honoured(tmp_path, monkeypatch):
    """Keyed delivery goes through the published gateway injector, not the local REPL agent."""
    _write_plugin_config(tmp_path, monkeypatch, {"allow_running_turn_injection": True})
    context, manager = _context()
    injector = MagicMock(return_value=True)
    manager.set_running_turn_injector(object(), injector)

    assert context.deliver_to_running_turn("go", session_key="agent:main:local:dm:local-1") is True
    injector.assert_called_once_with(content="go", plugin_id="notify-plugin",
                                     session_key="agent:main:local:dm:local-1")


def test_no_injector_and_no_cli_host_means_nothing_is_delivered(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {"allow_running_turn_injection": True})
    context, _manager = _context()
    assert context.deliver_to_running_turn("go", session_key="agent:main:local:dm:local-1") is False


def test_gateway_path_routes_the_notice_to_the_running_turn(tmp_path, monkeypatch):
    """The gateway's injector delivers to the live agent of the session named by the key."""
    _write_plugin_config(tmp_path, monkeypatch, {"allow_running_turn_injection": True})
    context, manager = _context()

    class _Runner:
        pass

    runner = _Runner()

    def _deliver(*, content, plugin_id, session_key, plugin_home=None):
        runner.delivered = (content, plugin_id, session_key)
        return True

    manager.set_running_turn_injector(object(), _deliver)
    assert context.deliver_to_running_turn("done", session_key="s1") is True
    assert runner.delivered == ("done", "notify-plugin", "s1")


def test_empty_content_is_refused(tmp_path, monkeypatch):
    _write_plugin_config(tmp_path, monkeypatch, {"allow_running_turn_injection": True})
    context, manager = _context()
    manager._cli_ref = SimpleNamespace(_agent=_Agent())
    assert context.deliver_to_running_turn("") is False
    assert context.deliver_to_running_turn("   ") is False
