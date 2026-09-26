"""Windows cron scripts must run on the ABI they were built for.

``cron/scheduler_script.py::_windows_cron_python_invocation`` answers "which interpreter and
which site-packages?" for a cron script child. On a PM-managed Windows install it read the
dependency tree through ``pm.environments.selected_venv``, which answers with the leftover
pre-PM ``<root>/venv`` when no generation is recorded (it falls back to ``base_venv``). That
tree was built for whichever interpreter created it, so overlaying it on the managed store
Python 3.14 loads a cp311 ``pydantic_core`` and every script dies with
``ModuleNotFoundError: No module named 'pydantic_core._pydantic_core'`` — the same class as
the gateway crash in #122183 / #123650, in the cron sibling no PR covers.

``committed_venv`` is the resolver that never answers with the in-tree venv. With none
committed there is no tree this install ever provisioned, so the call keeps the child's own
interpreter instead of borrowing the stale one.
"""

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "win32",
    reason="_windows_cron_python_invocation is the Windows cron spawn path (#122183/#123650)",
)


def _write_venv(venv: Path, base: Path, *, version: str | None = None, uv: bool = True) -> Path:
    """A fake Windows venv with the ``Scripts/python.exe`` shim a cron child is handed."""
    site = venv / "Lib" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    (venv / "Scripts").mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    shim = venv / "Scripts" / "python.exe"
    shim.write_text("", encoding="utf-8")
    (base / "python.exe").write_text("", encoding="utf-8")
    lines = [f"home = {base}"]
    if uv:
        lines.append("uv = 0.11.14")
    if version:
        lines.append(f"version_info = {version}")
    (venv / "pyvenv.cfg").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return venv


def _site_packages_in(env_overlay: dict) -> Path | None:
    for entry in env_overlay.get("PYTHONPATH", "").split(os.pathsep):
        if Path(entry).name == "site-packages":
            return Path(entry)
    return None


def test_managed_install_never_overlays_the_stale_in_tree_venv(tmp_path, monkeypatch):
    """A managed install with no committed generation keeps the child's own interpreter.

    ``selected_venv`` answers the leftover pre-PM ``<root>/venv`` here, which is built for a
    different interpreter: its compiled extensions cannot load on the store Python.
    """
    from cron import scheduler_script as sched_script

    repo = tmp_path / "repo"
    stale = _write_venv(repo / "venv", tmp_path / "base")
    store = tmp_path / "store" / "python.exe"
    store.parent.mkdir(parents=True)
    store.write_text("", encoding="utf-8")
    child = _write_venv(tmp_path / "child", tmp_path / "childbase").parent

    monkeypatch.setattr(sched_script, "_read_windows_pyvenv_cfg", lambda _dir: {})
    monkeypatch.setattr(
        "hermes_cli._launchers.resolve_store_python", lambda _root: store
    )
    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: None)
    monkeypatch.setattr("pm.environments.selected_venv", lambda _root: stale)

    interpreter, env_overlay = sched_script._windows_cron_python_invocation(
        str(child / "Scripts" / "python.exe")
    )

    overlay = _site_packages_in(env_overlay)
    assert overlay is None or not overlay.is_relative_to(stale), (
        "a stale in-tree venv must never be overlaid on the managed store Python"
    )


def test_managed_install_uses_the_committed_generation(tmp_path, monkeypatch):
    """With a generation committed, the child's site-packages are that generation's."""
    from cron import scheduler_script as sched_script

    repo = tmp_path / "repo"
    stale = _write_venv(repo / "venv", tmp_path / "base")
    committed = _write_venv(
        tmp_path / "installs" / "gen" / "venv", tmp_path / "genbase",
        version=f"{sys.version_info[0]}.{sys.version_info[1]}.7",
    )
    store = tmp_path / "store" / "python.exe"
    store.parent.mkdir(parents=True)
    store.write_text("", encoding="utf-8")
    child = _write_venv(tmp_path / "child", tmp_path / "childbase")

    monkeypatch.setattr(sched_script, "_read_windows_pyvenv_cfg", lambda _dir: {})
    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda _root: store)
    monkeypatch.setattr("pm.environments.committed_venv", lambda _root: committed)
    monkeypatch.setattr("pm.environments.selected_venv", lambda _root: stale)

    _, env_overlay = sched_script._windows_cron_python_invocation(
        str(child / "Scripts" / "python.exe")
    )

    overlay = _site_packages_in(env_overlay)
    assert overlay is not None
    assert overlay == committed / "Lib" / "site-packages"


def test_non_managed_install_keeps_its_own_venv_overlay(tmp_path, monkeypatch):
    """With no store Python the uv base-interpreter overlay is unchanged: the legacy
    installs that depend on it must keep working."""
    from cron import scheduler_script as sched_script

    venv = _write_venv(tmp_path / "venv", tmp_path / "base")
    child = _write_venv(venv, tmp_path / "base2")

    monkeypatch.setattr("hermes_cli._launchers.resolve_store_python", lambda _root: None)

    interpreter, env_overlay = sched_script._windows_cron_python_invocation(
        str(child / "Scripts" / "python.exe")
    )

    assert interpreter == str(tmp_path / "base2" / "python.exe")
    overlay = _site_packages_in(env_overlay)
    assert overlay == venv / "Lib" / "site-packages"
    assert env_overlay["VIRTUAL_ENV"] == str(venv)
