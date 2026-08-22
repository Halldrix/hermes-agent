import re, sys
src = open(r"C:\Users\theki\AppData\Local\hermes\hermes-agent\hermes_cli\gateway_windows.py", encoding="utf-8").read()
markers = {
    "_task_action_matches_expected def": "def _task_action_matches_expected",
    "_launcher_is_ours (vbs)": "_launcher_is_ours",
    "vbs check in _launcher_is_ours": 'with_suffix(".vbs")',
    "pre_pids before /Run comment": "Snapshot BEFORE triggering",
    "timeout 30s default": "timeout_s: float = 30.0",
    "/End retry": '["/End", "/TN"',
    "run accepted success": "return True\n    # A gateway was already running",
}
ok = True
for name, m in markers.items():
    found = m in src
    ok &= found
    print(("FOUND  " if found else "MISSING") + f"  {name}")
gw = open(r"C:\Users\theki\AppData\Local\hermes\hermes-agent\hermes_cli\gateway.py", encoding="utf-8").read()
w = "_started_via_task = _ok or not _pre_pids" in gw
print(("FOUND  " if w else "MISSING") + "  watcher: _ok or not _pre_pids")
print("ALL_OK" if ok and w else "SOME_MISSING")
