## What does this PR do?

Escape the parent job object for the post-update Windows gateway spawn by routing through the Task Scheduler when the gateway's Scheduled Task exists.

This is the **second half of the fix for #84185** — the companion to #84212 (which addresses only the silent/false success report). Where #84212 makes the updater honest ("did the process actually survive?"), this PR makes the spawn actually *succeed* in the first place by leaving the parent job object via `schtasks /Run /tn <Hermes_Gateway>`.

## Related Issue

Refs: #84185
Refs: #84212

This PR does not close #84185 on its own — #84212 fixes the silent lie, this PR fixes the silent death.

## Evidence (validated against a real Windows 11 host)

| Spawn method | Flags | Survives parent-job teardown? |
|---|---|---|
| `subprocess.Popen` | +`CREATE_BREAKAWAY_FROM_JOB` | **No** — `CreateProcess` succeeds, no `OSError`, but the child is silently kept inside the job and killed with it |
| `subprocess.Popen` | fallback without breakaway | **No** (control) |
| `schtasks /Run /tn <task>` | via Task Scheduler | **Yes** — the scheduler runs the gateway outside any job containing the updater |

Crucially, `breakaway_error=None` on the test host: no failure signal, no detection path. The current ✓ is doubly invisible.

**Live-host validation by @jrleal10 (see review thread):** `schtasks /Run` works against real At-logon tasks without any re-registration — a dead profile gateway was revived in ~26s on the box that filed #91675. That feedback drove the revision described below.

## Type of Change

- [x] Bug fix (non-breaking change which fixes a bug)

## Changes Made

### Revision 2 (live-host feedback from @jrleal10 + platform/concurrency review)

- **Snapshot before trigger.** `pre_pids = set(find_gateway_pids())` now runs **before** `schtasks /Run` in both `_spawn_via_scheduled_task()` (`hermes_cli/gateway_windows.py`) and the restart-watcher's inline script (`hermes_cli/gateway.py`). Previously the snapshot was taken after `/Run` returned, so a task-spawned python visible before the call returned could land in `pre_pids` and make a successful start report failure. A regression test asserts the ordering.
- **Cold-start-aware success criterion + longer window.** Poll timeout raised 6s → 30s. Additionally: if `/Run` was accepted and **no gateway existed before**, the task route counts as success even when the poll window expires — cold starts (e.g. Telegram connect measured at ~26s on real hosts) can exceed any bounded poll, and falling back to `_spawn_detached()` while the task-spawned process is still importing is exactly the dual-gateway race this PR must not introduce. When a pre-existing gateway was running and no new PID appears, the path retries once via `/End` + `/Run` instead of falling back to the non-escaping direct spawn (see `IgnoreNew` below).
- **No delete+create in the hot path.** The task action points at the stable `.vbs` launcher path, and `/Run` executes whatever content is currently on disk — so refreshing the embedded Python path after an update is a plain file write (`_write_task_script()`), never a task re-registration. This removes two real-world hazards confirmed on live hosts: (a) delete+create can hit UAC `Access denied` where `/Run` alone works; (b) re-registering clobbers user-customized launchers. `_install_scheduled_task()` only runs when the refresh guard fails.
- **Customized launchers are respected.** New helper `_launcher_is_ours()` regenerates scripts only when the on-disk `.vbs` matches our template byte-for-byte or doesn't exist. A user-replaced `.cmd` supervisor wrapper (custom wscript/vbs hidden launchers, external watchdogs) is never touched. New helper `_task_action_matches_expected()` detects tasks whose action no longer points at our `.vbs` (user-edited in `taskschd.msc`) so those definitions are triggered as-is, not overwritten.
- **`IgnoreNew` suppression handled.** With `MultipleInstancesPolicy=IgnoreNew`, `/Run` can return exit 0 while silently starting nothing if a draining gateway keeps the task in "Running" state. That path now ends the in-flight instance and retries once via `/End` + `/Run`, preserving the job-escape guarantee instead of dropping into direct spawn inside the dying parent job.

### Original implementation

- **`hermes_cli/gateway_windows.py`** — helper `_spawn_via_scheduled_task()`: triggers the gateway's own Scheduled Task via `schtasks /Run`, then polls `_wait_for_gateway_ready()` until a **new** gateway process appears. Returns `False` when no task is registered, script refresh/registration fails, the trigger failed, or no new process showed up in time (with the cold-start exception above). Best-effort and Windows-only; safe to call from any post-update spawn point.
- **`hermes_cli/update_cmd.py`** — `_cold_start_windows_gateway_after_update()`: (1) prefer `_spawn_via_scheduled_task()` and print `✓ ... (via Scheduled Task)`; (2) fall back to `_spawn_detached()` only when no task exists or the trigger hard-fails; (3) gate the fallback ✓ behind `_wait_for_gateway_ready()` and print an explicit `✗ ... did not survive` + `hermes gateway start` recovery hint when the spawned process never comes up.
- **`hermes_cli/gateway.py`** — `_spawn_gateway_restart_watcher()`: injects the same Scheduled-Task escape into the inline watcher script so unmapped gateways also respawn through the Task Scheduler when registered. Snapshots pre-existing PIDs **before** triggering, counting only new processes. Falls back to the `subprocess.Popen(breakaway)` → `Popen(no breakaway)` chain only when the task route hard-fails.
- **Tests** — `tests/hermes_cli/test_update_gateway_schtasks_escape.py`: 14 tests covering the full contract, including the revision behaviors: PID snapshot ordering (before trigger), `/Run`-accepted-with-no-prior-gateway counting as success, skip-reinstall when task action and launcher are current, pre-existing-gateway-must-not-pass, schtasks failure paths, and watcher source-level assertions (no delete+create in the hot path, snapshot-before-trigger, cold-start criterion).

## How to Test

```bash
scripts/run_tests.sh tests/hermes_cli/test_update_gateway_schtasks_escape.py -v
scripts/run_tests.sh tests/hermes_cli/test_update_cold_start_gateway_liveness.py -v
# plus the neighboring update-path suites:
scripts/run_tests.sh tests/hermes_cli/test_gateway_windows.py tests/hermes_cli/test_update_venv_health.py tests/hermes_cli/test_update_orphan_backend_reap.py
```

**Real-Windows validation:**
- Job-object escape reproduced end-to-end on a real Windows 11 host via a minimal two-process harness (parent creates a `KILL_ON_JOB_CLOSE` job, spawns the "updater" inside it, closes the handle). `subprocess.Popen`+breakaway → child killed; `schtasks /Run` → child survives.
- Revision 2 verified on a second W11 host: all suite tests pass in the host venv (14/14), plus a live Task Scheduler round-trip (`/Create` throwaway copy → `/Run` → `/End` → `/Delete` cleanup) confirming the guards behave correctly against a registered task whose action does not point at our `.vbs`.
- Independent live-host data from @jrleal10 (two-profile box, At-logon tasks) validated the underlying primitive: see the [review thread](https://github.com/NousResearch/hermes-agent/pull/84409#issuecomment-5376962627).

**Known follow-ups (intentionally out of scope here):**
- Extend the task-route escape to manual `hermes gateway start` / `restart()` (hole 1 of #91675) — kept out to keep this PR reviewable.
- Heartbeat-based phase-2 readiness confirmation (PID-scoped liveness probe) as a stricter success signal.
- Optional opt-out config flag for the task-route start.

## Checklist

- [x] I have read the Contributing Guide (`CONTRIBUTING.md`)
- [x] My commit messages follow Conventional Commits (`fix(update): escape parent job via schtasks for Windows gateway post-update spawn`, `fix(gateway): address live-host feedback on schtasks spawn path`, `refactor(gateway): refine schtasks spawn per platform review`)
- [x] I have searched for existing PRs and confirmed none cover this fix (#84212 is complementary and explicitly out-of-scope on the job-object escape)
- [x] I have tested this via the test suite (14/14 passing locally and on a real Windows 11 host)
- [x] I have added tests for the new behavior
- [x] N/A — No docs update needed (no new user-facing config / behavior change)
- [x] N/A — No changelog entry (bug fix)
