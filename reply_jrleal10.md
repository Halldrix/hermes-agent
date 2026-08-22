@jrleal10 this is outstanding — thank you for testing the actual platform primitive against real At-logon tasks instead of just reading the diff. Your three findings were all correct, and all three are now fixed on the branch (`44bdbeb94c`, plus `563eef5729`):

**1. `pre_pids` snapshotted after `/Run`** — confirmed, exactly the bug you described. The snapshot now happens **before** the trigger in both `_spawn_via_scheduled_task()` and the restart-watcher's inline script. There's also a regression test asserting the ordering (`find` → `run`).

**2. 6s poll too short / dual-spawn race** — agreed, and your timeline makes the case unarguable. Two changes:
- Poll window raised 6s → **30s**.
- More importantly, the success criterion changed: if `/Run` was accepted and **no gateway existed before**, the task route now counts as success even when the poll window expires, instead of falling back to `_spawn_detached()` while the task-spawned process is still importing. That fallback was precisely the dual-gateway race you flagged.

**3. `hermes gateway start` still direct-spawns** — correct, this PR intentionally scopes to the update cold-start path (#84185). Extending the escape hatch to manual `start()`/`restart()` is the follow-up we want to do next; we'd rather keep this PR reviewable than grow its blast radius. We opened it as the natural continuation of #84185 and would welcome your re-test there once it lands.

**Your two landmines drove a bigger change than you asked for:**
- `_spawn_via_scheduled_task()` no longer does delete+create at all in the common path. Since the task action points at the stable `.vbs` path and `/Run` executes whatever content is currently on disk, refreshing the embedded Python path after an update is a plain file write — no re-registration, so no UAC `Access denied` path.
- The refresh only happens when the on-disk `.vbs` matches our generated template byte-for-byte (or doesn't exist). If a user replaced their launcher — like the custom `.cmd` supervisor wrapper on `arthur_tutor` — we never touch it; we just `/Run` whatever task definition exists. A user-edited action in `taskschd.msc` is likewise respected, not overwritten.
- One thing your data helped confirm from platform semantics: `IgnoreNew` can silently suppress `/Run` while a draining gateway keeps the task "Running" (exit 0, nothing started). On that path we now retry once via `/End` + `/Run` instead of dropping into the non-escaping direct spawn.

We verified the branch on our W11 test box (14/14 tests green, live `/Run`→`/End`→cleanup round-trip against a throwaway copy of the registered task), but that box has a different task shape than yours — **your offer to re-test is very welcome**, especially since your box has both profiles with different launcher setups. The specific things worth exercising: (a) cold start via `/Run` after killing a profile gateway, (b) that your `arthur_tutor` wrapper survives an update untouched.
