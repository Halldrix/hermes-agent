#!/usr/bin/env bash
# pr-radar-test.sh -- self-check for scripts/pr-radar.sh.
#
# Stubs `gh` with a state file, flips the state while the radar polls, and
# asserts that every transition lands in the log exactly once -- including the
# ERROR line for a PR that stopped being readable.
#
# Usage: bash pr-radar-test.sh
set -uo pipefail

RADAR="$(dirname "$(readlink -f "$0")")/pr-radar.sh"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin"

cat > "$TMP/bin/gh" <<'STUB'
#!/usr/bin/env bash
n=0; [ -f "$STUB_STATE" ] && n=$(cat "$STUB_STATE")
case "$n" in
  0) printf 'aaa111\t2026-11-10T04:08:21Z\tOPEN\n' ;;
  1) printf 'bbb222\t2026-11-10T05:00:00Z\tOPEN\n' ;;
  2) printf 'bbb222\t2026-11-10T05:00:00Z\tCLOSED\n' ;;
  *) echo "GraphQL: Could not resolve to a PullRequest with the number 79610." >&2; exit 1 ;;
esac
STUB
chmod +x "$TMP/bin/gh"

export STUB_STATE="$TMP/state" PR_RADAR_LOG="$TMP/radar.log"
export PATH="$TMP/bin:$PATH"
echo 0 > "$STUB_STATE"

timeout 14 bash "$RADAR" 79610 owner/repo pr 2 &
radar=$!
sleep 3; echo 1 > "$STUB_STATE"
sleep 3; echo 2 > "$STUB_STATE"
sleep 3; echo 9 > "$STUB_STATE"   # unreadable -> ERROR line
sleep 3
kill "$radar" 2>/dev/null
wait "$radar" 2>/dev/null

lines=$(wc -l < "$PR_RADAR_LOG")
sha=$(cut -f2 "$TMP/radar.log" | sort -u | tr '\n' ' ')
last_state=$(cut -f4 "$TMP/radar.log" | tail -1)

if [ "$lines" = "4" ] && [ "$sha" = "ERROR aaa111 bbb222 " ] \
   && [ "$last_state" != "OPEN" ] && grep -q "Could not resolve" "$TMP/radar.log"; then
  echo "PASS: 4 transitions logged (aaa111 -> bbb222 -> CLOSED -> ERROR)"
  exit 0
fi
echo "FAIL: lines=$lines sha=[$sha] last_state=$last_state"
cat "$TMP/radar.log"
exit 1
