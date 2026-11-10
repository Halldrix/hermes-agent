#!/usr/bin/env bash
# pr-radar.sh -- detect movement on a PR/issue while you review it.
#
# Usage:
#   bash pr-radar.sh --once <number> [repo] [kind]        # print state, exit
#   bash pr-radar.sh <number> [repo] [kind] [interval]    # poll, log changes
#
# kind: pr (default) | issue
#
# Loop mode appends one line per change to $PR_RADAR_LOG (default
# ${TMPDIR:-/tmp}/pr-radar-<number>.log):
#   <utc-timestamp>\t<head-sha>\t<updated-at>\t<state>
#
# Read the tail of that log between phases. A changed head means the code you
# reviewed moved; a changed updated_at means someone commented or CI moved.
# An ERROR line means the thing under watch stopped being readable (deleted,
# renamed, or the token lost access) -- that is a signal, not silence.
set -euo pipefail

once=0
case "${1:-}" in
  --once) once=1; shift ;;
esac

number=${1:?usage: pr-radar.sh [--once] <number> [repo] [kind] [interval]}
repo=${2:-NousResearch/hermes-agent}
kind=${3:-pr}
interval=${4:-60}
log=${PR_RADAR_LOG:-${TMPDIR:-/tmp}/pr-radar-${number}.log}

state() {
  if [ "$kind" = issue ]; then
    gh issue view "$number" --repo "$repo" --json updatedAt,state \
      --jq '[.updatedAt,.state] | @tsv'
  else
    gh pr view "$number" --repo "$repo" --json headRefOid,updatedAt,state \
      --jq '[.headRefOid,.updatedAt,.state] | @tsv'
  fi
}

if [ "$once" = 1 ]; then
  state
  exit 0
fi

last=""
while :; do
  err=$(mktemp)
  if current=$(state 2>"$err"); then
    rm -f "$err"
  else
    reason=$(head -1 "$err" | tr '\t\n' '  ' | cut -c1-160)
    rm -f "$err"
    current=$(printf 'ERROR\t-\t%s' "${reason:-unreadable}")
  fi
  if [ "$current" != "$last" ]; then
    printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$current" | tee -a "$log"
    last=$current
  fi
  sleep "$interval"
done
