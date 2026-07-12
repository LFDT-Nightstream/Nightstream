#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
LEAN_TIMEOUT_SECONDS="${LEAN_TIMEOUT_SECONDS:-900}"
NON_LEAN_TIMEOUT_SECONDS=300

if [[ ! "$LEAN_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] ||
   (( LEAN_TIMEOUT_SECONDS < 1 || LEAN_TIMEOUT_SECONDS > 900 )); then
  echo "LEAN_TIMEOUT_SECONDS must be an integer between 1 and 900" >&2
  exit 2
fi

run_capped() {
  local seconds="$1"
  local status
  shift
  echo "[bounded] ${seconds}s: $*"
  perl -e '
    use strict;
    use warnings;
    my $seconds = shift @ARGV;
    my $pid = fork();
    die "fork failed: $!\n" unless defined $pid;
    if ($pid == 0) {
      setpgrp(0, 0);
      exec @ARGV;
      die "exec failed: $!\n";
    }
    $SIG{ALRM} = sub {
      kill "TERM", -$pid;
      select undef, undef, undef, 2;
      kill "KILL", -$pid;
      exit 124;
    };
    alarm $seconds;
    waitpid $pid, 0;
    alarm 0;
    my $status = $?;
    exit(($status & 127) ? 128 + ($status & 127) : $status >> 8);
  ' "$seconds" "$@" || {
    status=$?
    if (( status == 124 )); then
      echo "[bounded] command exceeded ${seconds}s" >&2
    fi
    return "$status"
  }
}

static_checks() {
  "$ROOT/scripts/check-layer-imports.sh"
  "$ROOT/scripts/check-generated-layout.sh"
  bash "$REPO_ROOT/scripts/audit_formal_lean.sh"
  run_capped "$NON_LEAN_TIMEOUT_SECONDS" python3 "$ROOT/scripts/check-assurance-data.py"

  local oversized
  oversized="$(
    find "$ROOT/Nightstream" "$ROOT/tests" -type f -name '*.lean' -print0 |
      xargs -0 wc -l |
      awk '$2 != "total" && $1 > 1500 { print $1 " " $2 }'
  )"
  if [[ -n "$oversized" ]]; then
    echo "[static] Lean source exceeds the 1,500-line limit:" >&2
    echo "$oversized" >&2
    exit 1
  fi

  local expected
  expected="$(find "$ROOT" -type f -name '*.expected' -not -path '*/.lake/*' -print -quit)"
  if [[ -n "$expected" ]]; then
    echo "[static] unreviewed generated output remains: ${expected#"$ROOT/"}" >&2
    exit 1
  fi
  echo "[static] structural checks passed"
}

lean_build() {
  (cd "$ROOT" && run_capped "$LEAN_TIMEOUT_SECONDS" lake build)
}

axiom_report() {
  (cd "$ROOT" && run_capped "$LEAN_TIMEOUT_SECONDS" lake build tests.Axioms)
}

executable_check() {
  (cd "$ROOT" && run_capped "$LEAN_TIMEOUT_SECONDS" lake exe check)
}

usage() {
  echo "usage: $0 {static|build|axioms|check|all}" >&2
  exit 2
}

case "${1:-}" in
  static)
    static_checks
    ;;
  build)
    lean_build
    ;;
  axioms)
    axiom_report
    ;;
  check)
    executable_check
    ;;
  all)
    static_checks
    lean_build
    axiom_report
    executable_check
    ;;
  *)
    usage
    ;;
esac
