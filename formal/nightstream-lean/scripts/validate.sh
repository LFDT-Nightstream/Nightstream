#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
LEAN_TIMEOUT_SECONDS="${LEAN_TIMEOUT_SECONDS:-900}"
NON_LEAN_TIMEOUT_SECONDS=300
LEAN_MEMORY_CAP_KB=25165824
LEAN_BUILD_TARGET="${LEAN_BUILD_TARGET:-}"

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
    use POSIX qw(WNOHANG);
    my $seconds = shift @ARGV;
    my $memory_cap_kb = 0 + ($ENV{"NIGHTSTREAM_MEMORY_CAP_KB"} // 0);
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
    if ($memory_cap_kb > 0) {
      while (1) {
        my $done = waitpid($pid, WNOHANG);
        last if $done == $pid;

        open my $ps, "-|", "ps", "-axo", "pgid=,rss=" or do {
          kill "TERM", -$pid;
          waitpid $pid, 0;
          die "cannot start RSS monitor: $!\n";
        };
        my $rss_kb = 0;
        while (my $line = <$ps>) {
          if ($line =~ /^\s*(\d+)\s+(\d+)\s*$/ && $1 == $pid) {
            $rss_kb += $2;
          }
        }
        unless (close $ps) {
          kill "TERM", -$pid;
          waitpid $pid, 0;
          die "RSS monitor failed; command terminated fail-closed\n";
        }
        if ($rss_kb > $memory_cap_kb) {
          kill "TERM", -$pid;
          select undef, undef, undef, 2;
          kill "KILL", -$pid;
          waitpid $pid, 0;
          print STDERR "[bounded] command exceeded ${memory_cap_kb} KiB RSS\n";
          exit 125;
        }
        select undef, undef, undef, 0.25;
      }
    } else {
      waitpid $pid, 0;
    }
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

run_lean_capped() (
  export LEAN_NUM_THREADS=1
  export NIGHTSTREAM_MEMORY_CAP_KB="$LEAN_MEMORY_CAP_KB"
  run_capped "$@"
)

run_non_lean_capped() (
  export NIGHTSTREAM_MEMORY_CAP_KB="$LEAN_MEMORY_CAP_KB"
  run_capped "$NON_LEAN_TIMEOUT_SECONDS" "$@"
)

static_checks() {
  "$ROOT/scripts/check-layer-imports.sh"
  "$ROOT/scripts/check-generated-layout.sh"
  bash "$ROOT/scripts/check-proof-ownership-contracts.sh"
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
  if [[ -n "$LEAN_BUILD_TARGET" ]]; then
    (cd "$ROOT" && run_lean_capped "$LEAN_TIMEOUT_SECONDS" lake build \
      "$LEAN_BUILD_TARGET")
  else
    (cd "$ROOT" && run_lean_capped "$LEAN_TIMEOUT_SECONDS" lake build)
  fi
}

axiom_report() {
  (cd "$ROOT" && run_lean_capped "$LEAN_TIMEOUT_SECONDS" lake build tests.Axioms)
}

executable_check() {
  (cd "$ROOT" && run_lean_capped "$LEAN_TIMEOUT_SECONDS" lake exe check)
}

usage() {
  echo "usage: $0 {static|build|axioms|check|all|bounded COMMAND...}" >&2
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
  bounded)
    shift
    if (( $# == 0 )); then
      usage
    fi
    run_non_lean_capped "$@"
    ;;
  *)
    usage
    ;;
esac
