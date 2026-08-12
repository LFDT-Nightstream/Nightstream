#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
LEAN_TIMEOUT_SECONDS="${LEAN_TIMEOUT_SECONDS:-1500}"
NON_LEAN_TIMEOUT_SECONDS=300
# Aggregate descendant RSS working ceiling. The user-approved 45 GB remains an
# emergency ceiling; focused validation should not need it.
LEAN_MEMORY_CAP_KB=25165824
LEAN_BUILD_TARGET="${LEAN_BUILD_TARGET:-}"
PROCESS_SNAPSHOT_SOURCE="$ROOT/scripts/process-tree-snapshot.c"
PROCESS_SNAPSHOT_BIN="$ROOT/.lake/build/nightstream-process-tree-snapshot"

if [[ ! "$LEAN_TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] ||
   (( LEAN_TIMEOUT_SECONDS < 1 || LEAN_TIMEOUT_SECONDS > 1500 )); then
  echo "LEAN_TIMEOUT_SECONDS must be an integer between 1 and 1500" >&2
  exit 2
fi

ensure_process_snapshot_helper() {
  if [[ -x "$PROCESS_SNAPSHOT_BIN" &&
        ! "$PROCESS_SNAPSHOT_SOURCE" -nt "$PROCESS_SNAPSHOT_BIN" ]]; then
    return 0
  fi
  local output_dir
  local temporary_bin
  output_dir="$(dirname "$PROCESS_SNAPSHOT_BIN")"
  temporary_bin="${PROCESS_SNAPSHOT_BIN}.tmp.$$"
  mkdir -p "$output_dir"
  if ! cc -std=c11 -O2 -Wall -Wextra -Werror \
      "$PROCESS_SNAPSHOT_SOURCE" -o "$temporary_bin"; then
    rm -f "$temporary_bin"
    echo "[bounded] process snapshot helper compilation failed" >&2
    return 1
  fi
  mv "$temporary_bin" "$PROCESS_SNAPSHOT_BIN"
}

run_capped() {
  local seconds="$1"
  local snapshot_bin=""
  local memory_cap_kb="${NIGHTSTREAM_MEMORY_CAP_KB:-0}"
  local status
  shift
  if [[ "$memory_cap_kb" =~ ^[0-9]+$ ]] &&
     (( memory_cap_kb > 0 )); then
    ensure_process_snapshot_helper
    snapshot_bin="$PROCESS_SNAPSHOT_BIN"
  fi
  echo "[bounded] ${seconds}s: $*"
  perl -e '
    use strict;
    use warnings;
    use POSIX qw(WNOHANG);
    use Time::HiRes qw(time);
    my $seconds = shift @ARGV;
    my $snapshot = shift @ARGV;
    my $memory_cap_kb = 0 + ($ENV{"NIGHTSTREAM_MEMORY_CAP_KB"} // 0);
    my $started_at = time;
    if ($memory_cap_kb > 0) {
      die "process snapshot helper is unavailable\n" unless length $snapshot;
      open my $probe, "-|", $snapshot
        or die "cannot start RSS monitor: $!\n";
      my $saw_process = 0;
      while (my $line = <$probe>) {
        $saw_process = 1
          if $line =~ /^\s*\d+\s+\d+\s+\d+\s+-?\d+\s+\S+\s+[AZ]\s*$/;
      }
      close $probe
        or die "RSS monitor preflight failed\n";
      die "RSS monitor preflight returned no process data\n"
        unless $saw_process;
    }
    my $pid = fork();
    die "fork failed: $!\n" unless defined $pid;
    if ($pid == 0) {
      setpgrp(0, 0) or die "setpgrp failed: $!\n";
      exec @ARGV;
      die "exec failed: $!\n";
    }
    my %known;
    my %live_known = ($pid => 1);
    my $root_status;
    my $terminate_tree = sub {
      my ($signal) = @_;
      kill $signal, -$pid;
      my @live = keys %live_known;
      kill $signal, @live if @live;
    };
    $SIG{ALRM} = sub {
      $terminate_tree->("TERM");
      select undef, undef, undef, 0.25;
      $terminate_tree->("KILL");
      exit 124;
    };
    for my $signal (qw(INT TERM HUP QUIT)) {
      $SIG{$signal} = sub {
        $terminate_tree->("TERM");
        select undef, undef, undef, 0.25;
        $terminate_tree->("KILL");
        exit 128;
      };
    }
    alarm $seconds;
    if ($memory_cap_kb > 0) {
      my $peak_rss_kb = 0;
      while (1) {
        open my $probe, "-|", $snapshot or do {
          $terminate_tree->("KILL");
          waitpid $pid, 0;
          die "cannot start RSS monitor: $!\n";
        };
        my (%parent, %group, %rss, %started, %state);
        while (my $line = <$probe>) {
          if ($line =~ /^\s*(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(\S+)\s+([AZ])\s*$/) {
            $parent{$1} = $2;
            $group{$1} = $3;
            $rss{$1} = $4;
            $started{$1} = $5;
            $state{$1} = $6;
          }
        }
        unless (close $probe) {
          $terminate_tree->("KILL");
          waitpid $pid, 0;
          die "RSS monitor failed; command terminated fail-closed\n";
        }

        # A child tool may create its own process group, so PGID equality is
        # not a complete process-tree test. Retain every previously observed
        # live identity, seed the original group, then close over PPID edges.
        my %tracked;
        for my $candidate (keys %known) {
          $tracked{$candidate} = 1
            if exists $started{$candidate} &&
              $known{$candidate} eq $started{$candidate};
        }
        $tracked{$pid} = 1 if exists $started{$pid};
        for my $candidate (keys %group) {
          $tracked{$candidate} = 1 if $group{$candidate} == $pid;
        }
        my $changed = 1;
        while ($changed) {
          $changed = 0;
          for my $candidate (keys %parent) {
            next if $tracked{$candidate};
            if ($tracked{$parent{$candidate}}) {
              $tracked{$candidate} = 1;
              $changed = 1;
            }
          }
        }
        for my $candidate (keys %tracked) {
          $known{$candidate} = $started{$candidate}
            if exists $started{$candidate};
        }
        %live_known = map { $_ => 1 }
          grep { exists $started{$_} && $known{$_} eq $started{$_} }
          keys %known;
        my $rss_kb = 0;
        for my $candidate (keys %live_known) {
          if (!exists $rss{$candidate} || $rss{$candidate} < 0) {
            $terminate_tree->("KILL");
            waitpid $pid, 0;
            die "RSS unavailable for tracked process; command terminated fail-closed\n";
          }
          $rss_kb += $rss{$candidate};
        }
        $peak_rss_kb = $rss_kb if $rss_kb > $peak_rss_kb;
        if ($rss_kb > $memory_cap_kb) {
          $terminate_tree->("STOP");
          $terminate_tree->("KILL");
          waitpid $pid, 0;
          print STDERR "[bounded] command exceeded ${memory_cap_kb} KiB RSS\n";
          exit 125;
        }

        if (!defined $root_status) {
          my $done = waitpid($pid, WNOHANG);
          $root_status = $? if $done == $pid;
        }
        if (defined $root_status) {
          delete $live_known{$pid};
          if (keys %live_known) {
            $terminate_tree->("STOP");
            $terminate_tree->("KILL");
          }
          last;
        }
        select undef, undef, undef, 0.25;
      }
      printf STDERR "[bounded] elapsed %.2fs; peak descendant RSS: %d KiB\n",
        time - $started_at, $peak_rss_kb;
    } else {
      waitpid $pid, 0;
      $root_status = $?;
      printf STDERR "[bounded] elapsed %.2fs\n", time - $started_at;
    }
    alarm 0;
    my $status = $root_status // $?;
    exit(($status & 127) ? 128 + ($status & 127) : $status >> 8);
  ' "$seconds" "$snapshot_bin" "$@" || {
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

is_lean_command() {
  local command_name="${1##*/}"
  [[ "$command_name" == "lake" || "$command_name" == "lean" ]]
}

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

rust_origin_check() {
  local evidence_dir="$ROOT/.lake/build/rust-origin"
  (cd "$REPO_ROOT" && run_non_lean_capped cargo test -p neo-fold-clean \
    --release --test system_formal_conformance \
    rust_origin_native_verifier_evidence_is_emitted_for_independent_checks -- \
    --exact --nocapture)
  (cd "$REPO_ROOT" && run_non_lean_capped cargo test -p neo-fold-clean \
    --release --test system_relation_artifact \
    rust_origin_relation_artifact_evidence_is_emitted_for_independent_checks -- \
    --exact --nocapture)
  local scope
  for scope in step terminal; do
    (cd "$REPO_ROOT" && run_non_lean_capped python3 \
      protocol-contract/check_rust_evidence.py \
      "$scope" \
      "$REPO_ROOT" \
      "$evidence_dir/native-$scope-evidence.json" \
      "$evidence_dir/native-$scope-corpus.json" \
      "$evidence_dir/native-$scope-replay.lean")
  done
  (cd "$REPO_ROOT" && run_non_lean_capped python3 \
    protocol-contract/check_relation_artifact_evidence.py \
    "$REPO_ROOT" \
    "$evidence_dir/relation-artifact-evidence.json")
  (cd "$ROOT" && run_lean_capped "$LEAN_TIMEOUT_SECONDS" lake build \
    Nightstream.Implementation.Rust.CanonicalConformance.OneSlot \
    Nightstream.Assurance.RelationArtifactBinding)
  for scope in step terminal; do
    (cd "$ROOT" && run_lean_capped "$LEAN_TIMEOUT_SECONDS" lake env lean \
      "$evidence_dir/native-$scope-replay.lean")
  done
  (cd "$ROOT" && run_lean_capped "$LEAN_TIMEOUT_SECONDS" lake env lean \
    "$evidence_dir/relation-artifact-replay.lean")
}

memory_monitor_self_test() {
  local pid_file
  pid_file="$(mktemp "${TMPDIR:-/tmp}/nightstream-monitor.XXXXXX")"
  local recorded_pid=""
  cleanup_recorded_child() {
    if [[ -s "$pid_file" ]]; then
      recorded_pid="$(<"$pid_file")"
      if [[ "$recorded_pid" =~ ^[0-9]+$ ]]; then
        kill -KILL "$recorded_pid" 2>/dev/null || true
      fi
    fi
  }
  local status=0
  (
    export NIGHTSTREAM_MEMORY_CAP_KB=131072
    run_capped 15 perl -e 'select undef, undef, undef, 0.5'
  ) || status=$?
  if (( status != 0 )); then
    cleanup_recorded_child
    rm -f "$pid_file"
    echo "[monitor-self-test] under-cap baseline failed ($status)" >&2
    return 1
  fi
  status=0
  (
    export NIGHTSTREAM_MEMORY_CAP_KB=131072
    run_capped 15 perl -e '
      my $pid_file = shift @ARGV;
      my $child = fork();
      die "fork failed: $!\n" unless defined $child;
      if ($child == 0) {
        setpgrp(0, 0) or die "setpgrp failed: $!\n";
        open my $out, ">", $pid_file or die "open failed: $!\n";
        print {$out} "$$\n";
        close $out or die "close failed: $!\n";
        my $resident = "x" x (256 * 1024 * 1024);
        sleep 10;
        exit(length($resident) == 0);
      }
      select undef, undef, undef, 1;
      exit 0;
    ' "$pid_file"
  ) || status=$?
  if (( status != 125 )); then
    cleanup_recorded_child
    rm -f "$pid_file"
    echo "[monitor-self-test] expected RSS termination (125), got $status" >&2
    return 1
  fi
  local child_pid
  child_pid="$(<"$pid_file")"
  rm -f "$pid_file"
  if [[ ! "$child_pid" =~ ^[0-9]+$ ]]; then
    echo "[monitor-self-test] invalid descendant pid" >&2
    return 1
  fi
  local child_state=""
  local attempt
  for (( attempt = 0; attempt < 20; attempt++ )); do
    child_state="$("$PROCESS_SNAPSHOT_BIN" --state "$child_pid" 2>/dev/null || true)"
    child_state="${child_state//[[:space:]]/}"
    if [[ -z "$child_state" || "$child_state" == Z* ]]; then
      echo "[monitor-self-test] descendant RSS cap passed"
      return 0
    fi
    sleep 0.05
  done
  kill -KILL "$child_pid" 2>/dev/null || true
  if [[ -n "$child_state" && "$child_state" != Z* ]]; then
    echo "[monitor-self-test] escaped descendant remains alive" >&2
    return 1
  fi
  echo "[monitor-self-test] descendant RSS cap passed"
}

usage() {
  echo "usage: $0 {static|build|axioms|check|rust-origin|monitor-self-test|all|bounded COMMAND...}" >&2
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
  rust-origin)
    rust_origin_check
    ;;
  monitor-self-test)
    memory_monitor_self_test
    ;;
  all)
    static_checks
    lean_build
    rust_origin_check
    axiom_report
    executable_check
    ;;
  bounded)
    shift
    if (( $# == 0 )); then
      usage
    fi
    if is_lean_command "$1"; then
      run_lean_capped "$LEAN_TIMEOUT_SECONDS" "$@"
    else
      run_non_lean_capped "$@"
    fi
    ;;
  *)
    usage
    ;;
esac
