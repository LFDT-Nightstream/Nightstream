#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
python_tool="${repo_root}/scripts/gpuprof/gpuprof.py"

usage() {
  cat <<'EOF'
Usage:
  scripts/gpuprof.sh [--all|all] [gate] [options]
  scripts/gpuprof.sh quick [gate] [options]
  scripts/gpuprof.sh nsys [gate] [options]
  scripts/gpuprof.sh ncu [gate] [--top N] [options]
  scripts/gpuprof.sh sanitize [gate] [--tool TOOL]...
  scripts/gpuprof.sh cpu [gate] [options]
  scripts/gpuprof.sh metadata [options]
  scripts/gpuprof.sh diff OLD_JSON NEW_JSON

Default:
  scripts/gpuprof.sh
    Same as: scripts/gpuprof.sh --all e2e_bench

Commands:
  --all, all  Build parity with cuda,perf-timers and collect the full bundle:
              gpuprof.json, trace.json, gpuprof.sqlite, nsys report, ncu
              reports, sanitizer logs, metadata.json, stdout.txt, stderr.txt.
  quick       Build and collect gpuprof.json + trace.json + nsys artifacts only.
  nsys        Same collection scope as quick; explicit name for timeline work.
  ncu         Build, run nsys, then run Nsight Compute on the top kernels.
  sanitize    Build, run nsys, then run compute-sanitizer tools.
  cpu         Build, run nsys, then rerun the gate under perf for CPU stacks.
  metadata    Write CUDA/Rust/GPU/tool metadata only.
  diff        Compare two gpuprof.json files.

Options:
  --artifacts DIR       Use a specific artifact directory.
  --top N              Number of top kernels for ncu/all. Default: 3.
  --ncu-launch-count N Matching launches to profile per NCU kernel. Default: 1.
  --cpu-perf-freq N    perf sample frequency. Default: 99.
  --tool TOOL          Sanitizer tool for sanitize mode: memcheck, racecheck,
                       initcheck, synccheck, or all. Can be repeated.
  --                  Pass remaining args through to gpuprof.py run.

Examples:
  scripts/gpuprof.sh --all e2e_bench
  scripts/gpuprof.sh quick e2e_bench
  scripts/gpuprof.sh ncu e2e_bench --top 5
  scripts/gpuprof.sh sanitize e2e_bench --tool memcheck
  scripts/gpuprof.sh cpu e2e_bench
  scripts/gpuprof.sh diff benchmark-results/a/gpuprof.json benchmark-results/b/gpuprof.json
EOF
}

build_parity() {
  (
    cd "${repo_root}/crates/neo-prover-cuda"
    cargo +nightly-2026-04-03 oxide build --features cuda,perf-timers
  )
}

print_bundle() {
  local artifact_dir="$1"
  printf '\nartifact bundle: %s\n' "${artifact_dir}"
  printf '  gpuprof.json:  %s\n' "${artifact_dir}/gpuprof.json"
  printf '  trace.json:    %s\n' "${artifact_dir}/trace.json"
  printf '  metadata.json: %s\n' "${artifact_dir}/metadata.json"
  printf '  nsys sqlite:   %s\n' "${artifact_dir}/gpuprof.sqlite"
  printf '  nsys report:   %s\n' "${artifact_dir}/gpuprof.nsys-rep"
  printf '  stdout/stderr: %s / %s\n' "${artifact_dir}/stdout.txt" "${artifact_dir}/stderr.txt"
  if [[ -d "${artifact_dir}/ncu" ]]; then
    printf '  ncu reports:   %s\n' "${artifact_dir}/ncu"
  else
    printf '  ncu reports:   not requested\n'
  fi
  if [[ -d "${artifact_dir}/sanitizer" ]]; then
    printf '  sanitizer:     %s\n' "${artifact_dir}/sanitizer"
  else
    printf '  sanitizer:     not requested\n'
  fi
  if [[ -d "${artifact_dir}/cpu" ]]; then
    printf '  cpu profile:   %s\n' "${artifact_dir}/cpu"
  else
    printf '  cpu profile:   not requested\n'
  fi
}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

mode="all"
if [[ $# -gt 0 ]]; then
  case "$1" in
    --all)
      mode="all"
      shift
      ;;
    all|quick|nsys|ncu|sanitize|cpu|metadata|diff)
      mode="$1"
      shift
      ;;
  esac
fi

if [[ "${mode}" == "diff" ]]; then
  if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
  fi
  cd "${repo_root}"
  exec python3 "${python_tool}" diff "$1" "$2"
fi

gate="e2e_bench"
if [[ "${mode}" != "metadata" && $# -gt 0 && "$1" != --* ]]; then
  gate="$1"
  shift
fi

top="3"
ncu_launch_count="1"
cpu_perf_freq="99"
artifact_dir=""
sanitize_tools=()
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --artifacts)
      artifact_dir="$2"
      shift 2
      ;;
    --top)
      top="$2"
      shift 2
      ;;
    --ncu-launch-count)
      ncu_launch_count="$2"
      shift 2
      ;;
    --cpu-perf-freq)
      cpu_perf_freq="$2"
      shift 2
      ;;
    --tool)
      sanitize_tools+=("$2")
      shift 2
      ;;
    --)
      shift
      extra_args+=("$@")
      break
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -z "${artifact_dir}" ]]; then
  if [[ "${mode}" == "metadata" ]]; then
    artifact_dir="${repo_root}/benchmark-results/gpuprof-metadata-${stamp}"
  else
    artifact_dir="${repo_root}/benchmark-results/gpuprof-${gate}-${mode}-${stamp}"
  fi
fi
mkdir -p "${artifact_dir}"

if [[ "${mode}" == "metadata" ]]; then
  cd "${repo_root}"
  python3 "${python_tool}" metadata --json "${artifact_dir}/metadata.json"
  printf 'metadata.json: %s\n' "${artifact_dir}/metadata.json"
  exit 0
fi

build_parity

run_args=(
  run "${gate}"
  --artifacts "${artifact_dir}"
  --json "${artifact_dir}/gpuprof.json"
  --trace-json "${artifact_dir}/trace.json"
  --metadata-json "${artifact_dir}/metadata.json"
  --keep-rep
)

case "${mode}" in
  all)
    run_args+=(--ncu-top "${top}" --ncu-launch-count "${ncu_launch_count}" --sanitize all)
    run_args+=(--cpu-profile perf --cpu-perf-freq "${cpu_perf_freq}")
    ;;
  ncu)
    run_args+=(--ncu-top "${top}" --ncu-launch-count "${ncu_launch_count}")
    ;;
  sanitize)
    if [[ ${#sanitize_tools[@]} -eq 0 ]]; then
      sanitize_tools=(all)
    fi
    for tool in "${sanitize_tools[@]}"; do
      run_args+=(--sanitize "${tool}")
    done
    ;;
  cpu)
    run_args+=(--cpu-profile perf --cpu-perf-freq "${cpu_perf_freq}")
    ;;
  quick|nsys)
    ;;
  *)
    printf 'unknown mode: %s\n\n' "${mode}" >&2
    usage >&2
    exit 2
    ;;
esac

cd "${repo_root}"
python3 "${python_tool}" "${run_args[@]}" "${extra_args[@]}"
print_bundle "${artifact_dir}"
