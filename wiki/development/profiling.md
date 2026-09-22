# Profiling & Perf Snapshots

Use the [`nightstream` benchmark](../../crates/nightstream/README.md#poseidon2-benchmark)
for current CPU and Metal measurements. The performance table in
[AGENTS.md](../../AGENTS.md#perf--constraint-debugging) gives the current commands.
The snapshots below apply only to the retained legacy implementation.

## Perf snapshot tests

All perf snapshots are `--ignored` by default. Pick by question:

| Question | Command |
|---|---|
| Cost of lifecycle fold/IVC append work for an F′ chain | `cargo test -p neo-fold-legacy --release --test perf_fibonacci_bits -- --ignored --nocapture fibonacci_bits_perf_snapshot` |
| R1CS shape the full-history audit circuit hands the decider | `cargo test -p neo-fold-legacy --release --test perf_fibonacci_bits -- --ignored --nocapture fibonacci_decider_r1cs_shape_snapshot` (chain length via `NEO_FOLD_FIB_DECIDER_VALUES`) |
| Committed width/rows of low-norm ring-action encodings | `cargo test -p neo-fold-legacy --release --test perf_ring_action_low_norm_prototype -- --nocapture` |

## Profiling scripts

Usage: `./scripts/<tool> <package> <test_file> <test_function> [--ignored]`

| Tool | Use case | Output |
|---|---|---|
| `profile_for_ai.sh` | Quick CPU profiling, filters system calls | `profile-output.txt` |
| `profile_xctrace.sh` | Full detail + Instruments GUI; add `--template <name>` (Allocations, Leaks, File Activity, System Trace, …) | `profile-xctrace.txt` + `.trace` |
| `profile_memory_deep.sh` | Memory-allocation debugging | text with allocation sites |

Builds use the `profiling` cargo profile (root `Cargo.toml`): release optimization
with thin LTO and full debug info, so symbols survive into the profiler.

## Perf-work conventions

- Perf instrumentation belongs in `engine/` (or the engine crates), never in
  `neo-fold-legacy/src/paper/` — the paper layer stays free of counters and probes.
- When comparing encodings or shapes, prefer adding a snapshot test under
  `crates/neo-fold-legacy/tests/perf/` over ad-hoc printouts, so numbers are
  reproducible and reviewable.
