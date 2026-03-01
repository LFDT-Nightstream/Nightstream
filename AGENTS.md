# AGENTS.md

## General
- We don't care about backwards compatibility because we are still in development. Keep the code simple and lean.
- Avoid adding new Rust features or ENVs unless it is explicitly approved.
- Never modify this file without explicit approval.
- No single file should ever exceed 1,500 lines of code unless explicitly confirmed by the user.
- Heavily avoid bloat. We want to maintain a compact and lean codebase.
- Proofs must remain compatible with on-chain verification targets. In proof/transcript/public-digest paths, use Poseidon2-only hashing unless explicitly approved otherwise.
- Do not introduce mixed hash families (e.g., Blake3/SHA prehashes) in protocol-binding paths without explicit user approval.

## Design & Architecture
- When evaluating design or architectural decisions, think from first principles: reduce the problem to its irreducible truths—axioms, physical laws, hard constraints—and derive every conclusion strictly from those, rejecting inherited conventions and unstated assumptions.
- Before proposing any architectural change: (1) list every assumption you are making, (2) challenge each by asking "is this a necessity or just a convention?", (3) discard any that fails. Only then derive your answer from what remains.

## Testing
- Never add tests in the same implementation file, always prefer to add them to a file inside tests/ (current or new)
- If you add a test to catch a problem, the test should fail if aims to confirm a problem.
- Always use `FoldingMode::Optimized` in tests. Never use `FoldingMode::PaperExact` unless the user explicitly approves it. PaperExact is an O(2^ell) brute-force reference engine meant only for correctness cross-checking, not general test usage.

## Build & Test Commands
- When running tests use --release eg cargo test --workspace --release
- For extra debugs use debug-logs eg --features paper-exact,debug-logs

## Perf & Constraint Debugging

Perf tests live in `crates/neo-fold/tests/suites/perf/single_addi_metrics_nightstream.rs`. All use `--ignored`.

Full constraint architecture report (main CCS, bus, Route-A claims, openings, timing):
```bash
NS_DEBUG_N=10 cargo test -p neo-fold --release --test perf -- --ignored --nocapture report_track_a_w0_w1_snapshot
```
N: number of riscv instructions + 1 (halt).

Other useful tests (all accept `NS_DEBUG_N`):
- `debug_trace_single_n_mixed_ops` — trace-wiring prove/verify + openings
- `debug_chunked_single_n_mixed_ops` — same in chunked trace mode
- `debug_trace_vs_chunked_single_n_mixed_ops` — side-by-side comparison
- `report_trace_vs_chunked_medians` — 5-run median timing
- `debug_trace_core_rows_per_cycle_equiv` — CCS rows/cycle (no prove, fast; uses `NS_DEBUG_T`)

## Profiling

| Tool | Use Case | Output |
|------|----------|--------|
| `profile_for_ai.sh` | Quick CPU profiling, filters system calls | `profile-output.txt` |
| `profile_xctrace.sh` | Full detail + Instruments GUI (supports `--template`) | `profile-xctrace.txt` + `.trace` |
| `profile_memory_deep.sh` | Memory allocation debugging | Text with allocation sites |

Usage: `./scripts/<tool> <package> <test_file> <test_function> [--ignored]`

For xctrace, add `--template <name>` (Allocations, Leaks, File Activity, System Trace, etc.)

Examples:
```bash
./scripts/profile_for_ai.sh neo-fold test_sha256_single_step test_sha256_preimage_4k --ignored
./scripts/profile_xctrace.sh neo-fold test_sha256_single_step test_sha256_preimage_4k --ignored
./scripts/profile_xctrace.sh neo-fold test_sha256_single_step test_sha256_preimage_4k --ignored --template Allocations
./scripts/profile_memory_deep.sh neo-fold test_sha256_single_step test_sha256_preimage_4k --ignored
```
