# Testing

## Hard policies (from [CLAUDE.md](../../CLAUDE.md))

1. **Always `--release`** — debug builds are unusably slow for proof code.
2. **5-minute cap per test invocation** — if a test needs more, shrink its work or
   mark it `#[ignore]` with a comment.
3. **`FoldingMode::Optimized` only** — `PaperExact` is an O(2^ℓ) reference engine and
   requires explicit approval.
4. **Tests live under `tests/`**, never inline in implementation files.
5. **A test added to catch a problem must fail while the problem exists.**

```bash
cargo test --workspace --release
cargo test -p neo-fold-clean --release --test system_fibonacci_bits_e2e -- --nocapture
# extra debugging output:
cargo test ... --features paper-exact,debug-logs
```

## neo-fold-clean test layout

Test files live in subdirectories of `crates/neo-fold-clean/tests/`; each file is
registered in `Cargo.toml` as a target named `<dir>_<file>` (so
`tests/system/lifecycle_redteam.rs` → `--test system_lifecycle_redteam`).

| Directory | Covers |
|---|---|
| `system/` | End-to-end chains (`fibonacci_bits_e2e`), lifecycle finalization/links/invariants, decider R1CS, terminal CE, production params, SHA-256 via Bellpepper, the `phase_1_*` F′ build-out suites |
| `direct_ccs/` | Direct-CCS frontend: R1CS round-trips and frontend red-team |
| `f_prime/` | F′ relation: R1CS scaffold, digest-circuit parity, source image, transcript red-team |
| `nifs/` | NIFS round-trip and isolated in-circuit NIFS.V |
| `reductions/` | Π_CCS split-NC verifier circuit (fe/nc/verifier), Π_RLC, Π_DEC, NIFS.V (+transcript), degree-7 Π_CCS, CCS-native Poseidon |
| `gadgets/` | R1CS builder primitives: booleans, u64/mux, Poseidon2, sum-check, transcript, alphabet sampling |
| `perf/` | `--ignored` perf snapshots — see [Profiling](profiling.md) |
| `support/` | Shared fixtures, including the Fibonacci F′ app fixture (`fibonacci_f_prime/`) and R1CS compiler fixtures |

Other crates keep their suites local: `neo-reductions/tests/` (engine parity, matrix
digests, digit-table parity), `neo-ajtai/tests/` (commit parity), etc.

## Red-team suites

The project treats tamper-rejection as a first-class test dimension. A red-team test
mutates one field of a proof/transcript/statement and asserts the verifier rejects
with the *specific* expected error:

- `system_lifecycle_redteam` — lifecycle-level tampers: audit-trail fields, final
  accumulator witnesses, counters, digests.
- `direct_ccs_redteam` — frontend-level tampers.
- `f_prime_transcript_redteam`, `reductions_nifs_v_transcript` — Fiat-Shamir binding:
  every absorbed datum must influence the challenges.
- `system_decider_ce_relation_isolation`, `nifs_r1cs_isolated` — relation/circuit
  isolation checks.

When you close a soundness gap, add the failing red-team case first — a test meant to
catch a problem must fail while the problem exists.

## Choosing the right e2e entry point

- Lifecycle behavior or folding changes → `system_fibonacci_bits_e2e`, then
  `system_lifecycle_redteam`.
- F′ structure/encoder changes → the relevant `system_phase_1_*` suite (they pin
  layout, fill, NIFS payloads, digests, recursive plan, and parity step by step).
- Decider changes → `system_decider_r1cs` plus the shape snapshot in `perf/`.
