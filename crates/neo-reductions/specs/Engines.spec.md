# PiCCS engines

## Purpose

`PaperExactEngine` is the independent correctness oracle. `OptimizedEngine`
computes the same polynomial with cached matrix tables. On native targets,
`CrossCheckEngine` runs both concurrently from the same transcript state. It
requires equal transcript checkpoints, selected proof surfaces, outputs, and
canonical proof bytes before it returns the optimized result.

The active protocol is `PiCcsProofVariant::PaperRectangularV1`. The old
`SplitNcV1` and `BlockLaneNcDelayedV1` variants are legacy diagnostic paths.
The normal optimized verifier does not accept them.

## Ownership

| Owner | Responsibility |
|---|---|
| `engines/pi_ccs_protocol.rs` | Neutral messages, canonical bytes, absolute gamma layout, initial and terminal equations |
| `engines/pi_ccs_rectangular.rs` | Shared Poseidon2 transcript order, SumCheck phase driver, proof assembly, verifier replay |
| `engines/paper_exact_engine/paper_rectangular.rs` | Direct FE, NC, and one-joint square evaluators |
| `engines/optimized_engine/paper_rectangular.rs` | Cached Boolean tables for the same FE and NC polynomials |
| `engines/crosscheck_engine` | Exact outputs, rounds, folds, terminals, and canonical-byte comparison |

PaperExact does not import an optimized oracle, transformed evaluator cache,
sparse cache, digit table, or binary-search matrix lookup. A source dependency
test enforces this boundary. Its gamma layout, corrected target, FE terminal,
and NC terminal are also literal local formulas. They do not call the
optimized engine's canonical formula helpers.

## Canonical equations

For `K` fresh sources, `k` running sources, `t` matrices, and `d` ring
coefficients, the absolute gamma blocks are:

| Block | Exponents |
|---|---|
| fresh CCS | `i`, for `0 <= i < K` |
| all-source norm | `K + i`, for `0 <= i < K + k` |
| carried evaluation | `2K + k + i + k*j + k*t*l` |

FE uses only the padded row cube. NC uses only the padded column cube. There
is no coefficient or Ajtai SumCheck axis. A second transcript phase and a
second terminal point are the only rectangular-domain changes.

## Executable references

- `PaperJointSquareOracle` executes the paper's one-polynomial square case.
- `PaperRectangularFeOracle` and `PaperRectangularNcOracle` execute the exact
  row/column decomposition.
- `OptimizedPaperRectangularFeOracle` and
  `OptimizedPaperRectangularNcOracle` fold cached Boolean tables in place.

## Evidence

| Property | Evidence |
|---|---|
| optimized proof bytes equal PaperExact | `tests/paper_rectangular_parity.rs` |
| cross-check prover and verifier start both engines concurrently | `crosscheck_starts_both_engines_concurrently` |
| public cross-check mode covers `n < m`, `n > m`, and carried inputs | `public_crosscheck_mode_enforces_exact_reference_parity` |
| both `n < m` and `n > m` | `paper_exact_and_optimized_are_byte_exact_for_both_rectangular_directions` |
| every FE and NC round polynomial and fold on nontrivial invalid witnesses | `every_round_polynomial_and_fold_matches_on_nontrivial_invalid_witnesses` |
| one-joint square decomposition, including an invalid witness | `square_joint_oracle_is_exactly_the_fe_nc_decomposition` |
| FE, NC, gamma, source-order, and output mutations fail | `canonical_verifier_rejects_independent_protocol_mutations` |
| PaperExact has no optimized dependency | `paper_exact_active_sources_have_no_optimized_dependency` |
| all 324 fixed-shape Rust gamma slots match Lean | `paper_rectangular_lean_artifact.rs` and `PiCcsPaperRectangular/Conformance.lean` |
| model-level square identity | `PaperRectangular.Square.joint_qAt_eq_fe_add_nc` and `joint_summedQ_eq_summedFe_add_summedNc` |

## Legacy boundary

All block/lane, device-backend, replay, and deferred computations are below
`optimized_engine::legacy_split_nc`. They are not available from the normal
optimized API, are not a PaperExact oracle, and are not accepted by the normal
optimized verifier. The current fixed-profile recursive circuit still uses
this explicit namespace. It is migration code and is not covered by the
`PaperRectangular` Lean theorem or the byte-equality claim.

## Acceptance commands

```text
cargo test -p neo-reductions --release --test paper_rectangular_parity
cargo test -p neo-reductions --release --features paper-exact --test paper_rectangular_parity crosscheck
cargo test -p neo-reductions --release --test paper_rectangular_lean_artifact
cargo test -p neo-reductions --release --test k_mcs_end_to_end
```
