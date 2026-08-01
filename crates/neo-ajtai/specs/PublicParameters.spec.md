# PublicParameters

## Purpose

- **What it is**: Supporting infrastructure for Ajtai: Poseidon2-based PRG for deterministic PP row expansion, balanced representation conversions for Goldilocks field elements, and error types.
- **Key invariant**: PRG is deterministic (same seed + row_idx -> same output) with domain separation preventing cross-circuit collisions.
- **Protocol role**: Seeded PP avoids materializing multi-GB matrices by expanding rows on-the-fly during commitment.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| `Setup(kappa, m) -> M` | Def 18, line 753 | `expand_row_v2(seed, row_idx, len) -> Vec<F>` | Deterministic row-by-row expansion of `M` |
| Balanced representative `bar(x)` | (used by decomp, not directly in paper) | `to_balanced_i128(x) -> i128` | Maps `F_q` to `[-(q-1)/2, (q-1)/2]` |

## Paper Anchors

Source: ./docs/superneo-paper/

- Definition 18 Setup, lines 753-756: `Setup(kappa, m) -> M` (PRG implements the deterministic version).
- (Balanced conversion is not directly in the paper but is used by `decomp_b`.)

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| `ProofSystem/Lattice.spec.md` | `SuperNeo/Lattice.lean` | Setup definition |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| PRG | `expand_row_v2(seed, row_idx, len) -> Vec<F>` | fn | Core | Poseidon2-based PRG with domain tag `neo/ajtai/prg/v2` |
| Balanced | `to_balanced_i128(x) -> i128` | fn | Core | Goldilocks to balanced representation `[-(q-1)/2, (q-1)/2]` |
| Balanced | `to_balanced_i64(x) -> i64` | fn | Core | Fast variant for hot paths |
| Error | `AjtaiError` | enum | Core | Error variants: `InvalidDimensions`, `InvalidInput`, `RangeViolation`, `SizeMismatch`, `EmptyInput`, `VerificationFailed`, `Internal` |
| Error | `AjtaiResult<T>` | type | Core | `Result` alias |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| PRG determinism: same seed + row_idx -> same output | Unit test | (none) |
| PRG domain separation: different domain tag -> different output | Unit test | (none) |
| Balanced conversion: `to_balanced_i128(0) == 0`, `to_balanced_i128(1) == 1` | Unit test | (none) |
| Balanced conversion: `to_balanced_i128(q-1) == -1` | Unit test | (none) |
| `to_balanced_i128` and `to_balanced_i64` agree | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Poseidon2 over Goldilocks is a secure PRF | Cryptographic assumption | Standard algebraic hash assumption; Poseidon2 security analysis applies |
| Domain tag `neo/ajtai/prg/v2` prevents cross-circuit collisions | Domain separation | Distinct tags produce independent PRG streams |
| `q = 2^64 - 2^32 + 1` is odd, so balanced representation is well-defined | neo-math spec | `(q-1)/2` is an integer |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-ccs::crypto::poseidon2_goldilocks`: Poseidon2 hash
- `neo-math`: `Fq` field type
- `p3-goldilocks`: Goldilocks field implementation

Downstream consumers:
- `neo-ajtai::commit`: seeded PP row expansion
- `neo-ajtai::decomp`: balanced conversion via `to_balanced_i64`

## Lean Oracle Conformance

All spec-derived tests (lean oracles + invariant obligations) live in `spec-tests/`.

| Test file | Oracle family | What it checks |
|---|---|---|
| (none) | (none) | Infrastructure module; no lean oracle vectors |

## Quality Expectations

- No `unsafe` (enforced crate-wide)
- `expand_row_v2` must be deterministic and reproducible across platforms
- `to_balanced_i64` must produce the same result as `to_balanced_i128` for all valid Goldilocks field elements
- `AjtaiError` variants cover all recoverable error conditions; panics are reserved for logic bugs only

## Acceptance Criteria

- `cargo test -p neo-ajtai --release` succeeds
- All invariant obligations have spec-tests
- PRG determinism verified across multiple invocations
- Balanced conversion edge cases (`0`, `1`, `q-1`, `(q-1)/2`) verified

## Out of Scope

- Alternative PRG constructions
- PRG security proofs
- Non-Goldilocks field support
