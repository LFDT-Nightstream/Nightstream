# SModule

## Purpose

- **What it is**: Global PP registry and `AjtaiSModule` implementing `SModuleHomomorphism<Fq, Commitment>` — the S-module structure ensuring commitment is compatible with the ring S-action.
- **Key invariant**: `commit(rho * Z) = rho * commit(Z)` (S-module linearity) and registry idempotency (same seed always produces same PP).
- **Protocol role**: Pi_RLC verification requires S-module homomorphism — the verifier checks `rho * c_old == c_new` without re-opening.

## Target Formulas (Paper -> Rust)

| Paper notation | Paper reference | Rust identifier | Notes |
|---|---|---|---|
| Homomorphism: `L(rho * z) = rho * L(z)` | Thm 2, line 319 | `AjtaiSModule` implementing `SModuleHomomorphism` | Implicit from Ajtai linearity |
| `M` (public parameter matrix) | Def 18, line 753 | `PP<RqEl>` via global registry | Shared across protocol layers |

## Paper Anchors

Source: ./docs/superneo-paper/

- Theorem 2, lines 319-321: Ajtai commitment is homomorphic (implicit S-module structure).
- (S-module homomorphism is an implementation detail, not explicitly formalized in the paper.)

## Lean Cross-Reference

| Lean spec | Lean module | Relationship |
|---|---|---|
| (none) | (none) | Implementation-only; no Lean counterpart |

## Contract Surface

| Group | Rust symbol | Kind | Role | Guarantee |
|---|---|---|---|---|
| Registry | `set_global_pp(pp) -> Result<(), AjtaiError>` | fn | Core | Register materialized PP |
| Registry | `set_global_pp_seeded(d, kappa, m, seed) -> Result<(), AjtaiError>` | fn | Core | Register seeded PP |
| Registry | `get_global_pp() -> Result<PPRef, AjtaiError>` | fn | Core | Get sole PP |
| Registry | `get_global_pp_for_dims(d, m) -> Result<PPRef, AjtaiError>` | fn | Core | Get PP by dimensions |
| Registry | `get_global_pp_for_z_len(z_len) -> Result<PPRef, AjtaiError>` | fn | Core | Get PP by witness length |
| Registry | `get_global_pp_seeded_params_for_dims(d, m) -> Result<(usize, [u8; 32]), AjtaiError>` | fn | Helper | Get seed params |
| Registry | `has_global_pp_for_dims(d, m) -> bool` | fn | Helper | Check existence |
| Registry | `has_seed_for_dims(d, m) -> bool` | fn | Helper | Check seed existence |
| Registry | `try_get_loaded_global_pp_for_dims(d, m) -> Option<PPRef>` | fn | Helper | Non-loading check |
| Registry | `unload_global_pp_for_dims(d, m) -> Result<bool, AjtaiError>` | fn | Core | Free memory, keep seed |
| S-Module | `AjtaiSModule` | struct | Core | Implements `SModuleHomomorphism` |
| S-Module | `AjtaiSModule::new(pp)` | fn | Core | From owned PP |
| S-Module | `AjtaiSModule::from_global()` | fn | Helper | From sole global PP |
| S-Module | `AjtaiSModule::from_global_for_dims(d, m)` | fn | Core | From global by dimensions |
| S-Module | `AjtaiSModule::from_global_for_z_len(z_len)` | fn | Core | From global by witness length |
| S-Module | `AjtaiSModule::kappa()` | fn | Helper | Get kappa without materializing PP |

## Invariant Obligations

| Invariant | Verification method | Lean theorem counterpart |
|---|---|---|
| Registry set/get round-trip | Unit test | (none) |
| Seeded PP unload/reload produces same commitment | Unit test | (none) |
| Dimension matching: `get_for_dims` returns correct PP | Unit test | (none) |

## Assumption Ledger

| Assumption | Source | Justification |
|---|---|---|
| Global mutable state is thread-safe | `OnceLock` / `RwLock` | Standard Rust concurrency primitives |
| Seeded PP expansion is deterministic | PRG determinism | Same seed + dimensions always produce the same matrix |

## Dependency and Consumer Map

Upstream dependencies:
- `neo-math`: `D`
- `neo-ccs`: `Mat`, `SModuleHomomorphism` trait

Downstream consumers:
- `neo-fold`: shard commitment via `SModuleHomomorphism` trait
- `neo-reductions`: `Pi_RLC`, `Pi_DEC` via `SModuleHomomorphism` trait

## Lean Oracle Conformance

All spec-derived tests (lean oracles + invariant obligations) live in `spec-tests/`.

| Test file | Oracle family | What it checks |
|---|---|---|
| (none) | (none) | Implementation-only module; no lean oracle vectors |

## Quality Expectations

- No `unsafe` (enforced crate-wide)
- Global registry uses interior mutability with proper synchronization (`RwLock` or equivalent)
- `unload_global_pp_for_dims` frees the materialized matrix but retains the seed for lazy re-expansion
- `AjtaiSModule::kappa()` does not trigger PP materialization

## Acceptance Criteria

- `cargo test -p neo-ajtai --release` succeeds
- All invariant obligations have spec-tests
- Registry round-trip works for both materialized and seeded PP
- `SModuleHomomorphism` trait methods produce results consistent with direct commitment

## Out of Scope

- Multi-party PP generation
- Distributed registry
- PP serialization format
