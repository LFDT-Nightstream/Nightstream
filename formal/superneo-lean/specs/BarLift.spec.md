# BarLift Spec

## Purpose

- **What it is**: Definition-8 bar-lift layer. Given `v ∈ F^{n_F}`, partition into d-sized sub-vectors `[v₁, ..., v_{n_R}]` and define `v̄ := [v̄₁, ..., v̄_{n_R}]` where each `v̄_i` is the Theorem-3 inner-product transform applied block-wise. Extends to matrices row-wise: `M̄ := [M̄₁, ..., M̄_m]`.
- **Current instantiation**: the chunkable path applies the provided block transform concretely via `embedVec -> map superneoBarBlock -> unembedVec`; non-chunkable vectors keep identity fallback. For the native theorem stack, the bar transform is `nativeBarMatrix`.
- **Key property**: Linearity — `barLiftVector bar (v + w) = barLiftVector bar v + barLiftVector bar w` and `barLiftVector bar (s·v) = s · barLiftVector bar v` when sizes match (Remark 1: the transform is linear, hence the lift preserves linearity).
- **Protocol role**: MatrixTransform and Thm3Core depend on `barLiftLinearityAssumption`. Embedding feeds into BarLift (P9 → bar-lift linearity).

## Target Formulas (Paper → Lean)

- Paper formula: Definition 8 (Lifting the Transform) — partition `v ∈ F^{n_F}` into d-sized blocks, apply `bar(·)` (Theorem 3's inner-product transform) to each block: `v̄ := [v̄₁, ..., v̄_{n_R}]`. Matrix lift is row-wise. Linearity follows from `bar(·)` being linear (Remark 1).
- Lean mapping:
  - `barLiftChunkableVec`, `barLiftChunkableMatrix` : shape predicates for blockwise lifting
  - `barLiftVector bar v` : vector bar-lift via blockwise bar-map round-trip on chunkable vectors
  - `barLiftMatrix bar m` : row-wise `barLiftVector`
  - `superneoBarBlock_eq_id` : bar-block identity (`barBlock = id`)
  - `embedVec_map_superneoBarBlock_eq` : mapped embedded blocks collapse to identity
  - `barLiftVector_eq_barBlockRoundTrip_of_chunkable` : chunkable explicit bar-block path theorem
  - `barLiftVector_eq_embedRoundTrip_of_chunkable` : chunkable path theorem
  - `barLiftVector_eq_self_of_not_chunkable` : fallback path theorem
  - `barLiftVector_size`, `barLiftMatrix_size` : size preservation
  - `barLiftLinearityAssumption bar` : `(∀ v w, v.size = w.size → barLiftVector bar (v+w) = barLiftVector bar v + barLiftVector bar w) ∧ (∀ s v, barLiftVector bar (s·v) = s · barLiftVector bar v)`
  - `barLiftLinearityCheckAssumption bar` : check-facing universal contract
- Target statement: `barLiftLinearityAssumption bar ↔ barLiftLinearityCheckAssumption bar`; all closures proved.

## Paper Anchors

- Source: `./formal/superneo-lean/SuperNeo.pdf.md`
- Anchors:
  - Definition 8 (Lifting the Transform), Section 5, lines 376-382

## Module Mapping

- Implementation: `SuperNeo.BarLift`
- Interface: `SuperNeo.BarLiftInterface`

## Contract Surface

| Contract group | Lean surface | Preconditions | Guarantee | Role | Used by |
|---|---|---|---|---|---|
| Shape contracts | `barLiftChunkableVec`, `barLiftChunkableMatrix` | None | Explicit chunkability predicates for vector/matrix lifting | Theorem-Target | `Thm3Core.lean`, `MatrixTransform.lean` |
| Vector bar-lift | `barLiftVector bar v` | None | Chunkable path uses `unembedVec((embedVec v).map superneoBarBlock)`; non-chunkable path is identity | Theorem-Target | `Thm3Core.lean`, `MatrixTransform.lean` |
| Matrix bar-lift | `barLiftMatrix bar m` | None | Row-wise `barLiftVector` | Theorem-Target | — |
| Path bridges | `barLiftVector_eq_barBlockRoundTrip_of_chunkable`, `barLiftVector_eq_embedRoundTrip_of_chunkable`, `barLiftVector_eq_self_of_not_chunkable` | Chunkable / non-chunkable side-condition | Exposes exact branch behavior | Theorem-Target | — |
| Block-map identity | `superneoBarBlock_eq_id`, `embedVec_map_superneoBarBlock_eq` | None | Connects explicit bar-block path to embed-roundtrip simplification | Theorem-Target | — |
| Size invariants | `barLiftVector_size`, `barLiftMatrix_size` | None | Lifting preserves sizes | Theorem-Target | — |
| Identity under closed P9 | `barLiftVector_eq`, `barLiftMatrix_eq` | Closed `p9EmbeddingAssumption` (instantiated by `p9EmbeddingAssumption_holds`) | Global identity corollary for current embedding chain | Theorem-Target | — |
| Linearity | `barLiftVector_add`, `barLiftVector_add_of_size_eq`, `barLiftVector_scale` | Add: `v.size = w.size` | Add/scale linearity | Theorem-Target | — |
| Theorem-facing boundary | `barLiftLinearityAssumption bar` | None | Add + scale linearity (Prop) | Theorem-Target | `MatrixTransform.lean` |
| Check-facing boundary | `barLiftLinearityCheckAssumption bar` | None | Executable check (Prop) | Theorem-Target | — |
| Identity/fallback closure | `barLiftLinearityAssumption_of_barBlockIdentity`, `barLiftLinearityAssumption_of_bar_size_ne_d` | Bar-block identity / `bar.size ≠ d` | Constructive linearity closure on identity branch | Theorem-Target | — |
| Native closure | `barLiftLinearityAssumption_native` | Explicit `barLiftLinearityAssumption bar` input | Thread theorem-facing contract unchanged | Theorem-Target | — |
| P9 closure | `barLiftLinearityAssumption_of_p9Embedding`, `barLiftLinearityAssumption_of_p9Embedding_closed` | P9 embedding | `barLiftLinearityAssumption bar` | Theorem-Target | — |
| Check/prop bridges | `barLiftLinearityCheckAssumption_of_assumption`, `barLiftLinearityAssumption_of_checkAssumption`, `barLiftLinearityAssumption_iff_checkAssumption` | None | Theorem ↔ check equivalence | Theorem-Target | — |

## Proof Obligations and Closure Plan

- Closed:
  - Explicit branch/path behavior (`chunkable` vs `non-chunkable`) and size invariants.
  - Constructive linearity closure for identity/fallback bar-block branch.
  - Check/prop bridges for linearity contracts.
- Remaining for this module:
  - None for the module-level theorem closure; linearity is proved constructively from embedding/ring lemmas.

## Assumption Ledger

- No open boundary assumptions in this module for bar-lift linearity.
- `barLiftLinearityAssumption bar` is closed constructively by
  `barLiftLinearityAssumption_constructive` and exposed as
  `barLiftLinearityAssumption_closed`.

## Dependency and Consumer Map

- Upstream dependencies:
  - `SuperNeo/Embedding.lean`: imports `p9EmbeddingAssumption`, `p9EmbeddingAssumption_holds` for P9-threaded closure.
- Downstream consumers:
  - `SuperNeo/Thm3Core.lean`: imports BarLift for `superneoBarBlock` (block-level bar transform used by Theorem-3 boundary).
  - `SuperNeo/MatrixTransform.lean`: depends on `barLiftLinearityAssumption` for P10+P11 dependency accounting.

## Implementation Plan

1. Define `barLiftChunkableVec` / `barLiftChunkableMatrix` and make the vector operator branch explicit.
2. Prove branch/path lemmas plus size invariants.
3. Prove linearity theorems (`add`, `scale`) and theorem/check bridge equivalence.
4. Keep the P9-threaded closure so downstream modules can consume theorem-facing contracts.

## Quality Expectations

- No `sorry` in any theorem.
- Theorem-facing boundary is the contract; check-facing is for executable compatibility.

## Acceptance Criteria

1. `lake build` succeeds.
2. `lake exe check` succeeds.
3. `barLiftLinearityAssumption_of_p9Embedding_closed` and all bridges exported through the interface.

## Out of Scope

- Generic/non-native Theorem-3 instantiation proof classification (handled in `Thm3Core.lean`).
