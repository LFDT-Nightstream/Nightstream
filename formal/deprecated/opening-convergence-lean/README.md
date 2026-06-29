# Opening Convergence — Lean Formalization

Status: deprecated / parked. This package is retained for historical theorem
context and does not describe the current active Rust implementation path.

Standalone Lean 4 project proving the soundness of Nightstream's opening
convergence pipeline: the reduction from ~600 family-level evaluation
claims down to 6 final Ajtai PCS openings.

**Central goal:** If the v1 convergence verifier accepts, then all six
original in-scope family opening obligations hold, with negligible failure
probability.

## Project Structure

```
formal/deprecated/opening-convergence-lean/
  lakefile.toml
  lean-toolchain                  (leanprover/lean4:v4.28.0)
  OpeningConvergence.lean         (root import)
  OpeningConvergence/
    Basic.lean                    (shared definitions & types)
    PayloadSemanticsInterface.lean
    PayloadSemantics.lean
    BatchEvalReductionInterface.lean
    BatchEvalReduction.lean
    SamePointAccumulationInterface.lean
    SamePointAccumulation.lean
    ConvergenceSoundnessInterface.lean
    ConvergenceSoundness.lean
    SuperNeoBoundaryInterface.lean
    SuperNeoBoundary.lean
    SuperNeoExtensionBridgeInterface.lean
    SuperNeoExtensionBridge.lean
    SuperNeoPhase1SumcheckBridgeInterface.lean
    SuperNeoPhase1SumcheckBridge.lean
    SuperNeoConvergenceClosureInterface.lean
    SuperNeoConvergenceClosure.lean
  specs/
    PayloadSemantics.spec.md
    BatchEvalReduction.spec.md
    SamePointAccumulation.spec.md
    ConvergenceSoundness.spec.md
    SuperNeoBoundary.spec.md
    SuperNeoExtensionBridge.spec.md
    SuperNeoPhase1SumcheckBridge.spec.md
    SuperNeoConvergenceClosure.spec.md
```

## Dependency

Depends on `superneo-lean` (sibling directory) for MLE, EqPoly,
SumCheck, Schwartz-Zippel, the extracted quadratic extension carrier
`SuperNeo.KExt`, and the concrete CE/lattice opening relation.

## Implementation Status

Closure legend:
- **Proved** — states and proves the correct mathematical proposition with no `sorry`
- **Integrated Boundary** — concrete theorem-facing instantiation exists and builds
- **Trusted Boundary** — explicit carried hypothesis / imported boundary that
  the current package uses but does not derive internally

The current project is **theorem-complete** for the frozen Nightstream
opening-convergence scope. `lake build` passes, there are no local theorem or
definition `sorry`s, the generic convergence package is fully proved, the
concrete split extension-field boundary `boundaryK` is integrated, and Module 7
proves the concrete end-to-end closure theorem against that boundary.

### Module 1: PayloadSemantics

| # | Theorem | Priority | Status | Notes |
|---|---------|----------|--------|-------|
| 1 | `unpackLinearity` | P0 | **Proved** | Real proof against the concrete `pack` / `unpack` / `mleEval` definitions |
| 2 | `payloadPcsConsistency` | P0 | **Proved** | Currently closes the carried coefficient-consistency boundary; it does not yet derive the full ring-lift/MLE commutation theorem internally |

### Module 2: BatchEvalReduction

| # | Theorem | Priority | Status | Notes |
|---|---------|----------|--------|-------|
| 3 | `claimedSumCorrectness` | P0 | **Proved** | Real proof from `eqPoly` symmetry, `mleEval`, and finite-sum rearrangement |
| 4 | `coefficientLinearization` | P0 | **Proved** | Real finite-root-count proof over the 54-coefficient collapse polynomial |
| 5 | `gammaLinearization` | P0 | **Proved** | Real finite-root-count proof via `Polynomial.ofFn` and `card_roots'` |
| 5b | `rhoLinearization` | P0 | **Proved** | Real finite-root-count proof for the cross-claim `rho^(i+1)` batching step |
| 6a | `phase1SoundnessCore` | P0 | **Proved** | Real deterministic composition of rho, gamma, and eta consistency against the good-event hypotheses |
| 6b | `phase1SoundnessFailureBound` | P0 | **Proved** | Uses the frozen union-bound bookkeeping definition for the five Phase 1 failure components |
| 6c | `phase1Soundness` | P0 | **Proved** | Composes the deterministic core with the failure-bound theorem under explicit good-event hypotheses, including `sumcheckTerminalCorrect` |
| — | `sameObjectPayloadUniqueness` | Bridge | **Proved** | Closes the same-object bridge using Phase 1 soundness plus the explicit opened-object functionality hypothesis |

### Module 3: SamePointAccumulation

| # | Theorem | Priority | Status | Notes |
|---|---------|----------|--------|-------|
| 7a | `phase2IdentityCollapse` | P0 | **Proved** | Real equality statement, no `sorry` |
| 7b | `phase2IdentityCollapseProvenance` | P1 | **Proved** | Real monotonicity/provenance statement, no `sorry` |
| 8 | `singletonPassthrough` | P0 | **Proved** | Real definitional equality (`rfl`) |

### Module 4: ConvergenceSoundness

| # | Theorem | Priority | Status | Notes |
|---|---------|----------|--------|-------|
| 9 | `finalOpeningAdequacy` | P0 | **Proved** | Uses the carried Phase 2→Phase 1 and Phase 1→original adequacy witnesses |
| 10 | `v1ConvergenceSoundness` | P0 | **Proved** | Uses `finalOpeningAdequacy` plus the frozen end-to-end bookkeeping bound |
| S1 | `accumulatorDedupSound` | Structural | **Proved** | Exposes the carried Phase 0 dedup uniqueness invariant from `V1PipelineResult` |
| S2 | `bucketPartitionDeterministic` | Structural | **Proved** | Follows from `List.Perm.countP_eq` |
| S3 | `collapseGroupingDeterministic` | Structural | **Proved** | Follows from `List.Perm.countP_eq` |
| S4 | `phase1PartitionExact` | Structural | **Proved** | Exposes the carried exact bucket-slot partition witness from `V1PipelineResult` |
| S5 | `phase2GroupingExact` | Structural | **Proved** | Exposes the carried exact Phase 2 grouping witness from `V1PipelineResult` |

### Module 5: SuperNeoBoundary

| Surface | Priority | Status | Notes |
|---------|----------|--------|-------|
| `boundary` | P0 | **Integrated Boundary** | Concrete base-field `AjtaiPCSBoundary SuperNeo.F` built from CE witness existence under an explicit registry, with schema and row-domain checks on the resolved opened object |
| `boundary_lookup_self` | Structural | **Proved** | Successful lookup returns the object keyed by that identity |

### Module 6: SuperNeoExtensionBridge

| Surface | Priority | Status | Notes |
|---------|----------|--------|-------|
| `pointToBaseCoeffs`, `payloadToSplitEvaluations` | P0 | **Proved** | Canonical split encoding from `SuperNeo.KExt` claims into the current base-field CE statement shape |
| `claimStatementK` | P0 | **Proved** | Frozen CE-statement translation for extension-field claims |
| `boundaryK` | P0 | **Integrated Boundary** | Concrete extension-field `AjtaiPCSBoundary SuperNeo.KExt` induced by split CE statements |
| `point_eq_of_split_blocks_eq`, `packedColumn_eq_of_split_blocks_eq` | Bridge | **Proved** | Split blocks determine the original extension-field point / packed column |
| `pointToBaseCoeffs_injective`, `payloadToSplitEvaluations_injective`, `claimStatementK_injective` | Bridge | **Proved** | The canonical split statement determines the original extension-field point and payload once the opened-object schema is fixed |

### Module 7: SuperNeoConvergenceClosure

| Surface | Priority | Status | Notes |
|---------|----------|--------|-------|
| `v1ConvergenceSoundness_boundaryK` | P0 | **Proved** | Concrete end-to-end specialization of the generic convergence theorem to the real extension-field SuperNeo boundary |

### Module 8: SuperNeoPhase1SumcheckBridge

| Surface | Priority | Status | Notes |
|---------|----------|--------|-------|
| `pointArray`, `rStarArray`, `trueColumnAtPoint`, `trueClaimAtPoint`, `phase1BatchedPolynomial`, `trueClaimTable` | P0 | **Proved / Defined** | Freezes both the actual degree-2 Phase 1 polynomial semantics and its Boolean-cube restriction |
| `phase1BatchedPolynomial_rStar_eq_trueClaimSum` | P0 | **Proved** | Makes the direct polynomial-at-`r*` target explicit inside Lean |
| `sumcheckTerminalCorrect_of_extensionFinalOracle` | P0 | **Proved** | Narrows the old abstract Phase 1 `sumcheckTerminalCorrect` boundary to a concrete extension-field SumCheck final-oracle witness plus an explicit guarded-`mleEvalK` bridge for the weighted scalar sum `Σ_i rho^(i+1) * eq(r_i, r*) * trueClaimLinearized_i` |

### Summary

| Category | Total |
|----------|-------|
| Core theorem declarations (Modules 1-4) | 20 |
| **Core theorems proved** | 20 |
| Module 6 bridge lemmas proved | 13 |
| Module 7 closure theorems proved | 1 |
| **Proof Missing** | 0 |
| Concrete boundary / bridge modules | 4 |

### Closure Readout

- **Scaffolding:** complete
- **Type-checking:** complete
- **Paper-faithful:** yes, for the frozen split-extension Nightstream boundary
- **Spec-faithful:** yes
- **Theorem-complete:** yes, for this package's frozen scope
- **Fully proof-complete from primitive facts:** not yet; see explicit trusted boundaries below

## Explicit Trusted Boundaries

These are not hidden `sorry`s. They are theorem-facing boundaries that remain
visible in the current package and therefore still matter if the goal is the
strongest possible repo-wide proof-complete closure.

1. **Payload coefficient consistency boundary**
   `payloadPcsConsistency` currently closes the carried coefficient hypothesis.
   It does not yet derive the full `R_F -> R_K` lift / MLE-commutation theorem
   from quotient-ring infrastructure.

2. **Sumcheck soundness / table-semantics boundary**
   `phase1SoundnessCore` still takes `sumcheckTerminalCorrect` as an explicit
   hypothesis, but the package no longer leaves that boundary completely
   abstract. Module 8 proves
   `sumcheckTerminalCorrect_of_extensionFinalOracle`, which reduces it to a
   concrete extension-field SumCheck witness:
   - `extensionSumcheckStatementTranscriptConsistent inst stmt tr`
   - a verifier-side terminal-value identification
   - an explicit `mleEvalK stmt.table ... = trueClaimSum` bridge

   The module now uses the proved extension-field theorems
   `mleEvalK_eq_mleByFoldingK_of_size` and
   `mleEvalLinearityAssumptionK_holds`, so the remaining semantic boundary is
   no longer missing extension-field MLE algebra or unnamed table objects.
   The canonical Boolean-cube objects are now explicit:
   - `trueColumnAtPoint`
   - `trueClaimAtPoint`
   - `phase1BatchedPolynomial`
   - `trueClaimTable`

   The remaining gap is more precise and more serious than a raw missing
   witness: the current `ExtensionSumCheck` surface is still table/MLE-based,
   while frozen Phase 1 targets the degree-2 polynomial
   `P(x) = Σ_i rho^(i+1) * eq(r_i, x) * g_i(x)`.

   So the remaining paper-faithful closure step is not just “show
   `mleEvalK trueClaimTable r* = trueClaimSum`”. It is to bridge or replace
   the current table/MLE final-oracle model so it matches the actual Phase 1
   degree-2 polynomial semantics at the verifier-derived point.

3. **Probability-model boundary**
   `phase1BadEventProbability` and `v1FailureProbability` are frozen
   bookkeeping expressions. The current package proves the arithmetic bound,
   not a fully modeled transcript-coin probability space.

4. **Adequacy-witness boundary**
   `finalOpeningAdequacy` and `v1ConvergenceSoundness` rely on the explicit
   carried witnesses:
   - `phase2ToPhase1Adequacy`
   - `phase1ToOriginalAdequacy`
   The package proves the composition is correct once those witnesses are
   provided; it does not derive them from lower-level CE facts internally.

5. **Concrete PCS boundary gap is closed**
   The older “abstract PCS only” caveat no longer applies at the package
   boundary. `boundaryK` and `v1ConvergenceSoundness_boundaryK` give a
   concrete extension-field specialization. The remaining boundaries are the
   four items above, not the absence of a concrete final theorem.

## Beyond Current Scope

This package is theorem-complete for the frozen v1 / Phase 2a protocol surface:

- Phase 1 point unification for the current safe buckets
- Phase 2a same-object same-point identity collapse
- concrete SuperNeo split-extension boundary
- concrete end-to-end closure theorem for the resulting 6-opening verifier

The next theorem frontier is **not** more work inside the current package.
It is the future **Phase 2b cross-object accumulation** target from
`docs/plans/2026-04-05-opening-convergence-design.md`.

Before that, the most direct remaining closure step for the current frozen
v1 / Phase 2a story is in the sibling `superneo-lean` package:

- finish extension-field SumCheck soundness/completeness over `SuperNeo.KExt`,
  so the remaining concrete SumCheck witness in Module 8 can be derived from
  imported theorems rather than carried as an explicit boundary

That future work should only start after the design is frozen on:

1. the accumulation scalar domain,
2. the broadened cross-family Phase 1 bucket,
3. the norm / MSIS model (or explicit `Π_DEC` scope),
4. the `AccumulatedEvalClaim` identity type.

Until those four items are frozen, adding a new Lean accumulation module
would be premature and would risk proving the wrong protocol.

| Infrastructure | Status |
|----------------|--------|
| Shared definitions (`Basic.lean`) | Concrete; PCS boundary is now field-free |
| Interface files (8) | Present, type-check |
| Spec files (8) | Present |
| Implementation files (8) | Present, with theorem-facing content still living mostly in interface files |
| `lake build` | Passes for `opening-convergence-lean`, including the root `OpeningConvergence` target |

## Build

```bash
cd formal/deprecated/opening-convergence-lean
lake build
```

## Recommended Closure Order

1. **Module 1** — PayloadSemantics (`unpackLinearity`)
2. **Module 2** — BatchEvalReduction (Theorems 3-6, bridge lemma)
3. **Module 3** — SamePointAccumulation (Theorem 7)
4. **Module 4** — ConvergenceSoundness (Theorems 9-10, structural)
5. **Module 5** — SuperNeoBoundary (registry-backed CE instantiation)
6. **Module 6** — SuperNeoExtensionBridge (split encoding + `boundaryK`)
7. **Module 7** — SuperNeoConvergenceClosure (concrete end-to-end specialization to `boundaryK`)
8. **Module 8** — SuperNeoPhase1SumcheckBridge (concrete Phase 1 final-oracle bridge to `sumcheckTerminalCorrect`)

Module 4 depends on Modules 1-3. Module 5 depends on the concrete SuperNeo CE
surfaces but not on the reduction proofs. Module 6 freezes the extension-field
encoding boundary, Module 7 closes the final concrete specialization step, and
Module 8 narrows the remaining Phase 1 SumCheck boundary to explicit
extension-field protocol objects.
