# ConvergenceSoundness Specification

## Purpose

Compose the full reduction pipeline (Modules 1-3) into the end-to-end
v1 verifier soundness theorem. This module is the actual payoff of the
Lean formalization effort.

## Target Formulas

### Theorem 9: FinalOpeningAdequacy

If the final Ajtai PCS openings verify for all 6 ReducedEvalClaims, then
the original Phase 0 FamilyEvalClaims are satisfied.

```
(forall k in Fin 6:
  pcs.verify(reduced[k].openedObject, reduced[k].point, reduced[k].payload))
  AND Phase1SoundnessHolds
  AND Phase2CollapseCoversOriginals
  -->
  ClaimsSatisfied(original Phase 0 claims)
```

This theorem connects the end of the pipeline back to the beginning.
Without it, the pipeline could be locally sound at every step but fail
to connect reduced claims to original claims.

The Lean surface carries two explicit adequacy witnesses inside
`V1PipelineResult` to make this composition honest:
- one explicit PCS boundary `AjtaiPCSBoundary`
- `phase2ToPhase1Adequacy`:
  a verified reduced opening implies the corresponding Phase 1 unified
  claim opening
- `phase1ToOriginalAdequacy`:
  a verified Phase 1 unified claim plus Phase 1 semantic correctness
  implies the corresponding original claim

### Theorem 10: V1ConvergenceSoundness

The single end-to-end theorem:

```
If the v1 convergence verifier accepts,
then all six original in-scope family opening obligations hold:

  forall family F in {Stage1Rows, Stage2RegisterReads, Stage2RegisterWrites,
                      Stage2RamEvents, Stage2TwistLinks, Stage3Continuity}:
    forall claim C emitted by family F:
      MLE_eval(committed_columns(C.openedObject), C.point)
        = C.payload.columnEvals

with total failure probability bounded by v1TotalErrorBound.
```

This is the theorem that gives operational certainty.

## Structural Lemmas

### S1: AccumulatorDedupSound

The carried v1 pipeline result explicitly includes the dedup key used for
the Phase 0 output together with its uniqueness invariant:
- equal-content duplicates have already been collapsed
- conflicting duplicates have already been rejected
- therefore no two distinct output positions share the same dedup key

### S2: BucketPartitionDeterministic

Bucketing depends only on claim content (schema), not insertion order.
Given any permutation of the same claims, the same bucket partition
results (up to within-bucket ordering).

### S3: CollapseGroupingDeterministic

Phase 2a grouping by (opened_object, point) is canonical and
order-preserving. Given the same input claims in the same order,
the same groups result.

### S4: Phase1PartitionExact

The carried v1 pipeline result explicitly includes the exact Phase 1
bucket-slot partition witness:
- every bucket slot points to a valid original claim index
- every original claim index appears in exactly one bucket slot

### S5: Phase2GroupingExact

The carried v1 pipeline result explicitly includes the exact Phase 2
grouping witness:
- every original claim index appears in exactly one reduced-claim group
- no original claim index is omitted or duplicated across groups

## Composition Structure

```
Phase 0: Accumulate + dedup claims (S1, S2)
  |
  v
Phase 1: Batch eval reduction per bucket (Theorems 3-6)
  |  Output: unified claims at r* with SameObjectPayloadUniqueness
  v
Phase 2: Identity collapse per group (Theorems 7-8, S3)
  |  Output: 6 ReducedEvalClaims
  v
PCS: Ajtai opens each ReducedEvalClaim (Theorem 9)
  |  Connects to payload semantics (Theorems 1-2)
  v
Theorem 10: V1ConvergenceSoundness (end-to-end)
```

## Explicit Type Signatures

| Lean symbol | Type | Lives in |
|---|---|---|
| `AjtaiPCSBoundary` | Structure | Explicit imported PCS boundary for one opened Ajtai object |
| `V1_OPENING_COUNT` | `Nat` (= 6) | Constant |
| `ClaimSatisfied` | `FamilyEvalClaim -> Prop` | One original claim satisfies the imported Ajtai PCS boundary |
| `ClaimsSatisfied` | `List FamilyEvalClaim -> Prop` | All original claims satisfy the imported Ajtai PCS boundary |
| `V1PipelineResult` | Structure | Full pipeline state, including Phase 0 dedup uniqueness, exact Phase 1/2 grouping witnesses, and explicit adequacy bridges under one PCS boundary |
| `Phase2CollapseCoversOriginals` | Prop | Every original claim is covered by some reduced claim |
| `V1VerifierAccepts` | Structure | All 4 verifier steps pass |
| `pipelineBucketParams` | `List (Nat x Nat x Nat)` | Frozen `(ell, N, m)` data extracted from the pipeline |
| `v1TotalErrorBound` | `Q` | Union bound over all buckets |
| `v1FailureProbability` | `Q` | Frozen end-to-end union-bound bookkeeping quantity |

## Error Bound

```
v1TotalErrorBound = sum over buckets b of phase1ErrorBound(ell_b, N_b, m_b)

where phase1ErrorBound(ell, N, m) = (2*ell + N*ell + AJTAI_D + m + N) / |K|
```

For v1 with |K| >= 2^128, this is negligible.

## Paper Anchors

- SuperNeo Section 7: End-to-end verifier soundness
- Nightstream architecture: Opening convergence pipeline
- Jolt Section 5: Composition of opening reductions

## Module Mapping

| Existing module | Import | What it provides |
|---|---|---|
| `OpeningConvergence.PayloadSemanticsInterface` | Theorems 1-2 | Payload correctness |
| `OpeningConvergence.BatchEvalReductionInterface` | Theorems 3-6, bridge | Phase 1 soundness |
| `OpeningConvergence.SamePointAccumulationInterface` | Theorems 7-8 | Phase 2 soundness |

## Contract Surface

| Lean symbol | Kind | Role | Guarantee |
|---|---|---|---|
| `finalOpeningAdequacy` | Theorem | P0 | Verified reduced openings imply `ClaimsSatisfied originalClaims` via the carried adequacy witnesses |
| `v1ConvergenceSoundness` | Theorem | P0 | `ClaimsSatisfied originalClaims` and the frozen end-to-end bookkeeping bound |
| `accumulatorDedupSound` | Theorem | S | Exposes the carried Phase 0 dedup uniqueness invariant |
| `bucketPartitionDeterministic` | Theorem | S | Bucketing is order-independent |
| `collapseGroupingDeterministic` | Theorem | S | Phase 2 grouping is canonical |
| `phase1PartitionExact` | Theorem | S | Exposes the carried exact Phase 1 bucket partition witness |
| `phase2GroupingExact` | Theorem | S | Exposes the carried exact Phase 2 grouping witness |
| `AjtaiPCSBoundary` | Structure | Boundary | Field-free PCS verifier carrier, instantiated concretely by the current base-field `SuperNeoBoundary` |
| `V1PipelineResult` | Structure | Composition | Full pipeline state with explicit Phase 0/1/2 structural witnesses and adequacy bridges under one PCS boundary |
| `V1VerifierAccepts` | Structure | Input | All verifier checks passed |
| `v1TotalErrorBound` | Definition | Bound | Union over all bucket error bounds |

This module is the generic payoff theorem. Module 7
(`SuperNeoConvergenceClosure`) supplies the concrete specialization to the real
split extension-field SuperNeo boundary.
