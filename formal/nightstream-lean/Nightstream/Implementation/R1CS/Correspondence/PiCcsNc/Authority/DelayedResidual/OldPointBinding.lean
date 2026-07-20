import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.ProjectionBinding

/-!
Contract: promote the compact delayed-projection identity to the complete
old-point relation consumed by the recursive authority transfer.

Owns: exact active-lane decoding of the 54-coefficient parent projection,
canonical padded-lane completion, and the exact-old-point-or-degree-53-root
result.

Does not own: transcript derivation, the production NC SumCheck, recursive
state continuity, Rust/R1CS columns, primitive security, costs, or row removal.

Emits constraints: no.

Authority boundary: `parent.yZcol` is fixed before `producerBeta`; the theorem
does not accept a pre-proved old-point relation or parent projection. The
padding premise is a concrete shape obligation and the one-point check remains
explicitly collision-bounded by `ProjectionCheck.BadRoot`.

| Stage path | Mathematical obligation | Excluded boundary |
|---|---|---|
| `nifs.pi_ccs.nc.delayed_projection.active` | decode the 54 active parent coefficients | production row provenance |
| `nifs.pi_ccs.nc.delayed_projection.padding` | complete the ten inactive lanes from explicit zeros | derivation of padding from acceptance |
| `nifs.pi_ccs.nc.delayed_projection.binding` | accepted identity yields the old-point relation or `BadRoot` | production acceptance and transcript derivation |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedParentProjection
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection

/-- Exact active coefficient view of the delayed parent sidecar. -/
def delayedParentActiveCoefficients (parent : DelayedParent) : List K :=
  (List.range ringDegree).map parent.yZcol

theorem delayedParentActiveCoefficients_length (parent : DelayedParent) :
    (delayedParentActiveCoefficients parent).length = ringDegree := by
  simp [delayedParentActiveCoefficients]

private theorem rangeMap_getD_of_lt
    (values : Nat → K) {index : Nat} (indexLt : index < ringDegree) :
    ((List.range ringDegree).map values).getD index K.zero = values index := by
  simp [List.getD_eq_getElem?_getD, indexLt]

/-- An accepted compact delayed identity binds every old-point lane, including
the production padding lanes, or exposes the existing degree-53 bad root.

`parentPadding` is deliberately separate from the one-point identity: the
identity evaluates exactly the 54 active coefficients while production owns
ten explicit zero-padding lanes in its 64-lane carrier. -/
theorem acceptedProjectionIdentity_implies_oldPointRelation_or_badRoot
    (shape : Shape) (radix : F) (parent : DelayedParent)
    (rawChildren : List (List F)) (producerBeta : K)
    (wellShaped : DelayedResidualShape shape parent.sCol rawChildren)
    (parentPadding : ∀ lane,
      ringDegree ≤ lane → lane < shape.laneDomain →
      parent.yZcol lane = K.zero)
    (accepted : Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
      (projectionIdentity shape radix rawChildren parent.sCol
        (delayedParentActiveCoefficients parent) producerBeta)) :
    OldPointSumcheckRelation shape radix parent rawChildren ∨
      Nightstream.SuperNeo.ProjectionCheck.BadRoot projectionOps
        (projectionIdentity shape radix rawChildren parent.sCol
          (delayedParentActiveCoefficients parent) producerBeta) := by
  rcases acceptedProjectionIdentity_implies_exact_or_badRoot
      shape radix rawChildren parent.sCol
      (delayedParentActiveCoefficients parent) producerBeta accepted with
    exact | badRoot
  · left
    refine ⟨wellShaped.oldSLength, wellShaped.childrenFit, ?_⟩
    intro lane laneLt
    by_cases active : lane < ringDegree
    · have coefficientsEqual :
          delayedParentActiveCoefficients parent =
            rawChildProjectionCoefficients
              shape radix rawChildren parent.sCol := by
        simpa [Nightstream.SuperNeo.ProjectionCheck.Identity.Exact,
          projectionIdentity] using exact
      have atLane := congrArg
        (fun coefficients => coefficients.getD lane K.zero)
        coefficientsEqual
      unfold delayedParentActiveCoefficients
        rawChildProjectionCoefficients at atLane
      change
        ((List.range ringDegree).map parent.yZcol).getD lane K.zero =
          ((List.range ringDegree).map fun current =>
            radixWeightedChildProjection
              shape radix rawChildren parent.sCol current).getD lane K.zero
        at atLane
      rw [rangeMap_getD_of_lt parent.yZcol active,
        rangeMap_getD_of_lt
          (fun current =>
            radixWeightedChildProjection
              shape radix rawChildren parent.sCol current) active]
        at atLane
      exact atLane
    · have laneGe : ringDegree ≤ lane := Nat.le_of_not_gt active
      rw [parentPadding lane laneGe laneLt]
      exact (radixWeightedChildProjection_eq_zero_of_ringDegree_le
        shape radix rawChildren parent.sCol lane laneGe).symm
  · exact Or.inr badRoot

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
