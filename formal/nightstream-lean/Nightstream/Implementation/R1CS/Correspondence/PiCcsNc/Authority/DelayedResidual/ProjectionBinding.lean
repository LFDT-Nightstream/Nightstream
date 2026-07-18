import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.ProjectionIdentity

/-!
Contract: connect the compact delayed-parent projection identity to the NC
cube sum used by the delayed residual.

Owns: honest completeness of the fixed-width projection identity and the
deterministic exact-or-bad-root transfer from an accepted identity to the
delayed cube sum.

Does not own: transcript sampling, Fiat--Shamir probability, accumulator
binding, production wires, SumCheck rows, or row-removal permission.

Emits constraints: no.

Authority boundary: the parent coefficient vector and raw child assignments
must be fixed independently before `producerBeta` is sampled. Acceptance of a
one-point equality is authority only up to the named projection bad-root
event.

| Stage path | Mathematical obligation | Explicit assumptions | Assurance tier | Permits row removal? |
|---|---|---|---|---|
| `pi_ccs.nc.authority.delayed_projection.completeness` | exact parent/child coefficients imply accepted projection identity | exact fixed-width coefficient equality | model-level | no |
| `pi_ccs.nc.authority.delayed_projection.soundness` | accepted identity makes the delayed cube sum use the claimed parent evaluation, or exposes the same degree-53 bad root | delayed-residual shape and accepted projection | model-level | no |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual

open Nightstream.Implementation.R1CS.PiCcsNc
open Nightstream.Implementation.R1CS.PiCcsNc.MixedPolynomial
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority.DelayedPackedProjection

/-- Honest completeness of the compact projection check. When the claimed
parent coefficients are exactly the independently reconstructed child
projection, the one-point identity accepts. This theorem supplies no
Fiat--Shamir or recursive-state authority. -/
theorem projectionIdentity_accepted_of_exact
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (s : List K) (claimedParentCoefficients : List K)
    (producerBeta : K)
    (claimedExact :
      claimedParentCoefficients =
        rawChildProjectionCoefficients shape radix rawChildren s) :
    Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
      (projectionIdentity shape radix rawChildren s
        claimedParentCoefficients producerBeta) := by
  have claimedLength : claimedParentCoefficients.length = ringDegree := by
    rw [claimedExact]
    simp [rawChildProjectionCoefficients]
  apply Nightstream.SuperNeo.ProjectionCheck.exact_is_accepted
  · exact projectionIdentity_wellFormed shape radix rawChildren s
      claimedParentCoefficients producerBeta claimedLength
  · simpa [Nightstream.SuperNeo.ProjectionCheck.Identity.Exact,
      projectionIdentity] using claimedExact

/-- Deterministic soundness of the compact delayed handle. Once the
transcript-chosen projection identity accepts, the delayed NC cube sum is
computed from the claimed state-parent coefficients, unless that same
challenge is a root of the nonzero parent-minus-child polynomial.

This is the exact model-level bridge needed before lowering the delayed
summand. It does not bound the bad-root probability or identify any claimed
coefficient, challenge, or SumCheck row with production wires. -/
theorem acceptedProjectionIdentity_implies_cubeSum_eq_claimed_or_badRoot
    (shape : Shape) (radix : F) (rawChildren : List (List F))
    (producerBeta batchWeight : K) (oldS : List K)
    (claimedParentCoefficients : List K)
    (wellShaped : DelayedResidualShape shape oldS rawChildren)
    (accepted : Nightstream.SuperNeo.ProjectionCheck.Accepted projectionOps
      (projectionIdentity shape radix rawChildren oldS
        claimedParentCoefficients producerBeta)) :
    delayedResidualCubeSum shape radix rawChildren
        producerBeta batchWeight oldS =
      K.mul batchWeight
        (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
          claimedParentCoefficients producerBeta) ∨
      Nightstream.SuperNeo.ProjectionCheck.BadRoot projectionOps
        (projectionIdentity shape radix rawChildren oldS
          claimedParentCoefficients producerBeta) := by
  rcases acceptedProjectionIdentity_implies_exact_or_badRoot
      shape radix rawChildren oldS claimedParentCoefficients producerBeta
      accepted with exact | badRoot
  · left
    have coefficientsEqual :
        claimedParentCoefficients =
          rawChildProjectionCoefficients shape radix rawChildren oldS := by
      simpa [Nightstream.SuperNeo.ProjectionCheck.Identity.Exact,
        projectionIdentity] using exact
    calc
      delayedResidualCubeSum shape radix rawChildren
          producerBeta batchWeight oldS =
          K.mul batchWeight
            (compactOldPointEvaluation
              shape radix rawChildren producerBeta oldS) :=
        delayedResidualCubeSum_eq_weightedProjectionEvaluation
          shape radix rawChildren producerBeta batchWeight oldS wellShaped
      _ = K.mul batchWeight
          (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
            (rawChildProjectionCoefficients shape radix rawChildren oldS)
            producerBeta) := by
        apply congrArg (K.mul batchWeight)
        exact (projectionIdentity_rhsEvaluation_eq_compact
          shape radix rawChildren oldS claimedParentCoefficients
          producerBeta).symm
      _ = K.mul batchWeight
          (Nightstream.SuperNeo.ProjectionCheck.eval projectionOps
            claimedParentCoefficients producerBeta) := by
        rw [coefficientsEqual]
  · exact Or.inr badRoot

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
