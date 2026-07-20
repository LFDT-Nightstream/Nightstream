import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.Acceptance
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionProjection
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority.DelayedProduction

/-!
Contract: close one adjacent production delayed-`yZcol` step from the next
raw combined-NC certificate back to the previous `Pi_CCS` output.

Assurance tier: model-level.

Owns: specialization of combined-NC acceptance to the verifier-computed
previous pending value; derivation of the degree-53 packed projection check;
cross-step recovery of the previous canonical `Pi_RLC` parent from the next
raw running assignments; and the previous packed-`yZcol` output conclusion.

Does not own: transcript sampling or domain separation, carriage of the
pending value in the recursive state, base or terminal boundary closure,
concrete combined-NC rows, commitment-key coordinate alignment, Ajtai
binding, Rust/R1CS refinement, costs, or row-removal permission.

Emits constraints: none.

Authority boundary: the successor polynomial and its terminal read only
`Sources.Data.runningAssignments`. The checked parent scalar is computed
from `DelayedProduction.outgoingPending`; the caller cannot provide a
projection match, a `ProjectionCheck.Accepted` premise, a source-output
binding predicate, or a child `y_zcol` sidecar. Equality with the previous
private parent follows from two genuine openings of each exact continued
child statement, outside the returned indexed binding-collision event.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.pi_ccs.nc.delayed_running.step.acceptance` | next raw combined-NC acceptance fixes the current NC truth and old-parent scalar | checked/derived | `accepted_next_implies_previous_packedYZcolBound_or_badEvent` |
| `nifs.pi_ccs.nc.delayed_running.step.producer` | the scalar equality instantiates the fixed 54-lane degree-53 pair identity | derived/security boundary | `ProducerBetaBadRoot` |
| `nifs.pi_ccs.nc.delayed_running.step.children` | next raw children are the exact previous `Pi_DEC` splits or expose one fresh binding collision | derived/security boundary | `accepted_next_implies_previous_packedYZcolBound_or_badEvent` |
| `nifs.pi_ccs.nc.delayed_running.step.output` | exact raw recomposition discharges the previous packed-`yZcol` output bound, modulo `Pi_RLC` mixing | derived/security boundary | `accepted_next_implies_previous_packedYZcolBound_or_badEvent` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionStep

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane.MixingSoundness
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.Degree
open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual

private abbrev ops := ConcreteCarrier.extensionOps

universe uPreviousState uNextState

variable
  {shape : SemanticShape}
  {PreviousState : Type uPreviousState}
  {NextState : Type uNextState}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- The only parent scalar accepted by this step: evaluation of the complete
previous verifier-computed pending vector at `producerBeta`. -/
def parentProjection
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousCertificate : FixedActive.Certificate previousContext)
    (producerBeta : K) : K :=
  DelayedPackedProjection.projectedValue
    (DelayedProduction.outgoingPending previousContext previousCertificate
      ).parentYZcol
    producerBeta

/-- Exact packed value reconstructed from all fourteen authoritative next
raw running assignments at the previous verifier-computed block point. -/
def rawPackedParent
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape NextState
        publicRingColumns publicFits verifierRows)
    (nextData : Data shape) : RingK :=
  PackedBlockAction.packedYZcol nextContext.covers
    (PiDEC.Raw.recomposeAssignment
      (DelayedRawChildren.rawRunningAssignments nextContext nextData))
    (DelayedProduction.outgoingPending previousContext previousCertificate
      ).oldBlock

/-- The fixed-width producer-projection identity agrees at `producerBeta`
without being exact. The identity contains all 54 active coefficients and
therefore has maximum degree 53. -/
def ProducerBetaBadRoot
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape NextState
        publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (producerBeta : K) : Prop :=
  ProjectionCheck.BadRoot DelayedPackedProjection.projectionOps
    (DelayedPackedProjection.pairIdentity
      (DelayedProduction.outgoingPending previousContext previousCertificate
        ).parentYZcol
      (rawPackedParent previousContext previousCertificate nextContext
        nextData)
      producerBeta)

/-- The combined-NC certificate determines both the next raw NC truth and the
exact 54-lane equality between the carried previous parent and the radix
recomposition of the next raw running table, unless one of the algebraic
events owned before commitment binding occurs.

This is the common acceptance half of both childwise and direct-parent
binding routes.  Its terminal reads `Sources.Data.runningAssignments`; no
child sidecar or semantic projection premise appears. -/
theorem accepted_next_implies_rawProjection_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape NextState
        publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (coins : Mixing.Coins PiCcsDomains.production.nc)
    (producerBeta batchWeight : K)
    (point : Point PiCcsDomains.production.nc)
    (sumcheckCertificate :
      FixedPhase.Certificate K ncSumcheckDegreeBound)
    (challengeSetSize : Nat)
    (accepted : FixedPhase.Accepted ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
        (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      (K.mul batchWeight
        (parentProjection previousContext previousCertificate producerBeta))
      point.coordinates sumcheckCertificate) :
    (Semantics.Nc.Truth nextData ∧
      (DelayedProduction.outgoingPending previousContext
          previousCertificate).parentYZcol =
        rawPackedParent previousContext previousCertificate nextContext
          nextData) ∨
      LaneSelectorRoot nextContext.covers nextData coins ∨
      BlockSelectorRoot nextContext.covers nextData coins ∨
      GammaPolynomialRoot nextContext.covers nextData coins ∨
      Acceptance.ResidualWeightRoot nextContext.covers nextData
        coins (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (parentProjection previousContext previousCertificate producerBeta)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock ∨
      (∃ round,
        FixedPhase.BadChallenge ops.toOps
          (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
            (ProductionProjection.productionWeights nextContext)
            producerBeta batchWeight
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          ncSumcheckDegreeBound challengeSetSize
          (K.mul batchWeight
            (parentProjection previousContext previousCertificate
              producerBeta))
          point.coordinates sumcheckCertificate round) ∨
      ProducerBetaBadRoot previousContext previousCertificate nextContext
        nextData producerBeta := by
  rcases Acceptance.accepted_implies_truth_and_parentProjection_or_badEvent
      noZeroDivisors nextContext.covers nextData coins
      (ProductionProjection.productionWeights nextContext)
      producerBeta batchWeight
      (parentProjection previousContext previousCertificate producerBeta)
      (DelayedProduction.outgoingPending previousContext
        previousCertificate).oldBlock
      point sumcheckCertificate challengeSetSize accepted with
    semantic | laneRoot | blockRoot | gammaRoot | residualRoot |
      sumcheckRoot
  · rcases semantic with ⟨nextNcTruth, parentScalar⟩
    have leftMatches :
        DelayedPackedProjection.PairLeftScalarMatches
          (DelayedProduction.outgoingPending previousContext
            previousCertificate).parentYZcol
          (CombinedNc.authoritativeRunningProjection nextContext.covers
            nextData (ProductionProjection.productionWeights nextContext)
            producerBeta
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          producerBeta := by
      simpa [parentProjection] using parentScalar
    have rightMatches :
        DelayedPackedProjection.PairRightScalarMatches
          (rawPackedParent previousContext previousCertificate nextContext
            nextData)
          (CombinedNc.authoritativeRunningProjection nextContext.covers
            nextData (ProductionProjection.productionWeights nextContext)
            producerBeta
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          producerBeta := by
      exact ProductionProjection.authoritativeRunningProjection_eq_projectedRawRecomposition
        nextContext nextData producerBeta
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
    have pairAccepted :
        DelayedPackedProjection.PairAccepted
          (DelayedProduction.outgoingPending previousContext
            previousCertificate).parentYZcol
          (rawPackedParent previousContext previousCertificate nextContext
            nextData)
          producerBeta :=
      DelayedPackedProjection.pairAccepted_of_scalar_matches
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).parentYZcol
        (rawPackedParent previousContext previousCertificate nextContext
          nextData)
        (CombinedNc.authoritativeRunningProjection nextContext.covers
          nextData (ProductionProjection.productionWeights nextContext)
          producerBeta
          (DelayedProduction.outgoingPending previousContext
            previousCertificate).oldBlock)
        producerBeta leftMatches rightMatches
    rcases DelayedPackedProjection.pairAccepted_implies_exact_or_badRoot
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).parentYZcol
        (rawPackedParent previousContext previousCertificate nextContext
          nextData)
        producerBeta pairAccepted with packedEqual | producerRoot
    · exact Or.inl ⟨nextNcTruth, packedEqual⟩
    · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
        Or.inr producerRoot
  · exact Or.inr <| Or.inl laneRoot
  · exact Or.inr <| Or.inr <| Or.inl blockRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inl gammaRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inl residualRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inl sumcheckRoot

/-- Exact raw-parent-state refinement contract.  If the radix recomposition
of the successor's authoritative raw running table is the predecessor's
canonical private `Pi_RLC` parent assignment, then the accepted combined-NC
certificate closes the predecessor packed output or exposes only the named
algebraic and source-mixing events.

The `recomposesCanonical` equality is intentionally the sole remaining
cross-step private-state obligation.  It is not a projection premise and it
mentions neither a public `yZcol` sidecar nor `ChildOpenings`.  A concrete
production refinement must derive it from its opening/state dataflow or
replace it by an explicit commitment/binding failure partition. -/
theorem accepted_next_of_rawRecomposition_implies_previous_packedYZcolBound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape NextState
        publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (recomposesCanonical :
      PiDEC.Raw.recomposeAssignment
          (DelayedRawChildren.rawRunningAssignments nextContext nextData) =
        PackedYZcol.canonicalParentAssignment previousContext previousData
          previousCertificate)
    (coins : Mixing.Coins PiCcsDomains.production.nc)
    (producerBeta batchWeight : K)
    (point : Point PiCcsDomains.production.nc)
    (sumcheckCertificate :
      FixedPhase.Certificate K ncSumcheckDegreeBound)
    (challengeSetSize : Nat)
    (accepted : FixedPhase.Accepted ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
        (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      (K.mul batchWeight
        (parentProjection previousContext previousCertificate producerBeta))
      point.coordinates sumcheckCertificate) :
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      LaneSelectorRoot nextContext.covers nextData coins ∨
      BlockSelectorRoot nextContext.covers nextData coins ∨
      GammaPolynomialRoot nextContext.covers nextData coins ∨
      Acceptance.ResidualWeightRoot nextContext.covers nextData
        coins (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (parentProjection previousContext previousCertificate producerBeta)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock ∨
      (∃ round,
        FixedPhase.BadChallenge ops.toOps
          (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
            (ProductionProjection.productionWeights nextContext)
            producerBeta batchWeight
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          ncSumcheckDegreeBound challengeSetSize
          (K.mul batchWeight
            (parentProjection previousContext previousCertificate
              producerBeta))
          point.coordinates sumcheckCertificate round) ∨
      ProducerBetaBadRoot previousContext previousCertificate nextContext
        nextData producerBeta ∨
      PiRlcSidecar.MixingCollision previousContext.covers
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        (PackedYZcol.sourceClaims previousContext previousCertificate) := by
  rcases accepted_next_implies_rawProjection_or_badEvent noZeroDivisors
      previousContext previousCertificate nextContext nextData coins
      producerBeta batchWeight point sumcheckCertificate challengeSetSize
      accepted with projection | laneRoot | blockRoot | gammaRoot |
        residualRoot | sumcheckRoot | producerRoot
  · rcases DelayedProduction.packedBound_or_mixingCollision_of_rawRecomposition
        previousContext previousData previousCertificate
        (DelayedRawChildren.rawRunningAssignments nextContext nextData)
        recomposesCanonical projection.2 with packed | mixing
    · exact Or.inl packed
    · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
        Or.inr <| Or.inr mixing
  · exact Or.inr <| Or.inl laneRoot
  · exact Or.inr <| Or.inr <| Or.inl blockRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inl gammaRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inl residualRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inl sumcheckRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inr <| Or.inl producerRoot

/-- Direct-parent production closure.  Unlike the older childwise route,
this theorem does not require the previous `ChildOpenings` family.  Accepted
strict `Pi_DEC` and genuine openings by the same ordered next raw assignments
bind their recomposition to the previous combined parent, or expose one
standard parent-opening collision.

All remaining successful-path data comes from the combined-NC certificate
and the actual raw assignment table. -/
theorem accepted_next_implies_previous_packedYZcolBound_or_parentBindingEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousParentValid : CE.Holds (semantics previousContext.key)
      productionGlobalParams
      (derive previousContext previousCertificate).piRlcOutput
      (PiRLC.combinedWitness (rlcAlgebra previousContext.key)
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment)))
    (previousPiDecAccepted : PiDEC.Accepted
      (decAlgebra previousContext.key)
      ((derive previousContext previousCertificate).piDecAttempt
        previousCertificate))
    (nextContext :
      FixedActive.Context shape NextState
        publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (rawChildrenValid : forall child,
      CE.Holds (semantics nextContext.key) productionGlobalParams
        (nextContext.input.running child)
        (DelayedRawChildren.rawRunningAssignment nextContext nextData child))
    (coins : Mixing.Coins PiCcsDomains.production.nc)
    (producerBeta batchWeight : K)
    (point : Point PiCcsDomains.production.nc)
    (sumcheckCertificate :
      FixedPhase.Certificate K ncSumcheckDegreeBound)
    (challengeSetSize : Nat)
    (accepted : FixedPhase.Accepted ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
        (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      (K.mul batchWeight
        (parentProjection previousContext previousCertificate producerBeta))
      point.coordinates sumcheckCertificate) :
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      LaneSelectorRoot nextContext.covers nextData coins ∨
      BlockSelectorRoot nextContext.covers nextData coins ∨
      GammaPolynomialRoot nextContext.covers nextData coins ∨
      Acceptance.ResidualWeightRoot nextContext.covers nextData
        coins (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (parentProjection previousContext previousCertificate producerBeta)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock ∨
      (∃ round,
        FixedPhase.BadChallenge ops.toOps
          (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
            (ProductionProjection.productionWeights nextContext)
            producerBeta batchWeight
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          ncSumcheckDegreeBound challengeSetSize
          (K.mul batchWeight
            (parentProjection previousContext previousCertificate
              producerBeta))
          point.coordinates sumcheckCertificate round) ∨
      ProducerBetaBadRoot previousContext previousCertificate nextContext
        nextData producerBeta ∨
      PiRlcSidecar.MixingCollision previousContext.covers
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        (PackedYZcol.sourceClaims previousContext previousCertificate) ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics previousContext.key) productionGlobalParams
        (derive previousContext previousCertificate).piRlcOutput.commitment) := by
  rcases accepted_next_implies_rawProjection_or_badEvent noZeroDivisors
      previousContext previousCertificate nextContext nextData coins
      producerBeta batchWeight point sumcheckCertificate challengeSetSize
      accepted with projection | laneRoot | blockRoot | gammaRoot |
        residualRoot | sumcheckRoot | producerRoot
  · rcases projection with ⟨_nextNcTruth, delayedProjection⟩
    rcases
        DelayedRawChildren.rawRunningAssignments_recompose_eq_parent_or_bindingCollision
          previousContext previousData previousCertificate nextContext nextData
          sameKey childrenContinue previousPiDecAccepted previousParentValid
          rawChildrenValid with recomposesCanonical | bindingCollision
    · rcases DelayedProduction.packedBound_or_mixingCollision_of_rawRecomposition
          previousContext previousData previousCertificate
          (DelayedRawChildren.rawRunningAssignments nextContext nextData)
          recomposesCanonical delayedProjection with bound | mixingCollision
      · exact Or.inl bound
      · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
          Or.inr <| Or.inr <| Or.inl mixingCollision
    · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
        Or.inr <| Or.inr <| Or.inr bindingCollision
  · exact Or.inr <| Or.inl laneRoot
  · exact Or.inr <| Or.inr <| Or.inl blockRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inl gammaRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inl residualRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inl sumcheckRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inr <| Or.inl producerRoot

/-- Direct-parent closure from the exact authority available in production.
The accepted combined-NC certificate first yields NC truth over the decoded
successor table.  Exact raw-table commitment alignment and that truth give
the commitment and fresh-norm authority for every raw running assignment.
Strict `Pi_DEC` then binds their radix recomposition to the independently
checked previous parent commitment and norm, or exposes one parent-opening
collision.

Neither the parent nor successor premises contain public-input or evaluation
sidecars. -/
theorem accepted_next_of_parentOpening_implies_previous_packedYZcolBound_or_bindingEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousParentBound : DelayedRawChildren.CanonicalParentBinding
      previousContext previousData previousCertificate)
    (previousPiDecAccepted : PiDEC.Accepted
      (decAlgebra previousContext.key)
      ((derive previousContext previousCertificate).piDecAttempt
        previousCertificate))
    (nextContext :
      FixedActive.Context shape NextState
        publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCommitments : DelayedRawChildren.RawRunningCommitmentsBound
      nextContext nextData)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (coins : Mixing.Coins PiCcsDomains.production.nc)
    (producerBeta batchWeight : K)
    (point : Point PiCcsDomains.production.nc)
    (sumcheckCertificate :
      FixedPhase.Certificate K ncSumcheckDegreeBound)
    (challengeSetSize : Nat)
    (accepted : FixedPhase.Accepted ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
        (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      (K.mul batchWeight
        (parentProjection previousContext previousCertificate producerBeta))
      point.coordinates sumcheckCertificate) :
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      LaneSelectorRoot nextContext.covers nextData coins ∨
      BlockSelectorRoot nextContext.covers nextData coins ∨
      GammaPolynomialRoot nextContext.covers nextData coins ∨
      Acceptance.ResidualWeightRoot nextContext.covers nextData
        coins (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (parentProjection previousContext previousCertificate producerBeta)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock ∨
      (∃ round,
        FixedPhase.BadChallenge ops.toOps
          (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
            (ProductionProjection.productionWeights nextContext)
            producerBeta batchWeight
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          ncSumcheckDegreeBound challengeSetSize
          (K.mul batchWeight
            (parentProjection previousContext previousCertificate
              producerBeta))
          point.coordinates sumcheckCertificate round) ∨
      ProducerBetaBadRoot previousContext previousCertificate nextContext
        nextData producerBeta ∨
      PiRlcSidecar.MixingCollision previousContext.covers
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        (PackedYZcol.sourceClaims previousContext previousCertificate) ∨
      Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics previousContext.key) productionGlobalParams
        (derive previousContext previousCertificate).piRlcOutput.commitment) := by
  rcases accepted_next_implies_rawProjection_or_badEvent noZeroDivisors
      previousContext previousCertificate nextContext nextData coins
      producerBeta batchWeight point sumcheckCertificate challengeSetSize
      accepted with projection | laneRoot | blockRoot | gammaRoot |
        residualRoot | sumcheckRoot | producerRoot
  · rcases projection with ⟨nextNcTruth, delayedProjection⟩
    rcases
        DelayedRawChildren.rawRunningAssignments_recompose_eq_parent_or_bindingCollision_of_ncTruth
          previousContext previousData previousCertificate nextContext nextData
          sameKey childrenContinue previousPiDecAccepted previousParentBound
          nextCommitments nextNcTruth with recomposesCanonical | bindingCollision
    · rcases DelayedProduction.packedBound_or_mixingCollision_of_rawRecomposition
          previousContext previousData previousCertificate
          (DelayedRawChildren.rawRunningAssignments nextContext nextData)
          recomposesCanonical delayedProjection with bound | mixingCollision
      · exact Or.inl bound
      · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
          Or.inr <| Or.inr <| Or.inl mixingCollision
    · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
        Or.inr <| Or.inr <| Or.inr bindingCollision
  · exact Or.inr <| Or.inl laneRoot
  · exact Or.inr <| Or.inr <| Or.inl blockRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inl gammaRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inl residualRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inl sumcheckRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inr <| Or.inl producerRoot

/-- One accepted successor combined-NC certificate binds the previous
packed-`yZcol` output, unless one explicitly named algebraic or commitment
event occurs.

The accepted SumCheck terminal is the raw `CombinedNc.sumcheckPolynomial`;
both the next NC truth and the delayed projection therefore come from the
same authoritative assignment table. Exact child continuity is equality of
the complete typed statement family, not a digest assertion. -/
theorem accepted_next_implies_previous_packedYZcolBound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (previousContext :
      FixedActive.Context shape PreviousState
        publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext :
      FixedActive.Context shape NextState
        publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextInput : SemanticInput nextContext nextData)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (previousOpenings :
      ChildOpenings previousContext previousData previousCertificate)
    (coins : Mixing.Coins PiCcsDomains.production.nc)
    (producerBeta batchWeight : K)
    (point : Point PiCcsDomains.production.nc)
    (sumcheckCertificate :
      FixedPhase.Certificate K ncSumcheckDegreeBound)
    (challengeSetSize : Nat)
    (accepted : FixedPhase.Accepted ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
        (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      (K.mul batchWeight
        (parentProjection previousContext previousCertificate producerBeta))
      point.coordinates sumcheckCertificate) :
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      LaneSelectorRoot nextContext.covers nextData coins ∨
      BlockSelectorRoot nextContext.covers nextData coins ∨
      GammaPolynomialRoot nextContext.covers nextData coins ∨
      Acceptance.ResidualWeightRoot nextContext.covers nextData
        coins (ProductionProjection.productionWeights nextContext)
        producerBeta batchWeight
        (parentProjection previousContext previousCertificate producerBeta)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock ∨
      (∃ round,
        FixedPhase.BadChallenge ops.toOps
          (CombinedNc.sumcheckPolynomial nextContext.covers nextData coins
            (ProductionProjection.productionWeights nextContext)
            producerBeta batchWeight
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          ncSumcheckDegreeBound challengeSetSize
          (K.mul batchWeight
            (parentProjection previousContext previousCertificate
              producerBeta))
          point.coordinates sumcheckCertificate round) ∨
      ProducerBetaBadRoot previousContext previousCertificate nextContext
        nextData producerBeta ∨
      PiRlcSidecar.MixingCollision previousContext.covers
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        (PackedYZcol.sourceClaims previousContext previousCertificate) ∨
      (∃ child, Nonempty
        (Opening.BindingCollision (semantics nextContext.key)
          productionGlobalParams.b
          (nextContext.input.running child).commitment)) := by
  rcases Acceptance.accepted_implies_truth_and_parentProjection_or_badEvent
      noZeroDivisors nextContext.covers nextData coins
      (ProductionProjection.productionWeights nextContext)
      producerBeta batchWeight
      (parentProjection previousContext previousCertificate producerBeta)
      (DelayedProduction.outgoingPending previousContext
        previousCertificate).oldBlock
      point sumcheckCertificate challengeSetSize accepted with
    semantic | laneRoot | blockRoot | gammaRoot | residualRoot |
      sumcheckRoot
  · rcases semantic with ⟨nextNcTruth, parentScalar⟩
    have leftMatches :
        DelayedPackedProjection.PairLeftScalarMatches
          (DelayedProduction.outgoingPending previousContext
            previousCertificate).parentYZcol
          (CombinedNc.authoritativeRunningProjection nextContext.covers
            nextData (ProductionProjection.productionWeights nextContext)
            producerBeta
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          producerBeta := by
      simpa [parentProjection] using parentScalar
    have rightMatches :
        DelayedPackedProjection.PairRightScalarMatches
          (rawPackedParent previousContext previousCertificate nextContext
            nextData)
          (CombinedNc.authoritativeRunningProjection nextContext.covers
            nextData (ProductionProjection.productionWeights nextContext)
            producerBeta
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          producerBeta := by
      exact ProductionProjection.authoritativeRunningProjection_eq_projectedRawRecomposition
        nextContext nextData producerBeta
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
    have pairAccepted :
        DelayedPackedProjection.PairAccepted
          (DelayedProduction.outgoingPending previousContext
            previousCertificate).parentYZcol
          (rawPackedParent previousContext previousCertificate nextContext
            nextData)
          producerBeta :=
      DelayedPackedProjection.pairAccepted_of_scalar_matches
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).parentYZcol
        (rawPackedParent previousContext previousCertificate nextContext
          nextData)
        (CombinedNc.authoritativeRunningProjection nextContext.covers
          nextData (ProductionProjection.productionWeights nextContext)
          producerBeta
          (DelayedProduction.outgoingPending previousContext
            previousCertificate).oldBlock)
        producerBeta leftMatches rightMatches
    rcases DelayedPackedProjection.pairAccepted_implies_exact_or_badRoot
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).parentYZcol
        (rawPackedParent previousContext previousCertificate nextContext
          nextData)
        producerBeta pairAccepted with packedEqual | producerRoot
    · rcases
        DelayedRawChildren.rawRunningAssignments_eq_previousChildren_or_freshBindingCollision
          previousContext previousData previousCertificate nextContext
          nextData nextInput nextNcTruth sameKey childrenContinue
          previousOpenings with childrenEqual | bindingCollision
      · have recomposesPrevious :=
          DelayedRawChildren.rawRunningAssignments_recompose_eq_previousParent
            previousContext previousData previousCertificate nextContext
            nextData childrenEqual
        have recomposesCanonical :
            PiDEC.Raw.recomposeAssignment
                (DelayedRawChildren.rawRunningAssignments nextContext
                  nextData) =
              PackedYZcol.canonicalParentAssignment previousContext
                previousData previousCertificate := by
          simpa [PackedYZcol.canonicalParentAssignment,
            SemanticFold.combinedAssignment, SemanticFold.assignments,
            CertificateRefinement.semanticWitness] using recomposesPrevious
        have delayedProjection :
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).parentYZcol =
              PackedBlockAction.packedYZcol previousContext.covers
                (PiDEC.Raw.recomposeAssignment
                  (DelayedRawChildren.rawRunningAssignments nextContext
                    nextData))
                (DelayedProduction.outgoingPending previousContext
                  previousCertificate).oldBlock := by
          simpa [rawPackedParent] using packedEqual
        rcases DelayedProduction.packedBound_or_mixingCollision_of_rawRecomposition
            previousContext previousData previousCertificate
            (DelayedRawChildren.rawRunningAssignments nextContext nextData)
            recomposesCanonical delayedProjection with bound | mixingCollision
        · exact Or.inl bound
        · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
            Or.inr <| Or.inr <| Or.inl mixingCollision
      · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
          Or.inr <| Or.inr <| Or.inr bindingCollision
    · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
        Or.inr <| Or.inl producerRoot
  · exact Or.inr <| Or.inl laneRoot
  · exact Or.inr <| Or.inr <| Or.inl blockRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inl gammaRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inl residualRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inl sumcheckRoot

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionStep
