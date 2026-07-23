import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionNifs
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionStep

/-!
One-fold delayed production composition.

Assurance tier: model-level, pending concrete state/decoder refinement.

Owns: conversion of the successor's raw NC acceptance into the exact
combined-NC premise; application of the adjacent-step raw-child theorem; and
promotion of the previous delayed refinement to the independent semantic
fold.

Does not own: derivation of pending continuity from the recursive accumulator
digest, decoding either `Sources.Data` from final columns, base/terminal chain
boundaries, Rust/R1CS rows, primitive security, costs, or row removal.

Emits constraints: none.

Authority boundary: the successor terminal reads only its raw running
assignments. `pendingContinue` is an explicit temporary boundary in this leaf;
the final production-chain theorem must derive it from exact
children-plus-pending state recomputation or return `BindingFailure`.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `nifs.production.sequence.pending` | successor carries exactly the verifier-computed previous pending value | explicit composition boundary | `acceptedNext_implies_previousSemanticFold_or_badEvent` |
| `nifs.production.sequence.raw_nc` | specialize successor NC acceptance to that pending value | derived | same theorem |
| `nifs.production.sequence.projection` | recover the previous packed output or a named algebraic/binding event | derived/security boundary | `ProductionStep` |
| `nifs.production.sequence.semantic` | fill the delayed output equation and obtain `SemanticFold.Holds` | derived | `DelayedRefinement.toSemanticFold` |
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionSequence

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.OutputClaims.EvaluationHomomorphism.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
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

/-- Exhaustive adjacent-step failure partition. The order matches
`ProductionStep.accepted_next_implies_previous_packedYZcolBound_or_badEvent`
exactly so composition cannot hide a generic output mismatch. -/
def PreviousClosureBadEvent
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (challengeSetSize : Nat) : Prop :=
  LaneSelectorRoot nextContext.covers nextData nextContext.ncCoins ∨
  BlockSelectorRoot nextContext.covers nextData nextContext.ncCoins ∨
  GammaPolynomialRoot nextContext.covers nextData nextContext.ncCoins ∨
  Acceptance.ResidualWeightRoot nextContext.covers nextData
      nextContext.ncCoins (ProductionProjection.productionWeights nextContext)
      nextContext.producerBeta nextContext.batchWeight
      (CombinedNc.ProductionStep.parentProjection previousContext previousCertificate
        nextContext.producerBeta)
      (DelayedProduction.outgoingPending previousContext
        previousCertificate).oldBlock ∨
  (∃ round,
    FixedPhase.BadChallenge ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData
        nextContext.ncCoins (ProductionProjection.productionWeights nextContext)
        nextContext.producerBeta nextContext.batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      ncSumcheckDegreeBound challengeSetSize
      (K.mul nextContext.batchWeight
        (CombinedNc.ProductionStep.parentProjection previousContext previousCertificate
          nextContext.producerBeta))
      (CombinedNc.ProductionPiCcs.ncPoint nextContext
        nextCertificate).coordinates
      nextCertificate.piCcs.nc.toSumCheck round) ∨
  CombinedNc.ProductionStep.ProducerBetaBadRoot previousContext previousCertificate
      nextContext nextData nextContext.producerBeta ∨
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
      (nextContext.input.running child).commitment))

/-- The exact private-state mismatch still requiring concrete production
refinement.  The left side is recomputed from the successor's authoritative
raw running-assignment table; the right side is the predecessor's canonical
source/challenge parent assignment.  This is strictly narrower than an
output-binding failure and contains no public `yZcol` sidecar. -/
def RawParentStateMismatch
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape) : Prop :=
  Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
      (DelayedRawChildren.rawRunningAssignments nextContext nextData) ≠
    PackedYZcol.canonicalParentAssignment previousContext previousData
      previousCertificate

/-- Algebraic failures of the raw-parent-state recursive edge.  The
assignment/state mismatch is deliberately separate so a concrete opening or
state-binding refinement cannot be mistaken for SumCheck soundness. -/
def RawParentStateBadEvent
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (challengeSetSize : Nat) : Prop :=
  LaneSelectorRoot nextContext.covers nextData nextContext.ncCoins ∨
  BlockSelectorRoot nextContext.covers nextData nextContext.ncCoins ∨
  GammaPolynomialRoot nextContext.covers nextData nextContext.ncCoins ∨
  Acceptance.ResidualWeightRoot nextContext.covers nextData
      nextContext.ncCoins (ProductionProjection.productionWeights nextContext)
      nextContext.producerBeta nextContext.batchWeight
      (CombinedNc.ProductionStep.parentProjection previousContext
        previousCertificate nextContext.producerBeta)
      (DelayedProduction.outgoingPending previousContext
        previousCertificate).oldBlock ∨
  (∃ round,
    FixedPhase.BadChallenge ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData
        nextContext.ncCoins (ProductionProjection.productionWeights nextContext)
        nextContext.producerBeta nextContext.batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      ncSumcheckDegreeBound challengeSetSize
      (K.mul nextContext.batchWeight
        (CombinedNc.ProductionStep.parentProjection previousContext
          previousCertificate nextContext.producerBeta))
      (CombinedNc.ProductionPiCcs.ncPoint nextContext
        nextCertificate).coordinates
      nextCertificate.piCcs.nc.toSumCheck round) ∨
  CombinedNc.ProductionStep.ProducerBetaBadRoot previousContext
      previousCertificate nextContext nextData nextContext.producerBeta ∨
  PiRlcSidecar.MixingCollision previousContext.covers
      previousCertificate.piRlcChallenges
      (InputAuthority.productAssignments previousData
        previousContext.alignment)
      (DelayedProduction.outgoingPending previousContext
        previousCertificate).oldBlock
      (PackedYZcol.sourceClaims previousContext previousCertificate)

/-- Exact failures after the predecessor's canonical parent opening is
checked independently.  The former unconstrained private-state mismatch is
replaced by the standard two-openings-of-one-parent-commitment event. -/
def ParentOpeningClosureBadEvent
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (challengeSetSize : Nat) : Prop :=
  LaneSelectorRoot nextContext.covers nextData nextContext.ncCoins ∨
  BlockSelectorRoot nextContext.covers nextData nextContext.ncCoins ∨
  GammaPolynomialRoot nextContext.covers nextData nextContext.ncCoins ∨
  nextContext.batchWeight = K.zero ∨
  (Acceptance.ResidualWeightRoot nextContext.covers nextData
      nextContext.ncCoins (ProductionProjection.productionWeights nextContext)
      nextContext.producerBeta nextContext.batchWeight
      (CombinedNc.ProductionStep.parentProjection previousContext
        previousCertificate nextContext.producerBeta)
      (DelayedProduction.outgoingPending previousContext
        previousCertificate).oldBlock ∧
    nextContext.batchWeight ≠ K.zero) ∨
  (∃ round,
    FixedPhase.BadChallenge ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData
        nextContext.ncCoins (ProductionProjection.productionWeights nextContext)
        nextContext.producerBeta nextContext.batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      ncSumcheckDegreeBound challengeSetSize
      (K.mul nextContext.batchWeight
        (CombinedNc.ProductionStep.parentProjection previousContext
          previousCertificate nextContext.producerBeta))
      (CombinedNc.ProductionPiCcs.ncPoint nextContext
        nextCertificate).coordinates
      nextCertificate.piCcs.nc.toSumCheck round) ∨
  CombinedNc.ProductionStep.ProducerBetaBadRoot previousContext
      previousCertificate nextContext nextData nextContext.producerBeta ∨
  PiRlcSidecar.MixingCollision previousContext.covers
      previousCertificate.piRlcChallenges
      (InputAuthority.productAssignments previousData
        previousContext.alignment)
      (DelayedProduction.outgoingPending previousContext
        previousCertificate).oldBlock
      (PackedYZcol.sourceClaims previousContext previousCertificate) ∨
  Nonempty (PiDEC.ParentOpeningBindingCollision
    (semantics previousContext.key) productionGlobalParams
    (derive previousContext previousCertificate).piRlcOutput.commitment)

/-- A raw-accepted successor closes the predecessor packed output from its
actual running-assignment table, except for the one exact private-parent state
mismatch or a named algebraic event.  The theorem does not require the
predecessor raw acceptance, paper relation, child openings, or any sidecar.

The successor acceptance is the post-extraction raw predicate; finite
backward composition obtains it from successor claims only after the
successor's own packed equation has already been established. -/
theorem acceptedNext_implies_previousPackedYZcolBound_or_rawParentStateMismatch_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (pendingContinue :
      nextContext.pending = some
        (DelayedProduction.outgoingPending previousContext
          previousCertificate))
    (nextAccepted : CombinedNc.ProductionPiCcs.Accepted nextContext nextData
      nextCertificate) :
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      RawParentStateMismatch previousContext previousData previousCertificate
        nextContext nextData ∨
      RawParentStateBadEvent previousContext previousData previousCertificate
        nextContext nextData nextCertificate nextContext.challengeSetSize := by
  by_cases recomposesCanonical :
      Phi81Relation.EvaluationHomomorphism.PiDEC.Raw.recomposeAssignment
          (DelayedRawChildren.rawRunningAssignments nextContext nextData) =
        PackedYZcol.canonicalParentAssignment previousContext previousData
          previousCertificate
  · have combinedAccepted : FixedPhase.Accepted ops.toOps
        (CombinedNc.sumcheckPolynomial nextContext.covers nextData
          nextContext.ncCoins
          (ProductionProjection.productionWeights nextContext)
          nextContext.producerBeta nextContext.batchWeight
          (DelayedProduction.outgoingPending previousContext
            previousCertificate).oldBlock)
        (K.mul nextContext.batchWeight
          (CombinedNc.ProductionStep.parentProjection previousContext
            previousCertificate nextContext.producerBeta))
        (CombinedNc.ProductionPiCcs.ncPoint nextContext
          nextCertificate).coordinates
        nextCertificate.piCcs.nc.toSumCheck := by
      simpa [CombinedNc.ProductionPiCcs.NcAccepted,
        CombinedNc.ProductionPiCcs.rawPolynomial,
        CombinedNc.ProductionPiCcs.rawInitial, pendingContinue,
        CombinedNc.ProductionStep.parentProjection] using nextAccepted.nc
    rcases
        CombinedNc.ProductionStep.accepted_next_of_rawRecomposition_implies_previous_packedYZcolBound_or_badEvent
          noZeroDivisors previousContext previousData previousCertificate
          nextContext nextData recomposesCanonical nextContext.ncCoins
          nextContext.producerBeta nextContext.batchWeight
          (CombinedNc.ProductionPiCcs.ncPoint nextContext nextCertificate)
          nextCertificate.piCcs.nc.toSumCheck nextContext.challengeSetSize
          combinedAccepted with packed | bad
    · exact Or.inl packed
    · exact Or.inr (Or.inr (by
        simpa [RawParentStateBadEvent] using bad))
  · exact Or.inr (Or.inl recomposesCanonical)

/-- One raw-accepted successor closes its predecessor from a separately
checked canonical parent commitment and norm. Pending continuity specializes
the actual combined-NC polynomial; exact child continuity and raw-table
commitment alignment bind the decoded table to that parent or expose a
parent-opening collision.

No predecessor packed equation, child-opening family, or public child
`y_zcol` sidecar occurs among the premises. -/
theorem acceptedNext_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousParentBound : DelayedRawChildren.CanonicalParentBinding
      previousContext previousData previousCertificate)
    (previousPiDecAccepted : PiDEC.Accepted
      (decAlgebra previousContext.key)
      ((derive previousContext previousCertificate).piDecAttempt
        previousCertificate))
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextCommitments : DelayedRawChildren.RawRunningCommitmentsBound
      nextContext nextData)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (pendingContinue :
      nextContext.pending = some
        (DelayedProduction.outgoingPending previousContext
          previousCertificate))
    (nextAccepted : CombinedNc.ProductionPiCcs.Accepted nextContext nextData
      nextCertificate) :
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      ParentOpeningClosureBadEvent previousContext previousData
        previousCertificate nextContext nextData nextCertificate
        nextContext.challengeSetSize := by
  by_cases batchWeightZero : nextContext.batchWeight = K.zero
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inl batchWeightZero
  have combinedAccepted : FixedPhase.Accepted ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData
        nextContext.ncCoins (ProductionProjection.productionWeights nextContext)
        nextContext.producerBeta nextContext.batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      (K.mul nextContext.batchWeight
        (CombinedNc.ProductionStep.parentProjection previousContext
          previousCertificate nextContext.producerBeta))
      (CombinedNc.ProductionPiCcs.ncPoint nextContext
        nextCertificate).coordinates
      nextCertificate.piCcs.nc.toSumCheck := by
    simpa [CombinedNc.ProductionPiCcs.NcAccepted,
      CombinedNc.ProductionPiCcs.rawPolynomial,
      CombinedNc.ProductionPiCcs.rawInitial, pendingContinue,
      CombinedNc.ProductionStep.parentProjection] using nextAccepted.nc
  rcases
    CombinedNc.ProductionStep.accepted_next_of_parentOpening_implies_previous_packedYZcolBound_or_bindingEvent
      noZeroDivisors previousContext previousData previousCertificate
      previousParentBound previousPiDecAccepted nextContext nextData
      nextCommitments sameKey childrenContinue nextContext.ncCoins
      nextContext.producerBeta
      nextContext.batchWeight
      (CombinedNc.ProductionPiCcs.ncPoint nextContext nextCertificate)
      nextCertificate.piCcs.nc.toSumCheck nextContext.challengeSetSize
      combinedAccepted with packed | laneRoot | blockRoot | gammaRoot |
        residualRoot | sumcheckRoot | producerRoot | mixing | binding
  · exact Or.inl packed
  · exact Or.inr <| Or.inl laneRoot
  · exact Or.inr <| Or.inr <| Or.inl blockRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inl gammaRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inl ⟨residualRoot, batchWeightZero⟩
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inr <| Or.inl sumcheckRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inr <| Or.inr <| Or.inl producerRoot
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inr <| Or.inr <| Or.inr <| Or.inl mixing
  · exact Or.inr <| Or.inr <| Or.inr <| Or.inr <| Or.inr <|
      Or.inr <| Or.inr <| Or.inr <| Or.inr binding

/-- A physically accepted successor derives the previous packed output before
the previous claims-level execution is extracted.  The carried pending
equality is not authority here; accumulator-state recomputation discharges it
in the production state leaf. -/
theorem acceptedNext_implies_previousPackedYZcolBound_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousOpenings :
      ChildOpenings previousContext previousData previousCertificate)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextInput : SemanticInput nextContext nextData)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (pendingContinue :
      nextContext.pending = some
        (DelayedProduction.outgoingPending previousContext
          previousCertificate))
    (nextAccepted : CombinedNc.ProductionNifs.Accepted nextContext nextData
      nextCertificate) :
    Terminal.PackedYZcolBoundAtBlock previousContext.covers previousData
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∨
      PreviousClosureBadEvent previousContext previousData
        previousCertificate nextContext nextData nextCertificate
        nextContext.challengeSetSize := by
  have combinedAccepted : FixedPhase.Accepted ops.toOps
      (CombinedNc.sumcheckPolynomial nextContext.covers nextData
        nextContext.ncCoins (ProductionProjection.productionWeights nextContext)
        nextContext.producerBeta nextContext.batchWeight
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock)
      (K.mul nextContext.batchWeight
        (CombinedNc.ProductionStep.parentProjection previousContext
          previousCertificate nextContext.producerBeta))
      (CombinedNc.ProductionPiCcs.ncPoint nextContext
        nextCertificate).coordinates
      nextCertificate.piCcs.nc.toSumCheck := by
    simpa [CombinedNc.ProductionPiCcs.NcAccepted,
      CombinedNc.ProductionPiCcs.rawPolynomial,
      CombinedNc.ProductionPiCcs.rawInitial, pendingContinue,
      CombinedNc.ProductionStep.parentProjection] using nextAccepted.piCcs.nc
  rcases
      CombinedNc.ProductionStep.accepted_next_implies_previous_packedYZcolBound_or_badEvent
        noZeroDivisors previousContext previousData previousCertificate
        nextContext nextData nextInput sameKey childrenContinue
        previousOpenings nextContext.ncCoins
        nextContext.producerBeta nextContext.batchWeight
        (CombinedNc.ProductionPiCcs.ncPoint nextContext nextCertificate)
        nextCertificate.piCcs.nc.toSumCheck nextContext.challengeSetSize
        combinedAccepted with
    packed | bad
  · exact Or.inl packed
  · exact Or.inr bad

/-- A physically accepted successor closes the previous delayed fold or
exposes one exact adjacent-step bad event. The carried pending equality is not
semantic authority here; the accumulator-binding composition discharges it
later. -/
theorem acceptedNext_implies_previousSemanticFold_or_badEvent
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousRefinement :
      CombinedNc.ProductionNifs.DelayedRefinement previousContext previousData
        previousCertificate)
    (nextContext : FixedActive.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextInput : SemanticInput nextContext nextData)
    (sameKey : nextContext.key = previousContext.key)
    (childrenContinue :
      nextContext.input.running =
        outputChildren previousContext previousCertificate)
    (pendingContinue :
      nextContext.pending = some
        (DelayedProduction.outgoingPending previousContext
          previousCertificate))
    (nextAccepted : CombinedNc.ProductionNifs.Accepted nextContext nextData
      nextCertificate) :
    SemanticFold.Holds previousContext previousData
        (derive previousContext previousCertificate).piRlcOutput
        (outputChildren previousContext previousCertificate) ∨
      PreviousClosureBadEvent previousContext previousData
        previousCertificate nextContext nextData nextCertificate
        nextContext.challengeSetSize := by
  rcases acceptedNext_implies_previousPackedYZcolBound_or_badEvent
      noZeroDivisors previousContext previousData previousCertificate
      previousRefinement.children nextContext nextData nextCertificate
      nextInput sameKey childrenContinue pendingContinue nextAccepted with
    packed | bad
  · exact Or.inl (previousRefinement.toSemanticFold (by simpa using packed))
  · exact Or.inr bad

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionSequence
