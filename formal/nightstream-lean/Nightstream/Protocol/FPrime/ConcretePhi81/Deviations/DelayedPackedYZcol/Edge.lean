import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.CombinedNc
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Continuity
import Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.PaperStep

/-!
Recursive edge for the one-fold delayed packed-`yZcol` deviation.

Owns: conversion of a successor's already-closed public message to actual
raw combined-NC acceptance; recovery of exact child/pending continuity from
two recomputed accumulator bindings; closure of the predecessor's packed
projection from the successor raw assignments; and promotion of that
predecessor to the independent paper transition.

Does not own: base or terminal boundaries, concrete transcript hashing,
commitment hardness, Rust/R1CS refinement, costs, or rows.

Emits constraints: no.

Authority boundary: no premise states raw-old-block authority, source
projection agreement, output-column binding, child `yZcol`, or generic
implementation refinement.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.delayed.edge.failure` | enumerate only the exact algebraic, opening, and accumulator-binding failure branches | security partition | `Failure` |
| `fprime.delayed.edge.continuity` | successor raw acceptance plus recomputed state bindings recovers exact children and pending state | derived/security boundary | `acceptedPair_of_nextPacked_implies_previousClosed_or_failure` |
| `fprime.delayed.edge.projection` | successor raw assignments close the predecessor packed projection | derived/security boundary | `acceptedPair_of_nextPacked_implies_previousClosed_or_failure` |
| `fprime.delayed.edge.transition` | a closed predecessor projection promotes its opening-derived step to the independent paper transition | derived | `acceptedPair_of_nextPacked_implies_previousClosed_or_failure` |
-/

namespace Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Edge

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
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
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

namespace ProductionPiCcs

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc
  (ncPoint rawInitial rawPolynomial NcAccepted Accepted BadEvent YRingUnbound
    accepted_of_messageAccepted_and_packed ncAccepted_implies_truth_or_badEvent)

end ProductionPiCcs

namespace ProductionStep

export Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.CombinedNc
  (parentProjection ProducerBetaBadRoot
    accepted_next_of_parentOpening_implies_previous_packedYZcolBound_or_bindingEvent)

end ProductionStep

private abbrev ops := ConcreteCarrier.extensionOps

universe uPreviousState uNextState uEncoding uDigest

variable
  {shape : SemanticShape}
  {PreviousState : Type uPreviousState}
  {NextState : Type uNextState}
  {Encoding : Type uEncoding}
  {Digest : Type uDigest}
  {publicRingColumns verifierRows : Nat}
  {publicFits : ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Exact recursive-edge failure partition. Every constructor is either one
algebraic root/collision or one binding failure; there is no generic escape
constructor. -/
inductive Failure
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (nextContext : FixedActive.Context shape NextState publicRingColumns
      publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext) : Prop where
  | previousYRingUnbound
      (failure : ProductionPiCcs.YRingUnbound previousContext previousData
        previousCertificate) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | previousPiCcs
      (failure : ProductionPiCcs.BadEvent previousContext previousData
        previousCertificate) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | nextPiCcs
      (failure : ProductionPiCcs.BadEvent nextContext nextData
        nextCertificate) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | batchWeightZero
      (failure : nextContext.batchWeight = K.zero) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | laneSelectorRoot
      (failure : LaneSelectorRoot nextContext.covers nextData
        nextContext.ncCoins) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | blockSelectorRoot
      (failure : BlockSelectorRoot nextContext.covers nextData
        nextContext.ncCoins) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | gammaPolynomialRoot
      (failure : GammaPolynomialRoot nextContext.covers nextData
        nextContext.ncCoins) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | residualWeightRoot
      (failure :
        Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.ResidualWeightRoot
          nextContext.covers nextData nextContext.ncCoins
          (Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.productionWeights
            nextContext)
          nextContext.producerBeta nextContext.batchWeight
          (ProductionStep.parentProjection previousContext previousCertificate
            nextContext.producerBeta)
          (DelayedProduction.outgoingPending previousContext
            previousCertificate).oldBlock)
      (nonzero : nextContext.batchWeight ≠ K.zero) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | sumcheckCollision
      (failure : ∃ round,
        FixedPhase.BadChallenge ops.toOps
          (Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.sumcheckPolynomial
            nextContext.covers nextData nextContext.ncCoins
            (Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.productionWeights
              nextContext)
            nextContext.producerBeta nextContext.batchWeight
            (DelayedProduction.outgoingPending previousContext
              previousCertificate).oldBlock)
          ncSumcheckDegreeBound nextContext.challengeSetSize
          (K.mul nextContext.batchWeight
            (ProductionStep.parentProjection previousContext
              previousCertificate nextContext.producerBeta))
          (ProductionPiCcs.ncPoint nextContext nextCertificate).coordinates
          nextCertificate.piCcs.nc.toSumCheck round) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | producerBetaBadRoot
      (failure : ProductionStep.ProducerBetaBadRoot previousContext
        previousCertificate nextContext nextData nextContext.producerBeta) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | piRlcMixing
      (failure : PiRlcSidecar.MixingCollision previousContext.covers
        previousCertificate.piRlcChallenges
        (InputAuthority.productAssignments previousData
          previousContext.alignment)
        (DelayedProduction.outgoingPending previousContext
          previousCertificate).oldBlock
        (PackedYZcol.sourceClaims previousContext previousCertificate)) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | parentOpeningBinding
      (failure : Nonempty (PiDEC.ParentOpeningBindingCollision
        (semantics previousContext.key) productionGlobalParams
        (derive previousContext previousCertificate).piRlcOutput.commitment)) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate
  | accumulatorBinding
      (failure : Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure
        scheme) :
      Failure scheme previousContext previousData previousCertificate
        nextContext nextData nextCertificate

/-- One accepted successor closes the predecessor using its authoritative raw
running assignments, or yields one constructor of `Failure`. -/
theorem acceptedPair_of_nextPacked_implies_previousClosed_or_failure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousCarrier :
      FixedActive.CanonicalOpening.SourceInput.Carrier shape
        publicRingColumns publicFits)
    (previousContext : FixedActive.CanonicalOpening.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousCertificate : FixedActive.Certificate
      (previousCarrier.install previousContext).full)
    (previousAccepted : PaperStep.PaperStepAccepted previousCarrier
      previousContext previousCertificate)
    (nextCarrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (nextContext : FixedActive.CanonicalOpening.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextCertificate : FixedActive.Certificate
      (nextCarrier.install nextContext).full)
    (nextAccepted : PaperStep.PaperStepAccepted nextCarrier nextContext
      nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      (nextCarrier.install nextContext).full.covers nextCarrier.data
      (ProductionPiCcs.ncPoint (nextCarrier.install nextContext).full
        nextCertificate).block
      nextCertificate.piCcs.output)
    (sameKey : nextContext.key = previousContext.key)
    (previousBinds : StateBinds scheme stateDigest
      (derive (previousCarrier.install previousContext).full
        previousCertificate).piRlcOutput
      (outputChildren (previousCarrier.install previousContext).full
        previousCertificate)
      (some (DelayedProduction.outgoingPending
        (previousCarrier.install previousContext).full previousCertificate)))
    (nextBinds : StateBinds scheme stateDigest
      (nextCarrier.opening.parent nextContext.key nextCarrier.system)
      (nextCarrier.install nextContext).full.input.running
      (nextCarrier.install nextContext).full.pending) :
    (Terminal.PackedYZcolBoundAtBlock
        (previousCarrier.install previousContext).full.covers
        previousCarrier.data
        (DelayedProduction.outgoingPending
          (previousCarrier.install previousContext).full
          previousCertificate).oldBlock
        previousCertificate.piCcs.output ∧
      FixedActive.PaperProfile.Transition
        (FixedActive.paperProfileOf
          (previousCarrier.install previousContext).full)
        (previousCarrier.install previousContext).full.input
        (outputChildren (previousCarrier.install previousContext).full
          previousCertificate)) ∨
      Failure scheme (previousCarrier.install previousContext).full
        previousCarrier.data previousCertificate
        (nextCarrier.install nextContext).full nextCarrier.data
        nextCertificate := by
  let previousFull := (previousCarrier.install previousContext).full
  let nextFull := (nextCarrier.install nextContext).full
  by_cases batchWeightZero : nextFull.batchWeight = K.zero
  · exact Or.inr (.batchWeightZero batchWeightZero)
  have nextRaw : ProductionPiCcs.Accepted nextFull nextCarrier.data
      nextCertificate :=
    ProductionPiCcs.accepted_of_messageAccepted_and_packed nextFull
      nextCarrier.data nextCertificate nextAccepted.piCcs (by
        simpa [nextFull] using nextPacked)
  rcases ProductionPiCcs.ncAccepted_implies_truth_or_badEvent
      noZeroDivisors nextFull nextCarrier.data nextCertificate nextRaw.nc with
    nextTruth | nextBad
  · have nextRunning : RunningAuthority.Accepted nextFull := by
      simpa [nextFull] using
        (nextCarrier.runningAuthority_of_ncTruth nextContext (by
          change nextCarrier.NcTruth
          exact nextTruth))
    rcases Continuity.of_piDec_and_stateBindings scheme stateDigest
        previousFull previousCertificate previousAccepted.piDecAccepted
        nextFull nextRunning
        (nextCarrier.opening.parent nextContext.key nextCarrier.system)
        (by rfl) (by simpa [previousFull] using previousBinds)
        (by simpa [nextFull] using nextBinds) with
      continuity | bindingFailure
    · have nextCommitments : DelayedRawChildren.RawRunningCommitmentsBound
          nextFull nextCarrier.data :=
        DelayedRawChildren.rawRunningCommitmentsBound_of_semanticInput nextFull
          nextCarrier.data (by
            simpa [nextFull] using nextCarrier.semanticInput nextContext)
      have combinedAccepted : FixedPhase.Accepted ops.toOps
          (Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.sumcheckPolynomial
            nextFull.covers nextCarrier.data nextFull.ncCoins
            (Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.BlockLaneCombinedNc.productionWeights
              nextFull)
            nextFull.producerBeta nextFull.batchWeight
            (DelayedProduction.outgoingPending previousFull
              previousCertificate).oldBlock)
          (K.mul nextFull.batchWeight
            (ProductionStep.parentProjection previousFull previousCertificate
              nextFull.producerBeta))
          (ProductionPiCcs.ncPoint nextFull nextCertificate).coordinates
          nextCertificate.piCcs.nc.toSumCheck := by
        simpa [ProductionPiCcs.NcAccepted,
          ProductionPiCcs.rawPolynomial, ProductionPiCcs.rawInitial,
          continuity.2, ProductionStep.parentProjection] using nextRaw.nc
      rcases
          ProductionStep.accepted_next_of_parentOpening_implies_previous_packedYZcolBound_or_bindingEvent
            noZeroDivisors previousFull previousCarrier.data
            previousCertificate previousAccepted.canonicalParent
            previousAccepted.piDecAccepted nextFull nextCarrier.data
            nextCommitments (by simpa [nextFull, previousFull] using sameKey)
            continuity.1 nextFull.ncCoins nextFull.producerBeta
            nextFull.batchWeight
            (ProductionPiCcs.ncPoint nextFull nextCertificate)
            nextCertificate.piCcs.nc.toSumCheck nextFull.challengeSetSize
            combinedAccepted with
        packed | laneRoot | blockRoot | gammaRoot | residualRoot |
          sumcheckRoot | producerRoot | mixing | parentBinding
      · rcases
          PaperStep.accepted_and_packed_implies_transition_or_yRingUnbound_or_badEvent
            noZeroDivisors previousCarrier previousContext
            previousCertificate previousAccepted (by
              simpa [previousFull] using packed) with
        paper | yRing | previousBad
        · exact Or.inl ⟨packed, paper⟩
        · exact Or.inr (.previousYRingUnbound yRing)
        · exact Or.inr (.previousPiCcs previousBad)
      · exact Or.inr (.laneSelectorRoot laneRoot)
      · exact Or.inr (.blockSelectorRoot blockRoot)
      · exact Or.inr (.gammaPolynomialRoot gammaRoot)
      · exact Or.inr (.residualWeightRoot residualRoot batchWeightZero)
      · exact Or.inr (.sumcheckCollision sumcheckRoot)
      · exact Or.inr (.producerBetaBadRoot producerRoot)
      · exact Or.inr (.piRlcMixing mixing)
      · exact Or.inr (.parentOpeningBinding parentBinding)
    · exact Or.inr (.accumulatorBinding bindingFailure)
  · exact Or.inr (.nextPiCcs nextBad)

end Nightstream.Protocol.FPrime.ConcretePhi81.Deviations.DelayedPackedYZcol.Edge
