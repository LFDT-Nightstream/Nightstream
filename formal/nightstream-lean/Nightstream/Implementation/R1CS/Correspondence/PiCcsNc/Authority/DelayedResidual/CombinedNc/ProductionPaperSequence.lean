import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperNifs
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperChecker
import Nightstream.Implementation.R1CS.Correspondence.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionState

/-!
One-fold delayed production composition into the independent paper transition.

Protocol: SuperNeo Sections 7.3--7.5.
Phase: two adjacent raw-production NIFS executions.
Constraint family: typed semantic composition only; this file emits no rows.

Assurance tier: model-level security partition.

Owns: composition of the minimal previous paper refinement with accumulator
state binding, canonical parent opening, successor raw running commitments,
and successor combined-NC acceptance; and the exact adjacent-step failure
partition.

Does not own: base or terminal chain boundaries, derivation of the three
`Pi_DEC` paper-shape facts from generated rows, concrete accumulator encoding,
Rust refinement, costs, or row removal.

Authority boundary: the successor closes the predecessor's packed projection
from authoritative raw running assignments. Accumulator equality gains
authority only through two recomputed `StateBinds` premises. No child opening,
public child `y_zcol` sidecar, generic output mismatch, or implementation
refinement failure occurs in the result.
-/

namespace Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperSequence

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.Authority
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending

universe uPreviousState uNextState uEncoding uDigest

variable {shape : SemanticShape}
variable {PreviousState : Type uPreviousState}
variable {NextState : Type uNextState}
variable {Encoding : Type uEncoding}
variable {Digest : Type uDigest}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Two adjacent accepted raw-production executions yield the predecessor's
actual-child paper transition or one exact algebraic/binding event. The
previous positive record is built internally and never includes running
authority or child openings. -/
theorem acceptedPair_implies_previousPaperTransition_or_namedFailure
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (stateDigest : Digest)
    (previousContext : FixedActive.Context shape PreviousState
      publicRingColumns publicFits verifierRows)
    (previousData : Data shape)
    (previousCertificate : FixedActive.Certificate previousContext)
    (previousInput : SemanticInput previousContext previousData)
    (previousCanonicalPublicInput : forall child,
      (outputChildren previousContext previousCertificate child).publicInput =
        (FixedActive.PaperProfile.decPublicInputSplit
          (FixedActive.paperProfileOf previousContext)).split
          (derive previousContext previousCertificate).piRlcOutput.publicInput
          child)
    (previousParentEvaluationSize :
      (derive previousContext previousCertificate).piRlcOutput.evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf previousContext)).count
          (derive previousContext
            previousCertificate).piRlcOutput.constraintSystem)
    (previousChildEvaluationSize : forall child,
      (outputChildren previousContext previousCertificate
          child).evaluations.size =
        (FixedActive.PaperProfile.decEvaluationArity
          (FixedActive.paperProfileOf previousContext)).count
          (derive previousContext
            previousCertificate).piRlcOutput.constraintSystem)
    (previousParentBound : DelayedRawChildren.CanonicalParentBinding
      previousContext previousData previousCertificate)
    (previousAccepted : ProductionNifs.Accepted previousContext previousData
      previousCertificate)
    (nextContext : FixedActive.Context shape NextState publicRingColumns
      publicFits verifierRows)
    (nextData : Data shape)
    (nextCertificate : FixedActive.Certificate nextContext)
    (nextCommitments : DelayedRawChildren.RawRunningCommitmentsBound
      nextContext nextData)
    (nextAccepted : ProductionNifs.Accepted nextContext nextData
      nextCertificate)
    (nextParent : Phi81Relation.CEStatement
      (RelationShape shape publicRingColumns publicFits)
      (CommitmentValue verifierRows))
    (nextParentBound : nextContext.runningParent = some nextParent)
    (sameKey : nextContext.key = previousContext.key)
    (previousBinds : StateBinds scheme stateDigest
      (derive previousContext previousCertificate).piRlcOutput
      (outputChildren previousContext previousCertificate)
      (some (DelayedProduction.outgoingPending previousContext
        previousCertificate)))
    (nextBinds : StateBinds scheme stateDigest nextParent
      nextContext.input.running nextContext.pending) :
    FixedActive.PaperProfile.Transition
        (FixedActive.paperProfileOf previousContext) previousContext.input
        (outputChildren previousContext previousCertificate) ∨
      ProductionPiCcs.YRingUnbound previousContext previousData
        previousCertificate ∨
      ProductionPiCcs.BadEvent previousContext previousData
        previousCertificate ∨
      ProductionSequence.ParentOpeningClosureBadEvent previousContext
        previousData previousCertificate nextContext nextData nextCertificate
        nextContext.challengeSetSize ∨
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  rcases
      ProductionPaperNifs.accepted_implies_paperRefinement_or_yRingUnbound_or_badEvent
        noZeroDivisors previousContext previousData previousCertificate
        previousInput previousCanonicalPublicInput
        previousParentEvaluationSize previousChildEvaluationSize
        previousParentBound
        previousAccepted with
    refinement | yRingUnbound | previousBad
  · rcases
        ProductionState.acceptedNext_of_stateBinding_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent
          noZeroDivisors scheme stateDigest previousContext previousData
          previousCertificate previousParentBound previousAccepted.tail nextContext
          nextData nextCertificate nextCommitments nextAccepted nextParent
          nextParentBound sameKey previousBinds nextBinds with
      packed | closureBad | bindingFailure
    · apply Or.inl
      exact refinement.toPaperTransition (by simpa using packed)
    · exact Or.inr (Or.inr (Or.inr (Or.inl closureBad)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr bindingFailure)))
  · exact Or.inr (Or.inl yRingUnbound)
  · exact Or.inr (Or.inr (Or.inl previousBad))

/-- Backward one-fold composition for the compact opening-derived checker.

The successor's packed equation is the backward-induction hypothesis. It
first converts the successor's public message acceptance to raw NC
acceptance. NC truth then proves the computed incoming children, rather than
checking a prover-carried running family. Two executable state-binding checks
derive child/pending continuity, after which the successor raw table closes
the predecessor's packed equation. -/
theorem checkedPair_of_nextPacked_implies_previousPackedAndPaper_or_namedFailure
    [DecidableEq Digest]
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
    (previousAccepted : ProductionPaperNifs.PaperStepAccepted previousCarrier
      previousContext previousCertificate)
    (nextCarrier : FixedActive.CanonicalOpening.SourceInput.Carrier shape
      publicRingColumns publicFits)
    (nextContext : FixedActive.CanonicalOpening.Context shape NextState
      publicRingColumns publicFits verifierRows)
    (nextCertificate : FixedActive.Certificate
      (nextCarrier.install nextContext).full)
    (nextAccepted : ProductionPaperNifs.PaperStepAccepted nextCarrier
      nextContext nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      (nextCarrier.install nextContext).full.covers nextCarrier.data
      (ProductionPiCcs.ncPoint (nextCarrier.install nextContext).full
        nextCertificate).block
      nextCertificate.piCcs.output)
    (sameKey : nextContext.key = previousContext.key)
    (previousStateAccepted :
      ProductionChecker.stateBindingCheck scheme stateDigest
        (derive (previousCarrier.install previousContext).full
          previousCertificate).piRlcOutput
        (outputChildren (previousCarrier.install previousContext).full
          previousCertificate)
        (some (DelayedProduction.outgoingPending
          (previousCarrier.install previousContext).full
          previousCertificate)) = true)
    (nextStateAccepted :
      ProductionChecker.stateBindingCheck scheme stateDigest
        (nextCarrier.opening.parent nextContext.key nextCarrier.system)
        (nextCarrier.install nextContext).full.input.running
        (nextCarrier.install nextContext).full.pending = true) :
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
      ProductionPiCcs.YRingUnbound
        (previousCarrier.install previousContext).full previousCarrier.data
        previousCertificate ∨
      ProductionPiCcs.BadEvent
        (previousCarrier.install previousContext).full previousCarrier.data
        previousCertificate ∨
      ProductionPiCcs.BadEvent
        (nextCarrier.install nextContext).full nextCarrier.data
        nextCertificate ∨
      ProductionSequence.ParentOpeningClosureBadEvent
        (previousCarrier.install previousContext).full previousCarrier.data
        previousCertificate (nextCarrier.install nextContext).full
        nextCarrier.data nextCertificate
        (nextCarrier.install nextContext).full.challengeSetSize ∨
      Nightstream.Protocol.FPrime.AccumulatorBinding.BindingFailure scheme := by
  let previousFull := (previousCarrier.install previousContext).full
  let nextFull := (nextCarrier.install nextContext).full
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
    have nextCommitments : DelayedRawChildren.RawRunningCommitmentsBound
        nextFull nextCarrier.data :=
      DelayedRawChildren.rawRunningCommitmentsBound_of_semanticInput nextFull
        nextCarrier.data (by
          simpa [nextFull] using nextCarrier.semanticInput nextContext)
    have previousBinds : StateBinds scheme stateDigest
        (derive previousFull previousCertificate).piRlcOutput
        (outputChildren previousFull previousCertificate)
        (some (DelayedProduction.outgoingPending previousFull
          previousCertificate)) :=
      (ProductionChecker.stateBindingCheck_eq_true_iff scheme stateDigest
        (derive previousFull previousCertificate).piRlcOutput
        (outputChildren previousFull previousCertificate)
        (some (DelayedProduction.outgoingPending previousFull
          previousCertificate))).1 (by
            simpa [previousFull] using previousStateAccepted)
    have nextBinds : StateBinds scheme stateDigest
        (nextCarrier.opening.parent nextContext.key nextCarrier.system)
        nextFull.input.running nextFull.pending :=
      (ProductionChecker.stateBindingCheck_eq_true_iff scheme stateDigest
        (nextCarrier.opening.parent nextContext.key nextCarrier.system)
        nextFull.input.running nextFull.pending).1 (by
          simpa [nextFull] using nextStateAccepted)
    rcases
        ProductionState.piDecAndRunningStateBinding_implies_continuity_or_failure
          scheme stateDigest previousFull previousCertificate
          previousAccepted.piDecAccepted nextFull nextRunning
          (nextCarrier.opening.parent nextContext.key nextCarrier.system)
          (by rfl)
          previousBinds nextBinds with
      continuity | bindingFailure
    · rcases
          ProductionSequence.acceptedNext_of_parentOpening_implies_previousPackedYZcolBound_or_badEvent
            noZeroDivisors previousFull previousCarrier.data
            previousCertificate previousAccepted.canonicalParent
            previousAccepted.piDecAccepted nextFull nextCarrier.data
            nextCertificate nextCommitments (by simpa [nextFull, previousFull]
              using sameKey) continuity.1 continuity.2 nextRaw with
        previousPacked | closureBad
      · rcases
            ProductionPaperNifs.paperStepAccepted_and_packed_implies_refinement_or_yRingUnbound_or_badEvent
              noZeroDivisors previousCarrier previousContext
              previousCertificate previousAccepted (by
                simpa [previousFull] using previousPacked) with
          refinement | yRingUnbound | previousBad
        · exact Or.inl ⟨previousPacked, refinement.toPaperTransition (by
              simpa [previousFull] using previousPacked)⟩
        · exact Or.inr (Or.inl yRingUnbound)
        · exact Or.inr (Or.inr (Or.inl previousBad))
      · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl closureBad))))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr bindingFailure))))
  · exact Or.inr (Or.inr (Or.inr (Or.inl nextBad)))

end Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc.ProductionPaperSequence
