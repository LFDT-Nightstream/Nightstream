import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveBoundary
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ActivePins
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized.ProductionColumnBindings

/-!
Selected combined-NC rows at the active one-fold boundary.

Owns: composition of literal selected-row satisfaction, the generated
source-column bindings, and the remaining executable production checks into
the claims-level active contract; and the adjacent-step packed-`y_zcol`
conclusion under the positive successor induction value.

Does not own: construction of the physical assignment, proof that the runtime
populates the generated columns, the remaining non-NC rows, the terminal
anchor, parent or raw-child openings, commitment binding, `y_ring`, primitive
security, costs, or row removal.  In particular, no producer-carried
`CeClaim.y_zcol`, digest, projection equation, source-row satisfaction, or
semantic acceptance predicate is accepted as authority.

The successor packed premise is the explicit one-fold induction value.  It is
derived from the terminal opening for the last step and then propagated
backward by `ActiveTrace`; this leaf does not claim that one step binds its own
current output.

Emits constraints: none.

Assurance tier: model-level composition.  The selected combined-NC rows are
artifact-checked for the fixed generated profile; the runtime column and
remaining-check bindings below are not yet Rust-conformant.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.selected.active` | selected rows plus exact runtime bindings reconstruct claims acceptance and propagate packed `y_zcol` one predecessor | derived/refinement |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsActiveComposition

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
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveEvaluator
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open PackedWitness

universe uOuterKey uAppState uWitness uDigest uTranscriptState uEncoding

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {Encoding : Type uEncoding}
variable {shape : SemanticShape}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}

/-- Executable checks outside the selected combined-NC row family.

Every field is a Boolean result from the canonical verifier surface.  The
structure deliberately contains neither `NcMessageAccepted` nor any raw or
packed projection proposition: the NC field is reconstructed from selected
rows below, while packed authority arrives only through the delayed successor
or terminal edge. -/
structure RemainingRuntimeChecks
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input)) : Prop where
  outer : FixedOneCanonical.outerCheck machine setup input.fixedOne = true
  running : FixedActive.Canonical.RunningAuthority.check
      (ProductionContext.canonical setup input) = true
  fe : Fe.check
      (ProductionContext.full setup input).feMachine
      (ProductionContext.full setup input).initialState
      (ProductionContext.full setup input).profile
      (ProductionContext.full setup input).piCcsInput
      (ProductionContext.full setup input).feCoins
      certificate.piCcs.output certificate.piCcs.fe = true
  sampler : Sampler.Checker.certificateCheck
      (ProductionContext.full setup input) certificate = true
  piDec : DerivedPiDec.Checker.check
      (ProductionContext.full setup input) certificate = true
  incomingState :
    ProductionChecker.stateBindingCheck scheme incomingStateDigest
        ((ProductionContext.canonical setup input).input.parent.materialize
          (ProductionContext.canonical setup input).input.system)
        (ProductionContext.full setup input).input.running input.pending = true
  outgoingState :
    ProductionChecker.stateBindingCheck scheme outgoingStateDigest
        (derive (ProductionContext.full setup input) certificate).piRlcOutput
        (outputChildren (ProductionContext.full setup input) certificate)
        (some (DelayedProduction.outgoingPending
          (ProductionContext.full setup input) certificate)) = true

/-- Selected combined-NC rows plus exact generated-column bindings fill the
only omitted child of `RemainingRuntimeChecks`, producing the complete active
claims contract.  The proof derives all logical acceptance records from the
corresponding executable checks. -/
theorem generatedEmittedRowsSatisfy_implies_claimsAccepted
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (assignment : Nat -> Nat)
    (selectedRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1)
    (columnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup input) certificate
      (PhysicalAgreement.reconstructedAssignment assignment))
    (remaining : RemainingRuntimeChecks scheme incomingStateDigest
      outgoingStateDigest machine setup input certificate) :
    ActiveBoundary.ClaimsAccepted scheme incomingStateDigest
      outgoingStateDigest machine setup input template witnesses certificate := by
  have consequences : SourceRowsSoundness.Consequences
      (PhysicalAgreement.reconstructedAssignment assignment) :=
    SelectedRowsSoundness.generatedEmittedRowsSatisfy_implies_consequences
      selectedRows selectorOne constantOne
  have reconstructedConstantOne :
      PhysicalAgreement.reconstructedAssignment assignment 0 = 1 :=
    PhysicalAgreement.reconstructed_constantOne constantOne
  have exactDataflow : ProductionMessageAcceptance.ExactDataflow
      (ProductionContext.full setup input) certificate
      (PhysicalAgreement.reconstructedAssignment assignment) :=
    ProductionColumnBindings.columnBindings_imply_exactDataflow
      (ProductionContext.full setup input) certificate
      (PhysicalAgreement.reconstructedAssignment assignment)
      reconstructedConstantOne consequences columnBindings
  have ncAccepted : ProductionPiCcs.NcMessageAccepted
      (ProductionContext.full setup input) certificate :=
    ProductionMessageAcceptance.consequences_imply_ncMessageAccepted
      (ProductionContext.full setup input) certificate
      (PhysicalAgreement.reconstructedAssignment assignment)
      consequences exactDataflow
  have piCcsAccepted : ProductionPiCcs.MessageAccepted
      (ProductionContext.full setup input) certificate := {
    fe := (Fe.check_eq_true_iff_accepted
      (ProductionContext.full setup input).feMachine
      (ProductionContext.full setup input).initialState
      (ProductionContext.full setup input).profile
      (ProductionContext.full setup input).piCcsInput
      (ProductionContext.full setup input).feCoins
      certificate.piCcs.output certificate.piCcs.fe).mp remaining.fe
    nc := ncAccepted
  }
  have messageAccepted : ProductionNifs.MessageAccepted
      (ProductionContext.full setup input) certificate := {
    running :=
      (FixedActive.Canonical.RunningAuthority.check_eq_true_iff_accepted
        (ProductionContext.canonical setup input)).mp remaining.running
    piCcs := piCcsAccepted
    sampler :=
      (Sampler.Checker.certificateCheck_eq_true_iff_accepted
        (ProductionContext.full setup input) certificate).mp remaining.sampler
    tail := {
      sourceStructures := FixedActive.Canonical.Context.sourceStructures
        (ProductionContext.canonical setup input)
      piDecRecomposition :=
        (DerivedPiDec.Checker.check_eq_true_iff_recomposition
          (ProductionContext.full setup input) certificate).mp remaining.piDec
    }
  }
  exact {
    outer := remaining.outer
    nifs := (PackedWitnessProduction.messageCheck_eq_true_iff_accepted
      (ProductionContext.canonical setup input) certificate).mpr
        messageAccepted
    incomingState := remaining.incomingState
    outgoingState := remaining.outgoingState
  }

/-- Stronger active-assignment entry point.  Constant-one and the recursive
selector are consequences of the same-profile encoder pins and exact
selector-total row, rather than caller-supplied equations. -/
theorem generatedEmittedAssignmentSatisfies_implies_claimsAccepted
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (assignment : Nat -> Nat)
    (generatedAssignment :
      ActivePins.GeneratedEmittedAssignmentSatisfies assignment)
    (columnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup input) certificate
      (PhysicalAgreement.reconstructedAssignment assignment))
    (remaining : RemainingRuntimeChecks scheme incomingStateDigest
      outgoingStateDigest machine setup input certificate) :
    ActiveBoundary.ClaimsAccepted scheme incomingStateDigest
      outgoingStateDigest machine setup input template witnesses certificate := by
  rcases ActivePins.generatedEmittedAssignmentSatisfies_implies_pins
      generatedAssignment with ⟨constantOne, selectorOne⟩
  exact generatedEmittedRowsSatisfy_implies_claimsAccepted scheme
    incomingStateDigest outgoingStateDigest machine setup input template
    witnesses certificate assignment generatedAssignment.selectedRows
    selectorOne constantOne columnBindings remaining

/-- Boolean form of the same refinement.  This is the exact field consumed by
`ActiveTrace.Step.checked`, so a concrete runtime bridge need not repackage a
caller-provided logical acceptance object. -/
theorem generatedEmittedRowsSatisfy_implies_claimsCheck
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (assignment : Nat -> Nat)
    (selectedRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy assignment)
    (selectorOne : assignment Metadata.steadySelectorColumn = 1)
    (constantOne : assignment 0 = 1)
    (columnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup input) certificate
      (PhysicalAgreement.reconstructedAssignment assignment))
    (remaining : RemainingRuntimeChecks scheme incomingStateDigest
      outgoingStateDigest machine setup input certificate) :
    ActiveBoundary.claimsCheck scheme incomingStateDigest outgoingStateDigest
      machine setup input certificate = true := by
  apply (ActiveBoundary.claimsCheck_eq_true_iff scheme incomingStateDigest
    outgoingStateDigest machine setup input template witnesses certificate).mpr
  exact generatedEmittedRowsSatisfy_implies_claimsAccepted scheme
    incomingStateDigest outgoingStateDigest machine setup input template
    witnesses certificate assignment selectedRows selectorOne constantOne
    columnBindings remaining

/-- Boolean active-assignment entry point with selector and constant pins
derived from the generated active-row certificate. -/
theorem generatedEmittedAssignmentSatisfies_implies_claimsCheck
    [DecidableEq Digest]
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (incomingStateDigest outgoingStateDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (input :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (template : Data shape)
    (witnesses : Fin shape.runningCount -> Matrix shape)
    (certificate : FixedActive.Certificate
      (ProductionContext.full setup input))
    (assignment : Nat -> Nat)
    (generatedAssignment :
      ActivePins.GeneratedEmittedAssignmentSatisfies assignment)
    (columnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup input) certificate
      (PhysicalAgreement.reconstructedAssignment assignment))
    (remaining : RemainingRuntimeChecks scheme incomingStateDigest
      outgoingStateDigest machine setup input certificate) :
    ActiveBoundary.claimsCheck scheme incomingStateDigest outgoingStateDigest
      machine setup input certificate = true := by
  rcases ActivePins.generatedEmittedAssignmentSatisfies_implies_pins
      generatedAssignment with ⟨constantOne, selectorOne⟩
  exact generatedEmittedRowsSatisfy_implies_claimsCheck scheme
    incomingStateDigest outgoingStateDigest machine setup input template
    witnesses certificate assignment generatedAssignment.selectedRows
    selectorOne constantOne columnBindings remaining

/-- One active delayed edge from literal selected-row satisfaction.

The successor packed equation is a positive terminal- or successor-derived
induction value.  On success, the successor's raw full-witness table closes
the predecessor output.  Every failure is owned by the delayed projection,
parent opening/key alignment, raw-child commitment alignment, SumCheck, or
accumulator binding path; no generic output-unbound or `y_ring` branch occurs
in this y-zcol-only theorem. -/
theorem generatedEmittedRowsPair_of_nextPacked_implies_previousPacked_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAssignment : Nat -> Nat)
    (previousRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy
        previousAssignment)
    (previousSelectorOne :
      previousAssignment Metadata.steadySelectorColumn = 1)
    (previousConstantOne : previousAssignment 0 = 1)
    (previousColumnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup previousInput) previousCertificate
      (PhysicalAgreement.reconstructedAssignment previousAssignment))
    (previousRemaining : RemainingRuntimeChecks scheme
      previousIncomingDigest sharedStateDigest machine setup previousInput
      previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAssignment : Nat -> Nat)
    (nextRows :
      SelectiveArtifactPairs.Artifact.GeneratedEmittedRowsSatisfy
        nextAssignment)
    (nextSelectorOne : nextAssignment Metadata.steadySelectorColumn = 1)
    (nextConstantOne : nextAssignment 0 = 1)
    (nextColumnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup nextInput) nextCertificate
      (PhysicalAgreement.reconstructedAssignment nextAssignment))
    (nextRemaining : RemainingRuntimeChecks scheme sharedStateDigest
      nextOutgoingDigest machine setup nextInput nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      (ProductionContext.full setup nextInput).covers
      (decodedData nextTemplate nextWitnesses)
      (ProductionPiCcs.ncPoint (ProductionContext.full setup nextInput)
        nextCertificate).block nextCertificate.piCcs.output) :
    Terminal.PackedYZcolBoundAtBlock
        (ProductionContext.full setup previousInput).covers
        (decodedData previousTemplate previousWitnesses)
        (ProductionPiCcs.ncPoint (ProductionContext.full setup previousInput)
          previousCertificate).block previousCertificate.piCcs.output ∨
      ProductionBoundary.ParentOpeningRecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses) previousCertificate
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      ActiveBoundary.ParentOpeningActiveBindingFailure setup previousInput
        nextInput previousTemplate nextTemplate previousWitnesses nextWitnesses
        previousCertificate := by
  have previousAccepted :=
    generatedEmittedRowsSatisfy_implies_claimsAccepted scheme
      previousIncomingDigest sharedStateDigest machine setup previousInput
      previousTemplate previousWitnesses previousCertificate
      previousAssignment previousRows previousSelectorOne previousConstantOne
      previousColumnBindings previousRemaining
  have nextAccepted :=
    generatedEmittedRowsSatisfy_implies_claimsAccepted scheme sharedStateDigest
      nextOutgoingDigest machine setup nextInput nextTemplate nextWitnesses
      nextCertificate nextAssignment nextRows nextSelectorOne nextConstantOne
      nextColumnBindings nextRemaining
  exact
    ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPacked_or_parentOpeningBadEvent
      noZeroDivisors scheme previousIncomingDigest sharedStateDigest
      nextOutgoingDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate previousAccepted nextInput
      nextTemplate nextWitnesses nextCertificate nextAccepted nextPacked

/-- Adjacent active delayed edge with both assignments carrying their exact
same-profile active pins.  This wrapper removes all four caller-supplied
constant/selector equations while preserving the lower-level theorem. -/
theorem generatedEmittedAssignmentPair_of_nextPacked_implies_previousPacked_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape shape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness shape publicRingColumns
        publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState shape
        publicRingColumns publicFits verifierRows 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data shape)
    (previousWitnesses : Fin shape.runningCount -> Matrix shape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAssignment : Nat -> Nat)
    (previousGeneratedAssignment :
      ActivePins.GeneratedEmittedAssignmentSatisfies previousAssignment)
    (previousColumnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup previousInput) previousCertificate
      (PhysicalAgreement.reconstructedAssignment previousAssignment))
    (previousRemaining : RemainingRuntimeChecks scheme
      previousIncomingDigest sharedStateDigest machine setup previousInput
      previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness shape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data shape)
    (nextWitnesses : Fin shape.runningCount -> Matrix shape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAssignment : Nat -> Nat)
    (nextGeneratedAssignment :
      ActivePins.GeneratedEmittedAssignmentSatisfies nextAssignment)
    (nextColumnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup nextInput) nextCertificate
      (PhysicalAgreement.reconstructedAssignment nextAssignment))
    (nextRemaining : RemainingRuntimeChecks scheme sharedStateDigest
      nextOutgoingDigest machine setup nextInput nextCertificate)
    (nextPacked : Terminal.PackedYZcolBoundAtBlock
      (ProductionContext.full setup nextInput).covers
      (decodedData nextTemplate nextWitnesses)
      (ProductionPiCcs.ncPoint (ProductionContext.full setup nextInput)
        nextCertificate).block nextCertificate.piCcs.output) :
    Terminal.PackedYZcolBoundAtBlock
        (ProductionContext.full setup previousInput).covers
        (decodedData previousTemplate previousWitnesses)
        (ProductionPiCcs.ncPoint (ProductionContext.full setup previousInput)
          previousCertificate).block previousCertificate.piCcs.output ∨
      ProductionBoundary.ParentOpeningRecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses) previousCertificate
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate ∨
      ActiveBoundary.ParentOpeningActiveBindingFailure setup previousInput
        nextInput previousTemplate nextTemplate previousWitnesses nextWitnesses
        previousCertificate := by
  rcases ActivePins.generatedEmittedAssignmentSatisfies_implies_pins
      previousGeneratedAssignment with
    ⟨previousConstantOne, previousSelectorOne⟩
  rcases ActivePins.generatedEmittedAssignmentSatisfies_implies_pins
      nextGeneratedAssignment with ⟨nextConstantOne, nextSelectorOne⟩
  exact
    generatedEmittedRowsPair_of_nextPacked_implies_previousPacked_or_namedFailure
      noZeroDivisors scheme previousIncomingDigest sharedStateDigest
      nextOutgoingDigest machine setup previousInput previousTemplate
      previousWitnesses previousCertificate previousAssignment
      previousGeneratedAssignment.selectedRows previousSelectorOne
      previousConstantOne previousColumnBindings previousRemaining nextInput
      nextTemplate nextWitnesses nextCertificate nextAssignment
      nextGeneratedAssignment.selectedRows nextSelectorOne nextConstantOne
      nextColumnBindings nextRemaining nextPacked

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsActiveComposition
