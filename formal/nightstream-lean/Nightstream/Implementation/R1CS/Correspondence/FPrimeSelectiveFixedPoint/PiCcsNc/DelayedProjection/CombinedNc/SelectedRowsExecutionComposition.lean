import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsActiveComposition
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.PackedWitnessExecutionBinding

/-!
Terminal-anchored selected-row composition for production delayed `y_zcol`.

Assurance tier: model-level composition over an artifact-checked selected-row
slice. Rust conformance remains conditional on the generated incoming
raw-old-block execution artifact and the distinct native terminal check.

Owns: the concrete two-step one-fold-delay composition from literal selected
combined-NC row satisfaction, exact active-column bindings, the remaining
Boolean runtime checks, and the final raw full-witness terminal check to both
adjacent packed `y_zcol` equations.

Does not own: generation/refinement of the incoming raw-old-block audit,
terminal invocation, Ajtai binding, finite-trace induction beyond two
adjacent steps, `y_ring`, costs, or row-removal permission.

Emits constraints: no.

Authority boundary: the outgoing terminal anchor is a separate executable
check over fourteen complete raw matrices.  The generated execution audit
belongs to the incoming pending state and must not be substituted for this
terminal edge. No `CeClaim.y_zcol`, digest, desired packed equation, semantic
acceptance, or generic output-binding proposition is supplied by the caller.

| Stable stage path | Obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.selected.execution.previous` | selected rows derive previous claims acceptance | checked/refinement |
| `f_prime.pi_ccs_nc.selected.execution.next` | selected rows derive successor claims acceptance | checked/refinement |
| `f_prime.pi_ccs_nc.selected.execution.terminal` | the actual raw terminal checker anchors successor packed authority | checked/refinement |
| `f_prime.pi_ccs_nc.selected.execution.delay` | successor anchor closes both the successor and its predecessor | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsExecutionComposition

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Sources
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Polynomial.Nc.BlockLane
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Materialized
open PackedWitness

namespace SelectedRows

export SelectedRowsActiveComposition
  (RemainingRuntimeChecks
    generatedEmittedAssignmentSatisfies_implies_claimsAccepted)

end SelectedRows

private abbrev productionShape := ProductionDomain.semanticShape

universe uOuterKey uAppState uWitness uDigest uTranscriptState uEncoding

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {Encoding : Type uEncoding}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= productionShape.carrierWidth}

/-- Literal selected rows for two adjacent active steps and the actual raw
terminal checker bind both delayed `y_zcol` outputs or expose only the exact
algebraic/commitment events owned by that track.

There is deliberately no packed premise and no semantic/refinement-failure
outcome.  The terminal checker establishes the successor equation; its
accepted combined-NC message then closes the predecessor one fold later. -/
theorem generatedEmittedAssignmentPairAndTerminal_implies_packed_or_namedFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape productionShape publicRingColumns publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (previousIncomingDigest sharedStateDigest nextOutgoingDigest : Digest)
    (machine :
      Machine OuterKey Digest AppState Witness productionShape
        publicRingColumns publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState productionShape
        publicRingColumns publicFits verifierRows 1)
    (previousInput :
      ProductionContext.Input OuterKey AppState Witness productionShape
        publicRingColumns publicFits verifierRows)
    (previousTemplate : Data productionShape)
    (previousWitnesses :
      Fin productionShape.runningCount -> Matrix productionShape)
    (previousCertificate : FixedActive.Certificate
      (ProductionContext.full setup previousInput))
    (previousAssignment : Nat -> Nat)
    (previousGeneratedAssignment :
      ActivePins.GeneratedEmittedAssignmentSatisfies previousAssignment)
    (previousColumnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup previousInput) previousCertificate
      (PhysicalAgreement.reconstructedAssignment previousAssignment))
    (previousRemaining : SelectedRows.RemainingRuntimeChecks scheme
      previousIncomingDigest sharedStateDigest machine setup previousInput
      previousCertificate)
    (nextInput :
      ProductionContext.Input OuterKey AppState Witness productionShape
        publicRingColumns publicFits verifierRows)
    (nextTemplate : Data productionShape)
    (nextWitnesses :
      Fin productionShape.runningCount -> Matrix productionShape)
    (nextCertificate : FixedActive.Certificate
      (ProductionContext.full setup nextInput))
    (nextAssignment : Nat -> Nat)
    (nextGeneratedAssignment :
      ActivePins.GeneratedEmittedAssignmentSatisfies nextAssignment)
    (nextColumnBindings : ProductionColumnBindings.ColumnBindings
      (ProductionContext.full setup nextInput) nextCertificate
      (PhysicalAgreement.reconstructedAssignment nextAssignment))
    (nextRemaining : SelectedRows.RemainingRuntimeChecks scheme
      sharedStateDigest nextOutgoingDigest machine setup nextInput
      nextCertificate)
    (terminalWitnesses :
      Fin productionGlobalParams.k -> Matrix productionShape)
    (terminal : PackedWitnessProduction.terminalCheck
      (ProductionContext.canonical setup nextInput) nextCertificate
      terminalWitnesses = true) :
    (Terminal.PackedYZcolBoundAtBlock
        (ProductionContext.full setup previousInput).covers
        (decodedData previousTemplate previousWitnesses)
        (ProductionPiCcs.ncPoint (ProductionContext.full setup previousInput)
          previousCertificate).block previousCertificate.piCcs.output /\
      Terminal.PackedYZcolBoundAtBlock
        (ProductionContext.full setup nextInput).covers
        (decodedData nextTemplate nextWitnesses)
        (ProductionPiCcs.ncPoint (ProductionContext.full setup nextInput)
          nextCertificate).block nextCertificate.piCcs.output) \/
      ProductionBoundary.ParentOpeningRecursiveBadEvent scheme
        (ProductionContext.full setup previousInput)
        (decodedData previousTemplate previousWitnesses)
        previousCertificate (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate \/
      ActiveBoundary.ParentOpeningActiveBindingFailure setup previousInput
        nextInput previousTemplate nextTemplate previousWitnesses nextWitnesses
        previousCertificate \/
      ProductionBoundary.ParentOpeningTerminalBadEvent
        (ProductionContext.full setup nextInput)
        (decodedData nextTemplate nextWitnesses) nextCertificate := by
  have previousAccepted :=
    SelectedRows.generatedEmittedAssignmentSatisfies_implies_claimsAccepted
      scheme previousIncomingDigest sharedStateDigest machine setup
      previousInput previousTemplate previousWitnesses previousCertificate
      previousAssignment previousGeneratedAssignment previousColumnBindings
      previousRemaining
  have nextAccepted :=
    SelectedRows.generatedEmittedAssignmentSatisfies_implies_claimsAccepted
      scheme sharedStateDigest nextOutgoingDigest machine setup nextInput
      nextTemplate nextWitnesses nextCertificate nextAssignment
      nextGeneratedAssignment nextColumnBindings nextRemaining
  rcases
      ActiveBoundary.claimsAcceptedTerminal_implies_packed_or_parentOpeningBadEvent
        scheme sharedStateDigest nextOutgoingDigest machine setup nextInput
        nextTemplate nextWitnesses nextCertificate nextAccepted
        terminalWitnesses terminal with
    nextPacked | terminalBad
  · rcases
        ActiveBoundary.claimsAcceptedPair_of_nextPacked_implies_previousPacked_or_parentOpeningBadEvent
          noZeroDivisors scheme previousIncomingDigest sharedStateDigest
          nextOutgoingDigest machine setup previousInput previousTemplate
          previousWitnesses previousCertificate previousAccepted nextInput
          nextTemplate nextWitnesses nextCertificate nextAccepted nextPacked with
      previousPacked | recursiveBad | bindingFailure
    · exact Or.inl ⟨previousPacked, nextPacked⟩
    · exact Or.inr (Or.inl recursiveBad)
    · exact Or.inr (Or.inr (Or.inl bindingFailure))
  · exact Or.inr (Or.inr (Or.inr terminalBad))

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.SelectedRowsExecutionComposition
