import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTraceRawProjection
import Nightstream.Implementation.R1CS.Correspondence.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact.ProductionExecution

/-!
Active trace composition from the fixed production raw-old-block rows.

Assurance tier: model-proved composition over the fixed generated production
rows.  The terminal package below contains the actual Rust placement and no
caller-selected layout, column map, or execution-audit proposition.

The final trace step alone owns the direct projection rows and terminal CE
opening.  Once those rows derive `ProjectionOpeningAccepted`, the existing
one-fold-delay induction handles the single-step terminal boundary, every
recursive predecessor edge, and the explicit no-pending base boundary.
No implementation-refinement failure or generic output-unbound branch is
introduced.

Owns: the fixed-production terminal row package, its conversion to both the
minimal projection anchor and the executable terminal checker, and the final
joint packed/paper trace composition.

Does not own: Rust artifact regeneration or drift testing, other combined-NC
row families, native commitment-key binding, `y_ring` discharge, transcript
primitive internals, costs, or row-removal authority.

Emits constraints: no; proof-only trace composition.

| Stable stage path | Mathematical obligation | Authority class |
|---|---|---|
| `f_prime.pi_ccs_nc.delayed.trace.fixed_terminal_rows` | every fixed generated projection row is satisfied by the transparent production assignment and the same witnesses satisfy terminal CE | checked execution premise |
| `f_prime.pi_ccs_nc.delayed.trace.fixed_terminal_check` | fixed rows derive the executable terminal checker over the same witness family | derived |
| `f_prime.pi_ccs_nc.delayed.trace.fixed_composition` | base plus terminal rows yield all packed and paper steps or only typed parent-opening/paper failures | derived |
-/

namespace Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace

open Nightstream.Protocol
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
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.Protocol.FPrime.ConcretePhi81.ActiveSemantics
open Nightstream.Protocol.FPrime.ConcretePhi81.AccumulatorBinding.DelayedPending
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual
open Nightstream.Implementation.R1CS.PiCcsNc.Authority.DelayedResidual.CombinedNc
open Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.TerminalRawOldBlockProjectionArtifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.CombinedNc.Generated.Execution.RawOldBlockProjectionRowAt
open PackedWitness

universe uOuterKey uAppState uWitness uDigest uTranscriptState uEncoding

variable {OuterKey : Type uOuterKey}
variable {AppState : Type uAppState}
variable {Witness : Type uWitness}
variable {Digest : Type uDigest}
variable {TranscriptState : Type uTranscriptState}
variable {Encoding : Type uEncoding}
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <=
    ProductionDomain.semanticShape.carrierWidth}

namespace Trace

/-- Fixed production terminal rows attached only to the final trace step.
The exact same ordered `finalWitnesses` family feeds both the generated
physical rows and terminal CE.  The only other witness is a typed assignment
for compiler-derived columns; no emitter, map-validity proof, semantic audit,
sidecar, or digest is supplied by the theorem caller. -/
def TerminalRawProjectionRowsChecked
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape ProductionDomain.semanticShape publicRingColumns
          publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness ProductionDomain.semanticShape
        publicRingColumns publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState
        ProductionDomain.semanticShape publicRingColumns publicFits
        verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing) : Prop :=
  match trace with
  | .single step =>
      ∃ (finalWitnesses : Fin productionGlobalParams.k ->
          PackedWitness.Matrix ProductionDomain.semanticShape)
        (internalWitness : Nat -> ProjectionProgram.F),
        (∀ row : Fin totalRows,
          RowHolds
            (physicalAssignment
              (DelayedProduction.outgoingPending
                (ProductionContext.full setup step.input) step.certificate)
              finalWitnesses internalWitness)
            (actualRow productionEmitterLayout row)) ∧
        TerminalCE.Holds
          (ProductionTerminal.TerminalCEBridge.semantics
            (ProductionContext.full setup step.input))
          (ProductionTerminal.TerminalCEBridge.terminalInstance
            (ProductionContext.full setup step.input) step.certificate
            (fun child => PackedWitness.unpack (finalWitnesses child)))
  | .cons _ tail => tail.TerminalRawProjectionRowsChecked

/-- Eliminate the internal row package into the already-audited semantic
terminal anchor.  This theorem is intentionally one-way: no caller can
manufacture physical row satisfaction from a desired projection equation. -/
theorem terminalRawProjectionRowsChecked_implies_terminalRawProjectionChecked
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape ProductionDomain.semanticShape publicRingColumns
          publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness ProductionDomain.semanticShape
        publicRingColumns publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState
        ProductionDomain.semanticShape publicRingColumns publicFits
        verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (rows : trace.TerminalRawProjectionRowsChecked) :
    trace.TerminalRawProjectionChecked := by
  induction trace with
  | single step =>
      rcases rows with ⟨finalWitnesses, internalWitness, physicalRows,
        terminalCE⟩
      exact ⟨fun child => unpack (finalWitnesses child),
        productionRows_projectionOpeningAccepted
          (context := ProductionContext.full setup step.input)
          (certificate := step.certificate)
          (finalWitnesses := finalWitnesses)
          (internalWitness := internalWitness)
          physicalRows terminalCE⟩
  | cons head tail inductionHypothesis =>
      exact inductionHypothesis rows

/-- The fixed generated terminal rows supply the complete executable terminal
checker over definitionally the same ordered raw `WitnessMat` family.  The
row theorem derives the projection equation; terminal CE supplies the
independent child commitment openings and norm bounds. -/
theorem terminalRawProjectionRowsChecked_implies_terminalChecked
    [DecidableEq Digest]
    {scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape ProductionDomain.semanticShape publicRingColumns
          publicFits)
        (CommitmentValue verifierRows)) Encoding Digest}
    {machine :
      Machine OuterKey Digest AppState Witness ProductionDomain.semanticShape
        publicRingColumns publicFits verifierRows 1}
    {setup :
      Setup OuterKey AppState Witness TranscriptState
        ProductionDomain.semanticShape publicRingColumns publicFits
        verifierRows 1}
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (rows : trace.TerminalRawProjectionRowsChecked) :
    trace.TerminalChecked := by
  induction trace with
  | single step =>
      rcases rows with ⟨finalWitnesses, internalWitness, physicalRows,
        terminalCE⟩
      have opening :=
        productionRows_projectionOpeningAccepted
          (context := ProductionContext.full setup step.input)
          (certificate := step.certificate)
          (finalWitnesses := finalWitnesses)
          (internalWitness := internalWitness)
          physicalRows terminalCE
      have projection :
          ProductionTerminal.projectionCheck
              (ProductionContext.full setup step.input) step.certificate
              (fun child => unpack (finalWitnesses child)) = true :=
        (ProductionTerminal.projectionCheck_eq_true_iff
          (ProductionContext.full setup step.input) step.certificate
          (fun child => unpack (finalWitnesses child))).2 opening.projection
      exact ⟨finalWitnesses,
        PackedWitnessProduction.terminalCheck_of_terminalCE_and_projection
          (ProductionContext.canonical setup step.input) step.certificate
          finalWitnesses terminalCE projection⟩
  | cons head tail inductionHypothesis =>
      exact inductionHypothesis rows

/-- Fixed-production one-fold-delay composition from exact terminal rows.  The base,
single-step terminal, recursive predecessor, and final terminal boundaries
are all explicit in `Trace`; the result contains only the established
parent-opening failure tree and no implementation-refinement branch. -/
theorem terminalRawProjectionRows_imply_baseAndAllPacked_or_parentOpeningFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape ProductionDomain.semanticShape publicRingColumns
          publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness ProductionDomain.semanticShape
        publicRingColumns publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState
        ProductionDomain.semanticShape publicRingColumns publicFits
        verifierRows 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (base : trace.headStep.2.input.pending = none)
    (terminal : trace.TerminalRawProjectionRowsChecked) :
    (trace.BaseNc ∧ trace.AllPacked) ∨ trace.ParentOpeningFailure := by
  exact terminalRawProjection_implies_baseAndAllPacked_or_parentOpeningFailure
    noZeroDivisors scheme machine setup trace base
    (terminalRawProjectionRowsChecked_implies_terminalRawProjectionChecked
      trace terminal)

/-- Strong fixed-row production composition.  Exact generated terminal rows
derive the executable terminal checker, after which the existing one-fold
trace theorem jointly preserves every packed `y_zcol` equation and every
successful paper step.  Its only failure branches are the typed parent-opening
tree and the independent paper failure tree. -/
theorem terminalRawProjectionRows_imply_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
    [DecidableEq Digest]
    (noZeroDivisors : NormRange.BaseFieldNoZeroDivisors)
    (scheme : Nightstream.Protocol.FPrime.AccumulatorBinding.Scheme
      (PendingFamilyPayload
        (RelationShape ProductionDomain.semanticShape publicRingColumns
          publicFits)
        (CommitmentValue verifierRows)) Encoding Digest)
    (machine :
      Machine OuterKey Digest AppState Witness ProductionDomain.semanticShape
        publicRingColumns publicFits verifierRows 1)
    (setup :
      Setup OuterKey AppState Witness TranscriptState
        ProductionDomain.semanticShape publicRingColumns publicFits
        verifierRows 1)
    (functionIndex : Fin 1)
    {incoming outgoing : Digest}
    (trace : Trace scheme machine setup incoming outgoing)
    (base : trace.headStep.2.input.pending = none)
    (terminal : trace.TerminalRawProjectionRowsChecked) :
    (trace.BaseNc ∧ trace.AllPacked ∧ trace.AllPaper functionIndex) ∨
      trace.ParentOpeningFailure ∨
      (trace.BaseNc ∧ trace.AllPacked ∧ trace.Failure) := by
  exact
    terminalChecked_implies_baseAllPackedAndAllPaper_or_parentOpeningFailure_or_paperFailure
      noZeroDivisors scheme machine setup functionIndex trace base
      (terminalRawProjectionRowsChecked_implies_terminalChecked trace terminal)

end Trace

end Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiCcsNc.DelayedProjection.ActiveTrace
