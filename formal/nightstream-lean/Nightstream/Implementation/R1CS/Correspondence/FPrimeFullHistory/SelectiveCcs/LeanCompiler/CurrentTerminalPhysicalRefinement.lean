import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentDeployment

/-!
Contract: derive the current fixed-one Terminal relation from the exact
Lean-emitted physical program.

Assurance tier: model-level.

Owns: reconstruction of the complete typed Terminal input from current
physical columns and soundness into the frozen Terminal checker.

Does not own: honest completeness, ownership, cost, Step semantics, Rust,
generated artifacts, or a security reduction.

Emits constraints: no new rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentTerminalPhysicalRefinement

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.State

section

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Nightstream.HyperNova.Construction2.Paper.Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    Nightstream.HyperNova.Construction2.Paper.TerminalRelations
      (Key dimensions TranscriptState verifierRows)
      (Running dimensions verifierRows)
      RunningWitness
      (Fresh dimensions verifierRows)
      FreshWitness 1)
variable
  (terminalChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      terminalRelations)
variable (widths : Widths) (footprints : Footprints)

local notation "Selected" =>
  ConcreteNifsPlain270Profile.selected dimensions
    (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
    defaultRunning machine terminalRelations terminalChecks widths footprints

/-- **Current Terminal CIR-SOUND.** Every satisfying finite assignment of the
exact Lean-emitted Terminal program constructs a decoded statement and proof
accepted by the frozen fixed-one Terminal relation. No semantic input or
decode equation is a premise. -/
theorem deployment_terminal_refines_from_physical_rows
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment :
      Fin
        ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalTerminal.program.toEncoding
              ).columnIds.length →
        F)
    (accepted :
      CurrentCompiler.Accepts
        (ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalTerminal.program.toEncoding
        (deployment_terminal_columns_ge_270 setup defaultRunning machine
          terminalRelations terminalChecks widths footprints deployment)
        assignment) :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    let stable :=
      StableRows.pulledAssignment
        (EncodingRows.columnIndex
          certificate.canonicalTerminal.program.toEncoding) assignment
    ∃ statement :
        Nightstream.HyperNova.Construction2.Paper.TerminalStatement AppState,
      ∃ proof :
          Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
            Selected,
        Columns.Decodes
            (certificate.baseProfile.family Selected)
            (CanonicalContexts.Terminal.input Selected) stable
            (terminalInputValues Selected statement proof) ∧
          Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
            Selected statement proof := by
  dsimp only
  let certificate :=
    ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  let stable :=
    StableRows.pulledAssignment
      (EncodingRows.columnIndex
        certificate.canonicalTerminal.program.toEncoding) assignment
  rcases
      terminalInput_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment stable with
    ⟨statement, proof, decoded⟩
  have decoded' :
      Columns.Decodes
        (certificate.baseProfile.family Selected)
        (CanonicalContexts.Terminal.input Selected) stable
        (terminalInputValues Selected statement proof) := by
    simpa [certificate] using decoded
  refine ⟨statement, proof, decoded', ?_⟩
  exact
    deployment_terminal_cir_sound
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment assignment statement proof accepted
        decoded'

end

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentTerminalPhysicalRefinement
