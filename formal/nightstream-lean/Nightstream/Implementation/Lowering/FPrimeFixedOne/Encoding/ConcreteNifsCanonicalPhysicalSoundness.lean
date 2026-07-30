import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofPhysicalDecode
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsActivatedProgram
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawSemantics
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: derive the selected semantic `nifsVerify` call directly from the
exact activated physical rows.

Owns: physical operand reconstruction, activation removal, deterministic
call evaluation, and output decoding for the canonical selected NIFS call.

Does not own: application-step decoding, terminal semantics, complete F-prime
assembly, Rust, or generated artifacts.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalPhysicalSoundness

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

section SelectedApplication

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    TerminalRelations
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

private abbrev CanonicalApplication :=
  ConcreteNifsCanonicalOperationalProfile.Application
    setup defaultRunning machine terminalRelations terminalChecks
      widths footprints

private noncomputable abbrev FamilyFor
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints) :=
  application.phase4.profile.family Selected

private abbrev FrameFor
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor setup defaultRunning machine terminalRelations terminalChecks
      widths footprints application)
    Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

/-- **Canonical physical NIFS soundness.** A satisfying active occurrence
constructs all semantic operands, computes the selected verifier call, and
decodes its output. No semantic input or decoding equation is a premise. -/
theorem active_soundness
    (application : CanonicalApplication setup defaultRunning machine
      terminalRelations terminalChecks widths footprints)
    (footprint :
      ConcreteNifsActivatedProgram.FootprintAlignment
        application.phase4.profile
        (ConcreteNifsCanonicalOperationalProfile.operational
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints application))
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor setup defaultRunning machine terminalRelations
      terminalChecks widths footprints application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (constantOne : assignment frame.one = 1)
    (activeOne : assignment frame.active = 1)
    (satisfied :
      Satisfies
        (ConcreteNifsActivatedProgram.rows
          application.phase4.profile
          (ConcreteNifsCanonicalOperationalProfile.operational
            setup defaultRunning machine terminalRelations terminalChecks
              widths footprints application)
          frame)
        assignment) :
    ∃ running : Running dimensions verifierRows,
      ∃ fresh : Fresh dimensions verifierRows,
        ∃ proof :
            SelectedProof
              (ConcreteNifsPlain270Profile.Shape dimensions)
              TranscriptState publicRingColumns (publicFits dimensions)
              verifierRows,
          ∃ output : Running dimensions verifierRows,
            frame.operands.Decodes
                (FamilyFor setup defaultRunning machine terminalRelations
                  terminalChecks widths footprints application)
                assignment
                (.cons running (.cons fresh (.cons proof .nil))) ∧
              callEval Selected Call.nifsVerify
                  (.cons running (.cons fresh (.cons proof .nil))) =
                some (.cons output .nil) ∧
              frame.outputs.Decodes
                (FamilyFor setup defaultRunning machine terminalRelations
                  terminalChecks widths footprints application)
                assignment (.cons output .nil) := by
  let profile :=
    ConcreteNifsCanonicalOperationalProfile.operational
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints application
  have activatedRaw :
      RawSatisfies
        (ConcreteNifsActivatedProgram.rawRows
          application.phase4.profile profile frame)
        assignment := by
    exact
      (satisfies_ownRows_iff frame.owner
        (ConcreteNifsActivatedProgram.rawRows
          application.phase4.profile profile frame)
        assignment).mp
        (by
          simpa [ConcreteNifsActivatedProgram.rows] using satisfied)
  have rawSatisfied :
      RawSatisfies
        (ConcreteNifsRawProgram.rawRows
          application.phase4.profile profile frame)
        assignment :=
    ActivatedRawProgram.active_sound frame.active
      (ConcreteNifsRawProgram.rawRows
        application.phase4.profile profile frame)
      (ConcreteNifsActivatedProgram.residuals
        application.phase4.profile profile frame)
      assignment
      (ConcreteNifsActivatedProgram.residuals_length
        application.phase4.profile profile footprint frame).symm
      activeOne activatedRaw
  rcases
      ConcreteNifsCanonicalProofPhysicalDecode.operands_decode_of_rawRows
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints application frame assignment constantOne
          rawSatisfied with
    ⟨running, fresh, proof, decodedInputs⟩
  rcases
      ConcreteNifsRawSemantics.call_result_and_output_of_rawRows
        GoldilocksField.goldilocks_euclidPrime
        application.phase4.profile profile frame assignment
        running fresh proof constantOne decodedInputs rawSatisfied with
    ⟨output, evaluated, decodedOutput⟩
  exact
    ⟨running, fresh, proof, output, decodedInputs, evaluated,
      decodedOutput⟩

end SelectedApplication

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalPhysicalSoundness
