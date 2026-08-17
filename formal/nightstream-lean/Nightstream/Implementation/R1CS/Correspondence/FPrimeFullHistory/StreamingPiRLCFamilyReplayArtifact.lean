import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCAuthority
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyReplayExecutionCertificate
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.TranscriptMachineDuplex

/-!
Contract: exact semantic boundary for the production PiRLC family replays.

Assurance tier: Rust-conformant source-row correspondence.

Owns both cursor-parity shapes, the physical input and output call traces,
their exact column executions, and refinement to the overwrite-duplex
`absorbSlice` operation used by `FamilyPhaseRelation`.

Does not own PiRLC arithmetic, the input residual, challenge carry, the family
cursor, selective lowering, or recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 524288

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplay.Artifact
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.PiRlcChallenge.TranscriptMachine
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyReplay
open Nightstream.SuperNeo.Concrete

theorem trace_pins_canonical (parity : CursorParity) (kind : ReplayKind) :
    ConstantPins.ValuesCanonical (trace parity kind).pins := by
  cases kind <;> simp [trace, ConstantPins.ValuesCanonical]

private theorem trace_call_member
    (parity : CursorParity) (kind : ReplayKind) (call : Poseidon2Call.Call)
    (member : call ∈ (trace parity kind).calls) :
    call ∈ (arm parity).poseidon2Calls := by
  cases kind with
  | input =>
      apply List.mem_of_mem_take
      simpa [trace] using member
  | output =>
      apply List.mem_of_mem_drop
      simpa [trace] using member

/-- Satisfaction of all replay rows reconstructs independent acceptance of
either compact input or output trace. -/
theorem trace_accepted
    (parity : CursorParity) (kind : ReplayKind)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (arm parity).Satisfied assignment) :
    (trace parity kind).Accepted assignment := by
  constructor
  · cases kind <;> simp [trace]
  · intro call member
    apply Poseidon2PermutationSound.poseidon2Permutation_renamed_sound
      call.columnMap call.columnMap_zero canonical one
    exact satisfied call (trace_call_member parity kind call member)

/-- Accepted rows refine the complete value-level slice, including Rust's
eager final normalization. -/
theorem execution_refines
    (parity : CursorParity) (kind : ReplayKind)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (arm parity).Satisfied assignment) :
    ColumnReplay.semanticExecuteSlice assignment canonical
        (ColumnReplay.decodeRun assignment canonical (startRun parity kind))
        (operations parity kind) =
      ColumnReplay.decodeRun assignment canonical (resultRun parity kind) := by
  apply ColumnReplay.executeSlice_sound canonical
    (trace_pins_canonical parity kind) one
    (trace_accepted parity kind assignment canonical one satisfied)
  exact execution parity kind

/-- Assignment values in one exact generated replay agree with the
independent overwrite-duplex bulk operation. -/
theorem replay_eq_absorbSlice
    (parity : CursorParity) (kind : ReplayKind)
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : (arm parity).Satisfied assignment) :
    PiRlcChallenge.TranscriptMachineDuplex.toDuplex
        (ColumnReplay.decodeRun assignment canonical
          (resultRun parity kind)).state =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        ((replayColumns parity kind).map assignment)
        (PiRlcChallenge.TranscriptMachineDuplex.toDuplex
          (ColumnReplay.decodeRun assignment canonical
            (startRun parity kind)).state) := by
  calc
    PiRlcChallenge.TranscriptMachineDuplex.toDuplex
        (ColumnReplay.decodeRun assignment canonical
          (resultRun parity kind)).state =
        PiRlcChallenge.TranscriptMachineDuplex.toDuplex
          (ColumnReplay.semanticExecuteSlice assignment canonical
            (ColumnReplay.decodeRun assignment canonical
              (startRun parity kind)) (operations parity kind)).state :=
      congrArg (fun run =>
        PiRlcChallenge.TranscriptMachineDuplex.toDuplex run.state)
        (execution_refines parity kind assignment canonical one satisfied).symm
    _ = Poseidon2Duplex.absorbSlice
          Poseidon2CanonicalConstants.selected
          ((replayColumns parity kind).map assignment)
          (PiRlcChallenge.TranscriptMachineDuplex.toDuplex
            (ColumnReplay.decodeRun assignment canonical
              (startRun parity kind)).state) := by
      simpa [operations] using
        (PiRlcChallenge.TranscriptMachineDuplex.semanticExecuteSlice_external_toDuplex
          assignment canonical
          (ColumnReplay.decodeRun assignment canonical
            (startRun parity kind)) (replayColumns parity kind))
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity kind).map assignment)
          (PiRlcChallenge.TranscriptMachineDuplex.toDuplex
            (ColumnReplay.decodeRun assignment canonical
              (startRun parity kind)).state) := by
      rfl

def stateAt
    (assignment : Nat → Nat) (columns : List Nat)
    (absorbed : Fin (rate + 1)) : BindingState where
  lanes := fun lane => assignment (columns.getD lane.val 0)
  absorbed := absorbed.val

@[simp] theorem decodedRun_eq_stateAt
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (columns : List Nat) (absorbed : Fin (rate + 1)) :
    PiRlcChallenge.TranscriptMachineDuplex.toDuplex
        (ColumnReplay.decodeRun assignment canonical
          (runFor columns absorbed)).state =
      stateAt assignment columns absorbed := by
  rfl

/-- Exact placement of both carried duplex states into the generated replay
columns. The cursor value is part of each equality. -/
structure FamilyStatesPlaced
    (parity : CursorParity) (assignment : Nat → Nat)
    (before after : FamilyState) : Prop where
  inputBefore : before.inputReplay =
    stateAt assignment (beforeColumns parity .input) (beforeAbsorbed parity)
  inputAfter : after.inputReplay =
    stateAt assignment (afterColumns parity .input) (afterAbsorbed parity)
  outputBefore : before.outputReplay =
    stateAt assignment (beforeColumns parity .output) (beforeAbsorbed parity)
  outputAfter : after.outputReplay =
    stateAt assignment (afterColumns parity .output) (afterAbsorbed parity)

/-- The algebraic values read by both replay traces. These equalities are
later derived from the fixed production PiRLC source layout. -/
structure ReplayValuesPlaced
    (parity : CursorParity) (assignment : Nat → Nat)
    (inputs : Source → RingF) (output : RingF) : Prop where
  input : (replayColumns parity .input).map assignment = phaseFields inputs
  output : (replayColumns parity .output).map assignment = ringFields output

/-- Exact input replay equality required by `FamilyTransition`. -/
theorem input_replay_exact
    (parity : CursorParity) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (before after : FamilyState)
    (states : FamilyStatesPlaced parity assignment before after)
    (satisfied : (arm parity).Satisfied assignment) :
    after.inputReplay =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        ((replayColumns parity .input).map assignment)
        before.inputReplay := by
  calc
    after.inputReplay =
        stateAt assignment (afterColumns parity .input)
          (afterAbsorbed parity) := states.inputAfter
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity .input).map assignment)
          (stateAt assignment (beforeColumns parity .input)
            (beforeAbsorbed parity)) := by
      simpa using replay_eq_absorbSlice parity .input assignment canonical one
        satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity .input).map assignment)
          before.inputReplay := by rw [states.inputBefore]

/-- Exact output replay equality required by `FamilyTransition`. -/
theorem output_replay_exact
    (parity : CursorParity) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (before after : FamilyState)
    (states : FamilyStatesPlaced parity assignment before after)
    (satisfied : (arm parity).Satisfied assignment) :
    after.outputReplay =
      Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
        ((replayColumns parity .output).map assignment)
        before.outputReplay := by
  calc
    after.outputReplay =
        stateAt assignment (afterColumns parity .output)
          (afterAbsorbed parity) := states.outputAfter
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity .output).map assignment)
          (stateAt assignment (beforeColumns parity .output)
            (beforeAbsorbed parity)) := by
      simpa using replay_eq_absorbSlice parity .output assignment canonical one
        satisfied
    _ = Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          ((replayColumns parity .output).map assignment)
          before.outputReplay := by rw [states.outputBefore]

/-- Both accepted replay traces give the two exact semantic equalities that
remain outside the 165,554 PiRLC source rows. -/
theorem family_replays_exact
    (parity : CursorParity) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (before after : FamilyState) (inputs : Source → RingF) (output : RingF)
    (states : FamilyStatesPlaced parity assignment before after)
    (values : ReplayValuesPlaced parity assignment inputs output)
    (satisfied : (arm parity).Satisfied assignment) :
    after.inputReplay =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (phaseFields inputs) before.inputReplay ∧
      after.outputReplay =
        Poseidon2Duplex.absorbSlice ProductPoseidon2.constants
          (ringFields output) before.outputReplay := by
  constructor
  · rw [← values.input]
    exact input_replay_exact parity assignment canonical one before after
      states satisfied
  · rw [← values.output]
    exact output_replay_exact parity assignment canonical one before after
      states satisfied

theorem artifact_valid : rawArtifact.Valid := rawArtifact_valid

/-- Both parity arms read the same exact contiguous PiRLC input columns. -/
theorem replayColumns_input_exact (parity : CursorParity) :
    replayColumns parity .input = List.range' 919 918 := by
  cases parity with
  | even => exact evenArm_inputColumns_exact
  | odd => exact oddArm_inputColumns_exact

/-- Both parity arms read the same exact contiguous PiRLC output columns. -/
theorem replayColumns_output_exact (parity : CursorParity) :
    replayColumns parity .output = List.range' 1837 54 := by
  cases parity with
  | even => exact evenArm_outputColumns_exact
  | odd => exact oddArm_outputColumns_exact

/-- The generated replay call counts are certified by bounded structural
leaves and their list decomposition. -/
theorem poseidon2Calls_length (parity : CursorParity) :
    (arm parity).poseidon2Calls.length =
      match parity with
      | .even => 242
      | .odd => 244 := by
  cases parity with
  | even => exact evenArm_poseidon2Calls_length
  | odd => exact oddArm_poseidon2Calls_length

theorem exact_shape :
    rawArtifact.sourceColumns = 165664 /\
      rawArtifact.even.rowCount = 145200 /\
      rawArtifact.even.columnCount = 310880 /\
      rawArtifact.odd.rowCount = 146400 /\
      rawArtifact.odd.columnCount = 312080 /\
      rawArtifact.even.poseidon2Calls.length = 242 /\
      rawArtifact.odd.poseidon2Calls.length = 244 := by
  refine ⟨rfl, rfl, rfl, rfl, rfl, ?_, ?_⟩
  · exact evenArm_poseidon2Calls_length
  · exact oddArm_poseidon2Calls_length

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact
