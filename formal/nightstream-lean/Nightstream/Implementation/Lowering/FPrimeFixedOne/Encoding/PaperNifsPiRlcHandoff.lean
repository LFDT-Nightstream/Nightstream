import Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptCursor
import Nightstream.Implementation.R1CS.Canonical.KPiCcsTranscriptSemantics
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgram
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerSound
import Nightstream.Implementation.R1CS.Correspondence.PiCcsOutputDigest.ActiveProfile

/-!
Contract: hand the selected paper `Pi_CCS` transcript state to the canonical
fixed-active `Pi_RLC` sampler without resetting its absorption cursor.

Owns: derivation of the exact 19,713-field output serialization from the
independently selected thirteen-matrix relation, proof that its outgoing cursor
is one, and the exact fresh-entry sampler handoff.

Does not own: transcript security, the values of the outgoing lanes, PiRLC row
soundness, a Rust layout, or generated constraints.

Assurance tier: model-level canonical encoding.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

abbrev SelectiveProfile (rows columns : Nat) :=
  Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.Selective.Profile
    rows columns

/-- Paper-joint shape selected by fixed-active arity and the independent
thirteen-role selective relation. -/
abbrev SelectedShape
    {rows columns : Nat}
    (profile : SelectiveProfile rows columns) :=
  (Nightstream.Implementation.R1CS.PiCcsOutputDigest.ActiveProfile.selectiveShape
    profile).paperShape

@[simp] theorem selected_freshCount
    {rows columns : Nat}
    (profile : SelectiveProfile rows columns) :
    (SelectedShape profile).freshCount = 1 := by
  rfl

@[simp] theorem selected_runningCount
    {rows columns : Nat}
    (profile : SelectiveProfile rows columns) :
    (SelectedShape profile).runningCount = 14 := by
  rfl

@[simp] theorem selected_sourceCount
    {rows columns : Nat}
    (profile : SelectiveProfile rows columns) :
    (SelectedShape profile).sourceCount = 15 := by
  rfl

@[simp] theorem selected_matrixCount
    {rows columns : Nat}
    (profile : SelectiveProfile rows columns) :
    (SelectedShape profile).matrixCount = 13 := by
  rfl

@[simp] theorem selected_coefficientCount
    {rows columns : Nat}
    (profile : SelectiveProfile rows columns) :
    (SelectedShape profile).coefficientCount = 54 := by
  rfl

@[simp] theorem selected_carriedEvaluationCount
    {rows columns : Nat}
    (profile : SelectiveProfile rows columns) :
    (SelectedShape profile).carriedEvaluationCount = 9828 := by
  rfl

/-- Exact output transcript width for one fresh source, fourteen running
sources, thirteen matrices, and fifty-four Phi81 coefficients. -/
theorem selected_outputFields_length
    {rows columns degree : Nat}
    (profile : SelectiveProfile rows columns)
    (input : KPiCcsTranscript.Input (SelectedShape profile) degree) :
    (KPiCcsTranscript.outputFields input).length = 19713 := by
  rw [KPiCcsTranscriptCursor.outputFields_length]
  rfl

/-- The selected `Pi_CCS` output leaves the Poseidon2 duplex cursor at one.
This is derived from the Lean-owned semantic relation shape, not a Rust matrix
count or measured trace. -/
theorem selected_afterOutput_absorbed
    {rows columns degree : Nat}
    (profile : SelectiveProfile rows columns)
    (input : KPiCcsTranscript.Input (SelectedShape profile) degree) :
    (KPiCcsTranscript.replay input).afterOutput.absorbed = 1 := by
  rw [KPiCcsTranscriptCursor.replay_afterOutput_absorbed,
    selected_outputFields_length profile input,
    SymbolicDuplexCursor.after_zero_19713]

/-- The canonical sampler preserves the outgoing lanes and cursor while
starting a fresh local entry receipt. -/
theorem samplerInitialBuilder_eq_handoff
    {rows columns degree : Nat}
    (profile : SelectiveProfile rows columns)
    (input : KPiCcsTranscript.Input (SelectedShape profile) degree) :
    PiRlcCanonicalSymbolicMachineHonest.initialBuilder
        (KPiCcsTranscript.replay input).afterOutput.lanes =
      SymbolicDuplex.start
        (KPiCcsTranscript.replay input).afterOutput.lanes
        (KPiCcsTranscript.replay input).afterOutput.absorbed := by
  rw [selected_afterOutput_absorbed profile input]
  rfl

/-- Satisfaction of the selected PiCCS transcript rows determines the exact
value-level state from which the canonical PiRLC sampler starts.  The caller
does not provide an independent transcript-state equality. -/
theorem decoded_samplerInitialBuilder_eq_valueReplay
    {rows columns degree : Nat}
    (profile : SelectiveProfile rows columns)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (input : KPiCcsTranscript.Input (SelectedShape profile) degree)
    (residues :
      ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied :
      Satisfies (KPiCcsTranscript.rows constants input) assignment) :
    SymbolicDuplexSemantics.decodedBuilder assignment
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder
          (KPiCcsTranscript.replay input).afterOutput.lanes) =
      (KPiCcsTranscriptSemantics.valueReplay
        constants assignment input).afterOutput := by
  have replaySemantics :=
    KPiCcsTranscriptSemantics.rows_replay_semantics
      constants assignment input residues constantWire satisfied
  have physicalHandoff :
      SymbolicDuplexSemantics.decodedBuilder assignment
          (PiRlcCanonicalSymbolicMachineHonest.initialBuilder
            (KPiCcsTranscript.replay input).afterOutput.lanes) =
        SymbolicDuplexSemantics.decodedBuilder assignment
          (KPiCcsTranscript.replay input).afterOutput := by
    unfold PiRlcCanonicalSymbolicMachineHonest.initialBuilder
      SymbolicDuplex.start SymbolicDuplexSemantics.decodedBuilder
    rw [selected_afterOutput_absorbed profile input]
  exact physicalHandoff.trans replaySemantics.2.2.2.2

/-! ## Physical transcript composition -/

/-- Exact selected transcript and sampler rows in causal order. -/
def rows
    {rowsCount columns degree : Nat}
    (profile : SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (input : KPiCcsTranscript.Input (SelectedShape profile) degree) :
    List Row :=
  KPiCcsTranscript.rows constants input ++
    PiRlcCanonicalSamplerProgram.rows duplexBase constants
      (KPiCcsTranscript.replay input).afterOutput.lanes

theorem piCcsRows_satisfied
    {rowsCount columns degree : Nat}
    (profile : SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (input : KPiCcsTranscript.Input (SelectedShape profile) degree)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows profile duplexBase constants input) assignment) :
    Satisfies (KPiCcsTranscript.rows constants input) assignment :=
  fun row member => satisfied row (List.mem_append_left _ member)

theorem samplerRows_satisfied
    {rowsCount columns degree : Nat}
    (profile : SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (input : KPiCcsTranscript.Input (SelectedShape profile) degree)
    (assignment : Nat → Nat)
    (satisfied : Satisfies (rows profile duplexBase constants input) assignment) :
    Satisfies
      (PiRlcCanonicalSamplerProgram.rows duplexBase constants
        (KPiCcsTranscript.replay input).afterOutput.lanes)
      assignment :=
  fun row member => satisfied row (List.mem_append_right _ member)

/-- Headline operational handoff: satisfaction of the combined rows both
binds the PiRLC initial state to the value-level PiCCS outgoing transcript and
forces every physical sampler output to the independent first-accepted
semantics. -/
theorem rows_bind_sampler_to_piCcs
    {rowsCount columns degree : Nat}
    (prime : EuclidPrime goldilocksP)
    (profile : SelectiveProfile rowsCount columns)
    (duplexBase : Nat) (constants : Poseidon2Schedule.Constants)
    (input : KPiCcsTranscript.Input (SelectedShape profile) degree)
    (assignment : Nat → Nat)
    (residues : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows profile duplexBase constants input) assignment)
    (coordinate : Fin PiRlcCanonicalSamplerProgram.coordinateCount) :
    let lanes := (KPiCcsTranscript.replay input).afterOutput.lanes
    let initial :=
      PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes
    let samplerSatisfied :=
      samplerRows_satisfied profile duplexBase constants input assignment
        satisfied
    let u64Satisfied :=
      PiRlcCanonicalSamplerProgram.u64Rows_satisfied
        duplexBase constants lanes assignment samplerSatisfied
    SymbolicDuplexSemantics.decodedBuilder assignment initial =
        (KPiCcsTranscriptSemantics.valueReplay
          constants assignment input).afterOutput ∧
      PiRlcCanonicalSamplerSound.physicalOutputValues
          (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
          coordinate assignment =
        (PiRlcCanonicalSamplerSound.semanticOutput
          prime duplexBase
          (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
          (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
          PiRlcCanonicalSamplerProgram.coordinateCount initial
          residues constantWire u64Satisfied coordinate).map
            (fun coefficient =>
              (Nightstream.SuperNeo.Concrete.Phi81StrongSet.embedCoefficient
                coefficient).val) := by
  dsimp only
  let lanes := (KPiCcsTranscript.replay input).afterOutput.lanes
  let samplerSatisfied :=
    samplerRows_satisfied profile duplexBase constants input assignment
      satisfied
  let u64Satisfied :=
    PiRlcCanonicalSamplerProgram.u64Rows_satisfied
      duplexBase constants lanes assignment samplerSatisfied
  let candidateSatisfied :=
    PiRlcCanonicalSamplerProgram.candidateRows_satisfied
      duplexBase constants lanes assignment samplerSatisfied
  let selectorSatisfied :=
    PiRlcCanonicalSamplerProgram.selectorRows_satisfied
      duplexBase constants lanes assignment samplerSatisfied
  refine
    ⟨decoded_samplerInitialBuilder_eq_valueReplay
        profile constants assignment input residues constantWire
        (piCcsRows_satisfied profile duplexBase constants input assignment
          satisfied),
      ?_⟩
  exact
    PiRlcCanonicalSamplerSound.outputs_eq_embeddedFirstAccepted
      prime duplexBase
      (PiRlcCanonicalSamplerProgram.u64Base duplexBase)
      (PiRlcCanonicalSamplerProgram.candidateBase duplexBase)
      (PiRlcCanonicalSamplerProgram.selectorBase duplexBase)
      PiRlcCanonicalSamplerProgram.coordinateCount
      (PiRlcCanonicalSymbolicMachineHonest.initialBuilder lanes)
      residues constantWire u64Satisfied candidateSatisfied selectorSatisfied
      coordinate

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsPiRlcHandoff
