import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCArtifact
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiRLCFamilySourceRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyReplayArtifact

/-!
Contract: complete source relation for one production PiRLC family phase.

Assurance tier: generated-source and Rust-replay correspondence.

Owns the fixed 146,224-column layout, all 146,114 algebraic source rows, the
cursor-parity Poseidon2 replay rows, and the implication from their joint
satisfaction to `FamilyPhaseRelation`.

Does not own normalized selective lowering, the Rust witness encoder, the
110-family sequence, recursive lifecycle integration, or the terminal zero
residual check.

Emits constraints: 275,114 rows for an even family and 276,314 rows for an
odd family.
-/

set_option autoImplicit false
set_option maxRecDepth 524288
set_option maxHeartbeats 4000000

namespace Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows

open Nightstream.Implementation.Nebula.ProductPiRlcRingCombinationSound
open Nightstream.Implementation.Nebula.ProductionStreamingPiRlcAuthority
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayArtifact
open Nightstream.SuperNeo.Concrete

private abbrev SourceLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.Layout

private abbrev InputFamilyLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.Layout

private abbrev InputPhaseLayout :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.Layout

private abbrev sourceRows :=
  Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.rows

private abbrev replayRowsFor (parity : CursorParity) : List Row :=
  (arm parity).poseidon2Calls.flatMap Poseidon2Call.Call.rows

/-- The 810 input openings reuse the exact arithmetic input columns. -/
def inputPhaseLayout : InputPhaseLayout where
  inputColumn :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout.input
  digitStart := fun source lane =>
    45456 + (source.val * 54 + lane.val) * 122
  zeroDigitStart := 45415
  dColumn := 144276
  kappaColumn := 144277
  outputColumn := fun output => 144278 + output.val
  seededRowStart := 144277

/-- The local commitment outputs feed the residual equations directly. -/
def inputFamilyLayout : InputFamilyLayout where
  phase := inputPhaseLayout
  beforeResidual := fun output => 144386 + output.val
  afterResidual := fun output => 144494 + output.val

/-- One fixed production layout for every source-row family. -/
def layout : SourceLayout where
  algebra :=
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcArtifact.Generated.layout
  input := inputFamilyLayout
  beforeChallenge := fun source lane =>
    144602 + source.val * 54 + lane.val
  afterChallenge := fun source lane =>
    145412 + source.val * 54 + lane.val
  beforeCursor := 146222
  afterCursor := 146223

theorem exact_layout :
    layout.algebra.base = 1675 /\
      layout.input.phase.zeroDigitStart = 45415 /\
      layout.input.phase.dColumn = 144276 /\
      layout.input.phase.kappaColumn = 144277 /\
      layout.input.phase.seededRowStart = 144277 /\
      layout.input.beforeResidual ⟨0, by decide⟩ = 144386 /\
      layout.input.afterResidual ⟨0, by decide⟩ = 144494 /\
      layout.beforeChallenge ⟨0, by decide⟩ ⟨0, by decide⟩ = 144602 /\
      layout.afterChallenge ⟨0, by decide⟩ ⟨0, by decide⟩ = 145412 /\
      layout.beforeCursor = 146222 /\
      layout.afterCursor = 146223 := by
  native_decide

/-- Family zero starts with cursor zero. Every 810-field input replay and
54-field output replay advances the cursor by two, so family parity selects
the physical cursor shape. -/
def parityFor (family : Family) : CursorParity :=
  if ProductPiRlcAlgebraRows.familyOrdinal family % 2 = 0 then
    .even
  else
    .odd

private theorem poseidonCall_rows_length (call : Poseidon2Call.Call) :
    call.rows.length = 600 := by
  simp [Poseidon2Call.Call.rows, Poseidon2Permutation.rows_length,
    Poseidon2Permutation.rowCount]

private theorem callsRows_length (calls : List Poseidon2Call.Call) :
    (calls.flatMap Poseidon2Call.Call.rows).length = calls.length * 600 := by
  induction calls with
  | nil => rfl
  | cons call rest inductionHypothesis =>
      simp only [List.flatMap_cons, List.length_append,
        poseidonCall_rows_length, inductionHypothesis, List.length_cons]
      omega

theorem replayRows_length :
    ∀ parity : CursorParity,
      (replayRowsFor parity).length =
        match parity with
        | .even => 129000
        | .odd => 130200 := by
  intro parity
  rw [callsRows_length]
  cases parity <;> native_decide

/-- Exact joint row order: algebraic source rows, then the complete input and
output Poseidon2 call rows for the selected cursor parity. -/
def rows
    (setup : InputBindingSetup) (family : Family) : List Row :=
  sourceRows setup layout family ++ replayRowsFor (parityFor family)

theorem rows_length
    (setup : InputBindingSetup) (family : Family) :
    (rows setup family).length =
      if ProductPiRlcAlgebraRows.familyOrdinal family % 2 = 0 then
        275114
      else
        276314 := by
  by_cases even : ProductPiRlcAlgebraRows.familyOrdinal family % 2 = 0
  · calc
      (rows setup family).length =
          146114 + (replayRowsFor (parityFor family)).length := by
        rw [rows, List.length_append,
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.rows_length]
      _ = 146114 + (replayRowsFor .even).length := by
        rw [parityFor, if_pos even]
      _ = 275114 := by rw [replayRows_length]; norm_num
      _ = if ProductPiRlcAlgebraRows.familyOrdinal family % 2 = 0 then
            275114 else 276314 := by rw [if_pos even]
  · calc
      (rows setup family).length =
          146114 + (replayRowsFor (parityFor family)).length := by
        rw [rows, List.length_append,
          Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.rows_length]
      _ = 146114 + (replayRowsFor .odd).length := by
        rw [parityFor, if_neg even]
      _ = 276314 := by rw [replayRows_length]; norm_num
      _ = if ProductPiRlcAlgebraRows.familyOrdinal family % 2 = 0 then
            275114 else 276314 := by rw [if_neg even]

/-- The fixed input columns decode to the exact source rings used by the
PiRLC arithmetic. -/
theorem inputsPlaced
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputPhaseRows.InputsPlaced
      layout.input.phase assignment
      (decodedInputs layout.algebra assignment canonical) := by
  intro source lane
  rfl

private def semanticInputColumns : List Nat :=
  (List.ofFn fun source : Source =>
    List.ofFn fun lane : Fin ringDegree => layout.algebra.input source lane).flatten

private def semanticOutputColumns : List Nat :=
  List.ofFn layout.algebra.output

private theorem replay_input_columns (parity : CursorParity) :
    replayColumns parity .input = semanticInputColumns := by
  cases parity <;> native_decide

private theorem replay_output_columns (parity : CursorParity) :
    replayColumns parity .output = semanticOutputColumns := by
  cases parity <;> native_decide

/-- Both generated replay traces read the exact algebra columns, without a
copy row or digest indirection. -/
theorem replayValuesPlaced
    (parity : CursorParity) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP) :
    ReplayValuesPlaced parity assignment
      (decodedInputs layout.algebra assignment canonical)
      (outputRing layout.algebra assignment canonical) := by
  constructor
  · rw [replay_input_columns]
    simp [semanticInputColumns, phaseFields, sourceBlocks, ringFields,
      decodedInputs, inputRing, wireField,
      ProductPiDecLinearCombination.fieldAt]
    rfl
  · rw [replay_output_columns]
    simp [semanticOutputColumns, ringFields, outputRing, wireField,
      ProductPiDecLinearCombination.fieldAt]
    rfl

private theorem source_satisfies
    {setup : InputBindingSetup} {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup family) assignment) :
    Satisfies (sourceRows setup layout family) assignment := by
  intro row member
  exact satisfies row (List.mem_append_left _ member)

private theorem replay_satisfies
    {setup : InputBindingSetup} {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup family) assignment) :
    Satisfies (replayRowsFor (parityFor family)) assignment := by
  intro row member
  exact satisfies row (List.mem_append_right _ member)

private theorem arm_satisfied
    {setup : InputBindingSetup} {family : Family} {assignment : Nat → Nat}
    (satisfies : Satisfies (rows setup family) assignment) :
    (arm (parityFor family)).Satisfied assignment := by
  intro call callMember row rowMember
  apply replay_satisfies satisfies row
  exact List.mem_flatMap.mpr ⟨call, callMember, rowMember⟩

/-- The complete accepted row family implies the exact semantic PiRLC phase.
No Poseidon2 replay equality remains as a premise. -/
theorem rows_sound
    {setup : InputBindingSetup} {family : Family} {assignment : Nat → Nat}
    {before after : FamilyState}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (range : ∀ source lane,
      assignment (layout.algebra.challengeSymbol source lane) < 5)
    (residualsPlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcInputFamilyRows.ResidualsPlaced
        layout.input assignment before.inputResidual after.inputResidual)
    (carryStatePlaced :
      Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCarryRows.StateColumnsPlaced
        (Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.carryLayout
          layout) assignment before after)
    (replayStatesPlaced :
      FamilyStatesPlaced (parityFor family) assignment before after)
    (cursorExact : before.familyCursor =
      ProductPiRlcAlgebraRows.familyOrdinal family)
    (satisfies : Satisfies (rows setup family) assignment) :
    FamilyPhaseRelation setup before after family
      (decodedInputs layout.algebra assignment canonical)
      (outputRing layout.algebra assignment canonical) := by
  have replayExact := family_replays_exact (parityFor family) assignment
    canonical one before after
    (decodedInputs layout.algebra assignment canonical)
    (outputRing layout.algebra assignment canonical)
    replayStatesPlaced (replayValuesPlaced _ assignment canonical)
    (arm_satisfied satisfies)
  exact
    Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilySourceRows.rows_sound
      canonical one range (inputsPlaced assignment canonical) residualsPlaced
      carryStatePlaced cursorExact replayExact.1 replayExact.2
      (source_satisfies satisfies)

end Nightstream.Implementation.Nebula.ProductionStreamingPiRlcFamilyCompleteRows
