import Nightstream.Implementation.R1CS.Core.Poseidon2Call

/-!
Contract: certifying trace semantics for production Poseidon2 sponge calls.

Each trace round carries only column wiring.  `Round.Valid` checks that the
exact global artifact contains the linear absorb/padding rows and the exact
renamed 600-row permutation call.  The theorem derives the round's functional
result from R1CS satisfaction; no digest value or hash-success fact is carried
in the certificate.
-/

namespace Nightstream.Implementation.R1CS.Poseidon2Sponge

set_option maxRecDepth 65536
set_option maxHeartbeats 5000000

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Call

inductive RoundKind where
  | absorb (chunkColumns : List Nat)
  | pad
deriving DecidableEq, Repr, Inhabited

structure Round where
  kind : RoundKind
  stateBeforeColumns : List Nat
  permutationInputColumns : List Nat
  permutationOutputColumns : List Nat
  definingRows : List Nat
  call : Call
deriving DecidableEq, Repr, Inhabited

def eight : List Nat := List.range 8

def Round.expectedDefinitionRows (round : Round) : List Row :=
  match round.kind with
  | .absorb chunkColumns =>
      (List.range chunkColumns.length).map fun lane =>
        builderLinearRow (round.permutationInputColumns.getD lane 0)
          [ (round.stateBeforeColumns.getD lane 0, 1)
          , (chunkColumns.getD lane 0, 1) ]
  | .pad =>
      [builderLinearRow (round.permutationInputColumns.getD 0 0)
        [(round.stateBeforeColumns.getD 0 0, 1), (0, 1)]]

def Round.metadataValid (round : Round) : Prop :=
  round.stateBeforeColumns.length = 8 ∧
  round.permutationInputColumns.length = 8 ∧
  round.permutationOutputColumns.length = 8 ∧
  round.call.inputColumns = round.permutationInputColumns ∧
  round.permutationOutputColumns =
    eight.map (fun lane => round.call.columnMap (601 + lane)) ∧
  match round.kind with
  | .absorb chunkColumns =>
      chunkColumns.length ≤ 4 ∧
      round.definingRows.length = chunkColumns.length ∧
      round.permutationInputColumns.drop chunkColumns.length =
        round.stateBeforeColumns.drop chunkColumns.length
  | .pad =>
      round.definingRows.length = 1 ∧
      round.permutationInputColumns.drop 1 =
        round.stateBeforeColumns.drop 1

def Round.Valid (round : Round) (programRows : List Row) : Prop :=
  round.metadataValid ∧
  round.call.Matches programRows ∧
  rowsIncluded round.expectedDefinitionRows programRows = true

instance (round : Round) (programRows : List Row) :
    Decidable (round.Valid programRows) := by
  unfold Round.Valid Round.metadataValid
  cases kind : round.kind <;> simp only [kind] <;> infer_instance

def Round.inputLane (assignment : Nat → Nat) (round : Round) (lane : Nat) : Nat :=
  match round.kind with
  | .absorb chunkColumns =>
      if lane < chunkColumns.length then
        (assignment (round.stateBeforeColumns.getD lane 0) +
          assignment (chunkColumns.getD lane 0)) % goldilocksP
      else
        assignment (round.stateBeforeColumns.getD lane 0)
  | .pad =>
      if lane = 0 then
        (assignment (round.stateBeforeColumns.getD 0 0) + 1) % goldilocksP
      else
        assignment (round.stateBeforeColumns.getD lane 0)

private theorem expectedRows_satisfy
    {round : Round} {programRows : List Row} {assignment : Nat → Nat}
    (valid : round.Valid programRows)
    (satisfies : Satisfies programRows assignment) :
    Satisfies round.expectedDefinitionRows assignment := by
  intro row member
  exact satisfies row (rowsIncluded_sound valid.2.2 row member)

private theorem twoTerm_canonical (left right : Nat) :
    CanonicalTerms [(left, 1), (right, 1)] := by
  simp [CanonicalTerms, goldilocksP]

private theorem getD_eq_of_drop_eq
    {left right : List Nat} {first lane : Nat}
    (firstLe : first ≤ lane)
    (dropEq : left.drop first = right.drop first) :
    left.getD lane 0 = right.getD lane 0 := by
  have atOffset := congrArg
    (fun columns : List Nat => (columns[lane - first]?).getD 0) dropEq
  simpa only [List.getElem?_drop, Nat.add_sub_of_le firstLe,
    ← List.getD_eq_getElem?_getD] using atOffset

private theorem output_column_eq
    {round : Round} (valid : round.metadataValid)
    (lane : Nat) (laneLt : lane < 8) :
    round.permutationOutputColumns.getD lane 0 =
      round.call.columnMap (601 + lane) := by
  have atLane := congrArg (fun columns : List Nat => columns.getD lane 0)
    valid.2.2.2.2.1
  simpa [eight, List.getD_eq_getElem?_getD, laneLt] using atLane

private theorem absorb_input_lane
    {round : Round} {chunkColumns : List Nat}
    (kind : round.kind = .absorb chunkColumns)
    {programRows : List Row} {assignment : Nat → Nat}
    (valid : round.Valid programRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment)
    (lane : Nat) (laneLt : lane < 8) :
    assignment (round.permutationInputColumns.getD lane 0) =
      round.inputLane assignment lane := by
  rw [Round.inputLane, kind]
  by_cases absorbed : lane < chunkColumns.length
  · simp only [absorbed, ↓reduceIte]
    have rowMember :
        builderLinearRow (round.permutationInputColumns.getD lane 0)
          [(round.stateBeforeColumns.getD lane 0, 1),
           (chunkColumns.getD lane 0, 1)] ∈
          round.expectedDefinitionRows := by
      rw [Round.expectedDefinitionRows, kind]
      exact List.mem_map.mpr ⟨lane, List.mem_range.mpr absorbed, rfl⟩
    have rowHolds := expectedRows_satisfy valid satisfies _ rowMember
    have defined := builderLinearRow_sound canonical one
      (round.permutationInputColumns.getD lane 0)
      [(round.stateBeforeColumns.getD lane 0, 1),
       (chunkColumns.getD lane 0, 1)]
      (twoTerm_canonical _ _) rowHolds
    simpa [lcEval, List.foldl] using defined
  · simp only [absorbed, ↓reduceIte]
    have kindValid := valid.1.2.2.2.2.2
    rw [kind] at kindValid
    exact congrArg assignment
      (getD_eq_of_drop_eq (Nat.le_of_not_gt absorbed) kindValid.2.2)

private theorem pad_input_lane
    {round : Round} (kind : round.kind = .pad)
    {programRows : List Row} {assignment : Nat → Nat}
    (valid : round.Valid programRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment)
    (lane : Nat) (laneLt : lane < 8) :
    assignment (round.permutationInputColumns.getD lane 0) =
      round.inputLane assignment lane := by
  rw [Round.inputLane, kind]
  by_cases isZero : lane = 0
  · subst lane
    simp only [↓reduceIte]
    have rowMember :
        builderLinearRow (round.permutationInputColumns.getD 0 0)
          [(round.stateBeforeColumns.getD 0 0, 1), (0, 1)] ∈
          round.expectedDefinitionRows := by
      simp [Round.expectedDefinitionRows, kind]
    have rowHolds := expectedRows_satisfy valid satisfies _ rowMember
    have defined := builderLinearRow_sound canonical one
      (round.permutationInputColumns.getD 0 0)
      [(round.stateBeforeColumns.getD 0 0, 1), (0, 1)]
      (twoTerm_canonical _ _) rowHolds
    simpa [lcEval, List.foldl, one] using defined
  · simp only [isZero, ↓reduceIte]
    have kindValid := valid.1.2.2.2.2.2
    rw [kind] at kindValid
    exact congrArg assignment
      (getD_eq_of_drop_eq (by omega) kindValid.2)

theorem input_lane_sound
    {round : Round} {programRows : List Row} {assignment : Nat → Nat}
    (valid : round.Valid programRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment)
    (lane : Nat) (laneLt : lane < 8) :
    assignment (round.permutationInputColumns.getD lane 0) =
      round.inputLane assignment lane := by
  cases kind : round.kind with
  | absorb chunkColumns =>
      exact absorb_input_lane kind valid canonical one satisfies lane laneLt
  | pad =>
      exact pad_input_lane kind valid canonical one satisfies lane laneLt

/-- One exact sponge round computes the extracted permutation of its intended
absorbed or padded state. -/
theorem round_sound
    {round : Round} {programRows : List Row} {assignment : Nat → Nat}
    (valid : round.Valid programRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    ∀ lane, lane < 8 →
      assignment (round.permutationOutputColumns.getD lane 0) =
        Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute
          (round.inputLane assignment) lane := by
  have callSound := Poseidon2Call.lanes_sound round.call programRows
    valid.2.1 canonical one satisfies
  intro lane laneLt
  rw [output_column_eq valid.1 lane laneLt]
  rw [callSound lane laneLt]
  apply Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute_congr
  intro inputLane inputLaneLt
  have mappedInput :
      round.call.columnMap (inputLane + 1) =
        round.permutationInputColumns.getD inputLane 0 := by
    have sourceLt : inputLane + 1 < 9 := by omega
    simp [Call.columnMap, sourceLt, valid.1.2.2.2.1]
  rw [mappedInput]
  exact input_lane_sound valid canonical one satisfies inputLane inputLaneLt

def stateAtColumns (assignment : Nat → Nat) (columns : List Nat) : Nat → Nat :=
  fun lane => assignment (columns.getD lane 0)

def semanticInput (assignment : Nat → Nat) (kind : RoundKind)
    (state : Nat → Nat) (lane : Nat) : Nat :=
  match kind with
  | .absorb chunkColumns =>
      if lane < chunkColumns.length then
        (state lane + assignment (chunkColumns.getD lane 0)) % goldilocksP
      else state lane
  | .pad => if lane = 0 then (state 0 + 1) % goldilocksP else state lane

def semanticRound (assignment : Nat → Nat) (round : Round)
    (state : Nat → Nat) : Nat → Nat :=
  Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute
    (semanticInput assignment round.kind state)

def runRounds (assignment : Nat → Nat) : List Round → (Nat → Nat) → Nat → Nat
  | [], state => state
  | round :: rest, state =>
      runRounds assignment rest (semanticRound assignment round state)

def Round.valueCount (round : Round) : Nat :=
  match round.kind with
  | .absorb chunkColumns => chunkColumns.length
  | .pad => 0

/-- The part of a generated sponge round that affects pure value execution.
Column identities are deliberately erased; only absorb length versus padding
remains. -/
inductive ValueSchedule where
  | absorb (count : Nat)
  | pad
deriving DecidableEq, Repr

def Round.valueSchedule (round : Round) : ValueSchedule :=
  match round.kind with
  | .absorb chunkColumns => .absorb chunkColumns.length
  | .pad => .pad

def valueSchedules (rounds : List Round) : List ValueSchedule :=
  rounds.map Round.valueSchedule

def valueInput (kind : RoundKind) (values : List Nat)
    (state : Nat → Nat) (lane : Nat) : Nat :=
  match kind with
  | .absorb chunkColumns =>
      if lane < chunkColumns.length then
        (state lane + values.getD lane 0) % goldilocksP
      else state lane
  | .pad => if lane = 0 then (state 0 + 1) % goldilocksP else state lane

def valueRound (round : Round) (values : List Nat)
    (state : Nat → Nat) : Nat → Nat :=
  Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute
    (valueInput round.kind values state)

private theorem valueCount_eq_of_schedule {left right : Round}
    (same : left.valueSchedule = right.valueSchedule) :
    left.valueCount = right.valueCount := by
  cases leftKind : left.kind with
  | pad =>
      cases rightKind : right.kind with
      | pad => simp [Round.valueCount, leftKind, rightKind]
      | absorb columns =>
          simp [Round.valueSchedule, leftKind, rightKind] at same
  | absorb leftColumns =>
      cases rightKind : right.kind with
      | pad =>
          simp [Round.valueSchedule, leftKind, rightKind] at same
      | absorb rightColumns =>
          have scheduleEq :
              ValueSchedule.absorb leftColumns.length =
                ValueSchedule.absorb rightColumns.length := by
            simpa [Round.valueSchedule, leftKind, rightKind] using same
          have lengthEq := ValueSchedule.absorb.inj scheduleEq
          simpa [Round.valueCount, leftKind, rightKind] using lengthEq

private theorem valueRound_eq_of_schedule {left right : Round}
    (same : left.valueSchedule = right.valueSchedule)
    (values : List Nat) (state : Nat → Nat) :
    valueRound left values state = valueRound right values state := by
  cases leftKind : left.kind with
  | pad =>
      cases rightKind : right.kind with
      | pad => simp [valueRound, valueInput, leftKind, rightKind]
      | absorb columns =>
          simp [Round.valueSchedule, leftKind, rightKind] at same
  | absorb leftColumns =>
      cases rightKind : right.kind with
      | pad =>
          simp [Round.valueSchedule, leftKind, rightKind] at same
      | absorb rightColumns =>
          have lengthEq : leftColumns.length = rightColumns.length := by
            have scheduleEq :
                ValueSchedule.absorb leftColumns.length =
                  ValueSchedule.absorb rightColumns.length := by
              simpa [Round.valueSchedule, leftKind, rightKind] using same
            exact ValueSchedule.absorb.inj scheduleEq
          unfold valueRound
          funext outputLane
          apply Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute_congr
          intro lane laneLt
          rw [leftKind, rightKind]
          simp only [valueInput]
          rw [lengthEq]

/-- Pure sponge execution: only input values and the certified absorb
schedule remain. Column numbers are not semantic inputs. -/
def runValueRounds : List Round → List Nat → (Nat → Nat) → Nat → Nat
  | [], _, state => state
  | round :: rest, values, state =>
      runValueRounds rest (values.drop round.valueCount)
        (valueRound round (values.take round.valueCount) state)

/-- Generated traces with the same absorb-length/padding schedule compute the
same pure sponge function, even when every artifact column number differs. -/
theorem runValueRounds_eq_of_schedules
    {left right : List Round}
    (same : valueSchedules left = valueSchedules right)
    (values : List Nat) (state : Nat → Nat) :
    runValueRounds left values state = runValueRounds right values state := by
  induction left generalizing right values state with
  | nil =>
      cases right with
      | nil => rfl
      | cons head tail =>
          simp [valueSchedules] at same
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil =>
          simp [valueSchedules] at same
      | cons rightHead rightTail =>
          simp only [valueSchedules, List.map_cons, List.cons.injEq] at same
          have countEq := valueCount_eq_of_schedule same.1
          simp only [runValueRounds]
          rw [countEq]
          rw [valueRound_eq_of_schedule same.1]
          exact inductionHypothesis same.2 _ _

def absorbedColumnsOf (rounds : List Round) : List Nat :=
  rounds.flatMap fun round =>
    match round.kind with
    | .absorb chunkColumns => chunkColumns
    | .pad => []

private theorem getD_map_eq_of_map_eq
    {columns : List Nat} {assignment : Nat → Nat} {values : List Nat}
    (mapped : columns.map assignment = values) (lane : Nat) :
    assignment (columns.getD lane 0) = values.getD lane (assignment 0) := by
  have atLane := congrArg (fun entries : List Nat => entries.getD lane (assignment 0)) mapped
  simpa [List.getD_eq_getElem?_getD] using atLane

private theorem runRounds_eq_runValueRounds
    (assignment : Nat → Nat) (rounds : List Round)
    (values : List Nat) (state : Nat → Nat)
    (inputs : (absorbedColumnsOf rounds).map assignment = values) :
    runRounds assignment rounds state = runValueRounds rounds values state := by
  induction rounds generalizing values state with
  | nil =>
      simp [runRounds, runValueRounds]
  | cons round rest inductionHypothesis =>
      cases kind : round.kind with
      | pad =>
          simp only [absorbedColumnsOf, kind, List.flatMap_cons, List.map_append,
            List.map_nil, List.nil_append] at inputs
          simp only [runRounds, runValueRounds, Round.valueCount, kind,
            List.drop_zero, List.take_zero, semanticRound, valueRound,
            semanticInput, valueInput]
          exact inductionHypothesis values _ inputs
      | absorb chunkColumns =>
          have mappedAppend :
              chunkColumns.map assignment ++
                (absorbedColumnsOf rest).map assignment = values := by
            simpa [absorbedColumnsOf, kind, List.map_append] using inputs
          have headValues :
              chunkColumns.map assignment = values.take chunkColumns.length := by
            have taken := congrArg (fun entries : List Nat => entries.take chunkColumns.length)
              mappedAppend
            simpa using taken
          have tailValues :
              (absorbedColumnsOf rest).map assignment =
                values.drop chunkColumns.length := by
            have dropped := congrArg (fun entries : List Nat => entries.drop chunkColumns.length)
              mappedAppend
            simpa using dropped
          have stateEq :
              semanticRound assignment round state =
                valueRound round (values.take chunkColumns.length) state := by
            have inputsEq :
                semanticInput assignment round.kind state =
                  valueInput round.kind (values.take chunkColumns.length) state := by
              funext lane
              simp only [semanticInput, valueInput, kind, Round.valueCount]
              by_cases inChunk : lane < chunkColumns.length
              · simp only [inChunk, ↓reduceIte]
                have mappedLane := getD_map_eq_of_map_eq headValues lane
                have laneTake : lane < (values.take chunkColumns.length).length := by
                  rw [List.length_take]
                  have enoughValues : chunkColumns.length ≤ values.length := by
                    rw [← mappedAppend]
                    simp
                  simp [Nat.min_eq_left enoughValues, inChunk]
                have defaultsAgree :
                    (values.take chunkColumns.length).getD lane (assignment 0) =
                      (values.take chunkColumns.length).getD lane 0 := by
                  simp only [List.getD_eq_getElem?_getD]
                  rw [List.getElem?_eq_getElem laneTake]
                  simp
                rw [mappedLane, defaultsAgree]
              · simp only [inChunk, ↓reduceIte]
            exact congrArg
              (fun input : Nat → Nat => fun lane =>
                Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute
                  input lane) inputsEq
          simp only [runRounds, runValueRounds, Round.valueCount, kind]
          exact (congrArg (runRounds assignment rest) stateEq).trans
            (inductionHypothesis (values.drop chunkColumns.length)
              (valueRound round (values.take chunkColumns.length) state) tailValues)

def linkedCheck : List Nat → List Round → Bool
  | _, [] => true
  | priorColumns, round :: rest =>
      decide (round.stateBeforeColumns = priorColumns) &&
        linkedCheck round.permutationOutputColumns rest

def finalColumns : List Nat → List Round → List Nat
  | priorColumns, [] => priorColumns
  | _, round :: rest => finalColumns round.permutationOutputColumns rest

private theorem inputLane_matches_semantic
    {assignment : Nat → Nat} {round : Round} {state : Nat → Nat}
    (stateMatches : ∀ lane, lane < 8 →
      assignment (round.stateBeforeColumns.getD lane 0) = state lane) :
    ∀ lane, lane < 8 →
      round.inputLane assignment lane =
        semanticInput assignment round.kind state lane := by
  intro lane laneLt
  cases kind : round.kind with
  | absorb chunkColumns =>
      simp only [Round.inputLane, semanticInput, kind]
      split <;> simp_all
  | pad =>
      simp only [Round.inputLane, semanticInput, kind]
      split <;> simp_all

/-- Composition theorem for an arbitrary number of exact sponge rounds. -/
theorem rounds_sound
    (programRows : List Row) {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment)
    (rounds : List Round)
    (roundValid : ∀ round ∈ rounds, round.Valid programRows)
    (priorColumns : List Nat)
    (initialState : Nat → Nat)
    (initialMatches : ∀ lane, lane < 8 →
      assignment (priorColumns.getD lane 0) = initialState lane)
    (linked : linkedCheck priorColumns rounds = true) :
    ∀ lane, lane < 8 →
      assignment (finalColumns priorColumns rounds |>.getD lane 0) =
        runRounds assignment rounds initialState lane := by
  induction rounds generalizing priorColumns initialState with
  | nil =>
      simpa [finalColumns, runRounds] using initialMatches
  | cons round rest inductionHypothesis =>
      simp only [linkedCheck, Bool.and_eq_true] at linked
      have valid := roundValid round (by simp)
      have localSound := round_sound valid canonical one satisfies
      have roundStateMatches : ∀ lane, lane < 8 →
          assignment (round.stateBeforeColumns.getD lane 0) =
            initialState lane := by
        rw [of_decide_eq_true linked.1]
        exact initialMatches
      have roundInputMatches := inputLane_matches_semantic roundStateMatches
      have outputMatches : ∀ lane, lane < 8 →
          assignment (round.permutationOutputColumns.getD lane 0) =
            semanticRound assignment round initialState lane := by
        intro lane laneLt
        rw [localSound lane laneLt]
        apply Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute_congr
        exact roundInputMatches
      apply inductionHypothesis
      · intro later laterMember
        exact roundValid later (by simp [laterMember])
      · exact outputMatches
      · exact linked.2

structure Trace where
  inputColumns : List Nat
  zeroColumn : Nat
  zeroRow : Nat
  rounds : List Round
  outputColumns : List Nat
deriving DecidableEq, Repr, Inhabited

/-- Pure value-level acceptance of a generated sponge trace.  This recomputes
the output lanes from the ordered absorbed values; it does not inspect the
trace's R1CS rows. -/
def Trace.ValueAccepted (trace : Trace) (assignment : Nat → Nat) : Prop :=
  ∀ lane, lane < trace.outputColumns.length →
    assignment (trace.outputColumns.getD lane 0) =
      runValueRounds trace.rounds (trace.inputColumns.map assignment)
        (fun _ => 0) lane

def Trace.valueCheck (trace : Trace) (assignment : Nat → Nat) : Bool :=
  (List.range trace.outputColumns.length).all fun lane =>
    decide
      (assignment (trace.outputColumns.getD lane 0) =
        runValueRounds trace.rounds (trace.inputColumns.map assignment)
          (fun _ => 0) lane)

theorem Trace.valueCheck_eq_true_iff (trace : Trace)
    (assignment : Nat → Nat) :
    trace.valueCheck assignment = true ↔ trace.ValueAccepted assignment := by
  simp only [Trace.valueCheck, List.all_eq_true, decide_eq_true_eq,
    Trace.ValueAccepted]
  constructor
  · intro checked lane laneLt
    exact checked lane (List.mem_range.mpr laneLt)
  · intro accepted lane laneMember
    exact accepted lane (List.mem_range.mp laneMember)

def Trace.zeroDefinitionRows (trace : Trace) : List Row :=
  [builderLinearRow trace.zeroColumn []]

def Trace.absorbedColumns (trace : Trace) : List Nat :=
  absorbedColumnsOf trace.rounds

/-- Structural certificate for a complete sponge call.  All fields are row
identity or wire-schedule facts. -/
structure Trace.Valid (trace : Trace) (programRows : List Row) : Prop where
  zeroIncluded : rowsIncluded trace.zeroDefinitionRows programRows = true
  roundsAccepted :
    trace.rounds.all (fun round => decide (round.Valid programRows)) = true
  linked : linkedCheck (List.replicate 8 trace.zeroColumn) trace.rounds = true
  inputsOwned : trace.absorbedColumns = trace.inputColumns
  finalOutput : trace.outputColumns =
    (finalColumns (List.replicate 8 trace.zeroColumn) trace.rounds).take 4
  outputLength : trace.outputColumns.length = 4
  terminalPad : trace.rounds.getLast?.map Round.kind = some .pad

instance (trace : Trace) (programRows : List Row) :
    Decidable (trace.Valid programRows) := by
  let conditions :=
    rowsIncluded trace.zeroDefinitionRows programRows = true ∧
    trace.rounds.all (fun round => decide (round.Valid programRows)) = true ∧
    linkedCheck (List.replicate 8 trace.zeroColumn) trace.rounds = true ∧
    trace.absorbedColumns = trace.inputColumns ∧
    trace.outputColumns =
      (finalColumns (List.replicate 8 trace.zeroColumn) trace.rounds).take 4 ∧
    trace.outputColumns.length = 4 ∧
    trace.rounds.getLast?.map Round.kind = some .pad
  have decision : Decidable conditions := inferInstance
  cases decision with
  | isTrue accepted =>
      exact isTrue {
        zeroIncluded := accepted.1
        roundsAccepted := accepted.2.1
        linked := accepted.2.2.1
        inputsOwned := accepted.2.2.2.1
        finalOutput := accepted.2.2.2.2.1
        outputLength := accepted.2.2.2.2.2.1
        terminalPad := accepted.2.2.2.2.2.2 }
  | isFalse rejected =>
      exact isFalse fun valid => rejected ⟨
        valid.zeroIncluded,
        valid.roundsAccepted,
        valid.linked,
        valid.inputsOwned,
        valid.finalOutput,
        valid.outputLength,
        valid.terminalPad⟩

theorem Trace.Valid.roundValid
    {trace : Trace} {programRows : List Row}
    (valid : trace.Valid programRows) {round : Round}
    (member : round ∈ trace.rounds) : round.Valid programRows := by
  have accepted := (List.all_eq_true.mp valid.roundsAccepted) round member
  exact of_decide_eq_true accepted

private theorem zeroColumn_sound
    {trace : Trace} {programRows : List Row} {assignment : Nat → Nat}
    (valid : trace.Valid programRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    assignment trace.zeroColumn = 0 := by
  have member : builderLinearRow trace.zeroColumn [] ∈
      trace.zeroDefinitionRows := by simp [Trace.zeroDefinitionRows]
  have rowHolds := satisfies _
    (rowsIncluded_sound valid.zeroIncluded _ member)
  have defined := builderLinearRow_sound canonical one trace.zeroColumn []
    (by simp [CanonicalTerms]) rowHolds
  simpa [lcEval] using defined

/-- Exact artifact-level soundness of a complete generated sponge trace. -/
theorem trace_sound
    {trace : Trace} {programRows : List Row} {assignment : Nat → Nat}
    (valid : trace.Valid programRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    ∀ lane, lane < 4 →
      assignment (trace.outputColumns.getD lane 0) =
        runRounds assignment trace.rounds (fun _ => 0) lane := by
  have zero := zeroColumn_sound valid canonical one satisfies
  have initialMatches : ∀ lane, lane < 8 →
      assignment ((List.replicate 8 trace.zeroColumn).getD lane 0) = 0 := by
    intro lane laneLt
    have replicateLt : lane < (List.replicate 8 trace.zeroColumn).length := by
      simpa using laneLt
    have getEq := List.getElem_eq_getD
      (l := List.replicate 8 trace.zeroColumn) (i := lane)
      (h := replicateLt) 0
    rw [← getEq]
    simpa only [List.getElem_replicate] using zero
  have allLanes := rounds_sound programRows canonical one satisfies
    trace.rounds (fun round member => valid.roundValid member)
    (List.replicate 8 trace.zeroColumn)
    (fun _ => 0) initialMatches valid.linked
  intro lane laneLt
  rw [valid.finalOutput]
  let final := finalColumns (List.replicate 8 trace.zeroColumn) trace.rounds
  change assignment ((final.take 4).getD lane 0) = _
  have finalLength : 4 ≤ final.length := by
    have outputLength := valid.outputLength
    rw [valid.finalOutput, List.length_take] at outputLength
    change min 4 final.length = 4 at outputLength
    omega
  have laneFinal : lane < final.length := by
    omega
  have laneTake : lane < (final.take 4).length := by
    rw [List.length_take, Nat.min_eq_left finalLength]
    exact laneLt
  have takeGet := List.getElem_eq_getD
    (l := final.take 4) (i := lane) (h := laneTake) 0
  rw [← takeGet]
  rw [List.getElem_take]
  rw [List.getElem_eq_getD (h := laneFinal) 0]
  exact allLanes lane (by omega)

/-- Artifact-level soundness stated as a pure hash of the ordered input
values, with no semantic dependence on artifact column numbers. -/
theorem trace_values_sound
    {trace : Trace} {programRows : List Row} {assignment : Nat → Nat}
    (valid : trace.Valid programRows)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    ∀ lane, lane < 4 →
      assignment (trace.outputColumns.getD lane 0) =
        runValueRounds trace.rounds
          (trace.inputColumns.map assignment) (fun _ => 0) lane := by
  have columnsAgree :
      (absorbedColumnsOf trace.rounds).map assignment =
        trace.inputColumns.map assignment := by
    exact congrArg (List.map assignment) (by
      simpa [Trace.absorbedColumns] using valid.inputsOwned)
  have pure := runRounds_eq_runValueRounds assignment trace.rounds
    (trace.inputColumns.map assignment) (fun _ => 0) columnsAgree
  intro lane laneLt
  rw [← pure]
  exact trace_sound valid canonical one satisfies lane laneLt

end Nightstream.Implementation.R1CS.Poseidon2Sponge
