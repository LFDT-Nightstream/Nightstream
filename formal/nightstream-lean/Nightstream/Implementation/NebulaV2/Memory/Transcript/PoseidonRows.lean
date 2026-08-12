import Nightstream.Implementation.NebulaV2.Memory.Transcript.HashFrameRows
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge
import Nightstream.SuperNeo.Concrete.Algebra

/-!
Contract: exact linked Poseidon2 duplex relation for the V2 memory challenge.

Assurance tier: implementation model and cryptographic primitive semantics.

Owns 14 frame absorbs, one terminal frame pad, four linked coordinate-tag
absorbs, eight exposed challenge limbs, row soundness to one continuous
eight-lane Poseidon2 state, and local honest completeness.

Does not own variable-frame authority, Fiat--Shamir unpredictability,
open-segment carry updates, absolute generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryTranscriptPoseidonRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2.MemoryTranscriptHashFrame
open Nightstream.Implementation.NebulaV2.MemoryTranscriptHashFrameRows
open Nightstream.Implementation.NebulaV2.CompactChainHashFrameRows
open Nightstream.Protocol.NebulaV2

def frameSchedule : List ValueSchedule :=
  List.replicate 13 (.absorb 4) ++ [.absorb 1, .pad]

def coordinateSchedule : List ValueSchedule :=
  List.replicate 4 (.absorb 1)

def expectedSchedule : List ValueSchedule :=
  frameSchedule ++ coordinateSchedule

theorem expectedSchedule_exact :
    expectedSchedule.length = 19 ∧
      (expectedSchedule.filter (· = .absorb 4)).length = 13 ∧
      (expectedSchedule.filter (· = .absorb 1)).length = 5 ∧
      (expectedSchedule.filter (· = .pad)).length = 1 := by
  decide

def representativeRound : ValueSchedule → Round
  | .absorb count =>
      { (default : Round) with
        kind := .absorb (List.replicate count 0) }
  | .pad =>
      { (default : Round) with kind := .pad }

theorem representativeRound_schedule (schedule : ValueSchedule) :
    (representativeRound schedule).valueSchedule = schedule := by
  cases schedule <;> simp [representativeRound, Round.valueSchedule]

def representativeRounds : List Round :=
  expectedSchedule.map representativeRound

theorem representativeRounds_schedule :
    valueSchedules representativeRounds = expectedSchedule := by
  rw [representativeRounds, valueSchedules, List.map_map]
  change
    expectedSchedule.map
        (fun schedule => (representativeRound schedule).valueSchedule) =
      expectedSchedule
  generalize expectedSchedule = schedules
  induction schedules with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.map_cons, List.cons.injEq]
      exact ⟨representativeRound_schedule head, inductionHypothesis⟩

def coordinateTags : List Nat := List.ofFn coordinateTag

theorem coordinateTags_length : coordinateTags.length = 4 := by
  simp [coordinateTags]

def prefixRoundCount (coordinate : Fin 4) : Nat :=
  16 + coordinate.val

def duplexValues (input : Input) (coordinate : Fin 4) : List Nat :=
  encode input ++ coordinateTags.take (coordinate.val + 1)

/-- Pure fixed Poseidon2 result for one extension-field coordinate. Lanes zero
and one are the two extension coefficients. -/
def pureCoordinate (input : Input) (coordinate : Fin 4)
    (limb : Fin 2) : Nat :=
  runValueRounds
    (representativeRounds.take (prefixRoundCount coordinate))
    (duplexValues input coordinate) (fun _ => 0) limb.val

theorem pureCoordinate_lt (input : Input) (coordinate : Fin 4)
    (limb : Fin 2) :
    pureCoordinate input coordinate limb < goldilocksP := by
  exact runValueRounds_canonical
    (representativeRounds.take (prefixRoundCount coordinate))
    (duplexValues input coordinate) (fun _ => 0)
    (by intro _; norm_num [goldilocksP]) limb.val

def pureF (input : Input) (coordinate : Fin 4) (limb : Fin 2) :
    Nightstream.SuperNeo.Concrete.F :=
  ⟨pureCoordinate input coordinate limb, by
    simpa [goldilocksP, Nightstream.SuperNeo.Concrete.goldilocksModulus]
      using pureCoordinate_lt input coordinate limb⟩

def pureK (input : Input) (coordinate : Fin 4) :
    Nightstream.SuperNeo.Concrete.K :=
  ⟨pureF input coordinate 0, pureF input coordinate 1⟩

/-- Exact two-pair challenge selected by the linked duplex schedule. -/
def pureChallenges (input : Input) :
    ProductState.Challenges Nightstream.SuperNeo.Concrete.K :=
  fun repetition =>
    { gamma1 := pureK input (Transcript.coordinateIndex repetition 0)
      gamma2 := pureK input (Transcript.coordinateIndex repetition 1) }

structure Layout where
  frame : MemoryTranscriptHashFrameRows.Layout
  zeroColumn : Nat
  coordinateTagColumn : Fin 4 → Nat
  rounds : List Round
deriving DecidableEq, Repr

def Layout.coordinateTagColumns (layout : Layout) : List Nat :=
  List.ofFn layout.coordinateTagColumn

def zeroRows (layout : Layout) : List Row :=
  [builderLinearRow layout.zeroColumn []]

def coordinateTagRows (layout : Layout) : List Row :=
  FixedFrame.rows layout.coordinateTagColumns coordinateTags

def rows (layout : Layout) : List Row :=
  MemoryTranscriptHashFrameRows.rows layout.frame ++
    (zeroRows layout ++ (coordinateTagRows layout ++
      layout.rounds.flatMap Round.rows))

def Layout.coordinateOutputColumn (layout : Layout)
    (coordinate : Fin 4) (limb : Fin 2) : Nat :=
  (finalColumns (List.replicate 8 layout.zeroColumn)
      (layout.rounds.take (prefixRoundCount coordinate))).getD limb.val 0

def Layout.challengeColumn (layout : Layout)
    (repetition coordinate limb : Fin 2) : Nat :=
  layout.coordinateOutputColumn
    (Transcript.coordinateIndex repetition coordinate) limb

/-- All certificate fields are structural row, column, or schedule facts.
No challenge value or row-satisfaction conclusion occurs here. -/
structure Layout.Valid (layout : Layout) : Prop where
  exactSchedule : valueSchedules layout.rounds = expectedSchedule
  roundsAccepted :
    layout.rounds.all (fun round => decide (round.Valid (rows layout))) = true
  linked :
    linkedCheck (List.replicate 8 layout.zeroColumn) layout.rounds = true
  exactAbsorbedColumns :
    absorbedColumnsOf layout.rounds =
      layout.frame.inputColumns ++ layout.coordinateTagColumns
  prefixAbsorbedColumns : ∀ coordinate,
    absorbedColumnsOf
        (layout.rounds.take (prefixRoundCount coordinate)) =
      layout.frame.inputColumns ++
        layout.coordinateTagColumns.take (coordinate.val + 1)
  roundRowsLength : (layout.rounds.flatMap Round.rows).length = 11458

theorem Layout.Valid.round_count_exact
    {layout : Layout} (valid : layout.Valid) :
    layout.rounds.length = 19 := by
  have lengths := congrArg List.length valid.exactSchedule
  simpa [valueSchedules, expectedSchedule_exact.1] using lengths

theorem Layout.Valid.roundValid
    {layout : Layout} (valid : layout.Valid)
    {round : Round} (member : round ∈ layout.rounds) :
    round.Valid (rows layout) := by
  have checked := List.all_eq_true.mp valid.roundsAccepted round member
  exact of_decide_eq_true checked

theorem rows_length_exact
    {layout : Layout} (valid : layout.Valid) :
    (rows layout).length = 11472 := by
  simp [rows, MemoryTranscriptHashFrameRows.rows_length_exact,
    zeroRows, coordinateTagRows, FixedFrame.rows, ConstantPins.rows,
    FixedFrame.pins, Layout.coordinateTagColumns, coordinateTags_length,
    valid.roundRowsLength]

private theorem frame_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryTranscriptHashFrameRows.rows layout.frame) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem coordinate_tag_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (coordinateTagRows layout) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem coordinate_tags_canonical :
    ∀ value ∈ coordinateTags, value < goldilocksP := by
  intro value member
  rw [coordinateTags, coordinateTags_exact] at member
  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
  rcases member with rfl | rfl | rfl | rfl <;> decide

theorem coordinate_tags_exact
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    layout.coordinateTagColumns.map assignment = coordinateTags := by
  exact FixedFrame.sound
    (by simp [Layout.coordinateTagColumns, coordinateTags_length])
    coordinate_tags_canonical canonical one (coordinate_tag_rows_hold holds)

private theorem zero_exact
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.zeroColumn = 0 := by
  have rowHolds :
      Satisfies [builderLinearRow layout.zeroColumn []] assignment := by
    intro row member
    have rowEqual : row = builderLinearRow layout.zeroColumn [] := by
      simpa using member
    subst row
    exact holds _ (by simp [rows, zeroRows])
  have defined := builderLinearRow_sound canonical one layout.zeroColumn []
    (by simp [CanonicalTerms]) (rowHolds _ (by simp))
  simpa [lcEval] using defined

private theorem linkedCheck_take
    (priorColumns : List Nat) (rounds : List Round) (count : Nat)
    (linked : linkedCheck priorColumns rounds = true) :
    linkedCheck priorColumns (rounds.take count) = true := by
  induction rounds generalizing priorColumns count with
  | nil => simp [linkedCheck]
  | cons round rest inductionHypothesis =>
      cases count with
      | zero => simp [linkedCheck]
      | succ count =>
          simp only [linkedCheck, Bool.and_eq_true] at linked
          simp only [List.take_succ_cons, linkedCheck, Bool.and_eq_true]
          exact ⟨linked.1,
            inductionHypothesis round.permutationOutputColumns count linked.2⟩

private theorem prefix_round_valid
    {layout : Layout} (valid : layout.Valid) (coordinate : Fin 4) :
    ∀ round ∈ layout.rounds.take (prefixRoundCount coordinate),
      round.Valid (rows layout) := by
  intro round member
  exact valid.roundValid (List.mem_of_mem_take member)

private theorem prefix_absorbed_values
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {input : Input}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : VariablePlaced layout.frame assignment input)
    (holds : Satisfies (rows layout) assignment)
    (coordinate : Fin 4) :
    (absorbedColumnsOf
        (layout.rounds.take (prefixRoundCount coordinate))).map assignment =
      duplexValues input coordinate := by
  rw [valid.prefixAbsorbedColumns coordinate, List.map_append]
  rw [MemoryTranscriptHashFrameRows.input_exact canonical one placed
    (frame_rows_hold holds)]
  rw [List.map_take, coordinate_tags_exact canonical one holds]
  rfl

private theorem prefix_schedules_exact
    {layout : Layout} (valid : layout.Valid) (coordinate : Fin 4) :
    valueSchedules (layout.rounds.take (prefixRoundCount coordinate)) =
      valueSchedules
        (representativeRounds.take (prefixRoundCount coordinate)) := by
  have schedules := valid.exactSchedule.trans representativeRounds_schedule.symm
  have prefixes := congrArg
    (List.take (prefixRoundCount coordinate)) schedules
  simpa [valueSchedules] using prefixes

/-- Every exposed limb is a row consequence of the same linked duplex state.
The theorem does not assume a challenge value. -/
theorem coordinate_output_exact
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {input : Input}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : VariablePlaced layout.frame assignment input)
    (holds : Satisfies (rows layout) assignment)
    (coordinate : Fin 4) (limb : Fin 2) :
    assignment (layout.coordinateOutputColumn coordinate limb) =
      pureCoordinate input coordinate limb := by
  let prefixRounds := layout.rounds.take (prefixRoundCount coordinate)
  have zero := zero_exact canonical one holds
  have initialMatches : ∀ lane, lane < 8 →
      assignment ((List.replicate 8 layout.zeroColumn).getD lane 0) = 0 := by
    intro lane laneLt
    have columnExact :
        (List.replicate 8 layout.zeroColumn).getD lane 0 =
          layout.zeroColumn := by
      rw [List.getD_eq_getElem?_getD]
      have inBounds : lane < (List.replicate 8 layout.zeroColumn).length := by
        simpa using laneLt
      rw [List.getElem?_eq_getElem inBounds]
      simp only [Option.getD_some]
      exact @List.getElem_replicate Nat layout.zeroColumn 8 lane inBounds
    rw [columnExact, zero]
  have rowSound := rounds_values_sound (rows layout) canonical one holds
    prefixRounds (prefix_round_valid valid coordinate)
    (List.replicate 8 layout.zeroColumn) (fun _ => 0) initialMatches
    (linkedCheck_take _ _ _ valid.linked)
    (duplexValues input coordinate)
    (prefix_absorbed_values valid canonical one placed holds coordinate)
  have schedules := prefix_schedules_exact valid coordinate
  have pureEqual := runValueRounds_eq_of_schedules schedules
    (duplexValues input coordinate) (fun _ => 0)
  calc
    assignment (layout.coordinateOutputColumn coordinate limb) =
        runValueRounds prefixRounds (duplexValues input coordinate)
          (fun _ => 0) limb.val := rowSound limb.val (by omega)
    _ = runValueRounds
          (representativeRounds.take (prefixRoundCount coordinate))
          (duplexValues input coordinate) (fun _ => 0) limb.val :=
      congrFun pureEqual limb.val
    _ = pureCoordinate input coordinate limb := rfl

theorem challenges_exact
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {input : Input}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : VariablePlaced layout.frame assignment input)
    (holds : Satisfies (rows layout) assignment) :
    ∀ repetition coordinate limb : Fin 2,
      assignment (layout.challengeColumn repetition coordinate limb) =
        pureCoordinate input
          (Transcript.coordinateIndex repetition coordinate) limb := by
  intro repetition coordinate limb
  exact coordinate_output_exact valid canonical one placed holds _ _

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (input : Input) : Prop where
  frame : MemoryTranscriptHashFrameRows.Honest layout.frame assignment input
  zero : assignment layout.zeroColumn = 0
  coordinateTags :
    layout.coordinateTagColumns.map assignment =
      MemoryTranscriptPoseidonRows.coordinateTags
  rounds : ∀ round ∈ layout.rounds, round.ExecutionWitness assignment

theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat} {input : Input}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment input) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows] at member
  rcases List.mem_append.mp member with frameMember | remaining
  · exact MemoryTranscriptHashFrameRows.rows_complete one honest.frame
      row frameMember
  rcases List.mem_append.mp remaining with zeroMember | remaining
  · have rowEqual : row = builderLinearRow layout.zeroColumn [] := by
      simpa [zeroRows] using zeroMember
    subst row
    apply builderLinearRow_complete one layout.zeroColumn []
    · simp [CanonicalTerms]
    · simpa [lcEval] using honest.zero
  rcases List.mem_append.mp remaining with tagMember | roundMember
  · exact FixedFrame.complete coordinate_tags_canonical
      honest.coordinateTags one row tagMember
  · rcases List.mem_flatMap.mp roundMember with
      ⟨round, roundInTrace, rowInRound⟩
    exact Round.execution_complete canonical one
      (honest.rounds round roundInTrace) row rowInRound

/-! ## Profile-indexed successor transcript -/

namespace ProfileIndexed

def duplexValues (profile : Profile.Identity) (input : Input)
    (coordinate : Fin 4) : List Nat :=
  encodeFor profile input ++ coordinateTags.take (coordinate.val + 1)

def pureCoordinate (profile : Profile.Identity) (input : Input)
    (coordinate : Fin 4) (limb : Fin 2) : Nat :=
  runValueRounds
    (representativeRounds.take (prefixRoundCount coordinate))
    (duplexValues profile input coordinate) (fun _ => 0) limb.val

theorem pureCoordinate_lt (profile : Profile.Identity) (input : Input)
    (coordinate : Fin 4) (limb : Fin 2) :
    pureCoordinate profile input coordinate limb < goldilocksP := by
  exact runValueRounds_canonical
    (representativeRounds.take (prefixRoundCount coordinate))
    (duplexValues profile input coordinate) (fun _ => 0)
    (by intro _; norm_num [goldilocksP]) limb.val

def pureF (profile : Profile.Identity) (input : Input)
    (coordinate : Fin 4) (limb : Fin 2) :
    Nightstream.SuperNeo.Concrete.F :=
  ⟨pureCoordinate profile input coordinate limb, by
    simpa [goldilocksP, Nightstream.SuperNeo.Concrete.goldilocksModulus]
      using pureCoordinate_lt profile input coordinate limb⟩

def pureK (profile : Profile.Identity) (input : Input)
    (coordinate : Fin 4) : Nightstream.SuperNeo.Concrete.K :=
  ⟨pureF profile input coordinate 0, pureF profile input coordinate 1⟩

def pureChallenges (profile : Profile.Identity) (input : Input) :
    ProductState.Challenges Nightstream.SuperNeo.Concrete.K :=
  fun repetition =>
    { gamma1 := pureK profile input (Transcript.coordinateIndex repetition 0)
      gamma2 := pureK profile input (Transcript.coordinateIndex repetition 1) }

def rows (profile : Profile.Identity) (layout : Layout) : List Row :=
  MemoryTranscriptHashFrameRows.ProfileIndexed.rows profile layout.frame ++
    (zeroRows layout ++ (coordinateTagRows layout ++
      layout.rounds.flatMap Round.rows))

structure Valid (profile : Profile.Identity) (layout : Layout) : Prop where
  exactSchedule : valueSchedules layout.rounds = expectedSchedule
  roundsAccepted :
    layout.rounds.all (fun round => decide (round.Valid (rows profile layout))) =
      true
  linked :
    linkedCheck (List.replicate 8 layout.zeroColumn) layout.rounds = true
  exactAbsorbedColumns :
    absorbedColumnsOf layout.rounds =
      layout.frame.inputColumns ++ layout.coordinateTagColumns
  prefixAbsorbedColumns : ∀ coordinate,
    absorbedColumnsOf
        (layout.rounds.take (prefixRoundCount coordinate)) =
      layout.frame.inputColumns ++
        layout.coordinateTagColumns.take (coordinate.val + 1)
  roundRowsLength : (layout.rounds.flatMap Round.rows).length = 11458

theorem rows_length_exact
    {profile : Profile.Identity} {layout : Layout}
    (valid : Valid profile layout) :
    (rows profile layout).length = 11472 := by
  simp [rows,
    MemoryTranscriptHashFrameRows.ProfileIndexed.rows_length_exact,
    zeroRows, coordinateTagRows, FixedFrame.rows, ConstantPins.rows,
    FixedFrame.pins, Layout.coordinateTagColumns, coordinateTags_length,
    valid.roundRowsLength]

private theorem frame_rows_hold
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows profile layout) assignment) :
    Satisfies
      (MemoryTranscriptHashFrameRows.ProfileIndexed.rows profile layout.frame)
      assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem coordinate_tag_rows_hold
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows profile layout) assignment) :
    Satisfies (coordinateTagRows layout) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem profile_coordinate_tags_exact
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows profile layout) assignment) :
    layout.coordinateTagColumns.map assignment = coordinateTags := by
  exact FixedFrame.sound
    (by simp [Layout.coordinateTagColumns, coordinateTags_length])
    coordinate_tags_canonical canonical one (coordinate_tag_rows_hold holds)

private theorem zero_exact
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows profile layout) assignment) :
    assignment layout.zeroColumn = 0 := by
  have rowHolds :
      Satisfies [builderLinearRow layout.zeroColumn []] assignment := by
    intro row member
    have rowEqual : row = builderLinearRow layout.zeroColumn [] := by
      simpa using member
    subst row
    exact holds _ (by simp [rows, zeroRows])
  have defined := builderLinearRow_sound canonical one layout.zeroColumn []
    (by simp [CanonicalTerms]) (rowHolds _ (by simp))
  simpa [lcEval] using defined

private theorem round_valid
    {profile : Profile.Identity} {layout : Layout}
    (valid : Valid profile layout)
    {round : Round} (member : round ∈ layout.rounds) :
    round.Valid (rows profile layout) := by
  have checked := List.all_eq_true.mp valid.roundsAccepted round member
  exact of_decide_eq_true checked

private theorem prefix_round_valid
    {profile : Profile.Identity} {layout : Layout}
    (valid : Valid profile layout) (coordinate : Fin 4) :
    ∀ round ∈ layout.rounds.take (prefixRoundCount coordinate),
      round.Valid (rows profile layout) := by
  intro round member
  exact round_valid valid (List.mem_of_mem_take member)

private theorem prefix_absorbed_values
    {profile : Profile.Identity} {layout : Layout}
    (valid : Valid profile layout)
    {assignment : Nat → Nat} {input : Input}
    (profileCanonical : ProfileCanonical profile)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : MemoryTranscriptHashFrameRows.VariablePlaced layout.frame
      assignment input)
    (holds : Satisfies (rows profile layout) assignment)
    (coordinate : Fin 4) :
    (absorbedColumnsOf
        (layout.rounds.take (prefixRoundCount coordinate))).map assignment =
      duplexValues profile input coordinate := by
  rw [valid.prefixAbsorbedColumns coordinate, List.map_append]
  rw [MemoryTranscriptHashFrameRows.ProfileIndexed.input_exact
    profileCanonical canonical one placed (frame_rows_hold holds)]
  rw [List.map_take, profile_coordinate_tags_exact canonical one holds]
  rfl

private theorem prefix_schedules_exact
    {profile : Profile.Identity} {layout : Layout}
    (valid : Valid profile layout) (coordinate : Fin 4) :
    valueSchedules (layout.rounds.take (prefixRoundCount coordinate)) =
      valueSchedules
        (representativeRounds.take (prefixRoundCount coordinate)) := by
  have schedules := valid.exactSchedule.trans representativeRounds_schedule.symm
  have prefixes := congrArg
    (List.take (prefixRoundCount coordinate)) schedules
  simpa [valueSchedules] using prefixes

theorem coordinate_output_exact
    {profile : Profile.Identity} {layout : Layout}
    (valid : Valid profile layout)
    {assignment : Nat → Nat} {input : Input}
    (profileCanonical : ProfileCanonical profile)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : MemoryTranscriptHashFrameRows.VariablePlaced layout.frame
      assignment input)
    (holds : Satisfies (rows profile layout) assignment)
    (coordinate : Fin 4) (limb : Fin 2) :
    assignment (layout.coordinateOutputColumn coordinate limb) =
      pureCoordinate profile input coordinate limb := by
  let prefixRounds := layout.rounds.take (prefixRoundCount coordinate)
  have zero := zero_exact canonical one holds
  have initialMatches : ∀ lane, lane < 8 →
      assignment ((List.replicate 8 layout.zeroColumn).getD lane 0) = 0 := by
    intro lane laneLt
    have columnExact :
        (List.replicate 8 layout.zeroColumn).getD lane 0 =
          layout.zeroColumn := by
      rw [List.getD_eq_getElem?_getD]
      have inBounds : lane < (List.replicate 8 layout.zeroColumn).length := by
        simpa using laneLt
      rw [List.getElem?_eq_getElem inBounds]
      simp only [Option.getD_some]
      exact @List.getElem_replicate Nat layout.zeroColumn 8 lane inBounds
    rw [columnExact, zero]
  have rowSound := rounds_values_sound (rows profile layout) canonical one holds
    prefixRounds (prefix_round_valid valid coordinate)
    (List.replicate 8 layout.zeroColumn) (fun _ => 0) initialMatches
    (linkedCheck_take _ _ _ valid.linked)
    (duplexValues profile input coordinate)
    (prefix_absorbed_values valid profileCanonical canonical one placed holds
      coordinate)
  have schedules := prefix_schedules_exact valid coordinate
  have pureEqual := runValueRounds_eq_of_schedules schedules
    (duplexValues profile input coordinate) (fun _ => 0)
  calc
    assignment (layout.coordinateOutputColumn coordinate limb) =
        runValueRounds prefixRounds (duplexValues profile input coordinate)
          (fun _ => 0) limb.val := rowSound limb.val (by omega)
    _ = runValueRounds
          (representativeRounds.take (prefixRoundCount coordinate))
          (duplexValues profile input coordinate) (fun _ => 0) limb.val :=
      congrFun pureEqual limb.val
    _ = pureCoordinate profile input coordinate limb := rfl

theorem challenges_exact
    {profile : Profile.Identity} {layout : Layout}
    (valid : Valid profile layout)
    {assignment : Nat → Nat} {input : Input}
    (profileCanonical : ProfileCanonical profile)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : MemoryTranscriptHashFrameRows.VariablePlaced layout.frame
      assignment input)
    (holds : Satisfies (rows profile layout) assignment) :
    ∀ repetition coordinate limb : Fin 2,
      assignment (layout.challengeColumn repetition coordinate limb) =
        pureCoordinate profile input
          (Transcript.coordinateIndex repetition coordinate) limb := by
  intro repetition coordinate limb
  exact coordinate_output_exact valid profileCanonical canonical one placed
    holds _ _

structure Honest (profile : Profile.Identity) (layout : Layout)
    (assignment : Nat → Nat) (input : Input) : Prop where
  frame : MemoryTranscriptHashFrameRows.ProfileIndexed.Honest profile
    layout.frame assignment input
  zero : assignment layout.zeroColumn = 0
  coordinateTags :
    layout.coordinateTagColumns.map assignment =
      MemoryTranscriptPoseidonRows.coordinateTags
  rounds : ∀ round ∈ layout.rounds, round.ExecutionWitness assignment

theorem rows_complete
    {profile : Profile.Identity} {layout : Layout} {assignment : Nat → Nat}
    {input : Input}
    (profileCanonical : ProfileCanonical profile)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest profile layout assignment input) :
    Satisfies (rows profile layout) assignment := by
  intro row member
  rw [rows] at member
  rcases List.mem_append.mp member with frameMember | remaining
  · exact MemoryTranscriptHashFrameRows.ProfileIndexed.rows_complete
      profileCanonical one honest.frame row frameMember
  rcases List.mem_append.mp remaining with zeroMember | remaining
  · have rowEqual : row = builderLinearRow layout.zeroColumn [] := by
      simpa [zeroRows] using zeroMember
    subst row
    apply builderLinearRow_complete one layout.zeroColumn []
    · simp [CanonicalTerms]
    · simpa [lcEval] using honest.zero
  rcases List.mem_append.mp remaining with tagMember | roundMember
  · exact FixedFrame.complete coordinate_tags_canonical
      honest.coordinateTags one row tagMember
  · rcases List.mem_flatMap.mp roundMember with
      ⟨round, roundInTrace, rowInRound⟩
    exact Round.execution_complete canonical one
      (honest.rounds round roundInTrace) row rowInRound

end ProfileIndexed

end Nightstream.Implementation.NebulaV2.MemoryTranscriptPoseidonRows
