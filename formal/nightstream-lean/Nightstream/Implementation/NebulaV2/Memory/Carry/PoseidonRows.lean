import Nightstream.Implementation.NebulaV2.Memory.Carry.HashFrameRows
import Nightstream.Implementation.R1CS.Core.Poseidon2Sponge

/-!
Contract: exact Poseidon2 sponge relation for the 117-field V2 memory-carry
digest.

Assurance tier: implementation model and cryptographic primitive semantics.

Owns the exact 29 full absorbs, one one-field absorb, one terminal padding
round, structural trace linkage, reduction from satisfying generated rows to
the pure fixed Poseidon2 digest, and honest row completeness from independent
permutation executions.

Does not own Poseidon2 collision resistance, the outer F-prime state-output
hash, absolute generated columns, or Rust conformance.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Poseidon2Sponge
open Nightstream.Implementation.NebulaV2.MemoryCarryHashFrame
open Nightstream.Implementation.NebulaV2.MemoryCarryHashFrameRows

def expectedSchedule : List ValueSchedule :=
  List.replicate 29 (.absorb 4) ++ [.absorb 1, .pad]

theorem expectedSchedule_exact :
    expectedSchedule.length = 31 ∧
      (expectedSchedule.filter (· = .absorb 4)).length = 29 ∧
      (expectedSchedule.filter (· = .absorb 1)).length = 1 ∧
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

/-- The pure, column-independent digest function selected by V2. -/
def pureDigest (values : List Nat) (lane : Nat) : Nat :=
  runValueRounds representativeRounds values (fun _ => 0) lane

def carryDigest (block : MemoryCarryParser.Block) : Fin 4 → Nat :=
  fun lane => pureDigest (frame block) lane.val

structure Layout where
  frame : MemoryCarryHashFrameRows.Layout
  trace : Trace
deriving DecidableEq, Repr

def rows (layout : Layout) : List Row :=
  MemoryCarryHashFrameRows.rows layout.frame ++ layout.trace.rows

/-- All validity fields are structural row/wire facts. No digest value or
row-satisfaction conclusion occurs in this certificate. -/
structure Layout.Valid (layout : Layout) : Prop where
  exactInputColumns :
    layout.trace.inputColumns = MemoryCarryHashFrameRows.inputColumns layout.frame
  exactSchedule : valueSchedules layout.trace.rounds = expectedSchedule
  traceValid : layout.trace.Valid (rows layout)

theorem Layout.Valid.round_count_exact
    {layout : Layout} (valid : layout.Valid) :
    layout.trace.rounds.length = 31 := by
  have lengths := congrArg List.length valid.exactSchedule
  simpa [valueSchedules, expectedSchedule_exact.1] using lengths

private theorem frame_rows_hold
    {layout : Layout} {assignment : Nat → Nat}
    (holds : Satisfies (rows layout) assignment) :
    Satisfies (MemoryCarryHashFrameRows.rows layout.frame) assignment := by
  intro row member
  exact holds row (by simp [rows, member])

private theorem exact_input_values
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed
      layout.frame.packing.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    layout.trace.inputColumns.map assignment = frame block := by
  rw [valid.exactInputColumns]
  exact MemoryCarryHashFrameRows.input_column_values canonical one placed
    (frame_rows_hold holds)

/-- Satisfying the exact generated trace computes the fixed V2 carry digest
from the same authority-bearing parser block. -/
theorem output_columns_eq_carryDigest
    {layout : Layout} (valid : layout.Valid)
    {assignment : Nat → Nat} {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : PublicBitBlock.Placed
      layout.frame.packing.publicBits assignment block)
    (holds : Satisfies (rows layout) assignment) :
    ∀ lane : Fin 4,
      assignment (layout.trace.outputColumns.getD lane.val 0) =
        carryDigest block lane := by
  have traceSound := trace_values_sound valid.traceValid canonical one holds
  have inputValues := exact_input_values valid canonical one placed holds
  have schedules :
      valueSchedules layout.trace.rounds =
        valueSchedules representativeRounds :=
    valid.exactSchedule.trans representativeRounds_schedule.symm
  have runEqual := runValueRounds_eq_of_schedules schedules
    (frame block) (fun _ => 0)
  intro lane
  calc
    assignment (layout.trace.outputColumns.getD lane.val 0) =
        runValueRounds layout.trace.rounds
          (layout.trace.inputColumns.map assignment) (fun _ => 0) lane.val :=
      traceSound lane.val lane.isLt
    _ = runValueRounds layout.trace.rounds
          (frame block) (fun _ => 0) lane.val := by rw [inputValues]
    _ = runValueRounds representativeRounds
          (frame block) (fun _ => 0) lane.val := congrFun runEqual lane.val
    _ = carryDigest block lane := rfl

structure Honest (layout : Layout) (assignment : Nat → Nat)
    (block : MemoryCarryParser.Block) : Prop where
  frame : MemoryCarryHashFrameRows.Honest layout.frame assignment block
  trace : layout.trace.ExecutionWitness assignment

/-- Independent frame placement and permutation executions satisfy the
complete local carry-digest relation. -/
theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat}
    {block : MemoryCarryParser.Block}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment block) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with frameMember | traceMember
  · exact MemoryCarryHashFrameRows.rows_complete one honest.frame
      row frameMember
  · exact Trace.execution_complete canonical one honest.trace row traceMember

end Nightstream.Implementation.NebulaV2.MemoryCarryPoseidonRows
