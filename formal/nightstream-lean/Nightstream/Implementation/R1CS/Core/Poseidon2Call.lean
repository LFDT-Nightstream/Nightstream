import Nightstream.Implementation.R1CS.Correspondence.Poseidon2.Poseidon2PermutationSound

/-!
Contract: compact, kernel-checked instantiation of the exact production
Poseidon2 permutation artifact at arbitrary builder columns.

The Rust emitter always allocates the isolated artifact's columns 9..608 as
one contiguous fresh interval.  Only columns 1..8 vary per call.  `Call`
therefore records ten small integers instead of copying 600 rows for every
hash invocation.  A generated artifact must still prove `rowsIncluded`; the
metadata itself has no semantic authority.
-/

namespace Nightstream.Implementation.R1CS.Poseidon2Call

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program

structure Call where
  rowStart : Nat
  rowEnd : Nat
  inputColumns : List Nat
  firstAllocatedColumn : Nat
deriving DecidableEq, Repr, Inhabited

/-- Column embedding from the isolated 609-column artifact into one builder
call site.  Column zero remains the global constant-one wire. -/
def Call.columnMap (call : Call) (column : Nat) : Nat :=
  if column = 0 then 0
  else if column < 9 then call.inputColumns.getD (column - 1) 0
  else call.firstAllocatedColumn + (column - 9)

@[simp] theorem Call.columnMap_zero (call : Call) : call.columnMap 0 = 0 := by
  simp [Call.columnMap]

def Call.rows (call : Call) : List Row :=
  Nightstream.Implementation.R1CS.Poseidon2Permutation.rows.map
    (renameRow call.columnMap)

def Call.programSlice (call : Call) (programRows : List Row) : List Row :=
  (programRows.drop call.rowStart).take (call.rowEnd - call.rowStart)

/-- Exact range identity, not merely set inclusion. -/
def Call.Matches (call : Call) (programRows : List Row) : Prop :=
  call.rows = call.programSlice programRows

instance (call : Call) (programRows : List Row) :
    Decidable (call.Matches programRows) := by
  unfold Call.Matches
  infer_instance

/-- Every satisfying global assignment induces a satisfying isolated
Poseidon assignment under the call's checked column embedding. -/
theorem sound
    (call : Call)
    (programRows : List Row)
    (rowMatch : call.Matches programRows)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    AgreeOn
      (Nightstream.Implementation.R1CS.Poseidon2PermutationSound.interpret
        (pullAssignment assignment call.columnMap))
      (pullAssignment assignment call.columnMap)
      (knownAfter
        Nightstream.Implementation.R1CS.Poseidon2Permutation.inputColumns
        Nightstream.Implementation.R1CS.Poseidon2Permutation.definitions) := by
  apply Nightstream.Implementation.R1CS.Poseidon2PermutationSound.poseidon2Permutation_renamed_sound
    call.columnMap call.columnMap_zero canonical one
  intro row member
  apply satisfies row
  have inSlice : row ∈ call.programSlice programRows := by
    rw [← rowMatch]
    exact member
  exact List.mem_of_mem_drop (List.mem_of_mem_take inSlice)

/-- Functional form used by hash/sponge compiler proofs: every global output
wire equals the extracted permutation applied only to this call's eight input
wires. -/
theorem outputs_sound
    (call : Call)
    (programRows : List Row)
    (rowMatch : call.Matches programRows)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    ∀ column ∈
        Nightstream.Implementation.R1CS.Poseidon2Permutation.outputColumns,
      assignment (call.columnMap column) =
        Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permuteState
          (pullAssignment assignment call.columnMap) column := by
  have callSound := sound call programRows rowMatch canonical one satisfies
  have functional :=
    Nightstream.Implementation.R1CS.Poseidon2PermutationSound.interpret_output_eq_permuteState
      (pullAssignment assignment call.columnMap)
  intro column member
  have known :=
    Nightstream.Implementation.R1CS.Poseidon2PermutationSound.outputs_known
      column member
  calc
    assignment (call.columnMap column) =
        pullAssignment assignment call.columnMap column := rfl
    _ = Nightstream.Implementation.R1CS.Poseidon2PermutationSound.interpret
          (pullAssignment assignment call.columnMap) column :=
        (callSound column known).symm
    _ = Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permuteState
          (pullAssignment assignment call.columnMap) column :=
        functional column member

private theorem outputColumns_generated :
    Nightstream.Implementation.R1CS.Poseidon2Permutation.outputColumns =
      (List.range 8).map (fun lane => 601 + lane) := by
  decide

private theorem outputColumn_mem (lane : Nat) (laneLt : lane < 8) :
    601 + lane ∈
      Nightstream.Implementation.R1CS.Poseidon2Permutation.outputColumns := by
  rw [outputColumns_generated]
  exact List.mem_map.mpr ⟨lane, List.mem_range.mpr laneLt, rfl⟩

/-- Eight-lane functional statement, independent of the call site's fresh
column numbers. -/
theorem lanes_sound
    (call : Call)
    (programRows : List Row)
    (rowMatch : call.Matches programRows)
    {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfies : Satisfies programRows assignment) :
    ∀ lane, lane < 8 →
      assignment (call.columnMap (601 + lane)) =
        Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute
          (fun inputLane => assignment (call.columnMap (inputLane + 1))) lane := by
  have outputs := outputs_sound call programRows rowMatch canonical one satisfies
  intro lane laneLt
  rw [outputs (601 + lane) (outputColumn_mem lane laneLt)]
  rw [Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permute_eq]
  change
    Nightstream.Implementation.R1CS.Poseidon2PermutationSound.interpret
        (Nightstream.Implementation.R1CS.Poseidon2PermutationSound.inputOnly
          (pullAssignment assignment call.columnMap)) (601 + lane) =
      Nightstream.Implementation.R1CS.Poseidon2PermutationSound.interpret
        (Nightstream.Implementation.R1CS.Poseidon2PermutationSound.inputOnly
          (Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permutationAssignment
            (fun inputLane => assignment (call.columnMap (inputLane + 1)))))
        (601 + lane)
  apply congrArg (fun state =>
    Nightstream.Implementation.R1CS.Poseidon2PermutationSound.interpret state
      (601 + lane))
  funext column
  unfold Nightstream.Implementation.R1CS.Poseidon2PermutationSound.inputOnly
  by_cases columnLt : column < 9
  · simp only [columnLt, ↓reduceIte]
    by_cases columnZero : column = 0
    · subst column
      simp [pullAssignment, Call.columnMap, one,
        Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permutationAssignment]
    · have columnPositive : 0 < column := Nat.pos_of_ne_zero columnZero
      simp [pullAssignment, Call.columnMap, columnZero, columnLt,
        Nightstream.Implementation.R1CS.Poseidon2PermutationSound.permutationAssignment,
        Nat.sub_add_cancel columnPositive]
  · simp [columnLt]

end Nightstream.Implementation.R1CS.Poseidon2Call
