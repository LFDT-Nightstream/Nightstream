import Nightstream.Implementation.R1CS.Poseidon2PermutationArtifact

/-!
Contract: artifact-level soundness and completeness of the exact production
Goldilocks Poseidon2 width-8 permutation rows.

The generated SSA certificate classifies every one of the 600 authoritative
Rust rows as a fresh linear or multiplication definition. The generic program
theorem then gives two non-vacuous guarantees:

- every satisfying assignment agrees with the deterministic interpreter;
- interpreting any canonical input assignment constructs a satisfying witness.

This proves circuit functionality without trusting the committed honest
witness. Native Rust `PERM` parity remains a distinct M5 refinement boundary.
-/

namespace Nightstream.Implementation.R1CS.Poseidon2PermutationSound

set_option maxRecDepth 65536
set_option maxHeartbeats 5000000

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Poseidon2Permutation

def interpret (state : Nat → Nat) : Nat → Nat := run state definitions

/-- Keep only constant-one and the eight permutation inputs. -/
def inputOnly (state : Nat → Nat) : Nat → Nat :=
  fun column => if column < 9 then state column else 0

/-- Executable eight-input semantics extracted from the exact SSA program. -/
def permuteState (state : Nat → Nat) : Nat → Nat :=
  interpret (inputOnly state)

/-- Column-zero/one based presentation of the eight-input permutation. -/
def permutationAssignment (lanes : Nat → Nat) : Nat → Nat :=
  fun column => if column = 0 then 1 else lanes (column - 1)

@[irreducible] def permute (lanes : Nat → Nat) (lane : Nat) : Nat :=
  permuteState (permutationAssignment lanes) (601 + lane)

theorem permute_eq (lanes : Nat → Nat) (lane : Nat) :
    permute lanes lane =
      permuteState (permutationAssignment lanes) (601 + lane) := by
  rw [permute]

theorem permute_congr {left right : Nat → Nat}
    (equalInputs : ∀ lane, lane < 8 → left lane = right lane)
    (lane : Nat) : permute left lane = permute right lane := by
  unfold permute permuteState
  apply congrArg (fun state => interpret state (601 + lane))
  funext column
  unfold inputOnly
  by_cases columnLt : column < 9
  · simp only [columnLt, ↓reduceIte]
    by_cases columnZero : column = 0
    · simp [columnZero, permutationAssignment]
    · have columnPositive : 0 < column := Nat.pos_of_ne_zero columnZero
      simp only [permutationAssignment, columnZero, ↓reduceIte]
      exact equalInputs (column - 1) (by omega)
  · simp [columnLt]

theorem outputs_known :
    ∀ column ∈ outputColumns, column ∈ knownAfter inputColumns definitions := by
  native_decide

theorem interpret_output_eq_permuteState (state : Nat → Nat) :
    ∀ column ∈ outputColumns,
      interpret state column = permuteState state column := by
  have inputsAgree : AgreeOn state (inputOnly state) inputColumns := by
    intro column member
    simp only [inputColumns, List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
      simp [inputOnly]
  have executionAgree := run_congr definitions_wellFormed inputsAgree
  intro column member
  exact executionAgree column (outputs_known column member)

/-- Every satisfying exact artifact agrees with the executable SSA semantics
on all input and derived columns. -/
theorem poseidon2Permutation_sound {z : Nat → Nat}
    (hcanon : ∀ column, z column < goldilocksP) (hone : z 0 = 1)
    (hsat : Satisfies rows z) :
    AgreeOn (interpret z) z (knownAfter inputColumns definitions) := by
  apply run_agrees_of_builder_satisfies definitions_wellFormed
  · intro column _
    rfl
  · exact hcanon
  · exact hone
  · exact definitions_canonical
  · exact hsat

/-- Reusable compiler rule for every production call site.  The Rust
emitter allocates the same 600-row permutation program under fresh column
names; a checked column renaming therefore transports the isolated artifact
theorem without re-proving Poseidon arithmetic at each call site. -/
theorem poseidon2Permutation_renamed_sound
    (columnMap : Nat → Nat)
    (mapsOne : columnMap 0 = 0)
    {z : Nat → Nat}
    (canonical : ∀ column, z column < goldilocksP)
    (one : z 0 = 1)
    (satisfies : Satisfies (rows.map (renameRow columnMap)) z) :
    AgreeOn (interpret (pullAssignment z columnMap))
      (pullAssignment z columnMap)
      (knownAfter inputColumns definitions) := by
  apply poseidon2Permutation_sound
  · intro column
    exact canonical (columnMap column)
  · simpa [pullAssignment, mapsOne] using one
  · intro row member
    apply (rowHolds_pull_iff z columnMap row).mpr
    exact satisfies (renameRow columnMap row)
      (List.mem_map.mpr ⟨row, member, rfl⟩)

/-- The eight artifact outputs are uniquely determined by the nine input
columns (constant one plus eight state lanes). -/
theorem poseidon2Permutation_outputs_unique
    {left right : Nat → Nat}
    (leftCanonical : ∀ column, left column < goldilocksP)
    (rightCanonical : ∀ column, right column < goldilocksP)
    (leftOne : left 0 = 1) (rightOne : right 0 = 1)
    (leftSat : Satisfies rows left) (rightSat : Satisfies rows right)
    (inputsEqual : AgreeOn left right inputColumns) :
    ∀ column ∈ outputColumns, left column = right column := by
  have leftRun := run_agrees_of_builder_satisfies definitions_wellFormed
    (z := left) (state := left) (by intro _ _; rfl)
    leftCanonical leftOne definitions_canonical leftSat
  have rightRun := run_agrees_of_builder_satisfies definitions_wellFormed
    (z := right) (state := left) inputsEqual
    rightCanonical rightOne definitions_canonical rightSat
  intro column member
  have known := outputs_known column member
  exact (leftRun column known).symm.trans (rightRun column known)

/-- Every canonical input state has a satisfying exact permutation witness,
constructed by the executable interpreter. -/
theorem poseidon2Permutation_complete {state : Nat → Nat}
    (canonical : ∀ column, state column < goldilocksP)
    (hone : state 0 = 1) : Satisfies rows (interpret state) := by
  exact run_satisfies_builder_rows definitions_wellFormed canonical
    (by decide) hone definitions_canonical

end Nightstream.Implementation.R1CS.Poseidon2PermutationSound
