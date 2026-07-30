import Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe
import Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: a same-payload diagnostic relation between two fixed-23 preimages.

Owns: the emitted linking program, its derived count, soundness, honest
completeness, conservation and cost.

Does not own: the sponge, either hash's placement, what the digests mean, or
the relation between the actual F-prime `hashPrior` and `hashNext` calls.

## Scope

The program below is exact for two preimages that have the same payload and
differ only in the normalized iteration coordinate. It is not a whole-Step
program.

The real F-prime calls do not have one shared payload. `hashPrior` reads the
current state and running input. `hashNext` reads the next state and folded
running output. These values can differ in an honest transition. Therefore,
placing these rows between the real calls would add false equalities and can
reject an honest transition. The individual hash recipes already apply the
correct prior/next mode to their own authoritative operands. A whole-program
assembly must use those two recipes and must not place this diagnostic program.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Reference
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Sponge23
open Nightstream.Implementation.R1CS.Canonical.Poseidon2RoundInduction
open Nightstream.Implementation.R1CS.Canonical.Poseidon2HashRecipe

/-! ## The two instances' preimage columns -/

/-- Slot `index` of `hashPrior`'s preimage. -/
def priorColumn (index : Nat) : Nat :=
  relocate hashPriorBase (inputColumn index)

/-- Slot `index` of `hashNext`'s preimage. -/
def nextColumn (index : Nat) : Nat :=
  relocate hashNextBase (inputColumn index)

theorem inputBase_eq : inputBase = 2527 := by decide

theorem priorColumn_ne_nextColumn (first second : Nat)
    (firstLt : first < sponge23Fields) (secondLt : second < sponge23Fields) :
    priorColumn first ≠ nextColumn second := by
  simp only [sponge23Fields] at firstLt secondLt
  have firstNonZero : inputColumn first ≠ 0 := by
    simp only [inputColumn, inputBase_eq]; omega
  have secondNonZero : inputColumn second ≠ 0 := by
    simp only [inputColumn, inputBase_eq]; omega
  have priorEq : priorColumn first = 2527 + first := by
    rw [priorColumn, relocate_pos _ _ firstNonZero]
    simp only [hashPriorBase, inputColumn, inputBase_eq, Nat.zero_add]
  have nextEq : nextColumn second = 2550 + (2527 + second) := by
    rw [nextColumn, relocate_pos _ _ secondNonZero]
    simp only [hashNextBase, spongeColumnTotal_eq, inputColumn, inputBase_eq]
  rw [priorEq, nextEq]
  omega

/-! ## The emitted program

One increment at slot zero, one equality at every later slot. -/

/-- **The emitted separation program.** -/
def separationRows : List Row :=
  KEquality.equalityRow [(nextColumn 0, 1)] [(priorColumn 0, 1), (0, 1)]
    :: (List.range (sponge23Fields - 1)).map
        (fun offset =>
          KEquality.equalityRow [(nextColumn (offset + 1), 1)]
            [(priorColumn (offset + 1), 1)])

/-- **The derived row count**, from the emitted list: one per preimage slot. -/
theorem separationRows_length :
    separationRows.length = sponge23Fields := by
  simp only [separationRows, List.length_cons, List.length_map,
    List.length_range, sponge23Fields]

theorem separationRows_length_eq : separationRows.length = 23 :=
  separationRows_length

/-- The linking allocates nothing: both instances' preimage columns already
exist. -/
def separationColumns : List Nat := []

theorem separationColumns_length : separationColumns.length = 0 := rfl

theorem separationColumns_nodup : separationColumns.Nodup := List.nodup_nil

/-! ## The preimages the two instances read -/

def priorPreimage (z : Nat → Nat) : Preimage :=
  fun index => z (priorColumn index.val)

def nextPreimage (z : Nat → Nat) : Preimage :=
  fun index => z (nextColumn index.val)

/-! ## Soundness

Satisfaction forces the next preimage to be the separated form of the prior
one, under this module's same-payload diagnostic relation. -/

private theorem slot_zero_forced
    (z : Nat → Nat) (wire : z 0 = 1)
    (residues : ∀ column, z column < goldilocksP)
    (satisfied : Satisfies separationRows z) :
    z (nextColumn 0) = (z (priorColumn 0) + 1) % goldilocksP := by
  have head : RowHolds z
      (KEquality.equalityRow [(nextColumn 0, 1)]
        [(priorColumn 0, 1), (0, 1)]) :=
    satisfied _ (by simp [separationRows])
  have equal := (KEquality.equalityRow_iff z _ _ wire).1 head
  rw [lcEval_singleton z (nextColumn 0) (residues _)] at equal
  rw [equal]
  simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul, wire]

private theorem slot_later_forced
    (z : Nat → Nat) (wire : z 0 = 1)
    (residues : ∀ column, z column < goldilocksP)
    (satisfied : Satisfies separationRows z)
    (offset : Nat) (bound : offset < sponge23Fields - 1) :
    z (nextColumn (offset + 1)) = z (priorColumn (offset + 1)) := by
  have member : KEquality.equalityRow [(nextColumn (offset + 1), 1)]
      [(priorColumn (offset + 1), 1)] ∈ separationRows := by
    refine List.mem_cons_of_mem _ (List.mem_map.2 ⟨offset, ?_, rfl⟩)
    exact List.mem_range.2 bound
  have equal := (KEquality.equalityRow_iff z _ _ wire).1 (satisfied _ member)
  rw [lcEval_singleton z _ (residues _), lcEval_singleton z _ (residues _)]
    at equal
  exact equal

/-- **The emitted program applies the separator.**

Satisfaction forces `hashNext`'s preimage to be `separatedPreimage true` of
`hashPrior`'s — not by the profile's promise, but by the rows. -/
theorem separationRows_applies
    (z : Nat → Nat) (wire : z 0 = 1)
    (residues : ∀ column, z column < goldilocksP)
    (satisfied : Satisfies separationRows z) :
    nextPreimage z = separatedPreimage true (priorPreimage z) := by
  funext index
  have indexLt : index.val < sponge23Fields := index.isLt
  simp only [nextPreimage, separatedPreimage, priorPreimage, and_true]
  by_cases atZero : index.val = 0
  · rw [if_pos atZero, atZero]
    exact slot_zero_forced z wire residues satisfied
  · rw [if_neg atZero]
    cases hval : index.val with
    | zero => exact absurd hval atZero
    | succ previous =>
        rw [hval] at indexLt
        exact slot_later_forced z wire residues satisfied previous (by
          simp only [sponge23Fields] at indexLt ⊢
          omega)

/-! ## Honest completeness

A caller whose two instances already read a preimage and its separated form
satisfies the program under its own assignment.  Nothing is allocated, so there
is no witness to extend. -/

theorem separationRows_honest
    (z : Nat → Nat) (wire : z 0 = 1)
    (residues : ∀ column, z column < goldilocksP)
    (separated : nextPreimage z = separatedPreimage true (priorPreimage z)) :
    Satisfies separationRows z := by
  intro row member
  simp only [separationRows, List.mem_cons] at member
  rcases member with rfl | later
  · refine (KEquality.equalityRow_iff z _ _ wire).2 ?_
    have atZero : z (nextColumn 0)
        = (z (priorColumn 0) + 1) % goldilocksP := by
      have := congrFun separated ⟨0, by decide⟩
      simpa only [nextPreimage, separatedPreimage, priorPreimage, and_true,
        reduceIte] using this
    rw [lcEval_singleton z (nextColumn 0) (residues _), atZero]
    simp only [lcEval, List.foldl, Nat.zero_add, Nat.one_mul, wire]
  · rcases List.mem_map.1 later with ⟨offset, offsetMember, rfl⟩
    have bound : offset + 1 < sponge23Fields := by
      have := List.mem_range.1 offsetMember
      simp only [sponge23Fields] at this ⊢
      omega
    refine (KEquality.equalityRow_iff z _ _ wire).2 ?_
    have same : z (nextColumn (offset + 1)) = z (priorColumn (offset + 1)) := by
      have := congrFun separated ⟨offset + 1, bound⟩
      simpa only [nextPreimage, separatedPreimage, priorPreimage,
        Nat.succ_ne_zero, false_and, reduceIte] using this
    rw [lcEval_singleton z _ (residues _), lcEval_singleton z _ (residues _),
      same]

/-- A changed nonzero slot is rejected by this same-payload diagnostic.

This theorem is the fail-closed scope guard: the program cannot be used as the
general F-prime prior/next relation because an honest Step can change payload
coordinates. -/
theorem changed_tail_rejected
    (z : Nat → Nat) (wire : z 0 = 1)
    (residues : ∀ column, z column < goldilocksP)
    (offset : Nat) (bound : offset < sponge23Fields - 1)
    (changed :
      z (nextColumn (offset + 1)) ≠ z (priorColumn (offset + 1))) :
    ¬ Satisfies separationRows z := by
  intro satisfied
  exact changed (slot_later_forced z wire residues satisfied offset bound)

/-! ## Row ownership

Section 2 item 3, positional — the contract `Poseidon2Ownership` settled and
`PiDecOwnership` and `CanonicalProgramOwnership` follow.

This recipe was added in cycle 396 and its item 3 was not written then.  The
checklist has to be walked for a *new* recipe too, not only for the ones a matrix
already lists. -/

/-- The receipt that emits a row: the slot-zero increment, or one later slot's
equality. -/
inductive RowOwner where
  | increment
  | equality (offset : Nat)
deriving DecidableEq, Repr

/-- The row a receipt emits. -/
def ownedRow : RowOwner → Row
  | .increment =>
      KEquality.equalityRow [(nextColumn 0, 1)] [(priorColumn 0, 1), (0, 1)]
  | .equality offset =>
      KEquality.equalityRow [(nextColumn (offset + 1), 1)]
        [(priorColumn (offset + 1), 1)]

/-- Every receipt, in program order. -/
def owners : List RowOwner :=
  RowOwner.increment :: (List.range (sponge23Fields - 1)).map RowOwner.equality

/-- **The program is the receipt list's image.** -/
theorem separationRows_eq_map_owners :
    separationRows = owners.map ownedRow := by
  simp only [separationRows, owners, List.map_cons, List.map_map,
    Function.comp_def, ownedRow]

theorem owners_nodup : owners.Nodup := by
  rw [owners, List.nodup_cons]
  constructor
  · intro member
    rcases List.mem_map.1 member with ⟨offset, _, equal⟩
    cases equal
  · refine Poseidon2Ownership.nodup_map_of_injective _ ?_ List.nodup_range
    intro first second equal
    cases equal
    rfl

/-- **Exactly one receipt per emitted row.** -/
theorem ownership_is_positional :
    separationRows.length = owners.length
      ∧ owners.Nodup
      ∧ separationRows = owners.map ownedRow := by
  refine ⟨?_, owners_nodup, separationRows_eq_map_owners⟩
  rw [separationRows_eq_map_owners, List.length_map]

/-! ## Conservation -/

theorem separationRows_conservation
    (row : Row) (member : row ∈ separationRows) (column : Nat)
    (mentioned : Mentions row.a column ∨ Mentions row.b column
      ∨ Mentions row.c column) :
    column = 0 ∨ (∃ index, index < sponge23Fields ∧ column = priorColumn index)
      ∨ ∃ index, index < sponge23Fields ∧ column = nextColumn index := by
  simp only [separationRows, List.mem_cons] at member
  rcases member with rfl | later
  · simp only [KEquality.equalityRow] at mentioned
    rcases mentioned with a | b | c
    · exact Or.inr (Or.inr ⟨0, by decide, by
        simpa only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] using a⟩)
    · exact Or.inl (by
        simpa only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] using b)
    · simp only [Mentions, List.map_cons, List.map_nil, List.mem_cons,
        List.not_mem_nil, or_false] at c
      rcases c with rfl | rfl
      · exact Or.inr (Or.inl ⟨0, by decide, rfl⟩)
      · exact Or.inl rfl
  · rcases List.mem_map.1 later with ⟨offset, offsetMember, rfl⟩
    have bound : offset + 1 < sponge23Fields := by
      have := List.mem_range.1 offsetMember
      simp only [sponge23Fields] at this ⊢
      omega
    simp only [KEquality.equalityRow] at mentioned
    rcases mentioned with a | b | c
    · exact Or.inr (Or.inr ⟨offset + 1, bound, by
        simpa only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] using a⟩)
    · exact Or.inl (by
        simpa only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] using b)
    · exact Or.inr (Or.inl ⟨offset + 1, bound, by
        simpa only [Mentions, List.map_cons, List.map_nil,
          List.mem_singleton] using c⟩)

/-! ## Cost -/

/-- **The linking's cost.**  One row per preimage slot, nothing allocated. -/
def separationCost : Lowering.Typed.Cost where
  recurringRows := sponge23Fields
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem separationCost_rows :
    separationRows.length = separationCost.recurringRows :=
  separationRows_length

theorem separationCost_columns :
    separationColumns.length = separationCost.auxiliaryColumns := rfl

/-! ## What this diagnostic owns

`Poseidon2HashRecipe.committed_separation_survives` says an applied separator is
not lost.  `separationRows_applies` says one is applied.  Together they are
an exact same-payload test: the emitted program both applies the separator and
carries it to the state.

This result does not connect the actual F-prime prior and next calls. Each call
must bind its own authoritative payload through its own hash recipe. -/

/-- **Applied and preserved, in one statement.** -/
theorem separation_applied_and_preserved
    (z : Nat → Nat) (state : Values) (wire : z 0 = 1)
    (residues : ∀ column, z column < goldilocksP)
    (satisfied : Satisfies separationRows z)
    (canonical : priorPreimage z ⟨0, by decide⟩ < goldilocksP) :
    nextPreimage z = separatedPreimage true (priorPreimage z)
      ∧ chunkAt (nextPreimage z) 0 ≠ chunkAt (priorPreimage z) 0
      ∧ absorbChunk (chunkAt (nextPreimage z) 0) state ⟨0, by decide⟩
          ≠ absorbChunk (chunkAt (priorPreimage z) 0) state ⟨0, by decide⟩ := by
  have applies := separationRows_applies z wire residues satisfied
  refine ⟨applies, ?_, ?_⟩
  · rw [applies]
    exact separatedPreimage_reaches_chunk_zero (priorPreimage z) canonical
  · rw [applies]
    exact separator_survives_absorption (priorPreimage z) state canonical

end Nightstream.Implementation.R1CS.Canonical.Poseidon2HashSeparation
