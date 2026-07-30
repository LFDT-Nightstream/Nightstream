import Nightstream.Implementation.R1CS.Canonical.PiDecRecipe

/-!
Contract: positional receipt ownership for Π_DEC's emitted program.

Owns: the receipt name, the row each receipt emits, and the proof that Π_DEC's
program is exactly the image of a duplicate-free receipt list.

Does not own: what the rows mean, what they cost, or which columns they touch —
those are `PiDecRecipe`'s.

## Why this exists

`PiDecRecipe.rows_owner_not_unique` exhibits one row attributable to two of
Π_DEC's own recomposition receipts, and has been carried since cycle 368 as an
obstruction to section 2 item 3.

**It is an obstruction to the wrong contract.**  `Poseidon2Ownership` settled
which contract is right, in its own header: making structural `Row` equality the
ABI is the error, because "two receipts emitting equal rows is degenerate rather
than incoherent".  Two Π_DEC checks that constrain the same relation *should*
emit the same row; a program that deduplicated them would be a different program.

A row program has positions, and position is what a receipt owns.  That contract
was built for Poseidon2 and never applied here — so item 3 was recorded as failing
for Π_DEC when what had failed was an ownership notion this tree had already
rejected.

`rows_owner_not_unique` stays true and stays guarded.  It is now a statement
about row *values*, which is not the ABI.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.PiDecOwnership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiDecRecipe

/-! ## Receipts

One constructor per atom of the five-part program, carrying the position within
that atom's own emitted list. -/

/-- The receipt that emits a row. -/
inductive RowOwner where
  | recomposition (index : Nat)
  | digit (index : Nat)
  | inactive (index : Nat)
  | padding (index : Nat)
  | consistency (index : Nat)
deriving DecidableEq, Repr

/-! ## The five atoms, as lists -/

def recompositionRows (claim : Decomposition) : List Row :=
  KRecomposition.recompositionsRows claim.base claim.recompositions

def digitRows (claim : Decomposition) : List Row :=
  KLowNormBatch.batchRows claim.xDigits

def inactiveRows (claim : Decomposition) : List Row :=
  claim.inactiveX.flatMap KZeroCheck.zeroRows

def paddingRows (claim : Decomposition) : List Row :=
  KZeroCheck.paddingRows claim.yRingPadding

def consistencyRows (claim : Decomposition) : List Row :=
  KConsistency.consistencyRows claim.consistency

theorem rows_eq_atoms (claim : Decomposition) :
    rows claim
      = recompositionRows claim ++ digitRows claim ++ inactiveRows claim
          ++ paddingRows claim ++ consistencyRows claim := by
  unfold rows recompositionRows digitRows inactiveRows paddingRows
    consistencyRows
  rfl

/-! ## Reading a list back from its positions

The one general fact this construction needs: a list is recovered by reading it
at every position.  Everything else is bookkeeping over the five atoms. -/

private def blank : Row := ⟨[], [], []⟩

theorem map_getD_range {α : Type} (list : List α) (fallback : α) :
    (List.range list.length).map (fun index => list.getD index fallback)
      = list := by
  induction list with
  | nil => rfl
  | cons head tail hypothesis =>
      rw [List.length_cons, List.range_succ_eq_map, List.map_cons,
        List.map_map]
      refine congrArg (head :: ·) ?_
      exact hypothesis

/-! ## The receipt list and the row each receipt emits -/

/-- The row a receipt emits. -/
def ownedRow (claim : Decomposition) : RowOwner → Row
  | .recomposition index => (recompositionRows claim).getD index blank
  | .digit index => (digitRows claim).getD index blank
  | .inactive index => (inactiveRows claim).getD index blank
  | .padding index => (paddingRows claim).getD index blank
  | .consistency index => (consistencyRows claim).getD index blank

/-- Every receipt, in program order. -/
def owners (claim : Decomposition) : List RowOwner :=
  (List.range (recompositionRows claim).length).map RowOwner.recomposition
    ++ (List.range (digitRows claim).length).map RowOwner.digit
    ++ (List.range (inactiveRows claim).length).map RowOwner.inactive
    ++ (List.range (paddingRows claim).length).map RowOwner.padding
    ++ (List.range (consistencyRows claim).length).map RowOwner.consistency

/-- **The program is the receipt list's image.** -/
theorem rows_eq_map_owners (claim : Decomposition) :
    rows claim = (owners claim).map (ownedRow claim) := by
  rw [rows_eq_atoms, owners]
  simp only [List.map_append, List.map_map, Function.comp_def, ownedRow,
    map_getD_range]

/-! ## The receipt list repeats nothing

Each atom's receipts are a range under an injective constructor, and the five
constructors are pairwise distinct, so no receipt appears twice. -/

private theorem nodup_map_of_injective {α β : Type} (f : α → β)
    (injective : ∀ first second, f first = f second → first = second) :
    ∀ {list : List α}, list.Nodup → (list.map f).Nodup
  | [], _ => by simp
  | head :: tail, nodup => by
      rw [List.nodup_cons] at nodup
      rw [List.map_cons, List.nodup_cons]
      refine ⟨?_, nodup_map_of_injective f injective nodup.2⟩
      intro member
      rcases List.mem_map.1 member with ⟨other, otherMember, equal⟩
      exact nodup.1 (injective _ _ equal ▸ otherMember)

private theorem nodup_map_range {constructor : Nat → RowOwner}
    (injective : ∀ first second, constructor first = constructor second →
      first = second) (count : Nat) :
    ((List.range count).map constructor).Nodup :=
  nodup_map_of_injective constructor injective List.nodup_range

theorem owners_nodup (claim : Decomposition) : (owners claim).Nodup := by
  unfold owners
  have single : ∀ (constructor : Nat → RowOwner),
      (∀ first second, constructor first = constructor second → first = second) →
      ∀ count, ((List.range count).map constructor).Nodup :=
    fun _ injective _ => nodup_map_range injective _
  refine List.nodup_append.2 ⟨List.nodup_append.2 ⟨List.nodup_append.2
    ⟨List.nodup_append.2
      ⟨single _ (fun _ _ h => by cases h; rfl) _,
        single _ (fun _ _ h => by cases h; rfl) _, ?_⟩,
      single _ (fun _ _ h => by cases h; rfl) _, ?_⟩,
    single _ (fun _ _ h => by cases h; rfl) _, ?_⟩,
    single _ (fun _ _ h => by cases h; rfl) _, ?_⟩ <;>
  · intro left leftMember right rightMember equal
    subst equal
    simp only [List.mem_append, List.mem_map] at leftMember rightMember
    rcases rightMember with ⟨index, _, rfl⟩
    simp at leftMember

/-- **Exactly one receipt per emitted row.**

The program and the receipt list have equal length, the receipt list repeats
nothing, and position `i` of the program is emitted by receipt `i`.  Nothing here
compares row values, so the two checks of `rows_owner_not_unique` are owned
separately — which is the correct contract for a row program. -/
theorem ownership_is_positional (claim : Decomposition) :
    (rows claim).length = (owners claim).length
      ∧ (owners claim).Nodup
      ∧ rows claim = (owners claim).map (ownedRow claim) := by
  refine ⟨?_, owners_nodup claim, rows_eq_map_owners claim⟩
  rw [rows_eq_map_owners, List.length_map]

/-- **The value-level failure is not a failure of this contract.**

`rows_owner_not_unique` exhibits one row value in two recomposition receipts.
Under positional ownership those are receipts `recomposition i` and
`recomposition j` with `i ≠ j`, both present and both distinct — so the program
still has exactly one receipt per row. -/
theorem duplicate_values_have_distinct_receipts
    (first second : Nat) (different : first ≠ second) :
    RowOwner.recomposition first ≠ RowOwner.recomposition second := by
  intro equal
  cases equal
  exact different rfl

end Nightstream.Implementation.R1CS.Canonical.PiDecOwnership
