import Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized

/-!
Contract: positional receipt ownership for the emitted program.

Owns: the receipt name, the row each receipt emits, and the proof that the
emitted program is exactly the image of a duplicate-free receipt list.

Does not own: what the rows mean, or what they cost.

## Why positional rather than by row value

`Poseidon2Layout.sboxRows_disjoint` and friends prove that distinct receipts
emit distinct row *values*.  That is true but it is the wrong ownership
contract: it makes structural `Row` equality the ABI, and two receipts emitting
equal rows is degenerate rather than incoherent.  A row program has positions,
and position is what a receipt should own.

The statement here is therefore: the emitted program is `allOwners.map`ped, the
owner list is duplicate-free, and it has the same length.  So position `i` of
the program is emitted by owner `i` and by no other — exactly one receipt per
row, with no appeal to row values at all.  The value-level lemmas remain true
and are now corollaries rather than the contract.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Program
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Schedule
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Normalized

/-! ## Receipts -/

/-- The receipt that emits a row: one step of one S-box, or one lane's terminal
binding. -/
inductive RowOwner where
  | sbox : Fin sboxCount → Fin columnsPerSbox → RowOwner
  | binding : Fin width → RowOwner
deriving DecidableEq

/-- The four rows of an S-box, indexed by step. -/
def sboxRowAt (frame : SboxFrame) : Fin columnsPerSbox → Row
  | ⟨0, _⟩ => rowSquare frame
  | ⟨1, _⟩ => rowFourth frame
  | ⟨2, _⟩ => rowSixth frame
  | _ => rowSeventh frame

theorem sboxRows_eq_map (frame : SboxFrame) :
    sboxRows frame
      = (List.finRange columnsPerSbox).map (sboxRowAt frame) := rfl

/-- The row a receipt emits, before normalization. -/
def ownedRow (layout : Layout) (constants : Constants) : RowOwner → Row
  | .sbox index step =>
      sboxRowAt (frameAt layout index (scheduleOf layout constants index)) step
  | .binding lane => bindRow (finalState layout lane) (layout.outputPort lane)

/-- Every receipt, in the order the program emits them. -/
def allOwners : List RowOwner :=
  (List.finRange sboxCount).flatMap
      (fun index => (List.finRange columnsPerSbox).map (RowOwner.sbox index))
    ++ (List.finRange width).map RowOwner.binding

/-! ## The program is exactly the receipts' image -/

theorem canonicalProgram_eq_map_owners
    (layout : Layout) (constants : Constants) :
    canonicalProgram layout constants
      = allOwners.map (ownedRow layout constants) := by
  unfold canonicalProgram permutationProgram sboxProgram bindingProgram
    terminalBindingRows allOwners
  rw [List.map_append, List.map_flatMap, List.map_map]
  congr 1

/-- **The emitted program is the receipts' image.** -/
theorem normalizedCanonicalProgram_eq_map_owners
    (layout : Layout) (constants : Constants) :
    normalizedCanonicalProgram layout constants
      = allOwners.map (fun owner => normalizeRow (ownedRow layout constants owner)) := by
  unfold normalizedCanonicalProgram normalizeProgram
  rw [canonicalProgram_eq_map_owners, List.map_map]
  rfl

/-! ## The receipt list is duplicate-free and the right length -/

theorem allOwners_length : allOwners.length = 352 := by
  unfold allOwners
  rw [List.length_append, List.length_map, List.length_finRange,
    length_flatMap_uniform _ _ columnsPerSbox (by intro x; simp),
    List.length_finRange]
  decide

/-- **An injective map preserves `Nodup`.**

The forward direction of `nodup_of_map_nodup`, which core Lean does not have —
`List.Nodup.map` is Mathlib's.  Kept beside its converse so a third private copy
does not get written. -/
theorem nodup_map_of_injective {α β : Type} (f : α → β)
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

/-- `List.Nodup.of_map` is not available without Mathlib: a duplicate in the
source would survive the map. -/
theorem nodup_of_map_nodup {α β : Type} (list : List α) (f : α → β)
    (nodup : (list.map f).Nodup) : list.Nodup := by
  induction list with
  | nil => simp
  | cons head tail hypothesis =>
      rw [List.map_cons, List.nodup_cons] at nodup
      rw [List.nodup_cons]
      exact ⟨fun member => nodup.1 (List.mem_map.2 ⟨head, member, rfl⟩),
        hypothesis nodup.2⟩

/-- A receipt's position in the program.  S-box `index` step `s` sits at
`4·index + s`; binding lane `l` at `344 + l`. -/
def ownerIndex : RowOwner → Nat
  | .sbox index step => columnsPerSbox * index.val + step.val
  | .binding lane => 344 + lane.val

set_option maxRecDepth 100000 in
theorem allOwners_index_eq : allOwners.map ownerIndex = List.range 352 := by
  decide

theorem allOwners_nodup : allOwners.Nodup :=
  nodup_of_map_nodup _ ownerIndex (allOwners_index_eq ▸ List.nodup_range)

/-- **Exactly one receipt per emitted row.**  The program and the receipt list
have equal length, the receipt list repeats nothing, and position `i` of the
program is emitted by receipt `i`.  Nothing here compares row values, so two
receipts emitting structurally equal rows would still be owned separately —
which is the correct contract for a row program. -/
theorem ownership_is_positional
    (layout : Layout) (constants : Constants) :
    (normalizedCanonicalProgram layout constants).length = allOwners.length
      ∧ allOwners.Nodup
      ∧ normalizedCanonicalProgram layout constants
          = allOwners.map
              (fun owner => normalizeRow (ownedRow layout constants owner)) := by
  refine ⟨?_, allOwners_nodup, normalizedCanonicalProgram_eq_map_owners _ _⟩
  rw [normalizedCanonicalProgram_length, allOwners_length]

/-- The receipt families are exactly the 344 S-box steps and the 8 bindings. -/
theorem allOwners_split :
    ((List.finRange sboxCount).flatMap
      (fun index => (List.finRange columnsPerSbox).map (RowOwner.sbox index))).length
        = 344
      ∧ ((List.finRange width).map RowOwner.binding).length = 8 := by
  constructor
  · rw [length_flatMap_uniform _ _ columnsPerSbox (by intro x; simp),
      List.length_finRange]
    decide
  · rw [List.length_map, List.length_finRange]
    decide


/-! ## Ownership on a carried entry

The sponge's calls are entered on carried states, so their programs are
`canonicalProgramFrom` rather than `canonicalProgram`.  Ownership is unchanged
in shape: the receipt list is the same `allOwners`, only the row each receipt
emits differs, and only in its scheduled input. -/

def ownedRowFrom (layout : Layout) (entry : State) (constants : Constants) :
    RowOwner → Row
  | .sbox index step =>
      sboxRowAt
        (frameAt layout index (scheduleOfFrom layout entry constants index)) step
  | .binding lane => bindRow (finalState layout lane) (layout.outputPort lane)

theorem canonicalProgramFrom_eq_map_owners
    (layout : Layout) (entry : State) (constants : Constants) :
    canonicalProgramFrom layout entry constants
      = allOwners.map (ownedRowFrom layout entry constants) := by
  unfold canonicalProgramFrom permutationProgram sboxProgram bindingProgram
    terminalBindingRows allOwners
  rw [List.map_append, List.map_flatMap, List.map_map]
  congr 1

/-- **Exactly one receipt per row of a carried-entry call.**  Same statement as
`ownership_is_positional`, at an arbitrary entry. -/
theorem ownership_is_positional_from
    (layout : Layout) (entry : State) (constants : Constants) :
    (canonicalProgramFrom layout entry constants).length = allOwners.length
      ∧ allOwners.Nodup
      ∧ canonicalProgramFrom layout entry constants
          = allOwners.map (ownedRowFrom layout entry constants) := by
  refine ⟨?_, allOwners_nodup, canonicalProgramFrom_eq_map_owners _ _ _⟩
  rw [canonicalProgramFrom_length, allOwners_length]

end Nightstream.Implementation.R1CS.Canonical.Poseidon2Ownership
