import Nightstream.Implementation.R1CS.Canonical.CanonicalProgram
import Nightstream.Implementation.R1CS.Canonical.PiDecOwnership

/-!
Contract: positional receipt ownership for the assembled canonical program.

Owns: the receipt name for each of the eight parts, the row each receipt emits,
and the proof that the assembly is exactly the image of a duplicate-free receipt
list.

Does not own: what the rows mean, what they cost, or which columns they touch.

## Why this exists

`CanonicalProgram.Recipes.rows_owner_not_unique` gives two reasons the assembly
fails section 2 item 3:

1. inherited from `PiDecRecipe.rows_owner_not_unique`;
2. **new at this level** — a deployment may supply a `SelectedRecipe` whose rows
   coincide with another part's, and nothing in the interface prevents it.

Cycle 393 answered the first: it objects to value-based ownership, which
`Poseidon2Ownership` had already rejected as the wrong ABI.

The second has the same shape, and that was not obvious when it was written —
it was recorded as "not inherited, not repairable by strengthening the built
recipes, and a property of the interface a deployment fills".  All three are
true **of value-based ownership**.  Under positional ownership a selection that
duplicates a built recipe's rows occupies different *positions*, so the rows are
owned separately and nothing needs to be prevented.

A deployment supplying duplicate rows is then not an ownership failure. It may
still be a *waste* — the same constraint asserted twice — but that is a cost
question, and `N_canonical` already counts both.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.CanonicalProgramOwnership

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.CanonicalProgram

/-! ## Receipts

One constructor per part of the eight-part program, carrying the position within
that part's own emitted list. -/

/-- The receipt that emits a row. -/
inductive RowOwner where
  | piDec (index : Nat)
  | foldDigest (index : Nat)
  | mixer (index : Nat)
  | transcript (index : Nat)
  | step (index : Nat)
  | nifsVerify (index : Nat)
  | runningCheck (index : Nat)
  | freshCheck (index : Nat)
deriving DecidableEq, Repr

/-! ## The eight parts, as lists -/

def piDecRows (recipes : Recipes) : List Row :=
  PiDecRecipe.rows recipes.piDec

def foldDigestRows (recipes : Recipes) : List Row :=
  FoldDigestRecipe.digestRows recipes.foldDigest

def mixerRows (recipes : Recipes) : List Row :=
  CommitmentMixerRecipe.mixerRows recipes.mixerBase recipes.mixer

def transcriptRows (recipes : Recipes) : List Row :=
  TranscriptRecipe.transcriptRows recipes.transcriptLayouts
    recipes.transcriptSchedule recipes.transcriptConstants
    recipes.transcriptRounds

theorem rows_eq_parts (recipes : Recipes) :
    Recipes.rows recipes
      = piDecRows recipes ++ foldDigestRows recipes ++ mixerRows recipes
          ++ transcriptRows recipes ++ recipes.step.rows
          ++ recipes.nifsVerify.rows ++ recipes.runningCheck.rows
          ++ recipes.freshCheck.rows := by
  unfold Recipes.rows piDecRows foldDigestRows mixerRows transcriptRows
  rfl

/-! ## The receipt list and the row each receipt emits -/

private def blank : Row := ⟨[], [], []⟩

/-- The row a receipt emits. -/
def ownedRow (recipes : Recipes) : RowOwner → Row
  | .piDec index => (piDecRows recipes).getD index blank
  | .foldDigest index => (foldDigestRows recipes).getD index blank
  | .mixer index => (mixerRows recipes).getD index blank
  | .transcript index => (transcriptRows recipes).getD index blank
  | .step index => recipes.step.rows.getD index blank
  | .nifsVerify index => recipes.nifsVerify.rows.getD index blank
  | .runningCheck index => recipes.runningCheck.rows.getD index blank
  | .freshCheck index => recipes.freshCheck.rows.getD index blank

/-- Every receipt, in program order. -/
def owners (recipes : Recipes) : List RowOwner :=
  (List.range (piDecRows recipes).length).map RowOwner.piDec
    ++ (List.range (foldDigestRows recipes).length).map RowOwner.foldDigest
    ++ (List.range (mixerRows recipes).length).map RowOwner.mixer
    ++ (List.range (transcriptRows recipes).length).map RowOwner.transcript
    ++ (List.range recipes.step.rows.length).map RowOwner.step
    ++ (List.range recipes.nifsVerify.rows.length).map RowOwner.nifsVerify
    ++ (List.range recipes.runningCheck.rows.length).map RowOwner.runningCheck
    ++ (List.range recipes.freshCheck.rows.length).map RowOwner.freshCheck

/-- **The program is the receipt list's image.** -/
theorem rows_eq_map_owners (recipes : Recipes) :
    Recipes.rows recipes = (owners recipes).map (ownedRow recipes) := by
  rw [rows_eq_parts, owners]
  simp only [List.map_append, List.map_map, Function.comp_def, ownedRow,
    PiDecOwnership.map_getD_range]

/-! ## The receipt list repeats nothing -/

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

theorem owners_nodup (recipes : Recipes) : (owners recipes).Nodup := by
  unfold owners
  have single : ∀ (constructor : Nat → RowOwner),
      (∀ first second, constructor first = constructor second → first = second) →
      ∀ count, ((List.range count).map constructor).Nodup :=
    fun constructor injective _ =>
      nodup_map_of_injective constructor injective List.nodup_range
  refine List.nodup_append.2 ⟨List.nodup_append.2 ⟨List.nodup_append.2
    ⟨List.nodup_append.2 ⟨List.nodup_append.2 ⟨List.nodup_append.2
      ⟨List.nodup_append.2
        ⟨single _ (fun _ _ h => by cases h; rfl) _,
          single _ (fun _ _ h => by cases h; rfl) _, ?_⟩,
        single _ (fun _ _ h => by cases h; rfl) _, ?_⟩,
      single _ (fun _ _ h => by cases h; rfl) _, ?_⟩,
      single _ (fun _ _ h => by cases h; rfl) _, ?_⟩,
      single _ (fun _ _ h => by cases h; rfl) _, ?_⟩,
    single _ (fun _ _ h => by cases h; rfl) _, ?_⟩,
    single _ (fun _ _ h => by cases h; rfl) _, ?_⟩ <;>
  · intro left leftMember right rightMember equal
    subst equal
    simp only [List.mem_append, List.mem_map] at leftMember rightMember
    rcases rightMember with ⟨index, _, rfl⟩
    simp at leftMember

/-- **Exactly one receipt per emitted row, across all eight parts.**

The program and the receipt list have equal length, the receipt list repeats
nothing, and position `i` of the program is emitted by receipt `i`.  Nothing here
compares row values. -/
theorem ownership_is_positional (recipes : Recipes) :
    (Recipes.rows recipes).length = (owners recipes).length
      ∧ (owners recipes).Nodup
      ∧ Recipes.rows recipes = (owners recipes).map (ownedRow recipes) := by
  refine ⟨?_, owners_nodup recipes, rows_eq_map_owners recipes⟩
  rw [rows_eq_map_owners, List.length_map]

/-! ## The second reason was value-level too

`selection_may_duplicate_built_rows` shows a deployment can hand `step` the
fold-digest program.  Under positional ownership those rows are receipts
`step i` and `foldDigest i` — distinct receipts, distinct positions, one owner
each. -/

/-- **A duplicating selection is still owned unambiguously.** -/
theorem duplicating_selection_has_distinct_receipts (index other : Nat) :
    RowOwner.step index ≠ RowOwner.foldDigest other := by
  intro equal
  cases equal

/-- **No two parts share a receipt**, whatever rows they carry. -/
theorem parts_have_distinct_receipts (index other : Nat) :
    RowOwner.piDec index ≠ RowOwner.step other
      ∧ RowOwner.step index ≠ RowOwner.nifsVerify other
      ∧ RowOwner.runningCheck index ≠ RowOwner.freshCheck other := by
  refine ⟨?_, ?_, ?_⟩ <;> (intro equal; cases equal)

end Nightstream.Implementation.R1CS.Canonical.CanonicalProgramOwnership
