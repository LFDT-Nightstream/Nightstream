import Nightstream.Implementation.NebulaV2.Application.Wasm.StateCodec
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Core.Program

/-!
Contract: reusable exact R1CS range block for one narrow unsigned word.

Assurance tier: implementation model.

Owns one Boolean row per little-endian bit, one integer-safe weighted
recomposition row, sound extraction of the bound `value < 2^width`, and an
honest satisfying assignment theorem. The width must fit below Goldilocks.

Does not own absolute generated columns, row inclusion in a generated
relation, a multiword codec schema, or native parsing.

Emits constraints: yes.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.BoundedWordRows

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.NebulaV2.WasmStateCodec

structure Layout where
  width : Nat
  valueColumn : Nat
  bitStart : Nat
deriving DecidableEq, Repr

def Layout.bitColumn (layout : Layout) (offset : Nat) : Nat :=
  layout.bitStart + offset

def Layout.bitRows (layout : Layout) : List Row :=
  (List.range layout.width).map fun offset =>
    bitRow (layout.bitColumn offset)

def Layout.terms (layout : Layout) : List (Nat × Nat) :=
  (List.range layout.width).map fun offset =>
    (layout.bitColumn offset, 2 ^ offset)

def Layout.recompositionRow (layout : Layout) : Row :=
  builderLinearRow layout.valueColumn layout.terms

def rows (layout : Layout) : List Row :=
  layout.bitRows ++ [layout.recompositionRow]

def Layout.digits (layout : Layout) (assignment : Nat → Nat) : List Nat :=
  (List.range layout.width).map fun offset =>
    assignment (layout.bitColumn offset)

def decoded (layout : Layout) (assignment : Nat → Nat) : Nat :=
  Nat.ofDigits 2 (layout.digits assignment)

theorem Layout.bitRows_length (layout : Layout) :
    layout.bitRows.length = layout.width := by
  simp [Layout.bitRows]

theorem rows_length (layout : Layout) :
    (rows layout).length = layout.width + 1 := by
  simp [rows, layout.bitRows_length]

theorem Layout.digits_length (layout : Layout) (assignment : Nat → Nat) :
    (layout.digits assignment).length = layout.width := by
  simp [Layout.digits]

private theorem ofDigits_range_map
    (base count : Nat) (digit : Nat → Nat) :
    Nat.ofDigits base ((List.range count).map digit) =
      (List.range count).foldl
        (fun value index => value + base ^ index * digit index) 0 := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      simp [List.range_succ, Nat.ofDigits_append, inductionHypothesis]

theorem Layout.terms_canonical
    (layout : Layout) (fits : 2 ^ layout.width ≤ goldilocksP) :
    CanonicalTerms layout.terms := by
  intro term member
  rcases List.mem_map.mp member with ⟨offset, offsetMember, rfl⟩
  have offsetBound := List.mem_range.mp offsetMember
  constructor
  · positivity
  · exact (Nat.pow_lt_pow_right (by decide) offsetBound).trans_le fits

/-- Canonical coefficients for a word whose full range can exceed the field.
The caller must prove each used power of two is below Goldilocks. -/
theorem Layout.terms_canonical_of_weight_bound
    (layout : Layout)
    (weightBound : ∀ offset, offset < layout.width →
      2 ^ offset < goldilocksP) :
    CanonicalTerms layout.terms := by
  intro term member
  rcases List.mem_map.mp member with ⟨offset, offsetMember, rfl⟩
  have offsetBound := List.mem_range.mp offsetMember
  exact ⟨by positivity, weightBound offset offsetBound⟩

private theorem bitRow_mem_rows (layout : Layout)
    {offset : Nat} (bound : offset < layout.width) :
    bitRow (layout.bitColumn offset) ∈ rows layout := by
  apply List.mem_append_left
  exact List.mem_map.mpr
    ⟨offset, List.mem_range.mpr bound, rfl⟩

private theorem recompositionRow_mem_rows (layout : Layout) :
    layout.recompositionRow ∈ rows layout := by
  simp [rows]

theorem canonical_digits_of_rows
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    ∀ digit ∈ layout.digits assignment, digit < 2 := by
  intro digit member
  rcases List.mem_map.mp member with
    ⟨offset, offsetMember, digitEqual⟩
  subst digit
  have bound := List.mem_range.mp offsetMember
  have atMostOne := bitRow_le_one goldilocks_euclidPrime
    (canonical (layout.bitColumn offset)) one
    (holds _ (bitRow_mem_rows layout bound))
  omega

theorem decoded_lt
    {layout : Layout} {assignment : Nat → Nat}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    decoded layout assignment < 2 ^ layout.width := by
  have digitsBinary := canonical_digits_of_rows canonical one holds
  simpa [decoded, layout.digits_length] using
    Nat.ofDigits_lt_base_pow_length (by decide : 1 < 2) digitsBinary

theorem lcEval_terms_eq_decoded_of_bound
    {layout : Layout} {assignment : Nat → Nat}
    (fits : 2 ^ layout.width ≤ goldilocksP)
    (decodedBound : decoded layout assignment < 2 ^ layout.width) :
    lcEval assignment layout.terms = decoded layout assignment := by
  have belowModulus := decodedBound.trans_le fits
  rw [lcEval]
  simp only [Layout.terms, List.foldl_map]
  rw [← ofDigits_range_map 2 layout.width
    (fun offset => assignment (layout.bitColumn offset))]
  exact Nat.mod_eq_of_lt belowModulus

/-- Exact field evaluation when the particular decoded word, rather than
the full declared word range, is below Goldilocks. This is required for a
canonical 64-bit Goldilocks word because `2^64` exceeds the modulus. -/
theorem lcEval_terms_eq_decoded_of_field_bound
    {layout : Layout} {assignment : Nat → Nat}
    (decodedBelowField : decoded layout assignment < goldilocksP) :
    lcEval assignment layout.terms = decoded layout assignment := by
  rw [lcEval]
  simp only [Layout.terms, List.foldl_map]
  rw [← ofDigits_range_map 2 layout.width
    (fun offset => assignment (layout.bitColumn offset))]
  exact Nat.mod_eq_of_lt decodedBelowField

/-- A linked recomposition row is sound when another authority already proves
the declared digits form a value below the word range. This theorem lets a
caller reuse parser-owned binary public bits without emitting duplicate
Boolean rows. -/
theorem recompositionRow_sound_of_decoded_bound
    {layout : Layout} {assignment : Nat → Nat}
    (fits : 2 ^ layout.width ≤ goldilocksP)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (decodedBound : decoded layout assignment < 2 ^ layout.width)
    (holds : RowHolds assignment layout.recompositionRow) :
    assignment layout.valueColumn = decoded layout assignment := by
  have fieldEqual := builderLinearRow_sound canonical one
    layout.valueColumn layout.terms (layout.terms_canonical fits) holds
  exact fieldEqual.trans
    (lcEval_terms_eq_decoded_of_bound fits decodedBound)

/-- Sound recomposition for an exact parser-owned word whose decoded value
is below Goldilocks. No modulo alias can satisfy this theorem. -/
theorem recompositionRow_sound_of_field_bound
    {layout : Layout} {assignment : Nat → Nat}
    (canonicalTerms : CanonicalTerms layout.terms)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (decodedBelowField : decoded layout assignment < goldilocksP)
    (holds : RowHolds assignment layout.recompositionRow) :
    assignment layout.valueColumn = decoded layout assignment := by
  have fieldEqual := builderLinearRow_sound canonical one
    layout.valueColumn layout.terms canonicalTerms holds
  exact fieldEqual.trans
    (lcEval_terms_eq_decoded_of_field_bound decodedBelowField)

/-- Honest linked recomposition is complete when the output column is the
integer decoded from the declared source columns. -/
theorem recompositionRow_complete_of_decoded
    {layout : Layout} {assignment : Nat → Nat}
    (fits : 2 ^ layout.width ≤ goldilocksP)
    (one : assignment 0 = 1)
    (equal : assignment layout.valueColumn = decoded layout assignment)
    (decodedBound : decoded layout assignment < 2 ^ layout.width) :
    RowHolds assignment layout.recompositionRow := by
  apply builderLinearRow_complete one layout.valueColumn layout.terms
    (layout.terms_canonical fits)
  rw [lcEval_terms_eq_decoded_of_bound fits decodedBound]
  exact equal

theorem lcEval_terms_eq_decoded
    {layout : Layout} {assignment : Nat → Nat}
    (fits : 2 ^ layout.width ≤ goldilocksP)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    lcEval assignment layout.terms = decoded layout assignment :=
  lcEval_terms_eq_decoded_of_bound fits
    (decoded_lt canonical one holds)

theorem recomposition_sound
    {layout : Layout} {assignment : Nat → Nat}
    (fits : 2 ^ layout.width ≤ goldilocksP)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.valueColumn = decoded layout assignment := by
  have fieldEqual := builderLinearRow_sound canonical one
    layout.valueColumn layout.terms (layout.terms_canonical fits)
    (holds _ (recompositionRow_mem_rows layout))
  exact fieldEqual.trans
    (lcEval_terms_eq_decoded fits canonical one holds)

/-- Satisfying the emitted rows proves the narrow integer bound. -/
theorem value_lt_twoPower
    {layout : Layout} {assignment : Nat → Nat}
    (fits : 2 ^ layout.width ≤ goldilocksP)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    assignment layout.valueColumn < 2 ^ layout.width := by
  rw [recomposition_sound fits canonical one holds]
  exact decoded_lt canonical one holds

/-- The satisfying bit columns are the unique fixed-width little-endian
encoding of the value column. This prevents a compiler from satisfying the
range rows with one word while exposing different public bits. -/
theorem digits_eq_encodeWord
    {layout : Layout} {assignment : Nat → Nat}
    (fits : 2 ^ layout.width ≤ goldilocksP)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (holds : Satisfies (rows layout) assignment) :
    layout.digits assignment =
      encodeWord layout.width (assignment layout.valueColumn) := by
  apply Nat.injOn_ofDigits (b := 2) (by decide) layout.width
  · exact ⟨layout.digits_length assignment,
      canonical_digits_of_rows canonical one holds⟩
  · exact ⟨encodeWord_length _ _, fun digit member =>
      encodeWord_binary _ _ digit member⟩
  · rw [ofDigits_encodeWord_of_bound
      (value_lt_twoPower fits canonical one holds)]
    exact (recomposition_sound fits canonical one holds).symm

/-- Exact honest placement used by compiler completeness. -/
structure Honest (layout : Layout) (assignment : Nat → Nat)
    (value : Nat) : Prop where
  valueBound : value < 2 ^ layout.width
  valuePlaced : assignment layout.valueColumn = value
  bitPlaced : ∀ offset (offsetBound : offset < layout.width),
    assignment (layout.bitColumn offset) =
      (encodeWord layout.width value).get
        ⟨offset, by simpa [encodeWord_length] using offsetBound⟩

private theorem honest_digits
    {layout : Layout} {assignment : Nat → Nat} {value : Nat}
    (honest : Honest layout assignment value) :
    layout.digits assignment = encodeWord layout.width value := by
  apply List.ext_get
  · rw [layout.digits_length, encodeWord_length]
  · intro index digitsBound wordBound
    simpa [Layout.digits] using honest.bitPlaced index
      (by simpa [layout.digits_length] using digitsBound)

private theorem bitRow_complete
    {layout : Layout} {assignment : Nat → Nat} {value : Nat}
    (one : assignment 0 = 1)
    (honest : Honest layout assignment value)
    {offset : Nat} (bound : offset < layout.width) :
    RowHolds assignment (bitRow (layout.bitColumn offset)) := by
  have placed := honest.bitPlaced offset bound
  have encodedBinary :
      (encodeWord layout.width value).get
          ⟨offset, by simpa [encodeWord_length] using bound⟩ < 2 :=
    encodeWord_binary _ _ _ (List.get_mem _ _)
  have root : assignment (layout.bitColumn offset) = 0 ∨
      assignment (layout.bitColumn offset) = 1 := by
    rw [placed]
    omega
  rcases root with zero | oneBit
  · simp [RowHolds, bitRow, lcEval, one, zero, goldilocksP]
  · simp [RowHolds, bitRow, lcEval, one, oneBit, goldilocksP]

private theorem honest_decoded
    {layout : Layout} {assignment : Nat → Nat} {value : Nat}
    (honest : Honest layout assignment value) :
    decoded layout assignment = value := by
  rw [decoded, honest_digits honest]
  exact ofDigits_encodeWord_of_bound honest.valueBound

/-- Every bounded value has a complete satisfying local row block when its
value and exact codec bits are placed at the declared columns. -/
theorem rows_complete
    {layout : Layout} {assignment : Nat → Nat} {value : Nat}
    (fits : 2 ^ layout.width ≤ goldilocksP)
    (one : assignment 0 = 1)
    (honest : Honest layout assignment value) :
    Satisfies (rows layout) assignment := by
  intro row member
  rw [rows, List.mem_append] at member
  rcases member with bitMember | recompositionMember
  · rcases List.mem_map.mp bitMember with
      ⟨offset, offsetMember, rowEqual⟩
    subst row
    exact bitRow_complete one honest (List.mem_range.mp offsetMember)
  · simp only [List.mem_singleton] at recompositionMember
    subst row
    apply builderLinearRow_complete one layout.valueColumn layout.terms
      (layout.terms_canonical fits)
    rw [lcEval_terms_eq_decoded_of_bound fits (by
      rw [honest_decoded honest]
      exact honest.valueBound)]
    rw [honest_decoded honest]
    exact honest.valuePlaced

end Nightstream.Implementation.NebulaV2.BoundedWordRows
