import Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
import Nightstream.Implementation.R1CS.Canonical.KMulHonest

/-!
Contract: honest completeness for the Lean-owned canonical-u64 recipe.

The witness is constructed from:

* an authoritative 64-bit source word;
* the caller assignment on pre-existing columns; and
* one global Goldilocks inverse primitive.

It writes exactly the recipe's 66 allocated columns and leaves every
caller-owned input column unchanged.  Neither row satisfaction nor a decoded
acceptance proposition is an input.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe

/-- Global field-inverse boundary used by the honest witness generator. -/
structure FieldInverse where
  inverse : Nat → Nat
  canonical : ∀ value, inverse value < goldilocksP
  zero : inverse 0 = 0
  correct : ∀ value, value < goldilocksP → value ≠ 0 →
    value * inverse value % goldilocksP = 1

/-- A fixed-width source word.  Only indices below 64 are consumed. -/
structure Source where
  bit : Nat → Bool

def sourceBit (source : Source) (index : Nat) : Nat :=
  (source.bit index).toNat

def sourceLow (source : Source) : Nat :=
  (List.range 32).foldl
    (fun value index => value + 2 ^ index * sourceBit source index) 0

def sourceHigh (source : Source) : Nat :=
  (List.range 32).foldl
    (fun value index =>
      value + 2 ^ index * sourceBit source (32 + index)) 0

/-- Little-endian integer denotation of the source word. -/
def sourceWord (source : Source) : Nat :=
  sourceLow source + 4294967296 * sourceHigh source

def highIsMax (source : Source) : Nat :=
  if sourceHigh source = highMax then 1 else 0

def highDifference (source : Source) : Nat :=
  (sourceHigh source + (goldilocksP - highMax)) % goldilocksP

/-- Honest assignment.  The recipe allocation is the contiguous interval
`[base, base + 66)`; all other columns retain their caller values. -/
def witness
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat) : Nat → Nat :=
  fun column =>
    if column < layout.base then initial column
    else if column < layout.base + 64 then
      sourceBit source (column - layout.base)
    else if column = highFlagColumn layout then highIsMax source
    else if column = inverseColumn layout then
      field.inverse (highDifference source)
    else initial column

private theorem range32_shape : List.range 32 =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31] := by decide

private theorem range64_shape : List.range 64 =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
     29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
     42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
     55, 56, 57, 58, 59, 60, 61, 62, 63] := by decide

theorem sourceBit_le_one (source : Source) (index : Nat) :
    sourceBit source index ≤ 1 := by
  cases value : source.bit index <;> simp [sourceBit, value]

theorem sourceBit_lt_modulus (source : Source) (index : Nat) :
    sourceBit source index < goldilocksP := by
  have bounded := sourceBit_le_one source index
  simp only [goldilocksP]
  omega

theorem sourceLow_le_highMax (source : Source) :
    sourceLow source ≤ highMax := by
  have b0 := sourceBit_le_one source 0
  have b1 := sourceBit_le_one source 1
  have b2 := sourceBit_le_one source 2
  have b3 := sourceBit_le_one source 3
  have b4 := sourceBit_le_one source 4
  have b5 := sourceBit_le_one source 5
  have b6 := sourceBit_le_one source 6
  have b7 := sourceBit_le_one source 7
  have b8 := sourceBit_le_one source 8
  have b9 := sourceBit_le_one source 9
  have b10 := sourceBit_le_one source 10
  have b11 := sourceBit_le_one source 11
  have b12 := sourceBit_le_one source 12
  have b13 := sourceBit_le_one source 13
  have b14 := sourceBit_le_one source 14
  have b15 := sourceBit_le_one source 15
  have b16 := sourceBit_le_one source 16
  have b17 := sourceBit_le_one source 17
  have b18 := sourceBit_le_one source 18
  have b19 := sourceBit_le_one source 19
  have b20 := sourceBit_le_one source 20
  have b21 := sourceBit_le_one source 21
  have b22 := sourceBit_le_one source 22
  have b23 := sourceBit_le_one source 23
  have b24 := sourceBit_le_one source 24
  have b25 := sourceBit_le_one source 25
  have b26 := sourceBit_le_one source 26
  have b27 := sourceBit_le_one source 27
  have b28 := sourceBit_le_one source 28
  have b29 := sourceBit_le_one source 29
  have b30 := sourceBit_le_one source 30
  have b31 := sourceBit_le_one source 31
  simp [sourceLow, range32_shape, highMax] at *
  omega

theorem sourceHigh_le_highMax (source : Source) :
    sourceHigh source ≤ highMax := by
  have b32 := sourceBit_le_one source 32
  have b33 := sourceBit_le_one source 33
  have b34 := sourceBit_le_one source 34
  have b35 := sourceBit_le_one source 35
  have b36 := sourceBit_le_one source 36
  have b37 := sourceBit_le_one source 37
  have b38 := sourceBit_le_one source 38
  have b39 := sourceBit_le_one source 39
  have b40 := sourceBit_le_one source 40
  have b41 := sourceBit_le_one source 41
  have b42 := sourceBit_le_one source 42
  have b43 := sourceBit_le_one source 43
  have b44 := sourceBit_le_one source 44
  have b45 := sourceBit_le_one source 45
  have b46 := sourceBit_le_one source 46
  have b47 := sourceBit_le_one source 47
  have b48 := sourceBit_le_one source 48
  have b49 := sourceBit_le_one source 49
  have b50 := sourceBit_le_one source 50
  have b51 := sourceBit_le_one source 51
  have b52 := sourceBit_le_one source 52
  have b53 := sourceBit_le_one source 53
  have b54 := sourceBit_le_one source 54
  have b55 := sourceBit_le_one source 55
  have b56 := sourceBit_le_one source 56
  have b57 := sourceBit_le_one source 57
  have b58 := sourceBit_le_one source 58
  have b59 := sourceBit_le_one source 59
  have b60 := sourceBit_le_one source 60
  have b61 := sourceBit_le_one source 61
  have b62 := sourceBit_le_one source 62
  have b63 := sourceBit_le_one source 63
  simp [sourceHigh, range32_shape, highMax] at *
  omega

theorem highIsMax_lt_modulus (source : Source) :
    highIsMax source < goldilocksP := by
  unfold highIsMax
  split <;> simp [goldilocksP]

theorem highDifference_lt_modulus (source : Source) :
    highDifference source < goldilocksP := by
  unfold highDifference
  exact Nat.mod_lt _ (by decide)

@[simp] theorem witness_before
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    {column : Nat} (before : column < layout.base) :
    witness field source layout initial column = initial column := by
  simp [witness, before]

@[simp] theorem witness_bit
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    {index : Nat} (bounded : index < 64) :
    witness field source layout initial (bitColumn layout index) =
      sourceBit source index := by
  unfold witness bitColumn
  rw [if_neg (by omega), if_pos (by omega)]
  congr 1
  omega

@[simp] theorem witness_highFlag
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat) :
    witness field source layout initial (highFlagColumn layout) =
      highIsMax source := by
  unfold witness highFlagColumn
  rw [if_neg (by omega), if_neg (by omega), if_pos rfl]

@[simp] theorem witness_inverse
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat) :
    witness field source layout initial (inverseColumn layout) =
      field.inverse (highDifference source) := by
  unfold witness inverseColumn highFlagColumn
  rw [if_neg (by omega), if_neg (by omega), if_neg (by omega), if_pos rfl]

theorem witness_input
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (below : InputBelowBase layout) :
    lcEval (witness field source layout initial) layout.input =
      lcEval initial layout.input := by
  apply KMulHonest.lcEval_congr
  intro column mentioned
  rcases List.mem_map.mp mentioned with ⟨term, termMember, columnEq⟩
  rcases term with ⟨termColumn, coefficient⟩
  simp only at columnEq
  subst termColumn
  exact witness_before field source layout initial
    (below column coefficient termMember)

theorem witness_constant
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    witness field source layout initial 0 = 1 := by
  rw [witness_before field source layout initial positive, constantWire]

private theorem bitTerms_eval
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (canonical : sourceWord source < goldilocksP) :
    lcEval (witness field source layout initial) (bitTerms layout) =
      sourceWord source := by
  unfold lcEval
  have raw :
      (bitTerms layout).foldl
          (fun value term =>
            value + term.2 * witness field source layout initial term.1) 0 =
        sourceWord source := by
    simp [bitTerms, sourceWord, sourceLow, sourceHigh, range32_shape,
      range64_shape, witness_bit]
    omega
  rw [raw, Nat.mod_eq_of_lt canonical]

private theorem lowTerms_eval
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat) :
    lcEval (witness field source layout initial) (lowTerms layout) =
      sourceLow source := by
  unfold lcEval
  have raw :
      (lowTerms layout).foldl
          (fun value term =>
            value + term.2 * witness field source layout initial term.1) 0 =
        sourceLow source := by
    simp [lowTerms, sourceLow, range32_shape, witness_bit]
  rw [raw, Nat.mod_eq_of_lt]
  exact Nat.lt_of_le_of_lt (sourceLow_le_highMax source) (by decide)

private theorem highTerms_eval
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat) :
    lcEval (witness field source layout initial) (highTerms layout) =
      sourceHigh source := by
  unfold lcEval
  have raw :
      (highTerms layout).foldl
          (fun value term =>
            value + term.2 * witness field source layout initial term.1) 0 =
        sourceHigh source := by
    simp [highTerms, sourceHigh, range32_shape, witness_bit]
  rw [raw, Nat.mod_eq_of_lt]
  exact Nat.lt_of_le_of_lt (sourceHigh_le_highMax source) (by decide)

private theorem highDifferenceTerms_eval
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    lcEval (witness field source layout initial)
        (highDifferenceTerms layout) =
      highDifference source := by
  unfold lcEval highDifferenceTerms highDifference
  have raw :
      (highTerms layout ++ [(0, goldilocksP - highMax)]).foldl
          (fun value (term : Nat × Nat) =>
            value + term.2 * witness field source layout initial term.1) 0 =
        sourceHigh source + (goldilocksP - highMax) := by
    simp [highTerms, sourceHigh, range32_shape, witness_bit,
      witness_constant field source layout initial positive constantWire]
  rw [raw]

private theorem bitRow_complete
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    {index : Nat} (bounded : index < 64) :
    RowHolds (witness field source layout initial)
      (bitRow (bitColumn layout index)) := by
  cases value : source.bit index <;>
    simp [RowHolds, bitRow, lcEval, sourceBit, value, goldilocksP,
      witness_bit field source layout initial bounded,
      witness_constant field source layout initial positive constantWire]

private theorem bitRows_complete
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    Satisfies (bitRows layout) (witness field source layout initial) := by
  intro row member
  rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
  apply bitRow_complete field source layout initial positive constantWire
  exact List.mem_range.mp indexMember

private theorem recomposition_complete
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    (below : InputBelowBase layout)
    (inputMatches : lcEval initial layout.input = sourceWord source)
    (canonical : sourceWord source < goldilocksP) :
    RowHolds (witness field source layout initial)
      (recompositionRow layout) := by
  have inputEval :
      lcEval (witness field source layout initial) layout.input =
        sourceWord source := by
    rw [witness_input field source layout initial below, inputMatches]
  have oneEval :
      lcEval (witness field source layout initial) [(0, 1)] = 1 := by
    simp [lcEval,
      witness_constant field source layout initial positive constantWire,
      goldilocksP]
  have bitsEval :=
    bitTerms_eval field source layout initial canonical
  simp only [RowHolds, recompositionRow, inputEval, oneEval, bitsEval,
    Nat.mul_one, Nat.mod_eq_of_lt canonical]

private theorem highFlagBit_complete
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    RowHolds (witness field source layout initial)
      (bitRow (highFlagColumn layout)) := by
  have flagEval :
      lcEval (witness field source layout initial)
          [(highFlagColumn layout, 1)] =
        highIsMax source := by
    simp [lcEval, Nat.mod_eq_of_lt (highIsMax_lt_modulus source)]
  have minusOneEval :
      lcEval (witness field source layout initial)
          [(highFlagColumn layout, 1), (0, goldilocksP - 1)] =
        (highIsMax source + (goldilocksP - 1)) % goldilocksP := by
    simp [lcEval,
      witness_constant field source layout initial positive constantWire]
  simp only [RowHolds, bitRow, flagEval, minusOneEval]
  unfold highIsMax
  split <;> simp [lcEval, goldilocksP]

private theorem highDefinition_complete
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    RowHolds (witness field source layout initial)
      (highDefinitionRow layout) := by
  have difference :=
    highDifferenceTerms_eval field source layout initial positive constantWire
  have flagEval :
      lcEval (witness field source layout initial)
          [(highFlagColumn layout, 1)] =
        highIsMax source := by
    simp [lcEval, Nat.mod_eq_of_lt (highIsMax_lt_modulus source)]
  simp only [RowHolds, highDefinitionRow, flagEval, difference]
  unfold highIsMax
  split
  case isTrue equal =>
    simp [highDifference, equal, highMax, goldilocksP, lcEval]
  case isFalse notEqual => simp [lcEval]

private theorem inverse_complete
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    RowHolds (witness field source layout initial)
      (inverseRow layout) := by
  have difference :=
    highDifferenceTerms_eval field source layout initial positive constantWire
  have inverseEval :
      lcEval (witness field source layout initial)
          [(inverseColumn layout, 1)] =
        field.inverse (highDifference source) := by
    simp [lcEval, Nat.mod_eq_of_lt (field.canonical _)]
  have targetEval :
      lcEval (witness field source layout initial)
          (oneMinusHighFlag layout) =
        ((goldilocksP - 1) * highIsMax source + 1) % goldilocksP := by
    simp [oneMinusHighFlag, lcEval,
      witness_constant field source layout initial positive constantWire]
  simp only [RowHolds, inverseRow, difference, inverseEval, targetEval]
  by_cases equal : sourceHigh source = highMax
  · simp [highIsMax, equal, highDifference, highMax, goldilocksP, field.zero]
  · have highBound := sourceHigh_le_highMax source
    have highLt : sourceHigh source < highMax := by omega
    have rawLt :
        sourceHigh source + (goldilocksP - highMax) < goldilocksP := by
      simp only [goldilocksP, highMax] at highLt ⊢
      omega
    have rawPositive :
        0 < sourceHigh source + (goldilocksP - highMax) := by
      simp only [goldilocksP, highMax]
      omega
    have differenceNonzero : highDifference source ≠ 0 := by
      rw [highDifference, Nat.mod_eq_of_lt rawLt]
      omega
    have inverseLaw := field.correct (highDifference source)
      (highDifference_lt_modulus source) differenceNonzero
    simpa [highIsMax, equal, goldilocksP] using inverseLaw

private theorem canonicality_complete
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (canonical : sourceWord source < goldilocksP) :
    RowHolds (witness field source layout initial)
      (canonicalityRow layout) := by
  have low := lowTerms_eval field source layout initial
  have flagEval :
      lcEval (witness field source layout initial)
          [(highFlagColumn layout, 1)] =
        highIsMax source := by
    simp [lcEval, Nat.mod_eq_of_lt (highIsMax_lt_modulus source)]
  simp only [RowHolds, canonicalityRow, flagEval, low]
  by_cases equal : sourceHigh source = highMax
  · have lowZero : sourceLow source = 0 := by
      simp only [sourceWord, equal, highMax, goldilocksP] at canonical
      omega
    simp [highIsMax, equal, lowZero, lcEval]
  · simp [highIsMax, equal, lcEval]

/-- The honest witness is canonical on every column when the caller assignment
is canonical. -/
theorem witness_canonical
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP) :
    ∀ column, witness field source layout initial column < goldilocksP := by
  intro column
  unfold witness
  split
  · exact initialCanonical column
  split
  · exact sourceBit_lt_modulus source _
  split
  · exact highIsMax_lt_modulus source
  split
  · exact field.canonical _
  · exact initialCanonical column

/-- Every authoritative canonical source word whose value matches the input
expression has a satisfying assignment for all 69 Lean-owned rows. -/
theorem complete
    (field : FieldInverse) (source : Source)
    (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    (below : InputBelowBase layout)
    (inputMatches : lcEval initial layout.input = sourceWord source)
    (canonical : sourceWord source < goldilocksP) :
    Satisfies (rows layout) (witness field source layout initial) := by
  intro row member
  unfold rows at member
  rcases List.mem_append.mp member with inBits | inTail
  · exact bitRows_complete field source layout initial positive constantWire
      row inBits
  · simp only [List.mem_cons, List.not_mem_nil, or_false] at inTail
    rcases inTail with rfl | rfl | rfl | rfl | rfl
    · exact recomposition_complete field source layout initial positive
        constantWire below inputMatches canonical
    · exact highFlagBit_complete field source layout initial positive
        constantWire
    · exact highDefinition_complete field source layout initial positive
        constantWire
    · exact inverse_complete field source layout initial positive constantWire
    · exact canonicality_complete field source layout initial canonical

end Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest
