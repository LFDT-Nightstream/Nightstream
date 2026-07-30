import Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe

/-!
Contract: semantic soundness of the Lean-owned canonical-u64 recipe.

Any canonical-residue assignment satisfying `CanonicalU64Recipe.rows`
contains 64 Boolean bits whose integer value:

* is strictly below the Goldilocks modulus; and
* equals the recipe's input linear combination.

The only arithmetic premise is the existing typed Euclid property of the
Goldilocks modulus.  No generated row or Rust witness is imported.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe

def bitValue (assignment : Nat → Nat) (layout : Layout) (index : Nat) : Nat :=
  assignment (bitColumn layout index)

def lowValue (assignment : Nat → Nat) (layout : Layout) : Nat :=
  (List.range 32).foldl
    (fun value index => value + 2 ^ index * bitValue assignment layout index) 0

def highValue (assignment : Nat → Nat) (layout : Layout) : Nat :=
  (List.range 32).foldl
    (fun value index =>
      value + 2 ^ index * bitValue assignment layout (32 + index)) 0

def bitsValue (assignment : Nat → Nat) (layout : Layout) : Nat :=
  (List.range 64).foldl
    (fun value index => value + 2 ^ index * bitValue assignment layout index) 0

private theorem range32_shape : List.range 32 =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
     31] := by decide

private theorem range64_shape : List.range 64 =
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
     30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
     44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
     58, 59, 60, 61, 62, 63] := by decide

theorem satisfies_bitRow
    {assignment : Nat → Nat} {layout : Layout}
    (satisfied : Satisfies (rows layout) assignment)
    {index : Nat} (bounded : index < 64) :
    RowHolds assignment (bitRow (bitColumn layout index)) := by
  apply satisfied
  unfold rows bitRows
  apply List.mem_append_left
  apply List.mem_map.mpr
  exact ⟨index, List.mem_range.mpr bounded, rfl⟩

theorem bitValue_le_one
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment)
    {index : Nat} (bounded : index < 64) :
    bitValue assignment layout index ≤ 1 := by
  exact Nightstream.Implementation.R1CS.bitRow_le_one prime
    (z := assignment) (c := bitColumn layout index)
    (canonical _) constantWire
    (satisfies_bitRow satisfied bounded)

theorem lowValue_le_highMax
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    lowValue assignment layout ≤ highMax := by
  have h0 := bitValue_le_one prime canonical constantWire satisfied
    (index := 0) (by decide)
  have h1 := bitValue_le_one prime canonical constantWire satisfied
    (index := 1) (by decide)
  have h2 := bitValue_le_one prime canonical constantWire satisfied
    (index := 2) (by decide)
  have h3 := bitValue_le_one prime canonical constantWire satisfied
    (index := 3) (by decide)
  have h4 := bitValue_le_one prime canonical constantWire satisfied
    (index := 4) (by decide)
  have h5 := bitValue_le_one prime canonical constantWire satisfied
    (index := 5) (by decide)
  have h6 := bitValue_le_one prime canonical constantWire satisfied
    (index := 6) (by decide)
  have h7 := bitValue_le_one prime canonical constantWire satisfied
    (index := 7) (by decide)
  have h8 := bitValue_le_one prime canonical constantWire satisfied
    (index := 8) (by decide)
  have h9 := bitValue_le_one prime canonical constantWire satisfied
    (index := 9) (by decide)
  have h10 := bitValue_le_one prime canonical constantWire satisfied
    (index := 10) (by decide)
  have h11 := bitValue_le_one prime canonical constantWire satisfied
    (index := 11) (by decide)
  have h12 := bitValue_le_one prime canonical constantWire satisfied
    (index := 12) (by decide)
  have h13 := bitValue_le_one prime canonical constantWire satisfied
    (index := 13) (by decide)
  have h14 := bitValue_le_one prime canonical constantWire satisfied
    (index := 14) (by decide)
  have h15 := bitValue_le_one prime canonical constantWire satisfied
    (index := 15) (by decide)
  have h16 := bitValue_le_one prime canonical constantWire satisfied
    (index := 16) (by decide)
  have h17 := bitValue_le_one prime canonical constantWire satisfied
    (index := 17) (by decide)
  have h18 := bitValue_le_one prime canonical constantWire satisfied
    (index := 18) (by decide)
  have h19 := bitValue_le_one prime canonical constantWire satisfied
    (index := 19) (by decide)
  have h20 := bitValue_le_one prime canonical constantWire satisfied
    (index := 20) (by decide)
  have h21 := bitValue_le_one prime canonical constantWire satisfied
    (index := 21) (by decide)
  have h22 := bitValue_le_one prime canonical constantWire satisfied
    (index := 22) (by decide)
  have h23 := bitValue_le_one prime canonical constantWire satisfied
    (index := 23) (by decide)
  have h24 := bitValue_le_one prime canonical constantWire satisfied
    (index := 24) (by decide)
  have h25 := bitValue_le_one prime canonical constantWire satisfied
    (index := 25) (by decide)
  have h26 := bitValue_le_one prime canonical constantWire satisfied
    (index := 26) (by decide)
  have h27 := bitValue_le_one prime canonical constantWire satisfied
    (index := 27) (by decide)
  have h28 := bitValue_le_one prime canonical constantWire satisfied
    (index := 28) (by decide)
  have h29 := bitValue_le_one prime canonical constantWire satisfied
    (index := 29) (by decide)
  have h30 := bitValue_le_one prime canonical constantWire satisfied
    (index := 30) (by decide)
  have h31 := bitValue_le_one prime canonical constantWire satisfied
    (index := 31) (by decide)
  simp [lowValue, range32_shape, highMax] at *
  omega

theorem highValue_le_highMax
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    highValue assignment layout ≤ highMax := by
  have h32 := bitValue_le_one prime canonical constantWire satisfied
    (index := 32) (by decide)
  have h33 := bitValue_le_one prime canonical constantWire satisfied
    (index := 33) (by decide)
  have h34 := bitValue_le_one prime canonical constantWire satisfied
    (index := 34) (by decide)
  have h35 := bitValue_le_one prime canonical constantWire satisfied
    (index := 35) (by decide)
  have h36 := bitValue_le_one prime canonical constantWire satisfied
    (index := 36) (by decide)
  have h37 := bitValue_le_one prime canonical constantWire satisfied
    (index := 37) (by decide)
  have h38 := bitValue_le_one prime canonical constantWire satisfied
    (index := 38) (by decide)
  have h39 := bitValue_le_one prime canonical constantWire satisfied
    (index := 39) (by decide)
  have h40 := bitValue_le_one prime canonical constantWire satisfied
    (index := 40) (by decide)
  have h41 := bitValue_le_one prime canonical constantWire satisfied
    (index := 41) (by decide)
  have h42 := bitValue_le_one prime canonical constantWire satisfied
    (index := 42) (by decide)
  have h43 := bitValue_le_one prime canonical constantWire satisfied
    (index := 43) (by decide)
  have h44 := bitValue_le_one prime canonical constantWire satisfied
    (index := 44) (by decide)
  have h45 := bitValue_le_one prime canonical constantWire satisfied
    (index := 45) (by decide)
  have h46 := bitValue_le_one prime canonical constantWire satisfied
    (index := 46) (by decide)
  have h47 := bitValue_le_one prime canonical constantWire satisfied
    (index := 47) (by decide)
  have h48 := bitValue_le_one prime canonical constantWire satisfied
    (index := 48) (by decide)
  have h49 := bitValue_le_one prime canonical constantWire satisfied
    (index := 49) (by decide)
  have h50 := bitValue_le_one prime canonical constantWire satisfied
    (index := 50) (by decide)
  have h51 := bitValue_le_one prime canonical constantWire satisfied
    (index := 51) (by decide)
  have h52 := bitValue_le_one prime canonical constantWire satisfied
    (index := 52) (by decide)
  have h53 := bitValue_le_one prime canonical constantWire satisfied
    (index := 53) (by decide)
  have h54 := bitValue_le_one prime canonical constantWire satisfied
    (index := 54) (by decide)
  have h55 := bitValue_le_one prime canonical constantWire satisfied
    (index := 55) (by decide)
  have h56 := bitValue_le_one prime canonical constantWire satisfied
    (index := 56) (by decide)
  have h57 := bitValue_le_one prime canonical constantWire satisfied
    (index := 57) (by decide)
  have h58 := bitValue_le_one prime canonical constantWire satisfied
    (index := 58) (by decide)
  have h59 := bitValue_le_one prime canonical constantWire satisfied
    (index := 59) (by decide)
  have h60 := bitValue_le_one prime canonical constantWire satisfied
    (index := 60) (by decide)
  have h61 := bitValue_le_one prime canonical constantWire satisfied
    (index := 61) (by decide)
  have h62 := bitValue_le_one prime canonical constantWire satisfied
    (index := 62) (by decide)
  have h63 := bitValue_le_one prime canonical constantWire satisfied
    (index := 63) (by decide)
  simp [highValue, range32_shape, highMax] at *
  omega

theorem bitsValue_eq_low_add_high
    (assignment : Nat → Nat) (layout : Layout) :
    bitsValue assignment layout =
      lowValue assignment layout +
        4294967296 * highValue assignment layout := by
  simp [bitsValue, lowValue, highValue, range32_shape, range64_shape]
  omega

private theorem lowTerms_eval
    (assignment : Nat → Nat) (layout : Layout)
    (lowBound : lowValue assignment layout ≤ highMax) :
    lcEval assignment (lowTerms layout) =
      lowValue assignment layout := by
  have raw :
      (lowTerms layout).foldl
          (fun value term => value + term.2 * assignment term.1) 0 =
        lowValue assignment layout := by
    simp [lowTerms, lowValue, bitValue, List.foldl_map]
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt]
  exact Nat.lt_of_le_of_lt lowBound (by decide)

private theorem highTerms_eval
    (assignment : Nat → Nat) (layout : Layout)
    (highBound : highValue assignment layout ≤ highMax) :
    lcEval assignment (highTerms layout) =
      highValue assignment layout := by
  have raw :
      (highTerms layout).foldl
          (fun value term => value + term.2 * assignment term.1) 0 =
        highValue assignment layout := by
    simp [highTerms, highValue, bitValue, List.foldl_map]
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt]
  exact Nat.lt_of_le_of_lt highBound (by decide)

private theorem highDifferenceTerms_eval
    (assignment : Nat → Nat) (layout : Layout) :
    lcEval assignment (highDifferenceTerms layout) =
      (lcEval assignment (highTerms layout) +
        (goldilocksP - highMax) * assignment 0) % goldilocksP := by
  unfold highDifferenceTerms lcEval
  rw [List.foldl_append]
  simp only [List.foldl]
  rw [Nat.add_mod]
  exact Nat.add_mod_mod _ _ _

private theorem singleton_eval
    (assignment : Nat → Nat)
    (column coefficient : Nat) :
    lcEval assignment [(column, coefficient)] =
      coefficient * assignment column % goldilocksP := by
  simp [lcEval]

private theorem one_eval
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1) :
    lcEval assignment [(0, 1)] = 1 := by
  simp [lcEval, constantWire, goldilocksP]

theorem bitsValue_lt_modulus
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    bitsValue assignment layout < goldilocksP := by
  let flag := assignment (highFlagColumn layout)
  let inverse := assignment (inverseColumn layout)
  let low := lowValue assignment layout
  let high := highValue assignment layout
  have lowBound : low ≤ highMax :=
    lowValue_le_highMax prime canonical constantWire satisfied
  have highBound : high ≤ highMax :=
    highValue_le_highMax prime canonical constantWire satisfied
  have lowEval := lowTerms_eval assignment layout lowBound
  have highEval := highTerms_eval assignment layout highBound
  have flagLt : flag < goldilocksP := canonical _
  have inverseLt : inverse < goldilocksP := canonical _
  have flagEval :
      lcEval assignment [(highFlagColumn layout, 1)] = flag := by
    rw [singleton_eval assignment]
    simp only [Nat.one_mul]
    change assignment (highFlagColumn layout) % goldilocksP =
      assignment (highFlagColumn layout)
    exact Nat.mod_eq_of_lt flagLt
  have inverseEval :
      lcEval assignment [(inverseColumn layout, 1)] = inverse := by
    rw [singleton_eval assignment]
    simp only [Nat.one_mul]
    change assignment (inverseColumn layout) % goldilocksP =
      assignment (inverseColumn layout)
    exact Nat.mod_eq_of_lt inverseLt
  have differenceEval :
      lcEval assignment (highDifferenceTerms layout) =
        (high + (goldilocksP - highMax)) % goldilocksP := by
    rw [highDifferenceTerms_eval, highEval, constantWire, Nat.mul_one]
  have targetEval :
      lcEval assignment (oneMinusHighFlag layout) =
        ((goldilocksP - 1) * flag + 1) % goldilocksP := by
    unfold oneMinusHighFlag lcEval
    simp only [List.foldl, constantWire, Nat.one_mul]
    simp only [flag, Nat.zero_add]
  have canonicality :
      flag * low % goldilocksP = 0 := by
    have rowHolds := satisfied (canonicalityRow layout) (by
      simp [rows])
    simpa [RowHolds, canonicalityRow, flagEval, lowEval, lcEval] using rowHolds
  have inverseEquation :
      ((high + (goldilocksP - highMax)) % goldilocksP) * inverse %
          goldilocksP =
        ((goldilocksP - 1) * flag + 1) % goldilocksP := by
    have rowHolds := satisfied (inverseRow layout) (by
      simp [rows])
    simpa [RowHolds, inverseRow, differenceEval, inverseEval, targetEval]
      using rowHolds
  have flagOrLow := prime flag low canonicality
  rw [Nat.mod_eq_of_lt flagLt,
    Nat.mod_eq_of_lt (by
      exact Nat.lt_of_le_of_lt lowBound (by decide : highMax < goldilocksP))]
    at flagOrLow
  rw [bitsValue_eq_low_add_high]
  rcases flagOrLow with flagZero | lowZero
  · have highLt : high < highMax := by
      by_cases highEq : high = highMax
      · simp only [highEq, goldilocksP, highMax, flagZero] at inverseEquation
        omega
      · omega
    change low + 4294967296 * high < goldilocksP
    simp only [highMax, goldilocksP] at lowBound highLt ⊢
    omega
  · change low + 4294967296 * high < goldilocksP
    simp only [highMax, goldilocksP] at lowZero highBound ⊢
    omega

private theorem bitTerms_eval
    (assignment : Nat → Nat) (layout : Layout)
    (bitsBound : bitsValue assignment layout < goldilocksP) :
    lcEval assignment (bitTerms layout) =
      bitsValue assignment layout := by
  unfold lcEval bitTerms bitsValue
  rw [range64_shape]
  simp only [List.map, List.foldl]
  exact Nat.mod_eq_of_lt bitsBound

/-- Complete semantic result of one canonical-u64 occurrence. -/
structure Refines
    (assignment : Nat → Nat) (layout : Layout) : Prop where
  input_eq :
    lcEval assignment layout.input = bitsValue assignment layout
  canonical :
    bitsValue assignment layout < goldilocksP
  bit :
    ∀ index, index < 64 → bitValue assignment layout index ≤ 1

/-- Satisfaction of the Lean-owned rows forces the unique canonical 64-bit
decomposition of the input expression. -/
theorem sound
    (prime : EuclidPrime goldilocksP)
    {assignment : Nat → Nat} {layout : Layout}
    (canonical : ∀ column, assignment column < goldilocksP)
    (constantWire : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    Refines assignment layout := by
  have bitsBound :=
    bitsValue_lt_modulus prime canonical constantWire satisfied
  have bitsEval := bitTerms_eval assignment layout bitsBound
  have recomposition := satisfied (recompositionRow layout) (by
    simp [rows])
  have inputEq :
      lcEval assignment layout.input = bitsValue assignment layout := by
    simpa [RowHolds, recompositionRow, one_eval assignment constantWire,
      bitsEval, Nat.mod_eq_of_lt
        (show lcEval assignment layout.input < goldilocksP by
          unfold lcEval
          exact Nat.mod_lt _ (by decide))] using recomposition
  exact {
    input_eq := inputEq
    canonical := bitsBound
    bit := fun index bounded =>
      bitValue_le_one prime canonical constantWire satisfied bounded
  }

end Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
