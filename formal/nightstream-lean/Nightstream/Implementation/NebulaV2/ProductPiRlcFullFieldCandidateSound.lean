import Nightstream.Implementation.NebulaV2.ProductPiRlcFullFieldCandidateRows
import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: row-derived soundness of one V2 full-field PiRLC candidate.

From only canonical field residues, the constant-one wire, and satisfaction of
the exact row program, this module derives:

* the unique canonical integer represented by the candidate expression;
* an accept bit equal to one exactly when that integer is below `q - 1`; and
* a residue equal to that integer modulo five.

No acceptance decision, candidate integer, quotient, or residue is supplied as
an assumption.
-/

set_option autoImplicit false
set_option maxRecDepth 30000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.NebulaV2.ProductPiRlcFullFieldCandidateSound

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.NebulaV2.ProductPiRlcFullFieldCandidateRows

def candidateValue (assignment : Nat -> Nat) (layout : Layout) : Nat :=
  CanonicalU64RecipeSound.bitsValue assignment (canonicalLayout layout)

def weightedValue (assignment : Nat -> Nat) (layout : Layout) : Nat :=
  (List.range 64).foldl
    (fun value index =>
      value + (2 ^ index % 5) *
        CanonicalU64RecipeSound.bitValue assignment
          (canonicalLayout layout) index) 0

def quotientValue (assignment : Nat -> Nat) (layout : Layout) : Nat :=
  (List.range quotientBitCount).foldl
    (fun value index =>
      value + 2 ^ index * assignment (quotientBitColumn layout index)) 0

def QuotientBitsBoolean (assignment : Nat -> Nat) (layout : Layout) : Prop :=
  ∀ index, index < quotientBitCount ->
    assignment (quotientBitColumn layout index) ≤ 1

private theorem one_eval
    (assignment : Nat -> Nat) (one : assignment 0 = 1) :
    lcEval assignment [(0, 1)] = 1 := by
  simp [lcEval, one, goldilocksP]

private theorem singleton_eval
    (assignment : Nat -> Nat) (column : Nat)
    (canonical : assignment column < goldilocksP) :
    lcEval assignment [(column, 1)] = assignment column := by
  simp [lcEval, Nat.mod_eq_of_lt canonical]

private theorem canonical_rows_hold
    {assignment : Nat -> Nat} {layout : Layout}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (CanonicalU64Recipe.rows (canonicalLayout layout)) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

/-- The embedded canonical-u64 occurrence derives the exact candidate value. -/
theorem canonical_refines
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    CanonicalU64RecipeSound.Refines assignment (canonicalLayout layout) :=
  CanonicalU64RecipeSound.sound goldilocks_euclidPrime canonical one
    (canonical_rows_hold satisfied)

theorem input_eq_candidateValue
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    lcEval assignment layout.candidate = candidateValue assignment layout :=
  (canonical_refines canonical one satisfied).input_eq

theorem candidateValue_lt_modulus
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    candidateValue assignment layout < goldilocksP :=
  (canonical_refines canonical one satisfied).canonical

private theorem acceptance_rows_hold
    {assignment : Nat -> Nat} {layout : Layout}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (acceptanceRows layout) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem residue_rows_hold
    {assignment : Nat -> Nat} {layout : Layout}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (residueRangeRows layout) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem quotient_bit_rows_hold
    {assignment : Nat -> Nat} {layout : Layout}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (quotientBitRows layout) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

private theorem quotient_upper_rows_hold
    {assignment : Nat -> Nat} {layout : Layout}
    (satisfied : Satisfies (rows layout) assignment) :
    Satisfies (quotientUpperRows layout) assignment := by
  intro row member
  exact satisfied row (by simp [rows, member])

theorem quotient_bits_boolean
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    QuotientBitsBoolean assignment layout := by
  intro index bounded
  apply bitRow_le_one goldilocks_euclidPrime (canonical _) one
  apply quotient_bit_rows_hold satisfied
  exact List.mem_map.mpr
    ⟨index, List.mem_range.mpr bounded, rfl⟩

private theorem oneMinusAccept_eval
    {assignment : Nat -> Nat} {layout : Layout}
    (one : assignment 0 = 1)
    (acceptLe : assignment (acceptColumn layout) ≤ 1) :
    lcEval assignment (oneMinusAccept layout) =
      1 - assignment (acceptColumn layout) := by
  have cases : assignment (acceptColumn layout) = 0 ∨
      assignment (acceptColumn layout) = 1 := by omega
  rcases cases with zero | oneValue
  · simp [oneMinusAccept, lcEval, one, zero, goldilocksP]
  · simp [oneMinusAccept, lcEval, one, oneValue, goldilocksP]

private theorem rejectionDifference_eval
    {assignment : Nat -> Nat} {layout : Layout}
    (one : assignment 0 = 1) :
    lcEval assignment (rejectionDifference layout) =
      (lcEval assignment layout.candidate + 1) % goldilocksP := by
  rw [rejectionDifference, KHorner.lcEval_append]
  simp [lcEval, one, Nat.add_mod]

private theorem add_one_mod_zero_iff
    {value : Nat} (valueLt : value < goldilocksP) :
    (value + 1) % goldilocksP = 0 ↔ value = goldilocksP - 1 := by
  constructor
  · intro zero
    by_cases last : value = goldilocksP - 1
    · exact last
    · have small : value + 1 < goldilocksP := by omega
      rw [Nat.mod_eq_of_lt small] at zero
      omega
  · intro last
    rw [last]
    simp [goldilocksP]

/-- The accept wire rejects exactly `q - 1`. -/
theorem acceptance_sound
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (acceptColumn layout) =
      if candidateValue assignment layout < goldilocksP - 1 then 1 else 0 := by
  have acceptance := acceptance_rows_hold satisfied
  have acceptLe : assignment (acceptColumn layout) ≤ 1 :=
    bitRow_le_one goldilocks_euclidPrime (canonical _) one
      (acceptance _ (by simp [acceptanceRows]))
  have candidateEq := input_eq_candidateValue canonical one satisfied
  have candidateLt := candidateValue_lt_modulus canonical one satisfied
  have differenceEq := rejectionDifference_eval (layout := layout) one
  rw [candidateEq] at differenceEq
  have zeroIff := add_one_mod_zero_iff candidateLt
  by_cases accepted : candidateValue assignment layout < goldilocksP - 1
  · rw [if_pos accepted]
    have notLast : candidateValue assignment layout ≠ goldilocksP - 1 := by
      omega
    have differenceNonzero :
        lcEval assignment (rejectionDifference layout) ≠ 0 := by
      rw [differenceEq]
      exact fun zero => notLast (zeroIff.mp zero)
    have zeroProduct := acceptance
      ⟨oneMinusAccept layout, rejectionDifference layout, []⟩
      (by simp [acceptanceRows])
    simp only [RowHolds] at zeroProduct
    rw [oneMinusAccept_eval one acceptLe] at zeroProduct
    simp only [lcEval, List.foldl, Nat.zero_mod] at zeroProduct
    rcases goldilocks_euclidPrime _ _ zeroProduct with firstZero | secondZero
    · have firstLt : 1 - assignment (acceptColumn layout) < goldilocksP := by
        omega
      rw [Nat.mod_eq_of_lt firstLt] at firstZero
      omega
    · apply False.elim
      apply differenceNonzero
      simpa only [lcEval, Nat.mod_mod] using secondZero
  · rw [if_neg accepted]
    have last : candidateValue assignment layout = goldilocksP - 1 := by
      omega
    have differenceZero :
        lcEval assignment (rejectionDifference layout) = 0 := by
      rw [differenceEq]
      exact zeroIff.mpr last
    have inverseProduct := acceptance
      ⟨rejectionDifference layout, [(inverseColumn layout, 1)],
        [(acceptColumn layout, 1)]⟩
      (by simp [acceptanceRows])
    simp only [RowHolds] at inverseProduct
    rw [differenceZero] at inverseProduct
    simp only [Nat.zero_mul, Nat.zero_mod] at inverseProduct
    rw [singleton_eval assignment _ (canonical _)] at inverseProduct
    exact inverseProduct.symm

private theorem fieldSub_eq_zero_iff
    {value amount : Nat}
    (valueLt : value < goldilocksP)
    (amountPositive : 0 < amount)
    (amountLt : amount < goldilocksP) :
    (value + (goldilocksP - amount)) % goldilocksP = 0 ↔
      value = amount := by
  have shifted : value + (goldilocksP - amount) =
      value + goldilocksP - amount := by omega
  rw [shifted]
  constructor
  · intro zero
    by_cases small : value < amount
    · have shiftedLt : value + goldilocksP - amount < goldilocksP := by
        omega
      rw [Nat.mod_eq_of_lt shiftedLt] at zero
      omega
    · have rearranged : value + goldilocksP - amount =
          (value - amount) + goldilocksP := by omega
      rw [rearranged, Nat.add_mod] at zero
      simp only [Nat.mod_self, Nat.add_zero, Nat.mod_mod] at zero
      rw [Nat.mod_eq_of_lt (by omega : value - amount < goldilocksP)] at zero
      omega
  · intro equal
    subst value
    simp [Nat.add_sub_cancel_left]

/-- The range polynomial restricts the residue to `0..4`. -/
theorem residue_range_sound
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (residueColumn layout) < 5 := by
  have residueRows := residue_rows_hold satisfied
  have first := residueRows
    ⟨[(residueColumn layout, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 1)],
      [(productColumn layout 0, 1)]⟩ (by simp [residueRangeRows])
  have second := residueRows
    ⟨[(productColumn layout 0, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 2)],
      [(productColumn layout 1, 1)]⟩ (by simp [residueRangeRows])
  have third := residueRows
    ⟨[(productColumn layout 1, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 3)],
      [(productColumn layout 2, 1)]⟩ (by simp [residueRangeRows])
  have fourth := residueRows
    ⟨[(productColumn layout 2, 1)],
      [(residueColumn layout, 1), (0, goldilocksP - 4)], []⟩
      (by simp [residueRangeRows])
  simp only [RowHolds, lcEval, List.foldl, one, Nat.one_mul,
    Nat.mul_one, Nat.zero_add, Nat.zero_mod,
    Nat.mod_eq_of_lt (canonical _)] at first second third fourth
  let residue := assignment (residueColumn layout)
  let product0 := assignment (productColumn layout 0)
  let product1 := assignment (productColumn layout 1)
  let product2 := assignment (productColumn layout 2)
  change residue * ((residue + (goldilocksP - 1)) % goldilocksP) %
      goldilocksP = product0 at first
  change product0 * ((residue + (goldilocksP - 2)) % goldilocksP) %
      goldilocksP = product1 at second
  change product1 * ((residue + (goldilocksP - 3)) % goldilocksP) %
      goldilocksP = product2 at third
  change product2 * ((residue + (goldilocksP - 4)) % goldilocksP) %
      goldilocksP = 0 at fourth
  change residue < 5
  have residueLt : residue < goldilocksP := canonical _
  have product0Lt : product0 < goldilocksP := canonical _
  have product1Lt : product1 < goldilocksP := canonical _
  have product2Lt : product2 < goldilocksP := canonical _
  rcases goldilocks_euclidPrime _ _ fourth with product2Zero | residueFour
  · rw [Nat.mod_eq_of_lt product2Lt] at product2Zero
    rw [product2Zero] at third
    rcases goldilocks_euclidPrime _ _ third with product1Zero | residueThree
    · rw [Nat.mod_eq_of_lt product1Lt] at product1Zero
      rw [product1Zero] at second
      rcases goldilocks_euclidPrime _ _ second with product0Zero | residueTwo
      · rw [Nat.mod_eq_of_lt product0Lt] at product0Zero
        rw [product0Zero] at first
        rcases goldilocks_euclidPrime _ _ first with residueZero | residueOne
        · rw [Nat.mod_eq_of_lt residueLt] at residueZero
          omega
        · have equalsOne : residue = 1 :=
            (fieldSub_eq_zero_iff residueLt (by decide) (by decide)).mp
              (by simpa only [Nat.mod_mod] using residueOne)
          omega
      · have equalsTwo : residue = 2 :=
          (fieldSub_eq_zero_iff residueLt (by decide) (by decide)).mp
            (by simpa only [Nat.mod_mod] using residueTwo)
        omega
    · have equalsThree : residue = 3 :=
        (fieldSub_eq_zero_iff residueLt (by decide) (by decide)).mp
          (by simpa only [Nat.mod_mod] using residueThree)
      omega
  · have equalsFour : residue = 4 :=
      (fieldSub_eq_zero_iff residueLt (by decide) (by decide)).mp
        (by simpa only [Nat.mod_mod] using residueFour)
    omega

private theorem range6_shape : List.range quotientBitCount =
    [0, 1, 2, 3, 4, 5] := by decide

private theorem upper_row_zero
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (satisfied : Satisfies (rows layout) assignment)
    (highOne : assignment (quotientBitColumn layout 5) = 1)
    (index : Nat) (bounded : index < 5) :
    assignment (quotientBitColumn layout index) = 0 := by
  have holds := quotient_upper_rows_hold satisfied
    ⟨[(quotientBitColumn layout 5, 1)],
      [(quotientBitColumn layout index, 1)], []⟩
    (List.mem_map.mpr ⟨index, List.mem_range.mpr bounded, rfl⟩)
  simp only [RowHolds, lcEval, List.foldl, Nat.one_mul, Nat.mul_one,
    Nat.zero_add, Nat.zero_mod, Nat.mod_eq_of_lt (canonical _), highOne,
    Nat.one_mul] at holds
  have bitMod :
      assignment (quotientBitColumn layout index) % goldilocksP = 0 := by
    simpa [goldilocksP, Nat.mod_mod] using holds
  rw [Nat.mod_eq_of_lt (canonical _)] at bitMod
  exact bitMod

/-- The six-bit quotient is in `0..32`, not `0..63`. -/
theorem quotientValue_le_32
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    quotientValue assignment layout ≤ 32 := by
  have bits := quotient_bits_boolean canonical one satisfied
  have b0 := bits 0 (by decide)
  have b1 := bits 1 (by decide)
  have b2 := bits 2 (by decide)
  have b3 := bits 3 (by decide)
  have b4 := bits 4 (by decide)
  have b5 := bits 5 (by decide)
  by_cases highZero : assignment (quotientBitColumn layout 5) = 0
  · simp [quotientValue, range6_shape, highZero] at *
    omega
  · have highOne : assignment (quotientBitColumn layout 5) = 1 := by omega
    have z0 := upper_row_zero canonical satisfied highOne 0 (by decide)
    have z1 := upper_row_zero canonical satisfied highOne 1 (by decide)
    have z2 := upper_row_zero canonical satisfied highOne 2 (by decide)
    have z3 := upper_row_zero canonical satisfied highOne 3 (by decide)
    have z4 := upper_row_zero canonical satisfied highOne 4 (by decide)
    simp [quotientValue, range6_shape, highOne, z0, z1, z2, z3, z4]

private theorem weighted_fold_bound
    (assignment : Nat -> Nat) (layout : Layout)
    (bits : forall index, index < 64 ->
      CanonicalU64RecipeSound.bitValue assignment
        (canonicalLayout layout) index ≤ 1)
    (indices : List Nat) (members : forall index, index ∈ indices -> index < 64)
    (initial : Nat) :
    indices.foldl
        (fun value index => value + (2 ^ index % 5) *
          CanonicalU64RecipeSound.bitValue assignment
            (canonicalLayout layout) index) initial ≤
      initial + 4 * indices.length := by
  induction indices generalizing initial with
  | nil => simp
  | cons head tail inductionHypothesis =>
      have headLt := members head (by simp)
      have headBit := bits head headLt
      have weightLt : 2 ^ head % 5 < 5 := Nat.mod_lt _ (by decide)
      have termLe :
          (2 ^ head % 5) *
              CanonicalU64RecipeSound.bitValue assignment
                (canonicalLayout layout) head ≤ 4 := by
        have weightLe : 2 ^ head % 5 ≤ 4 := by omega
        exact Nat.le_trans (Nat.mul_le_mul weightLe headBit)
          (by norm_num)
      have tailMembers : forall index, index ∈ tail -> index < 64 := by
        intro index member
        exact members index (by simp [member])
      have rest := inductionHypothesis tailMembers
        (initial + (2 ^ head % 5) *
          CanonicalU64RecipeSound.bitValue assignment
            (canonicalLayout layout) head)
      simp only [List.foldl_cons, List.length_cons] at rest ⊢
      omega

theorem weightedValue_le_256
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    weightedValue assignment layout ≤ 256 := by
  have refined := canonical_refines canonical one satisfied
  have bound := weighted_fold_bound assignment layout refined.bit
    (List.range 64) (fun index member => List.mem_range.mp member) 0
  simpa [weightedValue] using bound

private theorem weightedBitTerms_eval
    {assignment : Nat -> Nat} {layout : Layout}
    (weightedBound : weightedValue assignment layout ≤ 256) :
    lcEval assignment (weightedBitTerms layout) =
      weightedValue assignment layout := by
  unfold lcEval weightedBitTerms weightedValue
  simp only [List.foldl_map, CanonicalU64RecipeSound.bitValue]
  rw [Nat.mod_eq_of_lt]
  exact Nat.lt_of_le_of_lt weightedBound
    (by decide : 256 < goldilocksP)

private theorem quotientTerms_eval
    {assignment : Nat -> Nat} {layout : Layout}
    (quotientBound : quotientValue assignment layout ≤ 32) :
    lcEval assignment (quotientTerms layout) =
      5 * quotientValue assignment layout := by
  have raw :
      (quotientTerms layout).foldl
          (fun value term => value + term.2 * assignment term.1) 0 =
        5 * quotientValue assignment layout := by
    unfold quotientTerms quotientValue
    rw [range6_shape]
    simp only [List.map, List.foldl]
    ring
  unfold lcEval
  rw [raw, Nat.mod_eq_of_lt]
  have productLe : 5 * quotientValue assignment layout ≤ 160 := by
    exact Nat.mul_le_mul_left 5 quotientBound
  exact Nat.lt_of_le_of_lt productLe
    (by decide : 160 < goldilocksP)

private theorem modulo_equation_sound
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    weightedValue assignment layout =
      5 * quotientValue assignment layout +
        assignment (residueColumn layout) := by
  have weightedBound := weightedValue_le_256 canonical one satisfied
  have quotientBound := quotientValue_le_32 canonical one satisfied
  have residueBound := residue_range_sound canonical one satisfied
  have holds := satisfied (moduloFiveRow layout) (by simp [rows])
  have weightedEval := weightedBitTerms_eval weightedBound
  have quotientEval := quotientTerms_eval quotientBound
  have rightEval :
      lcEval assignment
          (quotientTerms layout ++ [(residueColumn layout, 1)]) =
        5 * quotientValue assignment layout +
          assignment (residueColumn layout) := by
    rw [KHorner.lcEval_append, quotientEval,
      singleton_eval assignment _ (canonical _)]
    rw [Nat.mod_eq_of_lt]
    have rightLe :
        5 * quotientValue assignment layout +
            assignment (residueColumn layout) ≤ 164 := by omega
    exact Nat.lt_of_le_of_lt rightLe
      (by decide : 164 < goldilocksP)
  rw [RowHolds, moduloFiveRow, weightedEval,
    one_eval assignment one, rightEval] at holds
  have weightedLt : weightedValue assignment layout < goldilocksP :=
    Nat.lt_of_le_of_lt weightedBound
      (by decide : 256 < goldilocksP)
  simpa [Nat.mod_eq_of_lt weightedLt] using holds

private theorem fold_mod_five
    (assignment : Nat -> Nat) (layout : Layout)
    (indices : List Nat) (left right : Nat)
    (initial : left % 5 = right % 5) :
    (indices.foldl
        (fun value index => value + 2 ^ index *
          CanonicalU64RecipeSound.bitValue assignment
            (canonicalLayout layout) index) left) % 5 =
      (indices.foldl
        (fun value index => value + (2 ^ index % 5) *
          CanonicalU64RecipeSound.bitValue assignment
            (canonicalLayout layout) index) right) % 5 := by
  induction indices generalizing left right with
  | nil => exact initial
  | cons head tail inductionHypothesis =>
      simp only [List.foldl_cons]
      apply inductionHypothesis
      simpa [Nat.add_mod, Nat.mul_mod, initial]

theorem candidateValue_mod_five_eq_weighted
    (assignment : Nat -> Nat) (layout : Layout) :
    candidateValue assignment layout % 5 =
      weightedValue assignment layout % 5 := by
  unfold candidateValue CanonicalU64RecipeSound.bitsValue weightedValue
  exact fold_mod_five assignment layout (List.range 64) 0 0 rfl

/-- The residue wire is the exact candidate integer modulo five. -/
theorem residue_sound
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    assignment (residueColumn layout) = candidateValue assignment layout % 5 := by
  have equation := modulo_equation_sound canonical one satisfied
  have residueBound := residue_range_sound canonical one satisfied
  have congruent := candidateValue_mod_five_eq_weighted assignment layout
  rw [equation] at congruent
  simpa [Nat.add_mod, Nat.mod_eq_of_lt residueBound] using congruent.symm

/-- Complete row-derived result for one V2 full-field candidate. -/
structure Refines (assignment : Nat -> Nat) (layout : Layout) : Prop where
  input : lcEval assignment layout.candidate = candidateValue assignment layout
  canonical : candidateValue assignment layout < goldilocksP
  accepted : assignment (acceptColumn layout) =
    if candidateValue assignment layout < goldilocksP - 1 then 1 else 0
  residue : assignment (residueColumn layout) =
    candidateValue assignment layout % 5

theorem sound
    {assignment : Nat -> Nat} {layout : Layout}
    (canonical : forall column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : Satisfies (rows layout) assignment) :
    Refines assignment layout where
  input := input_eq_candidateValue canonical one satisfied
  canonical := candidateValue_lt_modulus canonical one satisfied
  accepted := acceptance_sound canonical one satisfied
  residue := residue_sound canonical one satisfied

end Nightstream.Implementation.NebulaV2.ProductPiRlcFullFieldCandidateSound
