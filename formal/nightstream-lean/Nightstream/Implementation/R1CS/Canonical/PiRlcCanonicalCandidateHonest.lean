import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateSound
import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest

/-!
Contract: honest completeness for one Lean-owned `Pi_RLC` candidate
classification.

The witness is computed from the caller's authoritative source bits and prior
accepted-count expression.  It writes exactly the candidate recipe's 22
allocated columns.  The only arithmetic oracle is the same global
Goldilocks-inverse primitive used by the canonical-u64 witness.

Neither row satisfaction nor a semantic acceptance conclusion is an input.
-/

set_option autoImplicit false
set_option maxRecDepth 10000

namespace Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateHonest

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidate
open Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateSound
open Nightstream.SuperNeo.Folding.Nifs.NonInteractive.PiRlcSampler

abbrev FieldInverse :=
  Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeHonest.FieldInverse

def sourceValue (initial : Nat → Nat) (layout : Layout) : Nat :=
  chunkValue initial layout

def acceptValue (initial : Nat → Nat) (layout : Layout) : Nat :=
  if sourceValue initial layout =
      ProductionAlphabet.rejectionBucket then 0 else 1

def differenceValue (initial : Nat → Nat) (layout : Layout) : Nat :=
  (sourceValue initial layout + goldilocksP -
    ProductionAlphabet.rejectionBucket) % goldilocksP

def quotient (initial : Nat → Nat) (layout : Layout) : Nat :=
  sourceValue initial layout / ProductionAlphabet.alphabetSize

def residue (initial : Nat → Nat) (layout : Layout) : Nat :=
  sourceValue initial layout % ProductionAlphabet.alphabetSize

def quotientBit
    (initial : Nat → Nat) (layout : Layout) (offset : Nat) : Nat :=
  (quotient initial layout / 2 ^ offset) % 2

def product0 (initial : Nat → Nat) (layout : Layout) : Nat :=
  residue initial layout *
    ((residue initial layout + (goldilocksP - 1)) % goldilocksP) %
      goldilocksP

def product1 (initial : Nat → Nat) (layout : Layout) : Nat :=
  product0 initial layout *
    ((residue initial layout + (goldilocksP - 2)) % goldilocksP) %
      goldilocksP

def product2 (initial : Nat → Nat) (layout : Layout) : Nat :=
  product1 initial layout *
    ((residue initial layout + (goldilocksP - 3)) % goldilocksP) %
      goldilocksP

def cumulative (initial : Nat → Nat) (layout : Layout) : Nat :=
  lcEval initial layout.prior + acceptValue initial layout

/-- Honest assignment.  Caller-owned columns are retained verbatim. -/
def witness
    (field : FieldInverse) (layout : Layout)
    (initial : Nat → Nat) : Nat → Nat :=
  fun column =>
    if column = acceptColumn layout then
      acceptValue initial layout
    else if column = inverseColumn layout then
      field.inverse (differenceValue initial layout)
    else if column = residueColumn layout then
      residue initial layout
    else if column = quotientColumn layout then
      quotient initial layout
    else if column = productColumn layout 0 then
      product0 initial layout
    else if column = productColumn layout 1 then
      product1 initial layout
    else if column = productColumn layout 2 then
      product2 initial layout
    else if layout.base + 7 ≤ column ∧ column < layout.base + 21 then
      quotientBit initial layout (column - (layout.base + 7))
    else if column = cumulativeColumn layout then
      cumulative initial layout
    else
      initial column

private theorem range14 :
    List.range quotientBitCount =
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] := by
  decide

theorem quotient_lt_bound
    {initial : Nat → Nat} {layout : Layout}
    (sourceBits : SourceBitsBoolean initial layout) :
    quotient initial layout < 16384 := by
  have sourceBound := chunkValue_lt_bound sourceBits
  change sourceValue initial layout / 5 < 16384
  change sourceValue initial layout < 65536 at sourceBound
  omega

theorem residue_lt_five
    (initial : Nat → Nat) (layout : Layout) :
    residue initial layout < 5 := by
  unfold residue
  change sourceValue initial layout % 5 < 5
  exact Nat.mod_lt _ (by decide)

theorem quotientBit_le_one
    (initial : Nat → Nat) (layout : Layout) (offset : Nat) :
    quotientBit initial layout offset ≤ 1 := by
  unfold quotientBit
  have bounded :=
    Nat.mod_lt (quotient initial layout / 2 ^ offset) (by decide : 0 < 2)
  omega

theorem quotient_recomposes
    {initial : Nat → Nat} {layout : Layout}
    (sourceBits : SourceBitsBoolean initial layout) :
    (List.range quotientBitCount).foldl
        (fun value offset =>
          value + 2 ^ offset * quotientBit initial layout offset) 0 =
      quotient initial layout := by
  have bound := quotient_lt_bound sourceBits
  unfold quotientBit
  rw [range14]
  simp
  omega

theorem source_decomposes (initial : Nat → Nat) (layout : Layout) :
    sourceValue initial layout =
      5 * quotient initial layout + residue initial layout := by
  unfold quotient residue
  change sourceValue initial layout =
    5 * (sourceValue initial layout / 5) +
      sourceValue initial layout % 5
  omega

theorem acceptValue_le_one (initial : Nat → Nat) (layout : Layout) :
    acceptValue initial layout ≤ 1 := by
  unfold acceptValue
  split <;> omega

theorem acceptValue_lt_modulus (initial : Nat → Nat) (layout : Layout) :
    acceptValue initial layout < goldilocksP := by
  have bounded := acceptValue_le_one initial layout
  have modulus : 1 < goldilocksP := by decide
  omega

theorem differenceValue_lt_modulus
    (initial : Nat → Nat) (layout : Layout) :
    differenceValue initial layout < goldilocksP := by
  unfold differenceValue
  exact Nat.mod_lt _ (by decide)

theorem quotient_lt_modulus
    {initial : Nat → Nat} {layout : Layout}
    (sourceBits : SourceBitsBoolean initial layout) :
    quotient initial layout < goldilocksP :=
  Nat.lt_trans (quotient_lt_bound sourceBits) (by decide)

theorem residue_lt_modulus (initial : Nat → Nat) (layout : Layout) :
    residue initial layout < goldilocksP :=
  Nat.lt_trans (residue_lt_five initial layout) (by decide)

theorem quotientBit_lt_modulus
    (initial : Nat → Nat) (layout : Layout) (offset : Nat) :
    quotientBit initial layout offset < goldilocksP := by
  have bounded := quotientBit_le_one initial layout offset
  have modulus : 1 < goldilocksP := by decide
  omega

theorem product0_lt_modulus (initial : Nat → Nat) (layout : Layout) :
    product0 initial layout < goldilocksP := by
  unfold product0
  exact Nat.mod_lt _ (by decide)

theorem product1_lt_modulus (initial : Nat → Nat) (layout : Layout) :
    product1 initial layout < goldilocksP := by
  unfold product1
  exact Nat.mod_lt _ (by decide)

theorem product2_lt_modulus (initial : Nat → Nat) (layout : Layout) :
    product2 initial layout < goldilocksP := by
  unfold product2
  exact Nat.mod_lt _ (by decide)

@[simp] theorem witness_accept
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat) :
    witness field layout initial (acceptColumn layout) =
      acceptValue initial layout := by
  simp [witness, acceptColumn, inverseColumn, residueColumn, quotientColumn,
    productColumn, cumulativeColumn]

@[simp] theorem witness_inverse
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat) :
    witness field layout initial (inverseColumn layout) =
      field.inverse (differenceValue initial layout) := by
  simp [witness, acceptColumn, inverseColumn, residueColumn, quotientColumn,
    productColumn, cumulativeColumn]

@[simp] theorem witness_residue
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat) :
    witness field layout initial (residueColumn layout) =
      residue initial layout := by
  simp [witness, acceptColumn, inverseColumn, residueColumn, quotientColumn,
    productColumn, cumulativeColumn]

@[simp] theorem witness_quotient
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat) :
    witness field layout initial (quotientColumn layout) =
      quotient initial layout := by
  simp [witness, acceptColumn, inverseColumn, residueColumn, quotientColumn,
    productColumn, cumulativeColumn]

@[simp] theorem witness_product0
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat) :
    witness field layout initial (productColumn layout 0) =
      product0 initial layout := by
  unfold witness
  rw [if_neg (by simp only [acceptColumn, productColumn]; omega)]
  rw [if_neg (by simp [inverseColumn, productColumn])]
  rw [if_neg (by simp [residueColumn, productColumn])]
  rw [if_neg (by simp [quotientColumn, productColumn])]
  rw [if_pos rfl]

@[simp] theorem witness_product1
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat) :
    witness field layout initial (productColumn layout 1) =
      product1 initial layout := by
  unfold witness
  rw [if_neg (by simp only [acceptColumn, productColumn]; omega)]
  rw [if_neg (by simp [inverseColumn, productColumn])]
  rw [if_neg (by simp [residueColumn, productColumn])]
  rw [if_neg (by simp [quotientColumn, productColumn])]
  rw [if_neg (by simp [productColumn])]
  rw [if_pos rfl]

@[simp] theorem witness_product2
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat) :
    witness field layout initial (productColumn layout 2) =
      product2 initial layout := by
  unfold witness
  rw [if_neg (by simp only [acceptColumn, productColumn]; omega)]
  rw [if_neg (by simp [inverseColumn, productColumn])]
  rw [if_neg (by simp [residueColumn, productColumn])]
  rw [if_neg (by simp [quotientColumn, productColumn])]
  rw [if_neg (by simp [productColumn])]
  rw [if_neg (by simp [productColumn])]
  rw [if_pos rfl]

@[simp] theorem witness_quotientBit
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    {offset : Nat} (bounded : offset < quotientBitCount) :
    witness field layout initial (quotientBitColumn layout offset) =
      quotientBit initial layout offset := by
  change offset < 14 at bounded
  unfold witness quotientBitColumn
  rw [if_neg (by simp [acceptColumn]; omega)]
  rw [if_neg (by simp [inverseColumn]; omega)]
  rw [if_neg (by simp [residueColumn]; omega)]
  rw [if_neg (by simp [quotientColumn]; omega)]
  rw [if_neg (by simp [productColumn]; omega)]
  rw [if_neg (by simp [productColumn]; omega)]
  rw [if_neg (by simp [productColumn]; omega)]
  rw [if_pos (by omega)]
  congr 1
  omega

@[simp] theorem witness_cumulative
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat) :
    witness field layout initial (cumulativeColumn layout) =
      cumulative initial layout := by
  simp [witness, acceptColumn, inverseColumn, residueColumn, quotientColumn,
    productColumn, cumulativeColumn]

theorem witness_before
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    {column : Nat} (before : column < layout.base) :
    witness field layout initial column = initial column := by
  unfold witness
  rw [if_neg (by simp [acceptColumn]; omega)]
  rw [if_neg (by simp [inverseColumn]; omega)]
  rw [if_neg (by simp [residueColumn]; omega)]
  rw [if_neg (by simp [quotientColumn]; omega)]
  rw [if_neg (by simp [productColumn]; omega)]
  rw [if_neg (by simp [productColumn]; omega)]
  rw [if_neg (by simp [productColumn]; omega)]
  rw [if_neg (by omega)]
  rw [if_neg (by simp [cumulativeColumn]; omega)]

theorem witness_constant
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    witness field layout initial 0 = 1 := by
  rw [witness_before field layout initial positive, constantWire]

theorem witness_sourceBit
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (below : InputsBelowBase layout) (index : Fin sourceBitCount) :
    witness field layout initial (layout.sourceBit index) =
      initial (layout.sourceBit index) :=
  witness_before field layout initial (below.source index)

theorem witness_prior
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (below : InputsBelowBase layout) :
    lcEval (witness field layout initial) layout.prior =
      lcEval initial layout.prior := by
  apply Nightstream.Implementation.R1CS.Canonical.KMulHonest.lcEval_congr
  intro column mentioned
  rcases List.mem_map.mp mentioned with ⟨term, termMember, columnEq⟩
  rcases term with ⟨termColumn, coefficient⟩
  simp only at columnEq
  subst termColumn
  exact witness_before field layout initial
    (below.prior column coefficient termMember)

theorem witness_sourceBits
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (below : InputsBelowBase layout)
    (sourceBits : SourceBitsBoolean initial layout) :
    SourceBitsBoolean (witness field layout initial) layout := by
  intro index
  rw [witness_sourceBit field layout initial below index]
  exact sourceBits index

theorem witness_chunkValue
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (below : InputsBelowBase layout) :
    chunkValue (witness field layout initial) layout =
      sourceValue initial layout := by
  unfold chunkValue sourceValue
  congr 1
  unfold rawValue
  have go :
      ∀ (indices : List (Fin sourceBitCount)) (accumulator : Nat),
        indices.foldl
            (fun value index =>
              value + 2 ^ index.val *
                witness field layout initial (layout.sourceBit index))
            accumulator =
          indices.foldl
            (fun value index =>
              value + 2 ^ index.val * initial (layout.sourceBit index))
            accumulator := by
    intro indices accumulator
    induction indices generalizing accumulator with
    | nil => rfl
    | cons head tail inductionHypothesis =>
        simp only [List.foldl]
        rw [witness_sourceBit field layout initial below head]
        exact inductionHypothesis _
  exact go _ 0

theorem witness_quotientBits
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat) :
    QuotientBitsBoolean (witness field layout initial) layout := by
  intro offset bounded
  rw [witness_quotientBit field layout initial bounded]
  exact quotientBit_le_one initial layout offset

theorem witness_quotientValue
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (sourceBits : SourceBitsBoolean initial layout) :
    quotientValue (witness field layout initial) layout =
      quotient initial layout := by
  have recomposes := quotient_recomposes sourceBits
  have evaluated :
      quotientValue (witness field layout initial) layout =
        (List.range quotientBitCount).foldl
          (fun value offset =>
            value + 2 ^ offset * quotientBit initial layout offset) 0 := by
    unfold quotientValue
    rw [range14]
    simp only [List.foldl]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
    rw [witness_quotientBit field layout initial (by decide)]
  exact evaluated.trans recomposes

private theorem bitRow_complete
    (assignment : Nat → Nat) (constantWire : assignment 0 = 1)
    (column : Nat) (bounded : assignment column ≤ 1) :
    RowHolds assignment (bitRow column) := by
  have cases : assignment column = 0 ∨ assignment column = 1 := by
    omega
  rcases cases with zero | one
  · simp [RowHolds, bitRow, lcEval, constantWire, zero]
  · simp [RowHolds, bitRow, lcEval, constantWire, one, goldilocksP]

private theorem quotientBitRows_complete
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    Satisfies (quotientBitRows layout) (witness field layout initial) := by
  intro row member
  rcases List.mem_map.mp member with ⟨offset, inRange, rfl⟩
  apply bitRow_complete _ (witness_constant field layout initial
    positive constantWire) _
  rw [witness_quotientBit field layout initial (List.mem_range.mp inRange)]
  exact quotientBit_le_one initial layout offset

private theorem chunkTerms_witness_eval
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    (below : InputsBelowBase layout)
    (sourceBits : SourceBitsBoolean initial layout) :
    lcEval (witness field layout initial) (chunkTerms layout) =
      sourceValue initial layout := by
  rw [chunkTerms_eval
    (witness_constant field layout initial positive constantWire)
    (witness_sourceBits field layout initial below sourceBits)]
  exact witness_chunkValue field layout initial below

private theorem quotientTerms_witness_eval
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (sourceBits : SourceBitsBoolean initial layout) :
    lcEval (witness field layout initial) (quotientTerms layout) =
      quotient initial layout := by
  rw [quotientTerms_eval (witness_quotientBits field layout initial)]
  exact witness_quotientValue field layout initial sourceBits

private theorem differenceTerms_witness_eval
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    (below : InputsBelowBase layout)
    (sourceBits : SourceBitsBoolean initial layout) :
    lcEval (witness field layout initial) (differenceTerms layout) =
      differenceValue initial layout := by
  rw [differenceTerms_eval
    (witness_constant field layout initial positive constantWire)
    (witness_sourceBits field layout initial below sourceBits)]
  rw [witness_chunkValue field layout initial below]
  rfl

private theorem oneMinusAccept_witness_eval
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    lcEval (witness field layout initial) (oneMinusAccept layout) =
      1 - acceptValue initial layout := by
  have evaluated := oneMinusAccept_eval
    (layout := layout)
    (witness_constant field layout initial positive constantWire)
    (show witness field layout initial (acceptColumn layout) ≤ 1 by
      rw [witness_accept]
      exact acceptValue_le_one initial layout)
  simpa using evaluated

private theorem acceptanceRows_complete
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    (below : InputsBelowBase layout)
    (sourceBits : SourceBitsBoolean initial layout) :
    Satisfies (acceptanceRows layout) (witness field layout initial) := by
  have one :=
    witness_constant field layout initial positive constantWire
  have sourceBound := chunkValue_lt_bound sourceBits
  have differenceIff := difference_zero_iff sourceBound
  have differenceEval :=
    differenceTerms_witness_eval field layout initial positive constantWire
      below sourceBits
  have oneMinusEval :=
    oneMinusAccept_witness_eval field layout initial positive constantWire
  intro row member
  simp only [acceptanceRows, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · apply bitRow_complete _ one _
    rw [witness_accept]
    exact acceptValue_le_one initial layout
  · simp only [RowHolds]
    rw [oneMinusEval, differenceEval]
    by_cases rejected :
        sourceValue initial layout = ProductionAlphabet.rejectionBucket
    · have differenceZero : differenceValue initial layout = 0 := by
        apply differenceIff.mpr
        exact rejected
      simp [acceptValue, rejected, differenceZero, lcEval]
    · simp [acceptValue, rejected, lcEval]
  · simp only [RowHolds]
    rw [differenceEval]
    have inverseEval :
        lcEval (witness field layout initial)
            [(inverseColumn layout, 1)] =
          field.inverse (differenceValue initial layout) := by
      simp [lcEval, Nat.mod_eq_of_lt (field.canonical _)]
    have acceptEval :
        lcEval (witness field layout initial)
            [(acceptColumn layout, 1)] =
          acceptValue initial layout := by
      simp [lcEval, Nat.mod_eq_of_lt
        (acceptValue_lt_modulus initial layout)]
    rw [inverseEval, acceptEval]
    by_cases rejected :
        sourceValue initial layout = ProductionAlphabet.rejectionBucket
    · have differenceZero : differenceValue initial layout = 0 :=
        differenceIff.mpr rejected
      simp [acceptValue, rejected, differenceZero, field.zero]
    · have differenceNonzero : differenceValue initial layout ≠ 0 := by
        intro zero
        exact rejected (differenceIff.mp zero)
      have inverseCorrect := field.correct
        (differenceValue initial layout)
        (differenceValue_lt_modulus initial layout)
        differenceNonzero
      simpa [acceptValue, rejected] using inverseCorrect
  · simp only [RowHolds]
    rw [oneMinusEval]
    have inverseEval :
        lcEval (witness field layout initial)
            [(inverseColumn layout, 1)] =
          field.inverse (differenceValue initial layout) := by
      simp [lcEval, Nat.mod_eq_of_lt (field.canonical _)]
    rw [inverseEval]
    by_cases rejected :
        sourceValue initial layout = ProductionAlphabet.rejectionBucket
    · have differenceZero : differenceValue initial layout = 0 :=
        differenceIff.mpr rejected
      simp [acceptValue, rejected, differenceZero, field.zero, lcEval]
    · simp [acceptValue, rejected, lcEval]

private theorem residueRangeRows_complete
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1) :
    Satisfies (residueRangeRows layout) (witness field layout initial) := by
  have one :=
    witness_constant field layout initial positive constantWire
  have residueBound := residue_lt_five initial layout
  have residueCases :
      residue initial layout = 0 ∨ residue initial layout = 1 ∨
      residue initial layout = 2 ∨ residue initial layout = 3 ∨
      residue initial layout = 4 := by
    omega
  intro row member
  simp only [residueRangeRows, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with rfl | rfl | rfl | rfl
  · rcases residueCases with value | value | value | value | value <;>
      simp [RowHolds, lcEval, one, product0, value, goldilocksP]
  · rcases residueCases with value | value | value | value | value <;>
      simp [RowHolds, lcEval, one, product0, product1, value,
        goldilocksP]
  · rcases residueCases with value | value | value | value | value <;>
      simp [RowHolds, lcEval, one, product0, product1, product2, value,
        goldilocksP]
  · rcases residueCases with value | value | value | value | value <;>
      simp [RowHolds, lcEval, one, product0, product1, product2, value,
        goldilocksP]

private theorem quotientRecomposition_complete
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    (sourceBits : SourceBitsBoolean initial layout) :
    RowHolds (witness field layout initial)
      (quotientRecompositionRow layout) := by
  have one :=
    witness_constant field layout initial positive constantWire
  have quotientEval :=
    quotientTerms_witness_eval field layout initial sourceBits
  have quotientBound := quotient_lt_modulus sourceBits
  have singleton :
      lcEval (witness field layout initial)
          [(quotientColumn layout, 1)] =
        quotient initial layout := by
    simp [lcEval, Nat.mod_eq_of_lt quotientBound]
  simp only [RowHolds, quotientRecompositionRow, singleton, quotientEval]
  simp [lcEval, one, Nat.mod_eq_of_lt quotientBound]

private theorem decomposition_complete
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    (below : InputsBelowBase layout)
    (sourceBits : SourceBitsBoolean initial layout) :
    RowHolds (witness field layout initial) (decompositionRow layout) := by
  have one :=
    witness_constant field layout initial positive constantWire
  have sourceEval :=
    chunkTerms_witness_eval field layout initial positive constantWire below
      sourceBits
  have sourceBound := chunkValue_lt_bound sourceBits
  have sourceGoldilocks :
      sourceValue initial layout < goldilocksP :=
    Nat.lt_trans sourceBound (by decide)
  have rightValue :
      5 * quotient initial layout + residue initial layout =
        sourceValue initial layout := by
    exact (source_decomposes initial layout).symm
  have rightEval :
      lcEval (witness field layout initial)
          [(quotientColumn layout, 5), (residueColumn layout, 1)] =
        sourceValue initial layout := by
    simp [lcEval, rightValue, Nat.mod_eq_of_lt sourceGoldilocks]
  simp only [RowHolds, decompositionRow, sourceEval, rightEval]
  simp [lcEval, one, Nat.mod_eq_of_lt sourceGoldilocks]

private theorem cumulative_complete
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    (below : InputsBelowBase layout)
    (priorBound : lcEval initial layout.prior <
      ProductionAlphabet.candidateBound) :
    RowHolds (witness field layout initial) (cumulativeRow layout) := by
  have one :=
    witness_constant field layout initial positive constantWire
  have priorEval := witness_prior field layout initial below
  have sumBound :
      cumulative initial layout < goldilocksP := by
    unfold cumulative
    change lcEval initial layout.prior < 64 at priorBound
    have acceptBound := acceptValue_le_one initial layout
    have modulus : 65 < goldilocksP := by decide
    omega
  have leftEval :
      lcEval (witness field layout initial)
          [(cumulativeColumn layout, 1)] =
        cumulative initial layout := by
    simp [lcEval, Nat.mod_eq_of_lt sumBound]
  have rightEval :
      lcEval (witness field layout initial)
          (layout.prior ++ [(acceptColumn layout, 1)]) =
        cumulative initial layout := by
    rw [Nightstream.Implementation.R1CS.Canonical.KHorner.lcEval_append,
      priorEval]
    have acceptEval :
        lcEval (witness field layout initial)
            [(acceptColumn layout, 1)] =
          acceptValue initial layout := by
      simp [lcEval, Nat.mod_eq_of_lt
        (acceptValue_lt_modulus initial layout)]
    rw [acceptEval]
    exact Nat.mod_eq_of_lt sumBound
  simp only [RowHolds, cumulativeRow, leftEval, rightEval]
  simp [lcEval, one, Nat.mod_eq_of_lt sumBound]

/-- The constructed witness satisfies all 25 candidate rows. -/
theorem complete
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (positive : 0 < layout.base) (constantWire : initial 0 = 1)
    (below : InputsBelowBase layout)
    (sourceBits : SourceBitsBoolean initial layout)
    (priorBound : lcEval initial layout.prior <
      ProductionAlphabet.candidateBound) :
    Satisfies (rows layout) (witness field layout initial) := by
  have acceptance := acceptanceRows_complete field layout initial positive
    constantWire below sourceBits
  have residueRows := residueRangeRows_complete field layout initial positive
    constantWire
  have quotientRows := quotientBitRows_complete field layout initial positive
    constantWire
  have quotientRecomposition := quotientRecomposition_complete field layout
    initial positive constantWire sourceBits
  have decomposition := decomposition_complete field layout initial positive
    constantWire below sourceBits
  have cumulativeRowHolds := cumulative_complete field layout initial positive
    constantWire below priorBound
  intro row member
  unfold rows at member
  rcases List.mem_append.mp member with inHead | inTail
  · rcases List.mem_append.mp inHead with inAcceptanceResidue |
      inQuotientBits
    · rcases List.mem_append.mp inAcceptanceResidue with inAcceptance |
        inResidue
      · exact acceptance row inAcceptance
      · exact residueRows row inResidue
    · exact quotientRows row inQuotientBits
  simp only [List.mem_cons, List.not_mem_nil, or_false] at inTail
  rcases inTail with rfl | rfl | rfl
  · exact quotientRecomposition
  · exact decomposition
  · exact cumulativeRowHolds

theorem witness_canonical
    (field : FieldInverse) (layout : Layout) (initial : Nat → Nat)
    (initialCanonical : ∀ column, initial column < goldilocksP)
    (sourceBits : SourceBitsBoolean initial layout)
    (priorBound : lcEval initial layout.prior <
      ProductionAlphabet.candidateBound) :
    ∀ column, witness field layout initial column < goldilocksP := by
  intro column
  unfold witness
  split
  · exact acceptValue_lt_modulus initial layout
  split
  · exact field.canonical _
  split
  · exact residue_lt_modulus initial layout
  split
  · exact quotient_lt_modulus sourceBits
  split
  · exact product0_lt_modulus initial layout
  split
  · exact product1_lt_modulus initial layout
  split
  · exact product2_lt_modulus initial layout
  split
  · exact quotientBit_lt_modulus initial layout _
  split
  · unfold cumulative
    change lcEval initial layout.prior < 64 at priorBound
    have acceptBound := acceptValue_le_one initial layout
    have modulus : 65 < goldilocksP := by decide
    omega
  · exact initialCanonical column

end Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalCandidateHonest
