import NightstreamFPrime.Circuit.StraightLine
import NightstreamFPrime.Gadgets.Range.CanonicalU64

/-!
Owns exact decoding of one canonical 16-bit PiRLC sampler candidate.

Inputs:
- one caller-owned candidate expression;
- its 16 caller-owned little-endian Boolean bits.

Outputs:
- the exact quotient and remainder for division by five;
- one rejection flag which is one exactly for candidate `65535`.

The quotient and remainder hints are not authority. Fourteen Boolean quotient
bits, exact recomposition, the five-root remainder equation, and the original
candidate bits bind every output. The caller owns the canonical 16-bit input
proof; this gadget owns no Poseidon2 or transcript operation.
-/

namespace NightstreamFPrime.Gadgets.Sampling.Candidate16Five

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.GoldilocksPrime
open NightstreamFPrime.Circuit

def candidateBitCount : Nat := 16
def quotientBitCount : Nat := 14
def auxiliaryCount : Nat := 17
def exactRowCount : Nat := 18
def rejectionBucket : Nat := 65535

structure Interface where
  candidate : Nat → Expr
  candidateBit : Nat → Nat → Expr

def fieldOfNat (value : Nat) : F :=
  NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat value

def quotientExpr (offset : Nat) : Expr :=
  Expr.var offset

def remainderExpr (offset : Nat) : Expr :=
  Expr.var (offset + 1)

def quotientBitExpr (offset index : Nat) : Expr :=
  Expr.var (offset + 2 + index)

def rejectExpr (offset : Nat) : Expr :=
  Expr.var (offset + 16)

def acceptedExpr (offset : Nat) : Expr :=
  1 - rejectExpr offset

def centeredCoefficientExpr (offset : Nat) : Expr :=
  remainderExpr offset - 2

def weightedExpr (bit : Nat → Expr) (count : Nat) : Expr :=
  (List.range count).foldl (fun value index =>
    value + Expr.const (fieldOfNat (2 ^ index)) * bit index) 0

def candidateWordExpr (interface : Interface) (offset : Nat) : Expr :=
  weightedExpr (interface.candidateBit offset) candidateBitCount

/-- The exact low or high 16-bit view of one canonical-u64 decomposition. -/
def canonicalWindowInterface (wordOffset : Nat) (part : Fin 2) : Interface where
  candidate := fun _ =>
    NightstreamFPrime.Gadgets.Range.CanonicalU64.weightedExpr wordOffset
      (16 * part.val) candidateBitCount
  candidateBit := fun _ index =>
    NightstreamFPrime.Gadgets.Range.CanonicalU64.bitExpr wordOffset
      (16 * part.val + index)

def quotientWordExpr (offset : Nat) : Expr :=
  weightedExpr (quotientBitExpr offset) quotientBitCount

def productExpr (bit : Nat → Expr) : Nat → Expr
  | 0 => 1
  | count + 1 => productExpr bit count * bit count

def rejectRecipe (interface : Interface) (offset : Nat) : Expr :=
  productExpr (interface.candidateBit offset) candidateBitCount

def quotientRemainderHints (interface : Interface) (offset : Nat) : List Hint :=
  [.quotientFive (interface.candidate offset),
    .remainderFive (interface.candidate offset)]

def quotientBitHints (offset : Nat) : List Hint :=
  (List.range quotientBitCount).map fun index =>
    .bit (quotientExpr offset) index

def quotientBooleanConstraint (offset index : Nat) : Expr :=
  quotientBitExpr offset index * (quotientBitExpr offset index - 1)

def quotientBooleanOps (offset : Nat) : List Op :=
  (List.range quotientBitCount).map fun index =>
    .assertZero (quotientBooleanConstraint offset index)

def quotientRecompositionConstraint (offset : Nat) : Expr :=
  quotientExpr offset - quotientWordExpr offset

def divisionConstraint (interface : Interface) (offset : Nat) : Expr :=
  interface.candidate offset -
    (Expr.const (fieldOfNat 5) * quotientExpr offset + remainderExpr offset)

def remainderConstraint (offset : Nat) : Expr :=
  remainderExpr offset *
    (remainderExpr offset - 1) *
    (remainderExpr offset - 2) *
    (remainderExpr offset - 3) *
    (remainderExpr offset - 4)

def operations (interface : Interface) (offset : Nat) : List Op :=
  [ .witness (WitnessBatch.hinted offset
      (quotientRemainderHints interface offset)),
    .witness (WitnessBatch.hinted (offset + 2) (quotientBitHints offset)),
    .witness (WitnessBatch.arithmetic (offset + 16)
      [rejectRecipe interface offset]) ] ++
    quotientBooleanOps offset ++
    [ .assertZero (quotientRecompositionConstraint offset),
      .assertZero (divisionConstraint interface offset),
      .assertZero (remainderConstraint offset) ]

def main (interface : Interface) : Circuit Unit := fun offset =>
  ((), offset + auxiliaryCount, operations interface offset)

theorem localLength_eq (interface : Interface) (offset : Nat) :
    localLength (operations interface offset) = auxiliaryCount := by
  simp [operations, quotientBooleanOps, localLength, Op.localLength,
    WitnessBatch.outputLength, quotientRemainderHints, quotientBitHints,
    auxiliaryCount, quotientBitCount, Function.comp_def]

theorem rowCount_eq (interface : Interface) (offset : Nat) :
    NightstreamFPrime.Circuit.rowCount (operations interface offset) =
      exactRowCount := by
  simp [operations, quotientBooleanOps, NightstreamFPrime.Circuit.rowCount,
    Op.rowCount, quotientBitHints, exactRowCount, quotientBitCount,
    Function.comp_def]

theorem flatConstraints_length_eq (interface : Interface) (offset : Nat) :
    (flatConstraints (operations interface offset)).length = exactRowCount := by
  rw [flatConstraints_length_eq_rowCount]
  exact rowCount_eq interface offset

def bitValue (env : Env) (bit : Nat → Expr) (index : Nat) : Nat :=
  ((bit index).eval env).val

def weightedValue (env : Env) (bit : Nat → Expr) (count : Nat) : Nat :=
  (List.range count).foldl (fun value index =>
    value + 2 ^ index * bitValue env bit index) 0

def candidateValue (interface : Interface) (env : Env) (offset : Nat) : Nat :=
  weightedValue env (interface.candidateBit offset) candidateBitCount

def quotientValue (env : Env) (offset : Nat) : Nat :=
  weightedValue env (quotientBitExpr offset) quotientBitCount

def Assumptions (interface : Interface) (offset : Nat) (env : Env) : Prop :=
  (interface.candidate offset).VarsBelow offset ∧
    (∀ index, index < candidateBitCount →
      (interface.candidateBit offset index).VarsBelow offset) ∧
    ((interface.candidate offset).eval env).val =
      candidateValue interface env offset ∧
    ∀ index, index < candidateBitCount →
      bitValue env (interface.candidateBit offset) index < 2

structure Refines (interface : Interface) (offset : Nat) (env : Env) : Prop where
  remainder_eq : ((remainderExpr offset).eval env).val =
    ((interface.candidate offset).eval env).val % 5
  reject_eq : (rejectExpr offset).eval env =
    if ((interface.candidate offset).eval env).val = rejectionBucket then 1 else 0

abbrev SpecHolds := Refines

@[simp] private theorem fieldOfNat_val (value : Nat) :
    (fieldOfNat value).val = value % goldilocksModulus := by
  rfl

private theorem fieldOfNat_add (left right : Nat) :
    fieldOfNat left + fieldOfNat right = fieldOfNat (left + right) := by
  apply Fin.eq_of_val_eq
  simp [fieldOfNat, NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat,
    Fin.val_add, Nat.add_mod]

private theorem fieldOfNat_mul (left right : Nat) :
    fieldOfNat left * fieldOfNat right = fieldOfNat (left * right) := by
  apply Fin.eq_of_val_eq
  simp [fieldOfNat, NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat,
    Fin.val_mul, Nat.mul_mod]

private theorem fieldOfNat_val_self (value : F) :
    fieldOfNat value.val = value := by
  apply Fin.eq_of_val_eq
  simp [fieldOfNat, NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat,
    Nat.mod_eq_of_lt value.isLt]

private theorem weightedExpr_succ (bit : Nat → Expr) (count : Nat) :
    weightedExpr bit (count + 1) =
      weightedExpr bit count +
        Expr.const (fieldOfNat (2 ^ count)) * bit count := by
  simp [weightedExpr, List.range_succ, List.foldl_append]

private theorem weightedValue_succ (env : Env) (bit : Nat → Expr)
    (count : Nat) :
    weightedValue env bit (count + 1) =
      weightedValue env bit count + 2 ^ count * bitValue env bit count := by
  simp [weightedValue, List.range_succ, List.foldl_append]

private theorem weightedExpr_eval (env : Env) (bit : Nat → Expr)
    (count : Nat) :
    (weightedExpr bit count).eval env =
      fieldOfNat (weightedValue env bit count) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [weightedExpr_succ, weightedValue_succ]
      rw [Expr.eval_hadd, Expr.eval_hmul, Expr.eval_const,
        inductionHypothesis]
      rw [show (bit count).eval env = fieldOfNat (bitValue env bit count) by
        exact (fieldOfNat_val_self ((bit count).eval env)).symm,
        fieldOfNat_mul, fieldOfNat_add]

private theorem weightedValue_lt_twoPow
    (env : Env) (bit : Nat → Expr) (count : Nat)
    (binary : ∀ index, index < count → bitValue env bit index < 2) :
    weightedValue env bit count < 2 ^ count := by
  induction count with
  | zero => simp [weightedValue]
  | succ count inductionHypothesis =>
      rw [weightedValue_succ]
      have prefixBound := inductionHypothesis fun index bounded =>
        binary index (by omega)
      have last := binary count (by omega)
      have lastZeroOrOne : bitValue env bit count = 0 ∨
          bitValue env bit count = 1 := by omega
      rcases lastZeroOrOne with lastZero | lastOne
      · rw [lastZero, Nat.mul_zero, Nat.add_zero]
        exact lt_trans prefixBound (by simp [pow_succ])
      · rw [lastOne, Nat.mul_one]
        simp only [pow_succ]
        omega

/-- A proved canonical-u64 lane supplies every assumption of either adjacent
16-bit decoder window. -/
theorem canonicalWindowAssumptions
    {sourceInterface :
      NightstreamFPrime.Gadgets.Range.CanonicalU64.Interface}
    (wordOffset decoderOffset : Nat) (part : Fin 2) (env : Env)
    (wordEnd : wordOffset +
      NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount ≤
        decoderOffset)
    (canonical :
      NightstreamFPrime.Gadgets.Range.CanonicalU64.SpecHolds
        sourceInterface wordOffset env) :
    Assumptions (canonicalWindowInterface wordOffset part) decoderOffset env := by
  let bit := fun index =>
    NightstreamFPrime.Gadgets.Range.CanonicalU64.bitExpr wordOffset
      (16 * part.val + index)
  have partBound : part.val < 2 := part.isLt
  have binary : ∀ index, index < candidateBitCount →
      bitValue env bit index < 2 := by
    intro index bounded
    have sourceBit := canonical.bit_lt_two (16 * part.val + index) (by
      simp only [candidateBitCount,
        NightstreamFPrime.Gadgets.Range.CanonicalU64.bitCount] at bounded ⊢
      omega)
    simpa [bit, bitValue,
      NightstreamFPrime.Gadgets.Range.CanonicalU64.bitValue,
      NightstreamFPrime.Gadgets.Range.CanonicalU64.bitExpr] using sourceBit
  refine ⟨?_, ?_, ?_, binary⟩
  · apply
      NightstreamFPrime.Gadgets.Range.CanonicalU64.weightedExpr_varsBelow
    simp only [candidateBitCount,
      NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount] at wordEnd ⊢
    omega
  · intro index bounded
    apply NightstreamFPrime.Gadgets.Range.CanonicalU64.bitExpr_varsBelow
    simp only [NightstreamFPrime.Gadgets.Range.CanonicalU64.auxiliaryCount]
      at wordEnd
    simp only [candidateBitCount] at bounded
    omega
  · change
      ((NightstreamFPrime.Gadgets.Range.CanonicalU64.weightedExpr wordOffset
        (16 * part.val) candidateBitCount).eval env).val =
        weightedValue env bit candidateBitCount
    have expressionEq :
        NightstreamFPrime.Gadgets.Range.CanonicalU64.weightedExpr wordOffset
            (16 * part.val) candidateBitCount =
          weightedExpr bit candidateBitCount := by
      rfl
    rw [expressionEq, weightedExpr_eval]
    change weightedValue env bit candidateBitCount % goldilocksModulus =
      weightedValue env bit candidateBitCount
    rw [Nat.mod_eq_of_lt]
    exact lt_trans (weightedValue_lt_twoPow env bit candidateBitCount binary)
      (by norm_num [candidateBitCount, goldilocksModulus])

def productValue (env : Env) (bit : Nat → Expr) : Nat → Nat
  | 0 => 1
  | count + 1 => productValue env bit count * bitValue env bit count

private theorem productValue_succ (env : Env) (bit : Nat → Expr)
    (count : Nat) :
    productValue env bit (count + 1) =
      productValue env bit count * bitValue env bit count := by
  rfl

private theorem productValue_lt_two
    (env : Env) (bit : Nat → Expr) (count : Nat)
    (binary : ∀ index, index < count → bitValue env bit index < 2) :
    productValue env bit count < 2 := by
  induction count with
  | zero => simp [productValue]
  | succ count inductionHypothesis =>
      rw [productValue_succ]
      have prefixBound := inductionHypothesis fun index bounded =>
        binary index (by omega)
      have last := binary count (by omega)
      have prefixZeroOrOne : productValue env bit count = 0 ∨
          productValue env bit count = 1 := by omega
      have lastZeroOrOne : bitValue env bit count = 0 ∨
          bitValue env bit count = 1 := by omega
      rcases prefixZeroOrOne with prefixZero | prefixOne
      · simp [prefixZero]
      · rcases lastZeroOrOne with lastZero | lastOne
        · simp [prefixOne, lastZero]
        · simp [prefixOne, lastOne]

private theorem productValue_eq_one_iff
    (env : Env) (bit : Nat → Expr) (count : Nat)
    (binary : ∀ index, index < count → bitValue env bit index < 2) :
    productValue env bit count = 1 ↔
      ∀ index, index < count → bitValue env bit index = 1 := by
  induction count with
  | zero => simp [productValue]
  | succ count inductionHypothesis =>
      rw [productValue_succ]
      have prefixBinary : ∀ index, index < count → bitValue env bit index < 2 :=
        fun index bounded => binary index (by omega)
      have prefixLt := productValue_lt_two env bit count prefixBinary
      have lastLt := binary count (by omega)
      constructor
      · intro productEq index bounded
        have prefixZeroOrOne : productValue env bit count = 0 ∨
            productValue env bit count = 1 := by omega
        have lastZeroOrOne : bitValue env bit count = 0 ∨
            bitValue env bit count = 1 := by omega
        have prefixEq : productValue env bit count = 1 := by
          rcases prefixZeroOrOne with prefixZero | prefixOne
          · rw [prefixZero, Nat.zero_mul] at productEq
            omega
          · exact prefixOne
        have lastEq : bitValue env bit count = 1 := by
          rcases lastZeroOrOne with lastZero | lastOne
          · rw [lastZero, Nat.mul_zero] at productEq
            omega
          · exact lastOne
        by_cases before : index < count
        · exact (inductionHypothesis prefixBinary).mp prefixEq index before
        · have indexEq : index = count := by omega
          simpa [indexEq] using lastEq
      · intro allOne
        have prefixEq := (inductionHypothesis prefixBinary).mpr
          (fun index bounded => allOne index (by omega))
        have lastEq := allOne count (by omega)
        rw [prefixEq, lastEq]

private theorem weightedValue_eq_max_iff
    (env : Env) (bit : Nat → Expr) (count : Nat)
    (binary : ∀ index, index < count → bitValue env bit index < 2) :
    weightedValue env bit count = 2 ^ count - 1 ↔
      ∀ index, index < count → bitValue env bit index = 1 := by
  induction count with
  | zero => simp [weightedValue]
  | succ count inductionHypothesis =>
      rw [weightedValue_succ]
      have prefixBinary : ∀ index, index < count → bitValue env bit index < 2 :=
        fun index bounded => binary index (by omega)
      have prefixLt := weightedValue_lt_twoPow env bit count prefixBinary
      have lastLt := binary count (by omega)
      constructor
      · intro sumEq index bounded
        have lastZeroOrOne : bitValue env bit count = 0 ∨
            bitValue env bit count = 1 := by omega
        have lastEq : bitValue env bit count = 1 := by
          rcases lastZeroOrOne with lastZero | lastOne
          · simp only [lastZero, Nat.mul_zero, Nat.add_zero, pow_succ] at sumEq
            omega
          · exact lastOne
        have prefixEq : weightedValue env bit count = 2 ^ count - 1 := by
          simp only [lastEq, Nat.mul_one, pow_succ] at sumEq
          omega
        by_cases before : index < count
        · exact (inductionHypothesis prefixBinary).mp prefixEq index before
        · have indexEq : index = count := by omega
          simpa [indexEq] using lastEq
      · intro allOne
        have prefixEq := (inductionHypothesis prefixBinary).mpr
          (fun index bounded => allOne index (by omega))
        have lastEq := allOne count (by omega)
        rw [prefixEq, lastEq]
        simp only [Nat.mul_one, pow_succ]
        omega

private theorem rejectRecipe_eval (interface : Interface) (env : Env)
    (offset : Nat) :
    (rejectRecipe interface offset).eval env =
      fieldOfNat (productValue env (interface.candidateBit offset)
        candidateBitCount) := by
  unfold rejectRecipe
  generalize candidateBitCount = count
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [productExpr, productValue, Expr.eval_hmul, inductionHypothesis,
        show (interface.candidateBit offset count).eval env =
            fieldOfNat (bitValue env (interface.candidateBit offset) count) by
          exact (fieldOfNat_val_self
            ((interface.candidateBit offset count).eval env)).symm,
        fieldOfNat_mul]

private theorem boolean_of_constraint (env : Env) (offset index : Nat)
    (zero : (quotientBooleanConstraint offset index).eval env = 0) :
    bitValue env (quotientBitExpr offset) index < 2 := by
  let value := env (offset + 2 + index)
  have product : value * (value - 1) = 0 := by
    simpa [quotientBooleanConstraint, quotientBitExpr, bitValue, value] using zero
  rcases baseFieldNoZeroDivisors value (value - 1) product with
    valueZero | valueOne
  · change value.val < 2
    rw [valueZero]
    norm_num [goldilocksModulus]
  · have equalOne : value = 1 := sub_eq_zero.mp valueOne
    change value.val < 2
    rw [equalOne]
    norm_num [goldilocksModulus]

private theorem quotientBooleanConstraint_eval (env : Env)
    (offset index : Nat) :
    (quotientBooleanConstraint offset index).eval env =
      env (offset + 2 + index) * (env (offset + 2 + index) - 1) := by
  unfold quotientBooleanConstraint quotientBitExpr
  rw [Expr.eval_hmul, Expr.eval_sub]
  rw [show (1 : Expr).eval env = (1 : F) by
    apply Fin.eq_of_val_eq
    norm_num [Expr.eval, goldilocksModulus]]
  simp only [Expr.eval_var]

private theorem quotient_boolean_rows (interface : Interface) (env : Env)
    (offset : Nat) (rows : holds env (operations interface offset)) :
    ∀ index, index < quotientBitCount →
      bitValue env (quotientBitExpr offset) index < 2 := by
  intro index bounded
  have operationMember : .assertZero (quotientBooleanConstraint offset index) ∈
      operations interface offset := by
    unfold operations
    apply List.mem_append.mpr
    left
    apply List.mem_append.mpr
    right
    unfold quotientBooleanOps
    exact List.mem_map.mpr
      ⟨index, List.mem_range.mpr bounded, rfl⟩
  exact boolean_of_constraint env offset index
    (rows (.assertZero (quotientBooleanConstraint offset index)) operationMember)

private theorem quotient_recomposition_row (interface : Interface) (env : Env)
    (offset : Nat) (rows : holds env (operations interface offset)) :
    (quotientRecompositionConstraint offset).eval env = 0 := by
  exact rows (.assertZero (quotientRecompositionConstraint offset)) (by
    simp [operations])

private theorem division_row (interface : Interface) (env : Env)
    (offset : Nat) (rows : holds env (operations interface offset)) :
    (divisionConstraint interface offset).eval env = 0 := by
  exact rows (.assertZero (divisionConstraint interface offset)) (by
    simp [operations])

private theorem remainder_row (interface : Interface) (env : Env)
    (offset : Nat) (rows : holds env (operations interface offset)) :
    (remainderConstraint offset).eval env = 0 := by
  exact rows (.assertZero (remainderConstraint offset)) (by
    simp [operations])

private theorem reject_recipe_row (interface : Interface) (env : Env)
    (offset : Nat) (rows : holds env (operations interface offset)) :
    (rejectExpr offset).eval env = (rejectRecipe interface offset).eval env := by
  have witnessRows := rows
    (.witness (WitnessBatch.arithmetic (offset + 16)
      [rejectRecipe interface offset])) (by simp [operations])
  have equation := witnessRows
    (Expr.var (offset + 16) - rejectRecipe interface offset) (by
      simp [recipeConstraints])
  exact sub_eq_zero.mp (by simpa [rejectExpr] using equation)

private theorem remainder_lt_five (env : Env) (offset : Nat)
    (zero : (remainderConstraint offset).eval env = 0) :
    ((remainderExpr offset).eval env).val < 5 := by
  let remainder := (remainderExpr offset).eval env
  have product :
      remainder * (remainder - 1) * (remainder - 2) *
        (remainder - 3) * (remainder - 4) = 0 := by
    simpa [remainderConstraint, remainder] using zero
  have roots : remainder = 0 ∨ remainder = 1 ∨ remainder = 2 ∨
      remainder = 3 ∨ remainder = 4 := by
    rcases baseFieldNoZeroDivisors
        (remainder * (remainder - 1) * (remainder - 2) * (remainder - 3))
        (remainder - 4) product with prefix3 | root4
    · rcases baseFieldNoZeroDivisors
          (remainder * (remainder - 1) * (remainder - 2))
          (remainder - 3) prefix3 with prefix2 | root3
      · rcases baseFieldNoZeroDivisors
            (remainder * (remainder - 1)) (remainder - 2) prefix2 with
          prefix1 | root2
        · rcases baseFieldNoZeroDivisors remainder (remainder - 1) prefix1 with
          root0 | root1
          · exact Or.inl root0
          · exact Or.inr (Or.inl (sub_eq_zero.mp root1))
        · exact Or.inr (Or.inr (Or.inl (sub_eq_zero.mp root2)))
      · exact Or.inr (Or.inr (Or.inr (Or.inl (sub_eq_zero.mp root3))))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (sub_eq_zero.mp root4))))
  change remainder.val < 5
  rcases roots with root | root | root | root | root <;>
    rw [root] <;> norm_num [goldilocksModulus]

theorem soundness
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (rows : holds env (operations interface offset)) :
    SpecHolds interface offset env := by
  rcases assumptions with ⟨_candidateBelow, _bitsBelow, candidateEq, binary⟩
  have quotientBinary := quotient_boolean_rows interface env offset rows
  have quotientBound : quotientValue env offset < 2 ^ quotientBitCount :=
    weightedValue_lt_twoPow env (quotientBitExpr offset) quotientBitCount
      quotientBinary
  have quotientFieldEq : (quotientExpr offset).eval env =
      fieldOfNat (quotientValue env offset) := by
    have row := quotient_recomposition_row interface env offset rows
    have equality : (quotientExpr offset).eval env =
        (quotientWordExpr offset).eval env := by
      exact sub_eq_zero.mp (by
        simpa [quotientRecompositionConstraint] using row)
    rw [equality]
    exact weightedExpr_eval env (quotientBitExpr offset) quotientBitCount
  have remainderBound : ((remainderExpr offset).eval env).val < 5 :=
    remainder_lt_five env offset (remainder_row interface env offset rows)
  have divisionEq : (interface.candidate offset).eval env =
      fieldOfNat (5 * quotientValue env offset +
        ((remainderExpr offset).eval env).val) := by
    have row := division_row interface env offset rows
    have equality : (interface.candidate offset).eval env =
        fieldOfNat 5 * (quotientExpr offset).eval env +
          (remainderExpr offset).eval env := by
      exact sub_eq_zero.mp (by simpa [divisionConstraint] using row)
    calc
      (interface.candidate offset).eval env =
          fieldOfNat 5 * (quotientExpr offset).eval env +
            (remainderExpr offset).eval env := equality
      _ = fieldOfNat 5 * fieldOfNat (quotientValue env offset) +
          fieldOfNat ((remainderExpr offset).eval env).val := by
        rw [quotientFieldEq,
          fieldOfNat_val_self ((remainderExpr offset).eval env)]
      _ = fieldOfNat (5 * quotientValue env offset +
          ((remainderExpr offset).eval env).val) := by
        rw [fieldOfNat_mul, fieldOfNat_add]
  have divisionNat : ((interface.candidate offset).eval env).val =
      5 * quotientValue env offset + ((remainderExpr offset).eval env).val := by
    have values := congrArg Fin.val divisionEq
    have rightBound : 5 * quotientValue env offset +
        ((remainderExpr offset).eval env).val < goldilocksModulus := by
      simp only [quotientBitCount] at quotientBound
      norm_num [goldilocksModulus] at quotientBound ⊢
      omega
    simpa [fieldOfNat, NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat,
      Nat.mod_eq_of_lt rightBound] using values
  have remainderEq : ((remainderExpr offset).eval env).val =
      ((interface.candidate offset).eval env).val % 5 := by
    omega
  have productLt := productValue_lt_two env
    (interface.candidateBit offset) candidateBitCount binary
  have productIff := productValue_eq_one_iff env
    (interface.candidateBit offset) candidateBitCount binary
  have maximumIff := weightedValue_eq_max_iff env
    (interface.candidateBit offset) candidateBitCount binary
  have rejectEq : (rejectExpr offset).eval env =
      if ((interface.candidate offset).eval env).val = rejectionBucket then 1 else 0 := by
    rw [reject_recipe_row interface env offset rows,
      rejectRecipe_eval interface env offset]
    by_cases rejected : ((interface.candidate offset).eval env).val =
        rejectionBucket
    · simp only [rejected, if_pos]
      apply Fin.eq_of_val_eq
      have candidateMaximum : candidateValue interface env offset =
          2 ^ candidateBitCount - 1 := by
        rw [← candidateEq, rejected]
        norm_num [candidateBitCount, rejectionBucket]
      have allOne := maximumIff.mp candidateMaximum
      have productOne := productIff.mpr allOne
      simp [productOne, fieldOfNat,
        NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat]
    · simp only [rejected]
      apply Fin.eq_of_val_eq
      have notMaximum : candidateValue interface env offset ≠
          2 ^ candidateBitCount - 1 := by
        intro maximum
        apply rejected
        rw [candidateEq, maximum]
        norm_num [candidateBitCount, rejectionBucket]
      have productNotOne : productValue env
          (interface.candidateBit offset) candidateBitCount ≠ 1 := by
        intro productOne
        exact notMaximum (maximumIff.mpr (productIff.mp productOne))
      have productZero : productValue env
          (interface.candidateBit offset) candidateBitCount = 0 := by omega
      simp [productZero, fieldOfNat,
        NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat]
  exact ⟨remainderEq, rejectEq⟩

def bitNat (value index : Nat) : Nat :=
  (value / 2 ^ index) % 2

def bitWindowValue (value count : Nat) : Nat :=
  (List.range count).foldl (fun total index =>
    total + 2 ^ index * bitNat value index) 0

private theorem bitWindowValue_succ (value count : Nat) :
    bitWindowValue value (count + 1) =
      bitWindowValue value count + 2 ^ count * bitNat value count := by
  simp [bitWindowValue, List.range_succ, List.foldl_append]

private theorem bitWindowValue_eq_mod (value count : Nat) :
    bitWindowValue value count = value % 2 ^ count := by
  induction count with
  | zero => simp [bitWindowValue, Nat.mod_one]
  | succ count inductionHypothesis =>
      rw [bitWindowValue_succ, inductionHypothesis, Nat.mod_pow_succ]
      simp [bitNat]

private theorem hintBit_value (env : Env) (source : Expr) (index : Nat) :
    (Hint.eval env (.bit source index)).val = bitNat (source.eval env).val index := by
  change ((((source.eval env).val >>> index) &&& 1) % goldilocksModulus) = _
  rw [Nat.and_one_is_mod, Nat.shiftRight_eq_div_pow]
  apply Nat.mod_eq_of_lt
  exact lt_trans (Nat.mod_lt _ (by decide)) (by
    norm_num [goldilocksModulus])

def completeQuotientRemainder (interface : Interface) (env : Env)
    (offset : Nat) : Env :=
  executeHints env offset (quotientRemainderHints interface offset)

def completeQuotientBits (interface : Interface) (env : Env)
    (offset : Nat) : Env :=
  executeHints (completeQuotientRemainder interface env offset) (offset + 2)
    (quotientBitHints offset)

def completeEnv (interface : Interface) (env : Env) (offset : Nat) : Env :=
  executeRecipes (completeQuotientBits interface env offset) (offset + 16)
    [rejectRecipe interface offset]

private theorem quotientRemainderHints_readBelow
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    HintsReadBelow offset (quotientRemainderHints interface offset) := by
  intro hint member
  simp only [quotientRemainderHints, List.mem_cons, List.mem_nil_iff,
    or_false] at member
  rcases member with rfl | rfl <;> exact assumptions.1

private theorem quotientBitHints_readBelow (offset : Nat) :
    HintsReadBelow (offset + 2) (quotientBitHints offset) := by
  intro hint member
  rcases List.mem_map.mp member with ⟨index, _, rfl⟩
  simp [Hint.source, quotientExpr, Expr.VarsBelow]

private theorem completeQuotientRemainder_quotient
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    completeQuotientRemainder interface env offset offset =
      fieldOfNat (((interface.candidate offset).eval env).val / 5) := by
  have value := executeHints_value_of_readBelow env offset
    (quotientRemainderHints interface offset)
    (quotientRemainderHints_readBelow interface env offset assumptions) 0
      (by norm_num [quotientRemainderHints])
  simpa [completeQuotientRemainder, quotientRemainderHints, Hint.eval,
    Hint.ofNat, fieldOfNat,
    NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat] using value

private theorem completeQuotientRemainder_remainder
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    completeQuotientRemainder interface env offset (offset + 1) =
      fieldOfNat (((interface.candidate offset).eval env).val % 5) := by
  have value := executeHints_value_of_readBelow env offset
    (quotientRemainderHints interface offset)
    (quotientRemainderHints_readBelow interface env offset assumptions) 1
      (by norm_num [quotientRemainderHints])
  simpa [completeQuotientRemainder, quotientRemainderHints, Hint.eval,
    Hint.ofNat, fieldOfNat,
    NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat] using value

private theorem completeEnv_quotient
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    completeEnv interface env offset offset =
      fieldOfNat (((interface.candidate offset).eval env).val / 5) := by
  unfold completeEnv completeQuotientBits
  rw [executeRecipes_agrees_below _ (offset + 16)
      [rejectRecipe interface offset] offset (by omega),
    executeHints_agrees_below _ (offset + 2)
      (quotientBitHints offset) offset (by omega)]
  exact completeQuotientRemainder_quotient interface env offset assumptions

private theorem completeEnv_remainder
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    completeEnv interface env offset (offset + 1) =
      fieldOfNat (((interface.candidate offset).eval env).val % 5) := by
  unfold completeEnv completeQuotientBits
  rw [executeRecipes_agrees_below _ (offset + 16)
      [rejectRecipe interface offset] (offset + 1) (by omega),
    executeHints_agrees_below _ (offset + 2)
      (quotientBitHints offset) (offset + 1) (by omega)]
  exact completeQuotientRemainder_remainder interface env offset assumptions

private theorem completeEnv_source
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    (interface.candidate offset).eval (completeEnv interface env offset) =
      (interface.candidate offset).eval env := by
  apply Expr.eval_eq_of_agree_below _ offset _ _ assumptions.1
  intro index below
  unfold completeEnv completeQuotientBits completeQuotientRemainder
  rw [executeRecipes_agrees_below _ (offset + 16)
      [rejectRecipe interface offset] index (by omega),
    executeHints_agrees_below _ (offset + 2)
      (quotientBitHints offset) index (by omega),
    executeHints_agrees_below env offset
      (quotientRemainderHints interface offset) index below]

private theorem completeEnv_candidateBit
    (interface : Interface) (env : Env) (offset index : Nat)
    (assumptions : Assumptions interface offset env)
    (bounded : index < candidateBitCount) :
    (interface.candidateBit offset index).eval
        (completeEnv interface env offset) =
      (interface.candidateBit offset index).eval env := by
  apply Expr.eval_eq_of_agree_below _ offset _ _
    (assumptions.2.1 index bounded)
  intro current below
  unfold completeEnv completeQuotientBits completeQuotientRemainder
  rw [executeRecipes_agrees_below _ (offset + 16)
      [rejectRecipe interface offset] current (by omega),
    executeHints_agrees_below _ (offset + 2)
      (quotientBitHints offset) current (by omega),
    executeHints_agrees_below env offset
      (quotientRemainderHints interface offset) current below]

private theorem completeEnv_quotientBitValue
    (interface : Interface) (env : Env) (offset index : Nat)
    (assumptions : Assumptions interface offset env)
    (bounded : index < quotientBitCount) :
    bitValue (completeEnv interface env offset) (quotientBitExpr offset) index =
      bitNat (((interface.candidate offset).eval env).val / 5) index := by
  have position : index < (quotientBitHints offset).length := by
    simpa [quotientBitHints, quotientBitCount] using bounded
  have hinted := executeHints_value_of_readBelow
    (completeQuotientRemainder interface env offset) (offset + 2)
    (quotientBitHints offset) (quotientBitHints_readBelow offset) index position
  have quotientBefore :=
    completeQuotientRemainder_quotient interface env offset assumptions
  have quotientBound : ((interface.candidate offset).eval env).val / 5 <
      goldilocksModulus := by
    exact lt_of_le_of_lt (Nat.div_le_self _ 5)
      ((interface.candidate offset).eval env).isLt
  have outputBelow : offset + 2 + index < offset + 16 := by
    simp only [quotientBitCount] at bounded
    omega
  unfold bitValue
  simp only [quotientBitExpr, Expr.eval_var]
  rw [show completeEnv interface env offset (offset + 2 + index) =
      completeQuotientBits interface env offset (offset + 2 + index) by
        exact executeRecipes_agrees_below _ (offset + 16)
          [rejectRecipe interface offset] _ outputBelow,
    show completeQuotientBits interface env offset (offset + 2 + index) =
      Hint.eval (completeQuotientRemainder interface env offset)
        (.bit (quotientExpr offset) index) by
        simpa [completeQuotientBits, quotientBitHints] using hinted,
    hintBit_value]
  change bitNat
      ((quotientExpr offset).eval
        (completeQuotientRemainder interface env offset)).val index = _
  change bitNat
      (completeQuotientRemainder interface env offset offset).val index = _
  rw [quotientBefore]
  simp [fieldOfNat,
    NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat,
    Nat.mod_eq_of_lt quotientBound]

private theorem foldl_congr_mem
    {α β : Type} (items : List β) (left right : α → β → α)
    (initial : α)
    (equalStep : ∀ accumulator item, item ∈ items →
      left accumulator item = right accumulator item) :
    items.foldl left initial = items.foldl right initial := by
  induction items generalizing initial with
  | nil => rfl
  | cons item rest inductionHypothesis =>
      rw [List.foldl_cons, List.foldl_cons,
        equalStep initial item (by simp)]
      apply inductionHypothesis
      intro accumulator current member
      exact equalStep accumulator current (by simp [member])

private theorem completeEnv_quotientValue
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    quotientValue (completeEnv interface env offset) offset =
      ((interface.candidate offset).eval env).val / 5 := by
  let quotient := ((interface.candidate offset).eval env).val / 5
  have quotientLt : quotient < 2 ^ quotientBitCount := by
    have sourceLt : ((interface.candidate offset).eval env).val < 2 ^ 16 := by
      have weightedBound := weightedValue_lt_twoPow env
        (interface.candidateBit offset) candidateBitCount assumptions.2.2.2
      rw [assumptions.2.2.1]
      simpa [candidateBitCount] using weightedBound
    simp only [quotient, quotientBitCount]
    omega
  have pointwise : ∀ index ∈ List.range quotientBitCount,
      bitValue (completeEnv interface env offset) (quotientBitExpr offset) index =
        bitNat quotient index := by
    intro index member
    exact completeEnv_quotientBitValue interface env offset index assumptions
      (List.mem_range.mp member)
  unfold quotientValue weightedValue
  have equalFolds :
      (List.range quotientBitCount).foldl
          (fun value index => value + 2 ^ index *
            bitValue (completeEnv interface env offset)
              (quotientBitExpr offset) index) 0 =
        (List.range quotientBitCount).foldl
          (fun value index => value + 2 ^ index * bitNat quotient index) 0 := by
    apply foldl_congr_mem
    intro value index member
    rw [pointwise index member]
  rw [equalFolds]
  change bitWindowValue quotient quotientBitCount = quotient
  rw [bitWindowValue_eq_mod, Nat.mod_eq_of_lt quotientLt]

private theorem completeEnv_agreesOutside
    (interface : Interface) (env : Env) (offset : Nat) :
    AgreesOutside env (completeEnv interface env offset) offset auxiliaryCount := by
  have qr := executeHints_agreesOutside env offset
    (quotientRemainderHints interface offset)
  have bits := executeHints_agreesOutside
    (completeQuotientRemainder interface env offset) (offset + 2)
    (quotientBitHints offset)
  have reject := executeRecipes_agreesOutside
    (completeQuotientBits interface env offset) (offset + 16)
    [rejectRecipe interface offset]
  have all := (qr.append bits).append reject
  simpa [completeEnv, completeQuotientBits, completeQuotientRemainder,
    quotientRemainderHints, quotientBitHints, quotientBitCount,
    auxiliaryCount] using all

private theorem flatConstraints_quotientBooleanOps (offset : Nat) :
    flatConstraints (quotientBooleanOps offset) =
      (List.range quotientBitCount).map
        (quotientBooleanConstraint offset) := by
  unfold flatConstraints quotientBooleanOps
  generalize List.range quotientBitCount = indices
  induction indices with
  | nil => rfl
  | cons index rest inductionHypothesis =>
      simp [Op.flatConstraints, inductionHypothesis]

theorem flatConstraints_operations (interface : Interface) (offset : Nat) :
    flatConstraints (operations interface offset) =
      recipeConstraints (offset + 16) [rejectRecipe interface offset] ++
        ((List.range quotientBitCount).map
            (quotientBooleanConstraint offset) ++
          [quotientRecompositionConstraint offset,
            divisionConstraint interface offset,
            remainderConstraint offset]) := by
  change recipeConstraints offset [] ++
      (recipeConstraints (offset + 2) [] ++
        (recipeConstraints (offset + 16) [rejectRecipe interface offset] ++
          (flatConstraints (quotientBooleanOps offset) ++
            [quotientRecompositionConstraint offset,
              divisionConstraint interface offset,
              remainderConstraint offset]))) = _
  rw [flatConstraints_quotientBooleanOps]
  rfl

private theorem productExpr_varsBelow (bit : Nat → Expr) (count bound : Nat)
    (bitsBelow : ∀ index, index < count → (bit index).VarsBelow bound) :
    (productExpr bit count).VarsBelow bound := by
  induction count with
  | zero => trivial
  | succ count inductionHypothesis =>
      exact ⟨inductionHypothesis
          (fun index bounded => bitsBelow index (by omega)),
        bitsBelow count (by omega)⟩

private theorem weightedExpr_varsBelow (bit : Nat → Expr) (count bound : Nat)
    (bitsBelow : ∀ index, index < count → (bit index).VarsBelow bound) :
    (weightedExpr bit count).VarsBelow bound := by
  induction count with
  | zero => trivial
  | succ count inductionHypothesis =>
      rw [weightedExpr_succ]
      exact ⟨inductionHypothesis
          (fun index bounded => bitsBelow index (by omega)),
        ⟨trivial, bitsBelow count (by omega)⟩⟩

/-- Every decoder constraint reads only caller inputs and the exact logical
decoder interval. -/
theorem flatConstraints_varsBelow
    (interface : Interface) (offset : Nat)
    (candidateBelow : (interface.candidate offset).VarsBelow offset)
    (bitsBelow : ∀ index, index < candidateBitCount →
      (interface.candidateBit offset index).VarsBelow offset) :
    ∀ expression ∈ flatConstraints (operations interface offset),
      expression.VarsBelow (offset + auxiliaryCount) := by
  intro expression member
  rw [flatConstraints_operations] at member
  rcases List.mem_append.mp member with recipeMember | assertionMember
  · simp only [recipeConstraints, List.mem_singleton] at recipeMember
    subst expression
    apply Expr.VarsBelow.sub
    · simp [Expr.VarsBelow, auxiliaryCount]
    · apply productExpr_varsBelow
      intro index bounded
      exact Expr.VarsBelow.mono _ (bitsBelow index bounded) (by omega)
  · rcases List.mem_append.mp assertionMember with booleanMember | finalMember
    · rcases List.mem_map.mp booleanMember with ⟨index, indexMember, rfl⟩
      have bounded := List.mem_range.mp indexMember
      simp only [quotientBitCount] at bounded
      unfold quotientBooleanConstraint
      apply Expr.VarsBelow.mul
      · simp [quotientBitExpr, Expr.VarsBelow, auxiliaryCount]
        omega
      · apply Expr.VarsBelow.sub
        · simp [quotientBitExpr, Expr.VarsBelow, auxiliaryCount]
          omega
        · trivial
    · simp only [List.mem_cons, List.not_mem_nil, or_false] at finalMember
      rcases finalMember with rfl | rfl | rfl
      · apply Expr.VarsBelow.sub
        · simp [quotientExpr, Expr.VarsBelow, auxiliaryCount]
        · apply weightedExpr_varsBelow
          intro index bounded
          simp only [quotientBitCount] at bounded
          simp [quotientBitExpr, Expr.VarsBelow, auxiliaryCount]
          omega
      · apply Expr.VarsBelow.sub
        · exact Expr.VarsBelow.mono _ candidateBelow (by omega)
        · exact ⟨⟨trivial, by
              simp [quotientExpr, Expr.VarsBelow, auxiliaryCount]⟩,
            by
              simp [remainderExpr, Expr.VarsBelow, auxiliaryCount]⟩
      · have remainderBelow : (remainderExpr offset).VarsBelow
            (offset + auxiliaryCount) := by
          simp [remainderExpr, Expr.VarsBelow, auxiliaryCount]
        have differenceBelow (value : Nat) :
            (remainderExpr offset - (OfNat.ofNat value : Expr)).VarsBelow
              (offset + auxiliaryCount) :=
          Expr.VarsBelow.sub _ _ _ remainderBelow trivial
        exact ⟨⟨⟨⟨remainderBelow, differenceBelow 1⟩,
          differenceBelow 2⟩, differenceBelow 3⟩, differenceBelow 4⟩

private theorem completeEnv_holdsFlat
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    holdsFlat (completeEnv interface env offset) (operations interface offset) := by
  let completed := completeEnv interface env offset
  let sourceValue := ((interface.candidate offset).eval env).val
  let quotient := sourceValue / 5
  let remainder := sourceValue % 5
  have sourceEq := completeEnv_source interface env offset assumptions
  have quotientEq := completeEnv_quotient interface env offset assumptions
  have remainderEq := completeEnv_remainder interface env offset assumptions
  have quotientValueEq := completeEnv_quotientValue interface env offset assumptions
  have quotientEvalEq : (quotientExpr offset).eval completed =
      fieldOfNat quotient := by
    change completeEnv interface env offset offset = fieldOfNat quotient
    simpa [quotient, sourceValue] using quotientEq
  have remainderEvalEq : (remainderExpr offset).eval completed =
      fieldOfNat remainder := by
    change completeEnv interface env offset (offset + 1) = fieldOfNat remainder
    simpa [remainder, sourceValue] using remainderEq
  have quotientLtModulus : quotient < goldilocksModulus := by
    exact lt_of_le_of_lt (Nat.div_le_self sourceValue 5)
      ((interface.candidate offset).eval env).isLt
  have remainderLt : remainder < 5 := Nat.mod_lt _ (by decide)
  have binary : ∀ index, index < quotientBitCount →
      (quotientBooleanConstraint offset index).eval completed = 0 := by
    intro index bounded
    have bitEq := completeEnv_quotientBitValue interface env offset index
      assumptions bounded
    have bitLt : bitNat quotient index < 2 := by
      unfold bitNat
      exact Nat.mod_lt _ (by decide)
    have bitZeroOrOne : bitNat quotient index = 0 ∨ bitNat quotient index = 1 := by
      omega
    rcases bitZeroOrOne with bitZero | bitOne
    · have fieldZero : completed (offset + 2 + index) = 0 := by
        apply Fin.eq_of_val_eq
        change (completed (offset + 2 + index)).val = 0
        change (completed (offset + 2 + index)).val =
          bitNat quotient index at bitEq
        rw [bitZero] at bitEq
        exact bitEq
      rw [quotientBooleanConstraint_eval, fieldZero, zero_mul]
    · have fieldOne : completed (offset + 2 + index) = 1 := by
        apply Fin.eq_of_val_eq
        change (completed (offset + 2 + index)).val = (1 : F).val
        change (completed (offset + 2 + index)).val =
          bitNat quotient index at bitEq
        rw [bitOne] at bitEq
        simpa [goldilocksModulus] using bitEq
      rw [quotientBooleanConstraint_eval, fieldOne, sub_self, mul_zero]
  have quotientRecomposition :
      (quotientRecompositionConstraint offset).eval completed = 0 := by
    have wordEval := weightedExpr_eval completed (quotientBitExpr offset)
      quotientBitCount
    have equality : (quotientExpr offset).eval completed =
        (quotientWordExpr offset).eval completed := by
      calc
      (quotientExpr offset).eval completed = fieldOfNat quotient := quotientEvalEq
      _ = fieldOfNat (quotientValue completed offset) := by
        rw [quotientValueEq]
      _ = (quotientWordExpr offset).eval completed := wordEval.symm
    simpa only [quotientRecompositionConstraint, Expr.eval_sub] using
      sub_eq_zero.mpr equality
  have division : (divisionConstraint interface offset).eval completed = 0 := by
    have equality : (interface.candidate offset).eval completed =
        fieldOfNat 5 * (quotientExpr offset).eval completed +
          (remainderExpr offset).eval completed := by
      rw [sourceEq, quotientEvalEq, remainderEvalEq,
        fieldOfNat_mul, fieldOfNat_add]
      apply Fin.eq_of_val_eq
      have reordered : 5 * quotient + remainder = sourceValue := by
        have split := Nat.mod_add_div sourceValue 5
        omega
      have rhsBound : 5 * quotient + remainder < goldilocksModulus := by
        rw [reordered]
        exact ((interface.candidate offset).eval env).isLt
      simpa [fieldOfNat,
        NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat,
        Nat.mod_eq_of_lt rhsBound] using reordered.symm
    simpa only [divisionConstraint, Expr.eval_sub] using
      sub_eq_zero.mpr equality
  have remainderRoot : (remainderConstraint offset).eval completed = 0 := by
    simp only [remainderConstraint, Expr.eval_hmul, Expr.eval_sub]
    rw [remainderEvalEq]
    have cases : remainder = 0 ∨ remainder = 1 ∨ remainder = 2 ∨
        remainder = 3 ∨ remainder = 4 := by omega
    rcases cases with root | root | root | root | root <;>
      rw [root] <;>
      norm_num [Expr.eval, fieldOfNat,
        NightstreamFPrime.Gadgets.Range.CanonicalU64.fieldOfNat,
        goldilocksModulus]
  unfold holdsFlat
  rw [flatConstraints_operations, constraintsHold_append]
  constructor
  · exact executeRecipes_holds_recipeConstraints
      (completeQuotientBits interface env offset) (offset + 16)
      [rejectRecipe interface offset] (by
        constructor
        · unfold rejectRecipe
          apply productExpr_varsBelow
          intro index bounded
          exact Expr.VarsBelow.mono _
            (assumptions.2.1 index bounded) (by omega)
        · trivial)
  · rw [constraintsHold_append]
    constructor
    · intro expression member
      rcases List.mem_map.mp member with ⟨index, indexMember, rfl⟩
      exact binary index (List.mem_range.mp indexMember)
    · intro expression member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with rfl | rfl | rfl
      · exact quotientRecomposition
      · exact division
      · exact remainderRoot

theorem complete
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) := by
  refine ⟨completeEnv interface env offset, ?_,
    completeEnv_holdsFlat interface env offset assumptions⟩
  have agrees := completeEnv_agreesOutside interface env offset
  rw [localLength_eq]
  exact agrees

theorem completeness
    (interface : Interface) (env : Env) (offset : Nat)
    (assumptions : Assumptions interface offset env)
    (_specification : SpecHolds interface offset env) :
    ∃ completed,
      AgreesOutside env completed offset
        (localLength (operations interface offset)) ∧
      holdsFlat completed (operations interface offset) :=
  complete interface env offset assumptions

def circuit (interface : Interface) : FormalCircuit :=
  { main := main interface
    assumptions := Assumptions interface
    spec := SpecHolds interface
    privateCount := fun _ => auxiliaryCount
    rowCount := fun _ => exactRowCount
    privateCount_eq := by
      intro offset
      exact localLength_eq interface offset
    rowCount_eq := by
      intro offset
      exact flatConstraints_length_eq interface offset
    soundness := by
      intro env offset assumptions rows
      exact soundness interface env offset assumptions rows
    completeness := by
      intro env offset assumptions specification
      exact completeness interface env offset assumptions specification }

end NightstreamFPrime.Gadgets.Sampling.Candidate16Five
