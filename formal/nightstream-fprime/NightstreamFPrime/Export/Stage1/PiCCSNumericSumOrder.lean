import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.NumericCompletionSum
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Coefficients

/-! Scalar ordering identities for the existing numeric sum. The proofs
preserve ascending order and introduce no selected shape or new evaluator. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNumericSumOrder

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NumericCompletionSum (numericSum)

universe uField uIndex
variable {Field : Type uField}

private theorem numericSum_succ (ops : InterpolationOps Field)
    (count : Nat) (term : Nat → Field) :
    numericSum ops (count + 1) term = ops.add (numericSum ops count term) (term count) := by
  simp only [numericSum, Nat.fold_succ]

private theorem foldl_eq_add_sumMap {Index : Type uIndex}
    (ops : InterpolationOps Field) (laws : InterpolationEvaluationLaws ops)
    (indices : List Index) (term : Index → Field) (initial : Field) :
    indices.foldl (fun total index => ops.add total (term index)) initial =
      ops.add initial (FiniteSumAlgebra.sumMap ops indices term) := by
  induction indices generalizing initial with
  | nil => exact (laws.add_zero initial).symm
  | cons index rest ih =>
      rw [List.foldl_cons, ih]
      exact laws.add_assoc initial (term index) _

/-- The numeric prefix and canonical finite-index sum have exactly the same
terms and order. This includes the empty prefix. -/
theorem numericSum_eq_finSum (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (count : Nat) (term : Nat → Field) :
    numericSum ops count term =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices count) (fun index => term index.val) := by
  unfold numericSum
  rw [Nat.fold_eq_finRange_foldl]
  change (canonicalFinIndices count).foldl
      (fun total index => ops.add total (term index.val)) ops.zero = _
  rw [foldl_eq_add_sumMap ops laws, laws.zero_add]

private theorem numericSum_append (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (left right : Nat) (term : Nat → Field) :
    numericSum ops (left + right) term =
      ops.add (numericSum ops left term)
        (numericSum ops right (fun offset => term (left + offset))) := by
  induction right with
  | zero =>
      change numericSum ops left term = ops.add (numericSum ops left term) ops.zero
      exact (laws.add_zero _).symm
  | succ right ih =>
      rw [Nat.add_succ, numericSum_succ, ih, numericSum_succ]
      exact laws.add_assoc _ _ _

/-- Flatten complete blocks without changing their block-major, lane-major
order. Width zero and count zero are both included. -/
theorem numericSum_group (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (count width : Nat) (term : Nat → Field) :
    numericSum ops (count * width) term =
      numericSum ops count (fun block =>
        numericSum ops width (fun lane => term (block * width + lane))) := by
  induction count with
  | zero => rw [Nat.zero_mul]; rfl
  | succ count ih =>
      rw [Nat.succ_mul, numericSum_append ops laws, numericSum_succ, ih]

/-- Restrict a complete even row domain to exactly one parity. The chosen
bit is zero or one by its existing Fin type; terms need no value premise. -/
theorem numericSum_parity (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (count : Nat) (bit : Fin 2) (term : Nat → Field) :
    numericSum ops (2 * count) (fun row => if row % 2 = bit.val then term row else ops.zero) =
      numericSum ops count (fun pair => term (2 * pair + bit.val)) := by
  induction count with
  | zero => rfl
  | succ count ih =>
      have doubleSucc : 2 * (count + 1) = (2 * count + 1) + 1 := by omega
      have even : (2 * count) % 2 = 0 := by omega
      have odd : (2 * count + 1) % 2 = 1 := by omega
      rw [doubleSucc, numericSum_succ, numericSum_succ, ih, numericSum_succ]
      have bitCases : bit.val = 0 ∨ bit.val = 1 := by
        have bound := bit.isLt
        omega
      rcases bitCases with low | high
      · simp [low, even, odd, laws.add_zero, laws.zero_add]
      · simp [high, even, odd, laws.add_zero, laws.zero_add]

/-- A parity-restricted active prefix is the complete endpoint sum when
all omitted in-domain rows are zero. The last odd prefix row is retained. -/
theorem numericSum_prefix_parity_eq_finSum (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (remaining activeCount : Nat) (bit : Fin 2)
    (term : Nat → Field) (fits : activeCount ≤ 2 ^ (remaining + 1))
    (outsideZero : ∀ index, activeCount ≤ index → index < 2 ^ (remaining + 1) → term index = ops.zero) :
    numericSum ops activeCount (fun row => if row % 2 = bit.val then term row else ops.zero) =
      FiniteSumAlgebra.sumMap ops (canonicalFinIndices (2 ^ remaining))
        (fun pair => term (2 * pair.val + bit.val)) := by
  let restricted : Nat → Field := fun row => if row % 2 = bit.val then term row else ops.zero
  have prefixValue := NumericCompletionSum.numericSum_prefix_eq_vertexSum ops laws
    (remaining + 1) activeCount restricted fits (by
      intro index lower upper
      simp only [restricted, outsideZero index lower upper, ite_self])
  have fullValue := NumericCompletionSum.numericSum_eq_vertexSum ops laws (remaining + 1) restricted
  have fullCount : 2 ^ (remaining + 1) = 2 * 2 ^ remaining := by
    rw [Nat.pow_succ, Nat.mul_comm]
  calc
    _ = numericSum ops (2 ^ (remaining + 1)) restricted := prefixValue.trans fullValue.symm
    _ = numericSum ops (2 ^ remaining) (fun pair => term (2 * pair + bit.val)) := by
      rw [fullCount]
      exact numericSum_parity ops laws (2 ^ remaining) bit term
    _ = _ := numericSum_eq_finSum ops laws _ _
end NightstreamFPrime.Export.Stage1.PiCCSNumericSumOrder
