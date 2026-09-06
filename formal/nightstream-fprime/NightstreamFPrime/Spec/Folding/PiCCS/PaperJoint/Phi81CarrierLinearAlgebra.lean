import Mathlib.Data.List.OfFn
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout

/-!
Owns preservation of the scalar CCS matrix image under canonical Phi81
carrier completion. SuperNeo v1.1 Definition 19 uses the original scalar
matrix image; packing adds zero columns and does not change that image.
The proof splits symbolic finite sums, without evaluating a carrier.
-/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout

open PaperLinearAlgebra

universe uField

private theorem foldl_zeros
    {Field : Type uField} (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (count : Nat) (initial : Field) :
    (List.replicate count ops.zero).foldl ops.add initial = initial := by
  induction count generalizing initial with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [List.replicate_succ, List.foldl_cons, laws.add_zero]
      exact inductionHypothesis initial

private theorem foldl_ofFn
    {Field : Type uField} (ops : InterpolationOps Field)
    {count : Nat} (term : Fin count → Field) :
    (List.ofFn term).foldl ops.add ops.zero =
      (canonicalFinIndices count).foldl
        (fun total index => ops.add total (term index)) ops.zero := by
  calc
    (List.ofFn term).foldl ops.add ops.zero =
        ((List.ofFn (id : Fin count → Fin count)).map term).foldl
          ops.add ops.zero := by
      rw [List.map_ofFn]
      rfl
    _ = _ := by
      rw [List.foldl_map]
      rfl

private theorem foldl_ofFn_prefix
    {Field : Type uField} (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {prefixWidth fullWidth : Nat} (covered : prefixWidth ≤ fullWidth)
    (term : Fin fullWidth → Field) (prefixTerm : Fin prefixWidth → Field)
    (same : ∀ column, term (column.castLE covered) = prefixTerm column)
    (zeroTail : ∀ column, prefixWidth ≤ column.val → term column = ops.zero) :
    (List.ofFn term).foldl ops.add ops.zero =
      (List.ofFn prefixTerm).foldl ops.add ops.zero := by
  obtain ⟨tailWidth, rfl⟩ := Nat.exists_eq_add_of_le covered
  rw [List.ofFn_add, List.foldl_append]
  have prefixEq :
      (List.ofFn fun column : Fin prefixWidth =>
        term (column.castLE (Nat.le_add_right prefixWidth tailWidth))) =
      List.ofFn prefixTerm := by
    apply congrArg List.ofFn
    funext column
    exact same column
  have tailEq :
      (List.ofFn fun column : Fin tailWidth => term (column.natAdd prefixWidth)) =
      List.replicate tailWidth ops.zero := by
    rw [← List.ofFn_const]
    apply congrArg List.ofFn
    funext column
    apply zeroTail
    change prefixWidth ≤ prefixWidth + column.val
    omega
  rw [prefixEq, tailEq]
  exact foldl_zeros ops laws tailWidth _

/-- Canonical zero completion preserves every original scalar matrix image.
The finite column traversal retains its order and its initial zero. -/
theorem matrixVectorAt_extend
    {Field : Type uField} (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {cubeSize logicalWidth : Nat}
    (matrix : BooleanMatrix Field cubeSize logicalWidth)
    (assignment : Assignment Field logicalWidth)
    (vertex : BooleanVertex cubeSize) :
    matrixVectorAt ops (extendMatrix ops.zero matrix)
        (extendAssignment ops.zero assignment) vertex =
      matrixVectorAt ops matrix assignment vertex := by
  have sums := foldl_ofFn_prefix ops laws
    (logicalWidth_le_carrierWidth logicalWidth)
    (fun column => ops.mul (extendMatrix ops.zero matrix vertex column)
      (extendAssignment ops.zero assignment column))
    (fun column => ops.mul (matrix vertex column) (assignment column))
    (by
      intro column
      change ops.mul (extendMatrix ops.zero matrix vertex (embedLogical column))
          (extendAssignment ops.zero assignment (embedLogical column)) = _
      rw [extendMatrix_embedLogical, extendAssignment_embedLogical])
    (by
      intro column tail
      dsimp only
      rw [extendAssignment_tail_zero ops.zero assignment column tail,
        laws.mul_zero])
  unfold matrixVectorAt
  simpa only [foldl_ofFn] using sums

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.Phi81CarrierLayout
