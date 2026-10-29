import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanDomain
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.BooleanEvaluation

/-! SuperNeo v1.2 Section 7.3, Appendix B.2. -/

namespace NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

universe uField

open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

abbrev Assignment (Field : Type uField) (columns : Nat) := Fin columns -> Field

abbrev BooleanMatrix
    (Field : Type uField)
    (variables columns : Nat) :=
  BooleanVertex variables -> Fin columns -> Field

def matrixVectorAt
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {variables columns : Nat}
    (matrix : BooleanMatrix Field variables columns)
    (assignment : Assignment Field columns)
    (vertex : BooleanVertex variables) : Field :=
  (canonicalFinIndices columns).foldl
    (fun accumulated column =>
      ops.add accumulated
        (ops.mul (matrix vertex column) (assignment column)))
    ops.zero

theorem matrixVectorAt_zero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables columns : Nat}
    (matrix : BooleanMatrix Field variables columns)
    (vertex : BooleanVertex variables) :
    matrixVectorAt ops matrix (fun _ => ops.zero) vertex = ops.zero := by
  unfold matrixVectorAt
  generalize canonicalFinIndices columns = indices
  induction indices with
  | nil => rfl
  | cons _ indices inductionHypothesis =>
      rw [List.foldl_cons, laws.mul_zero, laws.add_zero]
      exact inductionHypothesis

theorem matrixVectorAt_zeroRow
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables columns : Nat}
    (matrix : BooleanMatrix Field variables columns)
    (assignment : Assignment Field columns)
    (vertex : BooleanVertex variables)
    (zeroRow : forall column, matrix vertex column = ops.zero) :
    matrixVectorAt ops matrix assignment vertex = ops.zero := by
  unfold matrixVectorAt
  generalize canonicalFinIndices columns = indices
  induction indices with
  | nil => rfl
  | cons column indices inductionHypothesis =>
      rw [List.foldl_cons, zeroRow column,
        laws.mul_comm ops.zero (assignment column), laws.mul_zero,
        laws.add_zero]
      exact inductionHypothesis

def oneHotAssignment
    {Field : Type uField}
    (ops : InterpolationOps Field)
    {columns : Nat}
    (selected : Fin columns)
    (value : Field) : Assignment Field columns :=
  fun column => if column = selected then value else ops.zero

private theorem foldl_absent_oneHot
    {Index : Type}
    [DecidableEq Index]
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index)
    (selected : Index)
    (term : Index -> Field)
    (absent : selected ∉ indices)
    (initial : Field) :
    indices.foldl (fun accumulated index =>
        ops.add accumulated
          (if index = selected then term index else ops.zero)) initial =
      initial := by
  induction indices generalizing initial with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      have headNe : head ≠ selected := by
        intro equal
        apply absent
        simp [equal]
      have absentTail : selected ∉ tail := by
        intro member
        exact absent (by simp [member])
      simp only [List.foldl_cons, ite_eq_right headNe]
      rw [laws.add_zero]
      exact inductionHypothesis absentTail initial

private theorem foldl_oneHot
    {Index : Type}
    [DecidableEq Index]
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (indices : List Index)
    (selected : Index)
    (term : Index -> Field)
    (nodup : indices.Nodup)
    (member : selected ∈ indices) :
    indices.foldl (fun accumulated index =>
        ops.add accumulated
          (if index = selected then term index else ops.zero)) ops.zero =
      term selected := by
  induction indices with
  | nil => simp at member
  | cons head tail inductionHypothesis =>
      simp only [List.foldl_cons]
      by_cases headEq : head = selected
      · subst head
        rw [ite_eq_left rfl, laws.zero_add]
        exact foldl_absent_oneHot ops laws tail selected term
          (List.nodup_cons.mp nodup).1
          (term selected)
      · have memberTail : selected ∈ tail := by
          simpa [Ne.symm headEq] using member
        rw [ite_eq_right headEq, laws.add_zero]
        exact inductionHypothesis (List.nodup_cons.mp nodup).2 memberTail

theorem matrixVectorAt_oneHot
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables columns : Nat}
    (matrix : BooleanMatrix Field variables columns)
    (selected : Fin columns)
    (value : Field)
    (vertex : BooleanVertex variables) :
    matrixVectorAt ops matrix (oneHotAssignment ops selected value) vertex =
      ops.mul (matrix vertex selected) value := by
  unfold matrixVectorAt
  have contributionFunction :
      (fun accumulated column =>
        ops.add accumulated
          (ops.mul (matrix vertex column)
            (oneHotAssignment ops selected value column))) =
      (fun accumulated column =>
        ops.add accumulated
          (if column = selected then
            ops.mul (matrix vertex column) value
          else
            ops.zero)) := by
    funext accumulated column
    by_cases equal : column = selected
    · simp [oneHotAssignment, equal]
    · simp [oneHotAssignment, equal, laws.mul_zero]
  rw [contributionFunction]
  exact foldl_oneHot ops laws (canonicalFinIndices columns) selected
    (fun column => ops.mul (matrix vertex column) value)
    (canonicalFinIndices_nodup columns)
    (by simp [canonicalFinIndices])

theorem matrixVectorAt_identityRow
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {variables columns : Nat}
    (matrix : BooleanMatrix Field variables columns)
    (assignment : Assignment Field columns)
    (vertex : BooleanVertex variables)
    (selected : Fin columns)
    (identityRow : forall column,
      matrix vertex column =
        if column = selected then ops.one else ops.zero) :
    matrixVectorAt ops matrix assignment vertex = assignment selected := by
  unfold matrixVectorAt
  have contributionFunction :
      (fun accumulated column =>
        ops.add accumulated
          (ops.mul (matrix vertex column) (assignment column))) =
      (fun accumulated column =>
        ops.add accumulated
          (if column = selected then assignment column else ops.zero)) := by
    funext accumulated column
    rw [identityRow column]
    by_cases equal : column = selected
    · rw [ite_eq_left equal, ite_eq_left equal, laws.one_mul]
    · rw [ite_eq_right equal, ite_eq_right equal,
        laws.mul_comm ops.zero (assignment column), laws.mul_zero]
  rw [contributionFunction]
  exact foldl_oneHot ops laws (canonicalFinIndices columns) selected assignment
    (canonicalFinIndices_nodup columns)
    (by simp [canonicalFinIndices])

end NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
