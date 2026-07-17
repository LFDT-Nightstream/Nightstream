import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanDomain
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanEvaluation

/-!
Shared finite linear algebra for paper-level joint `Pi_CCS` residual families.

Protocol: SuperNeo `Pi_CCS` (Section 7.3 / Appendix D.4).
Phase: explicit matrix-image construction before residual-family formulas.
Constraint family: shared arithmetic used by CCS and carried-evaluation leaves.

Owns: typed assignment columns, Boolean-row matrices, the canonical finite
dot product `(M z)(x)`, and exact reduction of a one-coordinate assignment to
its selected matrix contribution.

Does not own: a constraint polynomial, norm residual, carried target,
base-to-extension embedding, external row/bit serialization, SumCheck,
Fiat--Shamir, Rust, R1CS, or constraint counts.

Emits constraints: no.

Authority boundary: matrix entries and assignment columns are explicit data;
the matrix image is derived by a canonical increasing `Fin` traversal. Both
residual families consume this single owner so they cannot silently disagree
on column order or dot-product arithmetic.

| Shared owner | Mathematical object | Exact construction |
|---|---|---|
| `Assignment` | one finite `z` | `Fin columns -> Field` |
| `BooleanMatrix` | one `M_j` over the Boolean row domain | `BooleanVertex -> Fin columns -> Field` |
| `matrixVectorAt` | `(M_j z)(x)` | canonical-column fold of `M[x,c] * z[c]` |
| `matrixVectorAt_zero` | canonical zero assignment | every finite matrix-vector row is zero |
| `matrixVectorAt_oneHot` | one selected assignment coordinate | exact selected matrix entry times its value |
-/

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

universe uField

open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- An assignment with exactly the paper structure's declared column count. -/
abbrev Assignment (Field : Type uField) (columns : Nat) := Fin columns -> Field

/-- One `2^ell`-row matrix indexed by the shared semantic Boolean domain. -/
abbrev BooleanMatrix
    (Field : Type uField)
    (variables columns : Nat) :=
  BooleanVertex variables -> Fin columns -> Field

/-- One matrix-vector row `(M z)(x)`, derived as a finite dot product in
canonical assignment-column order. -/
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

/-- Every matrix sends the canonical zero assignment to zero. This theorem
owns the finite-column fold once, so source-semantics proofs do not need to
unfold a profile-specific carrier width. -/
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

/-- Assignment supported at exactly one selected coordinate. -/
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
      simp only [List.foldl_cons, if_neg headNe]
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
        rw [if_pos rfl, laws.zero_add]
        exact foldl_absent_oneHot ops laws tail selected term
          (List.nodup_cons.mp nodup).1
          (term selected)
      · have memberTail : selected ∈ tail := by
          simpa [Ne.symm headEq] using member
        rw [if_neg headEq, laws.add_zero]
        exact inductionHypothesis (List.nodup_cons.mp nodup).2 memberTail

/-- A one-coordinate assignment contributes exactly its selected matrix
entry. This structural theorem avoids reducing an entire padded carrier when
auditing whether one completed coordinate is semantically live. -/
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

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
