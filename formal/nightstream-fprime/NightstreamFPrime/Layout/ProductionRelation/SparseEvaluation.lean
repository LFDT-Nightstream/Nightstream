import NightstreamFPrime.Layout.ProductionRelation

/-!
Owns evaluation over the entries stored in a production sparse form. Its
equality with the canonical column sum includes repeated columns and
cancelling coefficients, without a distinctness premise.
-/

namespace NightstreamFPrime.Layout.ProductionRelation.SparseForm

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra

/-- Execute only stored entries; no logical-column enumeration is built. -/
def evalSparse {columns : Nat} (form : SparseForm columns)
    (assignment : Assignment F columns) : F :=
  form.entries.foldl (fun total entry =>
    total + entry.coefficient * assignment entry.column) 0

private theorem foldl_entries_from {columns : Nat}
    (entries : List (SparseEntry columns))
    (assignment : Assignment F columns) (initial : F) :
    entries.foldl (fun total entry =>
        total + entry.coefficient * assignment entry.column) initial =
      initial + entries.foldl (fun total entry =>
        total + entry.coefficient * assignment entry.column) 0 := by
  induction entries generalizing initial with
  | nil => simp
  | cons entry entries inductionHypothesis =>
      simp only [List.foldl_cons]
      rw [inductionHypothesis (initial + entry.coefficient * assignment entry.column),
        inductionHypothesis (0 + entry.coefficient * assignment entry.column)]
      simp only [zero_add]
      exact baseLaws.add_assoc _ _ _

/-- The stored-entry traversal is exactly the existing matrix-row meaning.
Every duplicate entry contributes to the same canonical column coefficient. -/
theorem evalSparse_eq_eval {columns : Nat}
    (form : SparseForm columns) (assignment : Assignment F columns) :
    form.evalSparse assignment = form.eval assignment := by
  rcases form with ⟨entries⟩
  induction entries with
  | nil => exact (empty_eval assignment).symm
  | cons entry entries inductionHypothesis =>
      change entries.foldl (fun total current =>
          total + current.coefficient * assignment current.column)
          (0 + entry.coefficient * assignment entry.column) =
        (add (singleton entry.column entry.coefficient) ⟨entries⟩).eval assignment
      rw [add_eval, singleton_eval, foldl_entries_from]
      simpa only [zero_add, evalSparse] using
        congrArg (fun value => entry.coefficient * assignment entry.column + value)
          inductionHypothesis

end NightstreamFPrime.Layout.ProductionRelation.SparseForm
