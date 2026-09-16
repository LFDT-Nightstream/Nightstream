import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra

/-! Evaluate an existing sparse form on extension-field reads through its
real and imaginary scalar evaluations. Weighted source aggregation preserves
every stored entry, including repeated columns. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSSparseEvaluation

universe uIndex

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.FiniteSumAlgebra
open NightstreamFPrime.Layout.ProductionRelation

/-- Use the existing scalar evaluator for both extension coordinates. -/
def evaluateK {columns : Nat} (form : SparseForm columns)
    (read : Fin columns → K) : K :=
  ⟨form.evalSparse (fun column => (read column).c0),
   form.evalSparse (fun column => (read column).c1)⟩

private theorem mul_embed_left (coefficient : F) (value : K) :
    extensionOps.mul (K.embed coefficient) value =
      ⟨coefficient * value.c0, coefficient * value.c1⟩ := by
  change K.mul (K.embed coefficient) value = _
  simp only [K.mul, K.embed, Fin.mul_zero, Fin.zero_mul, Fin.add_zero]

private theorem pair_fold_value {columns : Nat}
    (entries : List (SparseEntry columns)) (read : Fin columns → K)
    (initial : K) :
    (⟨entries.foldl (fun total entry =>
        total + entry.coefficient * (read entry.column).c0) initial.c0,
      entries.foldl (fun total entry =>
        total + entry.coefficient * (read entry.column).c1) initial.c1⟩ : K) =
      extensionOps.add initial
        (sumMap extensionOps entries (fun entry =>
          extensionOps.mul (K.embed entry.coefficient) (read entry.column))) := by
  induction entries generalizing initial with
  | nil => exact (extensionLaws.add_zero initial).symm
  | cons entry rest inductionHypothesis =>
      simp only [List.foldl_cons]
      calc
        _ = extensionOps.add
            ⟨initial.c0 + entry.coefficient * (read entry.column).c0,
             initial.c1 + entry.coefficient * (read entry.column).c1⟩
            (sumMap extensionOps rest (fun current =>
              extensionOps.mul (K.embed current.coefficient) (read current.column))) :=
          inductionHypothesis _
        _ = extensionOps.add
            (extensionOps.add initial
              (extensionOps.mul (K.embed entry.coefficient) (read entry.column)))
            (sumMap extensionOps rest (fun current =>
              extensionOps.mul (K.embed current.coefficient) (read current.column))) := by
          rw [mul_embed_left]
          rfl
        _ = _ := extensionLaws.add_assoc _ _ _

/-- Exact extension-field sum over all stored sparse entries. No distinctness
or coefficient restriction is needed. -/
theorem evaluateK_value {columns : Nat} (form : SparseForm columns)
    (read : Fin columns → K) :
    evaluateK form read =
      sumMap extensionOps form.entries (fun entry =>
        extensionOps.mul (K.embed entry.coefficient) (read entry.column)) := by
  calc
    evaluateK form read = extensionOps.add extensionOps.zero
        (sumMap extensionOps form.entries (fun entry =>
          extensionOps.mul (K.embed entry.coefficient) (read entry.column))) :=
      pair_fold_value form.entries read extensionOps.zero
    _ = _ := extensionLaws.zero_add _

private theorem evaluateK_embed {columns : Nat} (form : SparseForm columns)
    (read : Fin columns → F) :
    evaluateK form (fun column => K.embed (read column)) =
      K.embed (form.evalSparse read) := by
  simp only [evaluateK, K.embed, SparseForm.evalSparse, Fin.mul_zero,
    Fin.add_zero, List.foldl_fixed]

/-- Weighted extension-field reads commute with the same sparse evaluator.
The input list and the stored entries can both contain repetitions. -/
theorem weighted_readsK {Index : Type uIndex} {columns : Nat}
    (form : SparseForm columns) (indices : List Index) (weight : Index → K)
    (read : Index → Fin columns → K) :
    sumMap extensionOps indices (fun index =>
        extensionOps.mul (weight index) (evaluateK form (read index))) =
      evaluateK form (fun column =>
        sumMap extensionOps indices (fun index =>
          extensionOps.mul (weight index) (read index column))) := by
  calc
    _ = sumMap extensionOps indices (fun index =>
        extensionOps.mul (weight index)
          (sumMap extensionOps form.entries (fun entry =>
            extensionOps.mul (K.embed entry.coefficient) (read index entry.column)))) := by
      apply sumMap_congr
      intro index _
      rw [evaluateK_value]
    _ = sumMap extensionOps indices (fun index =>
        sumMap extensionOps form.entries (fun entry =>
          extensionOps.mul (weight index)
            (extensionOps.mul (K.embed entry.coefficient) (read index entry.column)))) := by
      apply sumMap_congr
      intro index _
      exact (sumMap_mul_left extensionOps extensionLaws (weight index) form.entries
        (fun entry => extensionOps.mul (K.embed entry.coefficient)
          (read index entry.column))).symm
    _ = sumMap extensionOps form.entries (fun entry =>
        sumMap extensionOps indices (fun index =>
          extensionOps.mul (weight index)
            (extensionOps.mul (K.embed entry.coefficient) (read index entry.column)))) :=
      sumMap_swap extensionOps extensionLaws indices form.entries
        (fun index entry => extensionOps.mul (weight index)
          (extensionOps.mul (K.embed entry.coefficient) (read index entry.column)))
    _ = sumMap extensionOps form.entries (fun entry =>
        extensionOps.mul (K.embed entry.coefficient)
          (sumMap extensionOps indices (fun index =>
            extensionOps.mul (weight index) (read index entry.column)))) := by
      apply sumMap_congr
      intro entry _
      calc
        _ = sumMap extensionOps indices (fun index =>
            extensionOps.mul (K.embed entry.coefficient)
              (extensionOps.mul (weight index) (read index entry.column))) := by
          apply sumMap_congr
          intro index _
          rw [← extensionLaws.mul_assoc,
            extensionLaws.mul_comm (weight index) (K.embed entry.coefficient),
            extensionLaws.mul_assoc]
        _ = _ := sumMap_mul_left extensionOps extensionLaws
          (K.embed entry.coefficient) indices
          (fun index => extensionOps.mul (weight index) (read index entry.column))
    _ = _ := (evaluateK_value form (fun column : Fin columns =>
      sumMap extensionOps indices (fun index =>
        extensionOps.mul (weight index) (read index column)))).symm

/-- Aggregate any finite list of weighted original reads before sparse
matrix evaluation. Repeated source indices and repeated entries are retained. -/
theorem weighted_reads {Index : Type uIndex} {columns : Nat}
    (form : SparseForm columns) (indices : List Index) (weight : Index → K)
    (read : Index → Fin columns → F) :
    sumMap extensionOps indices (fun index =>
        extensionOps.mul (weight index) (K.embed (form.evalSparse (read index)))) =
      evaluateK form (fun column =>
        sumMap extensionOps indices (fun index =>
          extensionOps.mul (weight index) (K.embed (read index column)))) := by
  simpa only [evaluateK_embed] using weighted_readsK form indices weight
    (fun (index : Index) (column : Fin columns) => K.embed (read index column))

end NightstreamFPrime.Export.Stage1.PiCCSSparseEvaluation
