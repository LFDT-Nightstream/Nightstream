import NightstreamFPrime.Layout.MatrixProgram
import NightstreamFPrime.Layout.ProductionRelation.SparseEvaluation
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch
import NightstreamFPrime.Export.Stage1.PiDECNativeSparseEvaluation

/-!
Evaluate a contiguous vector of existing sparse matrix rows. Each row and
port form is selected once, then all 54 output lanes are evaluated. The
existing batch loop shares the global point weight across all 14 ports.
No matrix rows, field values, or read functions are assumed correct here.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixSparseRange

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

private theorem ofFn_get {Alpha : Type} {size : Nat}
    (values : Fin size → Alpha) (index : Fin size) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

@[specialize] private def row {columns count : Nat} (firstRow : Nat)
    (read : Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (index : Nat) : Vector StoredRing matrixCount :=
  if firstRow ≤ index then
    if live : index - firstRow < count then
      let selected := forms.get ⟨index - firstRow, live⟩
      Vector.ofFn fun port =>
        let form := match meaningfulPort? port with
          | some meaningful => selected meaningful
          | none => SparseForm.empty
        Vector.ofFn fun output =>
          PiDECNativeSparseEvaluation.nativeEvalSparse form (read output)
    else Vector.replicate matrixCount (Vector.replicate ringDegree 0)
  else Vector.replicate matrixCount (Vector.replicate ringDegree 0)

private theorem row_value {columns count : Nat} (firstRow : Nat)
    (read : Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (index : Nat) (port : Fin matrixCount) (output : Fin ringDegree) :
    ((row firstRow read forms (firstRow + index)).get port).get output =
      if live : index < count then
        (match meaningfulPort? port with
          | some meaningful => (forms.get ⟨index, live⟩) meaningful
          | none => SparseForm.empty).evalSparse (read output)
      else 0 := by
  have lower : firstRow ≤ firstRow + index := by omega
  rw [row, if_pos lower, Nat.add_sub_cancel_left]
  by_cases live : index < count
  · simp only [dif_pos live, ofFn_get, PiDECNativeSparseEvaluation.nativeEvalSparse_eq_spec]
  · rw [dif_neg live, dif_neg live]
    change ((Vector.replicate matrixCount
      (Vector.replicate ringDegree (0 : F)))[port.val])[output.val] = 0
    rw [Vector.getElem_replicate, Vector.getElem_replicate]

/-- Compute the loaded rows in global order for one child, including the
explicit zero matrix port. The read family can use any parent accessor. -/
@[specialize] def sum {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity) (read : Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count) :
    Vector MaterializedRingK matrixCount :=
  PiDECEvaluationBatch.range firstRow count point (row firstRow read forms)

/-- Every output is the complete weighted sparse sum of the supplied rows.
Duplicate entries, cancellation and both field coordinates retain the
existing evalSparse and batch semantics. No read or row premise is needed. -/
theorem sum_value {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity) (read : Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (port : Fin matrixCount) (output : Fin ringDegree) :
    ((sum firstRow point read forms).get port).toRing output =
      NumericCompletionSum.numericSum extensionOps count (fun index =>
        extensionOps.mul (PiDECEvaluationWeights.weight point (firstRow + index))
          (K.embed (if live : index < count then
            (match meaningfulPort? port with
              | some meaningful => (forms.get ⟨index, live⟩) meaningful
              | none => SparseForm.empty).evalSparse (read output)
            else 0))) := by
  rw [sum, PiDECEvaluationBatch.range_value]
  apply congrArg (NumericCompletionSum.numericSum extensionOps count)
  funext index
  rw [row_value]

end NightstreamFPrime.Export.Stage1.PiDECMatrixSparseRange
