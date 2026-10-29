import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericRows
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixInvocation

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.ProductionRelation.RowSemantics (PortValues)
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

@[specialize] def prepare {columns : Nat} (read : Fin ringDegree → Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) :
    Vector (Vector PortValues 150) ringDegree :=
  Vector.ofFn fun output => PiDECPoseidonNumericRows.stored (read output) interface

theorem prepare_value {columns : Nat} (read : Fin ringDegree → Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (output : Fin ringDegree)
    (rowIndex : Fin 150) (matrix : Fin matrixCount) :
    (((prepare read interface).get output).get rowIndex).get matrix =
      (((PoseidonRetainedRows.rows interface).get
        ⟨rowIndex.val, by rw [PoseidonRetainedRows.rows_length]; exact rowIndex.isLt⟩).portForm matrix).evalSparse (read output) := by
  change (((Vector.ofFn (fun lane : Fin ringDegree =>
    PiDECPoseidonNumericRows.stored (read lane) interface))[output.val]).get rowIndex).get matrix = _
  rw [Vector.getElem_ofFn]
  exact PiDECPoseidonNumericRows.stored_value (read output) interface rowIndex matrix

private def row (prepared : Vector (Vector PortValues 150) ringDegree)
    (start index : Nat) : Vector StoredRing matrixCount :=
  if start ≤ index then
    if bounded : index - start < 150 then
      Vector.ofFn fun matrix => Vector.ofFn fun output =>
        ((prepared.get output).get ⟨index - start, bounded⟩).get matrix
    else Vector.replicate matrixCount (Vector.replicate ringDegree 0)
  else Vector.replicate matrixCount (Vector.replicate ringDegree 0)

private theorem row_value (prepared : Vector (Vector PortValues 150) ringDegree)
    (start index : Nat) (matrix : Fin matrixCount) (output : Fin ringDegree) :
    ((row prepared start (start + index)).get matrix).get output =
      if bounded : index < 150 then ((prepared.get output).get ⟨index, bounded⟩).get matrix
      else 0 := by
  have lower : start ≤ start + index := by omega
  rw [row, ite_eq_left lower, Nat.add_sub_cancel_left]
  by_cases bounded : index < 150
  · rw [dite_eq_left bounded, dite_eq_left bounded]
    change ((Vector.ofFn (fun port : Fin matrixCount =>
      Vector.ofFn (fun lane : Fin ringDegree =>
        ((prepared.get lane).get ⟨index, bounded⟩).get port)))[matrix.val])[output.val] = _
    rw [Vector.getElem_ofFn, Vector.getElem_ofFn]
  · rw [dite_eq_right bounded, dite_eq_right bounded]
    change ((Vector.replicate matrixCount
      (Vector.replicate ringDegree (0 : F)))[matrix.val])[output.val] = 0
    rw [Vector.getElem_replicate, Vector.getElem_replicate]

def sum {arity : Nat} (start : Nat) (point : CubePoint K arity)
    (prepared : Vector (Vector PortValues 150) ringDegree) :
    Vector MaterializedRingK matrixCount :=
  PiDECEvaluationBatch.range start 150 point (row prepared start)

theorem sum_value {arity : Nat} (start : Nat) (point : CubePoint K arity)
    (prepared : Vector (Vector PortValues 150) ringDegree)
    (matrix : Fin matrixCount) (output : Fin ringDegree) :
    ((sum start point prepared).get matrix).toRing output =
      NumericCompletionSum.numericSum extensionOps 150 (fun index =>
        extensionOps.mul (PiDECEvaluationWeights.weight point (start + index))
          (K.embed (if bounded : index < 150 then
            ((prepared.get output).get ⟨index, bounded⟩).get matrix
          else 0))) := by
  rw [sum, PiDECEvaluationBatch.range_value]
  apply congrArg (NumericCompletionSum.numericSum extensionOps 150)
  funext index
  rw [row_value]

theorem sum_prepare_value {columns arity : Nat}
    (read : Fin ringDegree → Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns)
    (start : Nat) (point : CubePoint K arity)
    (matrix : Fin matrixCount) (output : Fin ringDegree) :
    ((sum start point (prepare read interface)).get matrix).toRing output =
      NumericCompletionSum.numericSum extensionOps 150 (fun index =>
        extensionOps.mul (PiDECEvaluationWeights.weight point (start + index))
          (K.embed (if bounded : index < 150 then
            (((PoseidonRetainedRows.rows interface).get
              ⟨index, by rw [PoseidonRetainedRows.rows_length]; exact bounded⟩).portForm matrix).evalSparse (read output)
          else 0))) := by
  rw [sum_value]
  apply congrArg (NumericCompletionSum.numericSum extensionOps 150)
  funext index
  by_cases bounded : index < 150
  · simp only [dite_eq_left bounded, prepare_value]
  · simp only [dite_eq_right bounded]

end NightstreamFPrime.Export.Stage1.PiDECMatrixInvocation
