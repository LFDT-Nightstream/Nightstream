import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericRows
import NightstreamFPrime.Export.Stage1.PiDECEvaluationBatch

/-!
Prepare one Poseidon invocation for all Phi81 output reads, then weight its
94 global rows. The stored table contains the existing PortValues records;
the sum reads that table without repeating numeric invocation evaluation.
This computes one child's fourteen matrix contributions.
-/

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

/-- Materialize all 54 complete numeric invocations once. Each invocation
stores its 94 original port records, including all output pins. -/
def prepare {columns : Nat} (read : Fin ringDegree → Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) :
    Vector (Vector PortValues 94) ringDegree :=
  Vector.ofFn fun output => PiDECPoseidonNumericRows.stored (read output) interface

/-- Every prepared cell is the corresponding original sparse evaluation. -/
theorem prepare_value {columns : Nat} (read : Fin ringDegree → Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (output : Fin ringDegree)
    (rowIndex : Fin 94) (matrix : Fin matrixCount) :
    (((prepare read interface).get output).get rowIndex).get matrix =
      (((PoseidonSboxPlan.rows interface).get
        ⟨rowIndex.val, by rw [PoseidonSboxPlan.rows_length]; exact rowIndex.isLt⟩).portForm matrix).evalSparse (read output) := by
  change (((Vector.ofFn (fun lane : Fin ringDegree =>
    PiDECPoseidonNumericRows.stored (read lane) interface))[output.val]).get rowIndex).get matrix = _
  rw [Vector.getElem_ofFn]
  exact PiDECPoseidonNumericRows.stored_value (read output) interface rowIndex matrix

private def row (prepared : Vector (Vector PortValues 94) ringDegree)
    (start index : Nat) : Vector StoredRing matrixCount :=
  if start ≤ index then
    if bounded : index - start < 94 then
      Vector.ofFn fun matrix => Vector.ofFn fun output =>
        ((prepared.get output).get ⟨index - start, bounded⟩).get matrix
    else Vector.replicate matrixCount (Vector.replicate ringDegree 0)
  else Vector.replicate matrixCount (Vector.replicate ringDegree 0)

private theorem row_value (prepared : Vector (Vector PortValues 94) ringDegree)
    (start index : Nat) (matrix : Fin matrixCount) (output : Fin ringDegree) :
    ((row prepared start (start + index)).get matrix).get output =
      if bounded : index < 94 then ((prepared.get output).get ⟨index, bounded⟩).get matrix
      else 0 := by
  have lower : start ≤ start + index := by omega
  rw [row, if_pos lower, Nat.add_sub_cancel_left]
  by_cases bounded : index < 94
  · rw [dif_pos bounded, dif_pos bounded]
    change ((Vector.ofFn (fun port : Fin matrixCount =>
      Vector.ofFn (fun lane : Fin ringDegree =>
        ((prepared.get lane).get ⟨index, bounded⟩).get port)))[matrix.val])[output.val] = _
    rw [Vector.getElem_ofFn, Vector.getElem_ofFn]
  · rw [dif_neg bounded, dif_neg bounded]
    change ((Vector.replicate matrixCount
      (Vector.replicate ringDegree (0 : F)))[matrix.val])[output.val] = 0
    rw [Vector.getElem_replicate, Vector.getElem_replicate]

/-- Weight the 94 global rows from the prepared table. All fourteen matrices
share each computed point weight through the existing batch range loop. -/
def sum {arity : Nat} (start : Nat) (point : CubePoint K arity)
    (prepared : Vector (Vector PortValues 94) ringDegree) :
    Vector MaterializedRingK matrixCount :=
  PiDECEvaluationBatch.range start 94 point (row prepared start)

private theorem sum_value {arity : Nat} (start : Nat) (point : CubePoint K arity)
    (prepared : Vector (Vector PortValues 94) ringDegree)
    (matrix : Fin matrixCount) (output : Fin ringDegree) :
    ((sum start point prepared).get matrix).toRing output =
      NumericCompletionSum.numericSum extensionOps 94 (fun index =>
        extensionOps.mul (PiDECEvaluationWeights.weight point (start + index))
          (K.embed (if bounded : index < 94 then
            ((prepared.get output).get ⟨index, bounded⟩).get matrix
          else 0))) := by
  rw [sum, PiDECEvaluationBatch.range_value]
  apply congrArg (NumericCompletionSum.numericSum extensionOps 94)
  funext index
  rw [row_value]

/-- Every matrix/output coefficient is the original 94-row weighted sparse
evaluation for the supplied read family and interface. No row-validity,
constant-column, norm or expected-value premise is required. -/
theorem sum_prepare_value {columns arity : Nat}
    (read : Fin ringDegree → Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns)
    (start : Nat) (point : CubePoint K arity)
    (matrix : Fin matrixCount) (output : Fin ringDegree) :
    ((sum start point (prepare read interface)).get matrix).toRing output =
      NumericCompletionSum.numericSum extensionOps 94 (fun index =>
        extensionOps.mul (PiDECEvaluationWeights.weight point (start + index))
          (K.embed (if bounded : index < 94 then
            (((PoseidonSboxPlan.rows interface).get
              ⟨index, by rw [PoseidonSboxPlan.rows_length]; exact bounded⟩).portForm matrix).evalSparse (read output)
          else 0))) := by
  rw [sum_value]
  apply congrArg (NumericCompletionSum.numericSum extensionOps 94)
  funext index
  by_cases bounded : index < 94
  · simp only [dif_pos bounded, prepare_value]
  · simp only [dif_neg bounded]

end NightstreamFPrime.Export.Stage1.PiDECMatrixInvocation
