import Batteries.Data.Fin.Coding
import NightstreamFPrime.Export.Stage1.PiDECMatrixSparseRange
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocation
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange
import NightstreamFPrime.Lifecycle.Types

/-!
Owns shared matrix-row batches for all original sources, indexed source then
port. Sparse forms and prepared numeric invocations keep their existing
arithmetic owners. Original-source reads, loaded-row provenance, and the
complete evaluation-family link remain caller obligations.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixBatch

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.ProductionRelation.RowSemantics (PortValues)
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle (productionShape)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

abbrev Batch := Vector MaterializedRingK (productionShape.sourceCount * matrixCount)

abbrev PreparedInvocation :=
  Vector (Vector (Vector PortValues 94) ringDegree) productionShape.sourceCount

private theorem ofFn_get {Alpha : Type} {size : Nat}
    (values : Fin size → Alpha) (index : Fin size) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = values index
  rw [Vector.getElem_ofFn]

private def sparseRow {columns count : Nat} (firstRow : Nat)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count) (index : Nat) :
    Vector StoredRing (productionShape.sourceCount * matrixCount) :=
  if firstRow ≤ index then
    if live : index - firstRow < count then
      let selected := forms.get ⟨index - firstRow, live⟩
      let byPort : Vector (Vector StoredRing productionShape.sourceCount) matrixCount :=
        Vector.ofFn fun port =>
          let form := match meaningfulPort? port with
            | some meaningful => selected meaningful
            | none => SparseForm.empty
          Vector.ofFn fun source => Vector.ofFn fun output => form.evalSparse (read source output)
      Vector.ofFn fun code =>
        let decoded : Fin productionShape.sourceCount × Fin matrixCount := Fin.decodeProd code
        (byPort.get decoded.2).get decoded.1
    else Vector.replicate (productionShape.sourceCount * matrixCount) (Vector.replicate ringDegree 0)
  else Vector.replicate (productionShape.sourceCount * matrixCount) (Vector.replicate ringDegree 0)

private theorem sparseRow_value {columns count : Nat} (firstRow : Nat)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (index : Nat) (source : Fin productionShape.sourceCount)
    (port : Fin matrixCount) (output : Fin ringDegree) :
    ((sparseRow firstRow read forms (firstRow + index)).get
        (Fin.encodeProd (source, port))).get output =
      if live : index < count then
        (match meaningfulPort? port with
          | some meaningful => (forms.get ⟨index, live⟩) meaningful
          | none => SparseForm.empty).evalSparse (read source output)
      else 0 := by
  have lower : firstRow ≤ firstRow + index := by omega
  rw [sparseRow, if_pos lower, Nat.add_sub_cancel_left]
  by_cases live : index < count
  · simp only [dif_pos live, ofFn_get, Fin.decodeProd_encodeProd]
  · rw [dif_neg live, dif_neg live]
    change ((Vector.replicate (productionShape.sourceCount * matrixCount)
      (Vector.replicate ringDegree (0 : F)))[(Fin.encodeProd (source, port)).val])[output.val] = 0
    rw [Vector.getElem_replicate, Vector.getElem_replicate]

/-- One global row traversal and one point weight per row for all source/port
pairs. Each loaded row and port form is selected once before its source loop. -/
def sum {columns arity count : Nat} (firstRow : Nat) (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count) : Batch :=
  PiDECEvaluationBatch.range firstRow count point (sparseRow firstRow read forms)

/-- Every source/port retains the existing complete 54-lane weighted sparse
range. This equality needs no premise on reads, forms, point, or source values. -/
theorem sum_source_port {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) :
    ((sum firstRow point read forms).get (Fin.encodeProd (source, port))).toRing =
      ((PiDECMatrixSparseRange.sum firstRow point (read source) forms).get port).toRing := by
  funext output
  rw [sum, PiDECEvaluationBatch.range_value, PiDECMatrixSparseRange.sum_value]
  apply congrArg (NumericCompletionSum.numericSum extensionOps count)
  funext index
  rw [sparseRow_value]
  rfl

/-- Prepare each original source's numeric invocation once. This uses the
existing stored numeric evaluator and does not construct sparse plan rows. -/
def prepareInvocation {columns : Nat}
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) : PreparedInvocation :=
  Vector.ofFn fun source => PiDECMatrixInvocation.prepare (read source) interface

/-- The prepared source table is exactly the existing invocation constructor. -/
theorem prepareInvocation_source {columns : Nat}
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (interface : PoseidonSboxPlan.Interface columns) (source : Fin productionShape.sourceCount) :
    (prepareInvocation read interface).get source =
      PiDECMatrixInvocation.prepare (read source) interface := by
  simp only [prepareInvocation, ofFn_get]

private def invocationRow (prepared : PreparedInvocation) (start index : Nat) :
    Vector StoredRing (productionShape.sourceCount * matrixCount) :=
  if start ≤ index then
    if bounded : index - start < 94 then
      let selected := Vector.ofFn fun source : Fin productionShape.sourceCount =>
        Vector.ofFn fun output : Fin ringDegree =>
          ((prepared.get source).get output).get ⟨index - start, bounded⟩
      Vector.ofFn fun code =>
        let decoded : Fin productionShape.sourceCount × Fin matrixCount := Fin.decodeProd code
        Vector.ofFn fun output => ((selected.get decoded.1).get output).get decoded.2
    else Vector.replicate (productionShape.sourceCount * matrixCount) (Vector.replicate ringDegree 0)
  else Vector.replicate (productionShape.sourceCount * matrixCount) (Vector.replicate ringDegree 0)

private theorem invocationRow_value (prepared : PreparedInvocation) (start index : Nat)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) (output : Fin ringDegree) :
    ((invocationRow prepared start (start + index)).get
        (Fin.encodeProd (source, port))).get output =
      if bounded : index < 94 then
        (((prepared.get source).get output).get ⟨index, bounded⟩).get port
      else 0 := by
  have lower : start ≤ start + index := by omega
  rw [invocationRow, if_pos lower, Nat.add_sub_cancel_left]
  by_cases bounded : index < 94
  · simp only [dif_pos bounded, ofFn_get, Fin.decodeProd_encodeProd]
  · rw [dif_neg bounded, dif_neg bounded]
    change ((Vector.replicate (productionShape.sourceCount * matrixCount)
      (Vector.replicate ringDegree (0 : F)))[(Fin.encodeProd (source, port)).val])[output.val] = 0
    rw [Vector.getElem_replicate, Vector.getElem_replicate]

/-- One 94-row traversal of supplied numeric tables. Each stored source/lane
record is selected once per row and shared across all matrix ports. -/
def sumInvocation {arity : Nat} (start : Nat) (point : CubePoint K arity)
    (prepared : PreparedInvocation) : Batch :=
  PiDECEvaluationBatch.range start 94 point (invocationRow prepared start)

/-- Exact projection for arbitrary supplied prepared tables, including every
source, port, ring lane and both field coordinates. The point weights and
single-source arithmetic remain those of PiDECMatrixInvocation.sum. -/
theorem sumInvocation_source_port {arity : Nat} (start : Nat) (point : CubePoint K arity)
    (prepared : PreparedInvocation) (source : Fin productionShape.sourceCount)
    (port : Fin matrixCount) :
    ((sumInvocation start point prepared).get (Fin.encodeProd (source, port))).toRing =
      ((PiDECMatrixInvocation.sum start point (prepared.get source)).get port).toRing := by
  funext output
  rw [sumInvocation, PiDECEvaluationBatch.range_value, PiDECMatrixInvocation.sum_value]
  apply congrArg (NumericCompletionSum.numericSum extensionOps 94)
  funext index
  rw [invocationRow_value]

/-- Sum contiguous loaded invocations in global row order. Each source is
prepared once per interface, and all source/port totals use the existing
extension-ring batch sum. -/
def sumInvocations {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count) : Batch :=
  PiDECEvaluationBatch.sum count fun index =>
    if live : index < count then
      sumInvocation (firstRow + 94 * index) point
        (prepareInvocation read (interfaces.get ⟨index, live⟩))
    else PiDECEvaluationBatch.zero (productionShape.sourceCount * matrixCount)

/-- Every complete source/port value equals the existing single-source
invocation range with the same reads, interfaces, point and global offset. -/
theorem sumInvocations_source_port {columns arity count : Nat} (firstRow : Nat)
    (point : CubePoint K arity)
    (read : Fin productionShape.sourceCount → Fin ringDegree → Fin columns → F)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) :
    ((sumInvocations firstRow point read interfaces).get (Fin.encodeProd (source, port))).toRing =
      ((PiDECMatrixInvocationRange.sum firstRow point (read source) interfaces).get port).toRing := by
  funext output
  rw [sumInvocations, PiDECMatrixInvocationRange.sum,
    PiDECEvaluationBatch.sum_value, PiDECEvaluationBatch.sum_value]
  apply congrArg (NumericCompletionSum.numericSum extensionOps count)
  funext index
  by_cases live : index < count
  · simpa only [dif_pos live, prepareInvocation_source] using
      congrFun (sumInvocation_source_port (firstRow + 94 * index) point
        (prepareInvocation read (interfaces.get ⟨index, live⟩)) source port) output
  · simp only [dif_neg live, PiDECEvaluationBatch.zero_value]

end NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixBatch
