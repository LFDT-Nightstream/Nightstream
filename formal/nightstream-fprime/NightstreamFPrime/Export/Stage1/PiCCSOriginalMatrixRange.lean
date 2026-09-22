import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixPreservation
import NightstreamFPrime.Export.Stage1.PiCCSOriginalReadsPreservation
import NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixBatch
import NightstreamFPrime.Export.Stage1.PiDECMatrixSelectedBatch

/-!
Proof-only transport from loaded sparse rows and prepared numeric invocations
to complete original-source matrix ranges. Generic program arithmetic is proved
before selected specialization. Successful loads and their exact offsets are
the operational premises; no split or expected evaluation is supplied.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixRange

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic (StoredRing)
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = _
  rw [Vector.getElem_ofFn]

private theorem get_replicate {Alpha : Type} {count : Nat}
    (value : Alpha) (index : Fin count) :
    (Vector.replicate count value).get index = value := by
  change (Vector.replicate count value)[index.val] = _
  rw [Vector.getElem_replicate]

/-- The generic reference has an explicit zero suffix. The loaded-range
lemmas use successful rows; selected preservation also proves every active
lookup needed by the complete family. This reference is not a loader. -/
private def rows {columns : Nat} (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin ringDegree → Fin columns → F)
    (index : Nat) : Vector StoredRing matrixCount :=
  Vector.ofFn fun port => Vector.ofFn fun output =>
    if index < program.rowCount then
      ((program.row? columns sourceRow index).map fun forms =>
        (match meaningfulPort? port with
          | some meaningful => forms meaningful
          | none => SparseForm.empty).evalSparse (read output)).getD 0
    else 0

private theorem rows_value {columns : Nat} (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin ringDegree → Fin columns → F)
    (index : Nat) (port : Fin matrixCount) (output : Fin ringDegree) :
    ((rows program sourceRow read index).get port).get output =
      if index < program.rowCount then
        ((program.row? columns sourceRow index).map fun forms =>
          (match meaningfulPort? port with
            | some meaningful => forms meaningful
            | none => SparseForm.empty).evalSparse (read output)).getD 0
      else 0 := by
  simp only [rows, get_ofFn]

/-- Reference weighted range under one complete source read. -/
def range {columns arity : Nat} (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin ringDegree → Fin columns → F)
    (first count : Nat) (point : CubePoint K arity) : Vector MaterializedRingK matrixCount :=
  PiDECEvaluationBatch.range first count point (rows program sourceRow read)

private theorem numericSum_congr (count : Nat) (left right : Nat → K) :
    (∀ index, index < count → left index = right index) →
      NumericCompletionSum.numericSum extensionOps count left =
        NumericCompletionSum.numericSum extensionOps count right := by
  induction count with
  | zero => intro _; rfl
  | succ count inductionHypothesis =>
      intro equal
      change extensionOps.add (NumericCompletionSum.numericSum extensionOps count left)
          (left count) =
        extensionOps.add (NumericCompletionSum.numericSum extensionOps count right)
          (right count)
      rw [inductionHypothesis (fun index live =>
        equal index (Nat.lt_trans live (Nat.lt_succ_self count))),
        equal count (Nat.lt_succ_self count)]

private theorem sparse_range {columns arity count : Nat} (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin ringDegree → Fin columns → F)
    (first : Nat) (point : CubePoint K arity)
    (forms : Vector (MatrixProgram.RowForms columns) count)
    (loaded : ∀ index : Fin count,
      program.row? columns sourceRow (first + index.val) = some (forms.get index))
    (fits : first + count ≤ program.rowCount) (port : Fin matrixCount) :
    ((PiDECMatrixSparseRange.sum first point read forms).get port).toRing =
      ((range program sourceRow read first count point).get port).toRing := by
  funext output
  rw [PiDECMatrixSparseRange.sum_value, range, PiDECEvaluationBatch.range_value]
  apply numericSum_congr count
  intro index live
  have globalBound : first + index < program.rowCount := by omega
  rw [rows_value, if_pos globalBound, loaded ⟨index, live⟩, Option.map_some,
    Option.getD_some, dif_pos live]
  rfl

private theorem invocation_range {columns arity : Nat} (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin ringDegree → Fin columns → F)
    (point : CubePoint K arity) (blockIndex : Nat) (block : MatrixProgram.Poseidon.Block)
    (selected : program.blocks[blockIndex]? = some (.poseidon block))
    (invocation : Fin block.invocationCount) (interface : PoseidonSboxPlan.Interface columns)
    (loaded : PiDECPoseidonNumericBlock.loadInvocation? block columns invocation = some interface)
    (fits : ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      (Fin.encodeProd (invocation, (0 : Fin 86))).val + 86 ≤ program.rowCount)
    (port : Fin matrixCount) :
    let first := ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      (Fin.encodeProd (invocation, (0 : Fin 86))).val
    ((PiDECMatrixInvocation.sum first point (PiDECMatrixInvocation.prepare read interface)).get port).toRing =
      ((range program sourceRow read first 86 point).get port).toRing := by
  dsimp only
  let first := ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
    (Fin.encodeProd (invocation, (0 : Fin 86))).val
  change ((PiDECMatrixInvocation.sum first point
    (PiDECMatrixInvocation.prepare read interface)).get port).toRing = _
  funext output
  rw [PiDECMatrixInvocation.sum_prepare_value, range, PiDECEvaluationBatch.range_value]
  apply numericSum_congr 86
  intro index live
  have globalBound : first + index < program.rowCount := by
    change first + 86 ≤ program.rowCount at fits
    omega
  have encoded : ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      (Fin.encodeProd (invocation, (⟨index, live⟩ : Fin 86))).val = first + index := by
    dsimp only [first, Fin.encodeProd, Fin.mkDivMod]
    omega
  have sparse := PiDECMatrixSelectedBatch.invocation_sparse_value program sourceRow
    (read output) blockIndex block selected invocation interface loaded ⟨index, live⟩ port
  rw [encoded] at sparse
  rw [dif_pos live, rows_value, if_pos globalBound]
  have value := congrArg (fun result : Option F => result.getD 0) sparse
  exact congrArg (fun result : F =>
    extensionOps.mul (PiDECEvaluationWeights.weight point (first + index)) (K.embed result)) value

private theorem invocation_ranges {columns arity count : Nat} (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin ringDegree → Fin columns → F)
    (point : CubePoint K arity) (blockIndex : Nat) (block : MatrixProgram.Poseidon.Block)
    (selected : program.blocks[blockIndex]? = some (.poseidon block))
    (firstInvocation : Nat) (invocationsFit : firstInvocation + count ≤ block.invocationCount)
    (interfaces : Vector (PoseidonSboxPlan.Interface columns) count)
    (loaded : ∀ index : Fin count, PiDECPoseidonNumericBlock.loadInvocation? block columns
      ⟨firstInvocation + index.val, by omega⟩ = some (interfaces.get index))
    (fits : ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      86 * firstInvocation + 86 * count ≤ program.rowCount) (port : Fin matrixCount) :
    let first := ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      86 * firstInvocation
    ((PiDECMatrixInvocationRange.sum first point read interfaces).get port).toRing =
      ((range program sourceRow read first (86 * count) point).get port).toRing := by
  dsimp only
  let first := ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
    86 * firstInvocation
  change ((PiDECMatrixInvocationRange.sum first point read interfaces).get port).toRing = _
  unfold PiDECMatrixInvocationRange.sum range
  apply PiDECMatrixSelectedBatch.sumInvocationParts_eq_range first count point _ _ port port
  intro index live
  rw [dif_pos live]
  let invocation : Fin block.invocationCount := ⟨firstInvocation + index, by omega⟩
  have loadedIndex : PiDECPoseidonNumericBlock.loadInvocation? block columns invocation =
      some (interfaces.get ⟨index, live⟩) := loaded ⟨index, live⟩
  have startEq : ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      (Fin.encodeProd (invocation, (0 : Fin 86))).val = first + 86 * index := by
    dsimp only [invocation, first, Fin.encodeProd, Fin.mkDivMod]
    omega
  have invocationFits : ((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      (Fin.encodeProd (invocation, (0 : Fin 86))).val + 86 ≤ program.rowCount := by
    rw [startEq]
    change first + 86 * count ≤ program.rowCount at fits
    omega
  have single := invocation_range program sourceRow read point blockIndex block selected
    invocation (interfaces.get ⟨index, live⟩) loadedIndex invocationFits port
  dsimp only at single
  rw [startEq] at single
  exact single

private theorem range_evaluate {columns arity : Nat} (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row) (read : Fin ringDegree → Fin columns → F)
    (point : CubePoint K arity) (fits : program.rowCount ≤ 2 ^ arity)
    (port : Fin matrixCount) (output : Fin ringDegree) :
    ((range program sourceRow read 0 program.rowCount point).get port).toRing output =
      (BooleanTable.tabulate (fun vertex : BooleanVertex arity =>
        K.embed (((rows program sourceRow read (NumericBooleanDomain.index vertex)).get port).get output))).evaluate
          extensionOps point := by
  change ((PiDECEvaluationBatch.accumulate program.rowCount point
    (rows program sourceRow read)).get port).toRing output = _
  rw [PiDECEvaluationBatch.accumulate_child]
  apply PiDECEvaluationWeights.accumulate_prefix_eq_evaluate program.rowCount point
    (fun index => (rows program sourceRow read index).get port) fits
  intro index outside _
  funext lane
  rw [rows_value, if_neg (Nat.not_lt.mpr outside)]
  rfl

private theorem rows_image (program : MatrixProgram.Program)
    (sourceRow : Nat → Option R1CS.Row)
    (read : Fin ringDegree → Fin PiCCSSourceImages.logicalWidth → F)
    (vertex : BooleanVertex cubeVariables) (port : Fin matrixCount) (output : Fin ringDegree) :
    ((rows program sourceRow read (NumericBooleanDomain.index vertex)).get port).get output =
      ((PiCCSSourceImages.rowValues? program sourceRow (read output) vertex).map
        (fun values => values.get port)).getD 0 := by
  rw [rows_value, PiCCSSourceImages.rowValues?]
  by_cases live : NumericBooleanDomain.index vertex < program.rowCount
  · rw [if_pos live, if_pos live, PiDECMatrixNumericRows.row?_value]
    rfl
  · rw [if_neg live, if_neg live, Option.map_some, get_replicate, Option.getD_some]

private abbrev selectedProgram := PerApplicationMatrixProgram.matrixProgram
  Poseidon2HashChainV1Package.application
private abbrev selectedSource := PerApplicationCanonicalPackage.sourceRow
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev selectedRelation := PerApplicationFixedPoint.relation
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits

/-- One source's complete canonical reference range. This shares the exact
original reader used by the source/port batch runtime. -/
def originalRange (masks : Array (Array (Nat × Nat))) (point : PaperAlgebra.Point)
    (first count : Nat) (source : Fin productionShape.sourceCount) :
    Vector MaterializedRingK matrixCount :=
  range selectedProgram selectedSource
    (PiCCSOriginalReads.read (columns := PiCCSSourceImages.logicalWidth) (PiDECParentSparseRead.prepare ()) masks source) first count point

/-- Successful sparse loads at the stated offsets preserve every original
source and every matrix port. This includes all 54 lanes of fresh source 0. -/
theorem sparse_eq_range (masks : Array (Array (Nat × Nat))) (point : PaperAlgebra.Point)
    (first count : Nat) (forms : Vector (MatrixProgram.RowForms PiCCSSourceImages.logicalWidth) count)
    (loaded : ∀ index : Fin count, selectedProgram.row? PiCCSSourceImages.logicalWidth
      selectedSource (first + index.val) = some (forms.get index))
    (fits : first + count ≤ selectedProgram.rowCount)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) :
    ((PiCCSOriginalMatrixBatch.sum first point
      (PiCCSOriginalReads.read (columns := PiCCSSourceImages.logicalWidth) (PiDECParentSparseRead.prepare ()) masks) forms).get
      (Fin.encodeProd (source, port))).toRing =
      ((originalRange masks point first count source).get port).toRing := by
  rw [PiCCSOriginalMatrixBatch.sum_source_port]
  exact sparse_range selectedProgram selectedSource
    (PiCCSOriginalReads.read (columns := PiCCSSourceImages.logicalWidth) (PiDECParentSparseRead.prepare ()) masks source)
    first point forms loaded fits port

/-- Loaded contiguous numeric invocations preserve the same canonical range.
The runtime retains prepare/stored; sparse invocation rows occur only in proofs. -/
theorem invocations_eq_range (masks : Array (Array (Nat × Nat))) (point : PaperAlgebra.Point)
    (blockIndex : Nat) (block : MatrixProgram.Poseidon.Block)
    (selected : selectedProgram.blocks[blockIndex]? = some (.poseidon block))
    (firstInvocation count : Nat) (invocationsFit : firstInvocation + count ≤ block.invocationCount)
    (interfaces : Vector (PoseidonSboxPlan.Interface PiCCSSourceImages.logicalWidth) count)
    (loaded : ∀ index : Fin count, PiDECPoseidonNumericBlock.loadInvocation? block
      PiCCSSourceImages.logicalWidth ⟨firstInvocation + index.val, by omega⟩ = some (interfaces.get index))
    (fits : ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      86 * firstInvocation + 86 * count ≤ selectedProgram.rowCount)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) :
    let first := ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      86 * firstInvocation
    ((PiCCSOriginalMatrixBatch.sumInvocations first point
      (PiCCSOriginalReads.read (columns := PiCCSSourceImages.logicalWidth) (PiDECParentSparseRead.prepare ()) masks) interfaces).get
      (Fin.encodeProd (source, port))).toRing =
      ((originalRange masks point first (86 * count) source).get port).toRing := by
  dsimp only
  rw [PiCCSOriginalMatrixBatch.sumInvocations_source_port]
  exact invocation_ranges selectedProgram selectedSource
    (PiCCSOriginalReads.read (columns := PiCCSSourceImages.logicalWidth) (PiDECParentSparseRead.prepare ()) masks source)
    point blockIndex block selected firstInvocation invocationsFit interfaces loaded fits port

private theorem selected_rows_value (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (vertex : BooleanVertex cubeVariables)
    (port : Fin matrixCount) (output : Fin ringDegree) :
    ((rows selectedProgram selectedSource
      (PiCCSOriginalReads.read (columns := PiCCSSourceImages.logicalWidth) (PiDECParentSparseRead.prepare ()) masks source)
      (NumericBooleanDomain.index vertex)).get port).get output =
      PiRLC.rowRing selectedRelation.system (PiCCSOriginalReads.assignment masks source) port vertex output := by
  rw [rows_image, PiCCSOriginalReads.read_eq_kernelRead]
  change ((PiCCSSourceImages.matrixImage? selectedProgram selectedSource
    (PiCCSOriginalReads.assignment masks source) output vertex).map
      (fun values => values.get port)).getD 0 = _
  rw [PiCCSSourceImages.matrixImage_value, Option.getD_some]

private theorem evaluationFamily_matrix {logicalWidth : Nat}
    {publicFits : ringDegree * PaperAlgebra.publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
    (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
    (assignment : Phi81Relation.Assignment (PaperAlgebra.FullShape logicalWidth publicFits))
    (point : PaperAlgebra.Point) (port : Fin productionShape.matrixCount) :
    (PaperAlgebra.evaluationFamily
      (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation) assignment point).matrix port =
      fun output => (BooleanTable.tabulate (fun vertex =>
        K.embed (PiRLC.rowRing relation.system assignment port vertex output))).evaluate extensionOps point := by
  exact congrArg (fun family : PaperAlgebra.Evaluation => family.matrix port)
    (PaperAlgebra.evaluationFamily_eq_paper
      (Lifecycle.PiRLC.v1_1.InputBinding.relationSource relation).cubeLayout relation.system assignment point)

/-- The complete reference range equals the passed matrix-family reference.
The selected row bound and all omitted zero rows are derived here. No caller
supplies a matrix answer, tail condition, or successful split. -/
theorem range_eq_matrix (masks : Array (Array (Nat × Nat))) (point : PaperAlgebra.Point)
    (source : Fin productionShape.sourceCount) (port : Fin matrixCount) :
    ((originalRange masks point 0 selectedProgram.rowCount source).get port).toRing =
      (PiCCSOriginalMatrixPreservation.matrix masks point source port).toRing := by
  have fits : selectedProgram.rowCount ≤ 2 ^ cubeVariables := by
    rw [PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits]
    exact (PerApplicationFixedPoint.structuralPlan
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits).rowCount_le
  rw [PiCCSOriginalMatrixPreservation.matrix_eq_evaluationFamily,
    evaluationFamily_matrix selectedRelation (PiCCSOriginalReads.assignment masks source) point port]
  funext output
  rw [originalRange, range_evaluate selectedProgram selectedSource
    (PiCCSOriginalReads.read (columns := PiCCSSourceImages.logicalWidth) (PiDECParentSparseRead.prepare ()) masks source) point fits port output]
  apply congrArg (fun values : BooleanVertex cubeVariables → K =>
    (BooleanTable.tabulate values).evaluate extensionOps point)
  funext vertex
  exact congrArg K.embed (selected_rows_value masks source vertex port output)

end NightstreamFPrime.Export.Stage1.PiCCSOriginalMatrixRange
