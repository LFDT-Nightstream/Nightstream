import NightstreamFPrime.Export.Stage1.PiDECParentIntRead
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocationRange
import NightstreamFPrime.Export.Stage1.PiDECMatrixSparseRange
import NightstreamFPrime.Export.Stage1.PiDECCanonicalSourceCache
import NightstreamFPrime.Export.Stage1.PiDECPoseidonNumericBlock
import NightstreamFPrime.Export.Stage1.PiDECEvaluationFromBlocks

/-!
Connect one selected Poseidon invocation to its canonical matrix range.
The parent supplies digits; the selected program supplies all matrix rows.
Only block selection, interface loading and the executed range bound are
operational premises. No expected child or matrix value selects a result.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECMatrixSelectedBatch

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.ProductionRelation
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.PiDECAlgebra
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.MatrixProgram
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle

private abbrev selectedColumns :=
  PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application
private abbrev selectedShape := PaperAlgebra.FullShape selectedColumns
  (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)
private abbrev selectedPlan := PerApplicationFixedPoint.structuralPlan
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev selectedProgram := PerApplicationMatrixProgram.matrixProgram
  Poseidon2HashChainV1Package.application
private abbrev selectedSource := PerApplicationCanonicalPackage.sourceRow
  Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
private abbrev blockCount := Phi81ColumnLayout.blockCount selectedShape.carrierWidth

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

private theorem get_ofFn_function {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) : (Vector.ofFn values).get = values := by
  funext index
  exact get_ofFn values index

/-- Existing scalar splits in the existing stored block type. This reference
is not constructed by the executable parent reader. -/
def splitBlocks {count : Nat} (parents : Fin count → StoredRing)
    (block : Fin count) : Vector StoredRing productionGlobalParams.k :=
  Vector.ofFn fun child => Vector.ofFn fun input =>
    Radix.splitScalar ((parents block).get input) child

/-- The exact guarded read used by the invocation runner. -/
def parentRead {count columns : Nat} (parents : Fin count → StoredRing)
    (child : Fin productionGlobalParams.k) : Fin ringDegree → Fin columns → F :=
  let forms := PiDECParentSparseRead.prepare ()
  fun output column =>
    if live : column.val / ringDegree < count then
      PiDECParentSparseRead.read forms (parents ⟨column.val / ringDegree, live⟩)
        ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ child output
    else 0

private theorem parentRead_value (parents : Fin blockCount → StoredRing)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree)
    (column : Fin selectedColumns) :
    parentRead parents child output column =
      CarrierAction.kernelImage
        ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
        (((PiDECEvaluationFromBlocks.blockAt (splitBlocks parents))
          (column.val / ringDegree)).get child).get output := by
  dsimp only [parentRead]
  by_cases live : column.val / ringDegree < blockCount
  · rw [dif_pos live, PiDECParentSparseRead.read_eq_splitScalar]
    simp only [PiDECEvaluationFromBlocks.blockAt, dif_pos live,
      splitBlocks, get_ofFn, get_ofFn_function]
  · rw [dif_neg live]
    simp only [PiDECEvaluationFromBlocks.blockAt, dif_neg live,
      get_replicate, PiDECCommitmentFold.zero_value]
    rw [CarrierAction.kernelImage_eq_ringFMul, CarrierAction.ringFMul_zero_right]
    rfl

private theorem program_selected_row (program : MatrixProgram.Program)
    (columns : Nat) (source : Nat → Option R1CS.Row)
    (index : Nat) (block : MatrixProgram.Block)
    (selected : program.blocks[index]? = some block)
    (ordinal : Nat) (live : ordinal < block.rowCount) :
    program.row? columns source
      (((program.blocks.take index).map MatrixProgram.Block.rowCount).sum + ordinal) =
      block.row? columns source ordinal := by
  obtain ⟨indexBound, atIndex⟩ := List.getElem?_eq_some_iff.mp selected
  have dropped : program.blocks.drop index =
      block :: program.blocks.drop (index + 1) := by
    rw [List.drop_eq_getElem_cons indexBound, atIndex]
  have split : program.blocks = program.blocks.take index ++
      block :: program.blocks.drop (index + 1) := by
    rw [← dropped]
    exact (List.take_append_drop index program.blocks).symm
  have programEq : program =
      (MatrixProgram.Program.mk (program.blocks.take index)).append
        (MatrixProgram.Program.mk (block :: program.blocks.drop (index + 1))) := by
    cases program
    exact congrArg MatrixProgram.Program.mk split
  calc
    _ = (MatrixProgram.Program.mk (block :: program.blocks.drop (index + 1))).row?
        columns source ordinal := by
      have transport := MatrixProgram.Program.append_right_row?
        (MatrixProgram.Program.mk (program.blocks.take index))
        (MatrixProgram.Program.mk (block :: program.blocks.drop (index + 1)))
        columns source ordinal
      rw [← programEq] at transport
      exact transport
    _ = _ := MatrixProgram.Program.cons_first_row? _ _ columns source ordinal live

theorem invocation_sparse_value {columns : Nat}
    (program : MatrixProgram.Program) (source : Nat → Option R1CS.Row)
    (read : Fin columns → F) (blockIndex : Nat) (block : Poseidon.Block)
    (selected : program.blocks[blockIndex]? = some (.poseidon block))
    (invocation : Fin block.invocationCount)
    (interface : PoseidonSboxPlan.Interface columns)
    (loaded : PiDECPoseidonNumericBlock.loadInvocation? block columns invocation =
      some interface)
    (row : Fin 86) (port : Fin matrixCount) :
    some ((((PoseidonRetainedRows.rows interface).get
      ⟨row.val, by rw [PoseidonRetainedRows.rows_length]; exact row.isLt⟩).portForm port).evalSparse
        read) =
      (program.row? columns source
        (((program.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
          (Fin.encodeProd (invocation, row)).val)).map (fun forms =>
            (match meaningfulPort? port with
              | some meaningful => forms meaningful
              | none => SparseForm.empty).evalSparse read) := by
  rw [program_selected_row program columns source blockIndex (.poseidon block)
    selected _ (Fin.encodeProd (invocation, row)).isLt]
  change _ = (block.row? columns (Fin.encodeProd (invocation, row)).val).map _
  have value := PiDECPoseidonNumericBlock.row?_value block read
    (Fin.encodeProd (invocation, row)).val port
  simpa only [PiDECPoseidonNumericBlock.row?,
    PiDECPoseidonNumericBlock.loadRow?_encodeProd, loaded,
    Option.map_some, PiDECPoseidonNumericRows.stored_value] using value

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

/-- Every child, matrix port and output lane of the selected invocation is
the exact canonical weighted range. The whole parent carrier is available;
the explicit zero suffix follows the same guard as the canonical block reader.
Successful full splitting can subsequently identify splitBlocks with the
existing childBlocks used by familyFromBlocks_honestMessages. -/
theorem selectedInvocation_eq_range
    (parents : Fin blockCount → StoredRing) (point : PaperAlgebra.Point)
    (blockIndex : Nat) (block : Poseidon.Block)
    (selected : selectedProgram.blocks[blockIndex]? = some (.poseidon block))
    (invocation : Fin block.invocationCount)
    (interface : PoseidonSboxPlan.Interface selectedColumns)
    (loaded : PiDECPoseidonNumericBlock.loadInvocation? block selectedColumns invocation =
      some interface)
    (rangeFits :
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        (Fin.encodeProd (invocation, (0 : Fin 86))).val + 86 ≤ selectedProgram.rowCount)
    (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    let first :=
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        (Fin.encodeProd (invocation, (0 : Fin 86))).val
    ((PiDECMatrixInvocation.sum first point
      (PiDECMatrixInvocation.prepare (parentRead parents child) interface)).get port).toRing =
      ((PiDECEvaluationBatch.range first 86 point
        (PiDECEvaluationFromBlocks.matrixRow (splitBlocks parents) port)).get child).toRing := by
  dsimp only
  let first :=
    ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      (Fin.encodeProd (invocation, (0 : Fin 86))).val
  change ((PiDECMatrixInvocation.sum first point
    (PiDECMatrixInvocation.prepare (parentRead parents child) interface)).get port).toRing = _
  have countEq : selectedProgram.rowCount = selectedPlan.rowCount :=
    PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
  have covered : first + 86 ≤ selectedPlan.rowCount := by
    change first + 86 ≤ selectedProgram.rowCount at rangeFits
    exact rangeFits.trans_eq countEq
  funext output
  rw [PiDECMatrixInvocation.sum_prepare_value, PiDECEvaluationBatch.range_value]
  apply numericSum_congr 86
  intro index live
  rw [dif_pos live]
  have globalBound : first + index < selectedPlan.rowCount := by omega
  let globalRow : Fin selectedPlan.rowCount := ⟨first + index, globalBound⟩
  have encoded :
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        (Fin.encodeProd (invocation, (⟨index, live⟩ : Fin 86))).val = first + index := by
    dsimp only [first, Fin.encodeProd, Fin.mkDivMod]
    omega
  have sparse := invocation_sparse_value selectedProgram selectedSource
    (parentRead parents child output) blockIndex block selected invocation interface loaded
    ⟨index, live⟩ port
  rw [encoded] at sparse
  have canonical :=
    Poseidon2HashChainV1MatrixRows.compactProgram_row?_eq_structuralPlan_forms globalRow
  change selectedProgram.row? selectedColumns selectedSource (first + index) =
    some (selectedPlan.forms globalRow) at canonical
  rw [canonical, Option.map_some] at sparse
  have rowValue :
      (((PoseidonRetainedRows.rows interface).get
        ⟨index, by rw [PoseidonRetainedRows.rows_length]; exact live⟩).portForm port).evalSparse
          (parentRead parents child output) =
        (selectedPlan.portForm globalRow port).evalSparse
          (parentRead parents child output) := by
    exact Option.some.inj sparse
  rw [rowValue]
  apply congrArg (fun value : F =>
    extensionOps.mul (PiDECEvaluationWeights.weight point (first + index)) (K.embed value))
  rw [PiDECEvaluationFromBlocks.matrixRow, dif_pos globalBound,
    PiDECEvaluationFromBlocks.programForm_value,
    PiDECEvaluationBlockSupport.kernel_eq_evalSparse]
  apply congrArg (SparseForm.evalSparse (selectedPlan.portForm globalRow port))
  funext column
  exact parentRead_value parents child output column

/-- The same guarded parent reader over the existing centered integer cache.
The quotient/remainder addressing and zero branch match the field reader. -/
def intParentRead {count columns : Nat} (parents : Fin count → Vector Int ringDegree)
    (child : Fin productionGlobalParams.k) : Fin ringDegree → Fin columns → F :=
  let forms := PiDECParentSparseRead.prepare ()
  fun output column =>
    if live : column.val / ringDegree < count then
      PiDECParentIntRead.sparseRead forms (parents ⟨column.val / ringDegree, live⟩)
        ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ child output
    else 0

/-- Mapping the actual field parents to centered integers preserves the
complete guarded read. The bound covers all parent lanes, including tails;
no caller supplies a cache-correctness premise. -/
theorem intParentRead_map_valMinAbs {count columns : Nat}
    (parents : Fin count → StoredRing)
    (bounded : ∀ block input,
      centeredMagnitude ((parents block).get input) < Radix.combinedBound)
    (child : Fin productionGlobalParams.k) :
    intParentRead (columns := columns)
        (fun block => (parents block).map
          (fun value => ZMod.valMinAbs (n := goldilocksModulus) value)) child =
      parentRead (columns := columns) parents child := by
  funext output column
  dsimp only [intParentRead, parentRead]
  by_cases live : column.val / ringDegree < count
  · simp only [dif_pos live]
    exact PiDECParentIntRead.sparseRead_map_valMinAbs
      (PiDECParentSparseRead.prepare ()) (parents ⟨column.val / ringDegree, live⟩)
      (bounded ⟨column.val / ringDegree, live⟩)
      ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩ child output
  · simp only [dif_neg live]

/-- The integer-parent invocation computes the same selected canonical range.
Selection, interface loading and range bounds are unchanged. The additional
strict bound is exactly the existing loader check on every parent coefficient. -/
theorem selectedIntInvocation_eq_range
    (parents : Fin blockCount → StoredRing)
    (bounded : ∀ block input,
      centeredMagnitude ((parents block).get input) < Radix.combinedBound)
    (point : PaperAlgebra.Point)
    (blockIndex : Nat) (block : Poseidon.Block)
    (selected : selectedProgram.blocks[blockIndex]? = some (.poseidon block))
    (invocation : Fin block.invocationCount)
    (interface : PoseidonSboxPlan.Interface selectedColumns)
    (loaded : PiDECPoseidonNumericBlock.loadInvocation? block selectedColumns invocation =
      some interface)
    (rangeFits :
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        (Fin.encodeProd (invocation, (0 : Fin 86))).val + 86 ≤ selectedProgram.rowCount)
    (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    let first :=
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        (Fin.encodeProd (invocation, (0 : Fin 86))).val
    ((PiDECMatrixInvocation.sum first point
      (PiDECMatrixInvocation.prepare
        (intParentRead (fun block => (parents block).map
          (fun value => ZMod.valMinAbs (n := goldilocksModulus) value)) child)
        interface)).get port).toRing =
      ((PiDECEvaluationBatch.range first 86 point
        (PiDECEvaluationFromBlocks.matrixRow (splitBlocks parents) port)).get child).toRing := by
  dsimp only
  rw [intParentRead_map_valMinAbs parents bounded child]
  exact selectedInvocation_eq_range parents point blockIndex block selected
    invocation interface loaded rangeFits child port

theorem sumInvocationParts_eq_range {arity ports children : Nat}
    (first count : Nat) (point : CubePoint K arity)
    (rows : Nat → Vector StoredRing children)
    (parts : Nat → Vector PiRLCPartialTrace.MaterializedRingK ports)
    (port : Fin ports) (child : Fin children) :
    (∀ index, index < count →
      ((parts index).get port).toRing =
        ((PiDECEvaluationBatch.range (first + 86 * index) 86 point rows).get child).toRing) →
    ((PiDECEvaluationBatch.sum count parts).get port).toRing =
      ((PiDECEvaluationBatch.range first (86 * count) point rows).get child).toRing := by
  induction count with
  | zero =>
      intro _
      change ((PiDECEvaluationBatch.zero ports).get port).toRing =
        ((PiDECEvaluationBatch.zero children).get child).toRing
      rw [PiDECEvaluationBatch.zero_value, PiDECEvaluationBatch.zero_value]
  | succ count inductionHypothesis =>
      intro each
      have previous := inductionHypothesis (fun index live =>
        each index (Nat.lt_trans live (Nat.lt_succ_self count)))
      have last := each count (Nat.lt_succ_self count)
      have joined := PiDECEvaluationBatch.range_append first (86 * count) 86
        point rows child
      rw [PiDECEvaluationBatch.add_value] at joined
      have sumStep : PiDECEvaluationBatch.sum (count + 1) parts =
          PiDECEvaluationBatch.add (PiDECEvaluationBatch.sum count parts) (parts count) := by
        simp only [PiDECEvaluationBatch.sum, Nat.fold_succ]
      calc
        _ = ringKAdd ((PiDECEvaluationBatch.sum count parts).get port).toRing
            ((parts count).get port).toRing := by
          rw [sumStep, PiDECEvaluationBatch.add_value]
        _ = ringKAdd
            ((PiDECEvaluationBatch.range first (86 * count) point rows).get child).toRing
            ((PiDECEvaluationBatch.range (first + 86 * count) 86 point rows).get child).toRing := by
          rw [previous, last]
        _ = ((PiDECEvaluationBatch.range first (86 * count + 86) point rows).get child).toRing :=
          joined.symm
        _ = _ := by rw [Nat.mul_succ]

/-- The ordered integer-parent invocation range is exactly the corresponding
canonical matrix range for every child and port. The loaded interfaces must
be those selected by the existing program at firstInvocation + index. All
parent lanes retain the same strict bound as the single-invocation runner.
No expected value or caller source/read agreement is assumed. -/
theorem selectedIntInvocationRange_eq_range
    (parents : Fin blockCount → StoredRing)
    (bounded : ∀ block input,
      centeredMagnitude ((parents block).get input) < Radix.combinedBound)
    (point : PaperAlgebra.Point) (blockIndex : Nat) (block : Poseidon.Block)
    (selected : selectedProgram.blocks[blockIndex]? = some (.poseidon block))
    (firstInvocation count : Nat)
    (invocationsFit : firstInvocation + count ≤ block.invocationCount)
    (interfaces : Vector (PoseidonSboxPlan.Interface selectedColumns) count)
    (loaded : ∀ index : Fin count,
      PiDECPoseidonNumericBlock.loadInvocation? block selectedColumns
        ⟨firstInvocation + index.val, by omega⟩ = some (interfaces.get index))
    (rangeFits :
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        86 * firstInvocation + 86 * count ≤ selectedProgram.rowCount)
    (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    let first :=
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        86 * firstInvocation
    ((PiDECMatrixInvocationRange.sum first point
      (intParentRead (fun block => (parents block).map
        (fun value => ZMod.valMinAbs (n := goldilocksModulus) value)) child)
      interfaces).get port).toRing =
      ((PiDECEvaluationBatch.range first (86 * count) point
        (PiDECEvaluationFromBlocks.matrixRow (splitBlocks parents) port)).get child).toRing := by
  dsimp only
  let first :=
    ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      86 * firstInvocation
  change ((PiDECMatrixInvocationRange.sum first point
    (intParentRead (fun block => (parents block).map
      (fun value => ZMod.valMinAbs (n := goldilocksModulus) value)) child)
    interfaces).get port).toRing = _
  unfold PiDECMatrixInvocationRange.sum
  apply sumInvocationParts_eq_range first count point _ _ port child
  intro index live
  rw [dif_pos live]
  let invocation : Fin block.invocationCount := ⟨firstInvocation + index, by omega⟩
  have loadedIndex :
      PiDECPoseidonNumericBlock.loadInvocation? block selectedColumns invocation =
        some (interfaces.get ⟨index, live⟩) :=
    loaded ⟨index, live⟩
  have startEq :
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        (Fin.encodeProd (invocation, (0 : Fin 86))).val = first + 86 * index := by
    dsimp only [invocation, first, Fin.encodeProd, Fin.mkDivMod]
    omega
  have lastFits : first + 86 * index + 86 ≤ selectedProgram.rowCount := by
    change first + 86 * count ≤ selectedProgram.rowCount at rangeFits
    omega
  have invocationFits :
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        (Fin.encodeProd (invocation, (0 : Fin 86))).val + 86 ≤ selectedProgram.rowCount := by
    rw [startEq]
    exact lastFits
  have single := selectedIntInvocation_eq_range parents bounded point blockIndex block
    selected invocation (interfaces.get ⟨index, live⟩) loadedIndex invocationFits child port
  dsimp only at single
  rw [startEq] at single
  exact single

/-- A contiguous vector loaded by the exact canonical program yields its
complete canonical weighted range for every child and port. All parent
lanes retain the existing strict bound, including carried tails. The only
row premise records successful canonical loads; no expected matrix values
or caller read/source agreement are accepted. -/
theorem selectedIntSparseRange_eq_range
    (parents : Fin blockCount → StoredRing)
    (bounded : ∀ block input,
      centeredMagnitude ((parents block).get input) < Radix.combinedBound)
    (point : PaperAlgebra.Point) (firstRow count : Nat)
    (forms : Vector (MatrixProgram.RowForms selectedColumns) count)
    (loaded : ∀ index : Fin count,
      selectedProgram.row? selectedColumns selectedSource (firstRow + index.val) =
        some (forms.get index))
    (rangeFits : firstRow + count ≤ selectedProgram.rowCount)
    (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    ((PiDECMatrixSparseRange.sum firstRow point
      (intParentRead (fun block => (parents block).map
        (fun value => ZMod.valMinAbs (n := goldilocksModulus) value)) child)
      forms).get port).toRing =
      ((PiDECEvaluationBatch.range firstRow count point
        (PiDECEvaluationFromBlocks.matrixRow (splitBlocks parents) port)).get child).toRing := by
  rw [intParentRead_map_valMinAbs parents bounded child]
  have countEq : selectedProgram.rowCount = selectedPlan.rowCount :=
    PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
  have covered : firstRow + count ≤ selectedPlan.rowCount :=
    rangeFits.trans_eq countEq
  funext output
  rw [PiDECMatrixSparseRange.sum_value, PiDECEvaluationBatch.range_value]
  apply numericSum_congr count
  intro index live
  rw [dif_pos live]
  have globalBound : firstRow + index < selectedPlan.rowCount := by omega
  let globalRow : Fin selectedPlan.rowCount := ⟨firstRow + index, globalBound⟩
  have canonical :=
    Poseidon2HashChainV1MatrixRows.compactProgram_row?_eq_structuralPlan_forms globalRow
  change selectedProgram.row? selectedColumns selectedSource (firstRow + index) =
    some (selectedPlan.forms globalRow) at canonical
  have formsEq : forms.get ⟨index, live⟩ = selectedPlan.forms globalRow :=
    Option.some.inj ((loaded ⟨index, live⟩).symm.trans canonical)
  rw [formsEq]
  change extensionOps.mul (PiDECEvaluationWeights.weight point (firstRow + index))
      (K.embed ((selectedPlan.portForm globalRow port).evalSparse
        (parentRead parents child output))) = _
  apply congrArg (fun value : F =>
    extensionOps.mul (PiDECEvaluationWeights.weight point (firstRow + index)) (K.embed value))
  rw [PiDECEvaluationFromBlocks.matrixRow, dif_pos globalBound,
    PiDECEvaluationFromBlocks.programForm_value,
    PiDECEvaluationBlockSupport.kernel_eq_evalSparse]
  apply congrArg (SparseForm.evalSparse (selectedPlan.portForm globalRow port))
  funext column
  exact parentRead_value parents child output column

/-- Loading a selected block row through the canonical cache is exactly the
canonical program load at its global offset. Both some and none results
are preserved; no row or source agreement is supplied by the caller. -/
theorem selectedCachedBlockRow_eq_program
    (blockIndex : Nat) (block : MatrixProgram.Block)
    (selected : selectedProgram.blocks[blockIndex]? = some block)
    (localOrdinal : Nat) (live : localOrdinal < block.rowCount) :
    block.row? selectedColumns
        (fun source => (PiDECCanonicalSourceCache.stored
          Poseidon2HashChainV1Package.application)[source]?) localOrdinal =
      selectedProgram.row? selectedColumns selectedSource
        (((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
          localOrdinal) := by
  have sourceEq :
      (fun source => (PiDECCanonicalSourceCache.stored
        Poseidon2HashChainV1Package.application)[source]?) = selectedSource := by
    funext source
    exact PiDECCanonicalSourceCache.stored_value
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits source
  calc
    _ = block.row? selectedColumns selectedSource localOrdinal :=
      congrArg (fun source : Nat → Option R1CS.Row =>
        block.row? selectedColumns source localOrdinal) sourceEq
    _ = _ := (program_selected_row selectedProgram selectedColumns selectedSource
      blockIndex block selected localOrdinal live).symm

/-- A bounded slice of the already loaded invocation vector computes its
canonical subrange. Slice loads come from the original full-vector loads;
no new interface or value-agreement premise is supplied. -/
theorem selectedIntInvocationSlice_eq_range
    (parents : Fin blockCount → StoredRing)
    (bounded : ∀ block input,
      centeredMagnitude ((parents block).get input) < Radix.combinedBound)
    (point : PaperAlgebra.Point) (blockIndex : Nat) (block : Poseidon.Block)
    (selected : selectedProgram.blocks[blockIndex]? = some (.poseidon block))
    (firstInvocation count : Nat)
    (invocationsFit : firstInvocation + count ≤ block.invocationCount)
    (interfaces : Vector (PoseidonSboxPlan.Interface selectedColumns) count)
    (loaded : ∀ index : Fin count,
      PiDECPoseidonNumericBlock.loadInvocation? block selectedColumns
        ⟨firstInvocation + index.val, by omega⟩ = some (interfaces.get index))
    (rangeFits :
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        86 * firstInvocation + 86 * count ≤ selectedProgram.rowCount)
    (lo hi : Nat) (lo_le_hi : lo ≤ hi) (hi_le_count : hi ≤ count)
    (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    let first :=
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        86 * firstInvocation
    ((PiDECMatrixInvocationRange.sum (first + 86 * lo) point
      (intParentRead (fun block => (parents block).map
        (fun value => ZMod.valMinAbs (n := goldilocksModulus) value)) child)
      (interfaces.extract lo hi)).get port).toRing =
      ((PiDECEvaluationBatch.range (first + 86 * lo) (86 * (hi - lo)) point
        (PiDECEvaluationFromBlocks.matrixRow (splitBlocks parents) port)).get child).toRing := by
  have sliceFits : firstInvocation + lo + (min hi count - lo) ≤ block.invocationCount := by
    rw [Nat.min_eq_left hi_le_count]
    omega
  have sliceLoaded : ∀ index : Fin (min hi count - lo),
      PiDECPoseidonNumericBlock.loadInvocation? block selectedColumns
        ⟨firstInvocation + lo + index.val, by have := index.isLt; omega⟩ =
          some ((interfaces.extract lo hi).get index) := by
    intro index
    have localBound : index.val < hi - lo := by
      simpa only [Nat.min_eq_left hi_le_count] using index.isLt
    have fullBound : lo + index.val < count := by omega
    change PiDECPoseidonNumericBlock.loadInvocation? block selectedColumns
        ⟨firstInvocation + lo + index.val, by omega⟩ =
      some ((interfaces.extract lo hi)[index.val])
    rw [Vector.getElem_extract]
    simpa only [Nat.add_assoc, Vector.get] using loaded ⟨lo + index.val, fullBound⟩
  have sliceRangeFits :
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        86 * (firstInvocation + lo) + 86 * (min hi count - lo) ≤
          selectedProgram.rowCount := by
    rw [Nat.min_eq_left hi_le_count]
    omega
  have result := selectedIntInvocationRange_eq_range parents bounded point blockIndex block
    selected (firstInvocation + lo) (min hi count - lo) sliceFits
    (interfaces.extract lo hi) sliceLoaded sliceRangeFits child port
  dsimp only at result ⊢
  simpa only [Nat.min_eq_left hi_le_count, Nat.mul_add, Nat.add_assoc] using result

/-- A bounded slice of the already loaded sparse rows computes the matching
canonical subrange. The source, parent bounds and full-vector load authority
are unchanged; the slice introduces only its explicit index bounds. -/
theorem selectedIntSparseSlice_eq_range
    (parents : Fin blockCount → StoredRing)
    (bounded : ∀ block input,
      centeredMagnitude ((parents block).get input) < Radix.combinedBound)
    (point : PaperAlgebra.Point) (firstRow count : Nat)
    (forms : Vector (MatrixProgram.RowForms selectedColumns) count)
    (loaded : ∀ index : Fin count,
      selectedProgram.row? selectedColumns selectedSource (firstRow + index.val) =
        some (forms.get index))
    (rangeFits : firstRow + count ≤ selectedProgram.rowCount)
    (lo hi : Nat) (lo_le_hi : lo ≤ hi) (hi_le_count : hi ≤ count)
    (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    ((PiDECMatrixSparseRange.sum (firstRow + lo) point
      (intParentRead (fun block => (parents block).map
        (fun value => ZMod.valMinAbs (n := goldilocksModulus) value)) child)
      (forms.extract lo hi)).get port).toRing =
      ((PiDECEvaluationBatch.range (firstRow + lo) (hi - lo) point
        (PiDECEvaluationFromBlocks.matrixRow (splitBlocks parents) port)).get child).toRing := by
  have sliceLoaded : ∀ index : Fin (min hi count - lo),
      selectedProgram.row? selectedColumns selectedSource (firstRow + lo + index.val) =
        some ((forms.extract lo hi).get index) := by
    intro index
    have localBound : index.val < hi - lo := by
      simpa only [Nat.min_eq_left hi_le_count] using index.isLt
    have fullBound : lo + index.val < count := by omega
    change selectedProgram.row? selectedColumns selectedSource
        (firstRow + lo + index.val) = some ((forms.extract lo hi)[index.val])
    rw [Vector.getElem_extract]
    simpa only [Nat.add_assoc, Vector.get] using loaded ⟨lo + index.val, fullBound⟩
  have sliceRangeFits : firstRow + lo + (min hi count - lo) ≤ selectedProgram.rowCount := by
    rw [Nat.min_eq_left hi_le_count]
    omega
  have result := selectedIntSparseRange_eq_range parents bounded point
    (firstRow + lo) (min hi count - lo) (forms.extract lo hi) sliceLoaded sliceRangeFits
    child port
  simpa only [Nat.min_eq_left hi_le_count] using result

end NightstreamFPrime.Export.Stage1.PiDECMatrixSelectedBatch
