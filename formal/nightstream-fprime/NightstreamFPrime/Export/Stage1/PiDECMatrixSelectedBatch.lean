import NightstreamFPrime.Export.Stage1.PiDECParentSparseRead
import NightstreamFPrime.Export.Stage1.PiDECMatrixInvocation
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

private theorem invocation_sparse_value {columns : Nat}
    (program : MatrixProgram.Program) (source : Nat → Option R1CS.Row)
    (read : Fin columns → F) (blockIndex : Nat) (block : Poseidon.Block)
    (selected : program.blocks[blockIndex]? = some (.poseidon block))
    (invocation : Fin block.invocationCount)
    (interface : PoseidonSboxPlan.Interface columns)
    (loaded : PiDECPoseidonNumericBlock.loadInvocation? block columns invocation =
      some interface)
    (row : Fin 94) (port : Fin matrixCount) :
    some ((((PoseidonSboxPlan.rows interface).get
      ⟨row.val, by rw [PoseidonSboxPlan.rows_length]; exact row.isLt⟩).portForm port).evalSparse
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
        (Fin.encodeProd (invocation, (0 : Fin 94))).val + 94 ≤ selectedProgram.rowCount)
    (child : Fin productionGlobalParams.k) (port : Fin matrixCount) :
    let first :=
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        (Fin.encodeProd (invocation, (0 : Fin 94))).val
    ((PiDECMatrixInvocation.sum first point
      (PiDECMatrixInvocation.prepare (parentRead parents child) interface)).get port).toRing =
      ((PiDECEvaluationBatch.range first 94 point
        (PiDECEvaluationFromBlocks.matrixRow (splitBlocks parents) port)).get child).toRing := by
  dsimp only
  let first :=
    ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
      (Fin.encodeProd (invocation, (0 : Fin 94))).val
  change ((PiDECMatrixInvocation.sum first point
    (PiDECMatrixInvocation.prepare (parentRead parents child) interface)).get port).toRing = _
  have countEq : selectedProgram.rowCount = selectedPlan.rowCount :=
    PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
  have covered : first + 94 ≤ selectedPlan.rowCount := by
    change first + 94 ≤ selectedProgram.rowCount at rangeFits
    exact rangeFits.trans_eq countEq
  funext output
  rw [PiDECMatrixInvocation.sum_prepare_value, PiDECEvaluationBatch.range_value]
  apply numericSum_congr 94
  intro index live
  rw [dif_pos live]
  have globalBound : first + index < selectedPlan.rowCount := by omega
  let globalRow : Fin selectedPlan.rowCount := ⟨first + index, globalBound⟩
  have encoded :
      ((selectedProgram.blocks.take blockIndex).map MatrixProgram.Block.rowCount).sum +
        (Fin.encodeProd (invocation, (⟨index, live⟩ : Fin 94))).val = first + index := by
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
      (((PoseidonSboxPlan.rows interface).get
        ⟨index, by rw [PoseidonSboxPlan.rows_length]; exact live⟩).portForm port).evalSparse
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

end NightstreamFPrime.Export.Stage1.PiDECMatrixSelectedBatch
