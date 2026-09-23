import NightstreamFPrime.Export.Stage1.PiDECPadWeightedProduct
import NightstreamFPrime.Export.Stage1.PiDECEvaluationFromBlocks

/-!
The weighted Pad product equals the existing complete-block row range.
Consecutive block products can be summed in Lean to recover the full Pad
accumulator. All bounds come from the complete carrier and ring degree.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiDECPadBlockRange

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.MatrixCoefficientSource
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Lifecycle
open NumericCompletionSum (numericSum)

private abbrev selectedShape :=
  PaperAlgebra.FullShape
    (PerApplicationFixedPoint.logicalWidth Poseidon2HashChainV1Package.application)
    (PerApplicationFixedPoint.publicFits Poseidon2HashChainV1Package.application)

private abbrev blockCount := Phi81ColumnLayout.blockCount selectedShape.carrierWidth

private theorem carrierWidth_eq_blocks_of_shape (shape : Phi81Relation.Shape) :
    shape.carrierWidth = Phi81ColumnLayout.blockCount shape.carrierWidth * ringDegree := by
  change Phi81CarrierLayout.carrierWidth shape.logicalWidth =
    Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth shape.logicalWidth) * ringDegree
  rw [Phi81CarrierLayout.blockCount_carrierWidth]
  exact Phi81CarrierLayout.carrierWidth_eq shape.logicalWidth

private abbrev carrierWidth_eq_blocks := carrierWidth_eq_blocks_of_shape selectedShape

private theorem singleton_rowBlock {columns : Nat} (column : Fin columns)
    (block : Nat) (children : Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree) :
    ((PiDECEvaluationBlock.rowBlock (SparseForm.singleton column 1)
      block children).get child).get output =
      if column.val / ringDegree = block then
        CarrierAction.kernelImage
          ⟨column.val % ringDegree, Nat.mod_lt _ (by decide)⟩
          (children.get child).get output
      else 0 := by
  rw [PiDECEvaluationBlock.rowBlock_value]
  simp only [SparseForm.evalSparse, SparseForm.singleton,
    List.foldl_cons, List.foldl_nil, Fin.zero_add, Fin.one_mul]

private theorem singleton_kernel
    (column : Fin selectedShape.carrierWidth)
    (children : Nat → Vector StoredRing productionGlobalParams.k)
    (child : Fin productionGlobalParams.k) (output : Fin ringDegree) :
    ((PiDECEvaluationBlockSupport.kernel (SparseForm.singleton column 1)
      children).get child).get output =
      CarrierAction.kernelImage (Phi81ColumnLayout.decode column).2
        ((children (Phi81ColumnLayout.decode column).1.val).get child).get output := by
  rw [PiDECEvaluationBlockSupport.kernel_eq_fullBlockSum
    (SparseForm.singleton column 1) (Nat.le_of_eq carrierWidth_eq_blocks)]
  have selectedLt : column.val / ringDegree < blockCount :=
    (Phi81ColumnLayout.decode column).1.isLt
  calc
    _ = sumRange baseOps blockCount (fun block =>
        if block = column.val / ringDegree then
          CarrierAction.kernelImage (Phi81ColumnLayout.decode column).2
            ((children block).get child).get output
        else 0) := by
      apply sumRange_congr baseOps blockCount
      intro block _
      rw [singleton_rowBlock]
      simp only [eq_comm, Phi81ColumnLayout.decode]
    _ = _ := sumRange_select baseOps baseLaws blockCount
      (column.val / ringDegree)
      (fun block => CarrierAction.kernelImage (Phi81ColumnLayout.decode column).2
        ((children block).get child).get output) selectedLt

/-- Every Pad position of this complete carrier block selects the same
supplied child block and the exact basis lane. This includes carried tails. -/
theorem padRow_basis
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (block : Fin blockCount) (basis : Fin ringDegree)
    (child : Fin productionGlobalParams.k) :
    ((PiDECEvaluationFromBlocks.padRow blocks
      (block.val * ringDegree + basis.val)).get child).get =
      CarrierAction.kernelImage basis ((blocks block).get child).get := by
  funext output
  have within : block.val * ringDegree + basis.val < selectedShape.carrierWidth :=
    Phi81CarrierLayout.flatIndex_lt_carrierWidth
      (logicalWidth := selectedShape.logicalWidth) block basis
  rw [PiDECEvaluationFromBlocks.padRow, dif_pos within, singleton_kernel]
  change CarrierAction.kernelImage
      (Phi81ColumnLayout.decode
        (Phi81CarrierLayout.carrierColumn (logicalWidth := selectedShape.logicalWidth)
          block basis)).2
      ((PiDECEvaluationFromBlocks.blockAt blocks
        (Phi81ColumnLayout.decode
          (Phi81CarrierLayout.carrierColumn (logicalWidth := selectedShape.logicalWidth)
            block basis)).1.val).get child).get output = _
  rw [Phi81CarrierLayout.decode_carrierColumn,
    PiDECEvaluationFromBlocks.blockAt, dif_pos block.isLt]

/-- The same global point weight as the scalar row loop, at every basis lane. -/
def weights (point : PaperAlgebra.Point) (block : Fin blockCount) : Vector K ringDegree :=
  Vector.ofFn fun basis =>
    PiDECEvaluationWeights.weight point (block.val * ringDegree + basis.val)

private theorem weights_get (point : PaperAlgebra.Point) (block : Fin blockCount)
    (basis : Fin ringDegree) :
    (weights point block).get basis =
      PiDECEvaluationWeights.weight point (block.val * ringDegree + basis.val) := by
  change (Vector.ofFn (fun lane : Fin ringDegree =>
    PiDECEvaluationWeights.weight point (block.val * ringDegree + lane.val)))[basis.val] = _
  rw [Vector.getElem_ofFn]

/-- One block contributes to all children through the two weighted bar keys. -/
def products (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (point : PaperAlgebra.Point) (block : Fin blockCount) :
    Vector MaterializedRingK productionGlobalParams.k :=
  PiDECPadWeightedProduct.products (weights point block) (blocks block)

private theorem numericSum_congr (count : Nat) (left right : Nat → K) :
    (∀ index, index < count → left index = right index) →
      numericSum extensionOps count left = numericSum extensionOps count right := by
  induction count with
  | zero => intro _; rfl
  | succ count inductionHypothesis =>
      intro equal
      change extensionOps.add (numericSum extensionOps count left) (left count) =
        extensionOps.add (numericSum extensionOps count right) (right count)
      rw [inductionHypothesis (fun index live =>
        equal index (Nat.lt_trans live (Nat.lt_succ_self count))),
        equal count (Nat.lt_succ_self count)]

/-- The fast weighted product is exactly the batch row loop over this whole
carrier block. No point, norm, zero-lane or expected-value premise is needed. -/
theorem products_eq_range
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (point : PaperAlgebra.Point) (block : Fin blockCount)
    (child : Fin productionGlobalParams.k) :
    ((products blocks point block).get child).toRing =
      ((PiDECEvaluationBatch.range (block.val * ringDegree) ringDegree point
        (PiDECEvaluationFromBlocks.padRow blocks)).get child).toRing := by
  funext output
  rw [products, PiDECPadWeightedProduct.products_value, PiDECEvaluationBatch.range_value]
  apply numericSum_congr ringDegree
  intro index live
  rw [dif_pos live, weights_get, padRow_basis blocks block ⟨index, live⟩ child]
  rfl

/-- Sum consecutive complete block contributions, with no full witness array.
The proof below applies through the selected carrier block count. -/
def sumPrefix (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (point : PaperAlgebra.Point) (count : Nat) :
    Vector MaterializedRingK productionGlobalParams.k :=
  PiDECEvaluationBatch.sum count fun index =>
    if live : index < blockCount then products blocks point ⟨index, live⟩
    else PiDECEvaluationBatch.zero productionGlobalParams.k

private theorem prefix_succ
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (point : PaperAlgebra.Point) (count : Nat) :
    sumPrefix blocks point (count + 1) =
      PiDECEvaluationBatch.add (sumPrefix blocks point count)
        (if live : count < blockCount then products blocks point ⟨count, live⟩
         else PiDECEvaluationBatch.zero productionGlobalParams.k) := by
  simp only [sumPrefix, PiDECEvaluationBatch.sum, Nat.fold_succ]

/-- The block sumPrefix and the corresponding complete-row sumPrefix agree exactly. -/
theorem prefix_eq_range
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (point : PaperAlgebra.Point) (count : Nat) :
    count ≤ blockCount → ∀ child : Fin productionGlobalParams.k,
      ((sumPrefix blocks point count).get child).toRing =
        ((PiDECEvaluationBatch.range 0 (count * ringDegree) point
          (PiDECEvaluationFromBlocks.padRow blocks)).get child).toRing := by
  induction count with
  | zero =>
      intro _ child
      have zeroSum (term : Nat → K) :
          numericSum extensionOps 0 term = extensionOps.zero := rfl
      funext output
      rw [sumPrefix, PiDECEvaluationBatch.sum_value, PiDECEvaluationBatch.range_value]
      simp only [Nat.zero_mul, zeroSum]
  | succ count inductionHypothesis =>
      intro bounded child
      have live : count < blockCount :=
        Nat.lt_of_lt_of_le (Nat.lt_succ_self count) bounded
      have prior := inductionHypothesis (Nat.le_of_lt live) child
      calc
        _ = ringKAdd ((sumPrefix blocks point count).get child).toRing
            ((products blocks point ⟨count, live⟩).get child).toRing := by
          rw [prefix_succ, dif_pos live, PiDECEvaluationBatch.add_value]
        _ = ringKAdd
            ((PiDECEvaluationBatch.range 0 (count * ringDegree) point
              (PiDECEvaluationFromBlocks.padRow blocks)).get child).toRing
            ((PiDECEvaluationBatch.range (count * ringDegree) ringDegree point
              (PiDECEvaluationFromBlocks.padRow blocks)).get child).toRing := by
          rw [prior, products_eq_range]
        _ = ((PiDECEvaluationBatch.range 0 (count * ringDegree + ringDegree) point
              (PiDECEvaluationFromBlocks.padRow blocks)).get child).toRing := by
          simpa only [PiDECEvaluationBatch.add_value, Nat.zero_add] using
            (PiDECEvaluationBatch.range_append 0 (count * ringDegree) ringDegree
              point (PiDECEvaluationFromBlocks.padRow blocks) child).symm
        _ = _ := by rw [Nat.add_mul, Nat.one_mul]

/-- Summing every complete block recovers the full Pad accumulator. -/
theorem complete_eq_accumulate
    (blocks : Fin blockCount → Vector StoredRing productionGlobalParams.k)
    (point : PaperAlgebra.Point) (child : Fin productionGlobalParams.k) :
    ((sumPrefix blocks point blockCount).get child).toRing =
      ((PiDECEvaluationBatch.accumulate selectedShape.carrierWidth point
        (PiDECEvaluationFromBlocks.padRow blocks)).get child).toRing := by
  have rangeToAccumulator {arity k width count : Nat}
      (equal : width = count) (point : CubePoint K arity)
      (rows : Nat → Vector StoredRing k) (child : Fin k) :
      ((PiDECEvaluationBatch.range 0 count point rows).get child).toRing =
        ((PiDECEvaluationBatch.accumulate width point rows).get child).toRing := by
    rw [PiDECEvaluationBatch.accumulate]
    exact congrArg (fun n : Nat =>
      ((PiDECEvaluationBatch.range 0 n point rows).get child).toRing) equal.symm
  have widthPin : selectedShape.carrierWidth = 149597982 :=
    Poseidon2HashChainV1Setup.carrierWidth_eq
  have blockPin : blockCount = 2770333 := Poseidon2HashChainV1Setup.messageColumns_eq
  have covered : selectedShape.carrierWidth = blockCount * ringDegree := by
    rw [widthPin, blockPin]
    decide
  exact (prefix_eq_range blocks point blockCount (Nat.le_refl blockCount) child).trans
    (@rangeToAccumulator productionShape.cubeVariables productionGlobalParams.k
      selectedShape.carrierWidth (blockCount * ringDegree) covered point
      (PiDECEvaluationFromBlocks.padRow blocks) child)

end NightstreamFPrime.Export.Stage1.PiDECPadBlockRange
