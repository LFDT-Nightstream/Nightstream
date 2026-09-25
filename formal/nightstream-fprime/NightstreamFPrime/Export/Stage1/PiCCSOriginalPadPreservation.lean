import NightstreamFPrime.Export.Stage1.PiCCSOriginalPad
import NightstreamFPrime.Export.Stage1.PiCCSOriginalReadsPreservation
import NightstreamFPrime.Export.Stage1.PiDECPadBlockRange

/-! Proof-only transport from original-source block products to the complete
Pad evaluation. Fixed-width reference arrays do not enter runtime execution. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSOriginalPad

open NightstreamFPrime.Spec
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism
open NightstreamFPrime.Spec.Phi81Relation.EvaluationHomomorphism.StoredRingArithmetic
  (StoredRing)
open NightstreamFPrime.Spec.Folding.Nifs.StoredAssignmentArithmetic (StoredAssignment)
open NightstreamFPrime.Layout.ProductionRelation
open NightstreamFPrime.Export.Stage1.PiRLCPartialTrace (MaterializedRingK)
open NumericCompletionSum (numericSum)

private theorem get_ofFn {Alpha : Type} {count : Nat}
    (values : Fin count → Alpha) (index : Fin count) :
    (Vector.ofFn values).get index = values index := by
  change (Vector.ofFn values)[index.val] = _
  rw [Vector.getElem_ofFn]

private theorem numericSum_zero (term : Nat → K) :
    numericSum extensionOps 0 term = extensionOps.zero := rfl

private theorem blockValues_get (masks : Array (Array (Nat × Nat))) (block : Nat)
    (source : Fin productionShape.sourceCount) :
    ((blockValues masks block).get source).get =
      PiCCSOriginalReads.blockAt masks source block := by
  funext lane
  rw [blockValues, get_ofFn, get_ofFn]

private theorem weights_get (point : PaperAlgebra.Point) (block : Nat)
    (basis : Fin ringDegree) :
    (weights point block).get basis =
      PiDECEvaluationWeights.weight point (block * ringDegree + basis.val) := by
  rw [weights, get_ofFn]

private theorem blockAt_empty (masks : Array (Array (Nat × Nat))) (block : Nat)
    (source : Fin productionShape.sourceCount)
    (empty : (masks[block]?.getD #[]).isEmpty = true) :
    PiCCSOriginalReads.blockAt masks source block = ringFZero := by
  have absent : masks[block]?.getD #[] = #[] := Array.empty_of_isEmpty empty
  by_cases live : block < PiCCSSourceImages.blockCount
  · funext lane
    change PiCCSOriginalReads.blockAt masks source block lane = 0
    rw [PiCCSOriginalReads.blockAt_lane masks source ⟨block, live⟩ lane]
    simp [SignedUnitSourceInput.scalar, absent]
  · rw [PiCCSOriginalReads.blockAt, PiCCSSourceImages.blockAt, dif_neg live]

private theorem materializedRingK_ext (left right : MaterializedRingK)
    (equal : left.toRing = right.toRing) : left = right := by
  have arrays : left.values = right.values := by
    apply Array.ext (left.size_eq.trans right.size_eq.symm)
    intro index leftBound rightBound
    have live : index < ringDegree := by simpa only [left.size_eq] using leftBound
    exact congrFun equal ⟨index, live⟩
  cases left
  cases right
  cases arrays
  rfl

private theorem numericSum_all_zero (count : Nat) (term : Nat → K)
    (zero : ∀ index, term index = K.zero) :
    numericSum extensionOps count term = K.zero := by
  induction count with
  | zero => rfl
  | succ count ih =>
      change extensionOps.add (numericSum extensionOps count term) (term count) = K.zero
      rw [ih, zero count]
      rfl

private theorem products_eq_weighted (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) (block : Nat) :
    products point masks block =
      PiDECPadWeightedProduct.products (weights point block) (blockValues masks block) := by
  unfold products
  split_ifs with empty
  · apply Vector.ext
    intro source bound
    change (PiDECEvaluationBatch.zero productionShape.sourceCount).get ⟨source, bound⟩ =
      (PiDECPadWeightedProduct.products (weights point block) (blockValues masks block)).get
        ⟨source, bound⟩
    apply materializedRingK_ext
    funext output
    rw [PiDECEvaluationBatch.zero_value, PiDECPadWeightedProduct.products_value]
    change K.zero = _
    symm
    apply numericSum_all_zero
    intro basis
    split_ifs with live
    · rw [blockValues_get, blockAt_empty masks block ⟨source, bound⟩ empty,
        CarrierAction.kernelImage_eq_ringFMul, CarrierAction.ringFMul_zero_right]
      change extensionOps.mul _ extensionOps.zero = extensionOps.zero
      exact extensionLaws.mul_zero _
    · rfl
  · rfl

private theorem products_value (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) (block : Nat)
    (source : Fin productionShape.sourceCount) (output : Fin ringDegree) :
    ((products point masks block).get source).toRing output =
      numericSum extensionOps ringDegree (fun basis =>
        if live : basis < ringDegree then
          K.mul (PiDECEvaluationWeights.weight point (block * ringDegree + basis))
            (K.embed (CarrierAction.kernelImage ⟨basis, live⟩
              (PiCCSOriginalReads.blockAt masks source block) output))
        else K.zero) := by
  rw [products_eq_weighted, PiDECPadWeightedProduct.products_value]
  simp only [weights_get, blockValues_get]

/-- Every returned coefficient is the exact signed-mask Pad sum over the
requested complete blocks, with the original global point weights. -/
theorem range_value (firstBlock count : Nat) (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) (source : Fin productionShape.sourceCount)
    (output : Fin ringDegree) :
    ((range firstBlock count point masks).get source).toRing output =
      numericSum extensionOps count (fun offset =>
        numericSum extensionOps ringDegree (fun basis =>
          if live : basis < ringDegree then
            K.mul (PiDECEvaluationWeights.weight point
                ((firstBlock + offset) * ringDegree + basis))
              (K.embed (CarrierAction.kernelImage ⟨basis, live⟩
                (PiCCSOriginalReads.blockAt masks source (firstBlock + offset)) output))
          else K.zero)) := by
  rw [range, PiDECEvaluationBatch.sum_value]
  simp only [products_value]

private theorem numericSum_succ (count : Nat) (term : Nat → K) :
    numericSum extensionOps (count + 1) term =
      extensionOps.add (numericSum extensionOps count term) (term count) := by
  simp only [numericSum, Nat.fold_succ]

private theorem numericSum_append (leftCount rightCount : Nat) (term : Nat → K) :
    numericSum extensionOps (leftCount + rightCount) term =
      extensionOps.add (numericSum extensionOps leftCount term)
        (numericSum extensionOps rightCount (fun index => term (leftCount + index))) := by
  induction rightCount with
  | zero =>
      rw [Nat.add_zero, numericSum_zero]
      exact (extensionLaws.add_zero _).symm
  | succ count ih =>
      rw [Nat.add_succ, numericSum_succ, ih, numericSum_succ]
      exact extensionLaws.add_assoc _ _ _

/-- Adding adjacent complete-block ranges preserves every source coefficient,
with no restriction on the masks or the requested block indices. -/
theorem range_append (firstBlock leftCount rightCount : Nat) (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) (source : Fin productionShape.sourceCount) :
    ((range firstBlock (leftCount + rightCount) point masks).get source).toRing =
      ((PiDECEvaluationBatch.add (range firstBlock leftCount point masks)
        (range (firstBlock + leftCount) rightCount point masks)).get source).toRing := by
  funext output
  rw [PiDECEvaluationBatch.add_value]
  change ((range firstBlock (leftCount + rightCount) point masks).get source).toRing output =
    extensionOps.add (((range firstBlock leftCount point masks).get source).toRing output)
      (((range (firstBlock + leftCount) rightCount point masks).get source).toRing output)
  simp only [range, PiDECEvaluationBatch.sum_value]
  rw [numericSum_append]
  simp only [Nat.add_assoc]

private def referenceAssignments (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) :
    Vector (StoredAssignment PiCCSSourceImages.shape.carrierWidth) productionGlobalParams.k :=
  Vector.replicate productionGlobalParams.k
    (Vector.ofFn (PiCCSOriginalReads.assignment masks source))

private theorem referenceAssignments_get (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (child : Fin productionGlobalParams.k) :
    ((referenceAssignments masks source).get child).get =
      PiCCSOriginalReads.assignment masks source := by
  funext column
  change ((Vector.replicate productionGlobalParams.k
    (Vector.ofFn (PiCCSOriginalReads.assignment masks source)))[child.val])[column.val] = _
  rw [Vector.getElem_replicate, Vector.getElem_ofFn]

private def referenceBlocks (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount)
    (block : Fin PiCCSSourceImages.blockCount) :
    Vector StoredRing productionGlobalParams.k :=
  PiDECCommitmentFold.childBlocks (shape := PiCCSSourceImages.shape)
    (referenceAssignments masks source) block

private theorem referenceBlock_value (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount)
    (block : Fin PiCCSSourceImages.blockCount) (child : Fin productionGlobalParams.k) :
    ((referenceBlocks masks source block).get child).get =
      PiCCSOriginalReads.blockAt masks source block.val := by
  rw [referenceBlocks, PiDECCommitmentFold.childBlocks_value, referenceAssignments_get,
    PiCCSOriginalReads.blockAt, PiCCSSourceImages.blockAt, dif_pos block.isLt]
  rfl

private def referenceSlot : Fin productionGlobalParams.k := ⟨0, by decide⟩

private theorem products_eq_reference (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat)))
    (block : Fin PiCCSSourceImages.blockCount) (source : Fin productionShape.sourceCount) :
    ((products point masks block.val).get source).toRing =
      ((PiDECPadBlockRange.products (referenceBlocks masks source) point block).get
        referenceSlot).toRing := by
  funext output
  rw [products_value, PiDECPadBlockRange.products, PiDECPadWeightedProduct.products_value]
  simp only [PiDECPadBlockRange.weights, get_ofFn, referenceBlock_value]

private theorem range_succ (firstBlock count : Nat) (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) :
    range firstBlock (count + 1) point masks =
      PiDECEvaluationBatch.add (range firstBlock count point masks)
        (products point masks (firstBlock + count)) := by
  simp only [range, PiDECEvaluationBatch.sum, Nat.fold_succ]

private theorem range_eq_reference (firstBlock count : Nat) (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) (source : Fin productionShape.sourceCount) :
    firstBlock + count ≤ PiCCSSourceImages.blockCount →
      ((range firstBlock count point masks).get source).toRing =
        ((PiDECEvaluationBatch.range (firstBlock * ringDegree) (count * ringDegree) point
          (PiDECEvaluationFromBlocks.padRow (referenceBlocks masks source))).get
            referenceSlot).toRing := by
  induction count with
  | zero =>
      intro _
      funext output
      rw [range, PiDECEvaluationBatch.sum_value, PiDECEvaluationBatch.range_value]
      simp only [Nat.zero_mul, numericSum_zero]
  | succ count ih =>
      intro bounded
      have live : firstBlock + count < PiCCSSourceImages.blockCount := by omega
      have prior := ih (by omega)
      calc
        _ = ringKAdd ((range firstBlock count point masks).get source).toRing
            ((products point masks (firstBlock + count)).get source).toRing := by
          rw [range_succ, PiDECEvaluationBatch.add_value]
        _ = ringKAdd
            ((PiDECEvaluationBatch.range (firstBlock * ringDegree) (count * ringDegree) point
              (PiDECEvaluationFromBlocks.padRow (referenceBlocks masks source))).get
                referenceSlot).toRing
            ((PiDECEvaluationBatch.range ((firstBlock + count) * ringDegree) ringDegree point
              (PiDECEvaluationFromBlocks.padRow (referenceBlocks masks source))).get
                referenceSlot).toRing := by
          rw [prior, products_eq_reference point masks ⟨firstBlock + count, live⟩ source,
            PiDECPadBlockRange.products_eq_range]
        _ = ((PiDECEvaluationBatch.range (firstBlock * ringDegree)
              (count * ringDegree + ringDegree) point
              (PiDECEvaluationFromBlocks.padRow (referenceBlocks masks source))).get
                referenceSlot).toRing := by
          simpa only [PiDECEvaluationBatch.add_value, Nat.add_mul] using
            (PiDECEvaluationBatch.range_append (firstBlock * ringDegree)
              (count * ringDegree) ringDegree point
              (PiDECEvaluationFromBlocks.padRow (referenceBlocks masks source))
              referenceSlot).symm
        _ = _ := by rw [Nat.add_mul, Nat.one_mul]

private theorem referenceRow_value (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (index : Nat) (output : Fin ringDegree) :
    ((PiDECEvaluationFromBlocks.padRow (referenceBlocks masks source) index).get
      referenceSlot).get output =
      if live : index < PiCCSSourceImages.shape.carrierWidth then
        PiCCSSourceImages.kernelRead (PiCCSOriginalReads.assignment masks source)
          output (⟨index, live⟩ : Fin PiCCSSourceImages.shape.carrierWidth)
      else 0 := by
  by_cases live : index < PiCCSSourceImages.shape.carrierWidth
  · rw [PiDECEvaluationFromBlocks.padRow, dif_pos live, dif_pos live,
      PiDECEvaluationBlockSupport.kernel_eq_evalSparse]
    simp only [SparseForm.evalSparse, SparseForm.singleton,
      List.foldl_cons, List.foldl_nil, Fin.zero_add, Fin.one_mul]
    have blockLive : index / ringDegree < PiCCSSourceImages.blockCount :=
      (Phi81ColumnLayout.decode (⟨index, live⟩ : Fin PiCCSSourceImages.shape.carrierWidth)).1.isLt
    rw [PiDECEvaluationFromBlocks.blockAt, dif_pos blockLive, referenceBlock_value]
    rfl
  · rw [PiDECEvaluationFromBlocks.padRow, dif_neg live, dif_neg live]
    change ((Vector.replicate productionGlobalParams.k PiDECCommitmentFold.zero)[referenceSlot.val]).get
      output = 0
    rw [Vector.getElem_replicate, PiDECCommitmentFold.zero_value]
    rfl

/-- Flattening complete blocks preserves the ordinary scalar Pad row sum.
The bound states exact containment in the selected complete carrier. -/
theorem range_eq_rows (firstBlock count : Nat) (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) (source : Fin productionShape.sourceCount)
    (output : Fin ringDegree)
    (bounded : firstBlock + count ≤ PiCCSSourceImages.blockCount) :
    ((range firstBlock count point masks).get source).toRing output =
      numericSum extensionOps (count * ringDegree) (fun index =>
        K.mul (PiDECEvaluationWeights.weight point (firstBlock * ringDegree + index))
          (K.embed (if live : firstBlock * ringDegree + index <
                PiCCSSourceImages.shape.carrierWidth then
            PiCCSSourceImages.kernelRead (PiCCSOriginalReads.assignment masks source)
              output (⟨firstBlock * ringDegree + index, live⟩ :
                Fin PiCCSSourceImages.shape.carrierWidth)
          else 0))) := by
  rw [range_eq_reference firstBlock count point masks source bounded,
    PiDECEvaluationBatch.range_value]
  apply congrArg (numericSum extensionOps (count * ringDegree))
  funext index
  rw [referenceRow_value]
  rfl

private theorem carrierWidth_eq_blocks_of_shape (shape : Phi81Relation.Shape) :
    shape.carrierWidth = Phi81ColumnLayout.blockCount shape.carrierWidth * ringDegree := by
  change Phi81CarrierLayout.carrierWidth shape.logicalWidth =
    Phi81ColumnLayout.blockCount
      (Phi81CarrierLayout.carrierWidth shape.logicalWidth) * ringDegree
  rw [Phi81CarrierLayout.blockCount_carrierWidth]
  exact Phi81CarrierLayout.carrierWidth_eq shape.logicalWidth

private abbrev carrierWidth_eq_blocks :=
  carrierWidth_eq_blocks_of_shape PiCCSSourceImages.shape

/-- The full original-source block range is the complete paper Pad family.
This holds for every source, including fresh source 0 and all carrier tails. -/
theorem complete_eq_evaluationFamily (point : PaperAlgebra.Point)
    (masks : Array (Array (Nat × Nat))) (source : Fin productionShape.sourceCount) :
    ((range 0 PiCCSSourceImages.blockCount point masks).get source).toRing =
      (PaperAlgebra.evaluationFamily
        (Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation)
        (PiCCSOriginalReads.assignment masks source) point).pad := by
  calc
    _ = ((PiDECEvaluationBatch.accumulate PiCCSSourceImages.shape.carrierWidth point
          (PiDECEvaluationFromBlocks.padRow (referenceBlocks masks source))).get
            referenceSlot).toRing := by
      rw [range_eq_reference 0 PiCCSSourceImages.blockCount point masks source
        (by omega), Nat.zero_mul, ← carrierWidth_eq_blocks]
      rfl
    _ = ((PiDECEvaluationFromBlocks.familyFromBlocks
          (referenceBlocks masks source) point).get referenceSlot).pad := by
      rw [PiDECEvaluationFromBlocks.familyFromBlocks, get_ofFn]
    _ = (PiDECEvaluationHonestMessages.family
          (referenceAssignments masks source) point referenceSlot).pad := by
      rw [PiDECEvaluationFromBlocks.familyFromBlocks_eq_family]
      apply congrArg (fun assignments =>
        (PiDECEvaluationHonestMessages.family assignments point referenceSlot).pad)
      exact PiDECEvaluationFromBlocks.reference_childBlocks (referenceAssignments masks source)
    _ = _ := by
      rw [PiDECEvaluationHonestMessages.family_eq_evaluationFamily]
      change (PaperAlgebra.evaluationFamily
        (Lifecycle.PiRLC.v1_1.InputBinding.relationSource PiDECInputCheck.relation)
        ((referenceAssignments masks source).get referenceSlot).get point).pad = _
      rw [referenceAssignments_get]

end NightstreamFPrime.Export.Stage1.PiCCSOriginalPad
