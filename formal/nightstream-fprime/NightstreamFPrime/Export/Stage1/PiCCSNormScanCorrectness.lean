import NightstreamFPrime.Export.Stage1.PiCCSNormScan

/-! Exact coefficient sums for the checked original-mask norm scan. Every
complete block contributes all 27 adjacent pairs and all 17 original sources.
The source sum is linked to the existing nonlinear-message norm constructor. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormScanCorrectness

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open PiCCSNormSource

/-- The existing source-weighted norm cubics at one decoded adjacent pair. -/
def pairNorm (powers : Nat → K) (masks : Array (Nat × Nat)) (pair : Fin PairCount) :
    FixedPolynomial K 3 :=
  FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
    fun source => FixedPolynomial.scale extensionOps.toOps (powers source.val)
      (PiCCSFirstRoundPair.normPair extensionOps
        (K.embed (PiCCSNormCache.signedValue (sourceCode masks source (lowLane pair))))
        (K.embed (PiCCSNormCache.signedValue (sourceCode masks source (highLane pair)))))

/-- Canonical pair order, with the same scalar weight used by the scan. -/
def blockNorm (powers weight : Nat → K) (index : Nat) (masks : Array (Nat × Nat)) :
    FixedPolynomial K 3 :=
  FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices PairCount) fun pair =>
    FixedPolynomial.scale extensionOps.toOps (weight (pairIndex index pair))
      (pairNorm powers masks pair)

private theorem finish_list_fold {Index : Type} (powers : Nat → K)
    (step : PiCCSNormBuckets.Buckets → Index → PiCCSNormBuckets.Buckets)
    (term : Index → FixedPolynomial K 3)
    (effect : ∀ initial index, PiCCSNormBuckets.finish powers (step initial index) =
      FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers initial) (term index))
    (indices : List Index) (initial : PiCCSNormBuckets.Buckets) :
    PiCCSNormBuckets.finish powers (indices.foldl step initial) =
      FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers initial)
        (FixedPolynomial.sum extensionOps.toOps indices term) := by
  induction indices generalizing initial with
  | nil => exact (PiCCSPolynomialRange.add_zero extensionOps extensionLaws _).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis, effect]
      exact PiCCSPolynomialRange.add_assoc extensionOps extensionLaws _ _ _

private theorem finish_fin_fold (powers : Nat → K) {count : Nat}
    (step : PiCCSNormBuckets.Buckets → Fin count → PiCCSNormBuckets.Buckets)
    (term : Fin count → FixedPolynomial K 3)
    (effect : ∀ initial index, PiCCSNormBuckets.finish powers (step initial index) =
      FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers initial) (term index))
    (initial : PiCCSNormBuckets.Buckets) :
    PiCCSNormBuckets.finish powers
        (Nat.fold count (fun index inside current => step current ⟨index, inside⟩) initial) =
      FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers initial)
        (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices count) term) := by
  rw [Nat.fold_eq_finRange_foldl]
  exact finish_list_fold powers step term effect (canonicalFinIndices count) initial

private theorem sourceCode_empty (source : Fin productionShape.sourceCount)
    (lane : Fin ringDegree) : sourceCode #[] source lane = ⟨1, by decide⟩ := by
  simp [sourceCode]

private theorem pairNorm_empty (powers : Nat → K) (pair : Fin PairCount) :
    pairNorm powers #[] pair = FixedPolynomial.zero extensionOps.toOps 3 := by
  apply PiCCSPolynomialRange.coefficient_ext
  intro index
  simp only [pairNorm, PiCCSPolynomialRange.coefficient_sum, sourceCode_empty,
    ← PiCCSNormCache.pairTable_value, PiCCSNormBuckets.pairTable_diagonal,
    PiCCSPolynomialRange.scale_zero_polynomial extensionOps extensionLaws,
    PiCCSPolynomialRange.coefficient_zero,
    FiniteSumAlgebra.sumMap_zero extensionOps extensionLaws]

private theorem blockNorm_empty (powers weight : Nat → K) (index : Nat) :
    blockNorm powers weight index #[] = FixedPolynomial.zero extensionOps.toOps 3 := by
  apply PiCCSPolynomialRange.coefficient_ext
  intro coefficient
  simp only [blockNorm, PiCCSPolynomialRange.coefficient_sum, pairNorm_empty,
    PiCCSPolynomialRange.scale_zero_polynomial extensionOps extensionLaws,
    PiCCSPolynomialRange.coefficient_zero,
    FiniteSumAlgebra.sumMap_zero extensionOps extensionLaws]

private theorem finish_source_fold (powers : Nat → K) (weight : K)
    (masks : Array (Nat × Nat)) (pair : Fin PairCount)
    (initial : PiCCSNormBuckets.Buckets) :
    PiCCSNormBuckets.finish powers
        (Nat.fold productionShape.sourceCount (fun source inside current =>
          PiCCSNormBuckets.add current ⟨source, inside⟩
            (sourceCode masks ⟨source, inside⟩ (lowLane pair))
            (sourceCode masks ⟨source, inside⟩ (highLane pair)) weight) initial) =
      FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers initial)
        (FixedPolynomial.scale extensionOps.toOps weight (pairNorm powers masks pair)) := by
  let sourceStep : PiCCSNormBuckets.Buckets → Fin productionShape.sourceCount →
      PiCCSNormBuckets.Buckets := fun current source =>
    PiCCSNormBuckets.add current source
      (sourceCode masks source (lowLane pair)) (sourceCode masks source (highLane pair)) weight
  let sourceTerm : Fin productionShape.sourceCount → FixedPolynomial K 3 := fun source =>
    FixedPolynomial.scale extensionOps.toOps weight
      (PiCCSNormCache.weightedLookup (PiCCSNormCache.prepare powers) source
        (sourceCode masks source (lowLane pair)) (sourceCode masks source (highLane pair)))
  have sourceEffect : ∀ current source, PiCCSNormBuckets.finish powers (sourceStep current source) =
      FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers current) (sourceTerm source) :=
    fun current source => PiCCSNormBuckets.finish_add powers current source
      (sourceCode masks source (lowLane pair)) (sourceCode masks source (highLane pair)) weight
  have folded := finish_fin_fold powers (count := productionShape.sourceCount)
    sourceStep sourceTerm sourceEffect initial
  have sourceTerm_eq : sourceTerm = (fun source : Fin productionShape.sourceCount =>
      FixedPolynomial.scale extensionOps.toOps weight
        (FixedPolynomial.scale extensionOps.toOps (powers source.val)
          (PiCCSFirstRoundPair.normPair extensionOps
            (K.embed (PiCCSNormCache.signedValue (sourceCode masks source (lowLane pair))))
            (K.embed (PiCCSNormCache.signedValue (sourceCode masks source (highLane pair))))))) := by
    funext source
    change FixedPolynomial.scale extensionOps.toOps weight
        (PiCCSNormCache.weightedLookup (PiCCSNormCache.prepare powers) source
          (sourceCode masks source (lowLane pair)) (sourceCode masks source (highLane pair))) = _
    rw [PiCCSNormCache.weightedLookup_prepare]
  have sumEquality :
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount) sourceTerm =
        FixedPolynomial.scale extensionOps.toOps weight (pairNorm powers masks pair) := by
    calc
      _ = FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
          (fun source : Fin productionShape.sourceCount =>
            FixedPolynomial.scale extensionOps.toOps weight
              (FixedPolynomial.scale extensionOps.toOps (powers source.val)
                (PiCCSFirstRoundPair.normPair extensionOps
                  (K.embed (PiCCSNormCache.signedValue (sourceCode masks source (lowLane pair))))
                  (K.embed (PiCCSNormCache.signedValue (sourceCode masks source (highLane pair))))))) :=
        congrArg (FixedPolynomial.sum extensionOps.toOps
          (canonicalFinIndices productionShape.sourceCount)) sourceTerm_eq
      _ = _ :=
        (PiCCSPolynomialRange.scale_sum extensionOps extensionLaws weight
          (canonicalFinIndices productionShape.sourceCount)
          (fun source : Fin productionShape.sourceCount =>
            FixedPolynomial.scale extensionOps.toOps (powers source.val)
              (PiCCSFirstRoundPair.normPair extensionOps
                (K.embed (PiCCSNormCache.signedValue (sourceCode masks source (lowLane pair))))
                (K.embed (PiCCSNormCache.signedValue (sourceCode masks source (highLane pair))))))).symm
  exact folded.trans (congrArg
    (FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers initial)) sumEquality)

/-- Finishing one implemented block adds exactly the canonical 27-pair,
17-source coefficient sum to the arbitrary initial buckets. Empty-mask
omission is justified by its exact zero cubics, with no support premise. -/
theorem block_finish (powers weight : Nat → K) (index : Nat) (masks : Array (Nat × Nat))
    (initial : PiCCSNormBuckets.Buckets) :
    PiCCSNormBuckets.finish powers (PiCCSNormScan.block weight index masks initial) =
      FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers initial)
        (blockNorm powers weight index masks) := by
  by_cases omitted : masks.isEmpty
  · have empty : masks = #[] := Array.empty_of_isEmpty omitted
    subst masks
    simp only [PiCCSNormScan.block, Array.isEmpty_empty, if_true, blockNorm_empty]
    exact (PiCCSPolynomialRange.add_zero extensionOps extensionLaws _).symm
  · let pairStep : PiCCSNormBuckets.Buckets → Fin PairCount → PiCCSNormBuckets.Buckets :=
      fun accumulated pair =>
        Nat.fold productionShape.sourceCount (fun source inside current =>
          PiCCSNormBuckets.add current ⟨source, inside⟩
            (sourceCode masks ⟨source, inside⟩ (lowLane pair))
            (sourceCode masks ⟨source, inside⟩ (highLane pair))
            (weight (pairIndex index pair))) accumulated
    let pairTerm : Fin PairCount → FixedPolynomial K 3 := fun pair =>
      FixedPolynomial.scale extensionOps.toOps (weight (pairIndex index pair))
        (pairNorm powers masks pair)
    have pairEffect : ∀ accumulated pair,
        PiCCSNormBuckets.finish powers (pairStep accumulated pair) =
          FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers accumulated)
            (pairTerm pair) :=
      fun accumulated pair =>
        finish_source_fold powers (weight (pairIndex index pair)) masks pair accumulated
    have folded := finish_fin_fold powers (count := PairCount)
      pairStep pairTerm pairEffect initial
    rw [PiCCSNormScan.block, if_neg omitted]
    change PiCCSNormBuckets.finish powers
        (Nat.fold PairCount (fun pair inside accumulated => pairStep accumulated ⟨pair, inside⟩) initial) =
      FixedPolynomial.add extensionOps.toOps (PiCCSNormBuckets.finish powers initial)
        (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices PairCount) pairTerm)
    exact folded

/-- Every implemented ascending block range gives the exact direct polynomial
range. Missing array entries keep their original empty-mask zero meaning. -/
theorem range_finish (powers weight : Nat → K) (masks : Array (Array (Nat × Nat)))
    (start count : Nat) :
    PiCCSNormBuckets.finish powers (PiCCSNormScan.range weight masks start count) =
      PiCCSPolynomialRange.range extensionOps start count
        (fun index => blockNorm powers weight index (masks[index]?.getD #[])) := by
  induction count with
  | zero => exact PiCCSNormBuckets.finish_empty powers
  | succ count inductionHypothesis =>
      simpa only [PiCCSNormScan.range, Nat.fold_succ, block_finish, PiCCSPolynomialRange.range] using
        congrArg (fun polynomial => FixedPolynomial.add extensionOps.toOps polynomial
          (blockNorm powers weight (start + count) (masks[start + count]?.getD #[]))) inductionHypothesis

/-- This inner sum is the current normPolynomialWithPowers constructor at
its actual canonical endpoint messages. Fresh matrix images are arbitrary
because the norm reads only the original source-assignment fields. -/
theorem pairNorm_eq_normPolynomialWithPowers (powers : Nat → K)
    (masks : Array (Array (Nat × Nat))) (block : Fin PiCCSSourceImages.blockCount)
    (pair : Fin PairCount) (freshLow freshHigh : Vector F Spec.ProductionRelation.matrixCount) :
    pairNorm powers (masks[block.val]?.getD #[]) pair =
      PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers
        (PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
          (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false (pairSuffix block pair)) freshLow)
        (PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
          (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true (pairSuffix block pair)) freshHigh) := by
  let lowMessage : ProtocolPolynomial.OutputMessage K productionShape :=
    PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
      (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false (pairSuffix block pair)) freshLow
  let highMessage : ProtocolPolynomial.OutputMessage K productionShape :=
    PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
      (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true (pairSuffix block pair)) freshHigh
  change FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
      (fun source => FixedPolynomial.scale extensionOps.toOps (powers source.val)
        (PiCCSFirstRoundPair.normPair extensionOps
          (K.embed (PiCCSNormCache.signedValue (sourceCode (masks[block.val]?.getD #[]) source (lowLane pair))))
          (K.embed (PiCCSNormCache.signedValue (sourceCode (masks[block.val]?.getD #[]) source (highLane pair)))))) =
    FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
      (fun source => FixedPolynomial.scale extensionOps.toOps (powers source.val)
        (PiCCSFirstRoundPair.normPair extensionOps
          (lowMessage.sourceAssignment source) (highMessage.sourceAssignment source)))
  exact congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount))
    (funext (fun source =>
      congrArg (FixedPolynomial.scale extensionOps.toOps (powers source.val))
        (congrArg₂ (PiCCSFirstRoundPair.normPair extensionOps)
          (sourceCode_low_sourceAssignment masks source block pair freshLow)
          (sourceCode_high_sourceAssignment masks source block pair freshHigh))))

/-- The canonical outer pair sum also uses the original norm constructor,
with the same pair weights and complete original source order. -/
theorem blockNorm_eq_normPolynomialWithPowers (powers weight : Nat → K)
    (masks : Array (Array (Nat × Nat))) (block : Fin PiCCSSourceImages.blockCount)
    (freshLow freshHigh : Fin PairCount → Vector F Spec.ProductionRelation.matrixCount) :
    blockNorm powers weight block.val (masks[block.val]?.getD #[]) =
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices PairCount) (fun pair =>
        FixedPolynomial.scale extensionOps.toOps (weight (pairIndex block.val pair))
          (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers
            (PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
              (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) false (pairSuffix block pair)) (freshLow pair))
            (PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
              (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) true (pairSuffix block pair)) (freshHigh pair)))) := by
  unfold blockNorm
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices PairCount))
  funext pair
  exact congrArg (FixedPolynomial.scale extensionOps.toOps (weight (pairIndex block.val pair)))
    (pairNorm_eq_normPolynomialWithPowers powers masks block pair (freshLow pair) (freshHigh pair))


end NightstreamFPrime.Export.Stage1.PiCCSNormScanCorrectness
