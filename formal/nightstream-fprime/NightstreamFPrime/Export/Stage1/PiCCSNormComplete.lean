import NightstreamFPrime.Export.Stage1.PiCCSNormContribution
import NightstreamFPrime.Export.Stage1.PiCCSNormRangeMerge
import NightstreamFPrime.Export.Stage1.PiCCSGammaPowers

/-! Proof-only closure of the original-mask norm scan. The complete carrier
is grouped into its existing adjacent pairs; the remaining Boolean-domain
pairs are proved zero. No matrix or protocol-data table is evaluated. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormComplete

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open PiCCSNormContribution
open PiCCSNormSource

private abbrev message (masks : Array (Array (Nat × Nat)))
    (suffix : BooleanVertex (cubeVariables - 1)) (bit : Bool)
    (fresh : Vector F ProductionRelation.matrixCount) :=
  PiCCSAggregatedImages.nonlinearMessage (canonicalLayout ()) (assignments masks)
    (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) bit suffix) fresh

/-- The existing pair kernel's norm component at an absolute numeric suffix.
The out-of-domain branch retains the original numeric-pair zero convention. -/
def numericPairNorm (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables)
    (masks : Array (Array (Nat × Nat)))
    (freshLow freshHigh : Nat → Vector F ProductionRelation.matrixCount)
    (index : Nat) : FixedPolynomial K input.sumcheckDegreeBound :=
  if inside : index < 2 ^ (cubeVariables - 1) then
    let suffix := NumericBooleanDomain.vertex (cubeVariables - 1) ⟨index, inside⟩
    normTerm input powers (PiCCSFirstRound.equalitySelector extensionOps suffix alpha)
      (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers
        (message masks suffix false (freshLow index))
        (message masks suffix true (freshHigh index)))
  else FixedPolynomial.zero extensionOps.toOps input.sumcheckDegreeBound

private theorem numericPairNorm_block
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables)
    (masks : Array (Array (Nat × Nat)))
    (freshLow freshHigh : Nat → Vector F ProductionRelation.matrixCount)
    (block : Fin PiCCSSourceImages.blockCount) (pair : Fin PairCount) :
    numericPairNorm input powers alpha masks freshLow freshHigh (pairIndex block.val pair) =
      normTerm input powers (PiCCSFirstRound.equalitySelector extensionOps (pairSuffix block pair) alpha)
        (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers
          (message masks (pairSuffix block pair) false (freshLow (pairIndex block.val pair)))
          (message masks (pairSuffix block pair) true (freshHigh (pairIndex block.val pair)))) := by
  simp only [numericPairNorm, dif_pos (pairIndex_bound block pair), pairSuffix]

private theorem blockTerm_eq_range
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables)
    (masks : Array (Array (Nat × Nat)))
    (freshLow freshHigh : Nat → Vector F ProductionRelation.matrixCount)
    (block : Fin PiCCSSourceImages.blockCount) :
    normTerm input powers (headSelector alpha)
        (PiCCSNormScanCorrectness.blockNorm powers
          (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail)
          block.val (masks[block.val]?.getD #[])) =
      PiCCSPolynomialRange.range extensionOps (block.val * PairCount) PairCount
        (numericPairNorm input powers alpha masks freshLow freshHigh) := by
  rw [blockNorm_normTerms input powers alpha masks block
      (fun pair => freshLow (pairIndex block.val pair))
      (fun pair => freshHigh (pairIndex block.val pair)),
    PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices PairCount))
  funext pair
  simpa only [pairIndex] using
    (numericPairNorm_block input powers alpha masks freshLow freshHigh block pair).symm

private theorem carrier_pairs_at (logicalWidth : Nat) :
    Phi81CarrierLayout.carrierWidth logicalWidth =
      2 * (Phi81ColumnLayout.blockCount (Phi81CarrierLayout.carrierWidth logicalWidth) *
        PairCount) := by
  rw [Phi81CarrierLayout.blockCount_carrierWidth, Phi81CarrierLayout.carrierWidth_eq]
  change Phi81ColumnLayout.blockCount logicalWidth * 54 =
    2 * (Phi81ColumnLayout.blockCount logicalWidth * 27)
  omega

private theorem carrier_pairs :
    PiCCSSourceImages.shape.carrierWidth = 2 * (PiCCSSourceImages.blockCount * PairCount) := by
  simpa only [PiCCSSourceImages.shape, PiCCSSourceImages.blockCount,
    PaperAlgebra.FullShape, PaperAlgebra.fullShape, Phi81Relation.Shape.carrierWidth] using
    carrier_pairs_at PiCCSSourceImages.logicalWidth

private theorem carrier_pairs_le_domain :
    PiCCSSourceImages.blockCount * PairCount ≤ 2 ^ (cubeVariables - 1) := by
  have covered := (canonicalLayout ()).columns_le
  rw [carrier_pairs] at covered
  change 2 * (PiCCSSourceImages.blockCount * PairCount) ≤ 2 ^ 28 at covered
  change PiCCSSourceImages.blockCount * PairCount ≤ 2 ^ 27
  omega

private theorem numericPairNorm_padding
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables)
    (masks : Array (Array (Nat × Nat)))
    (freshLow freshHigh : Nat → Vector F ProductionRelation.matrixCount)
    (index : Nat) (beyond : PiCCSSourceImages.blockCount * PairCount ≤ index)
    (inside : index < 2 ^ (cubeVariables - 1)) :
    numericPairNorm input powers alpha masks freshLow freshHigh index =
      FixedPolynomial.zero extensionOps.toOps input.sumcheckDegreeBound := by
  have padding : PiCCSSourceImages.shape.carrierWidth ≤ 2 * index := by
    rw [carrier_pairs]
    exact Nat.mul_le_mul_left 2 beyond
  let suffix : BooleanVertex (cubeVariables - 1) :=
    NumericBooleanDomain.vertex (cubeVariables - 1) ⟨index, inside⟩
  rw [numericPairNorm, dif_pos inside]
  change normTerm input powers (PiCCSFirstRound.equalitySelector extensionOps suffix alpha)
      (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers
        (message masks suffix false (freshLow index))
        (message masks suffix true (freshHigh index))) = _
  rw [normTerm_innerPair]
  exact normTerm_padding_zero input
    (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail
      (NumericBooleanDomain.index suffix)) powers (headSelector alpha)
    masks suffix (freshLow index) (freshHigh index)
    (by simpa only [suffix, NumericBooleanDomain.index_vertex] using padding)

/-- Finishing the implemented whole-carrier norm scan and applying the
common head factor and gamma shifts gives exactly the full selected numeric
pair-norm sum. Missing masks and the entire padded suffix keep their proved
zero meaning. The result is equality of every retained polynomial coefficient. -/
theorem finished_norm_eq_fullPairSum
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (gamma : K) (alpha : CubePoint K cubeVariables)
    (masks : Array (Array (Nat × Nat)))
    (freshLow freshHigh : Nat → Vector F ProductionRelation.matrixCount) :
    normTerm input (TargetPolynomial.power extensionOps.toOps gamma) (headSelector alpha)
        (PiCCSNormBuckets.finish (TargetPolynomial.power extensionOps.toOps gamma)
          (PiCCSNormScan.range
            (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail)
            masks 0 PiCCSSourceImages.blockCount)) =
      PiCCSPolynomialRange.range extensionOps 0 (2 ^ (cubeVariables - 1))
        (numericPairNorm input (TargetPolynomial.power extensionOps.toOps gamma)
          alpha masks freshLow freshHigh) := by
  let powers := TargetPolynomial.power extensionOps.toOps gamma
  let term := numericPairNorm input powers alpha masks freshLow freshHigh
  change normTerm input powers (headSelector alpha)
      (PiCCSNormBuckets.finish powers (PiCCSNormScan.range
        (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail)
        masks 0 PiCCSSourceImages.blockCount)) =
    PiCCSPolynomialRange.range extensionOps 0 (2 ^ (cubeVariables - 1)) term
  have coveredRange :
      normTerm input powers (headSelector alpha)
          (PiCCSNormBuckets.finish powers (PiCCSNormScan.range
            (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail)
            masks 0 PiCCSSourceImages.blockCount)) =
        PiCCSPolynomialRange.range extensionOps 0 (PiCCSSourceImages.blockCount * PairCount) term := by
    rw [PiCCSNormScanCorrectness.range_finish, normTerm_range]
    calc
      _ = PiCCSPolynomialRange.range extensionOps 0 PiCCSSourceImages.blockCount
          (fun block => PiCCSPolynomialRange.range extensionOps (block * PairCount) PairCount term) := by
        rw [PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws,
          PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws]
        apply congrArg (FixedPolynomial.sum extensionOps.toOps
          (canonicalFinIndices PiCCSSourceImages.blockCount))
        funext block
        simpa only [Nat.zero_add, term] using
          blockTerm_eq_range input powers alpha masks freshLow freshHigh block
      _ = _ := by
        simpa only [Nat.zero_add] using
          (PiCCSPolynomialRange.range_group extensionOps extensionLaws 0
            PiCCSSourceImages.blockCount PairCount term).symm
  have coverage : PiCCSSourceImages.blockCount * PairCount +
      (2 ^ (cubeVariables - 1) - PiCCSSourceImages.blockCount * PairCount) =
        2 ^ (cubeVariables - 1) := by
    have bound := carrier_pairs_le_domain
    omega
  have complete := PiCCSPolynomialRange.range_append_zero extensionOps extensionLaws 0
    (PiCCSSourceImages.blockCount * PairCount)
    (2 ^ (cubeVariables - 1) - PiCCSSourceImages.blockCount * PairCount) term
    (by
      intro index lower upper
      have beyond : PiCCSSourceImages.blockCount * PairCount ≤ index := by
        simpa only [Nat.zero_add] using lower
      have inside : index < 2 ^ (cubeVariables - 1) := by
        simpa only [Nat.zero_add, coverage] using upper
      exact numericPairNorm_padding input powers alpha masks freshLow freshHigh index beyond inside)
  rw [coverage] at complete
  exact coveredRange.trans complete.symm

/-- The runner's prepared powers, prepared tensor weights and ordered worker
merge give the same complete norm sum. The number of workers is arbitrary;
it changes only the partition, not any retained polynomial coefficient. -/
theorem prepared_workers_eq_fullPairSum
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (gamma : K) (alpha : CubePoint K cubeVariables)
    (masks : Array (Array (Nat × Nat)))
    (freshLow freshHigh : Nat → Vector F ProductionRelation.matrixCount)
    (parts : Nat) (positive : 0 < parts) :
    let powers := PiCCSGammaPowers.lookup extensionOps.toOps gamma
      (PiCCSGammaPowers.prepare extensionOps.toOps gamma productionShape.sourceCount)
    let weight := PiCCSTensorWeights.lookup extensionOps alpha.coordinates.tail
      (PiCCSTensorWeights.prepare extensionOps alpha.coordinates.tail)
    normTerm input (TargetPolynomial.power extensionOps.toOps gamma) (headSelector alpha)
        ((Array.ofFn (fun index : Fin parts =>
          PiCCSNormBuckets.finish powers
            (PiCCSNormScan.range weight masks (PiCCSSourceImages.blockCount * index.val / parts)
              (PiCCSSourceImages.blockCount * (index.val + 1) / parts -
                PiCCSSourceImages.blockCount * index.val / parts)))).foldl
          (FixedPolynomial.add extensionOps.toOps) (FixedPolynomial.zero extensionOps.toOps 3)) =
      PiCCSPolynomialRange.range extensionOps 0 (2 ^ (cubeVariables - 1))
        (numericPairNorm input (TargetPolynomial.power extensionOps.toOps gamma)
          alpha masks freshLow freshHigh) := by
  dsimp only
  have powers_eq : PiCCSGammaPowers.lookup extensionOps.toOps gamma
      (PiCCSGammaPowers.prepare extensionOps.toOps gamma productionShape.sourceCount) =
      TargetPolynomial.power extensionOps.toOps gamma := by
    funext exponent
    exact PiCCSGammaPowers.lookup_prepare _ _ _ exponent
  have weight_eq : PiCCSTensorWeights.lookup extensionOps alpha.coordinates.tail
      (PiCCSTensorWeights.prepare extensionOps alpha.coordinates.tail) =
      NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail := by
    funext index
    exact PiCCSTensorWeights.lookup_prepare extensionOps
      (NumericBooleanDomain.WeightProductLaws.ofInterpolationEvaluationLaws extensionLaws) _ index
  rw [powers_eq, weight_eq]
  have merged := PiCCSNormRangeMerge.array_proportional_eq_range
    (TargetPolynomial.power extensionOps.toOps gamma)
    (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail)
    masks 0 PiCCSSourceImages.blockCount parts positive (Nat.zero_le _)
  simp only [Nat.sub_zero, Nat.zero_add] at merged
  rw [merged]
  exact finished_norm_eq_fullPairSum input gamma alpha masks freshLow freshHigh

end NightstreamFPrime.Export.Stage1.PiCCSNormComplete
