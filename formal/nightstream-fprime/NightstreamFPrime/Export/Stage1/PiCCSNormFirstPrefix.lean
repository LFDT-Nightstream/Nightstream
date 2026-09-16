import NightstreamFPrime.Export.Stage1.PiCCSNormPrefixSum

/-! The Q1 norm contribution uses the existing one-challenge range kernel.
The algebra is proved with symbolic protocol data. The selected theorem
supplies original-source endpoints last. File and worker IO are separate. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormFirstPrefix

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open PiCCSPolynomialRange
open PiCCSPrefixSelector (headPrefix tailWeight)

private abbrev power (gamma : K) := TargetPolynomial.power extensionOps.toOps gamma
private abbrev code (masks : Array (Array (Nat × Nat)))
    (source : Fin productionShape.sourceCount) (column : Nat) : Fin 3 :=
  PiCCSNormSource.sourceCode (masks[column / ringDegree]?.getD #[]) source
    ⟨column % ringDegree, Nat.mod_lt _ (by decide)⟩

private def firstValues (challenge : K) (codes : Nat → Fin 3) (groups : Nat) : Array K :=
  PiCCSSignedFirstFold.foldOne (PiCCSSignedFirstFold.prepare challenge) (4 * groups) codes

private theorem firstValues_size (challenge : K) (codes : Nat → Fin 3) (groups : Nat) :
    (firstValues challenge codes groups).size = 2 * groups := by
  rw [firstValues, PiCCSSignedFirstFold.foldOne_prepare, PrefixFold.foldOne_size,
    Array.size_ofFn]
  omega

private theorem firstValues_getD (challenge : K) (codes : Nat → Fin 3) (groups index : Nat)
    (inside : index < 2 * groups) :
    (firstValues challenge codes groups).getD index K.zero =
      PrefixFold.interpolate extensionOps challenge
        (K.embed (PiCCSNormCache.signedValue (codes (2 * index))))
        (K.embed (PiCCSNormCache.signedValue (codes (2 * index + 1)))) := by
  have low : 2 * index < 4 * groups := by omega
  have high : 2 * index + 1 < 4 * groups := by omega
  rw [firstValues, PiCCSSignedFirstFold.foldOne_prepare]
  change (PrefixFold.foldOne extensionOps _ challenge).getD index extensionOps.zero = _
  rw [PrefixFold.foldOne_getD extensionOps extensionLaws]
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_ofFn,
    dif_pos low, dif_pos high, Option.getD_some]

/-- The actual one-challenge bucket range in replayNormAfterFirst is the
same direct pair range over the existing SignedFirstFold output. -/
theorem range_eq_sourceRange (challenge : K) (codes : Nat → Fin 3)
    (weight : Nat → K) (groups : Nat) :
    PiCCSPrefixNorm.range challenge codes weight 0 groups =
      PiCCSNormPrefixSum.sourceRange (firstValues challenge codes groups) weight groups := by
  rw [PiCCSPrefixNorm.range_eq_reference]
  unfold PiCCSNormPrefixSum.sourceRange
  rw [PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws,
    PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices groups))
  funext index
  have low : 2 * index.val < 2 * groups := by omega
  have high : 2 * index.val + 1 < 2 * groups := by omega
  have lowLow : 2 * (2 * index.val) = 4 * index.val := by omega
  have highLow : 2 * (2 * index.val + 1) = 4 * index.val + 2 := by omega
  simp only [Nat.zero_add, firstValues_getD challenge codes groups (2 * index.val) low,
    firstValues_getD challenge codes groups (2 * index.val + 1) high,
    lowLow, highLow]

private theorem oneChallenge_endpoint (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (source : Fin productionShape.sourceCount)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount) (groups : Nat)
    (covered : PiCCSSourceImages.shape.carrierWidth ≤ 4 * groups)
    (fits : 4 * groups ≤ 2 ^ cubeVariables)
    (challenge : K) {remaining : Nat}
    (dimension : cubeVariables = [challenge].length + remaining + 1)
    (bit : Bool) (suffix : BooleanVertex remaining) :
    (firstValues challenge (code masks source) groups).getD
        (NumericBooleanDomain.index (.cons bit suffix)) K.zero =
      (ProtocolPolynomial.messageAt extensionOps
        (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPrefixRound.point extensionOps [challenge] dimension
          (if bit then K.one else K.zero) suffix)).sourceAssignment source := by
  have dimension' : cubeVariables = (remaining + 1) + [challenge].length := by omega
  have value := PiCCSNormPrefixSource.decoded_prefix_evaluate input masks source loaded
    (4 * groups) covered fits [challenge] dimension'
    ((BooleanVertex.cons bit suffix).toCubePoint extensionOps)
  rw [SumCheckTruthPath.evaluate_toCubePoint_eq_valueAt extensionOps extensionLaws] at value
  simp only [PrefixFold.zeroExtend, BooleanTable.valueAt_tabulate,
    PrefixFold.foldPrefix] at value
  rw [firstValues, PiCCSSignedFirstFold.foldOne_prepare]
  cases bit <;> simpa only [ProtocolPolynomial.messageAt, PiCCSPrefixRound.point,
    BooleanVertex.toCubePoint_coordinates, BooleanVertex.fieldCoordinates,
    SumCheckTruthPath.VertexEncoding.fieldCoordinates, Bool.false_eq_true,
    if_false, if_true] using value

-- Symbolic data keeps all coefficient tactics outside the selected 2^28 table.
private theorem inner_pair_of_endpoints
    (values : Fin productionShape.sourceCount → Array K) (gamma weight : K)
    (low high : ProtocolPolynomial.OutputMessage K productionShape) (index : Nat)
    (lowSource : ∀ source, (values source).getD (2 * index) K.zero = low.sourceAssignment source)
    (highSource : ∀ source, (values source).getD (2 * index + 1) K.zero = high.sourceAssignment source) :
    FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
      (fun source => FixedPolynomial.scale extensionOps.toOps (power gamma source.val)
        (FixedPolynomial.scale extensionOps.toOps weight
          (PiCCSFirstRoundPair.normPair extensionOps
            ((values source).getD (2 * index) K.zero)
            ((values source).getD (2 * index + 1) K.zero)))) =
      FixedPolynomial.scale extensionOps.toOps weight
        (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps (power gamma) low high) := by
  change _ = FixedPolynomial.scale extensionOps.toOps weight
    (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount) _)
  rw [PiCCSPolynomialRange.scale_sum extensionOps extensionLaws]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount))
  funext source
  rw [PiCCSNormPrefixSum.scale_commute, lowSource source, highSource source]

private theorem sum_of_endpoints
    (data : ProtocolPolynomial.Data K productionShape)
    (values : Fin productionShape.sourceCount → Array K)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K)
    {remaining : Nat} (dimension : cubeVariables = challenges.length + remaining + 1)
    (count : Nat) (covered : ∀ source, (values source).size ≤ 2 * count)
    (included : count ≤ 2 ^ remaining)
    (endpoints : ∀ (index : Fin (2 ^ remaining)) (source : Fin productionShape.sourceCount) (bit : Bool),
      (values source).getD (2 * index.val + if bit then 1 else 0) K.zero =
        (ProtocolPolynomial.messageAt extensionOps data
          (PiCCSPrefixRound.point extensionOps challenges dimension
            (if bit then K.one else K.zero) (NumericBooleanDomain.vertex remaining index))).sourceAssignment source) :
    PiCCSNormContribution.normTerm data.toVerifierInput (power gamma)
        (headPrefix extensionOps challenges alpha)
        (PiCCSNormPrefixSum.innerRange values gamma
          (tailWeight extensionOps challenges alpha) count) =
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) (fun index =>
        PiCCSPrefixComposition.normTerm data alpha gamma challenges dimension
          (NumericBooleanDomain.vertex remaining index)) := by
  rw [PiCCSNormPrefixSum.innerRange_eq_full _ gamma _ count remaining covered included,
    PiCCSNormContribution.normTerm_sum]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)))
  funext index
  have low (source : Fin productionShape.sourceCount) := endpoints index source false
  have high (source : Fin productionShape.sourceCount) := endpoints index source true
  simp only [Bool.false_eq_true, if_false, if_true, Nat.add_zero] at low high
  rw [inner_pair_of_endpoints values gamma _ _ _ index.val low high,
    ← PiCCSNormPrefixSum.normTerm_scaled_selector]
  unfold PiCCSPrefixComposition.normTerm
  rw [PiCCSPrefixSelector.selector_factor extensionOps extensionLaws challenges
    (NumericBooleanDomain.vertex remaining index) alpha dimension,
    NumericBooleanDomain.index_vertex]
  rfl

/-- The pure source loop of replayNormAfterFirst. Each source supplies the
existing PiCCSPrefixNorm.range, followed by its own gamma power. -/
def inner (masks : Array (Array (Nat × Nat))) (challenge gamma : K)
    (weight : Nat → K) (groups : Nat) : FixedPolynomial K 3 :=
  FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
    fun source => FixedPolynomial.scale extensionOps.toOps (power gamma source.val)
      (PiCCSPrefixNorm.range challenge (code masks source) weight 0 groups)

private theorem inner_eq_range (masks : Array (Array (Nat × Nat))) (challenge gamma : K)
    (weight : Nat → K) (groups : Nat) :
    inner masks challenge gamma weight groups =
      PiCCSNormPrefixSum.innerRange
        (fun source => firstValues challenge (code masks source) groups) gamma weight groups := by
  unfold inner PiCCSNormPrefixSum.innerRange
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount))
  funext source
  rw [range_eq_sourceRange]

private theorem group_fits {remaining : Nat} (challenge : K)
    (dimension : cubeVariables = [challenge].length + remaining + 1)
    (groups : Nat) (fits : 4 * groups ≤ 2 ^ cubeVariables) :
    groups ≤ 2 ^ remaining := by
  have arity : cubeVariables = remaining + 2 := by
    simp only [List.length_cons, List.length_nil] at dimension
    omega
  have total : 2 ^ cubeVariables = 4 * 2 ^ remaining := by
    rw [arity, Nat.pow_add]
    simp only [show (2 : Nat) ^ 2 = 4 from rfl]
    exact Nat.mul_comm _ _
  rw [total] at fits
  omega

/-- Q1 after its one consumed challenge: the actual selected norm range
equals the complete canonical normTerm sum. Source values are proved from
the original masks. All 17 cubics, carrier tails and high coefficients remain. -/
theorem selected_eq_normTerm_sum
    (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat)))
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount) (groups : Nat)
    (covered : PiCCSSourceImages.shape.carrierWidth ≤ 4 * groups)
    (fits : 4 * groups ≤ 2 ^ cubeVariables)
    (challenge : K) {remaining : Nat}
    (dimension : cubeVariables = [challenge].length + remaining + 1)
    (alpha : CubePoint K cubeVariables) (gamma : K) :
    PiCCSNormContribution.normTerm
        (PiCCSFirstRoundComposition.sourceData input masks).toVerifierInput (power gamma)
        (headPrefix extensionOps [challenge] alpha)
        (inner masks challenge gamma (tailWeight extensionOps [challenge] alpha) groups) =
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) (fun index =>
        PiCCSPrefixComposition.normTerm (PiCCSFirstRoundComposition.sourceData input masks)
          alpha gamma [challenge] dimension (NumericBooleanDomain.vertex remaining index)) := by
  have transport := sum_of_endpoints (PiCCSFirstRoundComposition.sourceData input masks)
    (fun source => firstValues challenge (code masks source) groups) alpha gamma [challenge] dimension
    groups (fun source => by rw [firstValues_size]) (group_fits challenge dimension groups fits)
    (fun index source bit => by
      have endpoint := oneChallenge_endpoint input masks source loaded groups covered fits challenge
        dimension bit (NumericBooleanDomain.vertex remaining index)
      cases bit
      · change (firstValues challenge (code masks source) groups).getD
          (0 + 2 * NumericBooleanDomain.index (NumericBooleanDomain.vertex remaining index)) K.zero = _ at endpoint
        simpa only [NumericBooleanDomain.index_vertex, Nat.zero_add, Nat.add_zero, if_false] using endpoint
      · change (firstValues challenge (code masks source) groups).getD
          (1 + 2 * NumericBooleanDomain.index (NumericBooleanDomain.vertex remaining index)) K.zero = _ at endpoint
        simpa only [NumericBooleanDomain.index_vertex, Nat.add_comm 1, if_true] using endpoint)
  exact (congrArg
    (PiCCSNormContribution.normTerm
      (PiCCSFirstRoundComposition.sourceData input masks).toVerifierInput (power gamma)
      (headPrefix extensionOps [challenge] alpha))
    (inner_eq_range masks challenge gamma (tailWeight extensionOps [challenge] alpha) groups)).trans transport

end NightstreamFPrime.Export.Stage1.PiCCSNormFirstPrefix
