import NightstreamFPrime.Export.Stage1.PiCCSNormPrefixSource
import NightstreamFPrime.Export.Stage1.PiCCSPrefixComposition
import NightstreamFPrime.Export.Stage1.PiCCSPrefixSelectorFactor

/-! Pure norm accumulation after retained prefix folds. Each source keeps its
own cubic before its gamma scale. This file proves coefficient formulas for
the existing bucket and direct range kernels, not file or worker-loop origin. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormPrefixSum

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

/-- The already proved source array: two code-table folds followed by the
existing scalar prefix fold. Source indices are never combined here. -/
def retained (masks : Array (Array (Nat × Nat))) (groups : Nat)
    (first second : K) (challenges : List K)
    (source : Fin productionShape.sourceCount) : Array K :=
  PrefixFold.foldPrefix extensionOps
    (PiCCSPrefixCodeFold.decode
      (PiCCSPrefixCodeFold.pairedTable (PiCCSPrefixNorm.values first) second)
      (PiCCSPrefixCodeFold.quadCodes (code masks source) 0 groups)) challenges

/-- The pure direct cubic update used by normAfterTwo and, after adjacent
pair decoding, normFromPrefix. Missing array endpoints are exactly zero. -/
def sourceRange (values : Array K) (weight : Nat → K) (count : Nat) :
    FixedPolynomial K 3 :=
  PiCCSPolynomialRange.range extensionOps 0 count fun index =>
    FixedPolynomial.scale extensionOps.toOps (weight index)
      (PiCCSFirstRoundPair.normPair extensionOps
        (values.getD (2 * index) K.zero) (values.getD (2 * index + 1) K.zero))

/-- Canonical source sum of separately weighted cubics. The source gamma
factor is outside each source's complete numeric pair range. -/
def innerRange (values : Fin productionShape.sourceCount → Array K)
    (gamma : K) (weight : Nat → K) (count : Nat) : FixedPolynomial K 3 :=
  FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
    fun source => FixedPolynomial.scale extensionOps.toOps (power gamma source.val)
      (sourceRange (values source) weight count)

private theorem normPair_zero :
    PiCCSFirstRoundPair.normPair extensionOps K.zero K.zero =
      FixedPolynomial.zero extensionOps.toOps 3 := by
  have diagonal := PiCCSNormBuckets.pairTable_diagonal ⟨1, by decide⟩
  rw [PiCCSNormCache.pairTable_value] at diagonal
  exact diagonal

private theorem getD_outside (values : Array K) (index : Nat)
    (outside : values.size ≤ index) : values.getD index K.zero = K.zero := by
  simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none outside, Option.getD_none]

/-- Extending a complete pair range adds only zero cubics. The final stored
value still pairs with zero when the length is odd or one. -/
theorem sourceRange_eq_full (values : Array K) (weight : Nat → K)
    (count fullCount : Nat) (covered : values.size ≤ 2 * count)
    (included : count ≤ fullCount) :
    sourceRange values weight count = sourceRange values weight fullCount := by
  unfold sourceRange
  have countSplit : fullCount = count + (fullCount - count) := by omega
  rw [countSplit]
  symm
  apply PiCCSPolynomialRange.range_append_zero extensionOps extensionLaws 0 count
    (fullCount - count)
  intro index lower _
  rw [getD_outside values (2 * index) (by omega),
    getD_outside values (2 * index + 1) (by omega), normPair_zero]
  exact scale_zero_polynomial extensionOps extensionLaws 3 (weight index)

/-- Swap only the two finite sums, after preserving each source cubic.
All absent suffix pairs are discharged by actual array sizes. -/
theorem innerRange_eq_full (values : Fin productionShape.sourceCount → Array K)
    (gamma : K) (weight : Nat → K) (count remaining : Nat)
    (covered : ∀ source, (values source).size ≤ 2 * count)
    (included : count ≤ 2 ^ remaining) :
    innerRange values gamma weight count =
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
          fun source => FixedPolynomial.scale extensionOps.toOps (power gamma source.val)
            (FixedPolynomial.scale extensionOps.toOps (weight index.val)
              (PiCCSFirstRoundPair.normPair extensionOps
                ((values source).getD (2 * index.val) K.zero)
                ((values source).getD (2 * index.val + 1) K.zero))) := by
  calc
    innerRange values gamma weight count =
        innerRange values gamma weight (2 ^ remaining) := by
      unfold innerRange
      apply congrArg (FixedPolynomial.sum extensionOps.toOps
        (canonicalFinIndices productionShape.sourceCount))
      funext source
      exact congrArg (FixedPolynomial.scale extensionOps.toOps (power gamma source.val))
        (sourceRange_eq_full (values source) weight count (2 ^ remaining)
          (covered source) included)
    _ = _ := by
      simp only [innerRange, sourceRange,
        PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws, Nat.zero_add,
        PiCCSPolynomialRange.scale_sum extensionOps extensionLaws]
      apply coefficient_ext
      intro coefficientIndex
      simp only [coefficient_sum]
      exact FiniteSumAlgebra.sumMap_swap extensionOps extensionLaws
        (canonicalFinIndices productionShape.sourceCount) (canonicalFinIndices (2 ^ remaining)) _

theorem scale_commute {degree : Nat} (left right : K)
    (polynomial : FixedPolynomial K degree) :
    FixedPolynomial.scale extensionOps.toOps left
        (FixedPolynomial.scale extensionOps.toOps right polynomial) =
      FixedPolynomial.scale extensionOps.toOps right
        (FixedPolynomial.scale extensionOps.toOps left polynomial) := by
  apply coefficient_ext
  intro index
  simp only [coefficient_scale]
  rw [← extensionLaws.mul_assoc, extensionLaws.mul_comm left right, extensionLaws.mul_assoc]

private theorem affine_mul_coefficients (a b c₀ c₁ c₂ c₃ : K) :
    (FixedPolynomial.mul extensionOps.toOps (FixedPolynomial.affine a b)
      (⟨[c₀, c₁, c₂, c₃], rfl⟩ : FixedPolynomial K 3)).coefficients =
    [extensionOps.add (extensionOps.mul a c₀) extensionOps.zero,
     extensionOps.add (extensionOps.mul a c₁)
       (extensionOps.add (extensionOps.mul b c₀) extensionOps.zero),
     extensionOps.add (extensionOps.mul a c₂) (extensionOps.mul b c₁),
     extensionOps.add (extensionOps.mul a c₃) (extensionOps.mul b c₂),
     extensionOps.mul b c₃] := rfl

private theorem mul_scaled_selector (scalar : K) (selector : FixedPolynomial K 1)
    (polynomial : FixedPolynomial K 3) :
    FixedPolynomial.mul extensionOps.toOps
        (FixedPolynomial.scale extensionOps.toOps scalar selector) polynomial =
      FixedPolynomial.mul extensionOps.toOps selector
        (FixedPolynomial.scale extensionOps.toOps scalar polynomial) := by
  rcases selector with ⟨selector, selectorLength⟩
  obtain ⟨a, b, rfl⟩ := List.length_eq_two.mp selectorLength
  rcases polynomial with ⟨coefficients, length⟩
  obtain ⟨c₀, c₁, c₂, c₃, rfl⟩ := List.length_eq_four.mp length
  apply coefficients_ext
  change (FixedPolynomial.mul extensionOps.toOps
      (FixedPolynomial.affine (extensionOps.mul scalar a) (extensionOps.mul scalar b))
      (⟨[c₀, c₁, c₂, c₃], rfl⟩ : FixedPolynomial K 3)).coefficients =
    (FixedPolynomial.mul extensionOps.toOps (FixedPolynomial.affine a b)
      (⟨[extensionOps.mul scalar c₀, extensionOps.mul scalar c₁,
          extensionOps.mul scalar c₂, extensionOps.mul scalar c₃], rfl⟩ :
        FixedPolynomial K 3)).coefficients
  rw [affine_mul_coefficients, affine_mul_coefficients]
  have commute (left right : K) :
      extensionOps.mul (extensionOps.mul scalar left) right =
        extensionOps.mul left (extensionOps.mul scalar right) := by
    rw [extensionLaws.mul_comm scalar left, extensionLaws.mul_assoc]
  simp only [commute]

theorem normTerm_scaled_selector
    (input : ProtocolPolynomial.VerifierInput K productionShape) (powers : Nat → K)
    (weight : K) (selector : FixedPolynomial K 1) (polynomial : FixedPolynomial K 3) :
    PiCCSNormContribution.normTerm input powers
        (FixedPolynomial.scale extensionOps.toOps weight selector) polynomial =
      PiCCSNormContribution.normTerm input powers selector
        (FixedPolynomial.scale extensionOps.toOps weight polynomial) := by
  unfold PiCCSNormContribution.normTerm
  rw [mul_scaled_selector]

private theorem source_pair (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (groups : Nat)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (covered : PiCCSSourceImages.shape.carrierWidth ≤ 4 * groups)
    (fits : 4 * groups ≤ 2 ^ cubeVariables)
    (first second : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = (first :: second :: challenges).length + remaining + 1)
    (index : Fin (2 ^ remaining)) (source : Fin productionShape.sourceCount) (bit : Bool) :
    (retained masks groups first second challenges source).getD
        (2 * index.val + if bit then 1 else 0) K.zero =
      (ProtocolPolynomial.messageAt extensionOps
        (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPrefixRound.point extensionOps (first :: second :: challenges) dimension
          (if bit then K.one else K.zero) (NumericBooleanDomain.vertex remaining index))).sourceAssignment source := by
  have endpoint := PiCCSNormPrefixSource.quadCodes_endpoint input masks source loaded groups covered fits
    first second challenges dimension bit (NumericBooleanDomain.vertex remaining index)
  cases bit
  · change (retained masks groups first second challenges source).getD
        (0 + 2 * NumericBooleanDomain.index (NumericBooleanDomain.vertex remaining index)) K.zero = _ at endpoint
    simpa only [NumericBooleanDomain.index_vertex, Nat.zero_add, Nat.add_zero, if_false] using endpoint
  · change (retained masks groups first second challenges source).getD
        (1 + 2 * NumericBooleanDomain.index (NumericBooleanDomain.vertex remaining index)) K.zero = _ at endpoint
    simpa only [NumericBooleanDomain.index_vertex, Nat.add_comm 1, if_true] using endpoint

private theorem inner_pair_value (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (groups : Nat)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (covered : PiCCSSourceImages.shape.carrierWidth ≤ 4 * groups)
    (fits : 4 * groups ≤ 2 ^ cubeVariables)
    (first second : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = (first :: second :: challenges).length + remaining + 1)
    (gamma weight : K) (index : Fin (2 ^ remaining)) :
    FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
      (fun source => FixedPolynomial.scale extensionOps.toOps (power gamma source.val)
        (FixedPolynomial.scale extensionOps.toOps weight
          (PiCCSFirstRoundPair.normPair extensionOps
            ((retained masks groups first second challenges source).getD (2 * index.val) K.zero)
            ((retained masks groups first second challenges source).getD (2 * index.val + 1) K.zero)))) =
      FixedPolynomial.scale extensionOps.toOps weight
        (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps (power gamma)
          (ProtocolPolynomial.messageAt extensionOps
            (PiCCSFirstRoundComposition.sourceData input masks)
            (PiCCSPrefixRound.point extensionOps (first :: second :: challenges) dimension K.zero
              (NumericBooleanDomain.vertex remaining index)))
          (ProtocolPolynomial.messageAt extensionOps
            (PiCCSFirstRoundComposition.sourceData input masks)
            (PiCCSPrefixRound.point extensionOps (first :: second :: challenges) dimension K.one
              (NumericBooleanDomain.vertex remaining index)))) := by
  change _ = FixedPolynomial.scale extensionOps.toOps weight
    (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount) _)
  rw [PiCCSPolynomialRange.scale_sum extensionOps extensionLaws]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount))
  funext source
  rw [scale_commute]
  apply congrArg (FixedPolynomial.scale extensionOps.toOps weight)
  apply congrArg (FixedPolynomial.scale extensionOps.toOps (power gamma source.val))
  have low := source_pair input masks groups loaded covered fits first second challenges dimension index source false
  have high := source_pair input masks groups loaded covered fits first second challenges dimension index source true
  exact congrArg₂ (PiCCSFirstRoundPair.normPair extensionOps) low high

/-- The complete pure direct norm contribution equals the canonical sum of
the original normTerm. Premises concern only source loading, domain geometry,
and retained lengths. No source-value or contribution equality is assumed. -/
theorem selected_eq_normTerm_sum
    (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat))) (groups : Nat)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (covered : PiCCSSourceImages.shape.carrierWidth ≤ 4 * groups)
    (fits : 4 * groups ≤ 2 ^ cubeVariables)
    (first second : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = (first :: second :: challenges).length + remaining + 1)
    (alpha : CubePoint K cubeVariables) (gamma : K) (count : Nat)
    (retainedCovered : ∀ source,
      (retained masks groups first second challenges source).size ≤ 2 * count)
    (included : count ≤ 2 ^ remaining) :
    PiCCSNormContribution.normTerm
        (PiCCSFirstRoundComposition.sourceData input masks).toVerifierInput (power gamma)
        (headPrefix extensionOps (first :: second :: challenges) alpha)
        (innerRange (retained masks groups first second challenges) gamma
          (tailWeight extensionOps (first :: second :: challenges) alpha) count) =
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) (fun index =>
        PiCCSPrefixComposition.normTerm (PiCCSFirstRoundComposition.sourceData input masks)
          alpha gamma (first :: second :: challenges) dimension
          (NumericBooleanDomain.vertex remaining index)) := by
  rw [innerRange_eq_full _ gamma _ count remaining retainedCovered included,
    PiCCSNormContribution.normTerm_sum]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)))
  funext index
  rw [inner_pair_value input masks groups loaded covered fits first second challenges dimension gamma]
  rw [← normTerm_scaled_selector]
  unfold PiCCSPrefixComposition.normTerm
  rw [PiCCSPrefixSelector.selector_factor extensionOps extensionLaws
    (first :: second :: challenges) (NumericBooleanDomain.vertex remaining index) alpha dimension,
    NumericBooleanDomain.index_vertex]
  rfl

private theorem decoded_getD {count : Nat} (values : Vector K count) (zeroCode : Fin count)
    (zeroValue : values.get zeroCode = K.zero) (codes : Array (Fin count)) (index : Nat) :
    (PiCCSPrefixCodeFold.decode values codes).getD index K.zero =
      values.get (codes.getD index zeroCode) := by
  simp only [PiCCSPrefixCodeFold.decode, Array.getD_eq_getD_getElem?, Array.getElem?_map]
  cases loaded : codes[index]? with
  | none => simpa only [loaded, Option.map_none, Option.getD_none] using zeroValue.symm
  | some code => simp only [Option.map_some, Option.getD_some]

/-- The actual 81-entry bucket accumulator gives the direct array cubic
range, including its true missing final endpoint with zero code 40. -/
theorem bucketRange_eq_sourceRange (first second : K) (codes : Array (Fin 81))
    (weight : Nat → K) (count : Nat) :
    PiCCSPrefixNormBuckets.range
        (PiCCSPrefixCodeFold.pairedTable (PiCCSPrefixNorm.values first) second)
        (fun index => codes.getD (2 * index) ⟨40, by decide⟩)
        (fun index => codes.getD (2 * index + 1) ⟨40, by decide⟩) weight 0 count =
      sourceRange
        (PiCCSPrefixCodeFold.decode
          (PiCCSPrefixCodeFold.pairedTable (PiCCSPrefixNorm.values first) second) codes)
        weight count := by
  rw [PiCCSPrefixNormBuckets.range_eq_reference]
  unfold sourceRange
  apply congrArg (PiCCSPolynomialRange.range extensionOps 0 count)
  funext index
  rw [decoded_getD _ ⟨40, by decide⟩ (PiCCSPrefixCodeFold.secondTable_zero first second),
    decoded_getD _ ⟨40, by decide⟩ (PiCCSPrefixCodeFold.secondTable_zero first second)]

/-- Pure normAfterTwo bucket formula, with one gamma scale per source. -/
def bucketInner (masks : Array (Array (Nat × Nat))) (groups : Nat)
    (first second gamma : K) (weight : Nat → K) (count : Nat) : FixedPolynomial K 3 :=
  FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount) fun source =>
    let codes := PiCCSPrefixCodeFold.quadCodes (code masks source) 0 groups
    FixedPolynomial.scale extensionOps.toOps (power gamma source.val)
      (PiCCSPrefixNormBuckets.range
        (PiCCSPrefixCodeFold.pairedTable (PiCCSPrefixNorm.values first) second)
        (fun index => codes.getD (2 * index) ⟨40, by decide⟩)
        (fun index => codes.getD (2 * index + 1) ⟨40, by decide⟩) weight 0 count)

/-- Bucket completion and the direct source formula are identical before
the shared consumed-alpha selector and global gamma factors are applied. -/
theorem bucketInner_eq_innerRange (masks : Array (Array (Nat × Nat))) (groups : Nat)
    (first second gamma : K) (weight : Nat → K) (count : Nat) :
    bucketInner masks groups first second gamma weight count =
      innerRange (retained masks groups first second []) gamma weight count := by
  unfold bucketInner innerRange
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount))
  funext source
  dsimp only
  rw [bucketRange_eq_sourceRange]
  rfl

/-- The selected bucket path has the same complete original normTerm sum,
with no decoded-file value or bucket correctness premise. -/
theorem selected_buckets_eq_normTerm_sum
    (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat))) (groups : Nat)
    (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (covered : PiCCSSourceImages.shape.carrierWidth ≤ 4 * groups)
    (fits : 4 * groups ≤ 2 ^ cubeVariables)
    (first second : K) {remaining : Nat}
    (dimension : cubeVariables = [first, second].length + remaining + 1)
    (alpha : CubePoint K cubeVariables) (gamma : K) (count : Nat)
    (retainedCovered : ∀ source,
      (retained masks groups first second [] source).size ≤ 2 * count)
    (included : count ≤ 2 ^ remaining) :
    PiCCSNormContribution.normTerm
        (PiCCSFirstRoundComposition.sourceData input masks).toVerifierInput (power gamma)
        (headPrefix extensionOps [first, second] alpha)
        (bucketInner masks groups first second gamma
          (tailWeight extensionOps [first, second] alpha) count) =
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) (fun index =>
        PiCCSPrefixComposition.normTerm (PiCCSFirstRoundComposition.sourceData input masks)
          alpha gamma [first, second] dimension (NumericBooleanDomain.vertex remaining index)) := by
  rw [bucketInner_eq_innerRange]
  exact selected_eq_normTerm_sum input masks groups loaded covered fits first second []
    dimension alpha gamma count retainedCovered included

end NightstreamFPrime.Export.Stage1.PiCCSNormPrefixSum
