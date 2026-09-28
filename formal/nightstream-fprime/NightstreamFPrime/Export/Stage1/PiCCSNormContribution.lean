import NightstreamFPrime.Export.Stage1.PiCCSNormSource
import NightstreamFPrime.Export.Stage1.PiCCSNormScanCorrectness
import NightstreamFPrime.Export.Stage1.PiCCSNormBuckets
import NightstreamFPrime.Export.Stage1.PiCCSCachedSelector

/-! Coefficient-level connection of the scanned norm cubic to the existing
pair kernel. The alpha-tail weight enters the cubic; the head equality
factor, degree widening and both gamma shifts remain in the norm branch. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSNormContribution

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open PiCCSPolynomialRange

/-- The inner cubic accumulated by the scalar norm scan. -/
def innerNormPair (weight : K) (powers : Nat → K)
    (low high : ProtocolPolynomial.OutputMessage K productionShape) : FixedPolynomial K 3 :=
  FixedPolynomial.scale extensionOps.toOps weight
    (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers low high)

/-- The head factor already used by equalitySelector. -/
def headSelector (alpha : CubePoint K cubeVariables) : FixedPolynomial K 1 :=
  match alpha.coordinates with
  | [] => FixedPolynomial.zero extensionOps.toOps 1
  | head :: _ => FixedPolynomial.affine (extensionOps.sub extensionOps.one head)
      (extensionOps.sub head (extensionOps.sub extensionOps.one head))

/-- The exact norm branch of pairPolynomialWithNorm after distributing its
outer constraint scale. Source gamma powers remain inside sourceNorm. -/
def normTerm (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (selector : FixedPolynomial K 1) (sourceNorm : FixedPolynomial K 3) :
    FixedPolynomial K input.sumcheckDegreeBound :=
  FixedPolynomial.scale extensionOps.toOps (powers productionShape.constraintOffset)
    (FixedPolynomial.scale extensionOps.toOps (powers productionShape.freshCount)
      (FixedPolynomial.widen extensionOps.toOps
        (show 4 ≤ input.sumcheckDegreeBound from Nat.le_max_right _ _)
        (FixedPolynomial.mul extensionOps.toOps selector sourceNorm)))

private theorem scale_add {degree : Nat} (scalar : K)
    (left right : FixedPolynomial K degree) :
    FixedPolynomial.scale extensionOps.toOps scalar (FixedPolynomial.add extensionOps.toOps left right) =
      FixedPolynomial.add extensionOps.toOps
        (FixedPolynomial.scale extensionOps.toOps scalar left)
        (FixedPolynomial.scale extensionOps.toOps scalar right) := by
  apply coefficient_ext
  intro index
  simp only [coefficient_scale, coefficient_add]
  exact extensionLaws.left_distrib _ _ _

private theorem scale_scale {degree : Nat} (left right : K)
    (polynomial : FixedPolynomial K degree) :
    FixedPolynomial.scale extensionOps.toOps left (FixedPolynomial.scale extensionOps.toOps right polynomial) =
      FixedPolynomial.scale extensionOps.toOps (extensionOps.mul left right) polynomial := by
  apply coefficient_ext
  intro index
  simp only [coefficient_scale]
  exact (extensionLaws.mul_assoc _ _ _).symm

private theorem widen_zero {degree target : Nat} (bound : degree ≤ target) :
    FixedPolynomial.widen extensionOps.toOps bound (FixedPolynomial.zero extensionOps.toOps degree) =
      FixedPolynomial.zero extensionOps.toOps target := by
  apply coefficients_ext
  change List.replicate (degree + 1) extensionOps.zero ++
      List.replicate (target - degree) extensionOps.zero =
    List.replicate (target + 1) extensionOps.zero
  rw [← List.replicate_add]
  congr 1
  omega

private theorem affine_mul_coefficients (a b c₀ c₁ c₂ c₃ : K) :
    (FixedPolynomial.mul extensionOps.toOps (FixedPolynomial.affine a b)
      (⟨[c₀, c₁, c₂, c₃], rfl⟩ : FixedPolynomial K 3)).coefficients =
    [extensionOps.add (extensionOps.mul a c₀) extensionOps.zero,
     extensionOps.add (extensionOps.mul a c₁)
       (extensionOps.add (extensionOps.mul b c₀) extensionOps.zero),
     extensionOps.add (extensionOps.mul a c₂) (extensionOps.mul b c₁),
     extensionOps.add (extensionOps.mul a c₃) (extensionOps.mul b c₂),
     extensionOps.mul b c₃] := rfl

private theorem mul_zero_norm (selector : FixedPolynomial K 1) :
    FixedPolynomial.mul extensionOps.toOps selector (FixedPolynomial.zero extensionOps.toOps 3) =
      FixedPolynomial.zero extensionOps.toOps 4 := by
  rcases selector with ⟨coefficients, length⟩
  obtain ⟨a, b, rfl⟩ := List.length_eq_two.mp length
  apply coefficients_ext
  change (FixedPolynomial.mul extensionOps.toOps (FixedPolynomial.affine a b)
      (⟨[extensionOps.zero, extensionOps.zero, extensionOps.zero, extensionOps.zero], rfl⟩ :
        FixedPolynomial K 3)).coefficients =
    [extensionOps.zero, extensionOps.zero, extensionOps.zero, extensionOps.zero, extensionOps.zero]
  rw [affine_mul_coefficients]
  simp only [extensionLaws.mul_zero, extensionLaws.add_zero]

private theorem mul_scaled_affine (scalar a b : K) (polynomial : FixedPolynomial K 3) :
    FixedPolynomial.mul extensionOps.toOps
        (FixedPolynomial.scale extensionOps.toOps scalar (FixedPolynomial.affine a b)) polynomial =
      FixedPolynomial.mul extensionOps.toOps (FixedPolynomial.affine a b)
        (FixedPolynomial.scale extensionOps.toOps scalar polynomial) := by
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

/-- This term is exactly the additive norm contribution in the existing pair
constructor; the base is that same constructor with its norm input zeroed. -/
theorem pairPolynomialWithNorm_split
    (input : ProtocolPolynomial.VerifierInput K productionShape) (powers : Nat → K)
    (alphaSelector priorSelector : FixedPolynomial K 1)
    (low high : ProtocolPolynomial.OutputMessage K productionShape)
    (padLow padHigh matrixLow matrixHigh : K) (sourceNorm : FixedPolynomial K 3) :
    PiCCSFirstRoundPair.pairPolynomialWithNorm extensionOps input powers alphaSelector priorSelector
        low high padLow padHigh matrixLow matrixHigh sourceNorm =
      FixedPolynomial.add extensionOps.toOps
        (PiCCSFirstRoundPair.pairPolynomialWithNorm extensionOps input powers alphaSelector priorSelector
          low high padLow padHigh matrixLow matrixHigh (FixedPolynomial.zero extensionOps.toOps 3))
        (normTerm input powers alphaSelector sourceNorm) := by
  unfold PiCCSFirstRoundPair.pairPolynomialWithNorm normTerm
  simp only [mul_zero_norm, widen_zero, scale_zero_polynomial extensionOps extensionLaws,
    PiCCSPolynomialRange.add_zero extensionOps extensionLaws]
  simp only [scale_add, PiCCSPolynomialRange.add_assoc extensionOps extensionLaws]

private theorem selector_factor (alpha : CubePoint K cubeVariables)
    (suffix : BooleanVertex (cubeVariables - 1)) (sourceNorm : FixedPolynomial K 3) :
    FixedPolynomial.mul extensionOps.toOps (PiCCSFirstRound.equalitySelector extensionOps suffix alpha)
        sourceNorm =
      FixedPolynomial.mul extensionOps.toOps (headSelector alpha)
        (FixedPolynomial.scale extensionOps.toOps
          (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail
            (NumericBooleanDomain.index suffix)) sourceNorm) := by
  rcases alpha with ⟨coordinates, length⟩
  cases coordinates with
  | nil =>
      change (0 : Nat) = 28 at length
      omega
  | cons head tail =>
      have selector := PiCCSCachedSelector.equalitySelector_prepare extensionOps extensionLaws
        (by decide : cubeVariables = (cubeVariables - 1) + 1) suffix
        (⟨head :: tail, length⟩ : CubePoint K cubeVariables)
      rw [← selector]
      simp only [PiCCSCachedSelector.equalitySelector, headSelector, List.tail_cons,
        PiCCSTensorWeights.lookup_prepare extensionOps
          (NumericBooleanDomain.WeightProductLaws.ofInterpolationEvaluationLaws extensionLaws)]
      exact mul_scaled_affine _ _ _ sourceNorm

/-- Move only the alpha-tail scalar into the inner cubic. The original head
selector, both gamma scales and exact target width remain unchanged. -/
theorem normTerm_innerPair (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables)
    (suffix : BooleanVertex (cubeVariables - 1))
    (low high : ProtocolPolynomial.OutputMessage K productionShape) :
    normTerm input powers (PiCCSFirstRound.equalitySelector extensionOps suffix alpha)
        (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers low high) =
      normTerm input powers (headSelector alpha)
        (innerNormPair (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail
          (NumericBooleanDomain.index suffix)) powers low high) := by
  unfold normTerm innerNormPair
  rw [selector_factor]

private def shiftLaws : TargetPolynomial.ShiftLaws extensionOps.toOps where
  one_mul := extensionLaws.one_mul
  mul_assoc := extensionLaws.mul_assoc
  mul_zero := extensionLaws.mul_zero
  mul_add := extensionLaws.left_distrib

/-- For actual gamma powers, the two existing shifts combine to the declared
normOffset. Each source's local gamma exponent remains inside the cubic. -/
theorem normTerm_gamma (input : ProtocolPolynomial.VerifierInput K productionShape)
    (gamma : K) (selector : FixedPolynomial K 1) (sourceNorm : FixedPolynomial K 3) :
    normTerm input (TargetPolynomial.power extensionOps.toOps gamma) selector sourceNorm =
      FixedPolynomial.scale extensionOps.toOps
        (TargetPolynomial.power extensionOps.toOps gamma productionShape.normOffset)
        (FixedPolynomial.widen extensionOps.toOps
          (show 4 ≤ input.sumcheckDegreeBound from Nat.le_max_right _ _)
          (FixedPolynomial.mul extensionOps.toOps selector sourceNorm)) := by
  unfold normTerm
  rw [scale_scale]
  apply congrArg (fun scalar => FixedPolynomial.scale extensionOps.toOps scalar
    (FixedPolynomial.widen extensionOps.toOps
      (show 4 ≤ input.sumcheckDegreeBound from Nat.le_max_right _ _)
      (FixedPolynomial.mul extensionOps.toOps selector sourceNorm)))
  exact (TargetPolynomial.power_add extensionOps.toOps shiftLaws gamma
    productionShape.constraintOffset productionShape.freshCount).symm

private theorem normPair_zero :
    PiCCSFirstRoundPair.normPair extensionOps K.zero K.zero =
      FixedPolynomial.zero extensionOps.toOps 3 := by
  have diagonal := PiCCSNormBuckets.pairTable_diagonal ⟨1, by decide⟩
  rw [PiCCSNormCache.pairTable_value] at diagonal
  exact diagonal

private theorem normPolynomial_zero (powers : Nat → K)
    (low high : ProtocolPolynomial.OutputMessage K productionShape)
    (lowZero : ∀ source, low.sourceAssignment source = K.zero)
    (highZero : ∀ source, high.sourceAssignment source = K.zero) :
    PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers low high =
      FixedPolynomial.zero extensionOps.toOps 3 := by
  change FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices productionShape.sourceCount)
    (fun source => FixedPolynomial.scale extensionOps.toOps (powers source.val)
      (PiCCSFirstRoundPair.normPair extensionOps (low.sourceAssignment source)
        (high.sourceAssignment source))) = _
  simp only [lowZero, highZero, normPair_zero, scale_zero_polynomial extensionOps extensionLaws]
  apply coefficient_ext
  intro index
  simp only [coefficient_sum, coefficient_zero,
    FiniteSumAlgebra.sumMap_zero extensionOps extensionLaws]

private abbrev endpointMessage (masks : Array (Array (Nat × Nat)))
    (suffix : BooleanVertex (cubeVariables - 1)) (bit : Bool)
    (fresh : Vector F ProductionRelation.matrixCount) :=
  PiCCSAggregatedImages.nonlinearMessage (PiCCSNormSource.canonicalLayout ())
    (PiCCSNormSource.assignments masks)
    (PiCCSFirstRound.endpointVertex (arity := cubeVariables) (by decide) bit suffix) fresh

/-- Beyond the complete carrier, both endpoint fields vanish and the entire
inner cubic is the exact four-coefficient zero, for any scalar weight. -/
theorem innerNormPair_padding_zero (weight : K) (powers : Nat → K)
    (masks : Array (Array (Nat × Nat))) (suffix : BooleanVertex (cubeVariables - 1))
    (freshLow freshHigh : Vector F ProductionRelation.matrixCount)
    (padding : PiCCSSourceImages.shape.carrierWidth ≤ 2 * NumericBooleanDomain.index suffix) :
    innerNormPair weight powers (endpointMessage masks suffix false freshLow)
        (endpointMessage masks suffix true freshHigh) = FixedPolynomial.zero extensionOps.toOps 3 := by
  have lowZero : ∀ source,
      (endpointMessage masks suffix false freshLow).sourceAssignment source = K.zero := by
    intro source
    apply PiCCSNormSource.sourceAssignment_padding
    change PiCCSSourceImages.shape.carrierWidth ≤ 0 + 2 * NumericBooleanDomain.index suffix
    omega
  have highZero : ∀ source,
      (endpointMessage masks suffix true freshHigh).sourceAssignment source = K.zero := by
    intro source
    apply PiCCSNormSource.sourceAssignment_padding
    change PiCCSSourceImages.shape.carrierWidth ≤ 1 + 2 * NumericBooleanDomain.index suffix
    omega
  unfold innerNormPair
  rw [normPolynomial_zero powers _ _ lowZero highZero]
  exact scale_zero_polynomial extensionOps extensionLaws 3 weight

/-- Head weighting, both gamma shifts and width completion preserve the
exact zero contribution of the padded domain. -/
theorem normTerm_padding_zero (input : ProtocolPolynomial.VerifierInput K productionShape)
    (weight : K) (powers : Nat → K) (selector : FixedPolynomial K 1)
    (masks : Array (Array (Nat × Nat))) (suffix : BooleanVertex (cubeVariables - 1))
    (freshLow freshHigh : Vector F ProductionRelation.matrixCount)
    (padding : PiCCSSourceImages.shape.carrierWidth ≤ 2 * NumericBooleanDomain.index suffix) :
    normTerm input powers selector
        (innerNormPair weight powers (endpointMessage masks suffix false freshLow)
          (endpointMessage masks suffix true freshHigh)) =
      FixedPolynomial.zero extensionOps.toOps input.sumcheckDegreeBound := by
  rw [innerNormPair_padding_zero weight powers masks suffix freshLow freshHigh padding]
  unfold normTerm
  rw [mul_zero_norm, widen_zero, scale_zero_polynomial extensionOps extensionLaws,
    scale_zero_polynomial extensionOps extensionLaws]

private theorem add_shuffle (a b c d : K) :
    extensionOps.add (extensionOps.add a b) (extensionOps.add c d) =
      extensionOps.add (extensionOps.add a c) (extensionOps.add b d) := by
  rw [extensionLaws.add_assoc, extensionLaws.add_assoc]
  apply congrArg (extensionOps.add a)
  rw [← extensionLaws.add_assoc, extensionLaws.add_comm b c, extensionLaws.add_assoc]

private theorem mul_add_norm (selector : FixedPolynomial K 1)
    (left right : FixedPolynomial K 3) :
    FixedPolynomial.mul extensionOps.toOps selector (FixedPolynomial.add extensionOps.toOps left right) =
      FixedPolynomial.add extensionOps.toOps
        (FixedPolynomial.mul extensionOps.toOps selector left)
        (FixedPolynomial.mul extensionOps.toOps selector right) := by
  rcases selector with ⟨selector, selectorLength⟩
  obtain ⟨a, b, rfl⟩ := List.length_eq_two.mp selectorLength
  rcases left with ⟨left, leftLength⟩
  obtain ⟨l₀, l₁, l₂, l₃, rfl⟩ := List.length_eq_four.mp leftLength
  rcases right with ⟨right, rightLength⟩
  obtain ⟨r₀, r₁, r₂, r₃, rfl⟩ := List.length_eq_four.mp rightLength
  apply coefficients_ext
  rw [add_coefficients]
  change (FixedPolynomial.mul extensionOps.toOps (FixedPolynomial.affine a b)
      (⟨[extensionOps.add l₀ r₀, extensionOps.add l₁ r₁,
          extensionOps.add l₂ r₂, extensionOps.add l₃ r₃], rfl⟩ : FixedPolynomial K 3)).coefficients =
    List.zipWith extensionOps.add
      (FixedPolynomial.mul extensionOps.toOps (FixedPolynomial.affine a b)
        (⟨[l₀, l₁, l₂, l₃], rfl⟩ : FixedPolynomial K 3)).coefficients
      (FixedPolynomial.mul extensionOps.toOps (FixedPolynomial.affine a b)
        (⟨[r₀, r₁, r₂, r₃], rfl⟩ : FixedPolynomial K 3)).coefficients
  rw [affine_mul_coefficients, affine_mul_coefficients, affine_mul_coefficients]
  simp only [List.zipWith_cons_cons, List.zipWith_nil_left,
    extensionLaws.left_distrib, extensionLaws.add_zero]
  rw [add_shuffle (extensionOps.mul a l₁) (extensionOps.mul a r₁)
      (extensionOps.mul b l₀) (extensionOps.mul b r₀),
    add_shuffle (extensionOps.mul a l₂) (extensionOps.mul a r₂)
      (extensionOps.mul b l₁) (extensionOps.mul b r₁),
    add_shuffle (extensionOps.mul a l₃) (extensionOps.mul a r₃)
      (extensionOps.mul b l₂) (extensionOps.mul b r₂)]

private theorem widen_add {degree target : Nat} (bound : degree ≤ target)
    (left right : FixedPolynomial K degree) :
    FixedPolynomial.widen extensionOps.toOps bound (FixedPolynomial.add extensionOps.toOps left right) =
      FixedPolynomial.add extensionOps.toOps (FixedPolynomial.widen extensionOps.toOps bound left)
        (FixedPolynomial.widen extensionOps.toOps bound right) := by
  apply coefficients_ext
  rw [add_coefficients]
  change (FixedPolynomial.add extensionOps.toOps left right).coefficients ++
      List.replicate (target - degree) extensionOps.zero =
    List.zipWith extensionOps.add
      (left.coefficients ++ List.replicate (target - degree) extensionOps.zero)
      (right.coefficients ++ List.replicate (target - degree) extensionOps.zero)
  rw [add_coefficients,
    List.zipWith_append (by rw [left.coefficients_length, right.coefficients_length])]
  simp only [List.zipWith_replicate', extensionLaws.add_zero]

private theorem normTerm_zero (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (selector : FixedPolynomial K 1) :
    normTerm input powers selector (FixedPolynomial.zero extensionOps.toOps 3) =
      FixedPolynomial.zero extensionOps.toOps input.sumcheckDegreeBound := by
  unfold normTerm
  rw [mul_zero_norm, widen_zero, scale_zero_polynomial extensionOps extensionLaws,
    scale_zero_polynomial extensionOps extensionLaws]

private theorem normTerm_add (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (selector : FixedPolynomial K 1) (left right : FixedPolynomial K 3) :
    normTerm input powers selector (FixedPolynomial.add extensionOps.toOps left right) =
      FixedPolynomial.add extensionOps.toOps (normTerm input powers selector left)
        (normTerm input powers selector right) := by
  simp only [normTerm, mul_add_norm, widen_add, scale_add]

/-- Apply the common head selector and gamma shifts after a complete cubic
sum. Every coefficient, including all widened high zeros, is preserved. -/
theorem normTerm_sum {Index : Type}
    (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (selector : FixedPolynomial K 1)
    (indices : List Index) (term : Index → FixedPolynomial K 3) :
    normTerm input powers selector (FixedPolynomial.sum extensionOps.toOps indices term) =
      FixedPolynomial.sum extensionOps.toOps indices
        (fun index => normTerm input powers selector (term index)) := by
  induction indices with
  | nil => exact normTerm_zero input powers selector
  | cons index indices inductionHypothesis =>
      simpa only [FixedPolynomial.sum, normTerm_add] using
        congrArg (FixedPolynomial.add extensionOps.toOps
          (normTerm input powers selector (term index))) inductionHypothesis

/-- The same linearity applies to the ascending range returned by NormScan. -/
theorem normTerm_range (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (selector : FixedPolynomial K 1) (start count : Nat)
    (term : Nat → FixedPolynomial K 3) :
    normTerm input powers selector (PiCCSPolynomialRange.range extensionOps start count term) =
      PiCCSPolynomialRange.range extensionOps start count
        (fun index => normTerm input powers selector (term index)) := by
  induction count with
  | zero => exact normTerm_zero input powers selector
  | succ count inductionHypothesis =>
      simpa only [PiCCSPolynomialRange.range, Nat.fold_succ, normTerm_add] using
        congrArg (fun polynomial => FixedPolynomial.add extensionOps.toOps polynomial
          (normTerm input powers selector (term (start + count)))) inductionHypothesis

/-- A complete scanned block supplies exactly the norm components of its
27 canonical pair kernels. The source cubics come from the existing mask
sum; only their shared head factor and gamma shifts are applied afterward. -/
theorem blockNorm_normTerms (input : ProtocolPolynomial.VerifierInput K productionShape)
    (powers : Nat → K) (alpha : CubePoint K cubeVariables)
    (masks : Array (Array (Nat × Nat))) (block : Fin PiCCSSourceImages.blockCount)
    (freshLow freshHigh : Fin PiCCSNormSource.PairCount → Vector F ProductionRelation.matrixCount) :
    normTerm input powers (headSelector alpha)
        (PiCCSNormScanCorrectness.blockNorm powers
          (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail)
          block.val (masks[block.val]?.getD #[])) =
      FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices PiCCSNormSource.PairCount)
        (fun pair => normTerm input powers
          (PiCCSFirstRound.equalitySelector extensionOps (PiCCSNormSource.pairSuffix block pair) alpha)
          (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps powers
            (endpointMessage masks (PiCCSNormSource.pairSuffix block pair) false (freshLow pair))
            (endpointMessage masks (PiCCSNormSource.pairSuffix block pair) true (freshHigh pair)))) := by
  rw [PiCCSNormScanCorrectness.blockNorm_eq_normPolynomialWithPowers
    powers (NumericBooleanDomain.tensorWeightCoordinates extensionOps alpha.coordinates.tail)
    masks block freshLow freshHigh, normTerm_sum]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps
    (canonicalFinIndices PiCCSNormSource.PairCount))
  funext pair
  simpa only [innerNormPair, PiCCSNormSource.pairSuffix, NumericBooleanDomain.index_vertex] using
    (normTerm_innerPair input powers alpha (PiCCSNormSource.pairSuffix block pair)
      (endpointMessage masks (PiCCSNormSource.pairSuffix block pair) false (freshLow pair))
      (endpointMessage masks (PiCCSNormSource.pairSuffix block pair) true (freshHigh pair))).symm

end NightstreamFPrime.Export.Stage1.PiCCSNormContribution
