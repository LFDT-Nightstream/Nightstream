import NightstreamFPrime.Export.Stage1.PiCCSFirstRoundComposition
import NightstreamFPrime.Export.Stage1.PiCCSPrefixRound

/-! Original-source contribution composition after a challenge prefix.
All branches use the same messageAt endpoints and complete suffix domain.
This module owns coefficient algebra, not retained-array or file provenance. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPrefixComposition

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open FiniteSumAlgebra (sumMap)
open PiCCSPolynomialRange (coefficient coefficient_ext coefficient_add coefficient_scale coefficient_sum)

private abbrev powers (gamma : K) := TargetPolynomial.power extensionOps.toOps gamma

private def endpoint (data : ProtocolPolynomial.Data K productionShape)
    (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (value : K) (suffix : BooleanVertex remaining) :=
  ProtocolPolynomial.messageAt extensionOps data
    (PiCCSPrefixRound.point extensionOps challenges dimension value suffix)

private def padValue (gamma : K) (message : ProtocolPolynomial.OutputMessage K productionShape) : K :=
  sumMap extensionOps (canonicalPadCoordinates productionShape) fun coordinate =>
    extensionOps.mul (powers gamma coordinate.localGammaExponent) (message.padImage coordinate)

private def matrixValue (gamma : K) (message : ProtocolPolynomial.OutputMessage K productionShape) : K :=
  sumMap extensionOps (canonicalMatrixCoordinates productionShape) fun coordinate =>
    extensionOps.mul (powers gamma coordinate.localGammaExponent) (message.matrixImage coordinate)

/-- Original Pad and matrix totals keep separate coordinates and one matrix shift. -/
def carriedTerm (data : ProtocolPolynomial.Data K productionShape) (gamma : K)
    (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) : FixedPolynomial K data.toVerifierInput.sumcheckDegreeBound :=
  let low := endpoint data challenges dimension extensionOps.zero suffix
  let high := endpoint data challenges dimension extensionOps.one suffix
  PiCCSCarriedMoments.carriedPair extensionOps
    (show 2 ≤ data.toVerifierInput.sumcheckDegreeBound from
      Nat.le_trans (by decide) (Nat.le_max_right _ _))
    (PiCCSPrefixSelector.selector extensionOps challenges suffix data.priorPoint)
    (powers gamma productionShape.matrixEvaluationOffset)
    (padValue gamma low) (padValue gamma high) (matrixValue gamma low) (matrixValue gamma high)

/-- The existing fresh constructor, at the exact original-source endpoints. -/
def freshTerm (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) : FixedPolynomial K data.toVerifierInput.sumcheckDegreeBound :=
  PiCCSFreshComplete.outerFresh data.toVerifierInput (powers gamma)
    (PiCCSFirstRoundPair.ccsPolynomialWithPowers extensionOps data.toVerifierInput (powers gamma)
      (PiCCSPrefixSelector.selector extensionOps challenges suffix alpha)
      (endpoint data challenges dimension extensionOps.zero suffix)
      (endpoint data challenges dimension extensionOps.one suffix))

/-- normPolynomialWithPowers retains every source's cubic before its canonical
source sum. The source assignments are not combined before the cubic. -/
def normTerm (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) : FixedPolynomial K data.toVerifierInput.sumcheckDegreeBound :=
  PiCCSNormContribution.normTerm data.toVerifierInput (powers gamma)
    (PiCCSPrefixSelector.selector extensionOps challenges suffix alpha)
    (PiCCSFirstRoundPair.normPolynomialWithPowers extensionOps (powers gamma)
      (endpoint data challenges dimension extensionOps.zero suffix)
      (endpoint data challenges dimension extensionOps.one suffix))

/-- Separate full-domain contribution sums, in canonical numeric suffix order. -/
def components (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1) :
    FixedPolynomial K data.toVerifierInput.sumcheckDegreeBound :=
  FixedPolynomial.add extensionOps.toOps
    (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
      carriedTerm data gamma challenges dimension (NumericBooleanDomain.vertex remaining index))
    (FixedPolynomial.add extensionOps.toOps
      (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        freshTerm data alpha gamma challenges dimension (NumericBooleanDomain.vertex remaining index))
      (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        normTerm data alpha gamma challenges dimension (NumericBooleanDomain.vertex remaining index)))

private theorem scale_add {degree : Nat} (scalar : K)
    (left right : FixedPolynomial K degree) :
    FixedPolynomial.scale extensionOps.toOps scalar (FixedPolynomial.add extensionOps.toOps left right) =
      FixedPolynomial.add extensionOps.toOps (FixedPolynomial.scale extensionOps.toOps scalar left)
        (FixedPolynomial.scale extensionOps.toOps scalar right) := by
  apply coefficient_ext
  intro index
  simp only [coefficient_scale, coefficient_add]
  exact extensionLaws.left_distrib _ _ _

private theorem pair_components (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (suffix : BooleanVertex remaining) :
    FixedPolynomial.add extensionOps.toOps (carriedTerm data gamma challenges dimension suffix)
      (FixedPolynomial.add extensionOps.toOps
        (freshTerm data alpha gamma challenges dimension suffix)
        (normTerm data alpha gamma challenges dimension suffix)) =
      PiCCSPrefixRound.pairPolynomial extensionOps data alpha gamma challenges dimension suffix := by
  unfold carriedTerm freshTerm normTerm PiCCSFreshComplete.outerFresh PiCCSNormContribution.normTerm
  rw [PiCCSPrefixRound.pairPolynomial, PiCCSFirstRoundPair.pairPolynomial,
    ← PiCCSFirstRoundPair.pairPolynomialWithTotals_eq extensionOps extensionLaws,
    PiCCSFirstRoundPair.pairPolynomialWithTotals,
    PiCCSCarriedMoments.pairPolynomialWithNorm_split extensionOps extensionLaws, scale_add]
  rfl

/-- Exact fixed-width polynomial equality for every dimension-correct prefix.
The endpoints are original messageAt values, not caller correctness premises. -/
theorem components_eq_roundPolynomial (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1) :
    components data alpha gamma challenges dimension =
      PiCCSPrefixRound.roundPolynomial extensionOps data alpha gamma challenges dimension := by
  unfold components
  rw [PiCCSPrefixRound.roundPolynomial, PiCCSPolynomialRange.range_eq_sum extensionOps extensionLaws]
  apply coefficient_ext
  intro index
  simp only [coefficient_add, coefficient_sum]
  rw [← FiniteSumAlgebra.sumMap_add extensionOps extensionLaws,
    ← FiniteSumAlgebra.sumMap_add extensionOps extensionLaws]
  apply FiniteSumAlgebra.sumMap_congr
  intro suffix _
  simpa only [PiCCSPrefixRound.numericPair, Nat.zero_add, dif_pos suffix.isLt, coefficient_add] using
    congrArg (fun polynomial => coefficient polynomial index)
      (pair_components data alpha gamma challenges dimension (NumericBooleanDomain.vertex remaining suffix))

/-- The same composed polynomial evaluates to the original completion sum. -/
theorem components_evaluate (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1) (value : K) :
    (components data alpha gamma challenges dimension).evaluate extensionOps.toOps value =
      HypercubeTruth.sumCompletions extensionOps.toOps
        (ProtocolPolynomial.polynomial extensionOps data alpha gamma) (challenges ++ [value]) remaining := by
  rw [components_eq_roundPolynomial]
  exact PiCCSPrefixRound.roundPolynomial_evaluate extensionOps extensionLaws
    data alpha gamma challenges dimension value

/-- Selected public input and original signed masks supply one common source.
No expected polynomial, valid-witness or contribution-correctness premise is used. -/
theorem selected_coefficients (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1) :
    (components (PiCCSFirstRoundComposition.sourceData input masks)
      (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
      challenges dimension).coefficients =
      (PiCCSPrefixRound.roundPolynomial extensionOps (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
        challenges dimension).coefficients :=
  congrArg FixedPolynomial.coefficients (components_eq_roundPolynomial
    (PiCCSFirstRoundComposition.sourceData input masks)
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma challenges dimension)

end NightstreamFPrime.Export.Stage1.PiCCSPrefixComposition
