import NightstreamFPrime.Export.Stage1.PiCCSCarriedPrefixSource
import NightstreamFPrime.Export.Stage1.PiCCSPrefixComposition
import NightstreamFPrime.Export.Stage1.PiCCSPrefixSelectorFactor

/-! Pure carried-prefix accumulation and complete-sum composition.
Pad and matrix arrays keep independent extents. This module proves scalar
updates, ordered joins, and selected-source coefficients; it does not prove
file decoding, mutable IO loops, or the origin of retained files. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSCarriedPrefixComplete

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open NumericCompletionSum (numericSum)
open FiniteSumAlgebra (sumMap)

universe uField
variable {Field : Type uField}

/-- The scalar projection of carriedFromPrefix on one parity. Indices are
absolute, so a chunk boundary cannot reset either parity or suffix weight. -/
def rangeMoment (ops : InterpolationOps Field) (weight : Nat → Field)
    (values : Array Field) (first count : Nat) (bit : Fin 2) : Field :=
  numericSum ops count fun offset =>
    let row := first + offset
    if row % 2 = bit.val then
      ops.mul (weight (row / 2)) (values.getD row ops.zero)
    else ops.zero

private theorem rangeMoment_succ_raw (ops : InterpolationOps Field)
    (weight : Nat → Field) (values : Array Field) (first count : Nat) (bit : Fin 2) :
    rangeMoment ops weight values first (count + 1) bit =
      ops.add (rangeMoment ops weight values first count bit)
        (if (first + count) % 2 = bit.val then
          ops.mul (weight ((first + count) / 2)) (values.getD (first + count) ops.zero)
        else ops.zero) := by
  simp only [rangeMoment, numericSum, Nat.fold_succ]

/-- Exactly the per-value update used by the retained scalar runner. -/
theorem rangeMoment_step (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (weight : Nat → Field) (values : Array Field) (first count : Nat) (bit : Fin 2) :
    rangeMoment ops weight values first (count + 1) bit =
      if (first + count) % 2 = bit.val then
        ops.add (rangeMoment ops weight values first count bit)
          (ops.mul (weight ((first + count) / 2)) (values.getD (first + count) ops.zero))
      else rangeMoment ops weight values first count bit := by
  rw [rangeMoment_succ_raw]
  split_ifs <;> simp only [laws.add_zero]

/-- Consecutive chunks join in the same numeric order as the runner. -/
theorem rangeMoment_append (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (weight : Nat → Field) (values : Array Field) (first left right : Nat) (bit : Fin 2) :
    rangeMoment ops weight values first (left + right) bit =
      ops.add (rangeMoment ops weight values first left bit)
        (rangeMoment ops weight values (first + left) right bit) := by
  induction right with
  | zero =>
      change rangeMoment ops weight values first left bit =
        ops.add (rangeMoment ops weight values first left bit) ops.zero
      exact (laws.add_zero _).symm
  | succ right ih =>
      rw [Nat.add_succ, rangeMoment_succ_raw, ih, rangeMoment_succ_raw]
      simp only [Nat.add_assoc]
      exact laws.add_assoc _ _ _

/-- The complete retained array, with no common Pad/matrix extent. -/
def moment (ops : InterpolationOps Field) (weight : Nat → Field)
    (values : Array Field) (bit : Fin 2) : Field :=
  rangeMoment ops weight values 0 values.size bit

/-- An odd last value, a singleton and the empty prefix all use the same
zero-extended full-domain sum. No endpoint-correctness premise is used here. -/
theorem moment_eq_finSum (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (remaining : Nat)
    (weight : Nat → Field) (values : Array Field)
    (fits : values.size ≤ 2 ^ (remaining + 1)) (bit : Fin 2) :
    moment ops weight values bit =
      sumMap ops (canonicalFinIndices (2 ^ remaining)) fun pair =>
        ops.mul (weight pair.val) (values.getD (2 * pair.val + bit.val) ops.zero) := by
  have complete := PiCCSNumericSumOrder.numericSum_prefix_parity_eq_finSum
    ops laws remaining values.size bit
    (fun row => ops.mul (weight (row / 2)) (values.getD row ops.zero)) fits (by
      intro row outside _
      simp only [Array.getD_eq_getD_getElem?, Array.getElem?_eq_none outside,
        Option.getD_none, laws.mul_zero])
  change numericSum ops values.size (fun row =>
    if (0 + row) % 2 = bit.val then
      ops.mul (weight ((0 + row) / 2)) (values.getD (0 + row) ops.zero)
    else ops.zero) = _
  simp only [Nat.zero_add]
  rw [complete]
  apply FiniteSumAlgebra.sumMap_congr
  intro pair _
  have quotient : (2 * pair.val + bit.val) / 2 = pair.val := by
    have bounded := bit.isLt
    omega
  rw [quotient]

/-- The exact ceil-division extent used by PrefixMain.foldedCount. A
singleton is still folded for every remaining challenge. -/
theorem foldPrefix_size (ops : InterpolationOps Field)
    (values : Array Field) (challenges : List Field) :
    (PrefixFold.foldPrefix ops values challenges).size =
      (values.size + 2 ^ challenges.length - 1) / 2 ^ challenges.length := by
  induction challenges generalizing values with
  | nil => simp only [PrefixFold.foldPrefix, List.length_nil, Nat.pow_zero,
      Nat.add_sub_cancel, Nat.div_one]
  | cons challenge challenges ih =>
      rw [PrefixFold.foldPrefix, ih, PrefixFold.foldOne_size]
      simp only [List.length_cons, Nat.pow_succ']
      have positive : 0 < 2 ^ challenges.length := Nat.two_pow_pos _
      have numerator :
          (values.size + 1) / 2 + 2 ^ challenges.length - 1 =
            (values.size + 2 * 2 ^ challenges.length - 1) / 2 := by omega
      rw [numerator, Nat.div_div_eq_div_mul]

private theorem weighted_join {Index : Type} (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (indices : List Index)
    (weight left right : Index → Field) (shift : Field) :
    ops.add (sumMap ops indices (fun index => ops.mul (weight index) (left index)))
        (ops.mul shift (sumMap ops indices (fun index => ops.mul (weight index) (right index)))) =
      sumMap ops indices (fun index =>
        ops.mul (weight index) (ops.add (left index) (ops.mul shift (right index)))) := by
  rw [← FiniteSumAlgebra.sumMap_mul_left ops laws, ← FiniteSumAlgebra.sumMap_add ops laws]
  apply FiniteSumAlgebra.sumMap_congr
  intro index _
  rw [laws.left_distrib]
  apply congrArg (ops.add (ops.mul (weight index) (left index)))
  rw [← laws.mul_assoc, laws.mul_comm shift, laws.mul_assoc]

/-- Join the four actual scalar accumulators into every polynomial
coefficient. The global matrix shift occurs once, after separate sums. -/
theorem carried_from_moments (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat} (bound : 2 ≤ degree)
    (remaining : Nat) (head : FixedPolynomial Field 1) (weight : Nat → Field)
    (shift : Field) (pad matrix : Array Field)
    (padFits : pad.size ≤ 2 ^ (remaining + 1))
    (matrixFits : matrix.size ≤ 2 ^ (remaining + 1)) :
    PiCCSCarriedMoments.carriedPair ops bound head shift
        (moment ops weight pad ⟨0, by decide⟩)
        (moment ops weight pad ⟨1, by decide⟩)
        (moment ops weight matrix ⟨0, by decide⟩)
        (moment ops weight matrix ⟨1, by decide⟩) =
      FixedPolynomial.sum ops.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSCarriedMoments.carriedPair ops bound
          (FixedPolynomial.scale ops.toOps (weight index.val) head) shift
          (pad.getD (2 * index.val) ops.zero) (pad.getD (2 * index.val + 1) ops.zero)
          (matrix.getD (2 * index.val) ops.zero) (matrix.getD (2 * index.val + 1) ops.zero) := by
  rw [PiCCSCarriedMoments.carriedPair_combined ops laws,
    PiCCSCarriedMoments.sum_carriedPair ops laws]
  simp only [moment_eq_finSum ops laws remaining weight pad padFits,
    moment_eq_finSum ops laws remaining weight matrix matrixFits, Nat.add_zero]
  simp only [weighted_join ops laws]
  rfl

private theorem low_index {remaining : Nat} (suffix : BooleanVertex remaining) :
    NumericBooleanDomain.index (.cons false suffix) = 2 * NumericBooleanDomain.index suffix := by
  change 0 + 2 * NumericBooleanDomain.index suffix = _
  omega

private theorem high_index {remaining : Nat} (suffix : BooleanVertex remaining) :
    NumericBooleanDomain.index (.cons true suffix) = 2 * NumericBooleanDomain.index suffix + 1 := by
  change 1 + 2 * NumericBooleanDomain.index suffix = _
  omega

private def carriedCoefficients
    (verifier : ProtocolPolynomial.VerifierInput K productionShape)
    (gamma : K) (challenges : List K) (pad matrix : Array K) : List K :=
  let weight := PiCCSPrefixSelector.tailWeight extensionOps challenges verifier.priorPoint
  (PiCCSCarriedMoments.carriedPair extensionOps
    (show 2 ≤ verifier.sumcheckDegreeBound from Nat.le_trans (by decide) (Nat.le_max_right _ _))
    (PiCCSPrefixSelector.headPrefix extensionOps challenges verifier.priorPoint)
    (TargetPolynomial.power extensionOps.toOps gamma productionShape.matrixEvaluationOffset)
    (moment extensionOps weight pad ⟨0, by decide⟩)
    (moment extensionOps weight pad ⟨1, by decide⟩)
    (moment extensionOps weight matrix ⟨0, by decide⟩)
    (moment extensionOps weight matrix ⟨1, by decide⟩)).coefficients

/-- Pure scalar projection of composePrefix fed by the original retained
Pad/matrix constructors. A failed matrix initializer stays failed. -/
def coefficients? (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K) : Option (List K) := do
  let gamma := (PiCCSPublicReplay.pre input).gamma
  let verifier := PiCCSPublicReplay.verifierInput input
  let originalMatrix ← PiCCSCarriedPrefixSource.matrixPrefix? masks gamma
  let pad := PrefixFold.foldPrefix extensionOps (PiCCSCarriedPrefixSource.padPrefix masks gamma) challenges
  let matrix := PrefixFold.foldPrefix extensionOps originalMatrix challenges
  return carriedCoefficients verifier gamma challenges pad matrix

/-- Selected original inputs close every endpoint and source-load premise.
Only the dimension-correct challenge prefix is a caller hypothesis. This
states complete carried coefficients, not the full fresh/norm composition. -/
theorem selected_coefficients (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1) :
    coefficients? input masks challenges =
      some ((FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSPrefixComposition.carriedTerm (PiCCSFirstRoundComposition.sourceData input masks)
          (PiCCSPublicReplay.pre input).gamma challenges dimension
          (NumericBooleanDomain.vertex remaining index)).coefficients) := by
  let gamma := (PiCCSPublicReplay.pre input).gamma
  let data := PiCCSFirstRoundComposition.sourceData input masks
  obtain ⟨originalMatrix, loaded, matrixSize⟩ :=
    Option.map_eq_some_iff.mp (PiCCSCarriedPrefixSource.matrixPrefix_size input masks gamma)
  let pad := PrefixFold.foldPrefix extensionOps (PiCCSCarriedPrefixSource.padPrefix masks gamma) challenges
  let matrix := PrefixFold.foldPrefix extensionOps originalMatrix challenges
  have padInitialFits : (PiCCSCarriedPrefixSource.padPrefix masks gamma).size ≤ 2 ^ cubeVariables := by
    rw [PiCCSCarriedPrefixSource.padPrefix_size]
    exact PiCCSAggregatedImages.selectedLayout.columns_le
  have matrixInitialFits : originalMatrix.size ≤ 2 ^ cubeVariables := by
    rw [matrixSize,
      PerApplicationMatrixProgram.matrixProgram_rowCount_eq_structuralPlan
        Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits]
    exact PerApplicationFixedPoint.structuralPlan_rowCount_le
      Poseidon2HashChainV1Package.application Poseidon2HashChainV1Package.fits
  have shape : cubeVariables = (remaining + 1) + challenges.length := by omega
  have padFits : pad.size ≤ 2 ^ (remaining + 1) :=
    PrefixFold.foldPrefix_fits extensionOps _ challenges (remaining + 1) (by
      rw [← shape]; exact padInitialFits)
  have matrixFits : matrix.size ≤ 2 ^ (remaining + 1) :=
    PrefixFold.foldPrefix_fits extensionOps _ challenges (remaining + 1) (by
      rw [← shape]; exact matrixInitialFits)
  have verifier : PiCCSPublicReplay.verifierInput input = data.toVerifierInput :=
    PiCCSFreshComplete.verifierInput_eq_sourceProtocolData input
      (PiCCSFirstRoundComposition.witness masks)
  unfold coefficients?
  dsimp only
  rw [loaded]
  simp only [bind, Option.bind]
  rw [verifier]
  apply congrArg some
  unfold carriedCoefficients
  apply congrArg FixedPolynomial.coefficients
  rw [carried_from_moments extensionOps extensionLaws _ remaining _ _ _ pad matrix padFits matrixFits]
  apply congrArg (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)))
  funext index
  let suffix := NumericBooleanDomain.vertex remaining index
  have padLow := PiCCSCarriedPrefixSource.padPrefix_endpoint input masks gamma challenges dimension false suffix
  have padHigh := PiCCSCarriedPrefixSource.padPrefix_endpoint input masks gamma challenges dimension true suffix
  have matrixLow := PiCCSCarriedPrefixSource.matrixPrefix_endpoint input masks gamma challenges dimension false suffix
  have matrixHigh := PiCCSCarriedPrefixSource.matrixPrefix_endpoint input masks gamma challenges dimension true suffix
  rw [loaded, Option.map_some] at matrixLow matrixHigh
  have matrixLowValue := Option.some.inj matrixLow
  have matrixHighValue := Option.some.inj matrixHigh
  simp only [low_index, high_index, suffix, NumericBooleanDomain.index_vertex] at padLow padHigh matrixLowValue matrixHighValue
  unfold PiCCSPrefixComposition.carriedTerm
  rw [PiCCSPrefixSelector.selector_factor extensionOps extensionLaws challenges suffix data.priorPoint dimension]
  simp only [suffix, NumericBooleanDomain.index_vertex]
  exact congrArg₂
    (fun (padValues matrixValues : K × K) =>
      PiCCSCarriedMoments.carriedPair extensionOps
        (show 2 ≤ data.toVerifierInput.sumcheckDegreeBound from
          Nat.le_trans (by decide) (Nat.le_max_right _ _))
        (FixedPolynomial.scale extensionOps.toOps
          (PiCCSPrefixSelector.tailWeight extensionOps challenges data.priorPoint index.val)
          (PiCCSPrefixSelector.headPrefix extensionOps challenges data.priorPoint))
        (TargetPolynomial.power extensionOps.toOps gamma productionShape.matrixEvaluationOffset)
        padValues.1 padValues.2 matrixValues.1 matrixValues.2)
    (congrArg₂ Prod.mk padLow padHigh) (congrArg₂ Prod.mk matrixLowValue matrixHighValue)

end NightstreamFPrime.Export.Stage1.PiCCSCarriedPrefixComplete
