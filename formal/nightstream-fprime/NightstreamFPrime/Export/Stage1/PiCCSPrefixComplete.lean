import NightstreamFPrime.Export.Stage1.PiCCSFreshPrefixComplete
import NightstreamFPrime.Export.Stage1.PiCCSNormFirstPrefix

/-! Complete later-round source coefficients from the existing fresh,
carried and norm contribution kernels. Q1 uses its one-challenge norm
range; Q2 onward uses retained source arrays. File IO and transcript replay
are separate from this pure coefficient composition. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPrefixComplete

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.ConcreteCarrier
open NightstreamFPrime.Lifecycle
open PiCCSPrefixSelector (headPrefix tailWeight)

private abbrev power (gamma : K) := TargetPolynomial.power extensionOps.toOps gamma

/-- Exact four-scalar norm group count, including the last carrier tail. -/
def groups : Nat := (PiCCSSourceImages.shape.carrierWidth + 3) / 4

/-- Exact extent used by the retained norm file consumer. -/
def retainedRows (consumed : Nat) : Nat :=
  (PiCCSSourceImages.shape.carrierWidth + 2 ^ consumed - 1) / 2 ^ consumed

def activePairs (consumed : Nat) : Nat := (retainedRows consumed + 1) / 2

private theorem ceil_two (width : Nat) :
    (width + 2 ^ 2 - 1) / 2 ^ 2 = (width + 3) / 4 := by
  omega

/-- Q2's bucket runner uses ceil(groups/2), the same exact retained extent. -/
theorem activePairs_two : activePairs 2 = (groups + 1) / 2 :=
  congrArg (fun count : Nat => (count + 1) / 2) (ceil_two PiCCSSourceImages.shape.carrierWidth)

private theorem ceil4_geometry (width : Nat) (fits : width ≤ 2 ^ 28) :
    width ≤ 4 * ((width + 3) / 4) ∧ 4 * ((width + 3) / 4) ≤ 2 ^ 28 := by
  omega

/-- Both bounds come from the existing full-carrier cube bound. -/
theorem groups_geometry :
    PiCCSSourceImages.shape.carrierWidth ≤ 4 * groups ∧
      4 * groups ≤ 2 ^ cubeVariables := by
  unfold groups
  exact ceil4_geometry PiCCSSourceImages.shape.carrierWidth
    (PiCCSNormSource.canonicalLayout ()).columns_le

private theorem ceil_after_two (width depth : Nat) :
    (((width + 3) / 4) + 2 ^ depth - 1) / 2 ^ depth =
      (width + 2 ^ (2 + depth) - 1) / 2 ^ (2 + depth) := by
  have positive : 0 < 2 ^ depth := Nat.two_pow_pos depth
  have numerator :
      (width + 3) / 4 + 2 ^ depth - 1 =
        (width + 4 * 2 ^ depth - 1) / 4 := by omega
  rw [numerator, Nat.div_div_eq_div_mul, Nat.pow_add]

/-- Code decoding retains exactly one value per four original scalars.
Later folds have the same ceil extent as the executable's foldedCount. -/
theorem retained_size (masks : Array (Array (Nat × Nat)))
    (first second : K) (challenges : List K)
    (source : Fin productionShape.sourceCount) :
    (PiCCSNormPrefixSum.retained masks groups first second challenges source).size =
      retainedRows (first :: second :: challenges).length := by
  rw [PiCCSNormPrefixSum.retained, PiCCSCarriedPrefixComplete.foldPrefix_size]
  simp only [PiCCSPrefixCodeFold.decode, Array.size_map,
    PiCCSPrefixCodeFold.quadCodes, Array.size_ofFn]
  have length : (first :: second :: challenges).length = 2 + challenges.length := by
    simp only [List.length_cons]
    omega
  rw [length]
  exact ceil_after_two PiCCSSourceImages.shape.carrierWidth challenges.length

private theorem groups_fit_after_two (groupCount arity remaining : Nat)
    (challenges : List K) (first second : K)
    (fits : 4 * groupCount ≤ 2 ^ arity)
    (dimension : arity = (first :: second :: challenges).length + remaining + 1) :
    groupCount ≤ 2 ^ ((remaining + 1) + challenges.length) := by
  have aritySplit : arity = 2 + ((remaining + 1) + challenges.length) := by
    simp only [List.length_cons] at dimension
    omega
  rw [aritySplit, Nat.pow_add] at fits
  change 4 * groupCount ≤ 4 * 2 ^ ((remaining + 1) + challenges.length) at fits
  omega

/-- Actual retained sizes supply complete pair coverage and fit the suffix
cube. No retained-length premise is passed to the selected composition. -/
theorem retained_geometry (masks : Array (Array (Nat × Nat)))
    (first second : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = (first :: second :: challenges).length + remaining + 1) :
    (∀ source,
      (PiCCSNormPrefixSum.retained masks groups first second challenges source).size ≤
        2 * activePairs (first :: second :: challenges).length) ∧
      activePairs (first :: second :: challenges).length ≤ 2 ^ remaining := by
  have sizes (source : Fin productionShape.sourceCount) :=
    retained_size masks first second challenges source
  have initialFits := groups_fit_after_two groups cubeVariables remaining challenges first second
    groups_geometry.2 dimension
  have fits (source : Fin productionShape.sourceCount) :
      (PiCCSNormPrefixSum.retained masks groups first second challenges source).size ≤
        2 ^ (remaining + 1) := by
    unfold PiCCSNormPrefixSum.retained
    apply PrefixFold.foldPrefix_fits extensionOps _ challenges (remaining + 1)
    simpa only [PiCCSPrefixCodeFold.decode, Array.size_map,
      PiCCSPrefixCodeFold.quadCodes, Array.size_ofFn] using initialFits
  constructor
  · intro source
    rw [sizes source]
    unfold activePairs
    omega
  · have sizeBound := fits ⟨0, by decide⟩
    rw [sizes ⟨0, by decide⟩, Nat.pow_succ] at sizeBound
    unfold activePairs
    omega

private def normCoefficients (verifier : ProtocolPolynomial.VerifierInput K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K)
    (inner : FixedPolynomial K 3) : List K :=
  (PiCCSNormContribution.normTerm verifier (power gamma)
    (headPrefix extensionOps challenges alpha) inner).coefficients

private theorem normCoefficients_transport
    (left right : ProtocolPolynomial.VerifierInput K productionShape) (same : left = right)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K)
    (inner : FixedPolynomial K 3) :
    normCoefficients left alpha gamma challenges inner =
      normCoefficients right alpha gamma challenges inner := by
  subst left
  rfl

private def firstNorm (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (challenge : K) : List K :=
  let coins := PiCCSPublicReplay.pre input
  normCoefficients (PiCCSPublicReplay.verifierInput input) coins.alpha coins.gamma [challenge]
    (PiCCSNormFirstPrefix.inner masks challenge coins.gamma
      (tailWeight extensionOps [challenge] coins.alpha) groups)

private def laterNorm (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (first second : K) (challenges : List K) : List K :=
  let coins := PiCCSPublicReplay.pre input
  let consumed := first :: second :: challenges
  normCoefficients (PiCCSPublicReplay.verifierInput input) coins.alpha coins.gamma consumed
    (PiCCSNormPrefixSum.innerRange
      (PiCCSNormPrefixSum.retained masks groups first second challenges) coins.gamma
      (tailWeight extensionOps consumed coins.alpha) (activePairs consumed.length))

private def bucketNorm (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (first second : K) : List K :=
  let coins := PiCCSPublicReplay.pre input
  normCoefficients (PiCCSPublicReplay.verifierInput input) coins.alpha coins.gamma [first, second]
    (PiCCSNormPrefixSum.bucketInner masks groups first second coins.gamma
      (tailWeight extensionOps [first, second] coins.alpha) (activePairs [first, second].length))

-- Every option/list operation is proved with symbolic data before selected
-- source data is supplied. This avoids reducing the complete source tables.
private def join (carried fresh : Option (List K)) (norm : List K) : Option (List K) := do
  let carried ← carried
  let fresh ← fresh
  return List.zipWith extensionOps.add carried (List.zipWith extensionOps.add fresh norm)

private theorem join_coefficients {degree : Nat}
    (carried fresh norm : FixedPolynomial K degree) :
    join (some carried.coefficients) (some fresh.coefficients) norm.coefficients =
      some ((FixedPolynomial.add extensionOps.toOps carried
        (FixedPolynomial.add extensionOps.toOps fresh norm)).coefficients) := by
  simp only [join, bind, Option.bind]
  rw [PiCCSPolynomialRange.add_coefficients, PiCCSPolynomialRange.add_coefficients]
  rfl

private theorem join_eq_round (data : ProtocolPolynomial.Data K productionShape)
    (alpha : CubePoint K cubeVariables) (gamma : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1)
    (carried fresh : Option (List K)) (norm : List K)
    (carriedValue : carried =
      some ((FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSPrefixComposition.carriedTerm data gamma challenges dimension
          (NumericBooleanDomain.vertex remaining index)).coefficients))
    (freshValue : fresh =
      some ((FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSPrefixComposition.freshTerm data alpha gamma challenges dimension
          (NumericBooleanDomain.vertex remaining index)).coefficients))
    (normValue : norm =
      (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSPrefixComposition.normTerm data alpha gamma challenges dimension
          (NumericBooleanDomain.vertex remaining index)).coefficients) :
    join carried fresh norm =
      some (PiCCSPrefixRound.roundPolynomial extensionOps data alpha gamma challenges dimension).coefficients := by
  rw [carriedValue, freshValue, normValue, join_coefficients]
  exact congrArg (fun polynomial => some polynomial.coefficients)
    (PiCCSPrefixComposition.components_eq_roundPolynomial data alpha gamma challenges dimension)

private theorem firstNorm_value (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (challenge : K) {remaining : Nat}
    (dimension : cubeVariables = [challenge].length + remaining + 1) :
    firstNorm input masks challenge =
      (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSPrefixComposition.normTerm (PiCCSFirstRoundComposition.sourceData input masks)
          (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
          [challenge] dimension (NumericBooleanDomain.vertex remaining index)).coefficients := by
  have original := PiCCSNormFirstPrefix.selected_eq_normTerm_sum input masks loaded groups
    groups_geometry.1 groups_geometry.2 challenge dimension
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
  have verifier := PiCCSFreshComplete.verifierInput_eq_sourceProtocolData input
    (PiCCSFirstRoundComposition.witness masks)
  exact (normCoefficients_transport _ _ verifier
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma [challenge]
    (PiCCSNormFirstPrefix.inner masks challenge (PiCCSPublicReplay.pre input).gamma
      (tailWeight extensionOps [challenge] (PiCCSPublicReplay.pre input).alpha) groups)).trans
        (congrArg FixedPolynomial.coefficients original)

private theorem laterNorm_value (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (first second : K) (challenges : List K) {remaining : Nat}
    (dimension : cubeVariables = (first :: second :: challenges).length + remaining + 1) :
    laterNorm input masks first second challenges =
      (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSPrefixComposition.normTerm (PiCCSFirstRoundComposition.sourceData input masks)
          (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
          (first :: second :: challenges) dimension
          (NumericBooleanDomain.vertex remaining index)).coefficients := by
  have geometry := retained_geometry masks first second challenges dimension
  have original := PiCCSNormPrefixSum.selected_eq_normTerm_sum input masks groups loaded
    groups_geometry.1 groups_geometry.2 first second challenges dimension
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
    (activePairs (first :: second :: challenges).length) geometry.1 geometry.2
  have verifier := PiCCSFreshComplete.verifierInput_eq_sourceProtocolData input
    (PiCCSFirstRoundComposition.witness masks)
  exact (normCoefficients_transport _ _ verifier
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
    (first :: second :: challenges)
    (PiCCSNormPrefixSum.innerRange (PiCCSNormPrefixSum.retained masks groups first second challenges)
      (PiCCSPublicReplay.pre input).gamma
      (tailWeight extensionOps (first :: second :: challenges) (PiCCSPublicReplay.pre input).alpha)
      (activePairs (first :: second :: challenges).length))).trans
        (congrArg FixedPolynomial.coefficients original)

private theorem bucketNorm_value (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (first second : K) {remaining : Nat}
    (dimension : cubeVariables = [first, second].length + remaining + 1) :
    bucketNorm input masks first second =
      (FixedPolynomial.sum extensionOps.toOps (canonicalFinIndices (2 ^ remaining)) fun index =>
        PiCCSPrefixComposition.normTerm (PiCCSFirstRoundComposition.sourceData input masks)
          (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
          [first, second] dimension (NumericBooleanDomain.vertex remaining index)).coefficients := by
  have geometry := retained_geometry masks first second [] dimension
  have original := PiCCSNormPrefixSum.selected_buckets_eq_normTerm_sum input masks groups loaded
    groups_geometry.1 groups_geometry.2 first second dimension
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
    (activePairs [first, second].length) geometry.1 geometry.2
  have verifier := PiCCSFreshComplete.verifierInput_eq_sourceProtocolData input
    (PiCCSFirstRoundComposition.witness masks)
  exact (normCoefficients_transport _ _ verifier
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma [first, second]
    (PiCCSNormPrefixSum.bucketInner masks groups first second (PiCCSPublicReplay.pre input).gamma
      (tailWeight extensionOps [first, second] (PiCCSPublicReplay.pre input).alpha)
      (activePairs [first, second].length))).trans (congrArg FixedPolynomial.coefficients original)

/-- Pure complete later-round coefficient constructor. It joins existing
contribution kernels. Q0 remains owned by PiCCSFirstRoundComposition. -/
def coefficients? (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat)))
    (challenges : List K) (remaining : Nat) : Option (List K) :=
  match challenges with
  | [] => none
  | [challenge] =>
      join (PiCCSCarriedPrefixComplete.coefficients? input masks [challenge])
        (PiCCSFreshPrefixComplete.coefficients? input masks [challenge] remaining)
        (firstNorm input masks challenge)
  | first :: second :: rest =>
      join (PiCCSCarriedPrefixComplete.coefficients? input masks (first :: second :: rest))
        (PiCCSFreshPrefixComplete.coefficients? input masks (first :: second :: rest) remaining)
        (laterNorm input masks first second rest)

/-- Every complete coefficient of Q1 through Q27 is the original-source
round polynomial. The only source constraint is the loader's block bound.
Prefix dimension and nonemptiness express exactly a later, nonterminal round. -/
theorem coefficients_eq_roundPolynomial (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (challenges : List K) (nonempty : challenges ≠ []) {remaining : Nat}
    (dimension : cubeVariables = challenges.length + remaining + 1) :
    coefficients? input masks challenges remaining =
      some (PiCCSPrefixRound.roundPolynomial extensionOps
        (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
        challenges dimension).coefficients := by
  cases challenges with
  | nil => exact False.elim (nonempty rfl)
  | cons first rest =>
      cases rest with
      | nil =>
          exact join_eq_round (PiCCSFirstRoundComposition.sourceData input masks)
            (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
            [first] dimension _ _ _
            (PiCCSCarriedPrefixComplete.selected_coefficients input masks [first] dimension)
            (PiCCSFreshPrefixComplete.coefficients_eq_fullSum input masks [first] dimension)
            (firstNorm_value input masks loaded first dimension)
      | cons second rest =>
          exact join_eq_round (PiCCSFirstRoundComposition.sourceData input masks)
            (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
            (first :: second :: rest) dimension _ _ _
            (PiCCSCarriedPrefixComplete.selected_coefficients input masks (first :: second :: rest) dimension)
            (PiCCSFreshPrefixComplete.coefficients_eq_fullSum input masks (first :: second :: rest) dimension)
            (laterNorm_value input masks loaded first second rest dimension)

/-- The Q2 bucket path uses the same complete fresh/carried kernels. -/
def bucketCoefficients? (input : PiCCSPublicReplay.Input) (masks : Array (Array (Nat × Nat)))
    (first second : K) (remaining : Nat) : Option (List K) :=
  join (PiCCSCarriedPrefixComplete.coefficients? input masks [first, second])
    (PiCCSFreshPrefixComplete.coefficients? input masks [first, second] remaining)
    (bucketNorm input masks first second)

/-- The actual two-challenge norm bucket formula composes to the same
original-source Q2 coefficients, with no expected-artifact premise. -/
theorem bucketCoefficients_eq_roundPolynomial (input : PiCCSPublicReplay.Input)
    (masks : Array (Array (Nat × Nat))) (loaded : masks.size ≤ PiCCSSourceImages.blockCount)
    (first second : K) {remaining : Nat}
    (dimension : cubeVariables = [first, second].length + remaining + 1) :
    bucketCoefficients? input masks first second remaining =
      some (PiCCSPrefixRound.roundPolynomial extensionOps
        (PiCCSFirstRoundComposition.sourceData input masks)
        (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma
        [first, second] dimension).coefficients := by
  exact join_eq_round (PiCCSFirstRoundComposition.sourceData input masks)
    (PiCCSPublicReplay.pre input).alpha (PiCCSPublicReplay.pre input).gamma [first, second] dimension _ _ _
    (PiCCSCarriedPrefixComplete.selected_coefficients input masks [first, second] dimension)
    (PiCCSFreshPrefixComplete.coefficients_eq_fullSum input masks [first, second] dimension)
    (bucketNorm_value input masks loaded first second dimension)

end NightstreamFPrime.Export.Stage1.PiCCSPrefixComplete
