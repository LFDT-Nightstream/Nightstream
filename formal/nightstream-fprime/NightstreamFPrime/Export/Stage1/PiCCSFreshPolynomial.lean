import NightstreamFPrime.Export.Stage1.PiCCSPolynomialRange

/-! The existing fresh CCS polynomial, with exact-zero terms and exponent-zero
factors skipped. All returned coefficients keep the original declared width. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSFreshPolynomial

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open ProtocolPolynomialDegree.Support (polynomialLaws)
open PiCCSPolynomialRange (coefficients_ext coefficient_ext coefficient_add
  coefficient_zero scale_zero_polynomial)

universe uField
variable {Field : Type uField} {Index : Type}

private theorem widen_zero (ops : Ops Field) {degree target : Nat}
    (bound : degree ≤ target) :
    FixedPolynomial.widen ops bound (FixedPolynomial.zero ops degree) =
      FixedPolynomial.zero ops target := by
  apply coefficients_ext
  change List.replicate (degree + 1) ops.zero ++
      List.replicate (target - degree) ops.zero = List.replicate (target + 1) ops.zero
  rw [← List.replicate_add]
  congr 1
  omega

private theorem zero_add (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (value : FixedPolynomial Field degree) :
    FixedPolynomial.add ops.toOps (FixedPolynomial.zero ops.toOps degree) value = value := by
  apply coefficient_ext
  intro index
  rw [coefficient_add, coefficient_zero, laws.zero_add]

/-- Skip only exponent-zero factors. Positive powers retain the exact
reference operation and order. -/
def productPowers (ops : InterpolationOps Field)
    (exponents : Index → Nat) (images : Index → FixedPolynomial Field 1) :
    (indices : List Index) → FixedPolynomial Field ((indices.map exponents).sum)
  | [] => FixedPolynomial.constant ops.one
  | index :: rest =>
      if zeroExponent : exponents index = 0 then
        FixedPolynomial.widen ops.toOps
          (by simpa only [List.map_cons, List.sum_cons, zeroExponent, Nat.zero_add] using
            Nat.le_refl ((rest.map exponents).sum))
          (productPowers ops exponents images rest)
      else
        FixedPolynomial.mul ops.toOps
          (ProtocolPolynomialDegree.Sparse.affinePower ops (images index) (exponents index))
          (productPowers ops exponents images rest)

private theorem zeroExponent_product_coefficients (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (exponent : Nat) (isZero : exponent = 0)
    (image : FixedPolynomial Field 1) {degree : Nat} (right : FixedPolynomial Field degree) :
    (FixedPolynomial.mul ops.toOps
      (ProtocolPolynomialDegree.Sparse.affinePower ops image exponent) right).coefficients =
        right.coefficients := by
  subst exponent
  exact FixedPolynomial.constant_one_mul ops.toOps (polynomialLaws laws) laws.one_mul right

/-- Equality includes every declared high coefficient; no evaluation
injectivity or nonzero-factor premise is used. -/
theorem productPowers_value (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (exponents : Index → Nat) (images : Index → FixedPolynomial Field 1)
    (indices : List Index) :
    productPowers ops exponents images indices =
      ProtocolPolynomialDegree.Sparse.productPowers ops exponents images indices := by
  induction indices with
  | nil => rfl
  | cons index rest ih =>
      by_cases zeroExponent : exponents index = 0
      · apply coefficients_ext
        calc
          (productPowers ops exponents images (index :: rest)).coefficients =
              (productPowers ops exponents images rest).coefficients := by
            simp only [productPowers, zeroExponent, dite_true, FixedPolynomial.widen,
              List.map_cons, List.sum_cons, Nat.zero_add, Nat.sub_self,
              List.replicate_zero, List.append_nil]
          _ = (ProtocolPolynomialDegree.Sparse.productPowers ops exponents images rest).coefficients :=
            congrArg FixedPolynomial.coefficients ih
          _ = (ProtocolPolynomialDegree.Sparse.productPowers ops exponents images
                (index :: rest)).coefficients := by
            exact (zeroExponent_product_coefficients ops laws (exponents index) zeroExponent
              (images index) (ProtocolPolynomialDegree.Sparse.productPowers ops exponents images rest)).symm
      · simp only [productPowers, zeroExponent, dite_false,
          ProtocolPolynomialDegree.Sparse.productPowers, ih]

private theorem affinePower_zero (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (exponent : Nat) (positive : 0 < exponent) :
    ProtocolPolynomialDegree.Sparse.affinePower ops
        (FixedPolynomial.zero ops.toOps 1) exponent =
      FixedPolynomial.zero ops.toOps exponent := by
  cases exponent with
  | zero => omega
  | succ exponent =>
      unfold ProtocolPolynomialDegree.Sparse.affinePower
      rw [FixedPolynomial.power,
        FixedPolynomial.mul_zero_right ops.toOps (polynomialLaws laws)]
      exact widen_zero ops.toOps _

private theorem referenceProduct_zero (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (exponents : Index → Nat) (images : Index → FixedPolynomial Field 1)
    (indices : List Index) (index : Index) (member : index ∈ indices)
    (positive : 0 < exponents index)
    (zeroImage : (images index).coefficients = [ops.zero, ops.zero]) :
    ProtocolPolynomialDegree.Sparse.productPowers ops exponents images indices =
      FixedPolynomial.zero ops.toOps ((indices.map exponents).sum) := by
  induction indices with
  | nil => simp at member
  | cons head rest ih =>
      rcases List.mem_cons.mp member with equal | member
      · subst head
        have zeroPolynomial : images index = FixedPolynomial.zero ops.toOps 1 := by
          apply coefficients_ext
          exact zeroImage
        rw [ProtocolPolynomialDegree.Sparse.productPowers, zeroPolynomial,
          affinePower_zero ops laws _ positive,
          FixedPolynomial.mul_zero_left ops.toOps (polynomialLaws laws)]
        rfl
      · rw [ProtocolPolynomialDegree.Sparse.productPowers, ih member]
        exact FixedPolynomial.mul_zero_right ops.toOps (polynomialLaws laws) _ _

/-- This test examines exact coefficients, not satisfaction of an endpoint
constraint. A positive-power identically zero affine factor kills the term. -/
def zeroFactor [DecidableEq Field] (ops : InterpolationOps Field) {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial Field matrixCount)
    (images : Fin matrixCount → FixedPolynomial Field 1) : Bool :=
  (canonicalFinIndices matrixCount).any fun index =>
    decide (0 < monomial.exponents index ∧
      (images index).coefficients = [ops.zero, ops.zero])

private theorem zeroFactor_value [DecidableEq Field] (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {matrixCount : Nat}
    (monomial : CCSResidualTable.Monomial Field matrixCount)
    (images : Fin matrixCount → FixedPolynomial Field 1)
    (isZero : zeroFactor ops monomial images = true) :
    ProtocolPolynomialDegree.Sparse.monomialPolynomial ops monomial images =
      FixedPolynomial.zero ops.toOps monomial.totalDegree := by
  have existsZero : ∃ index ∈ canonicalFinIndices matrixCount,
      0 < monomial.exponents index ∧ (images index).coefficients = [ops.zero, ops.zero] := by
    simpa only [zeroFactor, List.any_eq_true, decide_eq_true_eq] using isZero
  rcases existsZero with ⟨index, member, positive, zeroImage⟩
  unfold ProtocolPolynomialDegree.Sparse.monomialPolynomial
  rw [referenceProduct_zero ops laws _ _ _ index member positive zeroImage]
  exact scale_zero_polynomial ops laws monomial.totalDegree monomial.coefficient

/-- Same monomial coefficient and degree as the sparse syntax owner. -/
def monomial (ops : InterpolationOps Field) {matrixCount : Nat}
    (term : CCSResidualTable.Monomial Field matrixCount)
    (images : Fin matrixCount → FixedPolynomial Field 1) :
    FixedPolynomial Field term.totalDegree :=
  FixedPolynomial.scale ops.toOps term.coefficient
    (productPowers ops term.exponents images (canonicalFinIndices matrixCount))

theorem monomial_value (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {matrixCount : Nat}
    (term : CCSResidualTable.Monomial Field matrixCount)
    (images : Fin matrixCount → FixedPolynomial Field 1) :
    monomial ops term images = ProtocolPolynomialDegree.Sparse.monomialPolynomial ops term images := by
  unfold monomial ProtocolPolynomialDegree.Sparse.monomialPolynomial
  rw [productPowers_value ops laws]

private def gatedTerms [DecidableEq Field] (ops : InterpolationOps Field) {matrixCount : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial Field matrixCount)
    (selector : FixedPolynomial Field 1)
    (images : Fin matrixCount → FixedPolynomial Field 1) :
    List { term // term ∈ polynomial.terms } →
      FixedPolynomial Field polynomial.canonicalEqualityGatedDegreeBound
  | [] => FixedPolynomial.zero ops.toOps polynomial.canonicalEqualityGatedDegreeBound
  | term :: rest =>
      if zeroFactor ops term.val images then gatedTerms ops polynomial selector images rest
      else FixedPolynomial.add ops.toOps
        (FixedPolynomial.widen ops.toOps (by
          have bound := polynomial.term_totalDegree_succ_le_canonicalEqualityGatedDegreeBound
            term.val term.property
          omega)
          (FixedPolynomial.mul ops.toOps selector (monomial ops term.val images)))
        (gatedTerms ops polynomial selector images rest)

private theorem gatedTerms_value [DecidableEq Field] (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {matrixCount : Nat}
    (polynomial : CCSResidualTable.ConstraintPolynomial Field matrixCount)
    (selector : FixedPolynomial Field 1) (images : Fin matrixCount → FixedPolynomial Field 1)
    (terms : List { term // term ∈ polynomial.terms }) :
    gatedTerms ops polynomial selector images terms =
      FixedPolynomial.sum ops.toOps terms (fun term =>
        FixedPolynomial.widen ops.toOps (by
          have bound := polynomial.term_totalDegree_succ_le_canonicalEqualityGatedDegreeBound
            term.val term.property
          omega)
          (FixedPolynomial.mul ops.toOps selector
            (ProtocolPolynomialDegree.Sparse.monomialPolynomial ops term.val images))) := by
  induction terms with
  | nil => rfl
  | cons term rest ih =>
      by_cases isZero : zeroFactor ops term.val images = true
      · rw [gatedTerms, if_pos isZero, FixedPolynomial.sum,
          zeroFactor_value ops laws _ _ isZero,
          FixedPolynomial.mul_zero_right ops.toOps (polynomialLaws laws), widen_zero,
          zero_add ops laws, ih]
      · rw [gatedTerms, if_neg isZero, FixedPolynomial.sum,
          monomial_value ops laws, ih]

/-- Exact fresh contribution with the original syntax, degree, gamma source
weights and selector. Affine input coefficients are prepared once per source. -/
def ccsPolynomialWithPowers [DecidableEq Field] (ops : InterpolationOps Field) {shape : Shape}
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (selector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    FixedPolynomial Field input.constraintPolynomial.canonicalEqualityGatedDegreeBound :=
  FixedPolynomial.sum ops.toOps (canonicalFinIndices shape.freshCount) fun source =>
    let images := Vector.ofFn fun matrix : Fin shape.matrixCount =>
      FixedPolynomial.affine (low.freshMatrixImage source matrix)
        (ops.sub (high.freshMatrixImage source matrix) (low.freshMatrixImage source matrix))
    FixedPolynomial.scale ops.toOps (powers source.val)
      (gatedTerms ops input.constraintPolynomial selector images.get input.constraintPolynomial.terms.attach)

/-- Total coefficient equality to the current first-round reference kernel.
No image correctness, row validity or zero-pattern premise is required. -/
theorem ccsPolynomialWithPowers_value [DecidableEq Field] (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {shape : Shape}
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (selector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape) :
    ccsPolynomialWithPowers ops input powers selector low high =
      PiCCSFirstRoundPair.ccsPolynomialWithPowers ops input powers selector low high := by
  unfold ccsPolynomialWithPowers PiCCSFirstRoundPair.ccsPolynomialWithPowers
  apply congrArg (FixedPolynomial.sum ops.toOps (canonicalFinIndices shape.freshCount))
  funext source
  dsimp only
  apply congrArg (FixedPolynomial.scale ops.toOps (powers source.val))
  rw [gatedTerms_value ops laws]
  have stored :
      (Vector.ofFn (fun matrix : Fin shape.matrixCount =>
        FixedPolynomial.affine (low.freshMatrixImage source matrix)
          (ops.sub (high.freshMatrixImage source matrix) (low.freshMatrixImage source matrix)))).get =
      (fun matrix => FixedPolynomial.affine (low.freshMatrixImage source matrix)
        (ops.sub (high.freshMatrixImage source matrix) (low.freshMatrixImage source matrix))) := by
    funext matrix
    change (Vector.ofFn _)[matrix.val] = _
    rw [Vector.getElem_ofFn]
  rw [stored]
  rfl

private theorem positive_exponent_of_totalDegree_pos {matrixCount : Nat}
    (term : CCSResidualTable.Monomial Field matrixCount) (positive : 0 < term.totalDegree) :
    ∃ index ∈ canonicalFinIndices matrixCount, 0 < term.exponents index := by
  have findPositive (indices : List (Fin matrixCount)) :
      0 < (indices.map term.exponents).sum →
        ∃ index ∈ indices, 0 < term.exponents index := by
    induction indices with
    | nil => simp
    | cons index rest ih =>
        intro sumPositive
        by_cases headPositive : 0 < term.exponents index
        · exact ⟨index, List.mem_cons_self, headPositive⟩
        · have headZero : term.exponents index = 0 := Nat.eq_zero_of_not_pos headPositive
          have tailPositive : 0 < (rest.map term.exponents).sum := by
            simpa only [List.map_cons, List.sum_cons, headZero, Nat.zero_add] using sumPositive
          obtain ⟨found, member, foundPositive⟩ := ih tailPositive
          exact ⟨found, List.mem_cons_of_mem index member, foundPositive⟩
  exact findPositive (canonicalFinIndices matrixCount) positive

private theorem zeroFactor_of_positive_zero_images [DecidableEq Field]
    (ops : InterpolationOps Field) {matrixCount : Nat}
    (term : CCSResidualTable.Monomial Field matrixCount)
    (images : Fin matrixCount → FixedPolynomial Field 1)
    (positive : 0 < term.totalDegree)
    (zeroImages : ∀ matrix, (images matrix).coefficients = [ops.zero, ops.zero]) :
    zeroFactor ops term images = true := by
  obtain ⟨matrix, member, exponentPositive⟩ := positive_exponent_of_totalDegree_pos term positive
  simp only [zeroFactor, List.any_eq_true, decide_eq_true_eq]
  exact ⟨matrix, member, exponentPositive, zeroImages matrix⟩

private theorem gatedTerms_zero [DecidableEq Field] (ops : InterpolationOps Field)
    {matrixCount : Nat} (polynomial : CCSResidualTable.ConstraintPolynomial Field matrixCount)
    (selector : FixedPolynomial Field 1) (images : Fin matrixCount → FixedPolynomial Field 1)
    (positive : ∀ term, term ∈ polynomial.terms → 0 < term.totalDegree)
    (zeroImages : ∀ matrix, (images matrix).coefficients = [ops.zero, ops.zero])
    (terms : List { term // term ∈ polynomial.terms }) :
    gatedTerms ops polynomial selector images terms =
      FixedPolynomial.zero ops.toOps polynomial.canonicalEqualityGatedDegreeBound := by
  induction terms with
  | nil => rfl
  | cons term rest ih =>
      rw [gatedTerms, if_pos (zeroFactor_of_positive_zero_images ops term.val images
        (positive term.val term.property) zeroImages), ih]

/-- A constraint polynomial with no degree-zero term contributes the exact
fixed-width zero when both fresh endpoints are zero. The source weights and
selector are arbitrary; no relation-satisfaction premise is used. -/
theorem ccsPolynomialWithPowers_zero [DecidableEq Field] (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {shape : Shape}
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (selector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape)
    (positive : ∀ term, term ∈ input.constraintPolynomial.terms → 0 < term.totalDegree)
    (lowZero : ∀ source matrix, low.freshMatrixImage source matrix = ops.zero)
    (highZero : ∀ source matrix, high.freshMatrixImage source matrix = ops.zero) :
    ccsPolynomialWithPowers ops input powers selector low high =
      FixedPolynomial.zero ops.toOps input.constraintPolynomial.canonicalEqualityGatedDegreeBound := by
  have zeroDifference : ops.sub ops.zero ops.zero = ops.zero :=
    (FiniteSumAlgebra.sub_eq_zero_iff ops laws _ _).mpr rfl
  have sourceZero (source : Fin shape.freshCount) :
      (let images := Vector.ofFn (fun matrix : Fin shape.matrixCount =>
          FixedPolynomial.affine (low.freshMatrixImage source matrix)
            (ops.sub (high.freshMatrixImage source matrix) (low.freshMatrixImage source matrix)));
        FixedPolynomial.scale ops.toOps (powers source.val)
          (gatedTerms ops input.constraintPolynomial selector images.get
            input.constraintPolynomial.terms.attach)) =
        FixedPolynomial.zero ops.toOps input.constraintPolynomial.canonicalEqualityGatedDegreeBound := by
    dsimp only
    have zeroImages (matrix : Fin shape.matrixCount) :
        ((Vector.ofFn fun selected : Fin shape.matrixCount =>
          FixedPolynomial.affine (low.freshMatrixImage source selected)
            (ops.sub (high.freshMatrixImage source selected)
              (low.freshMatrixImage source selected))).get matrix).coefficients =
          [ops.zero, ops.zero] := by
      change FixedPolynomial.coefficients (degree := 1) ((Vector.ofFn _)[matrix.val]) = _
      rw [Vector.getElem_ofFn, lowZero source matrix, highZero source matrix, zeroDifference]
      rfl
    rw [gatedTerms_zero ops input.constraintPolynomial selector _ positive zeroImages]
    exact scale_zero_polynomial ops laws _ _
  unfold ccsPolynomialWithPowers
  apply coefficient_ext
  intro index
  rw [PiCCSPolynomialRange.coefficient_sum, coefficient_zero]
  simp_rw [sourceZero, coefficient_zero]
  exact FiniteSumAlgebra.sumMap_zero ops laws _

/-- The same zero-suffix fact for the unchanged reference fresh kernel. -/
theorem reference_ccsPolynomialWithPowers_zero [DecidableEq Field] (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {shape : Shape}
    (input : ProtocolPolynomial.VerifierInput Field shape) (powers : Nat → Field)
    (selector : FixedPolynomial Field 1)
    (low high : ProtocolPolynomial.OutputMessage Field shape)
    (positive : ∀ term, term ∈ input.constraintPolynomial.terms → 0 < term.totalDegree)
    (lowZero : ∀ source matrix, low.freshMatrixImage source matrix = ops.zero)
    (highZero : ∀ source matrix, high.freshMatrixImage source matrix = ops.zero) :
    PiCCSFirstRoundPair.ccsPolynomialWithPowers ops input powers selector low high =
      FixedPolynomial.zero ops.toOps input.constraintPolynomial.canonicalEqualityGatedDegreeBound := by
  rw [← ccsPolynomialWithPowers_value ops laws]
  exact ccsPolynomialWithPowers_zero ops laws input powers selector low high positive lowZero highZero

end NightstreamFPrime.Export.Stage1.PiCCSFreshPolynomial
