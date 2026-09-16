import NightstreamFPrime.Export.Stage1.PiCCSFirstRound

/-! Numeric ranges of existing fixed-width polynomial objects. Adjacent ranges
merge by coefficient addition, including all high zero coefficients. No claim
uses injectivity of polynomial evaluation. -/

set_option autoImplicit false

namespace NightstreamFPrime.Export.Stage1.PiCCSPolynomialRange

open NightstreamFPrime.Spec.SumCheck.Finite
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint

universe uField
variable {Field : Type uField}

/-- Fixed polynomial equality follows from equality of every stored coefficient. -/
theorem coefficients_ext {degree : Nat}
    (left right : FixedPolynomial Field degree)
    (equal : left.coefficients = right.coefficients) : left = right := by
  cases left
  cases right
  cases equal
  rfl

/-- Reduce only one coefficient-list constructor at each induction step. -/
theorem add_coefficients (ops : Ops Field) {degree : Nat}
    (left right : FixedPolynomial Field degree) :
    (FixedPolynomial.add ops left right).coefficients =
      List.zipWith ops.add left.coefficients right.coefficients := by
  induction degree with
  | zero =>
      rcases left with ⟨left, leftLength⟩
      rcases right with ⟨right, rightLength⟩
      rcases List.length_eq_one_iff.mp leftLength with ⟨leftHead, rfl⟩
      rcases List.length_eq_one_iff.mp rightLength with ⟨rightHead, rfl⟩
      rfl
  | succ degree inductionHypothesis =>
      rcases left with ⟨left, leftLength⟩
      rcases right with ⟨right, rightLength⟩
      cases left with
      | nil => simp at leftLength
      | cons leftHead leftTail =>
          cases right with
          | nil => simp at rightLength
          | cons rightHead rightTail =>
              have leftTailLength : leftTail.length = degree + 1 := by
                simp only [List.length_cons] at leftLength
                omega
              have rightTailLength : rightTail.length = degree + 1 := by
                simp only [List.length_cons] at rightLength
                omega
              exact congrArg (List.cons (ops.add leftHead rightHead))
                (inductionHypothesis ⟨leftTail, leftTailLength⟩
                  ⟨rightTail, rightTailLength⟩)

/-- Scaling preserves the complete existing constant-first coefficient list. -/
theorem scale_coefficients (ops : Ops Field) {degree : Nat}
    (scalar : Field) (polynomial : FixedPolynomial Field degree) :
    (FixedPolynomial.scale ops scalar polynomial).coefficients =
      polynomial.coefficients.map (ops.mul scalar) := by
  induction degree with
  | zero =>
      rcases polynomial with ⟨coefficients, length⟩
      rcases List.length_eq_one_iff.mp length with ⟨head, rfl⟩
      rfl
  | succ degree inductionHypothesis =>
      rcases polynomial with ⟨coefficients, length⟩
      cases coefficients with
      | nil => simp at length
      | cons head tail =>
          have tailLength : tail.length = degree + 1 := by
            simp only [List.length_cons] at length
            omega
          exact congrArg (List.cons (ops.mul scalar head))
            (inductionHypothesis ⟨tail, tailLength⟩)

/-- Read one statically bounded coefficient; no new polynomial carrier. -/
def coefficient {degree : Nat} (polynomial : FixedPolynomial Field degree)
    (index : Fin (degree + 1)) : Field :=
  polynomial.coefficients[index.val]'(by
    rw [polynomial.coefficients_length]
    exact index.isLt)

/-- Equality of all stored coefficients determines the polynomial object. -/
theorem coefficient_ext {degree : Nat} (left right : FixedPolynomial Field degree)
    (equal : ∀ index, coefficient left index = coefficient right index) : left = right := by
  apply coefficients_ext
  apply List.ext_getElem (by rw [left.coefficients_length, right.coefficients_length])
  intro index leftBound rightBound
  have bounded : index < degree + 1 := by
    simpa only [left.coefficients_length] using leftBound
  exact equal ⟨index, bounded⟩

/-- Stored coefficient of the existing fixed-width zero. -/
theorem coefficient_zero (ops : Ops Field) (degree : Nat) (index : Fin (degree + 1)) :
    coefficient (FixedPolynomial.zero ops degree) index = ops.zero := by
  simp only [coefficient, FixedPolynomial.zero, List.getElem_replicate]

/-- Existing coefficient addition, without an evaluation injectivity premise. -/
theorem coefficient_add (ops : Ops Field) {degree : Nat}
    (left right : FixedPolynomial Field degree) (index : Fin (degree + 1)) :
    coefficient (FixedPolynomial.add ops left right) index =
      ops.add (coefficient left index) (coefficient right index) := by
  unfold coefficient
  simp only [add_coefficients, List.getElem_zipWith]

/-- Existing coefficient scaling, without an evaluation injectivity premise. -/
theorem coefficient_scale (ops : Ops Field) {degree : Nat}
    (scalar : Field) (polynomial : FixedPolynomial Field degree) (index : Fin (degree + 1)) :
    coefficient (FixedPolynomial.scale ops scalar polynomial) index =
      ops.mul scalar (coefficient polynomial index) := by
  unfold coefficient
  simp only [scale_coefficients, List.getElem_map]

/-- Each coefficient of the original polynomial sum is the same finite sum. -/
theorem coefficient_sum {Index : Type} (ops : InterpolationOps Field) {degree : Nat}
    (indices : List Index) (term : Index → FixedPolynomial Field degree)
    (index : Fin (degree + 1)) :
    coefficient (FixedPolynomial.sum ops.toOps indices term) index =
      FiniteSumAlgebra.sumMap ops indices (fun item => coefficient (term item) index) := by
  induction indices with
  | nil => exact coefficient_zero ops.toOps degree index
  | cons item indices inductionHypothesis =>
      rw [FixedPolynomial.sum, coefficient_add, inductionHypothesis]
      rfl

/-- Scaling the existing zero polynomial preserves every zero coefficient. -/
theorem scale_zero_polynomial (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (degree : Nat) (scalar : Field) :
    FixedPolynomial.scale ops.toOps scalar (FixedPolynomial.zero ops.toOps degree) =
      FixedPolynomial.zero ops.toOps degree := by
  apply coefficient_ext
  intro index
  rw [coefficient_scale, coefficient_zero]
  exact laws.mul_zero scalar

private theorem zipWith_add_assoc (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (left middle right : List Field) :
    List.zipWith ops.add (List.zipWith ops.add left middle) right =
      List.zipWith ops.add left (List.zipWith ops.add middle right) := by
  induction left generalizing middle right with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      cases middle with
      | nil => rfl
      | cons middleHead middleTail =>
          cases right with
          | nil => rfl
          | cons rightHead rightTail =>
              simp only [List.zipWith_cons_cons]
              rw [laws.add_assoc, inductionHypothesis]

private theorem zipWith_add_zero (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) (values : List Field) :
    List.zipWith ops.add values (List.replicate values.length ops.zero) = values := by
  induction values with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      simp only [List.length_cons, List.replicate_succ, List.zipWith_cons_cons,
        laws.add_zero, inductionHypothesis]

/-- Coefficient addition preserves associativity for complete polynomial objects. -/
theorem add_assoc (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (left middle right : FixedPolynomial Field degree) :
    FixedPolynomial.add ops.toOps (FixedPolynomial.add ops.toOps left middle) right =
      FixedPolynomial.add ops.toOps left (FixedPolynomial.add ops.toOps middle right) := by
  apply coefficients_ext
  simp only [add_coefficients]
  exact zipWith_add_assoc ops laws left.coefficients middle.coefficients right.coefficients

/-- Adding the existing fixed-width zero preserves the polynomial object. -/
theorem add_zero (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (polynomial : FixedPolynomial Field degree) :
    FixedPolynomial.add ops.toOps polynomial (FixedPolynomial.zero ops.toOps degree) =
      polynomial := by
  apply coefficients_ext
  rw [add_coefficients]
  change List.zipWith ops.add polynomial.coefficients
    (List.replicate (degree + 1) ops.zero) = polynomial.coefficients
  rw [← polynomial.coefficients_length]
  exact zipWith_add_zero ops laws polynomial.coefficients

/-- Add exactly `count` complete polynomial objects from ascending indices
`start` through `start + count - 1`, with no index-list allocation. -/
def range (ops : InterpolationOps Field) {degree : Nat} (start count : Nat)
    (term : Nat → FixedPolynomial Field degree) : FixedPolynomial Field degree :=
  Nat.fold count (fun index _ accumulated =>
    FixedPolynomial.add ops.toOps accumulated (term (start + index)))
    (FixedPolynomial.zero ops.toOps degree)

private theorem range_count_zero (ops : InterpolationOps Field) {degree : Nat}
    (start : Nat) (term : Nat → FixedPolynomial Field degree) :
    range ops start 0 term = FixedPolynomial.zero ops.toOps degree := rfl

private theorem range_succ (ops : InterpolationOps Field) {degree : Nat}
    (start count : Nat) (term : Nat → FixedPolynomial Field degree) :
    range ops start (count + 1) term =
      FixedPolynomial.add ops.toOps (range ops start count term) (term (start + count)) := by
  simp only [range, Nat.fold_succ]

/-- Independent adjacent ranges merge as exactly the same coefficient object.
No field-size or evaluation-injectivity premise is needed. -/
theorem range_append (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (start leftCount rightCount : Nat) (term : Nat → FixedPolynomial Field degree) :
    range ops start (leftCount + rightCount) term =
      FixedPolynomial.add ops.toOps (range ops start leftCount term)
        (range ops (start + leftCount) rightCount term) := by
  induction rightCount with
  | zero =>
      simp only [Nat.add_zero, range_count_zero]
      exact (add_zero ops laws _).symm
  | succ rightCount inductionHypothesis =>
      simp only [Nat.add_succ, range_succ, inductionHypothesis, Nat.add_assoc]
      exact add_assoc ops laws _ _ _

/-- Group the same complete polynomial range into adjacent fixed-width
ranges. Both traversals use the existing numeric fold, without an index list. -/
theorem range_group (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (start count width : Nat) (term : Nat → FixedPolynomial Field degree) :
    range ops start (count * width) term =
      range ops 0 count (fun block => range ops (start + block * width) width term) := by
  induction count with
  | zero => simp only [Nat.zero_mul, range_count_zero]
  | succ count inductionHypothesis =>
      rw [Nat.succ_mul, range_append ops laws start (count * width) width term, range_succ]
      simpa only [Nat.zero_add] using
        congrArg (fun polynomial => FixedPolynomial.add ops.toOps polynomial
          (range ops (start + count * width) width term)) inductionHypothesis

/-- An appended suffix may be omitted only when each of its original
polynomial objects is the existing fixed-width zero. -/
theorem range_append_zero (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (start count suffix : Nat) (term : Nat → FixedPolynomial Field degree)
    (zeroSuffix : ∀ index, start + count ≤ index → index < start + (count + suffix) →
      term index = FixedPolynomial.zero ops.toOps degree) :
    range ops start (count + suffix) term = range ops start count term := by
  revert zeroSuffix
  induction suffix with
  | zero =>
      intro _
      simp only [Nat.add_zero]
  | succ suffix inductionHypothesis =>
      intro zeroSuffix
      have previous := inductionHypothesis (fun index lower upper =>
        zeroSuffix index lower (by omega))
      have lastZero := zeroSuffix (start + (count + suffix)) (by omega) (by omega)
      rw [Nat.add_succ, range_succ, previous, lastZero, add_zero ops laws]

/-- The full zero-start range is the existing first-round coefficient object.
The private `polynomialSum` remains unchanged and need not be exposed. -/
theorem range_eq_firstRound (ops : InterpolationOps Field) {shape : Shape}
    (data : ProtocolPolynomial.Data Field shape)
    (alpha : CubePoint Field shape.cubeVariables) (gamma : Field) {remaining : Nat}
    (dimension : shape.cubeVariables = remaining + 1) :
    range ops 0 (2 ^ remaining)
        (PiCCSFirstRound.numericPair ops data alpha gamma dimension) =
      PiCCSFirstRound.firstRound ops data alpha gamma dimension := by
  unfold range
  simp only [Nat.zero_add]
  rfl

/-- Scaling commutes with the existing coefficient-object finite sum. -/
theorem scale_sum {Index : Type} (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (scalar : Field) (indices : List Index) (term : Index → FixedPolynomial Field degree) :
    FixedPolynomial.scale ops.toOps scalar (FixedPolynomial.sum ops.toOps indices term) =
      FixedPolynomial.sum ops.toOps indices (fun item =>
        FixedPolynomial.scale ops.toOps scalar (term item)) := by
  apply coefficient_ext
  intro index
  rw [coefficient_scale, coefficient_sum, coefficient_sum]
  simp only [coefficient_scale]
  exact (FiniteSumAlgebra.sumMap_mul_left ops laws scalar indices _).symm

private theorem foldl_add_eq_add_sum {Index : Type} (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (indices : List Index) (term : Index → FixedPolynomial Field degree)
    (initial : FixedPolynomial Field degree) :
    indices.foldl (fun accumulated index => FixedPolynomial.add ops.toOps accumulated (term index))
        initial =
      FixedPolynomial.add ops.toOps initial (FixedPolynomial.sum ops.toOps indices term) := by
  induction indices generalizing initial with
  | nil => exact (add_zero ops laws initial).symm
  | cons index indices inductionHypothesis =>
      rw [List.foldl_cons, inductionHypothesis]
      exact add_assoc ops laws initial (term index) _

/-- The numeric range and the canonical finite-index sum keep the same
ascending order and complete coefficient objects. -/
theorem range_eq_sum (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops) {degree : Nat}
    (start count : Nat) (term : Nat → FixedPolynomial Field degree) :
    range ops start count term =
      FixedPolynomial.sum ops.toOps (canonicalFinIndices count)
        (fun index => term (start + index.val)) := by
  unfold range
  rw [Nat.fold_eq_finRange_foldl]
  change (canonicalFinIndices count).foldl
      (fun accumulated index => FixedPolynomial.add ops.toOps accumulated (term (start + index.val)))
      (FixedPolynomial.zero ops.toOps degree) = _
  rw [foldl_add_eq_add_sum ops laws]
  apply coefficient_ext
  intro index
  simp only [coefficient_add, coefficient_zero, laws.zero_add]

end NightstreamFPrime.Export.Stage1.PiCCSPolynomialRange
