import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.BooleanEvaluation
import Nightstream.SuperNeo.SumCheck.FixedPolynomial

/-!
Finite root counting for the exact coefficient representation used by paper
`Pi_CCS` SumCheck.

Assurance tier: model-level.

Owns: a monic difference quotient for constant-first fixed polynomials, its
evaluation identity over the existing interpolation operations, and the
degree bound on collisions of two distinct represented polynomial functions
over an explicit duplicate-free finite challenge list.

Does not own: challenge sampling, SumCheck execution, Fiat--Shamir, a concrete
field instantiation, Rust, R1CS, artifacts, minimality, or costs.

Emits constraints: no.

| Owned object | Exact equation or bound |
|---|---|
| divided difference | `P(x) - P(r) = (x - r) * Q(x)` |
| root count | `roots.length <= degree` |
| collision count | `collisions.length <= degree` |

The sole extra algebraic premise is the paper field law that multiplication
has no zero divisors.  Polynomial coefficients and evaluations use exactly
the verifier's existing `Finite.Ops`; no second polynomial evaluator or
degree oracle is introduced.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting

open Nightstream.SuperNeo.SumCheck.Finite

universe uField

/-- The exact cancellation law used by univariate root counting. -/
def NoZeroDivisors
    {Field : Type uField}
    (ops : InterpolationOps Field) : Prop :=
  forall left right,
    ops.mul left right = ops.zero -> left = ops.zero \/ right = ops.zero

private theorem polynomialLaws
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops) :
    FixedPolynomial.Laws ops.toOps := {
  add_assoc := laws.add_assoc
  add_comm := laws.add_comm
  zero_add := laws.zero_add
  add_zero := laws.add_zero
  mul_assoc := laws.mul_assoc
  mul_comm := laws.mul_comm
  mul_zero := laws.mul_zero
  left_distrib := laws.left_distrib
  right_distrib := laws.right_distrib
}

private theorem neg_zero
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops) :
    ops.neg ops.zero = ops.zero := by
  have cancelled :
      ops.add ops.zero (ops.neg ops.zero) = ops.zero :=
    laws.add_neg ops.zero
  simpa only [laws.zero_add] using cancelled

private theorem mul_neg
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (left right : Field) :
    ops.mul left (ops.neg right) = ops.neg (ops.mul left right) := by
  calc
    ops.mul left (ops.neg right) =
        ops.mul (ops.neg right) left := laws.mul_comm _ _
    _ = ops.neg (ops.mul right left) := laws.neg_mul _ _
    _ = ops.neg (ops.mul left right) := by rw [laws.mul_comm right left]

private theorem sub_self
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (value : Field) :
    ops.sub value value = ops.zero :=
  laws.add_neg value

private theorem sub_zero
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (value : Field) :
    ops.sub value ops.zero = value := by
  simp only [InterpolationOps.sub, neg_zero laws, laws.add_zero]

private theorem sub_eq_zero_implies_eq
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    {left right : Field}
    (zero : ops.sub left right = ops.zero) :
    left = right := by
  calc
    left = ops.add left ops.zero := (laws.add_zero left).symm
    _ = ops.add left (ops.add (ops.neg right) right) := by
      rw [laws.add_comm (ops.neg right) right, laws.add_neg right]
    _ = ops.add (ops.add left (ops.neg right)) right := by
      rw [laws.add_assoc]
    _ = ops.add ops.zero right := by
      simpa only [InterpolationOps.sub] using congrArg (fun value =>
        ops.add value right) zero
    _ = right := laws.zero_add right

/-- Remove the constant coefficient from a positive-degree fixed polynomial.
The remaining coefficients have exactly the predecessor width. -/
def tail
    {Field : Type uField}
    {degree : Nat}
    (polynomial : FixedPolynomial Field (degree + 1)) :
    FixedPolynomial Field degree where
  coefficients := polynomial.coefficients.tail
  coefficients_length := by
    have length := polynomial.coefficients_length
    cases coefficients : polynomial.coefficients with
    | nil => simp [coefficients] at length
    | cons coefficient coefficients =>
        simp only [coefficients, List.tail_cons, List.length_cons,
          Nat.succ.injEq] at length ⊢
        omega

/-- The divided difference
`(p(X) - p(anchor)) / (X - anchor)`, represented at the exact predecessor
degree.  The constant case is total but is not used by the inductive root
step. -/
def dividedDifference
    {Field : Type uField}
    (ops : InterpolationOps Field) :
    (degree : Nat) ->
      FixedPolynomial Field degree -> Field ->
        FixedPolynomial Field degree.pred
  | 0, _, _ => FixedPolynomial.zero ops.toOps 0
  | degree + 1, polynomial, anchor =>
      let polynomialTail := tail polynomial
      let recursive := dividedDifference ops degree polynomialTail anchor
      FixedPolynomial.add ops.toOps polynomialTail
        (FixedPolynomial.widen ops.toOps (Nat.pred_le degree)
          (FixedPolynomial.scale ops.toOps anchor recursive))

private theorem evaluate_degree_zero
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (polynomial : FixedPolynomial Field 0)
    (left right : Field) :
    polynomial.evaluate ops.toOps left =
      polynomial.evaluate ops.toOps right := by
  rcases polynomial with ⟨coefficients, length⟩
  cases coefficients with
  | nil => simp at length
  | cons coefficient coefficients =>
      cases coefficients with
      | nil =>
          simp [FixedPolynomial.evaluate, FixedPolynomial.toMessage,
            Message.evaluate, Message.evaluateCoefficients, laws.mul_zero,
            laws.add_zero]
      | cons next rest => simp at length

private theorem affineDifference
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (constant leftPoint rightPoint leftTail rightTail quotient : Field)
    (tailDifference :
      ops.sub leftTail rightTail =
        ops.mul (ops.sub leftPoint rightPoint) quotient) :
    ops.sub
        (ops.add constant (ops.mul leftPoint leftTail))
        (ops.add constant (ops.mul rightPoint rightTail)) =
      ops.mul (ops.sub leftPoint rightPoint)
        (ops.add leftTail (ops.mul rightPoint quotient)) := by
  letI : Std.Associative ops.add := ⟨laws.add_assoc⟩
  letI : Std.Commutative ops.add := ⟨laws.add_comm⟩
  letI : Std.Associative ops.mul := ⟨laws.mul_assoc⟩
  letI : Std.Commutative ops.mul := ⟨laws.mul_comm⟩
  calc
    ops.sub
        (ops.add constant (ops.mul leftPoint leftTail))
        (ops.add constant (ops.mul rightPoint rightTail)) =
      ops.add (ops.add constant (ops.mul leftPoint leftTail))
        (ops.add (ops.neg constant)
          (ops.neg (ops.mul rightPoint rightTail))) := by
          rw [InterpolationOps.sub, laws.neg_add]
    _ = ops.add (ops.mul leftPoint leftTail)
        (ops.neg (ops.mul rightPoint rightTail)) := by
      calc
        _ = ops.add (ops.add constant (ops.neg constant))
            (ops.add (ops.mul leftPoint leftTail)
              (ops.neg (ops.mul rightPoint rightTail))) := by ac_rfl
        _ = _ := by rw [laws.add_neg, laws.zero_add]
    _ = ops.add (ops.mul leftPoint leftTail)
        (ops.mul rightPoint (ops.neg rightTail)) := by
      rw [mul_neg laws]
    _ = ops.add
        (ops.mul (ops.sub leftPoint rightPoint) leftTail)
        (ops.mul rightPoint (ops.sub leftTail rightTail)) := by
      symm
      unfold InterpolationOps.sub
      rw [laws.right_distrib, laws.left_distrib, laws.neg_mul,
        mul_neg laws]
      calc
        _ = ops.add
            (ops.add (ops.mul leftPoint leftTail)
              (ops.neg (ops.mul rightPoint rightTail)))
            (ops.add (ops.neg (ops.mul rightPoint leftTail))
              (ops.mul rightPoint leftTail)) := by ac_rfl
        _ = _ := by
          rw [laws.add_comm (ops.neg (ops.mul rightPoint leftTail))]
          rw [laws.add_neg, laws.add_zero]
    _ = ops.add
        (ops.mul (ops.sub leftPoint rightPoint) leftTail)
        (ops.mul rightPoint
          (ops.mul (ops.sub leftPoint rightPoint) quotient)) := by
      rw [tailDifference]
    _ = ops.mul (ops.sub leftPoint rightPoint)
        (ops.add leftTail (ops.mul rightPoint quotient)) := by
      rw [laws.left_distrib]
      congr 1
      ac_rfl

/-- The constructed divided difference has the exact evaluation identity. -/
theorem evaluate_dividedDifference
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (degree : Nat)
    (polynomial : FixedPolynomial Field degree)
    (anchor point : Field) :
    ops.sub
        (polynomial.evaluate ops.toOps point)
        (polynomial.evaluate ops.toOps anchor) =
      ops.mul (ops.sub point anchor)
        ((dividedDifference ops degree polynomial anchor).evaluate
          ops.toOps point) := by
  induction degree with
  | zero =>
      rw [evaluate_degree_zero laws polynomial point anchor]
      rw [sub_self laws]
      simp [dividedDifference, FixedPolynomial.evaluate_zero,
        polynomialLaws laws, laws.mul_zero]
  | succ degree inductionHypothesis =>
      let polynomialTail := tail polynomial
      have coefficientsNonempty : polynomial.coefficients ≠ [] := by
        intro empty
        have length := polynomial.coefficients_length
        simp [empty] at length
      rcases List.exists_cons_of_ne_nil coefficientsNonempty with
        ⟨constant, coefficients, coefficientsEq⟩
      have tailCoefficients :
          polynomialTail.coefficients = coefficients := by
        simp [polynomialTail, tail, coefficientsEq]
      have pointEvaluation :
          polynomial.evaluate ops.toOps point =
            ops.add constant
              (ops.mul point
                (polynomialTail.evaluate ops.toOps point)) := by
        simp only [FixedPolynomial.evaluate, FixedPolynomial.toMessage,
          Message.evaluate, coefficientsEq, Message.evaluateCoefficients]
        congr 2
        simp [polynomialTail, tailCoefficients]
      have anchorEvaluation :
          polynomial.evaluate ops.toOps anchor =
            ops.add constant
              (ops.mul anchor
                (polynomialTail.evaluate ops.toOps anchor)) := by
        simp only [FixedPolynomial.evaluate, FixedPolynomial.toMessage,
          Message.evaluate, coefficientsEq, Message.evaluateCoefficients]
        congr 2
        simp [polynomialTail, tailCoefficients]
      rw [pointEvaluation, anchorEvaluation]
      rw [dividedDifference]
      simp only [FixedPolynomial.evaluate_add, FixedPolynomial.evaluate_widen,
        FixedPolynomial.evaluate_scale, polynomialLaws laws]
      exact affineDifference laws constant point anchor
        (polynomialTail.evaluate ops.toOps point)
        (polynomialTail.evaluate ops.toOps anchor)
        ((dividedDifference ops degree polynomialTail anchor).evaluate
          ops.toOps point)
        (inductionHypothesis polynomialTail)

/-- A root factors the polynomial through the exact predecessor-degree
divided difference. -/
theorem evaluate_eq_rootFactor
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    {degree : Nat}
    (polynomial : FixedPolynomial Field (degree + 1))
    (anchor point : Field)
    (anchorRoot :
      polynomial.evaluate ops.toOps anchor = ops.zero) :
    polynomial.evaluate ops.toOps point =
      ops.mul (ops.sub point anchor)
        ((dividedDifference ops (degree + 1) polynomial anchor).evaluate
          ops.toOps point) := by
  have identity :=
    evaluate_dividedDifference ops laws (degree + 1) polynomial anchor point
  rw [anchorRoot, sub_zero laws] at identity
  exact identity

/-- A nonzero polynomial function of declared degree `degree` has at most
`degree` roots in an explicit duplicate-free finite list. -/
theorem roots_count_le_degree
    {Field : Type uField}
    [DecidableEq Field]
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (noZeroDivisors : NoZeroDivisors ops)
    (degree : Nat)
    (polynomial : FixedPolynomial Field degree)
    (values : List Field)
    (valuesNodup : values.Nodup)
    (nonzero :
      (fun point => polynomial.evaluate ops.toOps point) ≠
        fun _ => ops.zero) :
    values.countP (fun value =>
      decide (polynomial.evaluate ops.toOps value = ops.zero)) <= degree := by
  induction degree generalizing values with
  | zero =>
      have noRoot : forall value,
          polynomial.evaluate ops.toOps value ≠ ops.zero := by
        intro value valueZero
        apply nonzero
        funext point
        exact (evaluate_degree_zero laws polynomial point value).trans valueZero
      have rootsEmpty :
          values.countP (fun value =>
            decide
              (polynomial.evaluate ops.toOps value = ops.zero)) = 0 := by
        rw [List.countP_eq_zero]
        intro value _member
        simpa using noRoot value
      omega
  | succ degree degreeInduction =>
      induction values with
      | nil => simp
      | cons anchor values valuesInduction =>
          have nodupParts : anchor ∉ values ∧ values.Nodup := by
            simpa using valuesNodup
          have anchorNotIn : anchor ∉ values := nodupParts.1
          have valuesNodupTail : values.Nodup := nodupParts.2
          by_cases anchorRoot :
              polynomial.evaluate ops.toOps anchor = ops.zero
          · let quotient :=
              dividedDifference ops (degree + 1) polynomial anchor
            have quotientNonzero :
                (fun point => quotient.evaluate ops.toOps point) ≠
                  fun _ => ops.zero := by
              intro quotientZero
              apply nonzero
              funext point
              have factor :=
                evaluate_eq_rootFactor ops laws polynomial anchor point
                  anchorRoot
              rw [congrFun quotientZero point, laws.mul_zero] at factor
              exact factor
            have quotientBound :
                values.countP (fun value =>
                    decide (quotient.evaluate ops.toOps value = ops.zero)) <=
                  degree :=
              degreeInduction quotient values valuesNodupTail quotientNonzero
            have tailBound :
                values.countP (fun value =>
                    decide (polynomial.evaluate ops.toOps value = ops.zero)) <=
                  values.countP (fun value =>
                    decide (quotient.evaluate ops.toOps value = ops.zero)) := by
              apply List.countP_mono_left
              intro value member valueRootBool
              have valueRoot :
                  polynomial.evaluate ops.toOps value = ops.zero := by
                simpa using valueRootBool
              have factor :=
                evaluate_eq_rootFactor ops laws polynomial anchor value
                  anchorRoot
              rw [valueRoot] at factor
              rcases noZeroDivisors _ _ factor.symm with
                valueMinusAnchorZero | quotientZero
              · have valueEq :=
                  sub_eq_zero_implies_eq laws valueMinusAnchorZero
                exact False.elim (anchorNotIn (valueEq ▸ member))
              · simpa using quotientZero
            have headCount :
                (anchor :: values).countP (fun value =>
                    decide
                      (polynomial.evaluate ops.toOps value = ops.zero)) =
                  values.countP (fun value =>
                    decide
                      (polynomial.evaluate ops.toOps value = ops.zero)) + 1 := by
              simp [anchorRoot]
            rw [headCount]
            omega
          · have tailResult :=
              valuesInduction valuesNodupTail
            simpa [anchorRoot] using tailResult

/-- Two distinct fixed-degree polynomial functions collide on at most their
declared degree many points of a duplicate-free challenge list. -/
theorem collisions_count_le_degree
    {Field : Type uField}
    [DecidableEq Field]
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (noZeroDivisors : NoZeroDivisors ops)
    (degree : Nat)
    (left right : FixedPolynomial Field degree)
    (values : List Field)
    (valuesNodup : values.Nodup)
    (different :
      (fun point => left.evaluate ops.toOps point) ≠
        fun point => right.evaluate ops.toOps point) :
    values.countP (fun value =>
      decide
        (left.evaluate ops.toOps value =
          right.evaluate ops.toOps value)) <= degree := by
  let ringLaws := polynomialLaws laws
  let difference :=
    FixedPolynomial.add ops.toOps left
      (FixedPolynomial.scale ops.toOps (ops.neg ops.one) right)
  have differenceEvaluation (point : Field) :
      difference.evaluate ops.toOps point =
        ops.sub
          (left.evaluate ops.toOps point)
          (right.evaluate ops.toOps point) := by
    simp only [difference, FixedPolynomial.evaluate_add,
      FixedPolynomial.evaluate_scale, ringLaws]
    rw [laws.neg_mul, laws.one_mul]
    rfl
  have differenceNonzero :
      (fun point => difference.evaluate ops.toOps point) ≠
        fun _ => ops.zero := by
    intro differenceZero
    apply different
    funext point
    apply sub_eq_zero_implies_eq laws
    rw [← differenceEvaluation]
    exact congrFun differenceZero point
  have rootBound :=
    roots_count_le_degree ops laws noZeroDivisors degree difference values
      valuesNodup differenceNonzero
  calc
    values.countP (fun value =>
        decide
          (left.evaluate ops.toOps value =
            right.evaluate ops.toOps value)) =
      values.countP (fun value =>
        decide (difference.evaluate ops.toOps value = ops.zero)) := by
          apply List.countP_congr
          intro value _member
          constructor
          · intro equal
            simp only [decide_eq_true_eq] at equal ⊢
            rw [differenceEvaluation, equal, sub_self laws]
          · intro zero
            simp only [decide_eq_true_eq] at zero ⊢
            apply sub_eq_zero_implies_eq laws
            rw [← differenceEvaluation]
            exact zero
    _ <= degree := rootBound

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting
