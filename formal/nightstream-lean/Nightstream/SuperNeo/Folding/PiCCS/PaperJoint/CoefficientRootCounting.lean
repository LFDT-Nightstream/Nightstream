import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.FiniteRootCounting

/-!
Coefficient-owned finite root counting for constant-first polynomials.

Assurance tier: model-level.

Owns: synthetic division of the exact verifier-visible coefficient list, its
evaluation factorization, preservation of coefficient nonzeroness after
division by a root, and a root bound requiring only that some represented
coefficient is nonzero.

Does not own: challenge sampling, a protocol event, multivariate alpha
specialization, SumCheck, Fiat--Shamir, production refinement, Rust, R1CS,
artifacts, or costs.

Emits constraints: no.

| Object | Authority | Guarantee |
|---|---|---|
| polynomial | constant-first coefficient list | no function-valued oracle |
| quotient | suffix evaluations at the root | exact linear-factor identity |
| nonzero premise | one represented coefficient is nonzero | at most `degree` roots |
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CoefficientRootCounting

open Nightstream.SuperNeo.SumCheck.Finite

universe uField

/-- Every represented coefficient is zero. -/
def AllZero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (coefficients : List Field) : Prop :=
  forall coefficient, coefficient ∈ coefficients -> coefficient = ops.zero

/-- Synthetic quotient in constant-first order. For
`a_0 + X * T(X)`, the constant quotient coefficient is `T(anchor)` and the
remaining quotient is obtained recursively from `T`. -/
def quotient
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (anchor : Field) : List Field -> List Field
  | [] => []
  | _ :: [] => []
  | _ :: (next :: rest) =>
      Message.evaluateCoefficients ops.toOps anchor (next :: rest) ::
        quotient ops anchor (next :: rest)

theorem quotient_length
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (anchor : Field) :
    forall coefficients : List Field,
      (quotient ops anchor coefficients).length = coefficients.length.pred
  | [] => rfl
  | [_] => rfl
  | _ :: next :: rest => by
      simp only [quotient, List.length_cons, Nat.pred_succ]
      rw [quotient_length ops anchor (next :: rest)]
      simp

private theorem neg_zero
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops) :
    ops.neg ops.zero = ops.zero := by
  have cancelled := laws.add_neg ops.zero
  simpa only [laws.zero_add] using cancelled

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
      simpa only [InterpolationOps.sub] using congrArg
        (fun value => ops.add value right) zero
    _ = right := laws.zero_add right

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

private theorem right_plus_sub
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (left right : Field) :
    ops.add right (ops.sub left right) = left := by
  unfold InterpolationOps.sub
  calc
    ops.add right (ops.add left (ops.neg right)) =
        ops.add (ops.add right left) (ops.neg right) :=
      (laws.add_assoc _ _ _).symm
    _ = ops.add (ops.add left right) (ops.neg right) := by
      rw [laws.add_comm right left]
    _ = ops.add left (ops.add right (ops.neg right)) :=
      laws.add_assoc _ _ _
    _ = ops.add left ops.zero := by rw [laws.add_neg]
    _ = left := laws.add_zero left

private theorem difference_mul_add_anchor
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (point anchor value : Field) :
    ops.add
        (ops.mul (ops.sub point anchor) value)
        (ops.mul anchor value) =
      ops.mul point value := by
  unfold InterpolationOps.sub
  rw [laws.right_distrib]
  calc
    ops.add
        (ops.add (ops.mul point value)
          (ops.mul (ops.neg anchor) value))
        (ops.mul anchor value) =
      ops.add (ops.mul point value)
        (ops.add (ops.neg (ops.mul anchor value))
          (ops.mul anchor value)) := by
        rw [laws.neg_mul]
        exact laws.add_assoc _ _ _
    _ = ops.add (ops.mul point value) ops.zero := by
      rw [laws.add_comm (ops.neg (ops.mul anchor value))]
      rw [laws.add_neg]
    _ = ops.mul point value := laws.add_zero _

private theorem quotient_step_evaluation
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (point anchor leftTail rightTail quotientValue : Field)
    (tailDifference :
      ops.sub leftTail rightTail =
        ops.mul (ops.sub point anchor) quotientValue) :
    ops.add rightTail (ops.mul point quotientValue) =
      ops.add leftTail (ops.mul anchor quotientValue) := by
  calc
    ops.add rightTail (ops.mul point quotientValue) =
        ops.add rightTail
          (ops.add
            (ops.mul (ops.sub point anchor) quotientValue)
            (ops.mul anchor quotientValue)) := by
          rw [difference_mul_add_anchor laws]
    _ = ops.add
        (ops.add rightTail
          (ops.mul (ops.sub point anchor) quotientValue))
        (ops.mul anchor quotientValue) := by
          rw [laws.add_assoc]
    _ = ops.add
        (ops.add rightTail (ops.sub leftTail rightTail))
        (ops.mul anchor quotientValue) := by
          rw [tailDifference]
    _ = ops.add leftTail (ops.mul anchor quotientValue) := by
      rw [right_plus_sub laws]

private theorem affineDifference
    {Field : Type uField}
    {ops : InterpolationOps Field}
    (laws : InterpolationEvaluationLaws ops)
    (constant point anchor leftTail rightTail quotientValue : Field)
    (tailDifference :
      ops.sub leftTail rightTail =
        ops.mul (ops.sub point anchor) quotientValue) :
    ops.sub
        (ops.add constant (ops.mul point leftTail))
        (ops.add constant (ops.mul anchor rightTail)) =
      ops.mul (ops.sub point anchor)
        (ops.add leftTail (ops.mul anchor quotientValue)) := by
  letI : Std.Associative ops.add := ⟨laws.add_assoc⟩
  letI : Std.Commutative ops.add := ⟨laws.add_comm⟩
  letI : Std.Associative ops.mul := ⟨laws.mul_assoc⟩
  letI : Std.Commutative ops.mul := ⟨laws.mul_comm⟩
  calc
    ops.sub
        (ops.add constant (ops.mul point leftTail))
        (ops.add constant (ops.mul anchor rightTail)) =
      ops.add (ops.add constant (ops.mul point leftTail))
        (ops.add (ops.neg constant)
          (ops.neg (ops.mul anchor rightTail))) := by
          rw [InterpolationOps.sub, laws.neg_add]
    _ = ops.add (ops.mul point leftTail)
        (ops.neg (ops.mul anchor rightTail)) := by
      calc
        _ = ops.add (ops.add constant (ops.neg constant))
            (ops.add (ops.mul point leftTail)
              (ops.neg (ops.mul anchor rightTail))) := by ac_rfl
        _ = _ := by rw [laws.add_neg, laws.zero_add]
    _ = ops.add (ops.mul point leftTail)
        (ops.mul anchor (ops.neg rightTail)) := by
      rw [mul_neg laws]
    _ = ops.add
        (ops.mul (ops.sub point anchor) leftTail)
        (ops.mul anchor (ops.sub leftTail rightTail)) := by
      symm
      unfold InterpolationOps.sub
      rw [laws.right_distrib, laws.left_distrib, laws.neg_mul,
        mul_neg laws]
      calc
        _ = ops.add
            (ops.add (ops.mul point leftTail)
              (ops.neg (ops.mul anchor rightTail)))
            (ops.add (ops.neg (ops.mul anchor leftTail))
              (ops.mul anchor leftTail)) := by ac_rfl
        _ = _ := by
          rw [laws.add_comm (ops.neg (ops.mul anchor leftTail))]
          rw [laws.add_neg, laws.add_zero]
    _ = ops.add
        (ops.mul (ops.sub point anchor) leftTail)
        (ops.mul anchor
          (ops.mul (ops.sub point anchor) quotientValue)) := by
      rw [tailDifference]
    _ = ops.mul (ops.sub point anchor)
        (ops.add leftTail (ops.mul anchor quotientValue)) := by
      rw [laws.left_distrib]
      congr 1
      ac_rfl

/-- Synthetic division satisfies the exact evaluator identity on the raw
constant-first list. -/
theorem evaluate_quotient
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (coefficients : List Field)
    (anchor point : Field) :
    ops.sub
        (Message.evaluateCoefficients ops.toOps point coefficients)
        (Message.evaluateCoefficients ops.toOps anchor coefficients) =
      ops.mul (ops.sub point anchor)
        (Message.evaluateCoefficients ops.toOps point
          (quotient ops anchor coefficients)) := by
  induction coefficients with
  | nil =>
      simp [quotient, Message.evaluateCoefficients, sub_self laws,
        laws.mul_zero]
  | cons constant coefficients inductionHypothesis =>
      cases coefficients with
      | nil =>
          simp [quotient, Message.evaluateCoefficients, sub_self laws,
            laws.mul_zero]
      | cons next rest =>
          let tail := next :: rest
          let quotientValue :=
            Message.evaluateCoefficients ops.toOps point
              (quotient ops anchor tail)
          have tailDifference :
              ops.sub
                  (Message.evaluateCoefficients ops.toOps point tail)
                  (Message.evaluateCoefficients ops.toOps anchor tail) =
                ops.mul (ops.sub point anchor) quotientValue := by
            simpa [tail, quotientValue] using inductionHypothesis
          have quotientEvaluation :
              Message.evaluateCoefficients ops.toOps point
                  (quotient ops anchor (constant :: tail)) =
                ops.add
                  (Message.evaluateCoefficients ops.toOps anchor tail)
                  (ops.mul point quotientValue) := by
            rfl
          rw [quotientEvaluation]
          rw [quotient_step_evaluation ops laws point anchor
            (Message.evaluateCoefficients ops.toOps point tail)
            (Message.evaluateCoefficients ops.toOps anchor tail)
            quotientValue tailDifference]
          exact affineDifference laws constant point anchor
            (Message.evaluateCoefficients ops.toOps point tail)
            (Message.evaluateCoefficients ops.toOps anchor tail)
            quotientValue tailDifference

private theorem evaluate_eq_zero_of_allZero
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (point : Field) :
    forall coefficients : List Field,
      AllZero ops coefficients ->
      Message.evaluateCoefficients ops.toOps point coefficients = ops.zero
  | [], _ => rfl
  | coefficient :: coefficients, allZero => by
      have coefficientZero := allZero coefficient (by simp)
      have tailZero :
          AllZero ops coefficients := by
        intro prior member
        exact allZero prior (by simp [member])
      simp only [Message.evaluateCoefficients, coefficientZero,
        evaluate_eq_zero_of_allZero ops laws point coefficients tailZero,
        laws.mul_zero, laws.zero_add]

/-- A root together with an all-zero synthetic quotient forces every source
coefficient to be zero. -/
theorem allZero_of_root_and_quotient
    {Field : Type uField}
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (anchor : Field) :
    forall coefficients : List Field,
      Message.evaluateCoefficients ops.toOps anchor coefficients = ops.zero ->
      AllZero ops (quotient ops anchor coefficients) ->
      AllZero ops coefficients
  | [], _, _ => by simp [AllZero]
  | [constant], root, _ => by
      intro value member
      have valueEq : value = constant := by simpa using member
      subst value
      simpa [Message.evaluateCoefficients, laws.mul_zero,
        laws.add_zero] using root
  | constant :: next :: rest, root, quotientZero => by
      let tail := next :: rest
      have tailRoot :
          Message.evaluateCoefficients ops.toOps anchor tail = ops.zero := by
        exact quotientZero _ (by
          simp [quotient, tail])
      have tailQuotientZero :
          AllZero ops (quotient ops anchor tail) := by
        intro coefficient member
        exact quotientZero coefficient (by
          simp [quotient, tail, member])
      have tailZero :=
        allZero_of_root_and_quotient ops laws anchor tail tailRoot
          tailQuotientZero
      have constantZero : constant = ops.zero := by
        have tailEvaluationZero :=
          evaluate_eq_zero_of_allZero ops laws anchor tail tailZero
        change ops.add constant
            (ops.mul anchor
              (Message.evaluateCoefficients ops.toOps anchor tail)) =
          ops.zero at root
        rw [tailEvaluationZero, laws.mul_zero, laws.add_zero] at root
        exact root
      intro coefficient member
      rcases List.mem_cons.mp member with coefficientEq | tailMember
      · exact coefficientEq ▸ constantZero
      · exact tailZero coefficient tailMember

/-- Coefficient-level root counting. Unlike the older function-distinct
variant, the premise is exactly the protocol fact that not every represented
coefficient is zero. -/
theorem roots_count_le_degree
    {Field : Type uField}
    [DecidableEq Field]
    (ops : InterpolationOps Field)
    (laws : InterpolationEvaluationLaws ops)
    (noZeroDivisors : FiniteRootCounting.NoZeroDivisors ops)
    (degree : Nat)
    (coefficients : List Field)
    (coefficientCount : coefficients.length = degree + 1)
    (values : List Field)
    (valuesNodup : values.Nodup)
    (nonzero : Not (AllZero ops coefficients)) :
    values.countP (fun value =>
      decide
        (Message.evaluateCoefficients ops.toOps value coefficients =
          ops.zero)) <= degree := by
  induction degree generalizing coefficients values with
  | zero =>
      cases coefficients with
      | nil => simp at coefficientCount
      | cons coefficient coefficients =>
          cases coefficients with
          | nil =>
              have coefficientNonzero : coefficient ≠ ops.zero := by
                intro coefficientZero
                exact nonzero (by
                  simp [AllZero, coefficientZero])
              have noRoot : forall value,
                  Message.evaluateCoefficients ops.toOps value [coefficient] ≠
                    ops.zero := by
                intro value
                simpa [Message.evaluateCoefficients, laws.mul_zero,
                  laws.add_zero] using coefficientNonzero
              have countZero :
                  values.countP (fun value =>
                    decide
                      (Message.evaluateCoefficients ops.toOps value
                        [coefficient] = ops.zero)) = 0 := by
                rw [List.countP_eq_zero]
                intro value _member
                simpa using noRoot value
              exact Nat.le_of_eq countZero
          | cons next rest => simp at coefficientCount
  | succ degree degreeInduction =>
      induction values with
      | nil => simp
      | cons anchor values valuesInduction =>
          have nodupParts : anchor ∉ values /\ values.Nodup := by
            simpa using valuesNodup
          have anchorNotIn := nodupParts.1
          have valuesNodupTail := nodupParts.2
          by_cases anchorRoot :
              Message.evaluateCoefficients ops.toOps anchor coefficients =
                ops.zero
          · let quotientCoefficients := quotient ops anchor coefficients
            have quotientCount :
                quotientCoefficients.length = degree + 1 := by
              rw [quotient_length, coefficientCount]
              simp
            have quotientNonzero :
                Not (AllZero ops quotientCoefficients) := by
              intro quotientZero
              exact nonzero
                (allZero_of_root_and_quotient ops laws anchor coefficients
                  anchorRoot quotientZero)
            have quotientBound :
                values.countP (fun value =>
                    decide
                      (Message.evaluateCoefficients ops.toOps value
                        quotientCoefficients = ops.zero)) <= degree :=
              degreeInduction quotientCoefficients quotientCount values
                valuesNodupTail quotientNonzero
            have tailBound :
                values.countP (fun value =>
                    decide
                      (Message.evaluateCoefficients ops.toOps value
                        coefficients = ops.zero)) <=
                  values.countP (fun value =>
                    decide
                      (Message.evaluateCoefficients ops.toOps value
                        quotientCoefficients = ops.zero)) := by
              apply List.countP_mono_left
              intro value member valueRootBool
              have valueRoot :
                  Message.evaluateCoefficients ops.toOps value coefficients =
                    ops.zero := by
                simpa using valueRootBool
              have factor :=
                evaluate_quotient ops laws coefficients anchor value
              rw [anchorRoot, sub_zero laws, valueRoot] at factor
              rcases noZeroDivisors _ _ factor.symm with
                valueMinusAnchorZero | quotientRoot
              · have valueEq :=
                  sub_eq_zero_implies_eq laws valueMinusAnchorZero
                exact False.elim (anchorNotIn (valueEq ▸ member))
              · simpa [quotientCoefficients] using quotientRoot
            have headCount :
                (anchor :: values).countP (fun value =>
                    decide
                      (Message.evaluateCoefficients ops.toOps value
                        coefficients = ops.zero)) =
                  values.countP (fun value =>
                    decide
                      (Message.evaluateCoefficients ops.toOps value
                        coefficients = ops.zero)) + 1 := by
              simp [anchorRoot]
            rw [headCount]
            omega
          · have tailResult :=
              valuesInduction valuesNodupTail
            simpa [anchorRoot] using tailResult

end Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.CoefficientRootCounting
