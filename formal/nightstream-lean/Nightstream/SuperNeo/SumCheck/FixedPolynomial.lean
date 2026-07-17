import Nightstream.SuperNeo.SumCheck.Polynomial

/-!
Typed fixed-width polynomials for honest SumCheck round construction.

Owns: a degree-indexed constant-first coefficient carrier, its exact conversion
to the verifier-visible finite message, and the algebraic closure facts needed
to build honest round polynomials.

Does not own: prover messages, canonical trailing-coefficient checks, challenge
generation, SumCheck acceptance, protocol polynomials, Rust, R1CS, or costs.

Emits constraints: no.

| Object | Mathematical obligation | Lean owner |
|---|---|---|
| fixed polynomial | exactly `degree + 1` constant-first coefficients | `FixedPolynomial` |
| verifier projection | conversion preserves width and Horner evaluation | `toMessage_coefficients_length`, `evaluate_eq_message_evaluate` |
| exact degree widening | append only verifier-visible high zero coefficients | `evaluate_widen` |
| algebraic closure | constant, affine, add, scale, convolution, natural power, finite sum | `evaluate_constant`, `evaluate_affine`, `evaluate_add`, `evaluate_scale`, `evaluate_mul`, `evaluate_power`, `evaluate_sum` |
-/

namespace Nightstream.SuperNeo.SumCheck.Finite

universe uField uIndex

/-- A constant-first coefficient list whose width is fixed by its degree. -/
structure FixedPolynomial (Field : Type uField) (degree : Nat) where
  coefficients : List Field
  coefficients_length : coefficients.length = degree + 1

namespace FixedPolynomial

/-- Algebraic laws needed to interpret coefficient operations as polynomial
operations. They are attached to the same verifier-visible operations used by
`Message.evaluate`; no separate evaluator is trusted. -/
structure Laws
    {Field : Type uField}
    (ops : Ops Field) : Prop where
  add_assoc : forall left middle right,
    ops.add (ops.add left middle) right =
      ops.add left (ops.add middle right)
  add_comm : forall left right, ops.add left right = ops.add right left
  zero_add : forall value, ops.add ops.zero value = value
  add_zero : forall value, ops.add value ops.zero = value
  mul_assoc : forall left middle right,
    ops.mul (ops.mul left middle) right =
      ops.mul left (ops.mul middle right)
  mul_comm : forall left right, ops.mul left right = ops.mul right left
  mul_zero : forall value, ops.mul value ops.zero = ops.zero
  left_distrib : forall left middle right,
    ops.mul left (ops.add middle right) =
      ops.add (ops.mul left middle) (ops.mul left right)
  right_distrib : forall left middle right,
    ops.mul (ops.add left middle) right =
      ops.add (ops.mul left right) (ops.mul middle right)

/-- Forget the static width while preserving constant-first order exactly. -/
def toMessage
    {Field : Type uField}
    {degree : Nat}
    (polynomial : FixedPolynomial Field degree) : Message Field where
  coefficients := polynomial.coefficients

/-- Evaluate with the same constant-first Horner machine as the finite
verifier. -/
def evaluate
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (polynomial : FixedPolynomial Field degree)
    (point : Field) : Field :=
  polynomial.toMessage.evaluate ops point

/-- Conversion to a raw message preserves the statically declared width. -/
@[simp] theorem toMessage_coefficients_length
    {Field : Type uField}
    {degree : Nat}
    (polynomial : FixedPolynomial Field degree) :
    polynomial.toMessage.coefficients.length = degree + 1 :=
  polynomial.coefficients_length

/-- The raw verifier derives exactly the declared degree from that width. -/
@[simp] theorem toMessage_degreeUpperBound
    {Field : Type uField}
    {degree : Nat}
    (polynomial : FixedPolynomial Field degree) :
    polynomial.toMessage.degreeUpperBound = degree := by
  simp [Message.degreeUpperBound]

/-- Fixed-width evaluation is definitionally the verifier-visible message
evaluation after conversion. -/
theorem evaluate_eq_message_evaluate
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (polynomial : FixedPolynomial Field degree)
    (point : Field) :
    polynomial.evaluate ops point =
      polynomial.toMessage.evaluate ops point :=
  rfl

private def addCoefficients
    {Field : Type uField}
    (ops : Ops Field) : List Field -> List Field -> List Field
  | [], right => right
  | left, [] => left
  | left :: lefts, right :: rights =>
      ops.add left right :: addCoefficients ops lefts rights

private theorem addCoefficients_length
    {Field : Type uField}
    (ops : Ops Field) :
    forall {left right : List Field},
      left.length = right.length ->
      (addCoefficients ops left right).length = left.length
  | [], [], _ => rfl
  | [], _ :: _, equal => by simp at equal
  | _ :: _, [], equal => by simp at equal
  | _ :: lefts, _ :: rights, equal => by
      simp only [List.length_cons, Nat.succ.injEq] at equal
      simp [addCoefficients, addCoefficients_length ops equal]

private def scaleCoefficients
    {Field : Type uField}
    (ops : Ops Field)
    (scalar : Field) : List Field -> List Field
  | [] => []
  | coefficient :: coefficients =>
      ops.mul scalar coefficient :: scaleCoefficients ops scalar coefficients

private theorem scaleCoefficients_length
    {Field : Type uField}
    (ops : Ops Field)
    (scalar : Field)
    (coefficients : List Field) :
    (scaleCoefficients ops scalar coefficients).length = coefficients.length := by
  induction coefficients with
  | nil => rfl
  | cons coefficient coefficients inductionHypothesis =>
      simp [scaleCoefficients, inductionHypothesis]

/-- Constant-first schoolbook convolution. Padding is explicit: multiplying
two nonempty lists of lengths `m` and `n` produces exactly `m + n - 1`
coefficients, even when the highest coefficient happens to be zero. -/
private def convolution
    {Field : Type uField}
    (ops : Ops Field) : List Field -> List Field -> List Field
  | [], _ => []
  | coefficient :: coefficients, right =>
      addCoefficients ops
        (scaleCoefficients ops coefficient right)
        (ops.zero :: convolution ops coefficients right)

private theorem addCoefficients_length_eq_max
    {Field : Type uField}
    (ops : Ops Field) :
    forall left right : List Field,
      (addCoefficients ops left right).length =
        max left.length right.length
  | [], right => by simp [addCoefficients]
  | _ :: lefts, [] => rfl
  | _ :: lefts, _ :: rights => by
      simp only [addCoefficients, List.length_cons,
        addCoefficients_length_eq_max ops lefts rights]
      omega

private theorem convolution_length
    {Field : Type uField}
    (ops : Ops Field)
    (left right : List Field)
    (leftNonempty : left.length > 0)
    (rightNonempty : right.length > 0) :
    (convolution ops left right).length =
      left.length + right.length - 1 := by
  induction left with
  | nil => simp at leftNonempty
  | cons coefficient coefficients inductionHypothesis =>
      cases coefficients with
      | nil =>
          simp only [convolution, addCoefficients_length_eq_max,
            scaleCoefficients_length, List.length_cons, List.length_nil,
            Nat.zero_add]
          omega
      | cons next rest =>
          have tailNonempty : (next :: rest).length > 0 := by simp
          rw [convolution]
          rw [addCoefficients_length_eq_max, scaleCoefficients_length]
          rw [List.length_cons, inductionHypothesis tailNonempty]
          simp only [List.length_cons]
          omega

/-- The zero polynomial at an arbitrary declared degree. -/
def zero
    {Field : Type uField}
    (ops : Ops Field)
    (degree : Nat) : FixedPolynomial Field degree where
  coefficients := List.replicate (degree + 1) ops.zero
  coefficients_length := by simp

/-- A degree-zero constant polynomial. -/
def constant
    {Field : Type uField}
    (value : Field) : FixedPolynomial Field 0 where
  coefficients := [value]
  coefficients_length := rfl

/-- A degree-one polynomial `constant + point * linear`. -/
def affine
    {Field : Type uField}
    (constant linear : Field) : FixedPolynomial Field 1 where
  coefficients := [constant, linear]
  coefficients_length := rfl

/-- Pointwise coefficient addition at one fixed degree. -/
def add
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (left right : FixedPolynomial Field degree) :
    FixedPolynomial Field degree where
  coefficients := addCoefficients ops left.coefficients right.coefficients
  coefficients_length := by
    rw [addCoefficients_length ops]
    · exact left.coefficients_length
    · rw [left.coefficients_length, right.coefficients_length]

/-- Left scalar multiplication at one fixed degree. -/
def scale
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (scalar : Field)
    (polynomial : FixedPolynomial Field degree) :
    FixedPolynomial Field degree where
  coefficients := scaleCoefficients ops scalar polynomial.coefficients
  coefficients_length := by
    rw [scaleCoefficients_length, polynomial.coefficients_length]

/-- Schoolbook multiplication. Declared degrees add and no high coefficient
is trimmed. -/
def mul
    {Field : Type uField}
    {leftDegree rightDegree : Nat}
    (ops : Ops Field)
    (left : FixedPolynomial Field leftDegree)
    (right : FixedPolynomial Field rightDegree) :
    FixedPolynomial Field (leftDegree + rightDegree) where
  coefficients := convolution ops left.coefficients right.coefficients
  coefficients_length := by
    rw [convolution_length ops left.coefficients right.coefficients]
    · rw [left.coefficients_length, right.coefficients_length]
      omega
    · rw [left.coefficients_length]
      omega
    · rw [right.coefficients_length]
      omega

/-- Exponentiation by repeated exact-width multiplication. The declared
degree is the source degree times the exponent, including all high zero
slots that arise for a lower actual degree. -/
def power
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (polynomial : FixedPolynomial Field degree) :
    (exponent : Nat) -> FixedPolynomial Field (degree * exponent)
  | 0 => constant ops.one
  | exponent + 1 => mul ops (power ops polynomial exponent) polynomial

/-- Scalar exponentiation in the exact multiplication order used by
`power`. -/
def valuePower
    {Field : Type uField}
    (ops : Ops Field)
    (value : Field) : Nat -> Field
  | 0 => ops.one
  | exponent + 1 => ops.mul (valuePower ops value exponent) value

/-- Right-associated sum over an explicit finite index list. -/
def sum
    {Field : Type uField}
    {Index : Type uIndex}
    {degree : Nat}
    (ops : Ops Field)
    (indices : List Index)
    (value : Index -> FixedPolynomial Field degree) :
    FixedPolynomial Field degree :=
  match indices with
  | [] => zero ops degree
  | index :: indices => add ops (value index) (sum ops indices value)

/-- Widen a polynomial to a larger declared degree by appending only high
zero coefficients. The proof argument is semantic authority for the target
width; the coefficient payload remains completely verifier-visible. -/
def widen
    {Field : Type uField}
    {degree target : Nat}
    (ops : Ops Field)
    (degree_le_target : degree <= target)
    (polynomial : FixedPolynomial Field degree) :
    FixedPolynomial Field target where
  coefficients := polynomial.coefficients ++
    List.replicate (target - degree) ops.zero
  coefficients_length := by
    rw [List.length_append, polynomial.coefficients_length,
      List.length_replicate]
    omega

private theorem evaluateCoefficients_zero_replicate
    {Field : Type uField}
    (ops : Ops Field)
    (laws : Laws ops)
    (point : Field) :
    forall width : Nat,
      Message.evaluateCoefficients ops point
          (List.replicate width ops.zero) = ops.zero
  | 0 => rfl
  | width + 1 => by
      simp only [List.replicate_succ, Message.evaluateCoefficients]
      rw [evaluateCoefficients_zero_replicate ops laws point width,
        laws.mul_zero, laws.zero_add]

private theorem evaluateCoefficients_append_zero_replicate
    {Field : Type uField}
    (ops : Ops Field)
    (laws : Laws ops)
    (point : Field) :
    forall coefficients : List Field, forall width : Nat,
      Message.evaluateCoefficients ops point
          (coefficients ++ List.replicate width ops.zero) =
        Message.evaluateCoefficients ops point coefficients
  | [], width => by
      simp only [List.nil_append]
      exact evaluateCoefficients_zero_replicate ops laws point width
  | coefficient :: coefficients, width => by
      simp only [List.cons_append, Message.evaluateCoefficients]
      rw [evaluateCoefficients_append_zero_replicate ops laws point
        coefficients width]

private theorem evaluateCoefficients_add
    {Field : Type uField}
    (ops : Ops Field)
    (laws : Laws ops)
    (point : Field) :
    forall left right : List Field,
      Message.evaluateCoefficients ops point
          (addCoefficients ops left right) =
        ops.add
          (Message.evaluateCoefficients ops point left)
          (Message.evaluateCoefficients ops point right)
  | [], right => (laws.zero_add _).symm
  | left :: lefts, [] => (laws.add_zero _).symm
  | left :: lefts, right :: rights => by
      simp only [addCoefficients, Message.evaluateCoefficients]
      rw [evaluateCoefficients_add ops laws point lefts rights]
      calc
        ops.add (ops.add left right)
            (ops.mul point
              (ops.add
                (Message.evaluateCoefficients ops point lefts)
                (Message.evaluateCoefficients ops point rights))) =
          ops.add (ops.add left right)
            (ops.add
              (ops.mul point
                (Message.evaluateCoefficients ops point lefts))
              (ops.mul point
                (Message.evaluateCoefficients ops point rights))) := by
            rw [laws.left_distrib]
        _ = ops.add
            (ops.add left
              (ops.mul point
                (Message.evaluateCoefficients ops point lefts)))
            (ops.add right
              (ops.mul point
                (Message.evaluateCoefficients ops point rights))) := by
          rw [laws.add_assoc left right]
          rw [(laws.add_assoc right
            (ops.mul point
              (Message.evaluateCoefficients ops point lefts))
            (ops.mul point
              (Message.evaluateCoefficients ops point rights))).symm]
          rw [laws.add_comm right
            (ops.mul point
              (Message.evaluateCoefficients ops point lefts))]
          rw [laws.add_assoc
            (ops.mul point
              (Message.evaluateCoefficients ops point lefts))]
          rw [(laws.add_assoc left
            (ops.mul point
              (Message.evaluateCoefficients ops point lefts))
            (ops.add right
              (ops.mul point
                (Message.evaluateCoefficients ops point rights)))).symm]

private theorem evaluateCoefficients_scale
    {Field : Type uField}
    (ops : Ops Field)
    (laws : Laws ops)
    (scalar point : Field) :
    forall coefficients : List Field,
      Message.evaluateCoefficients ops point
          (scaleCoefficients ops scalar coefficients) =
        ops.mul scalar
          (Message.evaluateCoefficients ops point coefficients)
  | [] => (laws.mul_zero scalar).symm
  | coefficient :: coefficients => by
      simp only [scaleCoefficients, Message.evaluateCoefficients]
      rw [evaluateCoefficients_scale ops laws scalar point coefficients]
      calc
        ops.add (ops.mul scalar coefficient)
            (ops.mul point
              (ops.mul scalar
                (Message.evaluateCoefficients ops point coefficients))) =
          ops.add (ops.mul scalar coefficient)
            (ops.mul scalar
              (ops.mul point
                (Message.evaluateCoefficients ops point coefficients))) := by
            rw [(laws.mul_assoc point scalar _).symm,
              laws.mul_comm point scalar,
              laws.mul_assoc]
        _ = ops.mul scalar
            (ops.add coefficient
              (ops.mul point
                (Message.evaluateCoefficients ops point coefficients))) :=
          (laws.left_distrib scalar coefficient _).symm

private theorem evaluateCoefficients_shift
    {Field : Type uField}
    (ops : Ops Field)
    (laws : Laws ops)
    (point : Field)
    (coefficients : List Field) :
    Message.evaluateCoefficients ops point (ops.zero :: coefficients) =
      ops.mul point
        (Message.evaluateCoefficients ops point coefficients) := by
  simp [Message.evaluateCoefficients, laws.zero_add]

private theorem evaluateCoefficients_convolution
    {Field : Type uField}
    (ops : Ops Field)
    (laws : Laws ops)
    (point : Field) :
    forall left right : List Field,
      left.length > 0 ->
      right.length > 0 ->
      Message.evaluateCoefficients ops point
          (convolution ops left right) =
        ops.mul
          (Message.evaluateCoefficients ops point left)
          (Message.evaluateCoefficients ops point right)
  | [], _, leftNonempty, _ => by simp at leftNonempty
  | coefficient :: coefficients, right, _, rightNonempty => by
      rw [convolution]
      rw [evaluateCoefficients_add ops laws point]
      rw [evaluateCoefficients_scale ops laws]
      rw [evaluateCoefficients_shift ops laws]
      cases coefficients with
      | nil =>
          simp only [convolution, Message.evaluateCoefficients]
          rw [laws.mul_zero, laws.add_zero, laws.add_zero]
      | cons next rest =>
          rw [evaluateCoefficients_convolution ops laws point
            (next :: rest) right (by simp) rightNonempty]
          simp only [Message.evaluateCoefficients]
          calc
            ops.add
                (ops.mul coefficient
                  (Message.evaluateCoefficients ops point right))
                (ops.mul point
                  (ops.mul
                    (Message.evaluateCoefficients ops point (next :: rest))
                    (Message.evaluateCoefficients ops point right))) =
              ops.add
                (ops.mul coefficient
                  (Message.evaluateCoefficients ops point right))
                (ops.mul
                  (ops.mul point
                    (Message.evaluateCoefficients ops point (next :: rest)))
                  (Message.evaluateCoefficients ops point right)) := by
                rw [laws.mul_assoc]
            _ = ops.mul
                (ops.add coefficient
                  (ops.mul point
                    (Message.evaluateCoefficients ops point (next :: rest))))
                (Message.evaluateCoefficients ops point right) :=
              (laws.right_distrib _ _ _).symm

/-- Every fixed-width zero polynomial evaluates to zero. -/
@[simp] theorem evaluate_zero
    {Field : Type uField}
    (ops : Ops Field)
    (laws : Laws ops)
    (degree : Nat)
    (point : Field) :
    (zero ops degree).evaluate ops point = ops.zero := by
  exact evaluateCoefficients_zero_replicate ops laws point (degree + 1)

/-- Degree widening preserves the polynomial function exactly. In
particular, an honest lower-degree polynomial may inhabit a fixed wider
verifier message without a canonical-high-coefficient rejection rule. -/
@[simp] theorem evaluate_widen
    {Field : Type uField}
    {degree target : Nat}
    (ops : Ops Field)
    (laws : Laws ops)
    (degree_le_target : degree <= target)
    (polynomial : FixedPolynomial Field degree)
    (point : Field) :
    (widen ops degree_le_target polynomial).evaluate ops point =
      polynomial.evaluate ops point := by
  exact evaluateCoefficients_append_zero_replicate ops laws point
    polynomial.coefficients (target - degree)

/-- Constant construction has its intended semantics. -/
@[simp] theorem evaluate_constant
    {Field : Type uField}
    (ops : Ops Field)
    (laws : Laws ops)
    (value point : Field) :
    (constant value).evaluate ops point = value := by
  simp [evaluate, toMessage, constant, Message.evaluate,
    Message.evaluateCoefficients, laws.mul_zero, laws.add_zero]

/-- Affine construction uses constant-first order. -/
@[simp] theorem evaluate_affine
    {Field : Type uField}
    (ops : Ops Field)
    (laws : Laws ops)
    (constant linear point : Field) :
    (affine constant linear).evaluate ops point =
      ops.add constant (ops.mul point linear) := by
  simp [evaluate, toMessage, affine, Message.evaluate,
    Message.evaluateCoefficients, laws.mul_zero, laws.add_zero]

/-- Fixed-degree coefficient addition evaluates as field addition. -/
@[simp] theorem evaluate_add
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (laws : Laws ops)
    (left right : FixedPolynomial Field degree)
    (point : Field) :
    (add ops left right).evaluate ops point =
      ops.add (left.evaluate ops point) (right.evaluate ops point) := by
  exact evaluateCoefficients_add ops laws point
    left.coefficients right.coefficients

/-- Fixed-degree scalar multiplication evaluates as scalar multiplication. -/
@[simp] theorem evaluate_scale
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (laws : Laws ops)
    (scalar : Field)
    (polynomial : FixedPolynomial Field degree)
    (point : Field) :
    (scale ops scalar polynomial).evaluate ops point =
      ops.mul scalar (polynomial.evaluate ops point) := by
  exact evaluateCoefficients_scale ops laws scalar point
    polynomial.coefficients

/-- Exact-width convolution evaluates as multiplication. -/
@[simp] theorem evaluate_mul
    {Field : Type uField}
    {leftDegree rightDegree : Nat}
    (ops : Ops Field)
    (laws : Laws ops)
    (left : FixedPolynomial Field leftDegree)
    (right : FixedPolynomial Field rightDegree)
    (point : Field) :
    (mul ops left right).evaluate ops point =
      ops.mul (left.evaluate ops point) (right.evaluate ops point) := by
  apply evaluateCoefficients_convolution ops laws point
  · rw [left.coefficients_length]
    omega
  · rw [right.coefficients_length]
    omega

/-- Exact-width polynomial powers evaluate as repeated scalar
multiplication. -/
@[simp] theorem evaluate_power
    {Field : Type uField}
    {degree : Nat}
    (ops : Ops Field)
    (laws : Laws ops)
    (polynomial : FixedPolynomial Field degree)
    (exponent : Nat)
    (point : Field) :
    (power ops polynomial exponent).evaluate ops point =
      valuePower ops (polynomial.evaluate ops point) exponent := by
  induction exponent with
  | zero =>
      exact evaluate_constant ops laws ops.one point
  | succ exponent inductionHypothesis =>
      rw [power, evaluate_mul ops laws, inductionHypothesis]
      rfl

/-- Evaluation commutes with the explicit finite polynomial sum. -/
@[simp] theorem evaluate_sum
    {Field : Type uField}
    {Index : Type uIndex}
    {degree : Nat}
    (ops : Ops Field)
    (laws : Laws ops)
    (indices : List Index)
    (value : Index -> FixedPolynomial Field degree)
    (point : Field) :
    (sum ops indices value).evaluate ops point =
      indices.foldr
        (fun index total => ops.add ((value index).evaluate ops point) total)
        ops.zero := by
  induction indices with
  | nil => exact evaluate_zero ops laws degree point
  | cons index indices inductionHypothesis =>
      rw [sum, evaluate_add ops laws, inductionHypothesis]
      rfl

end FixedPolynomial

end Nightstream.SuperNeo.SumCheck.Finite
