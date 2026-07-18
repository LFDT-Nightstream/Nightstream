import Nightstream.SuperNeo.SumCheck.FixedPolynomial

/-!
Focused semantic regressions for typed fixed-width SumCheck polynomials.

| Property | Failure caught |
|---|---|
| message width and degree are exact | coefficient trimming or padding drift |
| affine order is constant-first | verifier/semantic ordering mismatch |
| degree widening appends only high zeros | hidden trimming or changed evaluation |
| convolution preserves coefficients and evaluation | unsound honest-round multiplication |
| natural powers carry exact multiplied degree | sparse-monomial degree drift |
| explicit finite sums preserve evaluation | dropped or duplicated summands |
-/

namespace NightstreamTests.SumCheckFixedPolynomial

open Nightstream.SuperNeo.SumCheck.Finite
open Nightstream.SuperNeo.SumCheck.Finite.FixedPolynomial

private def natOps : Ops Nat where
  zero := 0
  one := 1
  add := Nat.add
  mul := Nat.mul

private def natLaws : Laws natOps where
  add_assoc := Nat.add_assoc
  add_comm := Nat.add_comm
  zero_add := Nat.zero_add
  add_zero := Nat.add_zero
  mul_assoc := Nat.mul_assoc
  mul_comm := Nat.mul_comm
  mul_zero := Nat.mul_zero
  left_distrib := Nat.mul_add
  right_distrib := Nat.add_mul

private def quadratic : FixedPolynomial Nat 2 :=
  mul natOps (affine 2 3) (affine 5 7)

/-- Schoolbook convolution is exact and constant-first. -/
example : quadratic.coefficients = [10, 29, 21] := by decide

/-- Conversion exposes all three coefficients and derives degree two. -/
example : quadratic.toMessage.coefficients.length = 3 /\
    quadratic.toMessage.degreeUpperBound = 2 := by decide

/-- Typed evaluation is exactly verifier-visible message evaluation. -/
example (point : Nat) :
    quadratic.evaluate natOps point =
      quadratic.toMessage.evaluate natOps point :=
  evaluate_eq_message_evaluate natOps quadratic point

/-- Convolution evaluation agrees with the product of both affine factors. -/
example : quadratic.evaluate natOps 4 = 462 := by decide

/-- Repeated multiplication carries the exact static degree and uses the
same evaluation order as scalar exponentiation. -/
example :
    let cubed := power natOps (affine 2 3) 3
    cubed.coefficients.length = 4 /\
      cubed.evaluate natOps 4 = valuePower natOps 14 3 := by
  decide

/-- Widening preserves all visible low coefficients, appends only high zeros,
and does not change evaluation. -/
example :
    let widened := widen natOps (by omega : 2 <= 5) quadratic
    widened.coefficients = [10, 29, 21, 0, 0, 0] /\
      widened.evaluate natOps 4 = quadratic.evaluate natOps 4 := by
  decide

/-- A finite family is summed without changing its fixed degree. -/
example :
    (sum natOps [0, 1, 2]
      (fun index => affine index (index + 1))).evaluate natOps 2 = 15 := by
  decide

/-- The generic algebraic closure theorem is available independently of the
concrete regression carrier. -/
example
    {Field : Type}
    {leftDegree rightDegree : Nat}
    (ops : Ops Field)
    (laws : Laws ops)
    (left : FixedPolynomial Field leftDegree)
    (right : FixedPolynomial Field rightDegree)
    (point : Field) :
    (mul ops left right).evaluate ops point =
      ops.mul (left.evaluate ops point) (right.evaluate ops point) :=
  evaluate_mul ops laws left right point

end NightstreamTests.SumCheckFixedPolynomial
