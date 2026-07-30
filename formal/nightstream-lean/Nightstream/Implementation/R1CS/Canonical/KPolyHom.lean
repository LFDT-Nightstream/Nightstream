import Nightstream.Implementation.R1CS.Canonical.KRingProjection

/-!
Contract: the polynomial-algebra vocabulary the projection homomorphism needs.

Owns: polynomial addition, scaling and convolution in the list representation,
uniform Horner evaluation, and the exact point where that differs from the row
program's `hornerValue`.

Does **not** own, and does not prove:

- that evaluating a convolution gives the product of the evaluations. The
  induction is one step given three ring laws for `mulPair` and `addPair` —
  distributivity over addition, associativity, and the zero law — and none of
  those is written. They are not free: `mulPair` and `addPair` reduce modulo the
  prime, so each law needs the same modular plumbing `KMul.karatsuba_identity`
  needed.
- that evaluation is a homomorphism on the *reduced* cyclotomic product, which
  additionally needs that subtracting a multiple of `Φ₈₁` leaves the evaluation
  unchanged at a root.
- that `polyMul` agrees with the frozen `rawMulCoeffK`.

`KRINGPROJECTION-HOMOMORPHISM` names the whole obligation. This module supplies
its vocabulary and nothing more.

## Why the list representation

The frozen ring is `Fin ringDegree → K` with `rawMulCoeffK` folding over an
index range. Multiplicativity is painful in that shape and immediate in the
recursive list shape, where `(a₀ :: a') · b = a₀·b + X·(a'·b)` makes the
induction one step.

Relating `polyMul` to `rawMulCoeffK` is a separate obligation, and until it is
done anything proved here is about *this* multiplication rather than the frozen
one. That gap is stated rather than glossed: the two are the same mathematics,
but sameness of mathematics is not sameness of definition, and this project has
been caught by that distinction before.

## `polyEval` versus `hornerValue`

`KHorner.hornerValue` special-cases a single coefficient to avoid emitting a
multiply-by-zero row. That optimization is right for a row program and wrong
for algebra, where the uniform recursion is what makes induction work. So the
algebra here uses `polyEval`, and `polyEval_singleton` records exactly where
the two differ.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPolyHom

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-- Uniform Horner evaluation, with no special case for a single
coefficient. -/
def polyEval (point : Pair) : List Pair → Pair
  | [] => ⟨0, 0⟩
  | c :: rest => addPair c (mulPair point (polyEval point rest))

/-- Coefficientwise addition, keeping the longer polynomial's tail. -/
def polyAdd : List Pair → List Pair → List Pair
  | [], right => right
  | left, [] => left
  | a :: left, b :: right => addPair a b :: polyAdd left right

/-- Multiply every coefficient by a scalar. -/
def polyScale (scalar : Pair) (poly : List Pair) : List Pair :=
  poly.map (mulPair scalar)

/-- Polynomial convolution, `(a₀ :: a') · b = a₀·b + X·(a'·b)`. -/
def polyMul : List Pair → List Pair → List Pair
  | [], _ => []
  | a :: left, right =>
      polyAdd (polyScale a right) (⟨0, 0⟩ :: polyMul left right)

/-! ## Modular arithmetic on pairs

`addPair` and `mulPair` reduce modulo the prime, so the ring laws hold only up
to that reduction. These are the steps the induction needs. -/

theorem addPair_zero_left (value : Pair) :
    addPair ⟨0, 0⟩ value = ⟨value.low % goldilocksP, value.high % goldilocksP⟩ := by
  unfold addPair
  simp

theorem polyEval_nil (point : Pair) : polyEval point [] = ⟨0, 0⟩ := rfl

theorem polyEval_cons (point : Pair) (c : Pair) (rest : List Pair) :
    polyEval point (c :: rest)
      = addPair c (mulPair point (polyEval point rest)) := rfl

/-- Where `polyEval` and `hornerValue` differ: on a single coefficient the row
program skips a multiplication by zero, so the algebra evaluates one extra
step. The values agree modulo canonicity of the coefficient. -/
theorem polyEval_singleton (point c : Pair) :
    polyEval point [c] = addPair c (mulPair point ⟨0, 0⟩) := rfl

theorem hornerValue_singleton (point c : Pair) : hornerValue point [c] = c := rfl

end Nightstream.Implementation.R1CS.Canonical.KPolyHom
