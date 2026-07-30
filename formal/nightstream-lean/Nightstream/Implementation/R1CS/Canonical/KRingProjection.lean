import Nightstream.Implementation.R1CS.Canonical.KFrames
import Nightstream.Implementation.R1CS.Canonical.KHorner

/-!
Contract: projecting one cyclotomic ring element.

Owns: the row program that evaluates a `RingK` element's coefficient vector at
the challenge, and its derived cost at the production degree.

Does not own: the trace function, the challenge's derivation, the quotient
identity, or the cost of a combine equation.

## Why the modulus is not divided out

`RingK` multiplication is convolution reduced modulo `Φ₈₁(X) = X⁵⁴ + X²⁷ + 1`.
Cycles 232–290 pursued a shortcut: if the challenge were a **root** of `Φ₈₁`
then evaluation would be a ring homomorphism, the reduction would be invisible,
and a ring product would cost one `KMul` instead of a convolution.

**That shortcut is impossible here, and the route is closed.** `Φ₈₁`'s roots are
primitive 81st roots of unity, so a root in `K` requires `81 ∣ |K*| = p² − 1`.
For Goldilocks `p = 2⁶⁴ − 2³² + 1` the 3-adic valuation of `p² − 1` is 1, not 4:
`v₃(p − 1) = 1` and `3 ∤ p + 1`. Equivalently `ord₈₁(p) = 27`, so `Φ₈₁` splits
into two irreducible factors of degree 27 and its roots live in `F_{p²⁷}` — never
in the quadratic extension the verifier works over. No challenge can satisfy the
premise, so every theorem carrying it was vacuous.

Production never made that assumption. `ProjectionProgram.ProjectionTrace.identity`
checks the quotient form

```text
Σᵢ ρᵢ · xᵢ  =  q · Φ₈₁ + out       (|q| = 53, maxDegree = 106)
```

as a *coefficient* identity tested at one challenge, with the quotient supplied
by the prover as 53 committed columns. That needs no condition on the challenge
at all, and it is the shape the canonical track now derives — independently, in
`KQuotient`, which reaches the same width 53 and the same degree 106 from the
degree bound rather than from the artifact.

## What survives

Horner evaluation is still the load-bearing gadget: the quotient identity is
tested by evaluating every coefficient vector at the challenge, so
`projectionRows` below is used more, not less. What died is only the claim that
a *ring product* collapses to three rows.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KRingProjection

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-- Production `d = φ(81)`, matching `SuperNeo.Concrete.ringDegree`. -/
def ringDegree : Nat := 54

/-- Middle coefficient of `Φ₈₁(X) = X⁵⁴ + X²⁷ + 1`, matching
`SuperNeo.Concrete.ringMiddleDegree`. -/
def ringMiddleDegree : Nat := 27

theorem ringDegree_eq : ringDegree = 54 := rfl

/-- **The projection of one ring element.**  Its coefficient vector, evaluated
at the challenge. -/
def projectionRows
    (beta : Carried) (base : Nat) (coefficients : List Carried) : List Row :=
  hornerRows beta (KFrames.frameAt base) coefficients 0

/-- **The derived cost of one ring projection.**  Three rows per multiplication,
one multiplication fewer than the degree. -/
theorem projectionRows_length
    (beta : Carried) (base : Nat) (coefficients : List Carried)
    (sized : coefficients.length = ringDegree) :
    (projectionRows beta base coefficients).length = 159 := by
  unfold projectionRows
  rw [hornerRows_length, sized]
  decide

/-- The projection allocates three columns per multiplication and nothing
else. -/
theorem projectionColumns
    (base : Nat) (count : Nat) :
    (KFrames.frameColumns base count).length = 3 * count :=
  KFrames.frameColumns_length base count

/-! ## The modulus vector

Carried as data because the quotient identity evaluates it at the challenge
like any other coefficient vector. It is not divided out and it is not assumed
to vanish. -/

/-- `Φ₈₁(X) = X⁵⁴ + X²⁷ + 1`, constant-first, matching `ProjectionCheck.eval`'s
coefficient order. -/
def modulusCoefficients : List Pair :=
  (List.range (ringDegree + 1)).map (fun index =>
    if index = 0 ∨ index = ringMiddleDegree ∨ index = ringDegree then
      ⟨1, 0⟩ else ⟨0, 0⟩)

theorem modulusCoefficients_length : modulusCoefficients.length = 55 := by
  unfold modulusCoefficients
  rw [List.length_map, List.length_range]
  decide

/-! ## No combine-equation cost is stated here

Cycles 253–290 carried `combineEquationCost count = 159 + 2·count·159 +
3·count + 2`, and `combineEquationCost 2 = 803`. Both are withdrawn.

Two defects, either one fatal:

- It was a **declared formula, not an emitted program**. `803` was arithmetic on
  a `def`, not the length of a row list. That is a count without a construction.
- It was a **subtotal presented as a total**. It priced a ring product at one
  `KMul` and omitted the quotient projection and the evaluation of `Φ₈₁`
  entirely — the two terms the impossible root assumption was hiding.

The replacement must be a fold over an emitted quotient-identity program, and
that program does not exist yet. Recording no number is correct; recording the
old one would be worse than recording none. -/

end Nightstream.Implementation.R1CS.Canonical.KRingProjection
