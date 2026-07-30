import Nightstream.Implementation.R1CS.Canonical.KPairLaws

/-!
Contract: evaluation is a ring map on the *unreduced* polynomial layer.

Owns: additivity, commutation with scaling, multiplicativity over `polyMul`,
the reconciliation of `hornerValue` with `polyEval`, and the evaluated form of
the frozen quotient identity's right-hand side.

Does **not** own, and does not prove: anything about the *reduced* cyclotomic
product. Relating the two is the coefficient identity `raw = reduced + q · Φ₈₁`,
which lives in `KQuotient` and is not written here.

## Why this is a separate module

`KPairLaws` imports `KPolyHom`, so the ring laws cannot be used inside
`KPolyHom` itself. Anything combining the vocabulary with the laws has to sit
above both. Discovered by trying the other order and getting an unknown
namespace.

## Canonicity is load-bearing

`polyAdd`'s base cases return the longer list unchanged, so
`polyEval (polyAdd [] q) = polyEval q` — but the statement demands
`addPair ⟨0,0⟩ (polyEval q)`, and `addPair` *reduces*. The two agree only
because `polyEval` always returns residues, which is what `polyEval_canonical`
establishes. Without it the base cases are false as stated, not merely hard.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPolyEval

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KPolyHom
open Nightstream.Implementation.R1CS.Canonical.KPairLaws

/-- Evaluation always returns residues, because every step reduces. -/
theorem polyEval_canonical (point : Pair) :
    ∀ poly : List Pair,
      (polyEval point poly).low < goldilocksP
        ∧ (polyEval point poly).high < goldilocksP
  | [] => ⟨by rw [polyEval_nil]; decide, by rw [polyEval_nil]; decide⟩
  | c :: rest => addPair_canonical c (mulPair point (polyEval point rest))

/-- **Evaluation distributes over coefficientwise addition.** -/
theorem polyEval_polyAdd (point : Pair) :
    ∀ left right : List Pair,
      polyEval point (polyAdd left right)
        = addPair (polyEval point left) (polyEval point right)
  | [], right => by
      show polyEval point right
        = addPair (polyEval point []) (polyEval point right)
      rw [polyEval_nil, addPair_zero_left_canonical _
        (polyEval_canonical point right).1 (polyEval_canonical point right).2]
  | a :: left, [] => by
      show polyEval point (a :: left)
        = addPair (polyEval point (a :: left)) (polyEval point [])
      rw [polyEval_nil, addPair_comm, addPair_zero_left_canonical _
        (polyEval_canonical point (a :: left)).1
        (polyEval_canonical point (a :: left)).2]
  | a :: left, b :: right => by
      show addPair (addPair a b)
          (mulPair point (polyEval point (polyAdd left right)))
        = addPair (addPair a (mulPair point (polyEval point left)))
            (addPair b (mulPair point (polyEval point right)))
      rw [polyEval_polyAdd point left right]
      exact addPair_addPair_regroup a b _ _ point

/-- **Evaluation commutes with scaling.** -/
theorem polyEval_polyScale (point scalar : Pair) :
    ∀ poly : List Pair,
      polyEval point (polyScale scalar poly)
        = mulPair scalar (polyEval point poly)
  | [] => by
      show polyEval point [] = mulPair scalar (polyEval point [])
      rw [polyEval_nil, mulPair_zero_right]
  | c :: rest => by
      show addPair (mulPair scalar c)
          (mulPair point (polyEval point (polyScale scalar rest)))
        = mulPair scalar (addPair c (mulPair point (polyEval point rest)))
      rw [polyEval_polyScale point scalar rest]
      exact scale_regroup scalar c point (polyEval point rest)

/-! ## Multiplicativity

The induction cycle 257 predicted would be one step given the ring laws. Each
rewrite names the law it uses:

- `polyEval_polyAdd` splits `polyMul`'s head-splitting definition;
- `polyEval_polyScale` handles the scaled head;
- `addPair_zero_left_canonical` discharges the shift's leading zero, which needs
  `mulPair_canonical`;
- `mulPair_addPair_distrib_right` expands the right-hand side;
- `mulPair_assoc` is what finally makes the two sides identical. -/

/-- **Evaluation is multiplicative on convolution.**  The first of
`KRINGPROJECTION-HOMOMORPHISM`'s three obligations. -/
theorem polyEval_polyMul (point : Pair) :
    ∀ left right : List Pair,
      polyEval point (polyMul left right)
        = mulPair (polyEval point left) (polyEval point right)
  | [], right => by
      show polyEval point [] = mulPair (polyEval point []) (polyEval point right)
      rw [polyEval_nil, mulPair_zero_left]
  | a :: left, right => by
      show polyEval point
          (polyAdd (polyScale a right) (⟨0, 0⟩ :: polyMul left right))
        = mulPair (addPair a (mulPair point (polyEval point left)))
            (polyEval point right)
      rw [polyEval_polyAdd, polyEval_polyScale, polyEval_cons,
        polyEval_polyMul point left right,
        addPair_zero_left_canonical _
          (mulPair_canonical point (mulPair (polyEval point left)
            (polyEval point right))).1
          (mulPair_canonical point (mulPair (polyEval point left)
            (polyEval point right))).2,
        mulPair_addPair_distrib_right, mulPair_assoc]

/-! ## Reconciling the two evaluators

`hornerValue` skips a multiply-by-zero on a single coefficient; `polyEval` does
not. They therefore agree exactly when the coefficients are residues, which the
modulus vector is. That is what lets a row program stated over `hornerValue` be
read as a statement about `polyEval`, which is where the ring laws live. -/

theorem hornerValue_eq_polyEval (point : Pair) :
    ∀ poly : List Pair,
      (∀ c ∈ poly, c.low < goldilocksP ∧ c.high < goldilocksP) →
      hornerValue point poly = polyEval point poly
  | [], _ => rfl
  | [c], canonical => by
      have residue := canonical c (by simp)
      show c = addPair c (mulPair point ⟨0, 0⟩)
      rw [mulPair_zero_right, addPair_comm,
        addPair_zero_left_canonical c residue.1 residue.2]
  | c :: next :: rest, canonical => by
      have tail : ∀ d ∈ next :: rest,
          d.low < goldilocksP ∧ d.high < goldilocksP :=
        fun d member => canonical d (List.mem_cons_of_mem _ member)
      show addPair c (mulPair point (hornerValue point (next :: rest)))
        = addPair c (mulPair point (polyEval point (next :: rest)))
      rw [hornerValue_eq_polyEval point (next :: rest) tail]

/-- The modulus vector holds only zeros and ones, so it is canonical and the
two evaluators agree on it. -/
theorem modulusCoefficients_canonical :
    ∀ c ∈ KRingProjection.modulusCoefficients,
      c.low < goldilocksP ∧ c.high < goldilocksP := by
  intro c member
  unfold KRingProjection.modulusCoefficients at member
  rcases List.mem_map.1 member with ⟨index, _, image⟩
  by_cases branch : index = 0 ∨ index = KRingProjection.ringMiddleDegree
      ∨ index = KRingProjection.ringDegree
  · rw [if_pos branch] at image
    rw [← image]
    exact ⟨by decide, by decide⟩
  · rw [if_neg branch] at image
    rw [← image]
    exact ⟨by decide, by decide⟩

/-! ## The production right-hand side

Cycles 262–290 carried three lemmas here — `polyEval_root_of_RootOfModulus`,
`polyEval_multiple_of_root` and `polyEval_add_multiple_of_root` — all
hypothesising `polyEval point Φ₈₁ = ⟨0,0⟩` so that the quotient term could be
deleted. **That hypothesis is unsatisfiable over `K`** (see `KRingProjection`:
`v₃(p² − 1) = 1`, so no primitive 81st root of unity exists there), so those
lemmas were vacuous on the only modulus this track uses. They are withdrawn.

What the frozen check actually tests is the quotient form
`Σᵢ ρᵢ · xᵢ = q · Φ₈₁ + out`, evaluated at the challenge. Evaluating that needs
**no** condition on the point: additivity and multiplicativity split it, and
`Φ₈₁(point)` is computed like any other coefficient vector. -/

/-- **The production right-hand side, evaluated.**  No hypothesis on the point:
the quotient term is computed, not assumed away.

This is what `ProjectionProgram.ProjectionTrace.identity`'s `rhs` becomes under
evaluation, and it is the replacement for the withdrawn root-based route. -/
theorem polyEval_quotientForm
    (point : Pair) (output modulus quotient : List Pair) :
    polyEval point (polyAdd output (polyMul quotient modulus))
      = addPair (polyEval point output)
          (mulPair (polyEval point quotient) (polyEval point modulus)) := by
  rw [polyEval_polyAdd, polyEval_polyMul]

end Nightstream.Implementation.R1CS.Canonical.KPolyEval
