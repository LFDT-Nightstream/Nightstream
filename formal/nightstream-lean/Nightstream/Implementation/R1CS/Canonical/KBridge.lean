import Nightstream.Implementation.R1CS.Canonical.KHorner

/-!
Contract: the canonical track's `Pair` arithmetic is `ProjectionProgram.K`
arithmetic.

Owns: the coordinate map from `K` to `Pair` and the proofs that it carries
extension addition, multiplication and Horner evaluation.

Does not own: any row program. This is a representation bridge only.

## Why two representations exist at all

The canonical track evaluates through `lcEval : (Nat → Nat) → LinComb → Nat`,
because columns hold naturals and every other canonical module is stated that
way. `ProjectionCheck` and `ProjectionProgram.K` work over `Fin goldilocksP`
pairs.

Working the row layer directly in `K` would mean coercing at every column read.
Working the projection layer in `Pair` would mean restating `ProjectionCheck`.
Bridging once, here, is cheaper than either, and it keeps the arithmetic
claims falsifiable: `KMul`'s formulas were checked against `K.mul` numerically
in cycle 238, and this module upgrades that spot check to a theorem.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KBridge

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.ProjectionProgram
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-- The coordinates of an extension element, as naturals. -/
def toPair (value : K) : Pair where
  low := value.c0.val
  high := value.c1.val

theorem toPair_zero : toPair K.zero = ⟨0, 0⟩ := rfl

/-! ## The operations agree

`Fin` arithmetic reduces modulo the prime at every step; `Pair` arithmetic
reduces once at the end. The two coincide, which is what these say. -/

theorem toPair_add (x y : K) : toPair (K.add x y) = addPair (toPair x) (toPair y) := by
  unfold toPair K.add addPair
  simp only [Fin.val_add]

theorem toPair_mul (x y : K) : toPair (K.mul x y) = mulPair (toPair x) (toPair y) := by
  have seven : (7 : Fin goldilocksP).val = 7 := rfl
  unfold toPair K.mul mulPair
  simp only [Fin.val_add, Fin.val_mul, seven, Pair.mk.injEq]
  refine ⟨?_, ?_⟩
  · rw [KMul.mul_mod_right_reduce, ← Nat.add_mod]
  · rw [← Nat.add_mod]

/-! ## Horner evaluation agrees

`ProjectionCheck.eval` is `foldr (fun c suffix => c + point * suffix) 0`.
`hornerValue` mirrors it with the final coefficient as a base case rather than
`c + point * 0`; these two lemmas are why that shortcut loses nothing. -/

theorem K_mul_zero (point : K) : K.mul point K.zero = K.zero := by
  unfold K.mul K.zero
  simp only [K.mk.injEq]
  refine ⟨?_, ?_⟩ <;>
    (apply Fin.eq_of_val_eq; simp [Fin.val_mul, Fin.val_add])

theorem K_add_zero (value : K) : K.add value K.zero = value := by
  unfold K.add K.zero
  simp

/-- **The translation loses nothing.**  Two extension elements with the same
`Pair` are the same element, so an equation proved in the row layer's vocabulary
transports back to `K` rather than only forward. -/
theorem toPair_injective {x y : K} (equal : toPair x = toPair y) : x = y := by
  unfold toPair at equal
  simp only [Pair.mk.injEq] at equal
  rcases x with ⟨x0, x1⟩
  rcases y with ⟨y0, y1⟩
  simp only [K.mk.injEq]
  exact ⟨Fin.ext equal.1, Fin.ext equal.2⟩

/-- **The bridge.**  The canonical reference computes exactly what
`ProjectionCheck.eval` computes, coordinatewise. -/
theorem toPair_eval (point : K) :
    ∀ coefficients : List K,
      toPair (SuperNeo.ProjectionCheck.eval K.ops coefficients point)
        = hornerValue (toPair point) (coefficients.map toPair)
  | [] => rfl
  | [c] => by
      show toPair (K.add c (K.mul point K.zero)) = _
      rw [K_mul_zero, K_add_zero]
      rfl
  | c :: next :: rest => by
      have tail := toPair_eval point (next :: rest)
      show toPair (K.add c (K.mul point
        (SuperNeo.ProjectionCheck.eval K.ops (next :: rest) point))) = _
      rw [toPair_add, toPair_mul, tail]
      rfl

end Nightstream.Implementation.R1CS.Canonical.KBridge
