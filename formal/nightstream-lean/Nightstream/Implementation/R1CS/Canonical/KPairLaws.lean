import Nightstream.Implementation.R1CS.Canonical.KPolyHom

/-!
Contract: the ring laws for `Pair` arithmetic.

Owns: commutativity, associativity, distributivity and the zero law for
`mulPair` and `addPair`, modulo the prime.

Does not own: the projection homomorphism, which needs these plus an induction;
or agreement with the frozen `rawMulCoeffK`.

## Why these are not free

`mulPair` and `addPair` reduce modulo the prime at every step, so each law is a
congruence rather than a syntactic identity. The pattern throughout is the same:
strip the inner reductions with congruence lemmas, expand the products with
`Nat.add_mul`/`Nat.mul_add`, generalize each monomial so nothing nonlinear
remains, and close with `omega`.

`ring` would do this in one line and is unavailable without Mathlib, which is
why `KMul.karatsuba_identity` needed the same treatment.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Canonical.KPairLaws

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical.KHorner

/-! ## Congruence plumbing -/

theorem mul_congr (a b c d : Nat)
    (left : a % goldilocksP = c % goldilocksP)
    (right : b % goldilocksP = d % goldilocksP) :
    a * b % goldilocksP = c * d % goldilocksP := by
  rw [Nat.mul_mod, left, right, ← Nat.mul_mod]

theorem add_congr (a b c d : Nat)
    (left : a % goldilocksP = c % goldilocksP)
    (right : b % goldilocksP = d % goldilocksP) :
    (a + b) % goldilocksP = (c + d) % goldilocksP := by
  rw [Nat.add_mod, left, right, ← Nat.add_mod]

theorem mod_self_congr (a : Nat) : a % goldilocksP % goldilocksP = a % goldilocksP :=
  Nat.mod_mod _ _

/-! ## Commutativity -/

theorem mulPair_comm (x y : Pair) : mulPair x y = mulPair y x := by
  unfold mulPair
  simp only [Pair.mk.injEq]
  constructor
  · rw [Nat.mul_comm x.low, Nat.mul_comm x.high]
  · rw [Nat.mul_comm x.low, Nat.mul_comm x.high, Nat.add_comm]

theorem addPair_comm (x y : Pair) : addPair x y = addPair y x := by
  unfold addPair
  simp only [Pair.mk.injEq]
  exact ⟨by rw [Nat.add_comm], by rw [Nat.add_comm]⟩

/-! ## The zero law -/

theorem mulPair_zero_right (x : Pair) : mulPair x ⟨0, 0⟩ = ⟨0, 0⟩ := by
  unfold mulPair
  simp

theorem mulPair_zero_left (x : Pair) : mulPair ⟨0, 0⟩ x = ⟨0, 0⟩ := by
  rw [mulPair_comm]; exact mulPair_zero_right x

/-! ## Distributivity

The load-bearing law: `polyMul`'s head-splitting identity turns into
`polyEval`'s multiplicativity only through this. -/

theorem mulPair_addPair_distrib_right (x y z : Pair) :
    mulPair (addPair x y) z = addPair (mulPair x z) (mulPair y z) := by
  unfold mulPair addPair
  simp only [Pair.mk.injEq]
  refine ⟨?_, ?_⟩
  · have strip :
        ((x.low + y.low) % goldilocksP * z.low
            + 7 * ((x.high + y.high) % goldilocksP * z.high)) % goldilocksP
          = ((x.low + y.low) * z.low
            + 7 * ((x.high + y.high) * z.high)) % goldilocksP := by
      refine add_congr _ _ _ _ ?_ ?_
      · exact mul_congr _ _ _ _ (mod_self_congr _) rfl
      · exact mul_congr _ _ _ _ rfl (mul_congr _ _ _ _ (mod_self_congr _) rfl)
    have expand :
        (x.low + y.low) * z.low + 7 * ((x.high + y.high) * z.high)
          = (x.low * z.low + 7 * (x.high * z.high))
            + (y.low * z.low + 7 * (y.high * z.high)) := by
      rw [Nat.add_mul, Nat.add_mul, Nat.mul_add]
      generalize x.low * z.low = a
      generalize y.low * z.low = b
      generalize x.high * z.high = c
      generalize y.high * z.high = d
      omega
    rw [strip, expand, Nat.add_mod]
  · have strip :
        ((x.low + y.low) % goldilocksP * z.high
            + (x.high + y.high) % goldilocksP * z.low) % goldilocksP
          = ((x.low + y.low) * z.high
            + (x.high + y.high) * z.low) % goldilocksP := by
      refine add_congr _ _ _ _ ?_ ?_
      · exact mul_congr _ _ _ _ (mod_self_congr _) rfl
      · exact mul_congr _ _ _ _ (mod_self_congr _) rfl
    have expand :
        (x.low + y.low) * z.high + (x.high + y.high) * z.low
          = (x.low * z.high + x.high * z.low)
            + (y.low * z.high + y.high * z.low) := by
      rw [Nat.add_mul, Nat.add_mul]
      generalize x.low * z.high = a
      generalize y.low * z.high = b
      generalize x.high * z.low = c
      generalize y.high * z.low = d
      omega
    rw [strip, expand, Nat.add_mod]

/-! ## Associativity

The `generalize`-then-`omega` pattern that closes distributivity does not close
this one: it compares `x.low * y.low * z.low` against `x.low * (y.low * z.low)`,
distinct terms, so generalizing one side leaves the other ungeneralized.

The fix is to normalize both sides to a common associated form first —
`Nat.mul_assoc` and `Nat.mul_left_comm` are enough to make `simp` treat the
products as an AC normal form — and only then compare. -/

theorem mulPair_assoc (x y z : Pair) :
    mulPair (mulPair x y) z = mulPair x (mulPair y z) := by
  unfold mulPair
  simp only [Pair.mk.injEq]
  refine ⟨?_, ?_⟩
  · have stripLeft :
        ((x.low * y.low + 7 * (x.high * y.high)) % goldilocksP * z.low
            + 7 * ((x.low * y.high + x.high * y.low) % goldilocksP * z.high))
              % goldilocksP
          = ((x.low * y.low + 7 * (x.high * y.high)) * z.low
            + 7 * ((x.low * y.high + x.high * y.low) * z.high)) % goldilocksP := by
      refine add_congr _ _ _ _ ?_ ?_
      · exact mul_congr _ _ _ _ (mod_self_congr _) rfl
      · exact mul_congr _ _ _ _ rfl (mul_congr _ _ _ _ (mod_self_congr _) rfl)
    have stripRight :
        (x.low * ((y.low * z.low + 7 * (y.high * z.high)) % goldilocksP)
            + 7 * (x.high * ((y.low * z.high + y.high * z.low) % goldilocksP)))
              % goldilocksP
          = (x.low * (y.low * z.low + 7 * (y.high * z.high))
            + 7 * (x.high * (y.low * z.high + y.high * z.low))) % goldilocksP := by
      refine add_congr _ _ _ _ ?_ ?_
      · exact mul_congr _ _ _ _ rfl (mod_self_congr _)
      · exact mul_congr _ _ _ _ rfl (mul_congr _ _ _ _ rfl (mod_self_congr _))
    rw [stripLeft, stripRight]
    congr 1
    simp only [Nat.add_mul, Nat.mul_add, Nat.mul_assoc, Nat.mul_left_comm,
      Nat.mul_comm]
    omega
  · have stripLeft :
        ((x.low * y.low + 7 * (x.high * y.high)) % goldilocksP * z.high
            + (x.low * y.high + x.high * y.low) % goldilocksP * z.low)
              % goldilocksP
          = ((x.low * y.low + 7 * (x.high * y.high)) * z.high
            + (x.low * y.high + x.high * y.low) * z.low) % goldilocksP := by
      refine add_congr _ _ _ _ ?_ ?_
      · exact mul_congr _ _ _ _ (mod_self_congr _) rfl
      · exact mul_congr _ _ _ _ (mod_self_congr _) rfl
    have stripRight :
        (x.low * ((y.low * z.high + y.high * z.low) % goldilocksP)
            + x.high * ((y.low * z.low + 7 * (y.high * z.high)) % goldilocksP))
              % goldilocksP
          = (x.low * (y.low * z.high + y.high * z.low)
            + x.high * (y.low * z.low + 7 * (y.high * z.high))) % goldilocksP := by
      refine add_congr _ _ _ _ ?_ ?_
      · exact mul_congr _ _ _ _ rfl (mod_self_congr _)
      · exact mul_congr _ _ _ _ rfl (mod_self_congr _)
    rw [stripLeft, stripRight]
    congr 1
    simp only [Nat.add_mul, Nat.mul_add, Nat.mul_assoc, Nat.mul_left_comm,
      Nat.mul_comm]
    omega

/-! ## Canonicity

`addPair` and `mulPair` reduce, so their results are always canonical. The
additive identity holds only on canonical values, which is why this is needed
before the polynomial induction rather than after. -/

theorem addPair_canonical (x y : Pair) :
    (addPair x y).low < goldilocksP ∧ (addPair x y).high < goldilocksP :=
  ⟨Nat.mod_lt _ (by decide), Nat.mod_lt _ (by decide)⟩

theorem mulPair_canonical (x y : Pair) :
    (mulPair x y).low < goldilocksP ∧ (mulPair x y).high < goldilocksP :=
  ⟨Nat.mod_lt _ (by decide), Nat.mod_lt _ (by decide)⟩

/-- Zero is the additive identity on canonical values.  It is *not* the identity
in general: `addPair ⟨0,0⟩ x` reduces `x`, which changes it when `x` is not
already a residue. -/
theorem addPair_zero_left_canonical (x : Pair)
    (lowLt : x.low < goldilocksP) (highLt : x.high < goldilocksP) :
    addPair ⟨0, 0⟩ x = x := by
  unfold addPair
  simp only [Nat.zero_add, Nat.mod_eq_of_lt lowLt, Nat.mod_eq_of_lt highLt]

/-! ## Regrouping

The two shapes the polynomial induction needs, each assembled from the laws
above rather than reproved from the modular arithmetic. -/

theorem addPair_assoc (x y z : Pair) :
    addPair (addPair x y) z = addPair x (addPair y z) := by
  unfold addPair
  simp only [Pair.mk.injEq]
  refine ⟨?_, ?_⟩ <;>
    (refine (add_congr _ _ _ _ (mod_self_congr _) rfl).trans ?_
     refine Eq.trans ?_ (add_congr _ _ _ _ rfl (mod_self_congr _)).symm
     rw [Nat.add_assoc])

theorem mulPair_addPair_distrib_left (x y z : Pair) :
    mulPair x (addPair y z) = addPair (mulPair x y) (mulPair x z) := by
  rw [mulPair_comm, mulPair_addPair_distrib_right, mulPair_comm y x,
    mulPair_comm z x]

theorem mulPair_left_comm (x y z : Pair) :
    mulPair x (mulPair y z) = mulPair y (mulPair x z) := by
  rw [← mulPair_assoc, mulPair_comm x y, mulPair_assoc]

/-- Middle-four exchange for `addPair`. -/
theorem addPair_exchange (a b c d : Pair) :
    addPair (addPair a b) (addPair c d)
      = addPair (addPair a c) (addPair b d) := by
  rw [addPair_assoc, addPair_assoc, ← addPair_assoc b c d,
    ← addPair_assoc c b d, addPair_comm b c]

/-- The shape `polyEval_polyAdd`'s cons case needs. -/
theorem addPair_addPair_regroup (a b u v point : Pair) :
    addPair (addPair a b) (mulPair point (addPair u v))
      = addPair (addPair a (mulPair point u)) (addPair b (mulPair point v)) := by
  rw [mulPair_addPair_distrib_left, addPair_exchange]

/-- The shape `polyEval_polyScale`'s cons case needs. -/
theorem scale_regroup (scalar c point r : Pair) :
    addPair (mulPair scalar c) (mulPair point (mulPair scalar r))
      = mulPair scalar (addPair c (mulPair point r)) := by
  rw [mulPair_addPair_distrib_left, mulPair_left_comm point scalar r]

/-! ## Powers

The per-degree reduction argument compares `β^d` against `β^{d−27}` and
`β^{d−54}`, so it needs powers and their additivity. Defined here rather than
with the polynomial layer because they are pure `Pair` arithmetic. -/

def powPair (base : Pair) : Nat → Pair
  | 0 => ⟨1, 0⟩
  | exponent + 1 => mulPair base (powPair base exponent)

theorem powPair_zero (base : Pair) : powPair base 0 = ⟨1, 0⟩ := rfl

theorem powPair_succ (base : Pair) (exponent : Nat) :
    powPair base (exponent + 1) = mulPair base (powPair base exponent) := rfl

theorem powPair_canonical (base : Pair) :
    ∀ exponent, (powPair base exponent).low < goldilocksP
      ∧ (powPair base exponent).high < goldilocksP
  | 0 => by
      rw [powPair_zero]
      exact ⟨by decide, by decide⟩
  | exponent + 1 => by
      rw [powPair_succ]
      exact mulPair_canonical _ _

theorem mulPair_one_left (value : Pair)
    (lowLt : value.low < goldilocksP) (highLt : value.high < goldilocksP) :
    mulPair ⟨1, 0⟩ value = value := by
  unfold mulPair
  simp only [Nat.one_mul, Nat.zero_mul, Nat.mul_zero, Nat.add_zero,
    Nat.zero_add, Nat.mul_one]
  simp only [Nat.mod_eq_of_lt lowLt, Nat.mod_eq_of_lt highLt]

/-- **Powers add.**  What lets `β^d` be split as `β^{d−54} · β^54`, which is the
whole content of the per-degree reduction argument. -/
theorem powPair_add (base : Pair) (left : Nat) :
    ∀ right, powPair base (left + right)
      = mulPair (powPair base left) (powPair base right)
  | 0 => by
      rw [Nat.add_zero, powPair_zero, mulPair_comm,
        mulPair_one_left _ (powPair_canonical base left).1
          (powPair_canonical base left).2]
  | right + 1 => by
      rw [show left + (right + 1) = (left + right) + 1 from by omega,
        powPair_succ, powPair_add base left right, powPair_succ,
        ← mulPair_assoc, ← mulPair_assoc, mulPair_comm base]

/-! ## The first reduction identity

For `54 ≤ d ≤ 80` a single fold suffices, and the identity it rests on is
`β^d + β^{d−27} + β^{d−54} = β^{d−54} · (β⁵⁴ + β²⁷ + 1)`. At a root the right
side vanishes, so the three powers cancel.

The root hypothesis is stated in `powPair` terms — `β⁵⁴ + β²⁷ + 1 = 0` — which
is what `polyEval` of the modulus vector unfolds to. -/

/-- Factoring out a shared multiplicand when one summand is the factor itself.
Conditional on canonicity, because it uses the identity law. -/
theorem mulPair_add_self (factor other : Pair)
    (lowLt : factor.low < goldilocksP) (highLt : factor.high < goldilocksP) :
    mulPair factor (addPair other ⟨1, 0⟩)
      = addPair (mulPair factor other) factor := by
  rw [mulPair_addPair_distrib_left, mulPair_comm factor ⟨1, 0⟩,
    mulPair_one_left factor lowLt highLt]

/-- **Degrees 54 through 80 fold once.**  Their three powers sum to zero at a
root of the modulus. -/
theorem reduction_single_fold (base : Pair) (degree : Nat)
    (atLeast : 54 ≤ degree)
    (root : addPair (addPair (powPair base 54) (powPair base 27)) ⟨1, 0⟩
      = ⟨0, 0⟩) :
    addPair (addPair (powPair base degree) (powPair base (degree - 27)))
        (powPair base (degree - 54))
      = ⟨0, 0⟩ := by
  have splitFull : powPair base degree
      = mulPair (powPair base (degree - 54)) (powPair base 54) := by
    rw [← powPair_add]
    congr 1
    omega
  have splitMid : powPair base (degree - 27)
      = mulPair (powPair base (degree - 54)) (powPair base 27) := by
    rw [← powPair_add]
    congr 1
    omega
  rw [splitFull, splitMid, ← mulPair_addPair_distrib_left,
    ← mulPair_add_self _ _ (powPair_canonical base (degree - 54)).1
      (powPair_canonical base (degree - 54)).2,
    root, mulPair_zero_right]

/-! ## The second reduction identity

Degrees 81 and above collapse in one step because `β⁸¹ = 1`. The obvious route
is cancellation — subtract the root from `β²⁷` times the root — and this tower
has no additive cancellation lemma.

It is not needed. Adding the root, which is zero, to `β⁸¹` and regrouping
reaches the same conclusion using only associativity:

```text
β⁸¹ = β⁸¹ + (β⁵⁴ + β²⁷ + 1)      -- adding zero
    = (β⁸¹ + β⁵⁴ + β²⁷) + 1      -- regrouping
    = 0 + 1                       -- β²⁷ times the root
    = 1
```
-/

/-- Multiplying the root by `β²⁷` gives the shifted relation. -/
theorem root_shifted (base : Pair)
    (root : addPair (addPair (powPair base 54) (powPair base 27)) ⟨1, 0⟩
      = ⟨0, 0⟩) :
    addPair (addPair (powPair base 81) (powPair base 54)) (powPair base 27)
      = ⟨0, 0⟩ := by
  have expand :
      mulPair (powPair base 27)
          (addPair (addPair (powPair base 54) (powPair base 27)) ⟨1, 0⟩)
        = addPair (addPair (powPair base 81) (powPair base 54))
            (powPair base 27) := by
    rw [mulPair_addPair_distrib_left, mulPair_addPair_distrib_left,
      ← powPair_add, ← powPair_add, mulPair_comm (powPair base 27) ⟨1, 0⟩,
      mulPair_one_left _ (powPair_canonical base 27).1
        (powPair_canonical base 27).2]
  rw [← expand, root, mulPair_zero_right]

/-- **Degrees 81 and above collapse.**  `β⁸¹ = 1` at a root, so `β^d = β^{d−81}`
and the second fold cancels the first.  Proved without additive cancellation. -/
theorem powPair_eightyOne (base : Pair)
    (root : addPair (addPair (powPair base 54) (powPair base 27)) ⟨1, 0⟩
      = ⟨0, 0⟩) :
    powPair base 81 = ⟨1, 0⟩ := by
  have addZero : powPair base 81
      = addPair (powPair base 81)
          (addPair (addPair (powPair base 54) (powPair base 27)) ⟨1, 0⟩) := by
    rw [root, addPair_comm, addPair_zero_left_canonical _
      (powPair_canonical base 81).1 (powPair_canonical base 81).2]
  rw [addZero, ← addPair_assoc, ← addPair_assoc, root_shifted base root,
    addPair_zero_left_canonical _ (by decide) (by decide)]

/-! ## Subtraction

The quotient's coefficients are differences, and every operation in this tower
so far has been `addPair`, `mulPair` or a `goldilocksP - 1` multiplier. This
supplies the missing one.

`Nat` has no negatives, so the complement is taken explicitly: `p - y % p` is
safe because `y % p < p`. The defining law is that adding `y` back recovers `x`,
which holds only for canonical `x` — the sixth place in this tower where that
condition is load-bearing. -/

def subPair (x y : Pair) : Pair where
  low := (x.low + (goldilocksP - y.low % goldilocksP)) % goldilocksP
  high := (x.high + (goldilocksP - y.high % goldilocksP)) % goldilocksP

theorem subPair_canonical (x y : Pair) :
    (subPair x y).low < goldilocksP ∧ (subPair x y).high < goldilocksP :=
  ⟨Nat.mod_lt _ (by decide), Nat.mod_lt _ (by decide)⟩

/-- The complement really is a multiple of the prime once the subtrahend is
added back.  This is the whole content of subtraction in `Nat`. -/
theorem complement_add (value : Nat) :
    (goldilocksP - value % goldilocksP) + value
      = goldilocksP * (1 + value / goldilocksP) := by
  have split := Nat.div_add_mod value goldilocksP
  have bound : value % goldilocksP < goldilocksP := Nat.mod_lt _ (by decide)
  rw [Nat.mul_add, Nat.mul_one]
  omega

/-- Subtracting zero is the identity on residues.  Not `rfl`: the complement is
`p - 0 = p`, so the sum still has to be reduced. -/
theorem subPair_zero_right (x : Pair)
    (lowLt : x.low < goldilocksP) (highLt : x.high < goldilocksP) :
    subPair x ⟨0, 0⟩ = x := by
  unfold subPair
  simp only [Nat.zero_mod, Nat.sub_zero, Nat.add_mod_right]
  simp only [Nat.mod_eq_of_lt lowLt, Nat.mod_eq_of_lt highLt]

/-- **Adding back recovers the original**, for canonical values. -/
theorem addPair_subPair (x y : Pair)
    (lowLt : x.low < goldilocksP) (highLt : x.high < goldilocksP) :
    addPair (subPair x y) y = x := by
  have component : ∀ a b : Nat, a < goldilocksP →
      ((a + (goldilocksP - b % goldilocksP)) % goldilocksP + b) % goldilocksP
        = a := by
    intro a b lt
    rw [Nat.mod_add_mod, Nat.add_assoc, complement_add,
      Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt lt]
  unfold addPair subPair
  simp only [component _ _ lowLt, component _ _ highLt]

end Nightstream.Implementation.R1CS.Canonical.KPairLaws
