import NightstreamFPrime.Spec.Compression.Lib.Cube
import Mathlib.Algebra.Field.Basic
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Data.Nat.Bitwise
import Mathlib.Tactic.FieldSimp

/-! The two points of the key polynomial (Rust `key.rs`, `weights.rs`).

The row point (`rowPoint`, `rowScale`, Rust `key::row_point`): `s_k = ω^(2^k) / (1 + ω^(2^k))` and
`scale = Π_k (1 + ω^(2^k))` give `ω^i = scale · eq(s, i)` for each row `i < 2^r`
(`rowPoint_spec`). The identity is exact, so the row point adds no error.

The prefix weight (`prefixEquality`, Rust `weights::prefix_equality`): the multilinear extension
of `j ↦ eq(point, j) · [j < count]` at `at`, in a number of steps linear in the variables. It is
`Σ_{j < count} eq(point, j) · eq(at, j)` for each count (`prefixEquality_spec`). An index is below
`count` exactly when, at the highest bit where they differ, the index has 0 and `count` has 1. -/

namespace NightstreamFPrime.Spec.Compression.Algebra

open Finset NightstreamFPrime.Spec.Compression.Lib

variable {K : Type} [Field K]

section RowPoint

/-- Rust `key::row_point`: the coordinates `s_k = ω^(2^k) / (1 + ω^(2^k))`, lowest row bit
first. -/
def rowPoint : ℕ → K → List K
  | 0, _ => []
  | r + 1, ω => ω / (1 + ω) :: rowPoint r (ω ^ 2)

/-- Rust `key::row_point`: the scale `Π_k (1 + ω^(2^k))`. -/
def rowScale : ℕ → K → K
  | 0, _ => 1
  | r + 1, ω => (1 + ω) * rowScale r (ω ^ 2)

theorem rowPoint_length : ∀ (r : ℕ) (ω : K), (rowPoint r ω).length = r
  | 0, _ => rfl
  | r + 1, ω => by simp [rowPoint, rowPoint_length r]

/-- The powers of `ω` below `2^r` are a scaled `eq` table when no `1 + ω^(2^k)` is zero. -/
theorem rowPoint_spec : ∀ (r : ℕ) (ω : K), (∀ k < r, 1 + ω ^ 2 ^ k ≠ 0) →
    ∀ i < 2 ^ r, ω ^ i = rowScale r ω * eqAt (rowPoint r ω) i := by
  intro r
  induction r with
  | zero =>
      intro ω _ i lt
      obtain rfl : i = 0 := by simpa using lt
      simp [rowScale, rowPoint, eqAt]
  | succ r ih =>
      intro ω regular i lt
      have first : 1 + ω ≠ 0 := by simpa using regular 0 (by omega)
      have rest : ∀ k < r, 1 + (ω ^ 2) ^ 2 ^ k ≠ 0 := fun k small => by
        rw [← pow_mul, ← pow_succ']
        exact regular (k + 1) (by omega)
      have half := ih (ω ^ 2) rest (i / 2) (by rw [pow_succ] at lt; omega)
      have split : ω ^ i = ω ^ (i % 2) * (ω ^ 2) ^ (i / 2) := by
        rw [← pow_mul, ← pow_add]
        congr 1
        omega
      simp only [rowScale, rowPoint, eqAt]
      rw [split, half]
      rcases Nat.mod_two_eq_zero_or_one i with even | odd
      · rw [even, ite_eq_right (by omega), pow_zero]
        field_simp
        ring
      · rw [odd, ite_eq_left rfl, pow_one]
        field_simp

end RowPoint

section Prefix

/-- The factors of the bits of `eq(point, ·) · eq(at, ·)`: `((1 − p_b)(1 − a_b), p_b · a_b)`. -/
def eqTerms (point low : List K) : List (K × K) :=
  List.zipWith (fun p a => ((1 - p) * (1 - a), p * a)) point low

/-- The product of the factors that the bits of `j` select. -/
def eqProd : List (K × K) → ℕ → K
  | [], _ => 1
  | t :: rest, j => (if j % 2 = 1 then t.2 else t.1) * eqProd rest (j / 2)

/-- The loop of Rust `prefix_equality` over the bits from the highest: bit `b` acts after each
higher bit, on `(sum, equal)`. -/
def prefixLoop (count : ℕ) (below : ℕ → K) : List (K × K) → ℕ → K × K
  | [], _ => (0, 1)
  | t :: rest, b =>
      let acc := prefixLoop count below rest (b + 1)
      if count.testBit b then (acc.1 + acc.2 * t.1 * below b, acc.2 * t.2)
      else (acc.1, acc.2 * t.1)

/-- Rust `weights::prefix_equality(point, count, at)`. -/
def prefixEquality (point : List K) (count : ℕ) (at' : List K) : K :=
  let terms := eqTerms point (at'.take point.length)
  let high := ((at'.drop point.length).map fun c => 1 - c).prod
  let below := fun b => ((terms.take b).map fun t => t.1 + t.2).prod
  if count >>> point.length ≠ 0 then below point.length * high
  else (prefixLoop count below terms 0).1 * high

theorem sum_range_double {M : Type} [AddCommMonoid M] (f : ℕ → M) :
    ∀ m : ℕ, ∑ j ∈ range (2 * m), f j = ∑ j ∈ range m, (f (2 * j) + f (2 * j + 1)) := by
  intro m
  induction m with
  | zero => simp
  | succ m ih =>
      rw [show 2 * (m + 1) = 2 * m + 1 + 1 by ring, sum_range_succ, sum_range_succ, ih,
        sum_range_succ, add_assoc]

/-- The sum below `c` splits at the lowest bit. -/
theorem sum_eqProd_cons (t : K × K) (rest : List (K × K)) (c : ℕ) :
    ∑ j ∈ range c, eqProd (t :: rest) j =
      (t.1 + t.2) * ∑ j ∈ range (c / 2), eqProd rest j +
        if c % 2 = 1 then t.1 * eqProd rest (c / 2) else 0 := by
  have pairs : ∑ j ∈ range (2 * (c / 2)), eqProd (t :: rest) j =
      (t.1 + t.2) * ∑ j ∈ range (c / 2), eqProd rest j := by
    rw [sum_range_double, mul_sum]
    refine sum_congr rfl fun j _ => ?_
    simp only [eqProd, show 2 * j % 2 = 0 by omega, show (2 * j + 1) % 2 = 1 by omega,
      show 2 * j / 2 = j by omega, show (2 * j + 1) / 2 = j by omega]
    simp
    ring
  rcases Nat.mod_two_eq_zero_or_one c with even | odd
  · rw [show c = 2 * (c / 2) by omega, pairs, ite_eq_right (by omega), add_zero,
      show 2 * (c / 2) / 2 = c / 2 by omega]
  · rw [show c = 2 * (c / 2) + 1 by omega, sum_range_succ, pairs, ite_eq_left (by omega),
      show (2 * (c / 2) + 1) / 2 = c / 2 by omega]
    simp [eqProd]

theorem sum_eqProd_full : ∀ terms : List (K × K),
    ∑ j ∈ range (2 ^ terms.length), eqProd terms j = (terms.map fun t => t.1 + t.2).prod
  | [] => by simp [eqProd]
  | t :: rest => by
      rw [sum_eqProd_cons, List.length_cons, pow_succ, show 2 ^ rest.length * 2 / 2 =
        2 ^ rest.length by omega, sum_eqProd_full rest, ite_eq_right (by omega), add_zero]
      simp

/-- The loop keeps `sum = below b · Σ_{j < count >> b} eqProd` and `equal = eqProd` at
`count >> b`. -/
theorem prefixLoop_spec (count : ℕ) (below : ℕ → K) :
    ∀ (terms : List (K × K)) (b : ℕ),
      (∀ k < terms.length, below (b + k + 1) = below (b + k) *
        ((terms.getD k (0, 0)).1 + (terms.getD k (0, 0)).2)) →
      count >>> b < 2 ^ terms.length →
      prefixLoop count below terms b =
        (below b * ∑ j ∈ range (count >>> b), eqProd terms j, eqProd terms (count >>> b)) := by
  intro terms
  induction terms with
  | nil =>
      intro b _ small
      obtain zero : count >>> b = 0 := by simpa using small
      simp [prefixLoop, zero, eqProd]
  | cons t rest ih =>
      intro b step small
      have shift : count >>> (b + 1) = count >>> b / 2 := by
        rw [Nat.shiftRight_succ]
      have rest_step : ∀ k < rest.length, below (b + 1 + k + 1) = below (b + 1 + k) *
          ((rest.getD k (0, 0)).1 + (rest.getD k (0, 0)).2) := fun k lt => by
        have := step (k + 1) (by simp; omega)
        rw [show b + (k + 1) + 1 = b + 1 + k + 1 by omega, show b + (k + 1) = b + 1 + k by omega]
          at this
        simpa using this
      have rest_small : count >>> (b + 1) < 2 ^ rest.length := by
        rw [shift]
        simp only [List.length_cons, pow_succ] at small
        omega
      have first : below (b + 1) = below b * (t.1 + t.2) := by simpa using step 0 (by simp)
      have bit : count.testBit b = decide (count >>> b % 2 = 1) := by
        rw [Nat.testBit_eq_decide_div_mod_eq, Nat.shiftRight_eq_div_pow]
      simp only [prefixLoop]
      rw [ih (b + 1) rest_step rest_small, shift, sum_eqProd_cons, bit, first]
      rcases Nat.mod_two_eq_zero_or_one (count >>> b) with even | odd
      · refine Prod.ext ?_ ?_ <;> simp [even, eqProd] <;> ring
      · refine Prod.ext ?_ ?_ <;> simp [odd, eqProd] <;> ring

theorem eqAt_mul_eqAt (point low : List K) (length : point.length = low.length) (j : ℕ) :
    eqAt point j * eqAt low j = eqProd (eqTerms point low) j := by
  induction point generalizing low j with
  | nil =>
      obtain rfl : low = [] := List.length_eq_zero_iff.mp (by simpa using length.symm)
      simp [eqAt, eqTerms, eqProd]
  | cons p rest ih =>
      obtain ⟨a, low, rfl⟩ := List.exists_cons_of_length_eq_add_one (by simpa using length.symm)
      simp only [eqAt, eqTerms, List.zipWith_cons_cons, eqProd]
      rw [← eqTerms, ← ih low (by simpa using length) (j / 2)]
      split <;> ring

theorem eqAt_zero : ∀ l : List K, eqAt l 0 = (l.map fun c => 1 - c).prod
  | [] => rfl
  | c :: rest => by simp [eqAt, eqAt_zero rest]

theorem eqAt_append_low (l₁ l₂ : List K) (j : ℕ) (small : j < 2 ^ l₁.length) :
    eqAt (l₁ ++ l₂) j = eqAt l₁ j * (l₂.map fun c => 1 - c).prod := by
  induction l₁ generalizing j with
  | nil =>
      obtain rfl : j = 0 := by simpa using small
      simp [eqAt_zero, eqAt]
  | cons a rest ih =>
      simp only [List.cons_append, eqAt]
      rw [ih (j / 2) (by simp only [List.length_cons, pow_succ] at small; omega)]
      ring

/-- Rust `prefix_equality` is `Σ_{j < count} eq(point, j) · eq(at, j)`, with every `j` below
`2^|point|`. -/
theorem prefixEquality_spec (point at' : List K) (count : ℕ) (long : point.length ≤ at'.length) :
    prefixEquality point count at' =
      ∑ j ∈ range (min count (2 ^ point.length)), eqAt point j * eqAt at' j := by
  have split : at' = at'.take point.length ++ at'.drop point.length := (List.take_append_drop _ _).symm
  have low_length : (at'.take point.length).length = point.length := by simp [long]
  have terms_length : (eqTerms point (at'.take point.length)).length = point.length := by
    simp [eqTerms, low_length]
  have each (j : ℕ) (small : j < 2 ^ point.length) : eqAt point j * eqAt at' j =
      eqProd (eqTerms point (at'.take point.length)) j *
        ((at'.drop point.length).map fun c => 1 - c).prod := by
    rw [split, eqAt_append_low _ _ j (by rwa [low_length]), ← mul_assoc,
      eqAt_mul_eqAt point _ low_length.symm, List.take_append_drop]
  have below_step : ∀ k < (eqTerms point (at'.take point.length)).length,
      (((eqTerms point (at'.take point.length)).take (0 + k + 1)).map fun t => t.1 + t.2).prod =
        (((eqTerms point (at'.take point.length)).take (0 + k)).map fun t => t.1 + t.2).prod *
        (((eqTerms point (at'.take point.length)).getD k (0, 0)).1 +
          ((eqTerms point (at'.take point.length)).getD k (0, 0)).2) := by
    intro k lt
    rw [zero_add, List.take_add_one, List.map_append, List.prod_append]
    simp [lt]
  unfold prefixEquality
  simp only
  split
  · rename_i large
    rw [Nat.shiftRight_eq_div_pow] at large
    rw [min_eq_right (by
      by_contra small
      exact large (Nat.div_eq_of_lt (by omega)))]
    rw [sum_congr rfl fun j member => each j (mem_range.mp member), ← sum_mul]
    congr 1
    have full := sum_eqProd_full (eqTerms point (at'.take point.length))
    rw [terms_length] at full
    rw [full, List.take_of_length_le (le_of_eq terms_length)]
  · rename_i small
    push Not at small
    rw [Nat.shiftRight_eq_div_pow, Nat.div_eq_zero_iff] at small
    have lt : count < 2 ^ point.length := by
      rcases small with zero | lt
      · exact absurd zero (pow_ne_zero _ two_ne_zero)
      · exact lt
    rw [min_eq_left lt.le]
    rw [prefixLoop_spec count _ _ 0 below_step (by
      rw [Nat.shiftRight_zero, terms_length]; exact lt)]
    simp only [List.take_zero, List.map_nil, List.prod_nil, one_mul, Nat.shiftRight_zero]
    rw [sum_mul]
    exact sum_congr rfl fun j member => (each j (by have := mem_range.mp member; omega)).symm

end Prefix

end NightstreamFPrime.Spec.Compression.Algebra
