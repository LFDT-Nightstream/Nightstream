import CompressionLean.Lib.Cube
import Mathlib.Algebra.BigOperators.Field
import Mathlib.Algebra.BigOperators.Intervals

namespace CompressionLean.Lib

variable {K : Type} [Field K]

structure Level (K : Type) where
  factors : List (ℕ → K)
  denominator : ℕ → K

namespace Level

def numerator (L : Level K) (x : ℕ) : K := (L.factors.map fun f => f x).prod

def up (L : Level K) (half : ℕ) : Level K where
  factors := [fun y => L.numerator y * L.denominator (y + half) + L.numerator (y + half) * L.denominator y]
  denominator := fun y => L.denominator y * L.denominator (y + half)

def above (L : Level K) (depth : ℕ) : ℕ → Level K
  | 0 => L
  | m + 1 => (L.above depth m).up (2 ^ (depth - m - 1))

theorem up_numerator (L : Level K) (half y : ℕ) :
    (L.up half).numerator y = L.numerator y * L.denominator (y + half) +
      L.numerator (y + half) * L.denominator y := by
  simp [up, numerator]

theorem up_denominator (L : Level K) (half y : ℕ) :
    (L.up half).denominator y = L.denominator y * L.denominator (y + half) := rfl

theorem above_fraction (L : Level K) (depth : ℕ) :
    ∀ m, m ≤ depth → (∀ y, y < 2 ^ (depth - m) → (L.above depth m).denominator y ≠ 0) →
      (∀ x, x < 2 ^ depth → L.denominator x ≠ 0) ∧
        ∑ y ∈ Finset.range (2 ^ (depth - m)),
            (L.above depth m).numerator y / (L.above depth m).denominator y =
          ∑ x ∈ Finset.range (2 ^ depth), L.numerator x / L.denominator x := by
  intro m
  induction m with
  | zero => intro _ nonzero; exact ⟨nonzero, rfl⟩
  | succ m ih =>
      intro fits nonzero
      set half := 2 ^ (depth - m - 1) with half_def
      have size : 2 ^ (depth - m) = half + half := by
        rw [half_def, ← two_mul, ← pow_succ']
        congr 1
        omega
      have size' : 2 ^ (depth - (m + 1)) = half := by rw [half_def]; congr 1
      have lower_nonzero : ∀ y, y < 2 ^ (depth - m) → (L.above depth m).denominator y ≠ 0 := by
        intro y lt
        rw [size] at lt
        by_cases low : y < half
        · have := nonzero y (by rw [size']; exact low)
          simp only [above, up] at this
          exact left_ne_zero_of_mul this
        · have := nonzero (y - half) (by rw [size']; omega)
          simp only [above, up] at this
          rw [show y - half + half = y by omega] at this
          exact right_ne_zero_of_mul this
      obtain ⟨leaves, sum⟩ := ih (by omega) lower_nonzero
      refine ⟨leaves, ?_⟩
      rw [← sum, size', size, Finset.sum_range_add]
      rw [← Finset.sum_add_distrib]
      apply Finset.sum_congr rfl
      intro y inside
      have low := lower_nonzero y (by rw [size]; simp at inside; omega)
      have high := lower_nonzero (half + y) (by rw [size]; simp at inside; omega)
      simp only [above]
      rw [up_numerator]
      simp only [up]
      rw [add_comm y half, div_add_div _ _ low high]
      congr 1
      ring

theorem root_fraction (L : Level K) (depth : ℕ)
    (nonzero : (L.above depth depth).denominator 0 ≠ 0) :
    (∀ x, x < 2 ^ depth → L.denominator x ≠ 0) ∧
      (L.above depth depth).numerator 0 / (L.above depth depth).denominator 0 =
        ∑ x ∈ Finset.range (2 ^ depth), L.numerator x / L.denominator x := by
  have result := above_fraction L depth depth le_rfl (by
    intro y lt
    rw [Nat.sub_self, pow_zero, Nat.lt_one_iff] at lt
    rw [lt]
    exact nonzero)
  simpa using result

end Level

end CompressionLean.Lib
