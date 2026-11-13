import NightstreamFPrime.Spec.Compression.Lib.Cube
import Mathlib.Algebra.BigOperators.Intervals

/-! Slices of a committed table (Rust `layout::Slice`). A slice is `2^vars` cells from `offset`, a
multiple of `2^vars`. A table that is zero outside a slice has the multilinear extension
`selector · local extension` (`Slice.mle_place`, Rust `Slice::selector`), and its sum against
another table is a sum over the slice (`Slice.sum_place`). The verifier opens `V` at the point of
the sum-check on the span, then the column point, then zeros (`mle_inverse_point`, Rust
`inverse_point`). -/

namespace NightstreamFPrime.Spec.Compression.Verifier

open Finset NightstreamFPrime.Spec.Compression.Lib

variable {K : Type} [Field K]

/-- `2^vars` cells of a table from `offset`. -/
structure Slice where
  offset : ℕ
  vars : ℕ

namespace Slice

/-- The slice is aligned and inside a cube of `n` variables. -/
def Fits (s : Slice) (n : ℕ) : Prop :=
  2 ^ s.vars ∣ s.offset ∧ s.offset + 2 ^ s.vars ≤ 2 ^ n ∧ s.vars ≤ n

/-- The table that is `f` on the slice and zero elsewhere. -/
def place (s : Slice) (f : ℕ → K) : ℕ → K :=
  fun x => if s.offset ≤ x ∧ x < s.offset + 2 ^ s.vars then f (x - s.offset) else 0

/-- Rust `Slice::selector`: the coordinates from `vars` select the bits of `offset`. -/
def selector (s : Slice) (point : List K) : K :=
  eqAt (point.drop s.vars) (s.offset / 2 ^ s.vars)

theorem sum_place (s : Slice) (n : ℕ) (fits : s.Fits n) (f g : ℕ → K) :
    ∑ x ∈ range (2 ^ n), s.place f x * g x = ∑ y ∈ range (2 ^ s.vars), f y * g (s.offset + y) := by
  obtain ⟨_, inside, _⟩ := fits
  rw [← sum_range_add_sum_Ico _ (show s.offset ≤ 2 ^ n by omega),
    ← sum_Ico_consecutive _ (show s.offset ≤ s.offset + 2 ^ s.vars by omega) inside]
  rw [sum_eq_zero (s := range s.offset) fun x member => by
      simp only [place, ite_eq_right (show ¬ (s.offset ≤ x ∧ x < s.offset + 2 ^ s.vars) by
        have := mem_range.mp member; omega), zero_mul],
    sum_eq_zero (s := Ico (s.offset + 2 ^ s.vars) (2 ^ n)) fun x member => by
      simp only [place, ite_eq_right (show ¬ (s.offset ≤ x ∧ x < s.offset + 2 ^ s.vars) by
        have := (mem_Ico.mp member).1; omega), zero_mul],
    zero_add, add_zero, sum_Ico_eq_sum_range, show s.offset + 2 ^ s.vars - s.offset = 2 ^ s.vars
      by omega]
  refine sum_congr rfl fun y member => ?_
  have := mem_range.mp member
  simp only [place, ite_eq_left (show s.offset ≤ s.offset + y ∧ s.offset + y < s.offset + 2 ^ s.vars
    by omega), Nat.add_sub_cancel_left]

theorem mle_place (s : Slice) (f : ℕ → K) (point : List K) (fits : s.Fits point.length) :
    mle point (s.place f) = s.selector point * mle (point.take s.vars) f := by
  obtain ⟨⟨j₀, aligned⟩, inside, short⟩ := fits
  have split : point = point.take s.vars ++ point.drop s.vars := (List.take_append_drop _ _).symm
  have low_length : (point.take s.vars).length = s.vars := by simp [short]
  have high_length : (point.drop s.vars).length = point.length - s.vars := by simp
  have block : 0 < 2 ^ s.vars := pow_pos two_pos _
  have j₀_lt : j₀ < 2 ^ (point.length - s.vars) := by
    have whole : 2 ^ point.length = 2 ^ s.vars * 2 ^ (point.length - s.vars) := by
      rw [← pow_add, Nat.add_sub_cancel' short]
    rw [aligned, whole] at inside
    by_contra large
    push Not at large
    have : 2 ^ s.vars * 2 ^ (point.length - s.vars) ≤ 2 ^ s.vars * j₀ :=
      Nat.mul_le_mul_left _ large
    omega
  have column (j : ℕ) : mle (point.take s.vars)
      (fun y => s.place f (y + 2 ^ (point.take s.vars).length * j)) =
      if j = j₀ then mle (point.take s.vars) f else 0 := by
    rw [low_length]
    split
    · rename_i same
      subst same
      refine mle_congr _ fun y small => ?_
      rw [low_length] at small
      simp only [place, aligned, ite_eq_left (show 2 ^ s.vars * j ≤ y + 2 ^ s.vars * j ∧
        y + 2 ^ s.vars * j < 2 ^ s.vars * j + 2 ^ s.vars by omega), Nat.add_sub_cancel]
    · rename_i different
      rw [← mle_const (point.take s.vars) 0]
      refine mle_congr _ fun y small => ?_
      rw [low_length] at small
      simp only [place, aligned]
      rw [ite_eq_right]
      rintro ⟨above, below⟩
      rcases Nat.lt_or_gt_of_ne different with less | more
      · have : 2 ^ s.vars * j + 2 ^ s.vars ≤ 2 ^ s.vars * j₀ := by
          have := Nat.mul_le_mul_left (2 ^ s.vars) (show j + 1 ≤ j₀ by omega)
          rw [mul_add, mul_one] at this
          exact this
        omega
      · have : 2 ^ s.vars * j₀ + 2 ^ s.vars ≤ 2 ^ s.vars * j := by
          have := Nat.mul_le_mul_left (2 ^ s.vars) (show j₀ + 1 ≤ j by omega)
          rw [mul_add, mul_one] at this
          exact this
        omega
  conv_lhs => rw [split, mle_append_split]
  simp only [column]
  rw [mle_eq_cubeSum, cubeSum_eq_sum_range]
  simp only [mul_ite, mul_zero]
  rw [sum_ite_eq' (range _) j₀, ite_eq_left (by rw [high_length]; exact mem_range.mpr j₀_lt),
    selector, aligned, Nat.mul_div_cancel_left _ block]

end Slice

/-- `eq(β, ·)` sums to one over the cube. -/
theorem sum_eqAt (β : List K) : ∑ y ∈ range (2 ^ β.length), eqAt β y = 1 := by
  have := mle_eq_cubeSum β (fun _ => (1 : K))
  rw [mle_const, cubeSum_eq_sum_range] at this
  rw [this]
  simp

/-- Rust `inverse_point`: `V` at the point of the span, then `column`, then zeros, is the
extension in `column` of the columns of `V`. -/
theorem mle_inverse_point (span : List K) (column : List K) (zeros : ℕ) (V : ℕ → K) :
    mle (span ++ column ++ List.replicate zeros 0) V =
      mle column (fun k => mle span (fun y => V (y + 2 ^ span.length * k))) := by
  rw [mle_append_zeros, mle_append_split]

end NightstreamFPrime.Spec.Compression.Verifier
