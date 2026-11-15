import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Ring

/-! Ring equations of `R[X]/(X^D + 1)` read at a point `δ` (Rust `ring`, SPEC.md section 6). A ring
element is its `D` coefficients. Its value at `δ` is the value of its reduced polynomial (`value`,
Rust `Point::value`), and lane `h` of a block with the key element `P` has the weight
`(X^h · P)(δ)` (`lanes`, Rust `Point::lanes`). The value of a product is the inner product of one
factor with the lane weights of the other (`value_mul`), and the pairing is symmetric
(`value_mul_comm`). `lanes_eq_split` is the negacyclic wrap that Rust computes: the terms below
`D − h` keep their sign and shift up by `h`, and the others change sign and shift down by
`D − h`. -/

namespace NightstreamFPrime.Spec.Compression.Algebra

open Finset

variable {R : Type} [CommRing R]

/-- The value at `δ` of the ring element with the coefficients `a`. -/
def value (D : ℕ) (δ : R) (a : ℕ → R) : R := ∑ l ∈ range D, a l * δ ^ l

/-- `X^h · a` in `R[X]/(X^D + 1)`, for `h < D`. -/
def mulX (D h : ℕ) (a : ℕ → R) : ℕ → R :=
  fun l => if h ≤ l then a (l - h) else -a (l + D - h)

/-- The product of `R[X]/(X^D + 1)`: `a · b = Σ_h b[h] · X^h · a`. -/
def mul (D : ℕ) (a b : ℕ → R) : ℕ → R := fun l => ∑ h ∈ range D, b h * mulX D h a l

/-- The weight `(X^h · P)(δ)` of lane `h`. -/
def lanes (D : ℕ) (δ : R) (P : ℕ → R) (h : ℕ) : R := value D δ (mulX D h P)

/-- `(X^h · X^l)(δ)`: `δ^(h + l)`, with the sign of the wrap when `h + l ≥ D`. -/
def monomial (D : ℕ) (δ : R) (h l : ℕ) : R :=
  if h + l < D then δ ^ (h + l) else -δ ^ (h + l - D)

theorem monomial_comm (D : ℕ) (δ : R) (h l : ℕ) : monomial D δ h l = monomial D δ l h := by
  simp only [monomial, Nat.add_comm h l]

theorem value_add (D : ℕ) (δ : R) (a b : ℕ → R) :
    value D δ (fun l => a l + b l) = value D δ a + value D δ b := by
  simp only [value, add_mul, sum_add_distrib]

theorem value_sub (D : ℕ) (δ : R) (a b : ℕ → R) :
    value D δ (fun l => a l - b l) = value D δ a - value D δ b := by
  simp only [value, sub_mul, sum_sub_distrib]

theorem value_smul (D : ℕ) (δ c : R) (a : ℕ → R) :
    value D δ (fun l => c * a l) = c * value D δ a := by
  simp only [value, mul_sum, mul_assoc]

theorem value_sum {ι : Type} (D : ℕ) (δ : R) (s : Finset ι) (a : ι → ℕ → R) :
    value D δ (fun l => ∑ i ∈ s, a i l) = ∑ i ∈ s, value D δ (a i) := by
  simp only [value, sum_mul]
  exact sum_comm

/-- The lane weights as a sum over the coefficients of `P`. -/
theorem lanes_eq_monomial (D : ℕ) (δ : R) (P : ℕ → R) {h : ℕ} (lane : h < D) :
    lanes D δ P h = ∑ l ∈ range D, P l * monomial D δ h l := by
  have low : ∑ l ∈ range h, mulX D h P l * δ ^ l =
      ∑ l ∈ Ico (D - h) D, P l * -δ ^ (h + l - D) := by
    rw [sum_Ico_eq_sum_range, show D - (D - h) = h by omega]
    refine sum_congr rfl fun k member => ?_
    have k_lt := mem_range.mp member
    simp only [mulX, ite_eq_right (show ¬ h ≤ k by omega)]
    rw [show k + D - h = D - h + k by omega, show h + (D - h + k) - D = k by omega]
    ring
  have high : ∑ l ∈ Ico h D, mulX D h P l * δ ^ l = ∑ l ∈ range (D - h), P l * δ ^ (h + l) := by
    rw [sum_Ico_eq_sum_range]
    refine sum_congr rfl fun k _ => ?_
    simp only [mulX, ite_eq_left (show h ≤ h + k by omega), show h + k - h = k by omega]
  have whole : ∑ l ∈ range D, P l * monomial D δ h l =
      ∑ l ∈ range (D - h), P l * δ ^ (h + l) + ∑ l ∈ Ico (D - h) D, P l * -δ ^ (h + l - D) := by
    rw [← sum_range_add_sum_Ico _ (show D - h ≤ D by omega)]
    congr 1
    · refine sum_congr rfl fun l member => ?_
      simp only [monomial, ite_eq_left (show h + l < D by have := mem_range.mp member; omega)]
    · refine sum_congr rfl fun l member => ?_
      simp only [monomial, ite_eq_right (show ¬ h + l < D by have := (mem_Ico.mp member).1; omega)]
  rw [lanes, value, ← sum_range_add_sum_Ico _ lane.le, low, high, whole, add_comm]

/-- Rust `Point::lanes`: `δ^h · Σ_{l < D − h} P[l] δ^l − Σ_{l ≥ D − h} P[l] δ^(l − (D − h))`. -/
theorem lanes_eq_split (D : ℕ) (δ : R) (P : ℕ → R) {h : ℕ} (lane : h < D) :
    lanes D δ P h = δ ^ h * ∑ l ∈ range (D - h), P l * δ ^ l -
      ∑ l ∈ Ico (D - h) D, P l * δ ^ (l - (D - h)) := by
  rw [lanes_eq_monomial D δ P lane, ← sum_range_add_sum_Ico _ (show D - h ≤ D by omega), mul_sum,
    sub_eq_add_neg, ← sum_neg_distrib]
  congr 1
  · refine sum_congr rfl fun l member => ?_
    simp only [monomial, ite_eq_left (show h + l < D by have := mem_range.mp member; omega)]
    ring
  · refine sum_congr rfl fun l member => ?_
    have := (mem_Ico.mp member).1
    simp only [monomial, ite_eq_right (show ¬ h + l < D by omega), show h + l - D = l - (D - h) by omega]
    ring

/-- The value of a product is the inner product of one factor with the lane weights of the other:
block `j` of a ring equation `Σ_j P_j · z_j = Y` contributes `Σ_h z_j[h] · (X^h · P_j)(δ)`. -/
theorem value_mul (D : ℕ) (δ : R) (a b : ℕ → R) :
    value D δ (mul D a b) = ∑ h ∈ range D, b h * lanes D δ a h := by
  simp only [value, mul, lanes, sum_mul]
  rw [sum_comm]
  exact sum_congr rfl fun h _ => by rw [mul_sum]; exact sum_congr rfl fun l _ => by ring

/-- The pairing is symmetric: `Σ_h b[h] · (X^h · a)(δ) = Σ_h a[h] · (X^h · b)(δ)`. So the lane
weights of `E` are the linear form `P ↦ (E · P)(δ)` (Rust `weights::Uses`). -/
theorem value_mul_comm (D : ℕ) (δ : R) (a b : ℕ → R) :
    ∑ h ∈ range D, b h * lanes D δ a h = ∑ h ∈ range D, a h * lanes D δ b h := by
  have expand (x y : ℕ → R) : ∑ h ∈ range D, y h * lanes D δ x h =
      ∑ h ∈ range D, ∑ l ∈ range D, y h * x l * monomial D δ h l := by
    refine sum_congr rfl fun h member => ?_
    rw [lanes_eq_monomial D δ x (mem_range.mp member), mul_sum]
    exact sum_congr rfl fun l _ => by ring
  rw [expand, expand, sum_comm]
  exact sum_congr rfl fun h _ => sum_congr rfl fun l _ => by rw [monomial_comm]; ring

/-- The conjugate `X ↦ X^(−1)` (Rust `ring::conjugate`, the map `bar` of SPEC.md section 2.2). -/
def conjugate (D : ℕ) (a : ℕ → R) : ℕ → R := fun l => if l = 0 then a 0 else -a (D - l)

/-- The constant term of `conjugate(a) · z` is `Σ_h a[h] · z[h]`. So `Eval_K`, read at its
constant term, is the multilinear extension of the witness at `r`. -/
theorem mul_conjugate_zero (D : ℕ) (a z : ℕ → R) :
    mul D (conjugate D a) z 0 = ∑ h ∈ range D, a h * z h := by
  simp only [mul, mulX]
  refine sum_congr rfl fun h member => ?_
  have lane := mem_range.mp member
  by_cases zero : h = 0
  · subst zero
    simp [conjugate]
    ring
  · rw [ite_eq_right (show ¬ h ≤ 0 by omega)]
    simp only [conjugate, ite_eq_right (show ¬ 0 + D - h = 0 by omega),
      show D - (0 + D - h) = h by omega, neg_neg]
    ring

end NightstreamFPrime.Spec.Compression.Algebra
