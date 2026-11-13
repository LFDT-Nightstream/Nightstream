import NightstreamFPrime.Spec.Compression.Algebra.Ring
import NightstreamFPrime.Spec.Compression.Lib.Density
import NightstreamFPrime.Spec.Compression.Lib.SumCheck

/-! The batch of the ring equations and the public input of the claims (Rust `weights::table` and
`weights::target`, SPEC.md section 6).

One claim has a witness of `n` blocks of `D` lanes, `E` ring equations `Σ_j P_{e,j} · z_j = Y_e`
and `m` public cells. Its weight on cell `D·j + h` is
`Σ_e ω^e · (X^h · P_{e,j})(δ) + ω^E · γ^(D·j + h) · [D·j + h < m]` (`claimWeight`), and its target is
`Σ_e ω^e · Y_e(δ) + ω^E · Σ_{x < m} γ^x · X_x` (`claimTarget`). The inner product of the witness with
the weights minus the target is a polynomial in `(ω, δ, γ)` whose coefficients are the errors of
the equations and of the public input (`claim_difference`). The claims are batched with powers of
`μ`. When an equation or a public cell of a claim fails, the batch vanishes with density at most
`(s − 1 + E + max(D − 1, m − 1)) / |K|` (`density_claims_le`): the term of Rust
`neo_compress::error_bits` with `E = κ + 1 + t`. -/

namespace NightstreamFPrime.Spec.Compression.Algebra

open Finset NightstreamFPrime.Spec.Compression.Lib

variable {K : Type} [Field K]

section Weights

/-- The weight of one claim on its cell `x = D·j + h`. -/
def claimWeight (D E m : ℕ) (P : ℕ → ℕ → ℕ → K) (ω δ γ : K) (x : ℕ) : K :=
  ∑ e ∈ range E, ω ^ e * lanes D δ (P e (x / D)) (x % D) + if x < m then ω ^ E * γ ^ x else 0

/-- The target of one claim (Rust `weights::target`). -/
def claimTarget (D E m : ℕ) (Y : ℕ → ℕ → K) (X : ℕ → K) (ω δ γ : K) : K :=
  ∑ e ∈ range E, ω ^ e * value D δ (Y e) + ω ^ E * ∑ x ∈ range m, γ ^ x * X x

/-- The left side `Σ_{j < n} P_{e,j} · z_j` of ring equation `e`. -/
def ringLeft (D n : ℕ) (P : ℕ → ℕ → ℕ → K) (z : ℕ → K) (e : ℕ) : ℕ → K :=
  fun l => ∑ j ∈ range n, mul D (P e j) (fun h => z (D * j + h)) l

/-- The witness against the weights, minus the target, is the batch of the errors. -/
theorem claim_difference (D n E m : ℕ) (P : ℕ → ℕ → ℕ → K) (Y : ℕ → ℕ → K) (X z : ℕ → K)
    (ω δ γ : K) (lanes_pos : 0 < D) (fits : m ≤ D * n) :
    ∑ x ∈ range (D * n), z x * claimWeight D E m P ω δ γ x - claimTarget D E m Y X ω δ γ =
      ∑ e ∈ range E, ω ^ e * (value D δ (ringLeft D n P z e) - value D δ (Y e)) +
        ω ^ E * ∑ x ∈ range m, γ ^ x * (z x - X x) := by
  have rings : ∑ x ∈ range (D * n), z x * ∑ e ∈ range E, ω ^ e * lanes D δ (P e (x / D)) (x % D) =
      ∑ e ∈ range E, ω ^ e * value D δ (ringLeft D n P z e) := by
    simp only [mul_sum]
    rw [sum_comm]
    refine sum_congr rfl fun e _ => ?_
    rw [show ringLeft D n P z e = fun l => ∑ j ∈ range n, mul D (P e j) (fun h => z (D * j + h)) l
      from rfl, value_sum, mul_sum]
    rw [sum_congr rfl fun x _ => show z x * (ω ^ e * lanes D δ (P e (x / D)) (x % D)) =
      (fun j h => z (D * j + h) * (ω ^ e * lanes D δ (P e j) h)) (x / D) (x % D) by
        simp only [Nat.div_add_mod], sum_range_mul_lanes D n lanes_pos
        (fun j h => z (D * j + h) * (ω ^ e * lanes D δ (P e j) h))]
    refine sum_congr rfl fun j _ => ?_
    rw [value_mul, mul_sum]
    exact sum_congr rfl fun h _ => by ring
  have publics : ∑ x ∈ range (D * n), z x * (if x < m then ω ^ E * γ ^ x else 0) =
      ω ^ E * ∑ x ∈ range m, γ ^ x * z x := by
    rw [← sum_range_add_sum_Ico _ fits, sum_eq_zero (s := Ico m (D * n)) fun x member => by
      rw [ite_eq_right (by have := (mem_Ico.mp member).1; omega), mul_zero], add_zero, mul_sum]
    exact sum_congr rfl fun x member => by rw [ite_eq_left (mem_range.mp member)]; ring
  simp only [claimWeight, claimTarget, mul_add, sum_add_distrib, rings, publics, mul_sub,
    sum_sub_distrib]
  ring
where
  sum_range_mul_lanes (D n : ℕ) (lanes_pos : 0 < D) (f : ℕ → ℕ → K) :
      ∑ x ∈ range (D * n), f (x / D) (x % D) = ∑ j ∈ range n, ∑ h ∈ range D, f j h := by
    induction n with
    | zero => simp
    | succ n ih =>
        rw [mul_add, mul_one, sum_range_add, ih, sum_range_succ]
        congr 1
        refine sum_congr rfl fun h member => ?_
        have lane := mem_range.mp member
        rw [Nat.add_comm, Nat.add_mul_div_left _ _ lanes_pos, Nat.div_eq_of_lt lane, zero_add,
          Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt lane]

end Weights

section Density

variable [Fintype K] [DecidableEq K]

omit [Fintype K] [DecidableEq K] in
theorem coeffPoly_coeff_eq {m : ℕ} (message : Fin m → K) (k : ℕ) (lt : k < m) :
    (coeffPoly message).coeff k = message ⟨k, lt⟩ := by
  rw [coeffPoly, Polynomial.finsetSum_coeff]
  simp only [Polynomial.coeff_C_mul_X_pow]
  rw [sum_eq_single ⟨k, lt⟩ (fun i _ different => by
    rw [ite_eq_right (fun same => different (Fin.ext same.symm))]) (by simp)]
  simp

/-- A nonzero polynomial of degree at most `n`, given by its coefficients, vanishes with density
at most `n / |K|`. -/
theorem density_poly_zero_le (n : ℕ) (a : ℕ → K) (i : ℕ) (le : i ≤ n) (nonzero : a i ≠ 0) :
    density (univ.filter fun c : K => ∑ k ∈ range (n + 1), a k * c ^ k = 0) ≤
      n / Fintype.card K := by
  have p_ne : coeffPoly (fun k : Fin (n + 1) => a k) ≠ 0 := fun zero => nonzero (by
    have := coeffPoly_coeff_eq (fun k : Fin (n + 1) => a k) i (by omega)
    rw [zero, Polynomial.coeff_zero] at this
    exact this.symm)
  refine (density_mono (T := zeros (coeffPoly fun k : Fin (n + 1) => a k)) fun c member => ?_).trans
    (density_zeros_le (coeffPoly_natDegree fun k : Fin (n + 1) => a k))
  unfold zeros
  rw [ite_eq_right p_ne]
  simp only [mem_filter, mem_univ, true_and] at member ⊢
  rw [coeffPoly_eval, ← member, Fin.sum_univ_eq_sum_range (fun k => a k * c ^ k)]

/-- A polynomial in `c` whose coefficient `i` is nonzero outside `SB` vanishes on pairs with
density at most `n / |K| + density SB`. -/
theorem density_batch_le {B : Type} [Fintype B] [Nonempty B] [DecidableEq B] (n : ℕ)
    (a : ℕ → B → K) (i : ℕ) (le : i ≤ n) (SB : Finset B) (live : ∀ b, b ∉ SB → a i b ≠ 0) :
    density (univ.filter fun p : K × B => ∑ k ∈ range (n + 1), a k p.2 * p.1 ^ k = 0) ≤
      n / Fintype.card K + density SB := by
  refine density_prod_le _ SB _ (by positivity) fun b outside => ?_
  simpa using density_poly_zero_le n (fun k => a k b) i le (live b outside)

/-- The ring part of the batch at `(ω, δ, γ)`: one claim's errors. -/
def claimBatch (D E m : ℕ) (L Y : ℕ → ℕ → K) (z X : ℕ → K) (ω δ γ : K) : K :=
  ∑ e ∈ range E, ω ^ e * (value D δ (L e) - value D δ (Y e)) +
    ω ^ E * ∑ x ∈ range m, γ ^ x * (z x - X x)

/-- Coefficient `k` of `claimBatch` as a polynomial in `ω`. -/
def claimCoeff (D E m : ℕ) (L Y : ℕ → ℕ → K) (z X : ℕ → K) (k : ℕ) (b : K × K) : K :=
  if k < E then value D b.1 (L k) - value D b.1 (Y k) else ∑ x ∈ range m, b.2 ^ x * (z x - X x)

omit [Fintype K] [DecidableEq K] in
theorem claimBatch_eq (D E m : ℕ) (L Y : ℕ → ℕ → K) (z X : ℕ → K) (ω δ γ : K) :
    claimBatch D E m L Y z X ω δ γ =
      ∑ k ∈ range (E + 1), claimCoeff D E m L Y z X k (δ, γ) * ω ^ k := by
  rw [sum_range_succ, claimBatch, claimCoeff, ite_eq_right (lt_irrefl E), mul_comm]
  congr 1
  exact sum_congr rfl fun k member => by
    rw [claimCoeff, ite_eq_left (mem_range.mp member), mul_comm]

/-- A claim fails when a coefficient of a ring equation differs or a public cell differs. -/
def ClaimFails (D E m : ℕ) (L Y : ℕ → ℕ → K) (z X : ℕ → K) : Prop :=
  (∃ e < E, ∃ l < D, L e l ≠ Y e l) ∨ ∃ x < m, z x ≠ X x

omit [DecidableEq K] in
theorem density_univ_eq_one : density (univ : Finset K) = 1 := by
  simp [density, Fintype.card_ne_zero]

theorem value_zero_density (D : ℕ) (a : ℕ → K) (l : ℕ) (lane : l < D) (nonzero : a l ≠ 0) :
    density (univ.filter fun δ : K => value D δ a = 0) ≤ ((D - 1 : ℕ) : ℝ) / Fintype.card K := by
  have := density_poly_zero_le (D - 1) a l (by omega) nonzero
  rw [show D - 1 + 1 = D by omega] at this
  exact this

/-- One failing claim: its batch vanishes on `(ω, δ, γ)` with density at most
`(E + max(D − 1, m − 1)) / |K|`. -/
theorem density_claimBatch_le (D E m : ℕ) (L Y : ℕ → ℕ → K) (z X : ℕ → K)
    (fails : ClaimFails D E m L Y z X) :
    density (univ.filter fun p : K × K × K => claimBatch D E m L Y z X p.1 p.2.1 p.2.2 = 0) ≤
      (E + max (D - 1) (m - 1) : ℕ) / Fintype.card K := by
  have shape : (univ.filter fun p : K × K × K => claimBatch D E m L Y z X p.1 p.2.1 p.2.2 = 0) =
      univ.filter fun p : K × K × K =>
        ∑ k ∈ range (E + 1), claimCoeff D E m L Y z X k p.2 * p.1 ^ k = 0 := by
    ext p
    simp only [mem_filter, mem_univ, true_and, claimBatch_eq]
  rw [shape]
  have bound (i : ℕ) (SB : Finset (K × K)) (le : i ≤ E)
      (small : density SB ≤ (max (D - 1) (m - 1) : ℕ) / Fintype.card K)
      (live : ∀ b, b ∉ SB → claimCoeff D E m L Y z X i b ≠ 0) :
      density (univ.filter fun p : K × K × K =>
        ∑ k ∈ range (E + 1), claimCoeff D E m L Y z X k p.2 * p.1 ^ k = 0) ≤
        (E + max (D - 1) (m - 1) : ℕ) / Fintype.card K := by
    refine (density_batch_le E (claimCoeff D E m L Y z X) i le SB live).trans ?_
    rw [Nat.cast_add, add_div]
    exact add_le_add le_rfl small
  rcases fails with ⟨e, e_lt, l, lane, differ⟩ | ⟨x, cell, differ⟩
  · refine bound e ((univ.filter fun δ : K => value D δ (fun k => L e k - Y e k) = 0) ×ˢ univ)
      e_lt.le ?_ fun b outside => ?_
    · rw [density_product, density_univ_eq_one, mul_one]
      refine (value_zero_density D _ l lane (sub_ne_zero.mpr differ)).trans ?_
      exact div_le_div_of_nonneg_right (by exact_mod_cast le_max_left _ _) (by positivity)
    · rw [claimCoeff, ite_eq_left e_lt, ← value_sub]
      simpa using outside
  · refine bound E (univ ×ˢ (univ.filter fun γ : K => ∑ x ∈ range m, (z x - X x) * γ ^ x = 0))
      le_rfl ?_ fun b outside => ?_
    · rw [density_product, density_univ_eq_one, one_mul]
      have := density_poly_zero_le (m - 1) (fun x => z x - X x) x (by omega)
        (sub_ne_zero.mpr differ)
      rw [show m - 1 + 1 = m by omega] at this
      exact this.trans (div_le_div_of_nonneg_right (by exact_mod_cast le_max_right _ _)
        (by positivity))
    · rw [claimCoeff, ite_eq_right (lt_irrefl E)]
      intro zero
      apply outside
      simp only [mem_product, mem_univ, mem_filter, true_and]
      rw [← zero]
      exact sum_congr rfl fun k _ => mul_comm _ _

/-- The claims batched with powers of `μ`: when one claim fails, the batch vanishes on
`(μ, ω, δ, γ)` with density at most `(s − 1 + E + max(D − 1, m − 1)) / |K|`. -/
theorem density_claims_le (s D E m : ℕ) (L Y : ℕ → ℕ → ℕ → K) (z X : ℕ → ℕ → K) (c : ℕ)
    (claim : c < s) (fails : ClaimFails D E m (L c) (Y c) (z c) (X c)) :
    density (univ.filter fun p : K × K × K × K => ∑ k ∈ range s, p.1 ^ k *
      claimBatch D E m (L k) (Y k) (z k) (X k) p.2.1 p.2.2.1 p.2.2.2 = 0) ≤
      ((s - 1 : ℕ) : ℝ) / Fintype.card K + (E + max (D - 1) (m - 1) : ℕ) / Fintype.card K := by
  let coeff : ℕ → K × K × K → K := fun k b =>
    if k < s then claimBatch D E m (L k) (Y k) (z k) (X k) b.1 b.2.1 b.2.2 else 0
  have shape : (univ.filter fun p : K × K × K × K => ∑ k ∈ range s, p.1 ^ k *
      claimBatch D E m (L k) (Y k) (z k) (X k) p.2.1 p.2.2.1 p.2.2.2 = 0) =
      univ.filter fun p : K × K × K × K => ∑ k ∈ range (s - 1 + 1), coeff k p.2 * p.1 ^ k = 0 := by
    ext p
    simp only [mem_filter, mem_univ, true_and, show s - 1 + 1 = s by omega]
    rw [sum_congr rfl fun k member => show coeff k p.2 * p.1 ^ k =
      p.1 ^ k * claimBatch D E m (L k) (Y k) (z k) (X k) p.2.1 p.2.2.1 p.2.2.2 by
        simp only [coeff, ite_eq_left (mem_range.mp member)]; ring]
  rw [shape]
  refine (density_batch_le (s - 1) coeff c (by omega)
    (univ.filter fun b : K × K × K =>
      claimBatch D E m (L c) (Y c) (z c) (X c) b.1 b.2.1 b.2.2 = 0)
    fun b outside => by
      simp only [coeff, ite_eq_left claim]
      simpa using outside).trans ?_
  exact add_le_add le_rfl (density_claimBatch_le D E m (L c) (Y c) (z c) (X c) fails)

end Density

end NightstreamFPrime.Spec.Compression.Algebra
