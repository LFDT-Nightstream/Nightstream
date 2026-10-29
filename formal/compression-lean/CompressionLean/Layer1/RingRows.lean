import CompressionLean.Lib.Roots
import Mathlib.Algebra.Polynomial.Div
import Mathlib.Algebra.Polynomial.BigOperators
import Mathlib.Tactic.LinearCombination

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {F K : Type} [Field F] [Field K] [Algebra F K]

structure RingRows (F : Type) [Field F] where
  rows : ℕ
  blocks : ℕ
  weight : ℕ → ℕ → F[X]
  target : ℕ → F[X]

namespace RingRows

variable (R : RingRows F) (Φ : F[X]) (Z : ℕ → F[X])

noncomputable def residual (ρ : ℕ) : F[X] :=
  ∑ b ∈ Finset.range R.blocks, R.weight ρ b * Z b - R.target ρ

def Holds : Prop := ∀ ρ, ρ < R.rows → Φ ∣ R.residual Z ρ

noncomputable def batched (lam : K) : K[X] :=
  ∑ ρ ∈ Finset.range R.rows, C (lam ^ ρ) * (R.residual Z ρ).map (algebraMap F K)

theorem natDegree_powers_le (n : ℕ) (c : ℕ → K) :
    (∑ σ ∈ Finset.range n, C (c σ) * X ^ σ).natDegree ≤ n - 1 := by
  apply natDegree_sum_le_of_forall_le
  intro σ member
  refine (natDegree_C_mul_le _ _).trans ?_
  rw [natDegree_X_pow]
  have := Finset.mem_range.mp member
  omega

theorem coeff_powers (n : ℕ) (c : ℕ → K) {σ : ℕ} (inside : σ < n) :
    (∑ τ ∈ Finset.range n, C (c τ) * X ^ τ).coeff σ = c σ := by
  simp [finsetSum_coeff, inside]

theorem batched_eval (lam ζ : K) :
    (R.batched Z lam).eval ζ =
      ∑ b ∈ Finset.range R.blocks,
          (∑ ρ ∈ Finset.range R.rows, lam ^ ρ * ((R.weight ρ b).map (algebraMap F K)).eval ζ) *
            ((Z b).map (algebraMap F K)).eval ζ -
        ∑ ρ ∈ Finset.range R.rows, lam ^ ρ * ((R.target ρ).map (algebraMap F K)).eval ζ := by
  simp only [batched, residual, eval_finsetSum, eval_mul, eval_C, Polynomial.map_sub,
    Polynomial.map_sum, Polynomial.map_mul, eval_sub, Finset.sum_mul, mul_sub, Finset.mul_sum]
  rw [Finset.sum_sub_distrib, Finset.sum_comm]
  congr 1
  apply Finset.sum_congr rfl
  intro b _
  apply Finset.sum_congr rfl
  intro ρ _
  ring

variable [Fintype K] [DecidableEq K]

open Classical in
noncomputable def lambdaBad : Finset K :=
  if h : ∃ ρ, ρ < R.rows ∧ R.residual Z ρ %ₘ Φ ≠ 0 then
    zeros (∑ σ ∈ Finset.range R.rows,
      C (algebraMap F K ((R.residual Z σ %ₘ Φ).coeff (R.residual Z h.choose %ₘ Φ).natDegree)) * X ^ σ)
  else ∅

noncomputable def zetaBad (lam : K) (Q : K[X]) : Finset K :=
  zeros (R.batched Z lam - Q * Φ.map (algebraMap F K))

theorem lambdaBad_card : (R.lambdaBad (K := K) Φ Z).card ≤ R.rows - 1 := by
  unfold lambdaBad
  split
  · exact zeros_card_le_of_natDegree_le (natDegree_powers_le (K := K) R.rows _)
  · simp

theorem ring_sound (monic : Φ.Monic) (proper : Φ ≠ 1) (lam ζ : K) (Q : K[X])
    (lam_good : lam ∉ R.lambdaBad Φ Z) (zeta_good : ζ ∉ R.zetaBad Φ Z lam Q)
    (check : (R.batched Z lam).eval ζ = Q.eval ζ * (Φ.map (algebraMap F K)).eval ζ) :
    R.Holds Φ Z := by
  classical
  by_contra failing
  have h : ∃ ρ, ρ < R.rows ∧ R.residual Z ρ %ₘ Φ ≠ 0 := by
    unfold Holds at failing
    push Not at failing
    obtain ⟨ρ, inside, not_dvd⟩ := failing
    exact ⟨ρ, inside, fun zero => not_dvd ((modByMonic_eq_zero_iff_dvd monic).mp zero)⟩
  obtain ⟨ρ₀, ρ₀_lt, ρ₀_rem⟩ := h
  have h' : ∃ ρ, ρ < R.rows ∧ R.residual Z ρ %ₘ Φ ≠ 0 := ⟨ρ₀, ρ₀_lt, ρ₀_rem⟩
  have bad_eq : R.lambdaBad Φ Z = zeros (∑ σ ∈ Finset.range R.rows,
      C (algebraMap F K ((R.residual Z σ %ₘ Φ).coeff (R.residual Z h'.choose %ₘ Φ).natDegree)) * X ^ σ) := by
    simp only [lambdaBad, dite_eq_left h']
  have live : (R.residual Z h'.choose %ₘ Φ).coeff (R.residual Z h'.choose %ₘ Φ).natDegree ≠ 0 :=
    leadingCoeff_ne_zero.mpr h'.choose_spec.2
  generalize (R.residual Z h'.choose %ₘ Φ).natDegree = l at bad_eq live
  have powers_nonzero : (∑ σ ∈ Finset.range R.rows,
      C (algebraMap F K ((R.residual Z σ %ₘ Φ).coeff l)) * X ^ σ).eval lam ≠ 0 := by
    have nonzero : (∑ σ ∈ Finset.range R.rows,
        C (algebraMap F K ((R.residual Z σ %ₘ Φ).coeff l)) * X ^ σ) ≠ 0 := by
      intro zero
      have := congrArg (fun p => p.coeff h'.choose) zero
      simp only [coeff_zero] at this
      rw [coeff_powers _ _ h'.choose_spec.1] at this
      exact live ((map_eq_zero_iff (algebraMap F K) (algebraMap F K).injective).mp this)
    intro zero
    apply lam_good
    rw [bad_eq]
    unfold zeros
    rw [ite_eq_right nonzero]
    simpa using zero
  have split : R.batched Z lam =
      (∑ σ ∈ Finset.range R.rows, C (lam ^ σ) * (R.residual Z σ %ₘ Φ).map (algebraMap F K)) +
        Φ.map (algebraMap F K) *
          ∑ σ ∈ Finset.range R.rows, C (lam ^ σ) * (R.residual Z σ /ₘ Φ).map (algebraMap F K) := by
    simp only [batched, Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro σ _
    conv_lhs => rw [← modByMonic_add_div (R.residual Z σ) Φ]
    simp only [Polynomial.map_add, Polynomial.map_mul]
    ring
  generalize hB : (∑ σ ∈ Finset.range R.rows, C (lam ^ σ) * (R.residual Z σ %ₘ Φ).map (algebraMap F K)) = B
    at split
  generalize hS : (∑ σ ∈ Finset.range R.rows, C (lam ^ σ) * (R.residual Z σ /ₘ Φ).map (algebraMap F K)) = S
    at split
  have rem_nonzero : B ≠ 0 := by
    intro zero
    have := congrArg (fun p => p.coeff l) (hB.trans zero)
    simp only [finsetSum_coeff, coeff_C_mul, coeff_map, coeff_zero] at this
    apply powers_nonzero
    rw [eval_finsetSum, ← this]
    apply Finset.sum_congr rfl
    intro σ _
    simp only [eval_mul, eval_C, eval_pow, eval_X]
    ring
  have degree_map : (Φ.map (algebraMap F K)).natDegree = Φ.natDegree :=
    natDegree_map_eq_of_injective (algebraMap F K).injective _
  have positive : 0 < Φ.natDegree := by
    by_contra zero
    push Not at zero
    exact proper (eq_one_of_monic_natDegree_zero monic (by omega))
  have rem_small : B.natDegree < Φ.natDegree := by
    rw [← hB]
    refine lt_of_le_of_lt (natDegree_sum_le_of_forall_le _ _ (n := Φ.natDegree - 1) ?_) (by omega)
    intro σ _
    refine (natDegree_C_mul_le _ _).trans ?_
    rw [natDegree_map_eq_of_injective (algebraMap F K).injective]
    have := natDegree_modByMonic_lt (R.residual Z σ) monic proper
    omega
  have nonzero : R.batched Z lam - Q * Φ.map (algebraMap F K) ≠ 0 := by
    rw [split]
    intro zero
    have equal : B = Φ.map (algebraMap F K) * (Q - S) := by linear_combination zero
    by_cases same : Q - S = 0
    · rw [same, mul_zero] at equal
      exact rem_nonzero equal
    · have lower : Φ.natDegree ≤ (Φ.map (algebraMap F K) * (Q - S)).natDegree := by
        rw [natDegree_mul (monic.map (algebraMap F K)).ne_zero same, degree_map]
        omega
      rw [← equal] at lower
      omega
  apply zeta_good
  unfold zetaBad zeros
  rw [ite_eq_right nonzero]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, eval_sub, eval_mul]
  rw [check, sub_self]

theorem zetaBad_card (D : ℕ) (lam : K) (Q : K[X])
    (weight_degree : ∀ ρ b, (R.weight ρ b).natDegree ≤ D - 1) (Z_degree : ∀ b, (Z b).natDegree ≤ D - 1)
    (target_degree : ∀ ρ, (R.target ρ).natDegree ≤ D - 1) (Q_degree : Q.natDegree ≤ D - 2)
    (Φ_degree : Φ.natDegree = D) (positive : 2 ≤ D) : (R.zetaBad Φ Z lam Q).card ≤ 2 * D - 2 := by
  apply zeros_card_le_of_natDegree_le
  have residual_degree : ∀ ρ, (R.residual Z ρ).natDegree ≤ 2 * D - 2 := by
    intro ρ
    refine (natDegree_sub_le _ _).trans (max_le ?_ ((target_degree ρ).trans (by omega)))
    refine natDegree_sum_le_of_forall_le _ _ fun b _ => (natDegree_mul_le).trans ?_
    have := weight_degree ρ b
    have := Z_degree b
    omega
  refine (natDegree_sub_le _ _).trans (max_le ?_ ?_)
  · refine natDegree_sum_le_of_forall_le _ _ fun ρ _ => (natDegree_C_mul_le _ _).trans ?_
    exact (natDegree_map_le).trans (residual_degree ρ)
  · refine (natDegree_mul_le).trans ?_
    have := natDegree_map_le (f := algebraMap F K) (p := Φ)
    omega

end RingRows

end CompressionLean.Layer1
