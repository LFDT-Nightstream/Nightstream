import Mathlib.Algebra.Polynomial.Roots

namespace CompressionLean.Lib

open Polynomial

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

noncomputable def zeros (p : K[X]) : Finset K :=
  if p = 0 then ∅ else Finset.univ.filter (fun c => p.eval c = 0)

theorem zeros_card_le (p : K[X]) : (zeros p).card ≤ p.natDegree := by
  unfold zeros
  split
  · simp
  · rename_i nonzero
    apply Polynomial.card_le_degree_of_subset_roots
    intro c inside
    simp only [Finset.mem_val, Finset.mem_filter] at inside
    exact (Polynomial.mem_roots nonzero).2 inside.2

theorem eval_ne_of_not_mem_zeros {p q : K[X]} (different : p ≠ q) {c : K}
    (outside : c ∉ zeros (p - q)) : p.eval c ≠ q.eval c := by
  intro equal
  apply outside
  unfold zeros
  rw [ite_eq_right (sub_ne_zero.mpr different)]
  simp [equal]

theorem zeros_card_le_of_natDegree_le {p : K[X]} {d : ℕ} (degree : p.natDegree ≤ d) :
    (zeros p).card ≤ d :=
  (zeros_card_le p).trans degree

omit [Field K] [Fintype K] in
theorem card_biUnion_le_mul {W : Type} (worlds : Finset W) (bad : W → Finset K) (d : ℕ)
    (each : ∀ w ∈ worlds, (bad w).card ≤ d) :
    (worlds.biUnion bad).card ≤ worlds.card * d := by
  calc (worlds.biUnion bad).card ≤ ∑ w ∈ worlds, (bad w).card := Finset.card_biUnion_le
    _ ≤ ∑ _w ∈ worlds, d := Finset.sum_le_sum each
    _ = worlds.card * d := by simp

section Line

omit [Fintype K] [DecidableEq K]

noncomputable def linePoly (a b : K) : K[X] := C a + X * C b

theorem linePoly_eval (a b c : K) : (linePoly a b).eval c = a + c * b := by
  simp only [linePoly, eval_add, eval_mul, eval_C, eval_X]

theorem linePoly_natDegree (a b : K) : (linePoly a b).natDegree ≤ 1 := by
  unfold linePoly
  refine (natDegree_add_le _ _).trans (max_le (by simp) ?_)
  exact (natDegree_mul_le).trans (by simp)

theorem linePoly_ne_zero {a b : K} (nonzero : a ≠ 0 ∨ b ≠ 0) : linePoly a b ≠ 0 := by
  intro zero
  have zeroth := congrArg (fun p => p.coeff 0) zero
  have first := congrArg (fun p => p.coeff 1) zero
  simp [linePoly, coeff_C] at zeroth first
  rcases nonzero with h | h
  · exact h zeroth
  · exact h first

end Line

end CompressionLean.Lib
