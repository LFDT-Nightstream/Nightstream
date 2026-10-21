import Mathlib.Algebra.Polynomial.Roots

/-!
Owns the bad sets of polynomial identity tests: the challenges at which a
nonzero polynomial vanishes, and their unions over the WHIR candidates.
A bad set has at most degree elements, so its density is `degree / |K|`.
-/

namespace CompressionLean.Layer1

open Polynomial

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

/-- The challenges at which `p` vanishes; empty for the zero polynomial. -/
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

/-- Off the bad set, distinct polynomials take distinct values. -/
theorem eval_ne_of_not_mem_zeros {p q : K[X]} (different : p ≠ q) {c : K}
    (outside : c ∉ zeros (p - q)) : p.eval c ≠ q.eval c := by
  intro equal
  apply outside
  unfold zeros
  rw [if_neg (sub_ne_zero.mpr different)]
  simp [equal]

theorem zeros_card_le_of_natDegree_le {p : K[X]} {d : ℕ} (degree : p.natDegree ≤ d) :
    (zeros p).card ≤ d :=
  (zeros_card_le p).trans degree

omit [Field K] [Fintype K] in
/-- A union of bad sets, one per candidate. -/
theorem card_biUnion_le_mul {W : Type} (worlds : Finset W) (bad : W → Finset K) (d : ℕ)
    (each : ∀ w ∈ worlds, (bad w).card ≤ d) :
    (worlds.biUnion bad).card ≤ worlds.card * d := by
  calc (worlds.biUnion bad).card ≤ ∑ w ∈ worlds, (bad w).card := Finset.card_biUnion_le
    _ ≤ ∑ _w ∈ worlds, d := Finset.sum_le_sum each
    _ = worlds.card * d := by simp

end CompressionLean.Layer1
