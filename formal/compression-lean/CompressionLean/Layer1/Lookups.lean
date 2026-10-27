import CompressionLean.Lib.LogUp
import CompressionLean.Layer1.Gkr
import Mathlib.Tactic.LinearCombination

/-!
Owns the four logUp identities of the layer-1 verifier and what each one
forces off its bad set:

- `twoSided`: a left and a right fraction sum that agree at `β` agree pole by
  pole, unless `β` is a root of `twoSidedPoly` (degree below the term count).
- `lookup_sound` (row lookup, `(β, γ)` drawn together): every run tuple
  `(r, e₁, e₂)` is a table tuple `(i, t₁(i), t₂(i))`. A bad `γ` merges a run
  tuple with a table tuple (at most `2·M` values); a bad `β` is a root.
- `scatter_sound` (slot scatter, `(β, μ)` drawn together): each pair's
  numerators are the sums of its runs' numerators, both coordinates.
- The norm and the block scatter use `twoSided` directly.
-/

namespace CompressionLean.Layer1

open CompressionLean.Lib
open Polynomial

variable {K : Type} [Field K] [DecidableEq K]

section TwoSided

variable {ι κ : Type}

/-- Both sides as one sum: the right side with negated numerators. -/
noncomputable def twoSidedPoly (left : Finset ι) (a b : ι → K) (right : Finset κ) (c d : κ → K) : K[X] :=
  residuePoly (left.disjSum right) (Sum.elim a fun k => -c k) (Sum.elim b d)

theorem twoSidedPoly_natDegree (left : Finset ι) (a b : ι → K) (right : Finset κ) (c d : κ → K) :
    (twoSidedPoly left a b right c d).natDegree ≤ left.card + right.card := by
  refine (residuePoly_natDegree _ _ _).trans ?_
  rw [Finset.card_disjSum]

theorem residue_disjSum (left : Finset ι) (a b : ι → K) (right : Finset κ) (c d : κ → K) (v : K) :
    residue (left.disjSum right) (Sum.elim a fun k => -c k) (Sum.elim b d) v =
      residue left a b v - residue right c d v := by
  unfold residue
  rw [Finset.sum_filter, Finset.sum_disjSum, Finset.sum_filter, Finset.sum_filter]
  simp only [Sum.elim_inl, Sum.elim_inr]
  rw [sub_eq_add_neg, ← Finset.sum_neg_distrib]
  congr 1
  apply Finset.sum_congr rfl
  intro k _
  split <;> simp

/-- Agreement at `β` off the poles, with a pole whose residues differ, puts `β`
among the roots of `twoSidedPoly`. -/
theorem twoSided [Fintype K] (left : Finset ι) (a b : ι → K) (right : Finset κ) (c d : κ → K) {v : K}
    (live : residue left a b v ≠ residue right c d v) (β : K)
    (no_pole_left : ∀ i ∈ left, β ≠ b i) (no_pole_right : ∀ k ∈ right, β ≠ d k)
    (equal : ∑ i ∈ left, a i / (β - b i) = ∑ k ∈ right, c k / (β - d k)) :
    β ∈ zeros (twoSidedPoly left a b right c d) := by
  apply fraction_sum_bad (left.disjSum right) _ _ (v := v)
  · rw [residue_disjSum]; exact sub_ne_zero.mpr live
  · intro term member
    rcases term with i | k
    · exact no_pole_left i (Finset.inl_mem_disjSum.mp member)
    · exact no_pole_right k (Finset.inr_mem_disjSum.mp member)
  · rw [Finset.sum_disjSum]
    simp only [Sum.elim_inl, Sum.elim_inr, neg_div, Finset.sum_neg_distrib]
    rw [equal, add_neg_cancel]

end TwoSided

section Joint

variable [Fintype K]

open Classical in
/-- Bad pairs of a joint draw: a bad second coordinate, or a root in the first. -/
noncomputable def jointBad (Γ : Finset K) (poly : K → K[X]) : Finset (K × K) :=
  Finset.univ.filter fun pair => pair.2 ∈ Γ ∨ pair.1 ∈ zeros (poly pair.2)

theorem jointBad_card (Γ : Finset K) (poly : K → K[X]) (g d : ℕ) (small : Γ.card ≤ g)
    (roots : ∀ γ, (zeros (poly γ)).card ≤ d) :
    (jointBad Γ poly).card ≤ (g + d) * Fintype.card K := by
  classical
  have split : jointBad Γ poly ⊆
      (Finset.univ ×ˢ Γ) ∪ Finset.univ.biUnion (fun γ => zeros (poly γ) ×ˢ {γ}) := by
    intro pair member
    simp only [jointBad, Finset.mem_filter, Finset.mem_univ, true_and] at member
    rcases member with inside | root
    · exact Finset.mem_union_left _ (Finset.mem_product.mpr ⟨Finset.mem_univ _, inside⟩)
    · refine Finset.mem_union_right _ (Finset.mem_biUnion.mpr ⟨pair.2, Finset.mem_univ _, ?_⟩)
      exact Finset.mem_product.mpr ⟨root, Finset.mem_singleton_self _⟩
  refine (Finset.card_le_card split).trans ((Finset.card_union_le _ _).trans ?_)
  rw [Finset.card_product, Finset.card_univ, add_mul]
  refine add_le_add (by rw [mul_comm]; exact Nat.mul_le_mul_right _ small) ?_
  refine Finset.card_biUnion_le.trans ?_
  calc ∑ γ, (zeros (poly γ) ×ˢ ({γ} : Finset K)).card ≤ ∑ _γ : K, d := by
        apply Finset.sum_le_sum
        intro γ _
        rw [Finset.card_product, Finset.card_singleton, mul_one]
        exact roots γ
    _ = d * Fintype.card K := by rw [Finset.sum_const, Finset.card_univ, smul_eq_mul, mul_comm]

end Joint

section Checks

variable [Fintype K]

omit [Fintype K] in
/-- A pole's residue on one side, when a single term has that pole. -/
theorem residue_range_single (n : ℕ) (c : ℕ → K) (distinct : ∀ x y, x < n → y < n → (x : K) = y → x = y)
    {x : ℕ} (inside : x < n) : residue (Finset.range n) c (fun y => (y : K)) (x : K) = c x := by
  unfold residue
  rw [Finset.sum_eq_single x]
  · intro y member different
    simp only [Finset.mem_filter, Finset.mem_range] at member
    exact absurd (distinct y x member.1 inside member.2) different
  · intro outside
    simp only [Finset.mem_filter, Finset.mem_range, not_and] at outside
    exact absurd trivial (outside inside)

/-- The value of histogram slot `t`: `t - H`, as `norm::table_sum` reads it. -/
def slotValue (H t : ℕ) : K := (t : K) - (H : K)

/-- Norm: `Σ_x 1/(β - z(x)) = Σ_t m_t/(β - (t - H))` over the `2H + 1` slots. -/
noncomputable def normPoly (n H : ℕ) (z : ℕ → K) (hist : ℕ → K) : K[X] :=
  twoSidedPoly (Finset.range (2 ^ n)) (fun _ => 1) z (Finset.range (2 * H + 1)) hist (slotValue H)

omit [Fintype K] in
theorem normPoly_natDegree (n H : ℕ) (z : ℕ → K) (hist : ℕ → K) :
    (normPoly n H z hist).natDegree ≤ 2 ^ n + (2 * H + 1) := by
  refine (twoSidedPoly_natDegree _ _ _ _ _ _).trans ?_
  simp

/-- Off its bad set, the norm identity puts every cell in `[-H, H]`. -/
theorem norm_sound (n H : ℕ) (z : ℕ → K) (hist : ℕ → K)
    (count : ∀ c : ℕ, 0 < c → c ≤ 2 ^ n → (c : K) ≠ 0) (β : K) (outside : β ∉ zeros (normPoly n H z hist))
    (no_pole_left : ∀ x, x < 2 ^ n → β ≠ z x) (no_pole_right : ∀ t, t < 2 * H + 1 → β ≠ slotValue H t)
    (equal : ∑ x ∈ Finset.range (2 ^ n), 1 / (β - z x) =
      ∑ t ∈ Finset.range (2 * H + 1), hist t / (β - slotValue H t)) :
    ∀ x, x < 2 ^ n → ∃ t, t < 2 * H + 1 ∧ z x = slotValue H t := by
  classical
  by_contra none
  push Not at none
  obtain ⟨x, inside, missing⟩ := none
  apply outside
  apply twoSided _ _ _ _ _ _ (v := z x) _ β (fun y m => no_pole_left y (Finset.mem_range.mp m))
    (fun t m => no_pole_right t (Finset.mem_range.mp m)) equal
  have right_zero : residue (Finset.range (2 * H + 1)) hist (slotValue H) (z x) = 0 := by
    unfold residue
    apply Finset.sum_eq_zero
    intro t member
    obtain ⟨inside_t, same⟩ := Finset.mem_filter.mp member
    exact absurd same.symm (missing t (Finset.mem_range.mp inside_t))
  rw [right_zero]
  unfold residue
  simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
  apply count
  · exact Finset.card_pos.mpr ⟨x, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr inside, rfl⟩⟩
  · exact (Finset.card_filter_le _ _).trans (by simp)

/-- `r + γ·(e₁ + γ·e₂)`, the tuple compression of `matrix.rs`. -/
def tuple (γ r e₁ e₂ : K) : K := r + γ * (e₁ + γ * e₂)

noncomputable def tuplePoly (r e₁ e₂ : K) : K[X] := C r + X * (C e₁ + X * C e₂)

omit [DecidableEq K] [Fintype K] in
theorem tuplePoly_eval (r e₁ e₂ γ : K) : (tuplePoly r e₁ e₂).eval γ = tuple γ r e₁ e₂ := by
  simp only [tuplePoly, tuple, eval_add, eval_mul, eval_C, eval_X]

omit [DecidableEq K] [Fintype K] in
theorem tuplePoly_natDegree (r e₁ e₂ : K) : (tuplePoly r e₁ e₂).natDegree ≤ 2 := by
  unfold tuplePoly
  refine (natDegree_add_le _ _).trans (max_le (by simp) ?_)
  refine (natDegree_mul_le).trans ?_
  have inner : (C e₁ + X * C e₂).natDegree ≤ 1 :=
    (natDegree_add_le _ _).trans (max_le (by simp) ((natDegree_mul_le).trans (by simp)))
  have := natDegree_X_le (R := K)
  omega

omit [DecidableEq K] [Fintype K] in
theorem tuplePoly_injective {r e₁ e₂ r' e₁' e₂' : K}
    (equal : tuplePoly r e₁ e₂ = tuplePoly r' e₁' e₂') : r = r' ∧ e₁ = e₁' ∧ e₂ = e₂' := by
  have c0 := congrArg (fun p => p.coeff 0) equal
  have c1 := congrArg (fun p => p.coeff 1) equal
  have c2 := congrArg (fun p => p.coeff 2) equal
  simp [tuplePoly, coeff_X_mul, coeff_C] at c0 c1 c2
  exact ⟨c0, c1, c2⟩

/-- The row lookup: runs `(r, e₁, e₂)`, `k < N`; the table `(i, t₁ i, t₂ i)`, `i < M`. -/
structure Lookup (K : Type) where
  runs : ℕ
  rows : ℕ
  run : ℕ → K × K × K
  table : ℕ → K × K
  mult : ℕ → K

namespace Lookup

variable (d : Lookup K)

/-- Run `k`'s tuple is the table tuple of some row. -/
def InTable (k : ℕ) : Prop :=
  ∃ i, i < d.rows ∧ d.run k = ((i : K), (d.table i).1, (d.table i).2)

def runTuple (γ : K) (k : ℕ) : K := tuple γ (d.run k).1 (d.run k).2.1 (d.run k).2.2

def tableTuple (γ : K) (i : ℕ) : K := tuple γ (i : K) (d.table i).1 (d.table i).2

open Classical in
/-- The `γ` that merge the first missing run tuple with a table tuple. -/
noncomputable def gammaBad : Finset K :=
  if h : ∃ k, k < d.runs ∧ ¬ d.InTable k then
    (Finset.range d.rows).biUnion fun i =>
      zeros (tuplePoly (d.run h.choose).1 (d.run h.choose).2.1 (d.run h.choose).2.2 -
        tuplePoly (i : K) (d.table i).1 (d.table i).2)
  else ∅

noncomputable def poly (γ : K) : K[X] :=
  twoSidedPoly (Finset.range d.runs) (fun _ => 1) (d.runTuple γ) (Finset.range d.rows) d.mult (d.tableTuple γ)

/-- The bad pairs `(β, γ)` of the row lookup. -/
noncomputable def bad : Finset (K × K) := jointBad d.gammaBad d.poly

theorem bad_card : d.bad.card ≤ (2 * d.rows + (d.runs + d.rows)) * Fintype.card K := by
  apply jointBad_card
  · unfold gammaBad
    split
    · refine (card_biUnion_le_mul _ _ 2 fun i _ => zeros_card_le_of_natDegree_le ?_).trans (by simp [mul_comm])
      exact (natDegree_sub_le _ _).trans (max_le (tuplePoly_natDegree _ _ _) (tuplePoly_natDegree _ _ _))
    · simp
  · intro γ
    refine zeros_card_le_of_natDegree_le ((twoSidedPoly_natDegree _ _ _ _ _ _).trans ?_)
    simp

/-- Off its bad pairs, the row lookup puts every run tuple in the table. -/
theorem sound (count : ∀ c : ℕ, 0 < c → c ≤ d.runs → (c : K) ≠ 0) (β γ : K) (outside : (β, γ) ∉ d.bad)
    (no_pole_left : ∀ k, k < d.runs → β ≠ d.runTuple γ k)
    (no_pole_right : ∀ i, i < d.rows → β ≠ d.tableTuple γ i)
    (equal : ∑ k ∈ Finset.range d.runs, 1 / (β - d.runTuple γ k) =
      ∑ i ∈ Finset.range d.rows, d.mult i / (β - d.tableTuple γ i)) :
    ∀ k, k < d.runs → d.InTable k := by
  classical
  by_contra none
  push Not at none
  have h : ∃ k, k < d.runs ∧ ¬ d.InTable k := none
  set k := h.choose with k_def
  have k_lt := h.choose_spec.1
  have k_missing := h.choose_spec.2
  simp only [bad, jointBad, Finset.mem_filter, Finset.mem_univ, true_and, not_or] at outside
  obtain ⟨gamma_good, beta_good⟩ := outside
  have apart : ∀ i, i < d.rows → d.runTuple γ k ≠ d.tableTuple γ i := by
    intro i inside equal_at
    apply gamma_good
    simp only [gammaBad, dif_pos h]
    refine Finset.mem_biUnion.mpr ⟨i, Finset.mem_range.mpr inside, ?_⟩
    have different : tuplePoly (d.run k).1 (d.run k).2.1 (d.run k).2.2 ≠
        tuplePoly (i : K) (d.table i).1 (d.table i).2 := by
      intro same
      obtain ⟨r, e₁, e₂⟩ := tuplePoly_injective same
      exact k_missing ⟨i, inside, Prod.ext r (Prod.ext e₁ e₂)⟩
    unfold zeros
    rw [if_neg (sub_ne_zero.mpr different)]
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, eval_sub, tuplePoly_eval]
    exact sub_eq_zero.mpr equal_at
  apply beta_good
  apply twoSided _ _ _ _ _ _ (v := d.runTuple γ k) _ β
    (fun j m => no_pole_left j (Finset.mem_range.mp m))
    (fun i m => no_pole_right i (Finset.mem_range.mp m)) equal
  have right_zero : residue (Finset.range d.rows) d.mult (d.tableTuple γ) (d.runTuple γ k) = 0 := by
    unfold residue
    apply Finset.sum_eq_zero
    intro i member
    obtain ⟨inside, same⟩ := Finset.mem_filter.mp member
    exact absurd same.symm (apart i (Finset.mem_range.mp inside))
  rw [right_zero]
  unfold residue
  simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
  apply count
  · exact Finset.card_pos.mpr ⟨k, Finset.mem_filter.mpr ⟨Finset.mem_range.mpr k_lt, rfl⟩⟩
  · exact (Finset.card_filter_le _ _).trans (by simp)

end Lookup

/-- The slot scatter: runs `k < N` with `coef k·(e₁ + μ·e₂)` at pole `tag k`;
pairs `x < P` with `U₁ x + μ·U₂ x` at pole `x`. -/
structure Scatter (K : Type) where
  runs : ℕ
  pairs : ℕ
  coef : ℕ → K
  e : ℕ → K × K
  tag : ℕ → K
  U : ℕ → K × K

namespace Scatter

variable (d : Scatter K)

/-- The runs' numerators summed at pair `x`. -/
def sums (x : ℕ) : K × K :=
  (∑ k ∈ (Finset.range d.runs).filter (fun k => d.tag k = (x : K)), d.coef k * (d.e k).1,
    ∑ k ∈ (Finset.range d.runs).filter (fun k => d.tag k = (x : K)), d.coef k * (d.e k).2)

open Classical in
/-- The `μ` that cancel the first wrong pair's two coordinates. -/
noncomputable def muBad : Finset K :=
  if h : ∃ x, x < d.pairs ∧ d.U x ≠ d.sums x then
    zeros (linePoly ((d.sums h.choose).1 - (d.U h.choose).1) ((d.sums h.choose).2 - (d.U h.choose).2))
  else ∅

noncomputable def poly (μ : K) : K[X] :=
  twoSidedPoly (Finset.range d.runs) (fun k => d.coef k * ((d.e k).1 + μ * (d.e k).2)) d.tag
    (Finset.range d.pairs) (fun x => (d.U x).1 + μ * (d.U x).2) (fun x => (x : K))

/-- The bad pairs `(β, μ)` of the slot scatter. -/
noncomputable def bad : Finset (K × K) := jointBad d.muBad d.poly

theorem bad_card : d.bad.card ≤ (1 + (d.runs + d.pairs)) * Fintype.card K := by
  apply jointBad_card
  · unfold muBad
    split
    · exact zeros_card_le_of_natDegree_le (linePoly_natDegree _ _)
    · simp
  · intro μ
    refine zeros_card_le_of_natDegree_le ((twoSidedPoly_natDegree _ _ _ _ _ _).trans ?_)
    simp

/-- Off its bad pairs, the slot scatter makes every pair the sum of its runs. -/
theorem sound (distinct : ∀ x y, x < d.pairs → y < d.pairs → (x : K) = y → x = y)
    (β μ : K) (outside : (β, μ) ∉ d.bad)
    (no_pole_left : ∀ k, k < d.runs → β ≠ d.tag k)
    (no_pole_right : ∀ x, x < d.pairs → β ≠ (x : K))
    (equal : ∑ k ∈ Finset.range d.runs, d.coef k * ((d.e k).1 + μ * (d.e k).2) / (β - d.tag k) =
      ∑ x ∈ Finset.range d.pairs, ((d.U x).1 + μ * (d.U x).2) / (β - (x : K))) :
    ∀ x, x < d.pairs → d.U x = d.sums x := by
  classical
  by_contra none
  push Not at none
  have h : ∃ x, x < d.pairs ∧ d.U x ≠ d.sums x := none
  set x := h.choose with x_def
  have x_lt := h.choose_spec.1
  have x_wrong := h.choose_spec.2
  simp only [bad, jointBad, Finset.mem_filter, Finset.mem_univ, true_and, not_or] at outside
  obtain ⟨mu_good, beta_good⟩ := outside
  have live : (d.sums x).1 - (d.U x).1 + μ * ((d.sums x).2 - (d.U x).2) ≠ 0 := by
    have nonzero : linePoly ((d.sums x).1 - (d.U x).1) ((d.sums x).2 - (d.U x).2) ≠ 0 := by
      apply linePoly_ne_zero
      by_contra both
      push Not at both
      exact x_wrong (Prod.ext (sub_eq_zero.mp both.1).symm (sub_eq_zero.mp both.2).symm)
    intro zero
    apply mu_good
    simp only [muBad, dif_pos h]
    unfold zeros
    rw [if_neg nonzero]
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, linePoly_eval]
    exact zero
  apply beta_good
  apply twoSided _ _ _ _ _ _ (v := (x : K)) _ β
    (fun k m => no_pole_left k (Finset.mem_range.mp m))
    (fun y m => no_pole_right y (Finset.mem_range.mp m)) equal
  rw [residue_range_single _ _ distinct x_lt]
  have residue_eq : residue (Finset.range d.runs) (fun k => d.coef k * ((d.e k).1 + μ * (d.e k).2))
      d.tag (x : K) = (d.sums x).1 + μ * (d.sums x).2 := by
    unfold residue sums
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro k _
    ring
  rw [residue_eq]
  intro same
  apply live
  linear_combination same

end Scatter

end Checks

end CompressionLean.Layer1
