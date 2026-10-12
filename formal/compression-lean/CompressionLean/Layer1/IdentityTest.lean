import CompressionLean.Layer1.Gkr

/-!
Owns the identity test of a table at a point whose coordinates are drawn one
by one: a table that is nonzero on its cube keeps a nonzero fold after each
coordinate, except for at most one bad value (`foldBad`). Its extension at
the whole point is then nonzero (`mle_ne_zero_of_path`).

The honest fold uses it for `α`; the early run point uses it to bind the
copied slot blocks to the setup column.
-/

namespace CompressionLean.Layer1

open Polynomial

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

/-- `d` is nonzero somewhere on the `m`-cube. -/
def NonzeroOn (m : ℕ) (d : ℕ → K) : Prop := ∃ i, i < 2 ^ m ∧ d i ≠ 0

open Classical in
/-- The value of the next coordinate that would zero the fold of `d` (on `m` variables). -/
noncomputable def foldBad (d : ℕ → K) (m : ℕ) (point : List K) : Finset K :=
  if h : ∃ i, i < 2 ^ (m - point.length - 1) ∧
      (fold point d (2 * i) ≠ 0 ∨ fold point d (2 * i + 1) ≠ 0) then
    zeros (linePoly (fold point d (2 * h.choose))
      (fold point d (2 * h.choose + 1) - fold point d (2 * h.choose)))
  else ∅

theorem foldBad_card (d : ℕ → K) (m : ℕ) (point : List K) : (foldBad d m point).card ≤ 1 := by
  unfold foldBad
  split
  · exact zeros_card_le_of_natDegree_le (linePoly_natDegree _ _)
  · simp

theorem foldBad_spec (d : ℕ → K) (m : ℕ) (point : List K)
    (h : ∃ i, i < 2 ^ (m - point.length - 1) ∧
      (fold point d (2 * i) ≠ 0 ∨ fold point d (2 * i + 1) ≠ 0)) :
    ∃ j, j < 2 ^ (m - point.length - 1) ∧
      (fold point d (2 * j) ≠ 0 ∨ fold point d (2 * j + 1) ≠ 0) ∧
      foldBad d m point = zeros (linePoly (fold point d (2 * j))
        (fold point d (2 * j + 1) - fold point d (2 * j))) := by
  classical
  exact ⟨h.choose, h.choose_spec.1, h.choose_spec.2, by simp only [foldBad, dif_pos h]⟩

/-- One coordinate off the bad value keeps the fold nonzero. -/
theorem nonzero_bindLow (d : ℕ → K) (m : ℕ) (point : List K) (short : point.length < m)
    (nonzero : NonzeroOn (m - point.length) (fold point d)) {c : K}
    (outside : c ∉ foldBad d m point) : NonzeroOn (m - (point ++ [c]).length) (fold (point ++ [c]) d) := by
  classical
  obtain ⟨i, lt, value⟩ := nonzero
  have size : 2 ^ (m - point.length) = 2 * 2 ^ (m - point.length - 1) := by
    rw [← pow_succ']
    congr 1
    omega
  have exists_pair : ∃ i, i < 2 ^ (m - point.length - 1) ∧
      (fold point d (2 * i) ≠ 0 ∨ fold point d (2 * i + 1) ≠ 0) := by
    refine ⟨i / 2, by rw [size] at lt; omega, ?_⟩
    rcases Nat.mod_two_eq_zero_or_one i with even | odd
    · left; rwa [show 2 * (i / 2) = i by omega]
    · right; rwa [show 2 * (i / 2) + 1 = i by omega]
  obtain ⟨j, j_lt, j_live, bad_eq⟩ := foldBad_spec d m point exists_pair
  have nonzero_line : linePoly (fold point d (2 * j)) (fold point d (2 * j + 1) - fold point d (2 * j)) ≠ 0 := by
    apply linePoly_ne_zero
    by_contra both
    push Not at both
    rcases j_live with h | h
    · exact h both.1
    · exact h (by rw [sub_eq_zero.mp both.2, both.1])
  refine ⟨j, by rw [List.length_append, List.length_singleton,
    show m - (point.length + 1) = m - point.length - 1 by omega]; exact j_lt, ?_⟩
  intro zero
  apply outside
  rw [bad_eq]
  unfold zeros
  rw [if_neg nonzero_line]
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, linePoly_eval]
  rw [fold_append_single] at zero
  simp only [bindLow] at zero
  rw [← zero]
  ring

/-- Along a point that avoided the bad values, a nonzero table has a nonzero extension. -/
theorem mle_ne_zero_of_path (d : ℕ → K) (m : ℕ) :
    ∀ (rest point : List K), point.length + rest.length = m →
      NonzeroOn (m - point.length) (fold point d) → Path (foldBad d m) point rest →
        fold (point ++ rest) d 0 ≠ 0 := by
  intro rest
  induction rest with
  | nil =>
      intro point length nonzero _
      obtain ⟨i, lt, value⟩ := nonzero
      simp only [List.length_nil, add_zero] at length
      rw [length, Nat.sub_self, pow_zero, Nat.lt_one_iff] at lt
      subst lt
      simpa using value
  | cons c rest ih =>
      intro point length nonzero path
      have next := nonzero_bindLow d m point (by simp at length; omega) nonzero path.1
      have := ih (point ++ [c]) (by simp at length ⊢; omega) next path.2
      simpa using this

theorem mle_ne_zero_of_path' (d : ℕ → K) (point : List K) (nonzero : NonzeroOn point.length d)
    (path : Path (foldBad d point.length) [] point) : mle point d ≠ 0 := by
  have := mle_ne_zero_of_path d point.length point [] (by simp) (by simpa [fold] using nonzero) path
  simpa [mle] using this

end CompressionLean.Layer1
