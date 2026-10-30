import CompressionLean.Lib.SumCheck
import CompressionLean.Game.Game

namespace CompressionLean.Game

open CompressionLean.Lib Polynomial

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

noncomputable def drawPoint (bad : List K → Finset K) : ℕ → List K → Game (List K)
  | 0, point => .pure point
  | m + 1, point => .draw K (bad point) fun c => drawPoint bad m (point ++ [c])

omit [DecidableEq K] in
theorem drawPoint_avoids (bad : List K → Finset K) :
    ∀ (m : ℕ) (point : List K), (drawPoint bad m point).Avoids fun r =>
      ∃ cs, r = point ++ cs ∧ cs.length = m ∧ Path bad point cs := by
  intro m
  induction m with
  | zero => intro point; exact ⟨[], by simp, rfl, trivial⟩
  | succ m ih =>
      intro point c outside
      refine (ih (point ++ [c])).mono ?_
      rintro r ⟨cs, rfl, length, path⟩
      exact ⟨c :: cs, by simp, by simp [length], outside, path⟩

omit [DecidableEq K] in
theorem drawPoint_cost (bad : List K → Finset K) (E : ℕ) (card : ∀ point, (bad point).card ≤ E) :
    ∀ (m : ℕ) (point : List K), (drawPoint bad m point).CostLe (m * ((E : ℝ) / Fintype.card K)) := by
  intro m
  induction m with
  | zero => intro point; simp [drawPoint, Game.CostLe]
  | succ m ih =>
      intro point
      refine ⟨m * ((E : ℝ) / Fintype.card K), ?_, fun c => ih _⟩
      have positive : (0 : ℝ) < Fintype.card K := by exact_mod_cast Fintype.card_pos
      have : ((bad point).card : ℝ) / Fintype.card K ≤ (E : ℝ) / Fintype.card K :=
        div_le_div_of_nonneg_right (by exact_mod_cast card point) positive.le
      push_cast
      linarith

variable {W : Type}

noncomputable def sumcheck (worlds : Finset W) (d : ℕ) (round : W → List K → K[X])
    (extra : List K → Finset K) : ℕ → List K → K → Game (List K × K)
  | 0, point, claim => .pure (point, claim)
  | rounds + 1, point, claim =>
      .recv (Fin (d + 1) → K) fun message =>
        .draw K (worlds.biUnion (fun w => zeros (roundPoly d claim message - round w point)) ∪
            extra point) fun c =>
          sumcheck worlds d round extra rounds (point ++ [c]) ((roundPoly d claim message).eval c)

theorem sumcheck_avoids (worlds : Finset W) (d : ℕ) (distinct : NodesDistinct K d)
    (truth : W → List K → K) (round : W → List K → K[X]) (extra : List K → Finset K) (bound : ℕ)
    (true_rounds : ∀ w ∈ worlds, RoundTruth d bound (truth w) (round w)) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (sumcheck worlds d round extra rounds point claim).Avoids fun result =>
        (∃ challenges, result.1 = point ++ challenges ∧ challenges.length = rounds ∧
          Path extra point challenges) ∧
        ∀ w ∈ worlds, claim ≠ truth w point → result.2 ≠ truth w result.1 := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _
      exact ⟨⟨[], by simp, rfl, trivial⟩, fun w _ false_claim => false_claim⟩
  | succ rounds ih =>
      intro point claim fits message c outside
      have inside : point.length < bound := by omega
      have rest := ih (point ++ [c]) ((roundPoly d claim message).eval c) (by simp; omega)
      refine rest.mono ?_
      rintro ⟨final, value⟩ ⟨⟨challenges, shape, length, path⟩, transfer⟩
      refine ⟨⟨c :: challenges, by simpa using shape, by simp [length], ?_, path⟩, ?_⟩
      · exact fun hit => outside (Finset.mem_union_right _ hit)
      · intro w member false_claim
        apply transfer w member
        have truths := true_rounds w member
        have differ : roundPoly d claim message ≠ round w point := by
          intro equal
          apply false_claim
          rw [← roundPoly_sum distinct claim message, equal, truths.halves point inside]
        rw [← truths.binds point c inside]
        exact eval_ne_of_not_mem_zeros differ
          (fun hit => outside (Finset.mem_union_left _ (Finset.mem_biUnion.mpr ⟨w, member, hit⟩)))

theorem sumcheck_cost (worlds : Finset W) (d : ℕ) (distinct : NodesDistinct K d)
    (round : W → List K → K[X]) (extra : List K → Finset K) (bound extraBound : ℕ)
    (degrees : ∀ w ∈ worlds, ∀ point : List K, point.length < bound → (round w point).natDegree ≤ d + 1)
    (extra_card : ∀ point : List K, (extra point).card ≤ extraBound) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (sumcheck worlds d round extra rounds point claim).CostLe
        (rounds * ((worlds.card * (d + 1) + extraBound : ℕ) / (Fintype.card K : ℝ))) := by
  intro rounds
  induction rounds with
  | zero => intro point claim _; simp [sumcheck, Game.CostLe]
  | succ rounds ih =>
      intro point claim fits message
      refine ⟨rounds * ((worlds.card * (d + 1) + extraBound : ℕ) / (Fintype.card K : ℝ)), ?_,
        fun c => ih _ _ (by simp; omega)⟩
      have card : (worlds.biUnion (fun w => zeros (roundPoly d claim message - round w point)) ∪
          extra point).card ≤ worlds.card * (d + 1) + extraBound := by
        refine (Finset.card_union_le _ _).trans (add_le_add ?_ (extra_card point))
        refine card_biUnion_le_mul worlds _ (d + 1) fun w member => zeros_card_le_of_natDegree_le ?_
        exact (natDegree_sub_le _ _).trans
          (max_le (roundPoly_natDegree distinct claim message) (degrees w member point (by omega)))
      have positive : (0 : ℝ) < Fintype.card K := by exact_mod_cast Fintype.card_pos
      have step : ∀ k : ℕ, k ≤ worlds.card * (d + 1) + extraBound →
          (k : ℝ) / Fintype.card K +
              rounds * ((worlds.card * (d + 1) + extraBound : ℕ) / (Fintype.card K : ℝ)) ≤
            ((rounds + 1 : ℕ) : ℝ) *
              ((worlds.card * (d + 1) + extraBound : ℕ) / (Fintype.card K : ℝ)) := by
        intro k le
        have scaled : (k : ℝ) / Fintype.card K ≤
            ((worlds.card * (d + 1) + extraBound : ℕ) : ℝ) / Fintype.card K :=
          div_le_div_of_nonneg_right (by exact_mod_cast le) positive.le
        push_cast at scaled ⊢
        linarith
      exact step _ card

end CompressionLean.Game
