import NightstreamFPrime.Spec.Compression.Lib.SumCheck
import NightstreamFPrime.Spec.Compression.Lib.Density
import NightstreamFPrime.Spec.Compression.Game.Restoration

/-! A sum-check as a game (Rust `sumcheck::verify`). Each round receives the coefficients of a
polynomial of degree at most `d`, rejects unless its values at 0 and 1 add to the claim, and draws
one coordinate. The bad set of a draw is the set of roots of the message minus the true round
polynomial, so it has at most `d` elements. On a path that avoids the bad sets, a false claim ends
in a false final claim (`sumcheck_avoids`); each draw has density at most `d / |K|`
(`sumcheck_densityLe`). -/

namespace NightstreamFPrime.Spec.Compression.Game

open NightstreamFPrime.Spec.Compression.Lib Polynomial

variable {K : Type} [Field K] [Fintype K] [DecidableEq K]

/-- The verifier of `rounds` rounds against the true round polynomials `round`. It ends in `none`
when a round check fails, and otherwise in the point and the final claim. -/
noncomputable def sumcheck (d : ℕ) (round : List K → K[X]) :
    ℕ → List K → K → Game (Option (List K × K))
  | 0, point, claim => .pure (some (point, claim))
  | rounds + 1, point, claim =>
      .recv (Fin (d + 1) → K) fun message =>
        if (coeffPoly message).eval 0 + (coeffPoly message).eval 1 = claim then
          .draw K (zeros (coeffPoly message - round point)) fun c =>
            sumcheck d round rounds (point ++ [c]) ((coeffPoly message).eval c)
        else .pure none

/-- Each accepted end extends the start point by `rounds` coordinates. -/
theorem sumcheck_ends (d : ℕ) (round : List K → K[X]) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), (sumcheck d round rounds point claim).Ends
      fun result => ∀ final value, result = some (final, value) →
        ∃ cs : List K, final = point ++ cs ∧ cs.length = rounds := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim final value equal
      simp only [Option.some.injEq, Prod.mk.injEq] at equal
      exact ⟨[], by simp [equal.1], rfl⟩
  | succ rounds ih =>
      intro point claim message
      dsimp only
      split
      · intro c
        refine Ends.mono (ih (point ++ [c]) _) ?_
        intro result ends final value equal
        obtain ⟨cs, rfl, length⟩ := ends final value equal
        exact ⟨c :: cs, by simp, by simp [length]⟩
      · intro final value equal
        simp at equal

/-- On a path that avoids the bad sets, an accepted false claim ends in a false final claim. -/
theorem sumcheck_avoids (d bound : ℕ) (truth : List K → K) (round : List K → K[X])
    (true_rounds : RoundTruth d bound truth round) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (sumcheck d round rounds point claim).Avoids fun result =>
        ∀ final value, result = some (final, value) → claim ≠ truth point →
          value ≠ truth final := by
  intro rounds
  induction rounds with
  | zero =>
      intro point claim _ final value equal false_claim
      simp only [Option.some.injEq, Prod.mk.injEq] at equal
      rw [← equal.1, ← equal.2]
      exact false_claim
  | succ rounds ih =>
      intro point claim fits message
      have inside : point.length < bound := by omega
      dsimp only
      split
      · rename_i checked
        intro c outside
        refine (ih (point ++ [c]) _ (by simp; omega)).mono ?_
        intro result transfer final value equal false_claim
        apply transfer final value equal
        have differ : coeffPoly message ≠ round point := by
          intro same
          apply false_claim
          rw [← checked, same, true_rounds.halves point inside]
        rw [← true_rounds.binds point c inside]
        exact eval_ne_of_not_mem_zeros differ outside
      · intro final value equal
        simp at equal

/-- Each draw of a sum-check of degree `d` has density at most `d / |K|`. -/
theorem sumcheck_densityLe (d bound : ℕ) (round : List K → K[X])
    (degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (sumcheck d round rounds point claim).DensityLe (d / Fintype.card K) := by
  intro rounds
  induction rounds with
  | zero => intro point claim _; trivial
  | succ rounds ih =>
      intro point claim fits message
      dsimp only
      split
      · refine ⟨density_zeros_le ((natDegree_sub_le _ _).trans
          (max_le (coeffPoly_natDegree message) (degree point (by omega)))), fun c => ?_⟩
        exact ih (point ++ [c]) _ (by simp; omega)
      · trivial

/-- The interactive bound: a false claim passes `rounds` rounds of degree `d` with probability at
most `rounds · d / |K|`. -/
theorem sumcheck_costLe (d bound : ℕ) (round : List K → K[X])
    (degree : ∀ point : List K, point.length < bound → (round point).natDegree ≤ d) :
    ∀ (rounds : ℕ) (point : List K) (claim : K), point.length + rounds ≤ bound →
      (sumcheck d round rounds point claim).CostLe (rounds * (d / Fintype.card K)) := by
  intro rounds
  induction rounds with
  | zero => intro point claim _; show (0 : ℝ) ≤ _; simp
  | succ rounds ih =>
      intro point claim fits message
      dsimp only
      split
      · refine ⟨rounds * (d / Fintype.card K), ?_, fun c => ih (point ++ [c]) _ (by simp; omega)⟩
        have := density_zeros_le ((natDegree_sub_le _ _).trans
          (max_le (coeffPoly_natDegree message) (degree point (by omega))))
        unfold density at this
        push_cast
        linarith
      · show (0 : ℝ) ≤ _
        positivity

end NightstreamFPrime.Spec.Compression.Game
