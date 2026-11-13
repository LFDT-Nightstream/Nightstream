import NightstreamFPrime.Spec.Compression.Game.Game
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-! The law of a game's outcome under a strategy. Each draw is uniform over its type (`uniform`)
and each message is the strategy's (`law`). The probability that `prob` gives an event is the
event's measure under the law (`prob_eq_law`). -/

namespace NightstreamFPrime.Spec.Compression.Game

open scoped BigOperators ENNReal

variable {α : Type}

/-- The uniform law on a nonempty finite type. -/
noncomputable def uniform (C : Type) [Fintype C] [Nonempty C] : PMF C :=
  PMF.ofFintype (fun _ => (Fintype.card C : ℝ≥0∞)⁻¹) (by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    exact ENNReal.mul_inv_cancel (by simp [Fintype.card_ne_zero]) (ENNReal.natCast_ne_top _))

/-- The law of a game's outcome: a draw is uniform and a message is the strategy's. -/
noncomputable def law (s : Strategy α) : Game α → PMF α
  | pure a => PMF.pure a
  | draw C _ next => (uniform C).bind fun c => law s (next c)
  | recv M next => law s (next (s M next))

theorem event_ne_top {β : Type} (p : PMF β) (event : Set β) : p.toOuterMeasure event ≠ ∞ := by
  rw [PMF.toOuterMeasure_apply]
  exact p.tsum_coe_indicator_ne_top event

/-- `prob` is the measure of the event under the game's law. -/
theorem prob_eq_law (s : Strategy α) (g : Game α) (E : α → Prop) :
    prob s g E = ((law s g).toOuterMeasure {a | E a}).toReal := by
  classical
  induction g with
  | pure a =>
      simp only [prob, law, PMF.toOuterMeasure_pure_apply, Set.mem_ofPred_eq]
      split_ifs <;> simp
  | draw C bad next ih =>
      have card : (Fintype.card C : ℝ≥0∞) ≠ 0 := by simp [Fintype.card_ne_zero]
      change 𝔼 c, prob s (next c) E = _
      simp only [law, PMF.toOuterMeasure_bind_apply, uniform, PMF.ofFintype_apply, tsum_fintype]
      rw [ENNReal.toReal_sum fun c _ =>
          ENNReal.mul_ne_top (ENNReal.inv_ne_top.mpr card) (event_ne_top _ _),
        Fintype.expect_eq_sum_div_card, div_eq_inv_mul, Finset.mul_sum]
      refine Finset.sum_congr rfl fun c _ => ?_
      rw [ih c, ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast]
  | recv M next ih =>
      simp only [prob, law]
      exact ih _

end NightstreamFPrime.Spec.Compression.Game
