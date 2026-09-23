import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Law
import NightstreamFPrime.Spec.Folding.PiRLC.CoordinateTerminalLaw

/-!
Owns the transfer of the interactive PiRLC extraction bound to challenges
drawn by the whole-vector sampler.

Experiment: each coordinate of the challenge vector is `sample` applied to its
own four independent uniform field values.
Per-call bound: `distance < 2^-132` from `Law`.
Cumulative bound: for `n` challenge coordinates, every acceptance test with
values in `[0, 1]` changes by at most `n * distance`. One PiRLC fold has
`n = K + k`; over `T` folds the loss is at most `T * n * distance`, with `T`
kept as a parameter.

The existing extractor keeps uniform challenges; its loss `n / |C|` stays a
separate term. The sampler has no failure event. No law is assigned to
Poseidon2.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

open Finset
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionStrongSet

/-! ### Uniform averages over finite types -/

/-- Uniform average over a finite type. -/
noncomputable def average {γ : Type*} [Fintype γ] (f : γ → ℝ) : ℝ :=
  (∑ x, f x) / Fintype.card γ

section Average

variable {γ δ : Type*} [Fintype γ] [Fintype δ]

theorem average_comp_equiv (e : γ ≃ δ) (f : δ → ℝ) :
    average (fun x => f (e x)) = average f := by
  unfold average
  rw [Equiv.sum_comp e f, Fintype.card_congr e]

theorem abs_average_sub_le [Nonempty γ] (f h : γ → ℝ) (bound : ℝ)
    (pointwise : ∀ x, |f x - h x| ≤ bound) : |average f - average h| ≤ bound := by
  unfold average
  have positive : (0 : ℝ) < Fintype.card γ := by exact_mod_cast Fintype.card_pos
  rw [← sub_div, ← sum_sub_distrib, abs_div, abs_of_pos positive, div_le_iff₀ positive]
  calc |∑ x, (f x - h x)| ≤ ∑ x, |f x - h x| := abs_sum_le_sum_abs _ _
    _ ≤ ∑ _x : γ, bound := sum_le_sum fun x _ => pointwise x
    _ = bound * Fintype.card γ := by rw [sum_const, card_univ, nsmul_eq_mul, mul_comm]

theorem average_nonnegative (f : γ → ℝ) (nonnegative : ∀ x, 0 ≤ f x) : 0 ≤ average f := by
  unfold average
  exact div_nonneg (sum_nonneg fun x _ => nonnegative x) (by positivity)

theorem average_le_one [Nonempty γ] (f : γ → ℝ) (atMostOne : ∀ x, f x ≤ 1) :
    average f ≤ 1 := by
  unfold average
  have positive : (0 : ℝ) < Fintype.card γ := by exact_mod_cast Fintype.card_pos
  rw [div_le_one positive]
  calc ∑ x, f x ≤ ∑ _x : γ, (1 : ℝ) := sum_le_sum fun x _ => atMostOne x
    _ = Fintype.card γ := by simp

/-- Averaging over a tuple is averaging over its first entry, then the rest. -/
theorem average_cons {α : Type*} [Fintype α] {n : ℕ} (f : (Fin (n + 1) → α) → ℝ) :
    average f = average (fun a : α => average (fun rest : Fin n → α => f (Fin.cons a rest))) := by
  rw [← average_comp_equiv (Fin.consEquiv (fun _ : Fin (n + 1) => α)) f]
  unfold average
  rw [Fintype.sum_prod_type, Fintype.card_prod, ← sum_div, div_div, Nat.cast_mul, mul_comm]
  rfl

end Average

/-! ### Hybrid argument -/

/-- If one sampled coordinate changes every `[0, 1]` test by at most `bound`,
then `n` independent sampled coordinates change it by at most `n * bound`. -/
theorem hybrid {α β : Type*} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    (g : α → β) (bound : ℝ)
    (single : ∀ test : β → ℝ, (∀ b, 0 ≤ test b) → (∀ b, test b ≤ 1) →
      |average (fun a => test (g a)) - average test| ≤ bound) :
    ∀ (n : ℕ) (test : (Fin n → β) → ℝ), (∀ c, 0 ≤ test c) → (∀ c, test c ≤ 1) →
      |average (fun v : Fin n → α => test (g ∘ v)) - average test| ≤ n * bound
  | 0, test, _, _ => by
      have sampled : ∀ v : Fin 0 → α, test (g ∘ v) = test Fin.elim0 :=
        fun v => congrArg test (Subsingleton.elim _ _)
      have uniform : ∀ w : Fin 0 → β, test w = test Fin.elim0 :=
        fun w => congrArg test (Subsingleton.elim _ _)
      unfold average
      simp only [sampled, uniform, sum_const, card_univ, nsmul_eq_mul]
      simp
  | n + 1, test, nonnegative, atMostOne => by
      let rest : β → ℝ := fun b => average (fun w : Fin n → β => test (Fin.cons b w))
      have sampled : average (fun v : Fin (n + 1) → α => test (g ∘ v)) =
          average (fun a => average (fun tail : Fin n → α => test (Fin.cons (g a) (g ∘ tail)))) := by
        rw [average_cons]
        simp only [Fin.comp_cons]
      have inner : ∀ a,
          |average (fun tail : Fin n → α => test (Fin.cons (g a) (g ∘ tail))) - rest (g a)| ≤
            n * bound :=
        fun a => hybrid g bound single n (fun w => test (Fin.cons (g a) w))
          (fun _ => nonnegative _) (fun _ => atMostOne _)
      have outer : |average (fun a => rest (g a)) - average rest| ≤ bound :=
        single rest (fun _ => average_nonnegative _ fun _ => nonnegative _)
          (fun _ => average_le_one _ fun _ => atMostOne _)
      rw [sampled, average_cons test]
      calc |average (fun a => average (fun tail : Fin n → α => test (Fin.cons (g a) (g ∘ tail)))) -
            average rest|
          ≤ |average (fun a => average (fun tail : Fin n → α => test (Fin.cons (g a) (g ∘ tail)))) -
              average (fun a => rest (g a))| + |average (fun a => rest (g a)) - average rest| :=
            abs_sub_le _ _ _
        _ ≤ n * bound + bound := add_le_add (abs_average_sub_le _ _ _ inner) outer
        _ = ((n + 1 : ℕ) : ℝ) * bound := by push_cast; ring

/-- The hybrid bound for any finite index type. -/
theorem hybrid_index {α β Index : Type*} [Fintype α] [Fintype β] [Nonempty α] [Nonempty β]
    [Fintype Index] [DecidableEq Index] (g : α → β) (bound : ℝ)
    (single : ∀ test : β → ℝ, (∀ b, 0 ≤ test b) → (∀ b, test b ≤ 1) →
      |average (fun a => test (g a)) - average test| ≤ bound)
    (test : (Index → β) → ℝ) (nonnegative : ∀ c, 0 ≤ test c) (atMostOne : ∀ c, test c ≤ 1) :
    |average (fun v : Index → α => test (g ∘ v)) - average test| ≤
      Fintype.card Index * bound := by
  let e := Fintype.equivFin Index
  let reindexSampled : (Fin (Fintype.card Index) → α) ≃ (Index → α) :=
    Equiv.arrowCongr e.symm (Equiv.refl α)
  let reindexUniform : (Fin (Fintype.card Index) → β) ≃ (Index → β) :=
    Equiv.arrowCongr e.symm (Equiv.refl β)
  have sampled : average (fun v : Index → α => test (g ∘ v)) =
      average (fun w : Fin (Fintype.card Index) → α => test (g ∘ (reindexSampled w))) :=
    (average_comp_equiv reindexSampled (fun v => test (g ∘ v))).symm
  have uniform : average test =
      average (fun w : Fin (Fintype.card Index) → β => test (reindexUniform w)) :=
    (average_comp_equiv reindexUniform test).symm
  have commute : ∀ w : Fin (Fintype.card Index) → α,
      g ∘ reindexSampled w = reindexUniform (g ∘ w) :=
    fun _ => rfl
  rw [sampled, uniform]
  simp only [commute]
  exact hybrid g bound single (Fintype.card Index) (fun w => test (reindexUniform w))
    (fun _ => nonnegative _) (fun _ => atMostOne _)

/-! ### The sampler instance -/

instance : Nonempty Scalar := ⟨fun _ => ⟨0, by decide⟩⟩

private theorem average_sample (test : Scalar → ℝ) :
    average (fun draw : Draw => test (sample draw)) = sampledExpect test := by
  unfold average sampledExpect
  rw [Fintype.card_congr drawIndex, Fintype.card_fin]

private theorem average_scalar (test : Scalar → ℝ) : average test = uniformExpect test := by
  unfold average uniformExpect
  rw [Fintype.card_congr scalarIndex, Fintype.card_fin]

/-- V6, cumulative form: for `n` challenge coordinates drawn by the sampler,
every acceptance test in `[0, 1]` changes by at most `n * distance`. -/
theorem vector_average_difference_abs_le {Index : Type*} [Fintype Index] [DecidableEq Index]
    (test : (Index → Scalar) → ℝ) (nonnegative : ∀ c, 0 ≤ test c) (atMostOne : ∀ c, test c ≤ 1) :
    |average (fun draws : Index → Draw => test (fun index => sample (draws index))) -
        average test| ≤ Fintype.card Index * distance :=
  hybrid_index sample distance
    (fun t low high => by
      rw [average_sample, average_scalar]
      exact expect_difference_abs_le t low high)
    test nonnegative atMostOne

/-! ### Transfer to the interactive PiRLC extractor -/

section Extractor

open NightstreamFPrime.Spec.Folding.PiRLC
open CoordinateRetry CoordinateOracle CoordinateOracleStar CoordinateTerminalLaw

variable {Index Assignment : Type*} [Fintype Index] [DecidableEq Index] [Fintype Assignment]

/-- Acceptance probability when the verifier draws every challenge coordinate
with the whole-vector sampler. -/
noncomputable def sampledRate (oracle : Oracle Index Scalar Assignment)
    (check : (Index → Scalar) → Assignment → Bool) : ℝ :=
  average (fun draws : Index → Draw =>
    (line oracle check).acceptance (fun index => sample (draws index)))

theorem rate_eq_average (oracle : Oracle Index Scalar Assignment)
    (check : (Index → Scalar) → Assignment → Bool) :
    (line oracle check).rate = average (line oracle check).acceptance := by
  unfold Line.rate Line.weight average
  rw [sum_div]

/-- Sampled acceptance exceeds uniform acceptance by at most `n * distance`. -/
theorem sampledRate_le (oracle : Oracle Index Scalar Assignment)
    (check : (Index → Scalar) → Assignment → Bool) :
    sampledRate oracle check ≤ (line oracle check).rate + Fintype.card Index * distance := by
  have close := vector_average_difference_abs_le (line oracle check).acceptance
    (line oracle check).nonnegative (line oracle check).atMostOne
  rw [rate_eq_average]
  unfold sampledRate
  linarith [(abs_le.mp close).2]

/-- V6: the existing uniform-challenge extractor returns with probability
at least the sampled acceptance minus `n / |C|` and `n * distance`.
The right-hand probability is still the uniform extractor experiment;
this theorem does not identify a Poseidon2 transcript execution. -/
theorem returningProbability_sampled_lower_bound (oracle : Oracle Index Scalar Assignment)
    (check : (Index → Scalar) → Assignment → Bool)
    (returns : (Index → Scalar) → Option Assignment →
      (Index → Outcome (Challenge := Scalar) (Assignment := Assignment)) → Bool)
    (returnsOnFork : ∀ vector initial outputs,
      0 < outcomeMass oracle check vector initial outputs → returns vector initial outputs = true) :
    sampledRate oracle check - Fintype.card Index * distance -
        (Fintype.card Index : ℝ) / Fintype.card Scalar ≤
      returningProbability oracle check returns := by
  have uniform := returningProbability_lower_bound oracle check returns returnsOnFork
  linarith [sampledRate_le oracle check]

end Extractor

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
