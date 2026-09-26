import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Fin
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Mathlib.SetTheory.Cardinal.Finite
import Mathlib.Algebra.BigOperators.Field
import NightstreamFPrime.Spec.AjtaiSetupV1.ReductionBias
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.Definition

/-!
Owns the exact law of the whole-vector sampler under uniform draws.

Experiment: one draw of four independent uniform field values.
Tests: every function from scalars to `[0, 1]`, so every event.
Per-call bound: sampled and uniform expectations differ by at most
`distance = r (N - r) / (N M)`, with `M = p^4`, `N = 5^54` and
`r = M mod N`; `distance < 2^-132`.

There is no sampler failure event. No law is assigned to Poseidon2.
-/

namespace NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler

open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open ProductionAlphabet ProductionStrongSet
open Finset

/-! ### V2: uniform draws give a uniform integer below `p^4` -/

/-- The base-`p` reading is a bijection, so each integer below `p^4` has
exactly one draw. -/
theorem drawIndex_fiber_card (value : Fin drawCount) :
    Nat.card {draw : Draw // drawIndex draw = value} = 1 := by
  rw [Nat.card_congr (drawIndex.subtypeEquiv (q := (· = value)) (fun _ => Iff.rfl)),
    Nat.card_unique]

/-! ### V1: exact fiber counts -/

/-- Residues in an initial segment that fits inside one block. -/
private theorem count_residue_within (modulus residue length : Nat)
    (within : length ≤ modulus) :
    Nat.count (fun value => value % modulus = residue) length =
      if residue < length then 1 else 0 := by
  rw [Nat.count_eq_card_filter_range]
  split_ifs with below
  · rw [Finset.card_eq_one]
    refine ⟨residue, ?_⟩
    ext value
    simp only [Finset.mem_filter, Finset.mem_range, Finset.mem_singleton]
    constructor
    · rintro ⟨lt, same⟩
      rwa [Nat.mod_eq_of_lt (lt_of_lt_of_le lt within)] at same
    · rintro rfl
      exact ⟨below, Nat.mod_eq_of_lt (lt_of_lt_of_le below within)⟩
  · rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
    intro value member same
    rw [Finset.mem_range] at member
    rw [Nat.mod_eq_of_lt (lt_of_lt_of_le member within)] at same
    exact below (same ▸ member)

/-- Each residue below `N` has `q` preimages below `M`, plus one if it lies
in the incomplete final block. -/
theorem residue_count (residue : Fin scalarCount) :
    Nat.count (fun value => value % scalarCount = residue.val) drawCount =
      quotient + if residue.val < remainder then 1 else 0 := by
  rw [AjtaiSetupV1.ReductionBias.event_count drawCount scalarCount
      (fun value => value = residue.val),
    count_residue_within scalarCount residue.val scalarCount le_rfl,
    count_residue_within scalarCount residue.val (drawCount % scalarCount)
      (Nat.mod_lt _ scalarCount_pos).le]
  simp only [residue.isLt, if_true, Nat.mul_one]
  rfl

private def finSubtypeEquiv (length : Nat) (event : Nat → Prop) :
    {value : Fin length // event value.val} ≃ {value : Nat // value < length ∧ event value} where
  toFun value := ⟨value.val.val, value.val.isLt, value.property⟩
  invFun value := ⟨⟨value.val, value.property.1⟩, value.property.2⟩
  left_inv _ := rfl
  right_inv _ := rfl

/-- Exact number of draws that give one scalar. -/
theorem sample_fiber_card (scalar : Scalar) :
    Nat.card {draw : Draw // sample draw = scalar} =
      quotient + if (scalarIndex scalar).val < remainder then 1 else 0 := by
  let event : Nat → Prop := fun value => value % scalarCount = (scalarIndex scalar).val
  letI := Nat.CountSet.fintype event drawCount
  rw [Nat.card_congr (Equiv.subtypeEquivRight (fun draw => sample_eq_iff draw scalar)),
    Nat.card_congr (drawIndex.subtypeEquiv (q := fun value : Fin drawCount => event value.val)
      (fun _ => Iff.rfl)),
    Nat.card_congr (finSubtypeEquiv drawCount event), Nat.card_eq_fintype_card,
    ← Nat.count_eq_card_fintype]
  exact residue_count (scalarIndex scalar)

/-! ### Generic finite facts; no concrete parameter is unfolded -/

/-- Grouping a sum by the value of `g`. -/
private theorem sum_comp_eq_sum_card {α β : Type*} [Fintype α] [Fintype β] [DecidableEq β]
    (g : α → β) (f : β → ℝ) :
    ∑ a, f (g a) = ∑ b, (Fintype.card {a // g a = b} : ℝ) * f b := by
  rw [← Finset.sum_fiberwise_of_maps_to (s := univ) (t := univ) (g := g)
    (fun _ _ => mem_univ _)]
  refine sum_congr rfl fun b _ => ?_
  have constant : ∀ a ∈ ({a ∈ univ | g a = b} : Finset α), f (g a) = f b := by
    intro a member
    rw [(mem_filter.mp member).2]
  rw [sum_congr rfl constant, sum_const, nsmul_eq_mul, Fintype.card_subtype]

/-- Weighted comparison over `Fin N`: weight `q + 1` below `r`, `q` above,
against the uniform weight, for any test in `[0, 1]`. -/
private theorem weighted_bound (modulus samples blocks rest : Nat)
    (modulusPositive : 0 < modulus) (samplesPositive : 0 < samples)
    (total : samples = blocks * modulus + rest) (restLe : rest ≤ modulus)
    (test : Fin modulus → ℝ)
    (nonnegative : ∀ value, 0 ≤ test value) (atMostOne : ∀ value, test value ≤ 1) :
    |(∑ value, ((blocks : ℝ) + if value.val < rest then 1 else 0) * test value) / samples -
        (∑ value, test value) / modulus| ≤
      ((rest * (modulus - rest) : Nat) : ℝ) / ((modulus * samples : Nat) : ℝ) := by
  set low : Fin modulus → Prop := fun value => value.val < rest
  have modulusReal : (0 : ℝ) < modulus := by exact_mod_cast modulusPositive
  have samplesReal : (0 : ℝ) < samples := by exact_mod_cast samplesPositive
  have denominator : (0 : ℝ) < modulus * samples := mul_pos modulusReal samplesReal
  have totalReal : (samples : ℝ) = blocks * modulus + rest := by exact_mod_cast total
  have restReal : (rest : ℝ) ≤ modulus := by exact_mod_cast restLe
  have target : ((rest * (modulus - rest) : Nat) : ℝ) / ((modulus * samples : Nat) : ℝ) =
      rest * (modulus - rest) / (modulus * samples) := by
    push_cast [Nat.cast_sub restLe]
    ring
  have weight : ∀ value : Fin modulus,
      ((blocks : ℝ) + if low value then 1 else 0) / samples - 1 / modulus =
        ((if low value then (modulus : ℝ) else 0) - rest) / (modulus * samples) := by
    intro value
    field_simp
    split_ifs <;> linarith
  have difference :
      (∑ value, ((blocks : ℝ) + if low value then 1 else 0) * test value) / samples -
          (∑ value, test value) / modulus =
        ∑ value, test value *
          (((if low value then (modulus : ℝ) else 0) - rest) / (modulus * samples)) := by
    rw [Finset.sum_div, Finset.sum_div, ← sum_sub_distrib]
    refine sum_congr rfl fun value _ => ?_
    rw [← weight value]
    ring
  have lowCount : (#({value ∈ univ | low value} : Finset (Fin modulus)) : ℝ) = rest := by
    have count : #({value ∈ univ | low value} : Finset (Fin modulus)) = min modulus rest :=
      Fin.card_filter_val_lt
    rw [count, min_eq_right restLe]
  have highCount : (#({value ∈ univ | ¬ low value} : Finset (Fin modulus)) : ℝ) =
      modulus - rest := by
    have split := Finset.card_filter_add_card_filter_not
      (s := (univ : Finset (Fin modulus))) low
    rw [card_univ, Fintype.card_fin] at split
    have splitReal : (#({value ∈ univ | low value} : Finset (Fin modulus)) : ℝ) +
        #({value ∈ univ | ¬ low value} : Finset (Fin modulus)) = modulus := by
      exact_mod_cast split
    rw [lowCount] at splitReal
    linarith
  have upper : ∑ value, test value *
        (((if low value then (modulus : ℝ) else 0) - rest) / (modulus * samples)) ≤
      rest * (modulus - rest) / (modulus * samples) := by
    calc ∑ value, test value *
          (((if low value then (modulus : ℝ) else 0) - rest) / (modulus * samples))
        ≤ ∑ value, (if low value then
            ((modulus : ℝ) - rest) / (modulus * samples) else 0) := by
          refine sum_le_sum fun value _ => ?_
          split_ifs
          · have positive : 0 ≤ ((modulus : ℝ) - rest) / (modulus * samples) :=
              div_nonneg (by linarith) denominator.le
            calc test value * (((modulus : ℝ) - rest) / (modulus * samples))
                ≤ 1 * (((modulus : ℝ) - rest) / (modulus * samples)) :=
                  mul_le_mul_of_nonneg_right (atMostOne value) positive
              _ = _ := one_mul _
          · have negative : ((0 : ℝ) - rest) / (modulus * samples) ≤ 0 :=
              div_nonpos_of_nonpos_of_nonneg (by simp) denominator.le
            exact mul_nonpos_of_nonneg_of_nonpos (nonnegative value) negative
      _ = rest * (modulus - rest) / (modulus * samples) := by
          rw [← sum_filter, sum_const, nsmul_eq_mul, lowCount]
          ring
  have lower : -(rest * (modulus - rest) / (modulus * samples)) ≤
      ∑ value, test value *
        (((if low value then (modulus : ℝ) else 0) - rest) / (modulus * samples)) := by
    calc -(rest * (modulus - rest) / (modulus * samples))
        = ∑ value, (if low value then 0 else
            -((rest : ℝ) / (modulus * samples))) := by
          rw [Finset.sum_ite, sum_const_zero, zero_add, sum_const, nsmul_eq_mul, highCount]
          ring
      _ ≤ ∑ value, test value *
          (((if low value then (modulus : ℝ) else 0) - rest) / (modulus * samples)) := by
          refine sum_le_sum fun value _ => ?_
          split_ifs
          · have positive : 0 ≤ ((modulus : ℝ) - rest) / (modulus * samples) :=
              div_nonneg (by linarith) denominator.le
            exact mul_nonneg (nonnegative value) positive
          · have negative : ((0 : ℝ) - rest) / (modulus * samples) ≤ 0 :=
              div_nonpos_of_nonpos_of_nonneg (by simp) denominator.le
            calc -((rest : ℝ) / (modulus * samples))
                = 1 * (((0 : ℝ) - rest) / (modulus * samples)) := by ring
              _ ≤ test value * (((0 : ℝ) - rest) / (modulus * samples)) :=
                  mul_le_mul_of_nonpos_right (atMostOne value) negative
  rw [difference, target]
  exact abs_le.mpr ⟨lower, upper⟩

/-! ### V1: comparison with uniform scalars -/

/-- Expectation of a test under the sampler with uniform draws. -/
noncomputable def sampledExpect (test : Scalar → ℝ) : ℝ :=
  (∑ draw : Draw, test (sample draw)) / drawCount

/-- Expectation of a test under uniform scalars. -/
noncomputable def uniformExpect (test : Scalar → ℝ) : ℝ :=
  (∑ scalar : Scalar, test scalar) / scalarCount

/-- Per-call statistical distance `r (N - r) / (N M)`. -/
noncomputable def distance : ℝ :=
  ((remainder * (scalarCount - remainder) : Nat) : ℝ) / ((scalarCount * drawCount : Nat) : ℝ)

theorem distance_lt : distance < 1 / 2 ^ 132 := by
  have exact : remainder * (scalarCount - remainder) * 2 ^ 132 < scalarCount * drawCount := by
    decide
  have denominator : (0 : ℝ) < ((scalarCount * drawCount : Nat) : ℝ) := by
    exact_mod_cast Nat.mul_pos scalarCount_pos drawCount_pos
  rw [distance, div_lt_div_iff₀ denominator (by positivity), one_mul]
  exact_mod_cast exact

theorem distance_nonnegative : 0 ≤ distance := by
  unfold distance
  positivity

/-- V1 main bound: for every test with values in `[0, 1]`, the sampled and
uniform expectations differ by at most `distance`. -/
theorem expect_difference_abs_le (test : Scalar → ℝ)
    (nonnegative : ∀ scalar, 0 ≤ test scalar) (atMostOne : ∀ scalar, test scalar ≤ 1) :
    |sampledExpect test - uniformExpect test| ≤ distance := by
  have grouped : ∑ draw : Draw, test (sample draw) =
      ∑ value : Fin scalarCount, ((quotient : ℝ) + if value.val < remainder then 1 else 0) *
        test (scalarIndex.symm value) := by
    rw [sum_comp_eq_sum_card sample test,
      ← Equiv.sum_comp scalarIndex.symm]
    refine sum_congr rfl fun value _ => ?_
    have count := sample_fiber_card (scalarIndex.symm value)
    rw [Nat.card_eq_fintype_card, Equiv.apply_symm_apply] at count
    rw [count]
    push_cast
    split_ifs <;> simp
  have uniform : ∑ scalar : Scalar, test scalar =
      ∑ value : Fin scalarCount, test (scalarIndex.symm value) :=
    (Equiv.sum_comp scalarIndex.symm test).symm
  rw [sampledExpect, uniformExpect, grouped, uniform, distance]
  exact weighted_bound scalarCount drawCount quotient remainder scalarCount_pos drawCount_pos
    drawCount_eq remainder_lt.le (fun value => test (scalarIndex.symm value))
    (fun value => nonnegative _) (fun value => atMostOne _)

end NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
