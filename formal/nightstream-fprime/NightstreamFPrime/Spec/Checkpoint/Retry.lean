import NightstreamFPrime.Spec.RandomOracle
import Mathlib.Algebra.BigOperators.Fin

/-! The retry loop of the extractor of Akita Theorem 10.17 (ePrint 2026/1983, Appendix E.2), with
a limit on the number of attempts. Each attempt draws one fresh uniform sample. The loop returns
the first sample that passes. -/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Checkpoint

open scoped BigOperators
open NightstreamFPrime.Spec.RandomOracle (expect_pos)

attribute [local instance low] Classical.propDecidable

variable {Sample : Type} [Fintype Sample]

theorem expect_cons (budget : ℕ) (value : (Fin (budget + 1) → Sample) → ℝ) :
    𝔼 tape, value tape = 𝔼 sample, 𝔼 rest : Fin budget → Sample, value (Fin.cons sample rest) := by
  rw [← Finset.expect_product' Finset.univ Finset.univ
    (fun sample rest => value (Fin.cons sample rest)), Finset.univ_product_univ]
  exact Fintype.expect_equiv (Fin.consEquiv fun _ => Sample).symm _ _ fun tape => by
    simp [Fin.consEquiv, Fin.cons_self_tail]

theorem expect_pair {Left Right : Type} [Fintype Left] [Fintype Right] (value : Left × Right → ℝ) :
    𝔼 pair, value pair = 𝔼 left, 𝔼 right, value (left, right) := by
  have split := Finset.expect_product' (Finset.univ : Finset Left) (Finset.univ : Finset Right)
    fun left right => value (left, right)
  rw [Finset.univ_product_univ] at split
  exact split

theorem expect_fst {Left Right : Type} [Fintype Left] [Fintype Right] [Nonempty Right]
    (value : Left → ℝ) : 𝔼 pair : Left × Right, value pair.1 = 𝔼 left, value left := by
  rw [expect_pair]
  exact Finset.expect_congr rfl fun left _ =>
    Finset.expect_const (s := (Finset.univ : Finset Right)) Finset.univ_nonempty (value left)

/-- The first sample of the tape that passes. -/
noncomputable def firstPass (pass : Sample → Prop) :
    (budget : ℕ) → (Fin budget → Sample) → Option Sample
  | 0, _ => none
  | budget + 1, tape =>
      if pass (tape 0) then some (tape 0) else firstPass pass budget (Fin.tail tape)

omit [Fintype Sample] in
theorem firstPass_pass {pass : Sample → Prop} {budget : ℕ} {tape : Fin budget → Sample}
    {sample : Sample} (found : firstPass pass budget tape = some sample) : pass sample := by
  induction budget with
  | zero => simp [firstPass] at found
  | succ budget induction =>
      unfold firstPass at found
      split at found
      · rename_i passes
        exact (Option.some.inj found) ▸ passes
      · exact induction found

/-- The number of runs of the loop: each attempt up to the first that passes costs `cost`. -/
noncomputable def loopCost (pass : Sample → Prop) (cost : Sample → ℕ) :
    (budget : ℕ) → (Fin budget → Sample) → ℕ
  | 0, _ => 0
  | budget + 1, tape =>
      cost (tape 0) + if pass (tape 0) then 0 else loopCost pass cost budget (Fin.tail tape)

private theorem chance_nonnegative (pass : Sample → Prop) :
    0 ≤ 𝔼 sample, if pass sample then (1 : ℝ) else 0 :=
  Finset.expect_nonneg fun _ _ => by split <;> norm_num

variable [Nonempty Sample]

private theorem chance_le_one (pass : Sample → Prop) :
    𝔼 sample, (if pass sample then (1 : ℝ) else 0) ≤ 1 :=
  (Finset.expect_le_expect fun _ _ => by split <;> norm_num).trans
    (Finset.expect_const Finset.univ_nonempty 1).le

private theorem expect_step (pass : Sample → Prop) (budget : ℕ) (now : Sample → ℝ) (later : ℝ)
    (value : (Fin (budget + 1) → Sample) → ℝ)
    (each : ∀ sample, 𝔼 rest : Fin budget → Sample, value (Fin.cons sample rest) =
      now sample + (1 - if pass sample then (1 : ℝ) else 0) * later) :
    𝔼 tape, value tape =
      (𝔼 sample, now sample) + (1 - 𝔼 sample, if pass sample then (1 : ℝ) else 0) * later := by
  rw [expect_cons]
  simp only [each]
  rw [Finset.expect_add_distrib, ← Finset.expect_mul, Finset.expect_sub_distrib,
    Finset.expect_const Finset.univ_nonempty]

/-- The loop returns a sample with law at most the law of one attempt, conditioned on a pass. -/
theorem firstPass_hit_le (pass : Sample → Prop) (value : Sample → ℝ)
    (nonnegative : ∀ sample, 0 ≤ value sample) (budget : ℕ) :
    (𝔼 sample, if pass sample then (1 : ℝ) else 0) *
        𝔼 tape : Fin budget → Sample, (firstPass pass budget tape).elim 0 value ≤
      𝔼 sample, if pass sample then value sample else 0 := by
  have total : 0 ≤ 𝔼 sample, if pass sample then value sample else 0 :=
    Finset.expect_nonneg fun sample _ => by split <;> simp [nonnegative]
  induction budget with
  | zero => simpa [firstPass] using total
  | succ budget induction =>
      rw [expect_step pass budget (fun sample => if pass sample then value sample else 0)
        (𝔼 tape : Fin budget → Sample, (firstPass pass budget tape).elim 0 value)]
      · nlinarith [mul_nonneg (sub_nonneg.mpr (chance_le_one pass)) (sub_nonneg.mpr induction)]
      · intro sample
        by_cases passes : pass sample <;> simp [firstPass, passes]

/-- The loop finds no sample with probability at most `1/(budget · p)`, where `p` is the chance
that one attempt passes. -/
theorem firstPass_none_le (pass : Sample → Prop) (budget : ℕ) :
    budget * ((𝔼 sample, if pass sample then (1 : ℝ) else 0) *
      𝔼 tape : Fin budget → Sample, if firstPass pass budget tape = none then (1 : ℝ) else 0) ≤
        1 := by
  induction budget with
  | zero => simp
  | succ budget induction =>
      rw [expect_step pass budget (fun _ => 0)
        (𝔼 tape : Fin budget → Sample, if firstPass pass budget tape = none then (1 : ℝ) else 0)]
      · have low := chance_nonnegative pass
        have high := chance_le_one pass
        have missLow : 0 ≤ 𝔼 tape : Fin budget → Sample,
            if firstPass pass budget tape = none then (1 : ℝ) else 0 :=
          Finset.expect_nonneg fun _ _ => by split <;> norm_num
        have missHigh : 𝔼 tape : Fin budget → Sample,
            (if firstPass pass budget tape = none then (1 : ℝ) else 0) ≤ 1 :=
          (Finset.expect_le_expect fun _ _ => by split <;> norm_num).trans
            (Finset.expect_const Finset.univ_nonempty 1).le
        have first := mul_le_mul_of_nonneg_left induction (sub_nonneg.mpr high)
        have second := mul_nonneg low missLow
        have third := mul_nonneg low (mul_nonneg (sub_nonneg.mpr high) (sub_nonneg.mpr missHigh))
        push_cast
        simp only [Finset.expect_const_zero, zero_add]
        nlinarith
      · intro sample
        by_cases passes : pass sample <;> simp [firstPass, passes]

/-- The expected cost of the loop is at most the expected cost of one attempt divided by the
chance that one attempt passes (the identity of Wald for a loop with a limit). -/
theorem loopCost_le (pass : Sample → Prop) (cost : Sample → ℕ) (budget : ℕ) :
    (𝔼 sample, if pass sample then (1 : ℝ) else 0) *
        𝔼 tape : Fin budget → Sample, (loopCost pass cost budget tape : ℝ) ≤
      𝔼 sample, (cost sample : ℝ) := by
  have total : 0 ≤ 𝔼 sample, (cost sample : ℝ) :=
    Finset.expect_nonneg fun sample _ => Nat.cast_nonneg _
  induction budget with
  | zero => simpa [loopCost] using total
  | succ budget induction =>
      rw [expect_step pass budget (fun sample => (cost sample : ℝ))
        (𝔼 tape : Fin budget → Sample, (loopCost pass cost budget tape : ℝ))]
      · nlinarith [mul_nonneg (sub_nonneg.mpr (chance_le_one pass)) (sub_nonneg.mpr induction),
          mul_nonneg (chance_nonnegative pass) total]
      · intro sample
        by_cases passes : pass sample <;> simp [loopCost, passes, Finset.expect_add_distrib]

/-- The expected cost of a loop that is entered after a base sample that passes is at most the
expected cost of one attempt, and is zero when no sample passes (Appendix E.2, "Cost"). -/
theorem loop_cost_le (pass : Sample → Prop) (cost : Sample → ℕ) (budget : ℕ) :
    𝔼 base, 𝔼 tape : Fin budget → Sample,
        (if pass base then (loopCost pass cost budget tape : ℝ) else 0) ≤
      if ∃ sample, pass sample then 𝔼 sample, (cost sample : ℝ) else 0 := by
  have same (base : Sample) : 𝔼 tape : Fin budget → Sample,
      (if pass base then (loopCost pass cost budget tape : ℝ) else 0) =
      (if pass base then (1 : ℝ) else 0) *
        𝔼 tape : Fin budget → Sample, (loopCost pass cost budget tape : ℝ) := by
    by_cases passes : pass base <;> simp [passes]
  simp only [same]
  rw [← Finset.expect_mul]
  by_cases possible : ∃ sample, pass sample
  · rw [ite_eq_left possible]
    exact loopCost_le pass cost budget
  · rw [ite_eq_right possible]
    have none (base : Sample) : ¬ pass base := fun passes => possible ⟨base, passes⟩
    simp [none]

/-- One retry loop after a base sample that passes. The loop fails when it finds no sample, or
when the base sample is in `gate` and the sample found has the key of the base sample. If each
key of a gated base sample has chance at most `ε` among the samples that pass, the loop fails with
probability at most `ε + 1/budget`. This is the binary sampling game of Akita Lemma 10.14 on one
table entry, and the success step of a checkpoint (Appendix E.2). -/
theorem fork_le {Key : Type} (pass gate : Sample → Prop) (key : Sample → Key) (ε : ℝ)
    (nonnegative : 0 ≤ ε) {budget : ℕ} (positive : 0 < budget)
    (rare : ∀ base, pass base → gate base →
      𝔼 sample, (if pass sample ∧ key sample = key base then (1 : ℝ) else 0) ≤ ε) :
    𝔼 base, 𝔼 tape : Fin budget → Sample,
        (if pass base ∧ (firstPass pass budget tape = none ∨
          gate base ∧ ∃ sample, firstPass pass budget tape = some sample ∧ key sample = key base)
          then (1 : ℝ) else 0) ≤
      if ∃ sample, pass sample then ε + 1 / budget else 0 := by
  by_cases possible : ∃ sample, pass sample
  · rw [ite_eq_left possible]
    obtain ⟨witness, passes⟩ := possible
    have chancePositive : 0 < 𝔼 sample, if pass sample then (1 : ℝ) else 0 :=
      expect_pos (fun _ => by split <;> norm_num) (item := witness) (by simp [passes])
    have budgetPositive : (0 : ℝ) < budget := by exact_mod_cast positive
    have split (base : Sample) (tape : Fin budget → Sample) :
        (if pass base ∧ (firstPass pass budget tape = none ∨
          gate base ∧ ∃ sample, firstPass pass budget tape = some sample ∧ key sample = key base)
          then (1 : ℝ) else 0) ≤
        (if pass base then (if firstPass pass budget tape = none then (1 : ℝ) else 0) else 0) +
          (if pass base ∧ gate base then (firstPass pass budget tape).elim 0
            (fun sample => if key sample = key base then (1 : ℝ) else 0) else 0) := by
      by_cases passesBase : pass base
      · rcases found : firstPass pass budget tape with _ | sample
        · by_cases gated : gate base <;> simp [passesBase, gated]
        · by_cases gated : gate base <;> by_cases same : key sample = key base <;>
            simp [passesBase, gated, same]
      · simp [passesBase]
    have missed : 𝔼 base, 𝔼 tape : Fin budget → Sample,
        (if pass base then (if firstPass pass budget tape = none then (1 : ℝ) else 0) else 0) ≤
          1 / budget := by
      have same (base : Sample) : 𝔼 tape : Fin budget → Sample,
          (if pass base then (if firstPass pass budget tape = none then (1 : ℝ) else 0) else 0) =
          (if pass base then (1 : ℝ) else 0) * 𝔼 tape : Fin budget → Sample,
            if firstPass pass budget tape = none then (1 : ℝ) else 0 := by
        by_cases passesBase : pass base <;> simp [passesBase]
      simp only [same]
      rw [← Finset.expect_mul, le_div_iff₀ budgetPositive, mul_comm]
      exact firstPass_none_le pass budget
    have repeated : 𝔼 base, 𝔼 tape : Fin budget → Sample,
        (if pass base ∧ gate base then (firstPass pass budget tape).elim 0
          (fun sample => if key sample = key base then (1 : ℝ) else 0) else 0) ≤ ε := by
      refine le_of_mul_le_mul_left ?_ chancePositive
      rw [Finset.mul_expect]
      calc
        _ ≤ 𝔼 base, (if pass base then ε else 0) := by
          refine Finset.expect_le_expect fun base _ => ?_
          by_cases inside : pass base ∧ gate base
          · simp only [ite_eq_left inside, ite_eq_left inside.1]
            refine (firstPass_hit_le pass _ (fun _ => by split <;> norm_num) budget).trans ?_
            refine le_trans (le_of_eq ?_) (rare base inside.1 inside.2)
            exact Finset.expect_congr rfl fun sample _ => by
              by_cases passesSample : pass sample <;> simp [passesSample]
          · simp only [ite_eq_right inside, Finset.expect_const_zero, mul_zero]
            split <;> simp [nonnegative]
        _ = (𝔼 sample, if pass sample then (1 : ℝ) else 0) * ε := by
          rw [Finset.expect_mul]
          exact Finset.expect_congr rfl fun base _ => by split <;> simp
    calc
      _ ≤ 𝔼 base, 𝔼 tape : Fin budget → Sample,
            ((if pass base then (if firstPass pass budget tape = none then (1 : ℝ) else 0) else 0) +
              (if pass base ∧ gate base then (firstPass pass budget tape).elim 0
                (fun sample => if key sample = key base then (1 : ℝ) else 0) else 0)) :=
        Finset.expect_le_expect fun base _ => Finset.expect_le_expect fun tape _ => split base tape
      _ ≤ ε + 1 / budget := by
        simp only [Finset.expect_add_distrib]
        linarith
  · rw [ite_eq_right possible]
    have none (base : Sample) : ¬ pass base := fun passes => possible ⟨base, passes⟩
    simp [none]

end NightstreamFPrime.Spec.Checkpoint
