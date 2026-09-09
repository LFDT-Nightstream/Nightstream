import NightstreamFPrime.Spec.Folding.Nifs.PaperWeakOracle
import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
The outer adversary call returns its original context and observed work.
The work contract includes private-coin acquisition and all fixed-key
preprocessing. Its PMF describes those private coins; no distribution table
is constructed or sampled by the reduction. Contexts and continuation work
are derived from this same call, with arbitrary value/work correlation.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Spec.Folding.Nifs.ContextPreparation

open PiRLC.PaperForkExtractionWork

variable {Tape Context Output : Type*}
  (tapes : PMF Tape) (prepare : Tape → Result Context)

/-- Execute the preparation once and pass its actual returned context to
the continuation. The dispatch/return transition adds one step. -/
def run (nextCall : Context → Result Output) (tape : Tape) : Result Output :=
  let prepared := prepare tape
  let continued := nextCall prepared.value
  ⟨continued.value, prepared.work + continued.work + 1⟩

theorem run_value (nextCall : Context → Result Output) (tape : Tape) :
    (run prepare nextCall tape).value = (nextCall (prepare tape).value).value := rfl

theorem run_work (nextCall : Context → Result Output) (tape : Tape) :
    (run prepare nextCall tape).work =
      (prepare tape).work + (nextCall (prepare tape).value).work + 1 := rfl

/-- The NIFS context law is the actual preparation-output marginal. -/
noncomputable def contexts : PMF Context := tapes.map fun tape => (prepare tape).value

theorem contexts_toReal (context : Context) :
    (contexts tapes prepare context).toReal =
      PaperWeakOracle.pushMass tapes (fun tape => (prepare tape).value) context := by
  classical
  have fiber : contexts tapes prepare context =
      ∑' tape : {tape // (prepare tape).value = context}, tapes tape.val := by
    rw [contexts, PMF.map_apply]
    simpa only [Set.indicator, Set.mem_setOf_eq, eq_comm] using
      (tsum_subtype {tape : Tape | (prepare tape).value = context}
        (fun tape : Tape => tapes tape)).symm
  rw [fiber]
  exact ENNReal.tsum_toReal_eq
    (fun tape : {tape // (prepare tape).value = context} => tapes.apply_ne_top tape.val)

/-- Every integrable observable keeps the exact probability and cost of the
actual returned context. No uniform law on contexts is assumed. -/
theorem value_hasSum (value : Context → ℝ)
    (summable : Summable fun tape => (tapes tape).toReal * value (prepare tape).value) :
    HasSum (fun context => (contexts tapes prepare context).toReal * value context)
      (∑' tape, (tapes tape).toReal * value (prepare tape).value) := by
  simp only [contexts_toReal]
  exact PaperWeakOracle.pushMass_value_hasSum tapes (fun tape => (prepare tape).value) value summable

/-- Nonnegative work is integrable under the context marginal exactly when
it is integrable on the original private tapes. Individual contexts and calls
have no imposed time bound. -/
theorem summable_iff (value : Context → ℝ) (nonnegative : ∀ context, 0 ≤ value context) :
    Summable (fun tape => (tapes tape).toReal * value (prepare tape).value) ↔
      Summable (fun context => (contexts tapes prepare context).toReal * value context) := by
  constructor
  · intro summable
    exact (value_hasSum tapes prepare value summable).summable
  · intro summable
    have weights : Summable (fun tape => (tapes tape).toReal) :=
      ENNReal.summable_toReal tapes.tsum_coe_ne_top
    let fiber := fun context => {tape : Tape | (prepare tape).value = context}
    have partition : ∀ tape : Tape, ∃! context, tape ∈ fiber context := by
      intro tape
      exact ⟨(prepare tape).value, rfl, fun _ same => same.symm⟩
    apply (summable_partition (s := fiber)
      (fun tape => mul_nonneg ENNReal.toReal_nonneg (nonnegative (prepare tape).value)) partition).mpr
    constructor
    · intro context
      apply ((weights.subtype (fun tape => (prepare tape).value = context)).mul_right (value context)).congr
      intro tape
      rw [tape.property]
      rfl
    · have fiberMean (context : Context) :
          (∑' tape : fiber context, (tapes tape.val).toReal * value (prepare tape.val).value) =
            (contexts tapes prepare context).toReal * value context := by
        calc
          _ = ∑' tape : fiber context, (tapes tape.val).toReal * value context := by
            apply tsum_congr
            intro tape
            rw [tape.property]
          _ = _ := by rw [tsum_mul_right, contexts_toReal]; rfl
      simpa only [fiberMean] using summable

/-- Extend the exact run clock to the continuation's conditional mean. -/
def clock (continuationMean : Context → ℝ) (tape : Tape) : ℝ :=
  (prepare tape).work + continuationMean (prepare tape).value + 1

theorem clock_eq_run_work (nextCall : Context → Result Output) (tape : Tape) :
    clock prepare (fun context => (nextCall context).work) tape =
      ((run prepare nextCall tape).work : ℝ) := by
  simp only [clock, run_work, Nat.cast_add, Nat.cast_one]

/-- Linearity adds the actual preparation cost once to the derived
continuation mean. It retains failed continuations and all preparation tapes. -/
theorem expected_work_eq (continuationMean : Context → ℝ)
    (nonnegative : ∀ context, 0 ≤ continuationMean context)
    (preparationSummable : Summable fun tape => (tapes tape).toReal * (prepare tape).work)
    (continuationSummable : Summable fun context =>
      (contexts tapes prepare context).toReal * continuationMean context) :
    Summable (fun tape => (tapes tape).toReal * clock prepare continuationMean tape) ∧
    (∑' tape, (tapes tape).toReal * clock prepare continuationMean tape) =
      (∑' tape, (tapes tape).toReal * (prepare tape).work) +
      (∑' context, (contexts tapes prepare context).toReal * continuationMean context) + 1 := by
  have continued := (summable_iff tapes prepare continuationMean nonnegative).mpr continuationSummable
  have pushforward := value_hasSum tapes prepare continuationMean continued
  have weights : Summable (fun tape => (tapes tape).toReal) :=
    ENNReal.summable_toReal tapes.tsum_coe_ne_top
  have weightSum : (∑' tape, (tapes tape).toReal) = 1 := by
    rw [← ENNReal.tsum_toReal_eq tapes.apply_ne_top, tapes.tsum_coe, ENNReal.toReal_one]
  have total := (preparationSummable.add continued).add weights
  refine ⟨?_, ?_⟩
  · simpa only [clock, mul_add, mul_one] using total
  · simp only [clock, mul_add, mul_one]
    rw [Summable.tsum_add (preparationSummable.add continued) weights,
      Summable.tsum_add preparationSummable continued, ← pushforward.tsum_eq, weightSum]

end NightstreamFPrime.Spec.Folding.Nifs.ContextPreparation
