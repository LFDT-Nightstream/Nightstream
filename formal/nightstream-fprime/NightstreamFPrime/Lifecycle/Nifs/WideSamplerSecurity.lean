import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import NightstreamFPrime.Lifecycle.Nifs.WideFiatShamir

/-! The wide key's exact transcript and its separate sampler error budget.
Fresh-batch extraction uses ScheduleLaw; adaptive raw-block observations
use OracleModel with all calls counted. Neither law is inferred for Poseidon2.
The actual-verifier event below is the same event used by WideFiatShamir. -/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.WideSamplerSecurity

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec NightstreamFPrime.Spec.Folding
open Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler
open PaperAlgebra MeasureTheory Spec.Folding.PiCCS.PaperJoint

variable {logicalWidth : Nat}
  {publicFits : ringDegree * publicRingColumns ≤ Phi81CarrierLayout.carrierWidth logicalWidth}
  (relation : ProductionKey.LogicalRelation logicalWidth publicFits)
  (ajtai : AjtaiKey (logicalWidth := logicalWidth) (publicFits := publicFits))

/-- No sampler-abort event remains, for any concrete starting state. -/
theorem no_sampler_abort (state : Poseidon2.State) :
    (PiRLC.Wide.Key.key relation ajtai).piRlcResponse state ≠ none := by
  rw [PiRLC.Wide.Key.key_response]
  simp only [ne_eq, reduceCtorEq, not_false_eq_true]

/-- Replaying a normalized query history from `seed` yields the response used
by the wide verification key. This holds for any seed and history; it does not
identify the verifier's PiCCS output state with a complete-transcript replay. -/
theorem response_history (seed : Poseidon2.State) (history : List Draw) (coordinate : Fin 17) :
    PiRLC.Wide.Key.response (TranscriptHistory.replay seed history) coordinate =
      Phi81StrongSet.embedScalar
        (sample (TranscriptHistory.answer seed (ScheduleLaw.queryAt history coordinate))) := by
  rw [TranscriptHistory.queryAt_sample]
  rfl

/-- Changing the sampler leaves the interactive source statement unchanged. -/
theorem statement_unchanged
    (running : Running (logicalWidth := logicalWidth) (publicFits := publicFits))
    (fresh : Fresh (logicalWidth := logicalWidth) (publicFits := publicFits)) :
    (PiRLC.Wide.Key.key relation ajtai).statement running fresh =
      (ProductionKey.key relation ajtai).statement running fresh := by
  unfold Spec.Folding.Nifs.PaperNonInteractive.Key.statement
  rfl

section Adaptive

variable {Context OracleState : Type*}
  (running : Context → Running (logicalWidth := logicalWidth) (publicFits := publicFits))
  (fresh : Context → Fresh (logicalWidth := logicalWidth) (publicFits := publicFits))
  (decode : OracleModel.Outcome OracleState → Context × Option (WideFiatShamir.RealOutput relation))

/-- The test checks the actual wide verifier and its exact child openings.
The supplied program/decoder defines the adversary experiment, not its
acceptance predicate. Raw lanes and the final consistent cache remain visible. -/
noncomputable def successTest (outcome : OracleModel.Outcome OracleState) : ℝ := by
  classical
  exact if WideFiatShamir.RealSuccess relation ajtai
    (running (decode outcome).1) (fresh (decode outcome).1) (decode outcome).2 then 1 else 0

private theorem successTest_bounds (outcome : OracleModel.Outcome OracleState) :
    0 ≤ successTest relation ajtai running fresh decode outcome ∧
      successTest relation ajtai running fresh decode outcome ≤ 1 := by
  classical
  unfold successTest
  split_ifs <;> norm_num

theorem adaptive_bias_bound (program : OracleModel.Program OracleState)
    (initial : OracleState) (q : Nat) :
    |average (fun tape : Fin q → Draw => successTest relation ajtai running fresh decode
        (OracleModel.run program initial OracleModel.empty q tape)) -
      balancedAverage (fun tape : Fin q → Draw => successTest relation ajtai running fresh decode
        (OracleModel.run program initial OracleModel.empty q tape))| ≤ q * distance :=
  OracleModel.run_bias_bound program initial q (successTest relation ajtai running fresh decode)
    (fun result => (successTest_bounds relation ajtai running fresh decode result).1)
    (fun result => (successTest_bounds relation ajtai running fresh decode result).2)

/-- The approximation premise is the explicit concrete-Poseidon2 boundary.
It names the actual success probability, so an unrelated scalar cannot
stand in for verifier acceptance. This theorem does not establish the
premise or identify balanced raw replies with a Fiat–Shamir extractor. -/
theorem concrete_bias_bound (program : OracleModel.Program OracleState)
    (initial : OracleState) (q : Nat)
    (law : PMF (Context × Option (WideFiatShamir.RealOutput relation)))
    (modelError : ℝ)
    (model : |WideFiatShamir.realSuccessProbability relation ajtai running fresh law -
      average (fun tape : Fin q → Draw => successTest relation ajtai running fresh decode
        (OracleModel.run program initial OracleModel.empty q tape))| ≤ modelError) :
    |WideFiatShamir.realSuccessProbability relation ajtai running fresh law -
      balancedAverage (fun tape : Fin q → Draw => successTest relation ajtai running fresh decode
        (OracleModel.run program initial OracleModel.empty q tape))| ≤ modelError + q * distance :=
  OracleModel.under_block_oracle_assumption program initial q
    (successTest relation ajtai running fresh decode)
    (fun result => (successTest_bounds relation ajtai running fresh decode result).1)
    (fun result => (successTest_bounds relation ajtai running fresh decode result).2)
    (WideFiatShamir.realSuccessProbability relation ajtai running fresh law) modelError model

end Adaptive

/-- Only PiCCS test failures are unioned: wide sampling is total. Per-call
bounds must hold in this same trace law, including the preceding history.
FS, hash, MSIS and extraction losses belong to their separate games. -/
theorem any_test_le {Trace : Type*} [MeasurableSpace Trace]
    (law : Measure Trace) [IsProbabilityMeasure law]
    (calls : Nat) (test : Fin calls → Set Trace)
    (testBound : ∀ index, law (test index) ≤ ENNReal.ofReal
      (PiCCS.PaperJoint.IndependentExecution.testError productionShape 9)) :
    law (⋃ index, test index) ≤ min 1 (ENNReal.ofReal ((calls : ℝ) *
      PiCCS.PaperJoint.IndependentExecution.testError productionShape 9)) := by
  apply le_min prob_le_one
  calc
    _ ≤ ∑ index : Fin calls, law (test index) := measure_iUnion_fintype_le law _
    _ ≤ ∑ _index : Fin calls, ENNReal.ofReal
        (PiCCS.PaperJoint.IndependentExecution.testError productionShape 9) := by
      exact Finset.sum_le_sum (fun index _ => testBound index)
    _ = _ := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg calls), ENNReal.ofReal_natCast]

end NightstreamFPrime.Lifecycle.Nifs.WideSamplerSecurity
