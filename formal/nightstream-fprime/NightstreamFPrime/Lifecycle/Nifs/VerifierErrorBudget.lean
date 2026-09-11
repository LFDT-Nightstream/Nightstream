import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import NightstreamFPrime.Lifecycle.Types
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution
import NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler.FieldBatchShortfall

/-!
The selected PiCCS test error and sampler-abort budget over a supplied number
of calls. The per-call probability hypotheses must hold in the same trace
experiment. Independence is not required. These events do not include the
Fiat--Shamir transfer, commitment/hash attacks, or a history extractor's loss.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.VerifierErrorBudget

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.Nifs.NonInteractive.PiRlcSampler
open MeasureTheory

/-- The actual selected shape and degree give this test numerator. -/
theorem test_error_eq : IndependentExecution.testError productionShape 9 =
    (13257 : ℝ) / (goldilocksModulus : ℝ) ^ 2 := by
  unfold IndependentExecution.testError
  rw [Nat.cast_pow]
  change (28 : ℝ) * 9 / (goldilocksModulus : ℝ) ^ 2 +
    13005 / (goldilocksModulus : ℝ) ^ 2 = _
  ring

/-- The proved 17-scalar field-comparison abort bound, with the actual
64-candidate and 16-bit sampler parameters. -/
noncomputable def batchAbortBound : ℝ :=
  17 * (Nat.choose ProductionAlphabet.candidateBound 11 : ℝ) /
    (ProductionAlphabet.chunkModulus : ℝ) ^ 11

noncomputable def perCall : ℝ :=
  IndependentExecution.testError productionShape 9 + batchAbortBound

private theorem test_nonnegative :
    0 ≤ IndependentExecution.testError productionShape 9 := by
  rw [test_error_eq]
  positivity

private theorem abort_nonnegative : 0 ≤ batchAbortBound := by
  unfold batchAbortBound
  positivity

private theorem comparison_abort_le :
    (FieldBatchShortfall.iidFieldBatchShortfallProbability : ℝ) ≤ batchAbortBound := by
  unfold batchAbortBound
  have lifted := (Rat.cast_le (K := ℝ)).mpr
    FieldBatchShortfall.iid_field_batch_shortfall_probability_le
  simpa only [Rat.cast_div, Rat.cast_mul, Rat.cast_natCast, Rat.cast_pow,
    Rat.cast_ofNat] using lifted

/-- Union the actual events in one probability space. The hypotheses can be
obtained from bounds conditional on the full preceding history; a claim only
about isolated uniform inputs does not discharge them for a transcript.
The count is supplied by the caller, not fixed by the production profile. -/
theorem any_test_or_sampler_abort_le
    {Trace : Type*} [MeasurableSpace Trace]
    (law : Measure Trace) [IsProbabilityMeasure law]
    (calls : Nat) (test abort : Fin calls → Set Trace)
    (testBound : ∀ index, law (test index) ≤
      ENNReal.ofReal (IndependentExecution.testError productionShape 9))
    (abortBound : ∀ index, law (abort index) ≤
      ENNReal.ofReal (FieldBatchShortfall.iidFieldBatchShortfallProbability : ℝ)) :
    law (⋃ index, test index ∪ abort index) ≤
      min 1 (ENNReal.ofReal ((calls : ℝ) * perCall)) := by
  apply le_min prob_le_one
  calc
    _ ≤ ∑ index : Fin calls, law (test index ∪ abort index) :=
      measure_iUnion_fintype_le law _
    _ ≤ ∑ _index : Fin calls, ENNReal.ofReal perCall := by
      apply Finset.sum_le_sum
      intro index _member
      calc
        _ ≤ law (test index) + law (abort index) := measure_union_le _ _
        _ ≤ ENNReal.ofReal (IndependentExecution.testError productionShape 9) +
            ENNReal.ofReal (FieldBatchShortfall.iidFieldBatchShortfallProbability : ℝ) :=
          add_le_add (testBound index) (abortBound index)
        _ ≤ ENNReal.ofReal (IndependentExecution.testError productionShape 9) +
            ENNReal.ofReal batchAbortBound :=
          _root_.add_le_add (le_refl _) (ENNReal.ofReal_le_ofReal comparison_abort_le)
        _ = ENNReal.ofReal perCall :=
          (ENNReal.ofReal_add test_nonnegative abort_nonnegative).symm
    _ = ENNReal.ofReal ((calls : ℝ) * perCall) := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg calls), ENNReal.ofReal_natCast]

end NightstreamFPrime.Lifecycle.Nifs.VerifierErrorBudget
