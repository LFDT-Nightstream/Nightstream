import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import NightstreamFPrime.Lifecycle.Types
import NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.IndependentExecution

/-!
The selected PiCCS test-error budget over a supplied number
of calls. The per-call probability hypotheses must hold in the same trace
experiment. Independence is not required. These events do not include the
Fiat--Shamir transfer, commitment/hash attacks, or a history extractor's loss.
-/

set_option autoImplicit false

namespace NightstreamFPrime.Lifecycle.Nifs.VerifierErrorBudget

open scoped BigOperators ENNReal
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open MeasureTheory

/-- The actual selected shape and degree give this test numerator. -/
theorem test_error_eq : IndependentExecution.testError productionShape 8 =
    (4589 : ℝ) / (goldilocksModulus : ℝ) ^ 2 := by
  unfold IndependentExecution.testError
  rw [Nat.cast_pow]
  change (28 : ℝ) * 8 / (goldilocksModulus : ℝ) ^ 2 +
    4365 / (goldilocksModulus : ℝ) ^ 2 = _
  ring

/-- Union the actual events in one probability space. The hypotheses can be
obtained from bounds conditional on the full preceding history; a claim only
about isolated uniform inputs does not discharge them for a transcript.
The count is supplied by the caller, not fixed by the production profile. -/
theorem any_test_error_le
    {Trace : Type*} [MeasurableSpace Trace]
    (law : Measure Trace) [IsProbabilityMeasure law]
    (calls : Nat) (test : Fin calls → Set Trace)
    (testBound : ∀ index, law (test index) ≤
      ENNReal.ofReal (IndependentExecution.testError productionShape 8)) :
    law (⋃ index, test index) ≤
      min 1 (ENNReal.ofReal ((calls : ℝ) * IndependentExecution.testError productionShape 8)) := by
  apply le_min prob_le_one
  calc
    _ ≤ ∑ index : Fin calls, law (test index) := measure_iUnion_fintype_le law _
    _ ≤ ∑ _index : Fin calls, ENNReal.ofReal (IndependentExecution.testError productionShape 8) :=
      Finset.sum_le_sum (fun index _ => testBound index)
    _ = ENNReal.ofReal ((calls : ℝ) * IndependentExecution.testError productionShape 8) := by
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, ENNReal.ofReal_mul (Nat.cast_nonneg calls), ENNReal.ofReal_natCast]

end NightstreamFPrime.Lifecycle.Nifs.VerifierErrorBudget
