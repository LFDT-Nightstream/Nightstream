import Nightstream.Implementation.Rust.CanonicalConformance.NativeStep.Conformance

/-! Focused regression for the bounded native `verify_step` differential. -/

namespace Nightstream.Tests.FPrimeNativeStepConformance

open Nightstream.Implementation.Rust.CanonicalConformance.NativeStep

#check generated_all_check
#check controlFlowAndCallConservation_of_mem_generated
#check nativeOutcome_eq_recorded_of_mem_generated

example : Generated.all.length = 11 := generated_case_count

example : Generated.rawEncoding.length = 14 := by
  rfl

example : Generated.honestRecursive.observed.executionOrder.length = 21 := by
  rfl

example :
    (Generated.honestRecursive.observed.transcript.map
      (fun transcript => transcript.orderedAppends.length)) = some 11 := by
  rfl

theorem honestBase_conserves_actual_calls :
    ControlFlowAndCallConservation Generated.honestBase := by
  exact controlFlowAndCallConservation_of_mem_generated
    Generated.honestBase (by simp [Generated.all])

theorem honestRecursive_conserves_actual_calls :
    ControlFlowAndCallConservation Generated.honestRecursive := by
  exact controlFlowAndCallConservation_of_mem_generated
    Generated.honestRecursive (by simp [Generated.all])

theorem nifsChildMutation_is_rejected :
    nativeOutcome Generated.nifsPiDecChildMutation =
      .rejected .nifsRejected := by
  exact nativeOutcome_eq_recorded_of_mem_generated
    Generated.nifsPiDecChildMutation (by simp [Generated.all])

theorem incomingHandleMutation_is_rejected :
    nativeOutcome Generated.incomingAccumulatorHandleMutation =
      .rejected .stateAuthorityMismatch := by
  exact nativeOutcome_eq_recorded_of_mem_generated
    Generated.incomingAccumulatorHandleMutation (by simp [Generated.all])

end Nightstream.Tests.FPrimeNativeStepConformance
