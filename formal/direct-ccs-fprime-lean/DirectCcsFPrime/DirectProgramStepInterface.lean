import DirectCcsFPrime.DirectProgramStep

/-!
Typed interface for deterministic direct-program step soundness.

Spec: `specs/DirectProgramStep.spec.md`
-/

namespace DirectCcsFPrime

namespace DirectProgramStepInterface

abbrev ComputedBoundaryStep :=
  @DirectProgramStep.ComputedBoundaryStep

abbrev computedBoundaryStep_functional :=
  @DirectProgramStep.computedBoundaryStep_functional

abbrev latest_currentBoundary_eq_compute :=
  @DirectProgramStep.latest_currentBoundary_eq_compute

abbrev latest_currentBoundary_functional :=
  @DirectProgramStep.latest_currentBoundary_functional

abbrev latest_publicImage_functional_of_accumulator_fields :=
  @DirectProgramStep.latest_publicImage_functional_of_accumulator_fields

abbrev terminal_soundness_of_concrete_program_and_msis :=
  @DirectProgramStep.terminal_soundness_of_concrete_program_and_msis

end DirectProgramStepInterface

end DirectCcsFPrime
