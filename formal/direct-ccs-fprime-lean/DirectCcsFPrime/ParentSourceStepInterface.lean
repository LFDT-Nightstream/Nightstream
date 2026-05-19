import DirectCcsFPrime.ParentSourceStep

/-!
Typed interface for parent-source derivation.

Spec: `specs/ParentSourceStep.spec.md`
-/

namespace DirectCcsFPrime

namespace ParentSourceStepInterface

abbrev Step :=
  @ParentSourceStep.Step

abbrev PiCCSFunctional :=
  @ParentSourceStep.PiCCSFunctional

abbrev PiRLCFunctional :=
  @ParentSourceStep.PiRLCFunctional

abbrev functional_of_stage_functional :=
  @ParentSourceStep.functional_of_stage_functional

abbrev ComputedPiCCS :=
  @ParentSourceStep.ComputedPiCCS

abbrev ComputedPiRLC :=
  @ParentSourceStep.ComputedPiRLC

abbrev computedPiCCS_functional :=
  @ParentSourceStep.computedPiCCS_functional

abbrev computedPiRLC_functional :=
  @ParentSourceStep.computedPiRLC_functional

abbrev functional_of_computed_stages :=
  @ParentSourceStep.functional_of_computed_stages

abbrev transition_accumulator_fields_functional_of_stage_functional :=
  @ParentSourceStep.transition_accumulator_fields_functional_of_stage_functional

abbrev transition_accumulator_fields_functional_of_stages_and_ajtaiCEOpening :=
  @ParentSourceStep.transition_accumulator_fields_functional_of_stages_and_ajtaiCEOpening

abbrev transition_accumulator_fields_functional_of_stages_statementCommitment_and_ajtaiCEOpening :=
  @ParentSourceStep.transition_accumulator_fields_functional_of_stages_statementCommitment_and_ajtaiCEOpening

end ParentSourceStepInterface

end DirectCcsFPrime
