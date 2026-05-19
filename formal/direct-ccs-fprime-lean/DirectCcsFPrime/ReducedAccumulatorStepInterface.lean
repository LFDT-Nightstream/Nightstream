import DirectCcsFPrime.ReducedAccumulatorStep

/-!
Typed interface for the reduced-handle accumulator step.

Spec: `specs/ReducedAccumulatorStep.spec.md`
-/

namespace DirectCcsFPrime

namespace ReducedAccumulatorStepInterface

abbrev AccumulatorHandle :=
  @ReducedAccumulatorStep.AccumulatorHandle

abbrev AuthorizedFunctional :=
  @ReducedAccumulatorStep.AuthorizedFunctional

abbrev ParentSourceFunctional :=
  @ReducedAccumulatorStep.ParentSourceFunctional

abbrev Step :=
  @ReducedAccumulatorStep.Step

abbrev step_fields_functional :=
  @ReducedAccumulatorStep.step_fields_functional

abbrev transition_accumulator_fields_functional :=
  @ReducedAccumulatorStep.transition_accumulator_fields_functional

abbrev canonical_authorized_functional_of_ajtaiCEOpening :=
  @ReducedAccumulatorStep.canonical_authorized_functional_of_ajtaiCEOpening

abbrev canonical_authorized_functional_of_statementCommitment_and_ajtaiCEOpening :=
  @ReducedAccumulatorStep.canonical_authorized_functional_of_statementCommitment_and_ajtaiCEOpening

abbrev transition_accumulator_fields_functional_of_ajtaiCEOpening :=
  @ReducedAccumulatorStep.transition_accumulator_fields_functional_of_ajtaiCEOpening

abbrev transition_accumulator_fields_functional_of_statementCommitment_and_ajtaiCEOpening :=
  @ReducedAccumulatorStep.transition_accumulator_fields_functional_of_statementCommitment_and_ajtaiCEOpening

end ReducedAccumulatorStepInterface

end DirectCcsFPrime
