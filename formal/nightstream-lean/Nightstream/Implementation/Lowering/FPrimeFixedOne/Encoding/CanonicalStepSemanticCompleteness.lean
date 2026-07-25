import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepCrossArmSeparation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepVisibleCompleteness

/-!
Contract: semantic honest-occurrence witnesses for canonical Step completion.

Owns:
- accepted base and recursive branch consequences;
- exact honest-active data for every canonical Step occurrence;
- honest-inactive data for either unselected private arm.

Does not own: temporary completion, cross-group separation, activation or mux
rows, physical receipt order, Rust behavior, or generated artifacts.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Step

namespace CanonicalStepCompleteness

/-- Acceptance on the base branch forces the initial-state check and fixes
the public result to the verifier-owned default running value. -/
theorem baseAccepted
    (parameters : Parameters)
    (input : StepInputFor parameters)
    (output : StepOutputFor parameters)
    (accepted : Accepts parameters input output)
    (iterationZero : input.iteration = 0) :
    stateEqual parameters input.z0 input.zi = true ∧
      output =
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
          parameters.setup parameters.machine input
          (fun _ => defaultRunning parameters) := by
  have fixed :
      fixedOneEval parameters input = some output :=
    (accepts_iff_fixedOne parameters input output).1 accepted
  unfold fixedOneEval at fixed
  by_cases initialState : input.z0 = input.zi
  · have outputEqual :
        output =
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
            parameters.setup parameters.machine input
            (fun _ => defaultRunning parameters) := by
      simpa [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
        iterationZero, initialState] using fixed.symm
    exact ⟨by simp [stateEqual, initialState], outputEqual⟩
  · simp [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
      iterationZero, initialState] at fixed

/-- Acceptance on the recursive branch exposes the exact successful NIFS
result, prior-link check, and public output selected by that result. -/
theorem recursiveAccepted
    (parameters : Parameters)
    (input : StepInputFor parameters)
    (output : StepOutputFor parameters)
    (accepted : Accepts parameters input output)
    (iterationNonzero : ¬ input.iteration = 0) :
    ∃ folded : parameters.Running,
      priorLinkAccepted parameters input = true ∧
        parameters.setup.nifs.verify
            (parameters.setup.verifierKeys Vocabulary.Step.selected)
            (input.running Vocabulary.Step.selected)
            input.fresh input.nifsProof =
          some folded ∧
        output =
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
            parameters.setup parameters.machine input (fun _ => folded) := by
  have fixed :
      fixedOneEval parameters input = some output :=
    (accepts_iff_fixedOne parameters input output).1 accepted
  unfold fixedOneEval at fixed
  by_cases priorPublic :
      parameters.machine.freshPublic input.fresh =
        parameters.machine.encodeInstance
          (parameters.machine.hash
            (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
              parameters.setup input))
  · cases verifierResult :
      parameters.setup.nifs.verify
        (parameters.setup.verifierKeys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
        (input.running
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
        input.fresh input.nifsProof with
    | none =>
        simp [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
          iterationNonzero, priorPublic, verifierResult] at fixed
    | some folded =>
        have verifierResultStep :
            parameters.setup.nifs.verify
                (parameters.setup.verifierKeys Vocabulary.Step.selected)
                (input.running Vocabulary.Step.selected)
                input.fresh input.nifsProof =
              some folded := by
          simpa only [step_selected_eq_canonical] using verifierResult
        have outputEqual :
            output =
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
                parameters.setup parameters.machine input
                (fun _ => folded) := by
          simpa [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
            iterationNonzero, priorPublic, verifierResult] using fixed.symm
        exact ⟨folded,
          (priorLinkAccepted_eq_true_iff parameters input).2 priorPublic,
          verifierResultStep, outputEqual⟩
  · simp [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
      iterationNonzero] at fixed
    exact (priorPublic (by
      simpa only
        [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage_eq_generic]
        using fixed.1)).elim

theorem baseSelectedRunning
    (parameters : Parameters)
    (input : StepInputFor parameters)
    (output : StepOutputFor parameters)
    (accepted : Accepts parameters input output)
    (iterationZero : input.iteration = 0) :
    selectedRunning output = defaultRunning parameters := by
  have branch :=
    baseAccepted parameters input output accepted iterationZero
  rw [branch.2]
  rfl

theorem recursiveSelectedRunning
    (parameters : Parameters)
    (input : StepInputFor parameters)
    (output : StepOutputFor parameters)
    (accepted : Accepts parameters input output)
    (iterationNonzero : ¬ input.iteration = 0) :
    priorLinkAccepted parameters input = true ∧
      parameters.setup.nifs.verify
          (parameters.setup.verifierKeys Vocabulary.Step.selected)
          (input.running Vocabulary.Step.selected)
          input.fresh input.nifsProof =
        some (selectedRunning output) := by
  rcases recursiveAccepted parameters input output accepted
      iterationNonzero with
    ⟨folded, prior, verifier, outputEqual⟩
  have selectedEqual : selectedRunning output = folded := by
    rw [outputEqual]
    rfl
  exact ⟨prior, selectedEqual ▸ verifier⟩

theorem acceptedResultValues
    (parameters : Parameters)
    (input : StepInputFor parameters)
    (output : StepOutputFor parameters)
    (accepted : Accepts parameters input output) :
    resultValuesFor parameters input (selectedRunning output) =
      stepResultValues parameters output := by
  by_cases iterationZero : input.iteration = 0
  · have branch :=
      baseAccepted parameters input output accepted iterationZero
    rw [branch.2]
    rfl
  · rcases recursiveAccepted parameters input output accepted
        iterationZero with
      ⟨folded, _, _, outputEqual⟩
    rw [outputEqual]
    rfl

/-- The common prefix and post-join continuation are honest for every exact
visible Step witness. -/
theorem alwaysHonest
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (input : StepInputFor parameters)
    (runningNext : parameters.Running)
    (visible :
      VisibleWitness parameters profile input runningNext) :
    (CanonicalCompletionPlans.Step.always
      parameters profile recipes).HonestActive visible.assignment := by
  let applyPlan :=
    CanonicalStepPlan.applyPlan parameters profile recipes
  let selector :=
    CanonicalStepPlan.selectorPlan parameters profile recipes
  let continuation :=
    CanonicalStepPlan.continuationHashPlan parameters profile recipes

  have applyResultEncoded :
      Columns.Encodes (profile.family parameters)
        applyPlan.resultColumns visible.assignment
        (afterStepValues parameters input) := by
    simpa [applyPlan, CanonicalStepPlan.applyPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterStep] using
      visible.afterStepEncoded
  have applyHonest :=
    applyPlan.honestActive visible.assignment
      (stepInputValues parameters input)
      (afterStepValues parameters input)
      visible.inputEncoded applyResultEncoded
      (((stepCall parameters).exec_eq_some_iff_holds _ _).1
        (stepCall_exec parameters input))

  have selectorResultEncoded :
      Columns.Encodes (profile.family parameters)
        selector.resultColumns visible.assignment
        (commonValues parameters input) := by
    simpa [selector, CanonicalStepPlan.selectorPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.common] using
      visible.commonEncoded
  have selectorHonest :=
    selector.honestActive visible.assignment
      (afterStepValues parameters input)
      (commonValues parameters input)
      visible.afterStepEncoded selectorResultEncoded
      (((iterationZeroCall parameters).exec_eq_some_iff_holds _ _).1
        (iterationZeroCall_exec parameters input))

  have continuationResultEncoded :
      Columns.Encodes (profile.family parameters)
        continuation.resultColumns visible.assignment
        (afterHashNextValues parameters input runningNext) := by
    simpa [continuation, CanonicalStepPlan.continuationHashPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterHashNext] using
      visible.finalEncoded
  have continuationHonest :=
    continuation.honestActive visible.assignment
      (continuationInputValues parameters input runningNext)
      (afterHashNextValues parameters input runningNext)
      visible.continuationInputEncoded continuationResultEncoded
      (((hashNextCall parameters).exec_eq_some_iff_holds _ _).1
        (hashNextCall_exec parameters input runningNext))

  change ArmPlan.HonestActiveOccurrences visible.assignment
    [applyPlan.occurrence, selector.occurrence,
      continuation.occurrence]
  exact ⟨applyHonest, selectorHonest, continuationHonest, True.intro⟩

/-- The selected base arm honestly proves the initial-state equality and
installs the verifier-owned default running claim. -/
theorem baseHonestActive
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (input : StepInputFor parameters)
    (runningNext : parameters.Running)
    (visible :
      VisibleWitness parameters profile input runningNext)
    (condition : stateEqual parameters input.z0 input.zi = true) :
    (CanonicalCompletionPlans.Step.onTrue
      parameters profile recipes defaultAdmissible
      ).HonestActive visible.assignment := by
  let equality :=
    CanonicalStepPlan.baseEqualityPlan parameters profile recipes
  let assertion :=
    CanonicalStepPlan.baseAssertionPlan parameters profile
  let literal :=
    CanonicalStepPlan.baseLiteralPlan
      parameters profile defaultAdmissible

  have equalityResultEncoded :
      Columns.Encodes (profile.family parameters)
        equality.resultColumns visible.assignment
        (afterBaseEqualityValues parameters input) := by
    simpa [equality, CanonicalStepPlan.baseEqualityPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterBaseEquality] using
      visible.afterBaseEqualityEncoded
  have equalityHonest :=
    equality.honestActive visible.assignment
      (commonValues parameters input)
      (afterBaseEqualityValues parameters input)
      visible.commonEncoded equalityResultEncoded
      (((baseStateEqualCall parameters).exec_eq_some_iff_holds _ _).1
        (baseStateEqualCall_exec parameters input))

  have assertionResultEncoded :
      Columns.Encodes (profile.family parameters)
        assertion.resultColumns visible.assignment
        (afterBaseEqualityValues parameters input) := by
    simpa [assertion, CanonicalStepPlan.baseAssertionPlan,
      PrimitivePlan.resultColumns] using
      visible.afterBaseEqualityEncoded
  have assertionHonest :=
    assertion.honestActive visible.assignment
      (afterBaseEqualityValues parameters input)
      (afterBaseEqualityValues parameters input)
      visible.afterBaseEqualityEncoded assertionResultEncoded
      (by exact ⟨condition, rfl⟩)

  have literalResultEncoded :
      Columns.Encodes (profile.family parameters)
        literal.resultColumns visible.assignment
        (afterBaseLiteralValues parameters input) := by
    simpa [literal, CanonicalStepPlan.baseLiteralPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterBaseLiteral] using
      visible.baseEncoded
  have literalHonest :=
    literal.honestActive visible.assignment
      (afterBaseEqualityValues parameters input)
      (afterBaseLiteralValues parameters input)
      visible.afterBaseEqualityEncoded literalResultEncoded
      (((baseDefaultCall parameters).exec_eq_some_iff_holds _ _).1
        (baseDefaultCall_exec parameters input))

  change ArmPlan.HonestActiveOccurrences visible.assignment
    [equality.occurrence, assertion.occurrence, literal.occurrence]
  exact ⟨equalityHonest, assertionHonest, literalHonest, True.intro⟩

/-- The selected recursive arm honestly enforces the public link and returns
the exact successful NIFS running value. -/
theorem recursiveHonestActive
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (input : StepInputFor parameters)
    (runningNext : parameters.Running)
    (visible :
      VisibleWitness parameters profile input runningNext)
    (conditions :
      priorLinkAccepted parameters input = true ∧
        parameters.setup.nifs.verify
            (parameters.setup.verifierKeys Vocabulary.Step.selected)
            (input.running Vocabulary.Step.selected)
            input.fresh input.nifsProof =
          some runningNext) :
    (CanonicalCompletionPlans.Step.onFalse
      parameters profile recipes).HonestActive visible.assignment := by
  let hash :=
    CanonicalStepPlan.recursiveHashPlan parameters profile recipes
  let fresh :=
    CanonicalStepPlan.recursiveFreshPublicPlan
      parameters profile recipes
  let encode :=
    CanonicalStepPlan.recursiveEncodePlan parameters profile recipes
  let equality :=
    CanonicalStepPlan.recursiveEncodedEqualityPlan
      parameters profile recipes
  let assertion :=
    CanonicalStepPlan.recursiveAssertionPlan parameters profile
  let nifs :=
    CanonicalStepPlan.recursiveNifsPlan parameters profile recipes

  have hashResultEncoded :
      Columns.Encodes (profile.family parameters)
        hash.resultColumns visible.assignment
        (afterHashValues parameters input) := by
    simpa [hash, CanonicalStepPlan.recursiveHashPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterHash] using
      visible.afterHashEncoded
  have hashHonest :=
    hash.honestActive visible.assignment
      (commonValues parameters input)
      (afterHashValues parameters input)
      visible.commonEncoded hashResultEncoded
      (((hashPriorCall parameters).exec_eq_some_iff_holds _ _).1
        (hashPriorCall_exec parameters input))

  have freshResultEncoded :
      Columns.Encodes (profile.family parameters)
        fresh.resultColumns visible.assignment
        (afterFreshPublicValues parameters input) := by
    simpa [fresh, CanonicalStepPlan.recursiveFreshPublicPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterFreshPublic] using
      visible.afterFreshPublicEncoded
  have freshHonest :=
    fresh.honestActive visible.assignment
      (afterHashValues parameters input)
      (afterFreshPublicValues parameters input)
      visible.afterHashEncoded freshResultEncoded
      (((freshPublicCall parameters).exec_eq_some_iff_holds _ _).1
        (freshPublicCall_exec parameters input))

  have encodeResultEncoded :
      Columns.Encodes (profile.family parameters)
        encode.resultColumns visible.assignment
        (afterEncodeValues parameters input) := by
    simpa [encode, CanonicalStepPlan.recursiveEncodePlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterEncode] using
      visible.afterEncodeEncoded
  have encodeHonest :=
    encode.honestActive visible.assignment
      (afterFreshPublicValues parameters input)
      (afterEncodeValues parameters input)
      visible.afterFreshPublicEncoded encodeResultEncoded
      (((encodeInstanceCall parameters).exec_eq_some_iff_holds _ _).1
        (encodeInstanceCall_exec parameters input))

  have equalityResultEncoded :
      Columns.Encodes (profile.family parameters)
        equality.resultColumns visible.assignment
        (afterEncodedEqualityValues parameters input) := by
    simpa [equality,
      CanonicalStepPlan.recursiveEncodedEqualityPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterEncodedEquality] using
      visible.afterEncodedEqualityEncoded
  have equalityHonest :=
    equality.honestActive visible.assignment
      (afterEncodeValues parameters input)
      (afterEncodedEqualityValues parameters input)
      visible.afterEncodeEncoded equalityResultEncoded
      (((encodedEqualCall parameters).exec_eq_some_iff_holds _ _).1
        (encodedEqualCall_exec parameters input))

  have assertionResultEncoded :
      Columns.Encodes (profile.family parameters)
        assertion.resultColumns visible.assignment
        (afterEncodedEqualityValues parameters input) := by
    simpa [assertion, CanonicalStepPlan.recursiveAssertionPlan,
      PrimitivePlan.resultColumns] using
      visible.afterEncodedEqualityEncoded
  have assertionHonest :=
    assertion.honestActive visible.assignment
      (afterEncodedEqualityValues parameters input)
      (afterEncodedEqualityValues parameters input)
      visible.afterEncodedEqualityEncoded assertionResultEncoded
      (by exact ⟨conditions.1, rfl⟩)

  have nifsResultEncoded :
      Columns.Encodes (profile.family parameters)
        nifs.resultColumns visible.assignment
        (afterNifsValues parameters input runningNext) := by
    simpa [nifs, CanonicalStepPlan.recursiveNifsPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterNifs] using
      visible.recursiveEncoded
  have nifsExecuted :
      (nifsVerifyCall parameters).exec
          (afterEncodedEqualityValues parameters input) =
        some (afterNifsValues parameters input runningNext) := by
    rw [nifsVerifyCall_exec parameters input]
    rw [conditions.2]
    rfl
  have nifsHonest :=
    nifs.honestActive visible.assignment
      (afterEncodedEqualityValues parameters input)
      (afterNifsValues parameters input runningNext)
      visible.afterEncodedEqualityEncoded nifsResultEncoded
      (((nifsVerifyCall parameters).exec_eq_some_iff_holds _ _).1
        nifsExecuted)

  change ArmPlan.HonestActiveOccurrences visible.assignment
    [hash.occurrence, fresh.occurrence, encode.occurrence,
      equality.occurrence, assertion.occurrence, nifs.occurrence]
  exact ⟨hashHonest, freshHonest, encodeHonest, equalityHonest,
    assertionHonest, nifsHonest, True.intro⟩

/-- An unselected base arm is satisfiable once its verifier-owned default
literal is present in the shared visible assignment. -/
theorem baseHonestInactive
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (input : StepInputFor parameters)
    (runningNext : parameters.Running)
    (visible : VisibleWitness parameters profile input runningNext) :
    (CanonicalCompletionPlans.Step.onTrue
      parameters profile recipes defaultAdmissible
      ).HonestInactive visible.assignment := by
  let equality :=
    CanonicalStepPlan.baseEqualityPlan parameters profile recipes
  let assertion :=
    CanonicalStepPlan.baseAssertionPlan parameters profile
  let literal :=
    CanonicalStepPlan.baseLiteralPlan
      parameters profile defaultAdmissible
  have literalEncoded :
      Columns.Encodes (profile.family parameters)
        literal.resultColumns visible.assignment
        (afterBaseLiteralValues parameters input) := by
    simpa [literal, CanonicalStepPlan.baseLiteralPlan,
      PrimitivePlan.resultColumns] using visible.baseEncoded
  have literalVisible : literal.InactiveVisible visible.assignment := by
    simp only [literal, CanonicalStepPlan.baseLiteralPlan,
      PrimitivePlan.InactiveVisible]
    exact ColumnBundle.decodes_of_encodes
      (profile.family parameters) _ _ visible.assignment _
      literalEncoded.1
  change ArmPlan.HonestInactiveOccurrences visible.assignment
    [equality.occurrence, assertion.occurrence, literal.occurrence]
  exact ⟨
    equality.honestInactive visible.assignment True.intro,
    assertion.honestInactive visible.assignment True.intro,
    literal.honestInactive visible.assignment literalVisible,
    True.intro⟩

/-- An unselected recursive arm is satisfiable independently of its visible
data and of NIFS acceptance. -/
theorem recursiveHonestInactive
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (assignment : ColumnId -> Field) :
    (CanonicalCompletionPlans.Step.onFalse
      parameters profile recipes).HonestInactive assignment := by
  let hash :=
    CanonicalStepPlan.recursiveHashPlan parameters profile recipes
  let fresh :=
    CanonicalStepPlan.recursiveFreshPublicPlan
      parameters profile recipes
  let encode :=
    CanonicalStepPlan.recursiveEncodePlan parameters profile recipes
  let equality :=
    CanonicalStepPlan.recursiveEncodedEqualityPlan
      parameters profile recipes
  let assertion :=
    CanonicalStepPlan.recursiveAssertionPlan parameters profile
  let nifs :=
    CanonicalStepPlan.recursiveNifsPlan parameters profile recipes
  change ArmPlan.HonestInactiveOccurrences assignment
    [hash.occurrence, fresh.occurrence, encode.occurrence,
      equality.occurrence, assertion.occurrence, nifs.occurrence]
  exact ⟨
    hash.honestInactive assignment True.intro,
    fresh.honestInactive assignment True.intro,
    encode.honestInactive assignment True.intro,
    equality.honestInactive assignment True.intro,
    assertion.honestInactive assignment True.intro,
    nifs.honestInactive assignment True.intro,
    True.intro⟩

end CanonicalStepCompleteness

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
