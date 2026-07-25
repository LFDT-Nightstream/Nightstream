import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalCrossArmSeparation

/-!
Contract: separation of the always-active Terminal selector from both branch
arms.

Owns:
- selector-temporary protection through the exact base and recursive SSA
  contexts;
- both completion orders between the selector group and either branch arm.

Does not own: semantic witnesses, assignments, branch rows, production Rust
behavior, or R1CS indices.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalCompletionPlans.Terminal

private theorem one_excludes_instruction (path : OwnerPath) :
    oneColumn.owner ≠ .typed (.instruction path) := by
  simp [oneColumn]

private theorem activation_excludes_instruction
    (target : OwnerPath)
    (selected : Bool) :
    (activationColumn SourceOwners.terminalBranchPath selected).owner ≠
      .typed (.instruction target) := by
  simp [activationColumn]

private theorem selector_pair
    {parameters : Parameters}
    {profile : Profile parameters}
    {secondInput secondOutput :
      Schema (typeSystem parameters)}
    {secondPrimitive :
      Primitive (SelectedSignature parameters)
        secondInput secondOutput}
    {secondPath : OwnerPath}
    {secondInputColumns : Columns secondInput}
    {selected : Bool}
    {selector :
      PrimitivePlan parameters profile
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.iterationZeroCall
          parameters)
        SourceOwners.terminalSelectorPath
        (CanonicalContexts.Terminal.input parameters)
        oneColumn oneColumn}
    (protection :
      selector.ProtectedExtension secondInputColumns)
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns oneColumn
        (activationColumn SourceOwners.terminalBranchPath selected))
    (different : SourceOwners.terminalSelectorPath ≠ secondPath) :
    IdsDisjoint selector.occurrence.temporaryIds
        second.occurrence.visibleIds ∧
      IdsDisjoint second.occurrence.temporaryIds
        selector.occurrence.visibleIds ∧
      IdsDisjoint selector.occurrence.temporaryIds
        second.occurrence.temporaryIds :=
  protection.crossPairwiseSeparated second
    (one_excludes_instruction SourceOwners.terminalSelectorPath)
    (activation_excludes_instruction
      SourceOwners.terminalSelectorPath selected)
    (one_excludes_instruction secondPath)
    (one_excludes_instruction secondPath)
    different
    (CanonicalPrimitivePlan.ContextExcludesOwner.input
      (terminalInputSchema parameters) secondPath)

/-- The Terminal selector group and both branch arms are mutually separated
in every completion order used by the honest assignment construction. -/
theorem always_arms_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (ArmPlan.PlansSeparated
        (always parameters profile recipes)
        (onTrue parameters profile recipes) ∧
      ArmPlan.PlansSeparated
        (onTrue parameters profile recipes)
        (always parameters profile recipes)) ∧
      (ArmPlan.PlansSeparated
        (always parameters profile recipes)
        (onFalse parameters profile recipes) ∧
      ArmPlan.PlansSeparated
        (onFalse parameters profile recipes)
        (always parameters profile recipes)) := by
  let selector :=
    CanonicalTerminalPlan.selectorPlan.{0}
      parameters profile recipes
  let baseEquality :=
    CanonicalTerminalPlan.baseEqualityPlan.{0}
      parameters profile recipes
  let baseAssertion :=
    CanonicalTerminalPlan.baseAssertionPlan.{0}
      parameters profile
  let hash :=
    CanonicalTerminalPlan.recursiveHashPlan.{0}
      parameters profile recipes
  let fresh :=
    CanonicalTerminalPlan.recursiveFreshPublicPlan.{0}
      parameters profile recipes
  let encode :=
    CanonicalTerminalPlan.recursiveEncodePlan.{0}
      parameters profile recipes
  let equality :=
    CanonicalTerminalPlan.recursiveEncodedEqualityPlan.{0}
      parameters profile recipes
  let priorAssertion :=
    CanonicalTerminalPlan.recursivePriorAssertionPlan.{0}
      parameters profile
  let runningCheck :=
    CanonicalTerminalPlan.recursiveRunningCheckPlan.{0}
      parameters profile recipes
  let runningAssertion :=
    CanonicalTerminalPlan.recursiveRunningAssertionPlan.{0}
      parameters profile
  let freshCheck :=
    CanonicalTerminalPlan.recursiveFreshCheckPlan.{0}
      parameters profile recipes
  let freshAssertion :=
    CanonicalTerminalPlan.recursiveFreshAssertionPlan.{0}
      parameters profile
  let alwaysPlan := always parameters profile recipes
  let basePlan := onTrue parameters profile recipes
  let recursivePlan := onFalse parameters profile recipes

  have selectorBase := selector.protectsResult
  have selectorAfterBase :=
    selectorBase.extend baseEquality (by decide)

  have selectorHash := selector.protectsResult
  have selectorFresh :=
    selectorHash.extend hash (by decide)
  have selectorEncode :=
    selectorFresh.extend fresh (by decide)
  have selectorEquality :=
    selectorEncode.extend encode (by decide)
  have selectorAfterEquality :=
    selectorEquality.extend equality (by decide)
  have selectorAfterRunning :=
    selectorAfterEquality.extend runningCheck (by decide)
  have selectorAfterFresh :=
    selectorAfterRunning.extend freshCheck (by decide)

  have basePairwise :
      ∀ firstOccurrence,
        firstOccurrence ∈ alwaysPlan.occurrences ->
      ∀ secondOccurrence,
        secondOccurrence ∈ basePlan.occurrences ->
        IdsDisjoint firstOccurrence.temporaryIds
            secondOccurrence.visibleIds ∧
          IdsDisjoint secondOccurrence.temporaryIds
            firstOccurrence.visibleIds ∧
          IdsDisjoint firstOccurrence.temporaryIds
            secondOccurrence.temporaryIds := by
    intro firstOccurrence firstMember
      secondOccurrence secondMember
    change firstOccurrence ∈ [selector.occurrence] at firstMember
    change secondOccurrence ∈
      [baseEquality.occurrence, baseAssertion.occurrence] at secondMember
    simp only [List.mem_singleton] at firstMember
    subst firstOccurrence
    simp only [List.mem_cons, List.not_mem_nil, or_false] at secondMember
    rcases secondMember with equal | equal
    · subst secondOccurrence
      exact selector_pair selectorBase baseEquality (by decide)
    · subst secondOccurrence
      exact selector_pair selectorAfterBase baseAssertion (by decide)

  have recursivePairwise :
      ∀ firstOccurrence,
        firstOccurrence ∈ alwaysPlan.occurrences ->
      ∀ secondOccurrence,
        secondOccurrence ∈ recursivePlan.occurrences ->
        IdsDisjoint firstOccurrence.temporaryIds
            secondOccurrence.visibleIds ∧
          IdsDisjoint secondOccurrence.temporaryIds
            firstOccurrence.visibleIds ∧
          IdsDisjoint firstOccurrence.temporaryIds
            secondOccurrence.temporaryIds := by
    intro firstOccurrence firstMember
      secondOccurrence secondMember
    change firstOccurrence ∈ [selector.occurrence] at firstMember
    change secondOccurrence ∈
      [hash.occurrence, fresh.occurrence, encode.occurrence,
        equality.occurrence, priorAssertion.occurrence,
        runningCheck.occurrence, runningAssertion.occurrence,
        freshCheck.occurrence, freshAssertion.occurrence] at secondMember
    simp only [List.mem_singleton] at firstMember
    subst firstOccurrence
    simp only [List.mem_cons, List.not_mem_nil, or_false] at secondMember
    rcases secondMember with
      equal | equal | equal | equal | equal |
        equal | equal | equal | equal
    · subst secondOccurrence
      exact selector_pair selectorHash hash (by decide)
    · subst secondOccurrence
      exact selector_pair selectorFresh fresh (by decide)
    · subst secondOccurrence
      exact selector_pair selectorEncode encode (by decide)
    · subst secondOccurrence
      exact selector_pair selectorEquality equality (by decide)
    · subst secondOccurrence
      exact selector_pair selectorAfterEquality priorAssertion (by decide)
    · subst secondOccurrence
      exact selector_pair selectorAfterEquality runningCheck (by decide)
    · subst secondOccurrence
      exact selector_pair selectorAfterRunning runningAssertion (by decide)
    · subst secondOccurrence
      exact selector_pair selectorAfterRunning freshCheck (by decide)
    · subst secondOccurrence
      exact selector_pair selectorAfterFresh freshAssertion (by decide)

  have selectorBaseControl :
      IdsDisjoint alwaysPlan.temporaryIds
        [oneColumn,
          activationColumn SourceOwners.terminalBranchPath true] := by
    intro id temporaryMember controlMember
    change id ∈ [selector.occurrence].flatMap
      ArmOccurrence.temporaryIds at temporaryMember
    rcases List.mem_flatMap.mp temporaryMember with
      ⟨occurrence, occurrenceMember, occurrenceTemporary⟩
    simp only [List.mem_singleton] at occurrenceMember
    subst occurrence
    exact
      (selector.occurrenceTemporariesDisjointControls
        oneColumn
        (activationColumn SourceOwners.terminalBranchPath true)
        (one_excludes_instruction
          SourceOwners.terminalSelectorPath)
        (activation_excludes_instruction
          SourceOwners.terminalSelectorPath true))
        id occurrenceTemporary controlMember

  have selectorRecursiveControl :
      IdsDisjoint alwaysPlan.temporaryIds
        [oneColumn,
          activationColumn SourceOwners.terminalBranchPath false] := by
    intro id temporaryMember controlMember
    change id ∈ [selector.occurrence].flatMap
      ArmOccurrence.temporaryIds at temporaryMember
    rcases List.mem_flatMap.mp temporaryMember with
      ⟨occurrence, occurrenceMember, occurrenceTemporary⟩
    simp only [List.mem_singleton] at occurrenceMember
    subst occurrence
    exact
      (selector.occurrenceTemporariesDisjointControls
        oneColumn
        (activationColumn SourceOwners.terminalBranchPath false)
        (one_excludes_instruction
          SourceOwners.terminalSelectorPath)
        (activation_excludes_instruction
          SourceOwners.terminalSelectorPath false))
        id occurrenceTemporary controlMember

  have baseAlwaysControl :
      IdsDisjoint basePlan.temporaryIds [oneColumn, oneColumn] := by
    intro id temporaryMember controlMember
    change id ∈
      [baseEquality.occurrence, baseAssertion.occurrence].flatMap
        ArmOccurrence.temporaryIds at temporaryMember
    rcases List.mem_flatMap.mp temporaryMember with
      ⟨occurrence, occurrenceMember, occurrenceTemporary⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false]
      at occurrenceMember
    rcases occurrenceMember with equal | equal
    · subst occurrence
      exact
        (baseEquality.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalBaseStateEqualPath)
          (one_excludes_instruction
            SourceOwners.terminalBaseStateEqualPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (baseAssertion.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalBaseAssertionPath)
          (one_excludes_instruction
            SourceOwners.terminalBaseAssertionPath))
          id occurrenceTemporary controlMember

  have recursiveAlwaysControl :
      IdsDisjoint recursivePlan.temporaryIds [oneColumn, oneColumn] := by
    intro id temporaryMember controlMember
    change id ∈
      [hash.occurrence, fresh.occurrence, encode.occurrence,
        equality.occurrence, priorAssertion.occurrence,
        runningCheck.occurrence, runningAssertion.occurrence,
        freshCheck.occurrence, freshAssertion.occurrence].flatMap
          ArmOccurrence.temporaryIds at temporaryMember
    rcases List.mem_flatMap.mp temporaryMember with
      ⟨occurrence, occurrenceMember, occurrenceTemporary⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false]
      at occurrenceMember
    rcases occurrenceMember with
      equal | equal | equal | equal | equal |
        equal | equal | equal | equal
    · subst occurrence
      exact
        (hash.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (fresh.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalRecursiveFreshPublicPath)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveFreshPublicPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (encode.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalRecursiveEncodePath)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveEncodePath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (equality.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalRecursiveEncodedEqualPath)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveEncodedEqualPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (priorAssertion.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalRecursivePriorAssertionPath)
          (one_excludes_instruction
            SourceOwners.terminalRecursivePriorAssertionPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (runningCheck.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalRecursiveRunningCheckPath)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveRunningCheckPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (runningAssertion.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalRecursiveRunningAssertionPath)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveRunningAssertionPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (freshCheck.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalRecursiveFreshCheckPath)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveFreshCheckPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (freshAssertion.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (one_excludes_instruction
            SourceOwners.terminalRecursiveFreshAssertionPath)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveFreshAssertionPath))
          id occurrenceTemporary controlMember

  exact ⟨
    ⟨
      CompletionSeparation.plansSeparated_of_pairwise
        alwaysPlan basePlan basePairwise selectorBaseControl,
      CompletionSeparation.plansSeparated_reverse_of_pairwise
        alwaysPlan basePlan basePairwise baseAlwaysControl⟩,
    ⟨
      CompletionSeparation.plansSeparated_of_pairwise
        alwaysPlan recursivePlan recursivePairwise
        selectorRecursiveControl,
      CompletionSeparation.plansSeparated_reverse_of_pairwise
        alwaysPlan recursivePlan recursivePairwise
        recursiveAlwaysControl⟩⟩

end CanonicalCompletionPlans.Terminal

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
