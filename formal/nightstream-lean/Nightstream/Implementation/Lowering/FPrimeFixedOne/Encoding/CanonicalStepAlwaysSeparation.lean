import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepCrossArmSeparation

/-!
Contract: separation of the always-active Step prefix and continuation from
both private branch arms.

Owns:
- protection of the apply and selector temporaries through each arm's exact
  SSA contexts;
- separation of the post-join continuation from each pre-join occurrence;
- both completion orders between the always group and either private arm.

Does not own: semantic witnesses, assignments, branch-control or mux rows,
production Rust behavior, or numeric R1CS indices.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalCompletionPlans.Step

private theorem protected_pair
    {parameters : Parameters}
    {profile : Profile parameters}
    {firstInput firstOutput secondInput secondOutput :
      Schema (typeSystem parameters)}
    {firstPrimitive :
      Primitive (SelectedSignature parameters)
        firstInput firstOutput}
    {secondPrimitive :
      Primitive (SelectedSignature parameters)
        secondInput secondOutput}
    {firstPath secondPath : OwnerPath}
    {firstInputColumns : Columns firstInput}
    {secondInputColumns : Columns secondInput}
    {selected : Bool}
    {first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns oneColumn oneColumn}
    (protection : first.ProtectedExtension secondInputColumns)
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns oneColumn
        (activationColumn SourceOwners.stepBranchPath selected))
    (different : firstPath ≠ secondPath)
    (firstInputExcludesSecond :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction secondPath)) firstInputColumns) :
    IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.visibleIds ∧
      IdsDisjoint second.occurrence.temporaryIds
        first.occurrence.visibleIds ∧
      IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.temporaryIds :=
  protection.crossPairwiseSeparated second
    (CanonicalStepPlan.one_excludes_instruction firstPath)
    (CanonicalStepPlan.activation_excludes_instruction
      SourceOwners.stepBranchPath firstPath selected)
    (CanonicalStepPlan.one_excludes_instruction secondPath)
    (CanonicalStepPlan.one_excludes_instruction secondPath)
    different firstInputExcludesSecond

private theorem continuation_pair
    {parameters : Parameters}
    {profile : Profile parameters}
    {secondInput secondOutput : Schema (typeSystem parameters)}
    {secondPrimitive :
      Primitive (SelectedSignature parameters)
        secondInput secondOutput}
    {secondPath : OwnerPath}
    {secondInputColumns : Columns secondInput}
    {selected : Bool}
    (continuation :
      PrimitivePlan parameters profile
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.hashNextCall
          parameters)
        SourceOwners.stepContinuationHashPath
        (CanonicalContexts.Step.continuationInput parameters)
        oneColumn oneColumn)
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns oneColumn
        (activationColumn SourceOwners.stepBranchPath selected))
    (different :
      SourceOwners.stepContinuationHashPath ≠ secondPath)
    (selectorDifferent :
      SourceOwners.stepSelectorPath ≠ secondPath)
    (applyDifferent :
      SourceOwners.stepApplyPath ≠ secondPath)
    (secondInputExcludesContinuation :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction
          SourceOwners.stepContinuationHashPath))
        secondInputColumns) :
    IdsDisjoint continuation.occurrence.temporaryIds
        second.occurrence.visibleIds ∧
      IdsDisjoint second.occurrence.temporaryIds
        continuation.occurrence.visibleIds ∧
      IdsDisjoint continuation.occurrence.temporaryIds
        second.occurrence.temporaryIds :=
  continuation.crossPairwiseSeparated second
    (CanonicalStepPlan.one_excludes_instruction
      SourceOwners.stepContinuationHashPath)
    (CanonicalStepPlan.activation_excludes_instruction
      SourceOwners.stepBranchPath
      SourceOwners.stepContinuationHashPath selected)
    (CanonicalStepPlan.one_excludes_instruction secondPath)
    (CanonicalStepPlan.one_excludes_instruction secondPath)
    different
    (CanonicalStepPlan.continuationInput_excludes
      parameters secondPath selectorDifferent applyDifferent)
    secondInputExcludesContinuation

/-- The always-active Step group and both private arms are mutually separated
in every order used by constructive honest completion. -/
theorem always_arms_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    (ArmPlan.PlansSeparated
        (always parameters profile recipes)
        (onTrue parameters profile recipes defaultAdmissible) ∧
      ArmPlan.PlansSeparated
        (onTrue parameters profile recipes defaultAdmissible)
        (always parameters profile recipes)) ∧
      (ArmPlan.PlansSeparated
        (always parameters profile recipes)
        (onFalse parameters profile recipes) ∧
      ArmPlan.PlansSeparated
        (onFalse parameters profile recipes)
        (always parameters profile recipes)) := by
  let applyPlan :=
    CanonicalStepPlan.applyPlan.{0} parameters profile recipes
  let selector :=
    CanonicalStepPlan.selectorPlan.{0} parameters profile recipes
  let continuation :=
    CanonicalStepPlan.continuationHashPlan.{0}
      parameters profile recipes
  let baseEquality :=
    CanonicalStepPlan.baseEqualityPlan.{0}
      parameters profile recipes
  let baseAssertion :=
    CanonicalStepPlan.baseAssertionPlan.{0}
      parameters profile
  let baseLiteral :=
    CanonicalStepPlan.baseLiteralPlan.{0}
      parameters profile defaultAdmissible
  let hash :=
    CanonicalStepPlan.recursiveHashPlan.{0}
      parameters profile recipes
  let fresh :=
    CanonicalStepPlan.recursiveFreshPublicPlan.{0}
      parameters profile recipes
  let encode :=
    CanonicalStepPlan.recursiveEncodePlan.{0}
      parameters profile recipes
  let equality :=
    CanonicalStepPlan.recursiveEncodedEqualityPlan.{0}
      parameters profile recipes
  let assertion :=
    CanonicalStepPlan.recursiveAssertionPlan.{0}
      parameters profile
  let nifs :=
    CanonicalStepPlan.recursiveNifsPlan.{0}
      parameters profile recipes
  let alwaysPlan := always parameters profile recipes
  let basePlan :=
    onTrue parameters profile recipes defaultAdmissible
  let recursivePlan := onFalse parameters profile recipes

  have applyCommon :=
    applyPlan.protectsResult.extend selector (by decide)
  have selectorCommon := selector.protectsResult
  have applyAfterBaseEquality :=
    applyCommon.extend baseEquality (by decide)
  have selectorAfterBaseEquality :=
    selectorCommon.extend baseEquality (by decide)

  have applyAfterHash :=
    applyCommon.extend hash (by decide)
  have applyAfterFresh :=
    applyAfterHash.extend fresh (by decide)
  have applyAfterEncode :=
    applyAfterFresh.extend encode (by decide)
  have applyAfterEquality :=
    applyAfterEncode.extend equality (by decide)
  have selectorAfterHash :=
    selectorCommon.extend hash (by decide)
  have selectorAfterFresh :=
    selectorAfterHash.extend fresh (by decide)
  have selectorAfterEncode :=
    selectorAfterFresh.extend encode (by decide)
  have selectorAfterEquality :=
    selectorAfterEncode.extend equality (by decide)

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
    change firstOccurrence ∈
      [applyPlan.occurrence, selector.occurrence,
        continuation.occurrence] at firstMember
    change secondOccurrence ∈
      [baseEquality.occurrence, baseAssertion.occurrence,
        baseLiteral.occurrence] at secondMember
    simp only [List.mem_cons, List.not_mem_nil, or_false]
      at firstMember secondMember
    rcases firstMember with firstEqual | firstEqual | firstEqual
    · subst firstOccurrence
      rcases secondMember with
        secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact protected_pair applyCommon baseEquality (by decide)
          (CanonicalPrimitivePlan.ContextExcludesOwner.input
            (stepInputSchema parameters)
            SourceOwners.stepBaseStateEqualPath)
      · subst secondOccurrence
        exact protected_pair applyAfterBaseEquality baseAssertion
          (by decide)
          (CanonicalPrimitivePlan.ContextExcludesOwner.input
            (stepInputSchema parameters)
            SourceOwners.stepBaseAssertionPath)
      · subst secondOccurrence
        exact protected_pair applyAfterBaseEquality baseLiteral
          (by decide)
          (CanonicalPrimitivePlan.ContextExcludesOwner.input
            (stepInputSchema parameters)
            SourceOwners.stepBaseDefaultPath)
    · subst firstOccurrence
      rcases secondMember with
        secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact protected_pair selectorCommon baseEquality (by decide)
          (CanonicalStepPlan.afterStep_excludes parameters
            SourceOwners.stepBaseStateEqualPath (by decide))
      · subst secondOccurrence
        exact protected_pair selectorAfterBaseEquality baseAssertion
          (by decide)
          (CanonicalStepPlan.afterStep_excludes parameters
            SourceOwners.stepBaseAssertionPath (by decide))
      · subst secondOccurrence
        exact protected_pair selectorAfterBaseEquality baseLiteral
          (by decide)
          (CanonicalStepPlan.afterStep_excludes parameters
            SourceOwners.stepBaseDefaultPath (by decide))
    · subst firstOccurrence
      rcases secondMember with
        secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact continuation_pair continuation baseEquality
          (by decide) (by decide) (by decide)
          (CanonicalStepPlan.common_excludes parameters
            SourceOwners.stepContinuationHashPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact continuation_pair continuation baseAssertion
          (by decide) (by decide) (by decide)
          (CanonicalStepPlan.afterBaseEquality_excludes parameters
            SourceOwners.stepContinuationHashPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact continuation_pair continuation baseLiteral
          (by decide) (by decide) (by decide)
          (CanonicalStepPlan.afterBaseEquality_excludes parameters
            SourceOwners.stepContinuationHashPath
            (by decide) (by decide) (by decide))

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
    change firstOccurrence ∈
      [applyPlan.occurrence, selector.occurrence,
        continuation.occurrence] at firstMember
    change secondOccurrence ∈
      [hash.occurrence, fresh.occurrence, encode.occurrence,
        equality.occurrence, assertion.occurrence,
        nifs.occurrence] at secondMember
    simp only [List.mem_cons, List.not_mem_nil, or_false]
      at firstMember secondMember
    rcases firstMember with firstEqual | firstEqual | firstEqual
    · subst firstOccurrence
      rcases secondMember with
        secondEqual | secondEqual | secondEqual |
          secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact protected_pair applyCommon hash (by decide)
          (CanonicalPrimitivePlan.ContextExcludesOwner.input
            (stepInputSchema parameters)
            SourceOwners.stepRecursiveHashPriorPath)
      · subst secondOccurrence
        exact protected_pair applyAfterHash fresh (by decide)
          (CanonicalPrimitivePlan.ContextExcludesOwner.input
            (stepInputSchema parameters)
            SourceOwners.stepRecursiveFreshPublicPath)
      · subst secondOccurrence
        exact protected_pair applyAfterFresh encode (by decide)
          (CanonicalPrimitivePlan.ContextExcludesOwner.input
            (stepInputSchema parameters)
            SourceOwners.stepRecursiveEncodePath)
      · subst secondOccurrence
        exact protected_pair applyAfterEncode equality (by decide)
          (CanonicalPrimitivePlan.ContextExcludesOwner.input
            (stepInputSchema parameters)
            SourceOwners.stepRecursiveEncodedEqualPath)
      · subst secondOccurrence
        exact protected_pair applyAfterEquality assertion (by decide)
          (CanonicalPrimitivePlan.ContextExcludesOwner.input
            (stepInputSchema parameters)
            SourceOwners.stepRecursiveAssertionPath)
      · subst secondOccurrence
        exact protected_pair applyAfterEquality nifs (by decide)
          (CanonicalPrimitivePlan.ContextExcludesOwner.input
            (stepInputSchema parameters)
            SourceOwners.stepRecursiveNifsPath)
    · subst firstOccurrence
      rcases secondMember with
        secondEqual | secondEqual | secondEqual |
          secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact protected_pair selectorCommon hash (by decide)
          (CanonicalStepPlan.afterStep_excludes parameters
            SourceOwners.stepRecursiveHashPriorPath (by decide))
      · subst secondOccurrence
        exact protected_pair selectorAfterHash fresh (by decide)
          (CanonicalStepPlan.afterStep_excludes parameters
            SourceOwners.stepRecursiveFreshPublicPath (by decide))
      · subst secondOccurrence
        exact protected_pair selectorAfterFresh encode (by decide)
          (CanonicalStepPlan.afterStep_excludes parameters
            SourceOwners.stepRecursiveEncodePath (by decide))
      · subst secondOccurrence
        exact protected_pair selectorAfterEncode equality (by decide)
          (CanonicalStepPlan.afterStep_excludes parameters
            SourceOwners.stepRecursiveEncodedEqualPath (by decide))
      · subst secondOccurrence
        exact protected_pair selectorAfterEquality assertion
          (by decide)
          (CanonicalStepPlan.afterStep_excludes parameters
            SourceOwners.stepRecursiveAssertionPath (by decide))
      · subst secondOccurrence
        exact protected_pair selectorAfterEquality nifs (by decide)
          (CanonicalStepPlan.afterStep_excludes parameters
            SourceOwners.stepRecursiveNifsPath (by decide))
    · subst firstOccurrence
      rcases secondMember with
        secondEqual | secondEqual | secondEqual |
          secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact continuation_pair continuation hash
          (by decide) (by decide) (by decide)
          (CanonicalStepPlan.common_excludes parameters
            SourceOwners.stepContinuationHashPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact continuation_pair continuation fresh
          (by decide) (by decide) (by decide)
          (CanonicalStepPlan.afterHash_excludes parameters
            SourceOwners.stepContinuationHashPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact continuation_pair continuation encode
          (by decide) (by decide) (by decide)
          (CanonicalStepPlan.afterFreshPublic_excludes parameters
            SourceOwners.stepContinuationHashPath
            (by decide) (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact continuation_pair continuation equality
          (by decide) (by decide) (by decide)
          (CanonicalStepPlan.afterEncode_excludes parameters
            SourceOwners.stepContinuationHashPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide))
      · subst secondOccurrence
        exact continuation_pair continuation assertion
          (by decide) (by decide) (by decide)
          (CanonicalStepPlan.afterEncodedEquality_excludes parameters
            SourceOwners.stepContinuationHashPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide))
      · subst secondOccurrence
        exact continuation_pair continuation nifs
          (by decide) (by decide) (by decide)
          (CanonicalStepPlan.afterEncodedEquality_excludes parameters
            SourceOwners.stepContinuationHashPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide))

  have alwaysControl :
      ∀ selected,
        IdsDisjoint alwaysPlan.temporaryIds
          [oneColumn,
            activationColumn SourceOwners.stepBranchPath selected] := by
    intro selected id temporaryMember controlMember
    change id ∈
      [applyPlan.occurrence, selector.occurrence,
        continuation.occurrence].flatMap
          ArmOccurrence.temporaryIds at temporaryMember
    rcases List.mem_flatMap.mp temporaryMember with
      ⟨occurrence, occurrenceMember, occurrenceTemporary⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false]
      at occurrenceMember
    rcases occurrenceMember with equal | equal | equal
    · subst occurrence
      exact
        (applyPlan.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath selected)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepApplyPath)
          (CanonicalStepPlan.activation_excludes_instruction
            SourceOwners.stepBranchPath
            SourceOwners.stepApplyPath selected))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (selector.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath selected)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepSelectorPath)
          (CanonicalStepPlan.activation_excludes_instruction
            SourceOwners.stepBranchPath
            SourceOwners.stepSelectorPath selected))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (continuation.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath selected)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepContinuationHashPath)
          (CanonicalStepPlan.activation_excludes_instruction
            SourceOwners.stepBranchPath
            SourceOwners.stepContinuationHashPath selected))
          id occurrenceTemporary controlMember

  have baseAlwaysControl :
      IdsDisjoint basePlan.temporaryIds [oneColumn, oneColumn] := by
    intro id temporaryMember controlMember
    change id ∈
      [baseEquality.occurrence, baseAssertion.occurrence,
        baseLiteral.occurrence].flatMap
          ArmOccurrence.temporaryIds at temporaryMember
    rcases List.mem_flatMap.mp temporaryMember with
      ⟨occurrence, occurrenceMember, occurrenceTemporary⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false]
      at occurrenceMember
    rcases occurrenceMember with equal | equal | equal
    · subst occurrence
      exact
        (baseEquality.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepBaseStateEqualPath)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepBaseStateEqualPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (baseAssertion.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepBaseAssertionPath)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepBaseAssertionPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (baseLiteral.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepBaseDefaultPath)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepBaseDefaultPath))
          id occurrenceTemporary controlMember

  have recursiveAlwaysControl :
      IdsDisjoint recursivePlan.temporaryIds [oneColumn, oneColumn] := by
    intro id temporaryMember controlMember
    change id ∈
      [hash.occurrence, fresh.occurrence, encode.occurrence,
        equality.occurrence, assertion.occurrence,
        nifs.occurrence].flatMap
          ArmOccurrence.temporaryIds at temporaryMember
    rcases List.mem_flatMap.mp temporaryMember with
      ⟨occurrence, occurrenceMember, occurrenceTemporary⟩
    simp only [List.mem_cons, List.not_mem_nil, or_false]
      at occurrenceMember
    rcases occurrenceMember with
      equal | equal | equal | equal | equal | equal
    · subst occurrence
      exact
        (hash.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (fresh.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveFreshPublicPath)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveFreshPublicPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (encode.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveEncodePath)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveEncodePath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (equality.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveEncodedEqualPath)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveEncodedEqualPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (assertion.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveAssertionPath)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveAssertionPath))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (nifs.occurrenceTemporariesDisjointControls
          oneColumn oneColumn
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveNifsPath)
          (CanonicalStepPlan.one_excludes_instruction
            SourceOwners.stepRecursiveNifsPath))
          id occurrenceTemporary controlMember

  exact ⟨
    ⟨
      CompletionSeparation.plansSeparated_of_pairwise
        alwaysPlan basePlan basePairwise (alwaysControl true),
      CompletionSeparation.plansSeparated_reverse_of_pairwise
        alwaysPlan basePlan basePairwise baseAlwaysControl⟩,
    ⟨
      CompletionSeparation.plansSeparated_of_pairwise
        alwaysPlan recursivePlan recursivePairwise
        (alwaysControl false),
      CompletionSeparation.plansSeparated_reverse_of_pairwise
        alwaysPlan recursivePlan recursivePairwise
        recursiveAlwaysControl⟩⟩

end CanonicalCompletionPlans.Step

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
