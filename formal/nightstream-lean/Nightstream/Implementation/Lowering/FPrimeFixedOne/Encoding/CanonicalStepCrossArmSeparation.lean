import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepCompletionSeparation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CrossArmSeparation

/-!
Contract: exact cross-arm column separation for canonical Step completion.

Owns:
- exclusion of every recursive instruction owner from the base-arm contexts
  and conversely;
- both valid selected/inactive completion orders for the Step branch.

Does not own: within-arm separation, semantic witnesses, branch rows, honest
assignments, production Rust behavior, or R1CS indices.

Emits constraints: no.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalCompletionPlans.Step

private theorem one_excludes_instruction (path : OwnerPath) :
    oneColumn.owner ≠ .typed (.instruction path) := by
  simp [oneColumn]

private theorem activation_excludes_instruction
    (target : OwnerPath)
    (selected : Bool) :
    (activationColumn SourceOwners.stepBranchPath selected).owner ≠
      .typed (.instruction target) := by
  simp [activationColumn]

private theorem afterStep_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterStep parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepApplyPath target
      [Ports.committedState parameters] applyDifferent)
    (CanonicalPrimitivePlan.ContextExcludesOwner.input
      (stepInputSchema parameters) target)

private theorem common_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.common parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepSelectorPath target
      [Ports.auxiliaryBit parameters] selectorDifferent)
    (afterStep_excludes parameters target applyDifferent)

private theorem afterBaseEquality_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (baseEqualityDifferent :
      SourceOwners.stepBaseStateEqualPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterBaseEquality parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepBaseStateEqualPath target
      [Ports.auxiliaryBit parameters] baseEqualityDifferent)
    (common_excludes parameters target selectorDifferent applyDifferent)

private theorem afterHash_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (hashDifferent : SourceOwners.stepRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterHash parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepRecursiveHashPriorPath target
      [Ports.auxiliaryDigest parameters] hashDifferent)
    (common_excludes parameters target selectorDifferent applyDifferent)

private theorem afterFreshPublic_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (freshDifferent :
      SourceOwners.stepRecursiveFreshPublicPath ≠ target)
    (hashDifferent : SourceOwners.stepRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterFreshPublic parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepRecursiveFreshPublicPath target
      [Ports.auxiliaryEncoded parameters] freshDifferent)
    (afterHash_excludes parameters target hashDifferent
      selectorDifferent applyDifferent)

private theorem afterEncode_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (encodeDifferent : SourceOwners.stepRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.stepRecursiveFreshPublicPath ≠ target)
    (hashDifferent : SourceOwners.stepRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterEncode parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepRecursiveEncodePath target
      [Ports.auxiliaryEncoded parameters] encodeDifferent)
    (afterFreshPublic_excludes parameters target freshDifferent
      hashDifferent selectorDifferent applyDifferent)

private theorem afterEncodedEquality_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (equalityDifferent :
      SourceOwners.stepRecursiveEncodedEqualPath ≠ target)
    (encodeDifferent : SourceOwners.stepRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.stepRecursiveFreshPublicPath ≠ target)
    (hashDifferent : SourceOwners.stepRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.stepSelectorPath ≠ target)
    (applyDifferent : SourceOwners.stepApplyPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Step.afterEncodedEquality parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.stepRecursiveEncodedEqualPath target
      [Ports.auxiliaryBit parameters] equalityDifferent)
    (afterEncode_excludes parameters target encodeDifferent
      freshDifferent hashDifferent selectorDifferent applyDifferent)

private theorem cross_pair
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
    (first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns oneColumn
        (activationColumn SourceOwners.stepBranchPath true))
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns oneColumn
        (activationColumn SourceOwners.stepBranchPath false))
    (different : firstPath ≠ secondPath)
    (firstInputExcludesSecond :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction secondPath)) firstInputColumns)
    (secondInputExcludesFirst :
      CanonicalPrimitivePlan.ContextExcludesOwner
        (.typed (.instruction firstPath)) secondInputColumns) :
    IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.visibleIds ∧
      IdsDisjoint second.occurrence.temporaryIds
        first.occurrence.visibleIds ∧
      IdsDisjoint first.occurrence.temporaryIds
        second.occurrence.temporaryIds :=
  first.crossPairwiseSeparated second
    (one_excludes_instruction firstPath)
    (activation_excludes_instruction firstPath false)
    (one_excludes_instruction secondPath)
    (activation_excludes_instruction secondPath true)
    different firstInputExcludesSecond secondInputExcludesFirst

private theorem cross_pair_of_first_no_temporaries
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
    (first :
      PrimitivePlan parameters profile firstPrimitive firstPath
        firstInputColumns oneColumn
        (activationColumn SourceOwners.stepBranchPath true))
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns oneColumn
        (activationColumn SourceOwners.stepBranchPath false))
    (noTemporaries : first.occurrence.temporaryIds = [])
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
  first.crossPairwiseSeparated_of_first_no_temporaries second
    noTemporaries
    (one_excludes_instruction secondPath)
    (activation_excludes_instruction secondPath true)
    different firstInputExcludesSecond

/-- Both Step branch-completion orders preserve every opposite-arm visible
coordinate, temporary coordinate, and control. -/
theorem arms_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    ArmPlan.PlansSeparated
        (onTrue parameters profile recipes defaultAdmissible)
        (onFalse parameters profile recipes) ∧
      ArmPlan.PlansSeparated
        (onFalse parameters profile recipes)
        (onTrue parameters profile recipes defaultAdmissible) := by
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
  let basePlan :=
    onTrue parameters profile recipes defaultAdmissible
  let recursivePlan :=
    onFalse parameters profile recipes

  have pairwise :
      ∀ firstOccurrence,
        firstOccurrence ∈ basePlan.occurrences ->
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
      [baseEquality.occurrence, baseAssertion.occurrence,
        baseLiteral.occurrence] at firstMember
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
        exact cross_pair baseEquality hash (by decide)
          (common_excludes parameters
            SourceOwners.stepRecursiveHashPriorPath
            (by decide) (by decide))
          (common_excludes parameters
            SourceOwners.stepBaseStateEqualPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality fresh (by decide)
          (common_excludes parameters
            SourceOwners.stepRecursiveFreshPublicPath
            (by decide) (by decide))
          (afterHash_excludes parameters
            SourceOwners.stepBaseStateEqualPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality encode (by decide)
          (common_excludes parameters
            SourceOwners.stepRecursiveEncodePath
            (by decide) (by decide))
          (afterFreshPublic_excludes parameters
            SourceOwners.stepBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality equality (by decide)
          (common_excludes parameters
            SourceOwners.stepRecursiveEncodedEqualPath
            (by decide) (by decide))
          (afterEncode_excludes parameters
            SourceOwners.stepBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality assertion (by decide)
          (common_excludes parameters
            SourceOwners.stepRecursiveAssertionPath
            (by decide) (by decide))
          (afterEncodedEquality_excludes parameters
            SourceOwners.stepBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality nifs (by decide)
          (common_excludes parameters
            SourceOwners.stepRecursiveNifsPath
            (by decide) (by decide))
          (afterEncodedEquality_excludes parameters
            SourceOwners.stepBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide))
    · subst firstOccurrence
      have noTemporaries :
          baseAssertion.occurrence.temporaryIds = [] := by
        rfl
      rcases secondMember with
        secondEqual | secondEqual | secondEqual |
          secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion hash noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveHashPriorPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion fresh noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveFreshPublicPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion encode noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveEncodePath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion equality noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveEncodedEqualPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion assertion noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveAssertionPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion nifs noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveNifsPath
            (by decide) (by decide) (by decide))
    · subst firstOccurrence
      have noTemporaries :
          baseLiteral.occurrence.temporaryIds = [] := by
        rfl
      rcases secondMember with
        secondEqual | secondEqual | secondEqual |
          secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseLiteral hash noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveHashPriorPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseLiteral fresh noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveFreshPublicPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseLiteral encode noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveEncodePath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseLiteral equality noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveEncodedEqualPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseLiteral assertion noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveAssertionPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseLiteral nifs noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.stepRecursiveNifsPath
            (by decide) (by decide) (by decide))

  have baseControl :
      IdsDisjoint basePlan.temporaryIds
        [oneColumn,
          activationColumn SourceOwners.stepBranchPath false] := by
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
          oneColumn
          (activationColumn SourceOwners.stepBranchPath false)
          (one_excludes_instruction
            SourceOwners.stepBaseStateEqualPath)
          (activation_excludes_instruction
            SourceOwners.stepBaseStateEqualPath false))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (baseAssertion.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath false)
          (one_excludes_instruction
            SourceOwners.stepBaseAssertionPath)
          (activation_excludes_instruction
            SourceOwners.stepBaseAssertionPath false))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (baseLiteral.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath false)
          (one_excludes_instruction
            SourceOwners.stepBaseDefaultPath)
          (activation_excludes_instruction
            SourceOwners.stepBaseDefaultPath false))
          id occurrenceTemporary controlMember

  have recursiveControl :
      IdsDisjoint recursivePlan.temporaryIds
        [oneColumn,
          activationColumn SourceOwners.stepBranchPath true] := by
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
          oneColumn
          (activationColumn SourceOwners.stepBranchPath true)
          (one_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (fresh.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath true)
          (one_excludes_instruction
            SourceOwners.stepRecursiveFreshPublicPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveFreshPublicPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (encode.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath true)
          (one_excludes_instruction
            SourceOwners.stepRecursiveEncodePath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveEncodePath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (equality.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath true)
          (one_excludes_instruction
            SourceOwners.stepRecursiveEncodedEqualPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveEncodedEqualPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (assertion.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath true)
          (one_excludes_instruction
            SourceOwners.stepRecursiveAssertionPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveAssertionPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (nifs.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.stepBranchPath true)
          (one_excludes_instruction
            SourceOwners.stepRecursiveNifsPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveNifsPath true))
          id occurrenceTemporary controlMember

  constructor
  · exact CompletionSeparation.plansSeparated_of_pairwise
      basePlan recursivePlan pairwise baseControl
  · exact CompletionSeparation.plansSeparated_reverse_of_pairwise
      basePlan recursivePlan pairwise recursiveControl

end CanonicalCompletionPlans.Step

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
