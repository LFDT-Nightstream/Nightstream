import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalCompletionSeparation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CrossArmSeparation

/-!
Contract: exact cross-arm column separation for canonical Terminal
completion.

Owns:
- exclusion of every recursive instruction owner from the base-arm contexts
  and conversely;
- both valid selected/inactive completion orders for the Terminal branch.

Does not own: within-arm separation, semantic witnesses, branch rows, honest
assignments, production Rust behavior, or R1CS indices.

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

private theorem branchInput_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.branchInput parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalSelectorPath target
      [Ports.auxiliaryBit parameters] selectorDifferent)
    (CanonicalPrimitivePlan.ContextExcludesOwner.input
      (terminalInputSchema parameters) target)

private theorem afterBaseEquality_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (baseEqualityDifferent :
      SourceOwners.terminalBaseStateEqualPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterBaseEquality parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalBaseStateEqualPath target
      [Ports.auxiliaryBit parameters] baseEqualityDifferent)
    (branchInput_excludes parameters target selectorDifferent)

private theorem afterHash_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterHash parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveHashPriorPath target
      [Ports.auxiliaryDigest parameters] hashDifferent)
    (branchInput_excludes parameters target selectorDifferent)

private theorem afterFreshPublic_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (freshDifferent :
      SourceOwners.terminalRecursiveFreshPublicPath ≠ target)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterFreshPublic parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveFreshPublicPath target
      [Ports.auxiliaryEncoded parameters] freshDifferent)
    (afterHash_excludes parameters target hashDifferent selectorDifferent)

private theorem afterEncode_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (encodeDifferent :
      SourceOwners.terminalRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.terminalRecursiveFreshPublicPath ≠ target)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterEncode parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveEncodePath target
      [Ports.auxiliaryEncoded parameters] encodeDifferent)
    (afterFreshPublic_excludes parameters target freshDifferent
      hashDifferent selectorDifferent)

private theorem afterEncodedEquality_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (equalityDifferent :
      SourceOwners.terminalRecursiveEncodedEqualPath ≠ target)
    (encodeDifferent :
      SourceOwners.terminalRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.terminalRecursiveFreshPublicPath ≠ target)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterEncodedEquality parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveEncodedEqualPath target
      [Ports.auxiliaryBit parameters] equalityDifferent)
    (afterEncode_excludes parameters target encodeDifferent
      freshDifferent hashDifferent selectorDifferent)

private theorem afterRunningCheck_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (runningDifferent :
      SourceOwners.terminalRecursiveRunningCheckPath ≠ target)
    (equalityDifferent :
      SourceOwners.terminalRecursiveEncodedEqualPath ≠ target)
    (encodeDifferent :
      SourceOwners.terminalRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.terminalRecursiveFreshPublicPath ≠ target)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterRunningCheck parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveRunningCheckPath target
      [Ports.auxiliaryBit parameters] runningDifferent)
    (afterEncodedEquality_excludes parameters target
      equalityDifferent encodeDifferent freshDifferent
      hashDifferent selectorDifferent)

private theorem afterFreshCheck_excludes
    (parameters : Parameters)
    (target : OwnerPath)
    (freshCheckDifferent :
      SourceOwners.terminalRecursiveFreshCheckPath ≠ target)
    (runningDifferent :
      SourceOwners.terminalRecursiveRunningCheckPath ≠ target)
    (equalityDifferent :
      SourceOwners.terminalRecursiveEncodedEqualPath ≠ target)
    (encodeDifferent :
      SourceOwners.terminalRecursiveEncodePath ≠ target)
    (freshDifferent :
      SourceOwners.terminalRecursiveFreshPublicPath ≠ target)
    (hashDifferent :
      SourceOwners.terminalRecursiveHashPriorPath ≠ target)
    (selectorDifferent : SourceOwners.terminalSelectorPath ≠ target) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction target))
      (CanonicalContexts.Terminal.afterFreshCheck parameters) :=
  CanonicalPrimitivePlan.ContextExcludesOwner.append
    (CanonicalPrimitivePlan.ContextExcludesOwner.instruction
      SourceOwners.terminalRecursiveFreshCheckPath target
      [Ports.auxiliaryBit parameters] freshCheckDifferent)
    (afterRunningCheck_excludes parameters target
      runningDifferent equalityDifferent encodeDifferent
      freshDifferent hashDifferent selectorDifferent)

/-- The complete recursive Terminal SSA context contains no coordinate owned
by the base equality instruction. -/
theorem afterFreshCheck_excludes_baseEquality
    (parameters : Parameters) :
    CanonicalPrimitivePlan.ContextExcludesOwner
      (.typed (.instruction SourceOwners.terminalBaseStateEqualPath))
      (CanonicalContexts.Terminal.afterFreshCheck parameters) :=
  afterFreshCheck_excludes parameters
    SourceOwners.terminalBaseStateEqualPath
    (by decide) (by decide) (by decide) (by decide)
    (by decide) (by decide) (by decide)

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
        (activationColumn SourceOwners.terminalBranchPath true))
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns oneColumn
        (activationColumn SourceOwners.terminalBranchPath false))
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
        (activationColumn SourceOwners.terminalBranchPath true))
    (second :
      PrimitivePlan parameters profile secondPrimitive secondPath
        secondInputColumns oneColumn
        (activationColumn SourceOwners.terminalBranchPath false))
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

/-- Both Terminal branch-completion orders preserve every opposite-arm
visible coordinate, temporary coordinate, and control. -/
theorem arms_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ArmPlan.PlansSeparated
        (onTrue parameters profile recipes)
        (onFalse parameters profile recipes) ∧
      ArmPlan.PlansSeparated
        (onFalse parameters profile recipes)
        (onTrue parameters profile recipes) := by
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
  let basePlan :=
    onTrue parameters profile recipes
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
      [baseEquality.occurrence, baseAssertion.occurrence] at firstMember
    change secondOccurrence ∈
      [hash.occurrence, fresh.occurrence, encode.occurrence,
        equality.occurrence, priorAssertion.occurrence,
        runningCheck.occurrence, runningAssertion.occurrence,
        freshCheck.occurrence, freshAssertion.occurrence] at secondMember
    simp only [List.mem_cons, List.not_mem_nil, or_false]
      at firstMember secondMember
    rcases firstMember with firstEqual | firstEqual
    · subst firstOccurrence
      rcases secondMember with
        secondEqual | secondEqual | secondEqual |
          secondEqual | secondEqual | secondEqual |
          secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact cross_pair baseEquality hash (by decide)
          (branchInput_excludes parameters
            SourceOwners.terminalRecursiveHashPriorPath
            (by decide))
          (branchInput_excludes parameters
            SourceOwners.terminalBaseStateEqualPath
            (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality fresh (by decide)
          (branchInput_excludes parameters
            SourceOwners.terminalRecursiveFreshPublicPath
            (by decide))
          (afterHash_excludes parameters
            SourceOwners.terminalBaseStateEqualPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality encode (by decide)
          (branchInput_excludes parameters
            SourceOwners.terminalRecursiveEncodePath
            (by decide))
          (afterFreshPublic_excludes parameters
            SourceOwners.terminalBaseStateEqualPath
            (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality equality (by decide)
          (branchInput_excludes parameters
            SourceOwners.terminalRecursiveEncodedEqualPath
            (by decide))
          (afterEncode_excludes parameters
            SourceOwners.terminalBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality priorAssertion (by decide)
          (branchInput_excludes parameters
            SourceOwners.terminalRecursivePriorAssertionPath
            (by decide))
          (afterEncodedEquality_excludes parameters
            SourceOwners.terminalBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality runningCheck (by decide)
          (branchInput_excludes parameters
            SourceOwners.terminalRecursiveRunningCheckPath
            (by decide))
          (afterEncodedEquality_excludes parameters
            SourceOwners.terminalBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality runningAssertion (by decide)
          (branchInput_excludes parameters
            SourceOwners.terminalRecursiveRunningAssertionPath
            (by decide))
          (afterRunningCheck_excludes parameters
            SourceOwners.terminalBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality freshCheck (by decide)
          (branchInput_excludes parameters
            SourceOwners.terminalRecursiveFreshCheckPath
            (by decide))
          (afterRunningCheck_excludes parameters
            SourceOwners.terminalBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair baseEquality freshAssertion (by decide)
          (branchInput_excludes parameters
            SourceOwners.terminalRecursiveFreshAssertionPath
            (by decide))
          (afterFreshCheck_excludes parameters
            SourceOwners.terminalBaseStateEqualPath
            (by decide) (by decide) (by decide) (by decide)
            (by decide) (by decide) (by decide))
    · subst firstOccurrence
      have noTemporaries :
          baseAssertion.occurrence.temporaryIds = [] := by
        rfl
      rcases secondMember with
        secondEqual | secondEqual | secondEqual |
          secondEqual | secondEqual | secondEqual |
          secondEqual | secondEqual | secondEqual
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion hash noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.terminalRecursiveHashPriorPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion fresh noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.terminalRecursiveFreshPublicPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion encode noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.terminalRecursiveEncodePath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion equality noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.terminalRecursiveEncodedEqualPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion priorAssertion noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.terminalRecursivePriorAssertionPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion runningCheck noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.terminalRecursiveRunningCheckPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion runningAssertion noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.terminalRecursiveRunningAssertionPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion freshCheck noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.terminalRecursiveFreshCheckPath
            (by decide) (by decide))
      · subst secondOccurrence
        exact cross_pair_of_first_no_temporaries
          baseAssertion freshAssertion noTemporaries (by decide)
          (afterBaseEquality_excludes parameters
            SourceOwners.terminalRecursiveFreshAssertionPath
            (by decide) (by decide))

  have baseControl :
      IdsDisjoint basePlan.temporaryIds
        [oneColumn,
          activationColumn SourceOwners.terminalBranchPath false] := by
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
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath false)
          (one_excludes_instruction
            SourceOwners.terminalBaseStateEqualPath)
          (activation_excludes_instruction
            SourceOwners.terminalBaseStateEqualPath false))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (baseAssertion.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath false)
          (one_excludes_instruction
            SourceOwners.terminalBaseAssertionPath)
          (activation_excludes_instruction
            SourceOwners.terminalBaseAssertionPath false))
          id occurrenceTemporary controlMember

  have recursiveControl :
      IdsDisjoint recursivePlan.temporaryIds
        [oneColumn,
          activationColumn SourceOwners.terminalBranchPath true] := by
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
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath true)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (fresh.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath true)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveFreshPublicPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveFreshPublicPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (encode.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath true)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveEncodePath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveEncodePath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (equality.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath true)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveEncodedEqualPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveEncodedEqualPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (priorAssertion.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath true)
          (one_excludes_instruction
            SourceOwners.terminalRecursivePriorAssertionPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursivePriorAssertionPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (runningCheck.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath true)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveRunningCheckPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveRunningCheckPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (runningAssertion.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath true)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveRunningAssertionPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveRunningAssertionPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (freshCheck.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath true)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveFreshCheckPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveFreshCheckPath true))
          id occurrenceTemporary controlMember
    · subst occurrence
      exact
        (freshAssertion.occurrenceTemporariesDisjointControls
          oneColumn
          (activationColumn SourceOwners.terminalBranchPath true)
          (one_excludes_instruction
            SourceOwners.terminalRecursiveFreshAssertionPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveFreshAssertionPath true))
          id occurrenceTemporary controlMember

  constructor
  · exact CompletionSeparation.plansSeparated_of_pairwise
      basePlan recursivePlan pairwise baseControl
  · exact CompletionSeparation.plansSeparated_reverse_of_pairwise
      basePlan recursivePlan pairwise recursiveControl

end CanonicalCompletionPlans.Terminal

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
