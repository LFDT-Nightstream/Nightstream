import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalCompletionPlans

/-!
Contract: whole-arm column-separation certificates for the exact canonical
Step completion plans.

Owns:
- recursive-arm SSA protection across all six primitive occurrences;
- the resulting ordered occurrence-separation theorem.

Does not own: semantic execution values, honest assignments, branch-control
rows, join rows, production Rust behavior, or R1CS indices.

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
    (target : OwnerPath) :
    (activationColumn SourceOwners.stepBranchPath false).owner ≠
      .typed (.instruction target) := by
  simp [activationColumn]

/-- The always-active Step prefix and continuation preserve temporary
isolation across the branch join. -/
theorem always_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (always parameters profile recipes).Separated := by
  let applyPlan :=
    CanonicalStepPlan.applyPlan.{0} parameters profile recipes
  let selector :=
    CanonicalStepPlan.selectorPlan.{0} parameters profile recipes
  let continuation :=
    CanonicalStepPlan.continuationHashPlan.{0}
      parameters profile recipes
  have applySelector := applyPlan.protectsResult
  have applyCommon :=
    applySelector.extend selector (by decide)
  have selectorCommon := selector.protectsResult
  have applyJoinDisjoint :
      IdsDisjoint applyPlan.occurrence.temporaryIds
        (CanonicalContexts.Step.joined parameters).toSchemaBundles.ids := by
    apply applyPlan.occurrenceTemporariesDisjointOwnerExcluded
    exact
      CanonicalPrimitivePlan.ContextExcludesOwner.id_excludes
        (CanonicalPrimitivePlan.ContextExcludesOwner.branch
          SourceOwners.stepBranchPath
          SourceOwners.stepApplyPath
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
            parameters))
  have selectorJoinDisjoint :
      IdsDisjoint selector.occurrence.temporaryIds
        (CanonicalContexts.Step.joined parameters).toSchemaBundles.ids := by
    apply selector.occurrenceTemporariesDisjointOwnerExcluded
    exact
      CanonicalPrimitivePlan.ContextExcludesOwner.id_excludes
        (CanonicalPrimitivePlan.ContextExcludesOwner.branch
          SourceOwners.stepBranchPath
          SourceOwners.stepSelectorPath
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
            parameters))
  have applyContinuation :=
    applyCommon.prepend
      (CanonicalContexts.Step.joined parameters)
      applyJoinDisjoint
  have selectorContinuation :=
    selectorCommon.prepend
      (CanonicalContexts.Step.joined parameters)
      selectorJoinDisjoint
  change ArmPlan.SeparatedOccurrences
    [applyPlan.occurrence, selector.occurrence,
      continuation.occurrence]
  apply CompletionSeparation.separatedOccurrences_of_pairwise
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with equal | equal | equal
    · subst occurrence
      exact applyPlan.occurrenceRowsSupported
    · subst occurrence
      exact selector.occurrenceRowsSupported
    · subst occurrence
      exact continuation.occurrenceRowsSupported
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with equal | equal | equal
    · subst occurrence
      exact applyPlan.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact selector.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact continuation.occurrenceTemporariesDisjointVisible
  · rw [List.pairwise_cons]
    constructor
    · intro occurrence member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with equal | equal
      · subst occurrence
        exact applySelector.pairwiseSeparated selector
          (one_excludes_instruction SourceOwners.stepApplyPath)
          (one_excludes_instruction SourceOwners.stepApplyPath)
          (by decide)
      · subst occurrence
        exact applyContinuation.pairwiseSeparated continuation
          (one_excludes_instruction SourceOwners.stepApplyPath)
          (one_excludes_instruction SourceOwners.stepApplyPath)
          (by decide)
    · rw [List.pairwise_cons]
      constructor
      · intro occurrence member
        simp only [List.mem_singleton] at member
        subst occurrence
        exact selectorContinuation.pairwiseSeparated continuation
          (one_excludes_instruction SourceOwners.stepSelectorPath)
          (one_excludes_instruction SourceOwners.stepSelectorPath)
          (by decide)
      · simp

/-- The exact recursive Step arm satisfies every separation premise needed
for constructive active/inactive completion. -/
theorem onFalse_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (onFalse parameters profile recipes).Separated := by
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
  have hashFresh := hash.protectsResult
  have hashEncode :=
    hashFresh.extend fresh (by decide)
  have hashEquality :=
    hashEncode.extend encode (by decide)
  have hashAssertion :=
    hashEquality.extend equality (by decide)
  have freshEncode := fresh.protectsResult
  have freshEquality :=
    freshEncode.extend encode (by decide)
  have freshAssertion :=
    freshEquality.extend equality (by decide)
  have encodeEquality := encode.protectsResult
  have encodeAssertion :=
    encodeEquality.extend equality (by decide)
  have equalityAssertion := equality.protectsResult
  have assertionNifs := assertion.protectsResult
  change ArmPlan.SeparatedOccurrences
    [ hash.occurrence, fresh.occurrence, encode.occurrence,
      equality.occurrence, assertion.occurrence, nifs.occurrence ]
  apply CompletionSeparation.separatedOccurrences_of_pairwise
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with
      equal | equal | equal | equal | equal | equal
    · subst occurrence
      exact hash.occurrenceRowsSupported
    · subst occurrence
      exact fresh.occurrenceRowsSupported
    · subst occurrence
      exact encode.occurrenceRowsSupported
    · subst occurrence
      exact equality.occurrenceRowsSupported
    · subst occurrence
      exact assertion.occurrenceRowsSupported
    · subst occurrence
      exact nifs.occurrenceRowsSupported
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with
      equal | equal | equal | equal | equal | equal
    · subst occurrence
      exact hash.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact fresh.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact encode.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact equality.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact assertion.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact nifs.occurrenceTemporariesDisjointVisible
  · rw [List.pairwise_cons]
    constructor
    · intro occurrence member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with equal | equal | equal | equal | equal
      · subst occurrence
        exact hashFresh.pairwiseSeparated fresh
          (one_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (by decide)
      · subst occurrence
        exact hashEncode.pairwiseSeparated encode
          (one_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (by decide)
      · subst occurrence
        exact hashEquality.pairwiseSeparated equality
          (one_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (by decide)
      · subst occurrence
        exact hashAssertion.pairwiseSeparated assertion
          (one_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (by decide)
      · subst occurrence
        exact hashAssertion.pairwiseSeparated nifs
          (one_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.stepRecursiveHashPriorPath)
          (by decide)
    · rw [List.pairwise_cons]
      constructor
      · intro occurrence member
        simp only [List.mem_cons, List.not_mem_nil, or_false] at member
        rcases member with equal | equal | equal | equal
        · subst occurrence
          exact freshEncode.pairwiseSeparated encode
            (one_excludes_instruction
              SourceOwners.stepRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.stepRecursiveFreshPublicPath)
            (by decide)
        · subst occurrence
          exact freshEquality.pairwiseSeparated equality
            (one_excludes_instruction
              SourceOwners.stepRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.stepRecursiveFreshPublicPath)
            (by decide)
        · subst occurrence
          exact freshAssertion.pairwiseSeparated assertion
            (one_excludes_instruction
              SourceOwners.stepRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.stepRecursiveFreshPublicPath)
            (by decide)
        · subst occurrence
          exact freshAssertion.pairwiseSeparated nifs
            (one_excludes_instruction
              SourceOwners.stepRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.stepRecursiveFreshPublicPath)
            (by decide)
      · rw [List.pairwise_cons]
        constructor
        · intro occurrence member
          simp only [List.mem_cons, List.not_mem_nil, or_false] at member
          rcases member with equal | equal | equal
          · subst occurrence
            exact encodeEquality.pairwiseSeparated equality
              (one_excludes_instruction
                SourceOwners.stepRecursiveEncodePath)
              (activation_excludes_instruction
                SourceOwners.stepRecursiveEncodePath)
              (by decide)
          · subst occurrence
            exact encodeAssertion.pairwiseSeparated assertion
              (one_excludes_instruction
                SourceOwners.stepRecursiveEncodePath)
              (activation_excludes_instruction
                SourceOwners.stepRecursiveEncodePath)
              (by decide)
          · subst occurrence
            exact encodeAssertion.pairwiseSeparated nifs
              (one_excludes_instruction
                SourceOwners.stepRecursiveEncodePath)
              (activation_excludes_instruction
                SourceOwners.stepRecursiveEncodePath)
              (by decide)
        · rw [List.pairwise_cons]
          constructor
          · intro occurrence member
            simp only [List.mem_cons, List.not_mem_nil, or_false] at member
            rcases member with equal | equal
            · subst occurrence
              exact equalityAssertion.pairwiseSeparated assertion
                (one_excludes_instruction
                  SourceOwners.stepRecursiveEncodedEqualPath)
                (activation_excludes_instruction
                  SourceOwners.stepRecursiveEncodedEqualPath)
                (by decide)
            · subst occurrence
              exact equalityAssertion.pairwiseSeparated nifs
                (one_excludes_instruction
                  SourceOwners.stepRecursiveEncodedEqualPath)
                (activation_excludes_instruction
                  SourceOwners.stepRecursiveEncodedEqualPath)
                (by decide)
          · rw [List.pairwise_cons]
            constructor
            · intro occurrence member
              simp only [List.mem_singleton] at member
              subst occurrence
              exact assertionNifs.pairwiseSeparated nifs
                (one_excludes_instruction
                  SourceOwners.stepRecursiveAssertionPath)
                (activation_excludes_instruction
                  SourceOwners.stepRecursiveAssertionPath)
                (by decide)
            · simp

end CanonicalCompletionPlans.Step

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
