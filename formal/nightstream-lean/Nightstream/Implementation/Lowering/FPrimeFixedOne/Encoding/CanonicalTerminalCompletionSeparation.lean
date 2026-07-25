import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalCompletionPlans

/-!
Contract: whole-arm column-separation certificates for the exact canonical
Terminal completion plans.

Owns:
- the singleton always-active plan;
- the base-arm pair;
- recursive-arm SSA protection across both assertion forks.

Does not own: semantic execution values, honest assignments, branch-control
rows, production Rust behavior, or R1CS indices.

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

/-- The singleton selector group is separated without any cross-occurrence
premise. -/
theorem always_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (always parameters profile recipes).Separated := by
  let selector :=
    CanonicalTerminalPlan.selectorPlan.{0}
      parameters profile recipes
  change ArmPlan.SeparatedOccurrences [selector.occurrence]
  exact ⟨
    selector.occurrenceRowsSupported,
    selector.occurrenceTemporariesDisjointVisible,
    by simp [IdsDisjoint],
    by simp [IdsDisjoint],
    by simp [IdsDisjoint],
    trivial⟩

/-- The exact Terminal base arm has a protected equality result followed by
its active assertion. -/
theorem onTrue_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (onTrue parameters profile recipes).Separated := by
  let equality :=
    CanonicalTerminalPlan.baseEqualityPlan.{0}
      parameters profile recipes
  let assertion :=
    CanonicalTerminalPlan.baseAssertionPlan.{0}
      parameters profile
  have equalityAssertion := equality.protectsResult
  change ArmPlan.SeparatedOccurrences
    [equality.occurrence, assertion.occurrence]
  apply CompletionSeparation.separatedOccurrences_of_pairwise
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with equal | equal
    · subst occurrence
      exact equality.occurrenceRowsSupported
    · subst occurrence
      exact assertion.occurrenceRowsSupported
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with equal | equal
    · subst occurrence
      exact equality.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact assertion.occurrenceTemporariesDisjointVisible
  · rw [List.pairwise_cons]
    constructor
    · intro occurrence member
      simp only [List.mem_singleton] at member
      subst occurrence
      exact equalityAssertion.pairwiseSeparated assertion
        (one_excludes_instruction
          SourceOwners.terminalBaseStateEqualPath)
        (activation_excludes_instruction
          SourceOwners.terminalBaseStateEqualPath true)
        (by decide)
    · simp

/-- The exact Terminal recursive arm preserves every earlier temporary set
across its two assertion forks and all later SSA extensions. -/
theorem onFalse_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (onFalse parameters profile recipes).Separated := by
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

  have hashFresh := hash.protectsResult
  have hashEncode :=
    hashFresh.extend fresh (by decide)
  have hashEquality :=
    hashEncode.extend encode (by decide)
  have hashAfterEquality :=
    hashEquality.extend equality (by decide)
  have hashAfterRunning :=
    hashAfterEquality.extend runningCheck (by decide)
  have hashAfterFresh :=
    hashAfterRunning.extend freshCheck (by decide)

  have freshEncode := fresh.protectsResult
  have freshEquality :=
    freshEncode.extend encode (by decide)
  have freshAfterEquality :=
    freshEquality.extend equality (by decide)
  have freshAfterRunning :=
    freshAfterEquality.extend runningCheck (by decide)
  have freshPublicAfterFresh :=
    freshAfterRunning.extend freshCheck (by decide)

  have encodeEquality := encode.protectsResult
  have encodeAfterEquality :=
    encodeEquality.extend equality (by decide)
  have encodeAfterRunning :=
    encodeAfterEquality.extend runningCheck (by decide)
  have encodeAfterFresh :=
    encodeAfterRunning.extend freshCheck (by decide)

  have equalityAfterEquality := equality.protectsResult
  have equalityAfterRunning :=
    equalityAfterEquality.extend runningCheck (by decide)
  have equalityAfterFresh :=
    equalityAfterRunning.extend freshCheck (by decide)

  have priorAfterEquality := priorAssertion.protectsResult
  have priorAfterRunning :=
    priorAfterEquality.extend runningCheck (by decide)
  have priorAfterFresh :=
    priorAfterRunning.extend freshCheck (by decide)

  have runningAfterRunning := runningCheck.protectsResult
  have runningAfterFresh :=
    runningAfterRunning.extend freshCheck (by decide)

  have runningAssertionAfterRunning :=
    runningAssertion.protectsResult
  have runningAssertionAfterFresh :=
    runningAssertionAfterRunning.extend freshCheck (by decide)

  have freshCheckAfterFresh := freshCheck.protectsResult

  change ArmPlan.SeparatedOccurrences
    [ hash.occurrence, fresh.occurrence, encode.occurrence,
      equality.occurrence, priorAssertion.occurrence,
      runningCheck.occurrence, runningAssertion.occurrence,
      freshCheck.occurrence, freshAssertion.occurrence ]
  apply CompletionSeparation.separatedOccurrences_of_pairwise
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with
      equal | equal | equal | equal | equal | equal | equal | equal | equal
    · subst occurrence
      exact hash.occurrenceRowsSupported
    · subst occurrence
      exact fresh.occurrenceRowsSupported
    · subst occurrence
      exact encode.occurrenceRowsSupported
    · subst occurrence
      exact equality.occurrenceRowsSupported
    · subst occurrence
      exact priorAssertion.occurrenceRowsSupported
    · subst occurrence
      exact runningCheck.occurrenceRowsSupported
    · subst occurrence
      exact runningAssertion.occurrenceRowsSupported
    · subst occurrence
      exact freshCheck.occurrenceRowsSupported
    · subst occurrence
      exact freshAssertion.occurrenceRowsSupported
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with
      equal | equal | equal | equal | equal | equal | equal | equal | equal
    · subst occurrence
      exact hash.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact fresh.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact encode.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact equality.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact priorAssertion.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact runningCheck.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact runningAssertion.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact freshCheck.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact freshAssertion.occurrenceTemporariesDisjointVisible
  · rw [List.pairwise_cons]
    constructor
    · intro occurrence member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with
        equal | equal | equal | equal | equal | equal | equal | equal
      · subst occurrence
        exact hashFresh.pairwiseSeparated fresh
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath false)
          (by decide)
      · subst occurrence
        exact hashEncode.pairwiseSeparated encode
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath false)
          (by decide)
      · subst occurrence
        exact hashEquality.pairwiseSeparated equality
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath false)
          (by decide)
      · subst occurrence
        exact hashAfterEquality.pairwiseSeparated priorAssertion
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath false)
          (by decide)
      · subst occurrence
        exact hashAfterEquality.pairwiseSeparated runningCheck
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath false)
          (by decide)
      · subst occurrence
        exact hashAfterRunning.pairwiseSeparated runningAssertion
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath false)
          (by decide)
      · subst occurrence
        exact hashAfterRunning.pairwiseSeparated freshCheck
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath false)
          (by decide)
      · subst occurrence
        exact hashAfterFresh.pairwiseSeparated freshAssertion
          (one_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath)
          (activation_excludes_instruction
            SourceOwners.terminalRecursiveHashPriorPath false)
          (by decide)
    · rw [List.pairwise_cons]
      constructor
      · intro occurrence member
        simp only [List.mem_cons, List.not_mem_nil, or_false] at member
        rcases member with
          equal | equal | equal | equal | equal | equal | equal
        · subst occurrence
          exact freshEncode.pairwiseSeparated encode
            (one_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath false)
            (by decide)
        · subst occurrence
          exact freshEquality.pairwiseSeparated equality
            (one_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath false)
            (by decide)
        · subst occurrence
          exact freshAfterEquality.pairwiseSeparated priorAssertion
            (one_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath false)
            (by decide)
        · subst occurrence
          exact freshAfterEquality.pairwiseSeparated runningCheck
            (one_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath false)
            (by decide)
        · subst occurrence
          exact freshAfterRunning.pairwiseSeparated runningAssertion
            (one_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath false)
            (by decide)
        · subst occurrence
          exact freshAfterRunning.pairwiseSeparated freshCheck
            (one_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath false)
            (by decide)
        · subst occurrence
          exact freshPublicAfterFresh.pairwiseSeparated freshAssertion
            (one_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath)
            (activation_excludes_instruction
              SourceOwners.terminalRecursiveFreshPublicPath false)
            (by decide)
      · rw [List.pairwise_cons]
        constructor
        · intro occurrence member
          simp only [List.mem_cons, List.not_mem_nil, or_false] at member
          rcases member with
            equal | equal | equal | equal | equal | equal
          · subst occurrence
            exact encodeEquality.pairwiseSeparated equality
              (one_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath)
              (activation_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath false)
              (by decide)
          · subst occurrence
            exact encodeAfterEquality.pairwiseSeparated priorAssertion
              (one_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath)
              (activation_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath false)
              (by decide)
          · subst occurrence
            exact encodeAfterEquality.pairwiseSeparated runningCheck
              (one_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath)
              (activation_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath false)
              (by decide)
          · subst occurrence
            exact encodeAfterRunning.pairwiseSeparated runningAssertion
              (one_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath)
              (activation_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath false)
              (by decide)
          · subst occurrence
            exact encodeAfterRunning.pairwiseSeparated freshCheck
              (one_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath)
              (activation_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath false)
              (by decide)
          · subst occurrence
            exact encodeAfterFresh.pairwiseSeparated freshAssertion
              (one_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath)
              (activation_excludes_instruction
                SourceOwners.terminalRecursiveEncodePath false)
              (by decide)
        · rw [List.pairwise_cons]
          constructor
          · intro occurrence member
            simp only [List.mem_cons, List.not_mem_nil, or_false] at member
            rcases member with
              equal | equal | equal | equal | equal
            · subst occurrence
              exact equalityAfterEquality.pairwiseSeparated priorAssertion
                (one_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath)
                (activation_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath false)
                (by decide)
            · subst occurrence
              exact equalityAfterEquality.pairwiseSeparated runningCheck
                (one_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath)
                (activation_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath false)
                (by decide)
            · subst occurrence
              exact equalityAfterRunning.pairwiseSeparated runningAssertion
                (one_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath)
                (activation_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath false)
                (by decide)
            · subst occurrence
              exact equalityAfterRunning.pairwiseSeparated freshCheck
                (one_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath)
                (activation_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath false)
                (by decide)
            · subst occurrence
              exact equalityAfterFresh.pairwiseSeparated freshAssertion
                (one_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath)
                (activation_excludes_instruction
                  SourceOwners.terminalRecursiveEncodedEqualPath false)
                (by decide)
          · rw [List.pairwise_cons]
            constructor
            · intro occurrence member
              simp only [List.mem_cons, List.not_mem_nil, or_false] at member
              rcases member with equal | equal | equal | equal
              · subst occurrence
                exact priorAfterEquality.pairwiseSeparated runningCheck
                  (one_excludes_instruction
                    SourceOwners.terminalRecursivePriorAssertionPath)
                  (activation_excludes_instruction
                    SourceOwners.terminalRecursivePriorAssertionPath false)
                  (by decide)
              · subst occurrence
                exact priorAfterRunning.pairwiseSeparated runningAssertion
                  (one_excludes_instruction
                    SourceOwners.terminalRecursivePriorAssertionPath)
                  (activation_excludes_instruction
                    SourceOwners.terminalRecursivePriorAssertionPath false)
                  (by decide)
              · subst occurrence
                exact priorAfterRunning.pairwiseSeparated freshCheck
                  (one_excludes_instruction
                    SourceOwners.terminalRecursivePriorAssertionPath)
                  (activation_excludes_instruction
                    SourceOwners.terminalRecursivePriorAssertionPath false)
                  (by decide)
              · subst occurrence
                exact priorAfterFresh.pairwiseSeparated freshAssertion
                  (one_excludes_instruction
                    SourceOwners.terminalRecursivePriorAssertionPath)
                  (activation_excludes_instruction
                    SourceOwners.terminalRecursivePriorAssertionPath false)
                  (by decide)
            · rw [List.pairwise_cons]
              constructor
              · intro occurrence member
                simp only [List.mem_cons, List.not_mem_nil, or_false] at member
                rcases member with equal | equal | equal
                · subst occurrence
                  exact runningAfterRunning.pairwiseSeparated
                    runningAssertion
                    (one_excludes_instruction
                      SourceOwners.terminalRecursiveRunningCheckPath)
                    (activation_excludes_instruction
                      SourceOwners.terminalRecursiveRunningCheckPath false)
                    (by decide)
                · subst occurrence
                  exact runningAfterRunning.pairwiseSeparated freshCheck
                    (one_excludes_instruction
                      SourceOwners.terminalRecursiveRunningCheckPath)
                    (activation_excludes_instruction
                      SourceOwners.terminalRecursiveRunningCheckPath false)
                    (by decide)
                · subst occurrence
                  exact runningAfterFresh.pairwiseSeparated freshAssertion
                    (one_excludes_instruction
                      SourceOwners.terminalRecursiveRunningCheckPath)
                    (activation_excludes_instruction
                      SourceOwners.terminalRecursiveRunningCheckPath false)
                    (by decide)
              · rw [List.pairwise_cons]
                constructor
                · intro occurrence member
                  simp only [List.mem_cons, List.not_mem_nil, or_false] at member
                  rcases member with equal | equal
                  · subst occurrence
                    exact
                      runningAssertionAfterRunning.pairwiseSeparated freshCheck
                        (one_excludes_instruction
                          SourceOwners.terminalRecursiveRunningAssertionPath)
                        (activation_excludes_instruction
                          SourceOwners.terminalRecursiveRunningAssertionPath
                          false)
                        (by decide)
                  · subst occurrence
                    exact
                      runningAssertionAfterFresh.pairwiseSeparated
                        freshAssertion
                        (one_excludes_instruction
                          SourceOwners.terminalRecursiveRunningAssertionPath)
                        (activation_excludes_instruction
                          SourceOwners.terminalRecursiveRunningAssertionPath
                          false)
                        (by decide)
                · rw [List.pairwise_cons]
                  constructor
                  · intro occurrence member
                    simp only [List.mem_singleton] at member
                    subst occurrence
                    exact freshCheckAfterFresh.pairwiseSeparated freshAssertion
                      (one_excludes_instruction
                        SourceOwners.terminalRecursiveFreshCheckPath)
                      (activation_excludes_instruction
                        SourceOwners.terminalRecursiveFreshCheckPath false)
                      (by decide)
                  · simp

end CanonicalCompletionPlans.Terminal

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
