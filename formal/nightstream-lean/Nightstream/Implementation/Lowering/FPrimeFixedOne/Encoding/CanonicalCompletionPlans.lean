import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepPlan
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalPlan
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompletionProtection

/-!
Contract: exact primitive-occurrence groups used by honest canonical
assignment construction.

Owns:
- the always-active, true-arm, and false-arm occurrence lists for Step;
- the always-active, true-arm, and false-arm occurrence lists for Terminal;
- their exact ordering inherited from the typed source programs.

Does not own: semantic execution witnesses, visible-coordinate construction,
temporary completion, branch-control rows, production Rust behavior, or
numeric R1CS indices.

Emits constraints: no new constraints; every occurrence is the occurrence of
an already selected canonical primitive receipt.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalCompletionPlans

private theorem one_excludes_instruction (path : OwnerPath) :
    oneColumn.owner ≠ .typed (.instruction path) := by
  simp [oneColumn]

private theorem activation_excludes_instruction
    (branchPath target : OwnerPath)
    (selected : Bool) :
    (activationColumn branchPath selected).owner ≠
      .typed (.instruction target) := by
  simp [activationColumn]

namespace Step

def always
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ArmPlan (signature parameters) (profile.family parameters)
      oneColumn oneColumn where
  occurrences :=
    [ (CanonicalStepPlan.applyPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalStepPlan.selectorPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalStepPlan.continuationHashPlan.{0}
        parameters profile recipes).occurrence ]

def onTrue
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    ArmPlan (signature parameters) (profile.family parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath true) where
  occurrences :=
    [ (CanonicalStepPlan.baseEqualityPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalStepPlan.baseAssertionPlan.{0}
        parameters profile).occurrence,
      (CanonicalStepPlan.baseLiteralPlan.{0}
        parameters profile defaultAdmissible).occurrence ]

def onFalse
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ArmPlan (signature parameters) (profile.family parameters)
      oneColumn (activationColumn SourceOwners.stepBranchPath false) where
  occurrences :=
    [ (CanonicalStepPlan.recursiveHashPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalStepPlan.recursiveFreshPublicPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalStepPlan.recursiveEncodePlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalStepPlan.recursiveEncodedEqualityPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalStepPlan.recursiveAssertionPlan.{0}
        parameters profile).occurrence,
      (CanonicalStepPlan.recursiveNifsPlan.{0}
        parameters profile recipes).occurrence ]

/-- The exact base-arm occurrence sequence satisfies every column-separation
premise required by constructive honest completion. -/
theorem onTrue_separated
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters)) :
    (onTrue parameters profile recipes defaultAdmissible).Separated := by
  let equality :=
    CanonicalStepPlan.baseEqualityPlan.{0}
      parameters profile recipes
  let assertion :=
    CanonicalStepPlan.baseAssertionPlan.{0} parameters profile
  let literal :=
    CanonicalStepPlan.baseLiteralPlan.{0}
      parameters profile defaultAdmissible
  change ArmPlan.SeparatedOccurrences
    [equality.occurrence, assertion.occurrence, literal.occurrence]
  have assertionTemps :
      assertion.occurrence.temporaryIds = [] := by
    rfl
  have literalTemps :
      literal.occurrence.temporaryIds = [] := by
    rfl
  apply CompletionSeparation.separatedOccurrences_of_pairwise
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with equal | equal | equal
    · subst occurrence
      exact equality.occurrenceRowsSupported
    · subst occurrence
      exact assertion.occurrenceRowsSupported
    · subst occurrence
      exact literal.occurrenceRowsSupported
  · intro occurrence member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with equal | equal | equal
    · subst occurrence
      exact equality.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact assertion.occurrenceTemporariesDisjointVisible
    · subst occurrence
      exact literal.occurrenceTemporariesDisjointVisible
  · rw [List.pairwise_cons]
    constructor
    · intro occurrence member
      simp only [List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with equal | equal
      · subst occurrence
        refine ⟨?_, ?_, ?_⟩
        · apply equality.occurrenceTemporariesDisjointOtherVisibleOfInput
            assertion
            (one_excludes_instruction
              SourceOwners.stepBaseStateEqualPath)
            (activation_excludes_instruction
              SourceOwners.stepBranchPath
              SourceOwners.stepBaseStateEqualPath true)
            (by decide)
          simpa [equality, assertion, PrimitivePlan.resultColumns,
            CanonicalContexts.Step.afterBaseEquality] using
            equality.occurrenceTemporariesDisjointResultColumns
        · intro id member
          rw [assertionTemps] at member
          simp at member
        · intro id _ member
          rw [assertionTemps] at member
          simp at member
      · subst occurrence
        refine ⟨?_, ?_, ?_⟩
        · apply equality.occurrenceTemporariesDisjointOtherVisibleOfInput
            literal
            (one_excludes_instruction
              SourceOwners.stepBaseStateEqualPath)
            (activation_excludes_instruction
              SourceOwners.stepBranchPath
              SourceOwners.stepBaseStateEqualPath true)
            (by decide)
          simpa [equality, literal, PrimitivePlan.resultColumns,
            CanonicalContexts.Step.afterBaseEquality] using
            equality.occurrenceTemporariesDisjointResultColumns
        · intro id member
          rw [literalTemps] at member
          simp at member
        · intro id _ member
          rw [literalTemps] at member
          simp at member
    · rw [List.pairwise_cons]
      constructor
      · intro occurrence member
        simp only [List.mem_singleton] at member
        subst occurrence
        constructor
        · intro id member
          rw [assertionTemps] at member
          simp at member
        · constructor
          · intro id member
            rw [literalTemps] at member
            simp at member
          · intro id member
            rw [assertionTemps] at member
            simp at member
      · simp

end Step

namespace Terminal

def always
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ArmPlan (signature parameters) (profile.family parameters)
      oneColumn oneColumn where
  occurrences :=
    [ (CanonicalTerminalPlan.selectorPlan.{0}
        parameters profile recipes).occurrence ]

def onTrue
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ArmPlan (signature parameters) (profile.family parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath true) where
  occurrences :=
    [ (CanonicalTerminalPlan.baseEqualityPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalTerminalPlan.baseAssertionPlan.{0}
        parameters profile).occurrence ]

def onFalse
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ArmPlan (signature parameters) (profile.family parameters)
      oneColumn (activationColumn SourceOwners.terminalBranchPath false) where
  occurrences :=
    [ (CanonicalTerminalPlan.recursiveHashPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalTerminalPlan.recursiveFreshPublicPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalTerminalPlan.recursiveEncodePlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalTerminalPlan.recursiveEncodedEqualityPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalTerminalPlan.recursivePriorAssertionPlan.{0}
        parameters profile).occurrence,
      (CanonicalTerminalPlan.recursiveRunningCheckPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalTerminalPlan.recursiveRunningAssertionPlan.{0}
        parameters profile).occurrence,
      (CanonicalTerminalPlan.recursiveFreshCheckPlan.{0}
        parameters profile recipes).occurrence,
      (CanonicalTerminalPlan.recursiveFreshAssertionPlan.{0}
        parameters profile).occurrence ]

end Terminal

end CanonicalCompletionPlans

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
