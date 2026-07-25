import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalCompleteness

/-!
Contract: artifact-independent physical completeness of the canonical
fixed-one Terminal encoding.

Owns: the public theorem constructing a satisfying physical assignment from
an accepted typed Terminal execution and admissible canonical codec values.

Does not own: production codecs or recipes, Rust behavior, numeric R1CS
indices, generated artifacts, or extraction.

Emits constraints: no new constraints; the witness satisfies exactly the
selected canonical receipt program.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal

namespace CanonicalTerminalCompleteness

/-- Every accepted typed Terminal execution with codec-admissible values has
a physical assignment satisfying exactly the canonical emission receipts,
while encoding the original typed input at the canonical input columns. -/
theorem physicalComplete
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (laws : FieldLaws)
    (statement : TerminalStatementFor parameters)
    (proof : TerminalProofFor parameters)
    (accepted : Accepts parameters statement proof)
    (admissible :
      AdmissibleExecution parameters profile statement proof) :
    ∃ assignment : ColumnId -> Field,
      (CanonicalTerminalSoundness.encoding
          parameters profile recipes).PhysicalSatisfies assignment ∧
        Columns.Encodes (profile.family parameters)
          (CanonicalContexts.Terminal.input parameters) assignment
          (terminalInputValues parameters statement proof) := by
  let always :=
    CanonicalCompletionPlans.Terminal.always parameters profile recipes
  let base :=
    CanonicalCompletionPlans.Terminal.onTrue parameters profile recipes
  let recursive :=
    CanonicalCompletionPlans.Terminal.onFalse parameters profile recipes
  rcases exists_visible parameters profile recipes statement proof
      admissible with
    ⟨visible⟩
  have alwaysSeparated :=
    CanonicalCompletionPlans.Terminal.always_separated
      parameters profile recipes
  have baseSeparated :=
    CanonicalCompletionPlans.Terminal.onTrue_separated
      parameters profile recipes
  have recursiveSeparated :=
    CanonicalCompletionPlans.Terminal.onFalse_separated
      parameters profile recipes
  have armCross :=
    CanonicalCompletionPlans.Terminal.arms_separated
      parameters profile recipes
  have alwaysCross :=
    CanonicalCompletionPlans.Terminal.always_arms_separated
      parameters profile recipes
  have oneVisible : oneColumn ∈ always.visibleIds := by
    simp only [always, CanonicalCompletionPlans.Terminal.always,
      ArmPlan.visibleIds, List.flatMap_cons, List.flatMap_nil,
      List.append_nil]
    change oneColumn ∈
      (PrimitivePlan.invoke
        (CanonicalTerminalPlan.selectorInvokePlan
          parameters profile recipes)).occurrence.visibleIds
    exact
      (CanonicalTerminalPlan.selectorInvokePlan
        parameters profile recipes).occurrenceOneMemVisible
  by_cases iterationZero : statement.iteration = 0
  · have trueOne :
        visible.assignment
            (activationColumn SourceOwners.terminalBranchPath true) =
          1 := by
      simpa [iterationZero] using visible.controls.2.1
    have falseZero :
        visible.assignment
            (activationColumn SourceOwners.terminalBranchPath false) =
          0 := by
      simpa [iterationZero] using visible.controls.2.2
    rcases CompletionSeparation.completeThreeGroups
        always base recursive laws
        visible.assignment visible.controls.1 trueOne falseZero
        (alwaysHonest parameters profile recipes statement proof visible)
        (baseHonestActive parameters profile recipes statement proof visible
          (baseAccepted parameters statement proof accepted iterationZero))
        (recursiveHonestInactive parameters profile recipes
          visible.assignment)
        alwaysSeparated baseSeparated recursiveSeparated armCross.1
        alwaysCross.1.1 alwaysCross.1.2
        alwaysCross.2.1 alwaysCross.2.2 oneVisible with
      ⟨assignment, agrees, alwaysRows, baseRows, recursiveRows⟩
    have alwaysAgrees :
        AgreesOn always.visibleIds visible.assignment assignment :=
      agreesOn_of_subset
        (fun id member =>
          List.mem_append_left (base.visibleIds ++ recursive.visibleIds)
            member)
        agrees
    have baseAgrees :
        AgreesOn base.visibleIds visible.assignment assignment :=
      agreesOn_of_subset
        (fun id member =>
          List.mem_append_right always.visibleIds
            (List.mem_append_left recursive.visibleIds member))
        agrees
    have recursiveAgrees :
        AgreesOn recursive.visibleIds visible.assignment assignment :=
      agreesOn_of_subset
        (fun id member =>
          List.mem_append_right always.visibleIds
            (List.mem_append_right base.visibleIds member))
        agrees
    refine ⟨assignment, ?_⟩
    exact physicalOfCompletedGroups parameters profile recipes statement
      proof visible assignment alwaysAgrees baseAgrees recursiveAgrees
      alwaysRows baseRows recursiveRows
  · have trueZero :
        visible.assignment
            (activationColumn SourceOwners.terminalBranchPath true) =
          0 := by
      simpa [iterationZero] using visible.controls.2.1
    have falseOne :
        visible.assignment
            (activationColumn SourceOwners.terminalBranchPath false) =
          1 := by
      simpa [iterationZero] using visible.controls.2.2
    rcases CompletionSeparation.completeThreeGroups
        always recursive base laws
        visible.assignment visible.controls.1 falseOne trueZero
        (alwaysHonest parameters profile recipes statement proof visible)
        (recursiveHonestActive parameters profile recipes statement proof
          visible
          (recursiveAccepted parameters statement proof accepted
            iterationZero))
        (baseHonestInactive parameters profile recipes visible.assignment)
        alwaysSeparated recursiveSeparated baseSeparated armCross.2
        alwaysCross.2.1 alwaysCross.2.2
        alwaysCross.1.1 alwaysCross.1.2 oneVisible with
      ⟨assignment, agrees, alwaysRows, recursiveRows, baseRows⟩
    have alwaysAgrees :
        AgreesOn always.visibleIds visible.assignment assignment :=
      agreesOn_of_subset
        (fun id member =>
          List.mem_append_left (recursive.visibleIds ++ base.visibleIds)
            member)
        agrees
    have recursiveAgrees :
        AgreesOn recursive.visibleIds visible.assignment assignment :=
      agreesOn_of_subset
        (fun id member =>
          List.mem_append_right always.visibleIds
            (List.mem_append_left base.visibleIds member))
        agrees
    have baseAgrees :
        AgreesOn base.visibleIds visible.assignment assignment :=
      agreesOn_of_subset
        (fun id member =>
          List.mem_append_right always.visibleIds
            (List.mem_append_right recursive.visibleIds member))
        agrees
    refine ⟨assignment, ?_⟩
    exact physicalOfCompletedGroups parameters profile recipes statement
      proof visible assignment alwaysAgrees baseAgrees recursiveAgrees
      alwaysRows baseRows recursiveRows

end CanonicalTerminalCompleteness

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
