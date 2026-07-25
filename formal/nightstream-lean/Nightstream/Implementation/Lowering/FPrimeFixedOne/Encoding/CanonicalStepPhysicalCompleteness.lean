import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepAlwaysSeparation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepSemanticCompleteness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepSoundness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ThreeGroupCompletion

/-!
Contract: artifact-independent physical completeness of the canonical
fixed-one Step encoding.

Owns:
- activation and one-port join completion from exact typed Step values;
- exact reconstruction of the selected receipt order;
- a satisfying physical assignment for every accepted typed Step execution.

Does not own: production codecs or recipes, Rust behavior, numeric R1CS
indices, generated artifacts, or extraction.

Emits constraints: no new constraints; the witness satisfies exactly the
selected canonical receipt program.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Step

namespace CanonicalStepCompleteness

/-- Completed occurrence groups, unchanged visible coordinates, branch
activation, and the one-port mux satisfy exactly the canonical Step receipts. -/
theorem physicalOfCompletedGroups
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (input : StepInputFor parameters)
    (runningNext : parameters.Running)
    (visible : VisibleWitness parameters profile input runningNext)
    (assignment : ColumnId -> Field)
    (alwaysAgrees :
      AgreesOn
        (CanonicalCompletionPlans.Step.always
          parameters profile recipes).visibleIds
        visible.assignment assignment)
    (baseAgrees :
      AgreesOn
        (CanonicalCompletionPlans.Step.onTrue
          parameters profile recipes defaultAdmissible).visibleIds
        visible.assignment assignment)
    (recursiveAgrees :
      AgreesOn
        (CanonicalCompletionPlans.Step.onFalse
          parameters profile recipes).visibleIds
        visible.assignment assignment)
    (alwaysRows :
      Satisfies
        (CanonicalCompletionPlans.Step.always
          parameters profile recipes).rows assignment)
    (baseRows :
      Satisfies
        (CanonicalCompletionPlans.Step.onTrue
          parameters profile recipes defaultAdmissible).rows assignment)
    (recursiveRows :
      Satisfies
        (CanonicalCompletionPlans.Step.onFalse
          parameters profile recipes).rows assignment)
    (baseRunning :
      input.iteration = 0 ->
        runningNext = defaultRunning parameters) :
    (CanonicalStepSoundness.encoding
        parameters profile recipes defaultAdmissible
      ).PhysicalSatisfies assignment ∧
      Columns.Encodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input) ∧
      Columns.Encodes (profile.family parameters)
        (CanonicalContexts.Step.result parameters) assignment
        (resultValuesFor parameters input runningNext) := by
  let always :=
    CanonicalCompletionPlans.Step.always parameters profile recipes
  let base :=
    CanonicalCompletionPlans.Step.onTrue
      parameters profile recipes defaultAdmissible
  let recursive :=
    CanonicalCompletionPlans.Step.onFalse parameters profile recipes
  let applyView :=
    CanonicalStepConstructionPlans.apply parameters profile recipes
  let selectorView :=
    CanonicalStepConstructionPlans.selector parameters profile recipes
  let baseEqualityView :=
    CanonicalStepConstructionPlans.baseEquality parameters profile recipes
  let baseLiteralView :=
    CanonicalStepConstructionPlans.baseLiteral
      parameters profile defaultAdmissible
  let recursiveHashView :=
    CanonicalStepConstructionPlans.recursiveHash parameters profile recipes
  let recursiveNifsView :=
    CanonicalStepConstructionPlans.recursiveNifs parameters profile recipes
  let continuationView :=
    CanonicalStepConstructionPlans.continuationHash
      parameters profile recipes
  let applyPlan :=
    CanonicalStepPlan.applyPlan parameters profile recipes
  let selectorPlan :=
    CanonicalStepPlan.selectorPlan parameters profile recipes
  let continuationPlan :=
    CanonicalStepPlan.continuationHashPlan parameters profile recipes

  have oneInAlways : oneColumn ∈ always.visibleIds := by
    change oneColumn ∈
      [applyPlan.occurrence, selectorPlan.occurrence,
        continuationPlan.occurrence].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨applyPlan.occurrence, by simp, ?_⟩
    change oneColumn ∈
      (PrimitivePlan.invoke applyView).occurrence.visibleIds
    exact applyView.occurrenceOneMemVisible
  have trueInBase :
      activationColumn SourceOwners.stepBranchPath true ∈
        base.visibleIds := by
    change activationColumn SourceOwners.stepBranchPath true ∈
      [ (CanonicalStepPlan.baseEqualityPlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.baseAssertionPlan
          parameters profile).occurrence,
        (CanonicalStepPlan.baseLiteralPlan
          parameters profile defaultAdmissible).occurrence
      ].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨
      (CanonicalStepPlan.baseEqualityPlan
        parameters profile recipes).occurrence,
      List.mem_cons_self, ?_⟩
    change activationColumn SourceOwners.stepBranchPath true ∈
      (PrimitivePlan.invoke baseEqualityView).occurrence.visibleIds
    exact baseEqualityView.occurrenceActiveMemVisible
  have falseInRecursive :
      activationColumn SourceOwners.stepBranchPath false ∈
        recursive.visibleIds := by
    change activationColumn SourceOwners.stepBranchPath false ∈
      [ (CanonicalStepPlan.recursiveHashPlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.recursiveFreshPublicPlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.recursiveEncodePlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.recursiveEncodedEqualityPlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.recursiveAssertionPlan
          parameters profile).occurrence,
        (CanonicalStepPlan.recursiveNifsPlan
          parameters profile recipes).occurrence
      ].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨
      (CanonicalStepPlan.recursiveHashPlan
        parameters profile recipes).occurrence,
      List.mem_cons_self, ?_⟩
    change activationColumn SourceOwners.stepBranchPath false ∈
      (PrimitivePlan.invoke recursiveHashView).occurrence.visibleIds
    exact recursiveHashView.occurrenceActiveMemVisible
  have controlsAgree :
      AgreesOn
        [oneColumn,
          activationColumn SourceOwners.stepBranchPath true,
          activationColumn SourceOwners.stepBranchPath false]
        visible.assignment assignment := by
    intro id member
    simp only [List.mem_cons, List.not_mem_nil, or_false] at member
    rcases member with oneMember | trueMember | falseMember
    · subst id
      exact alwaysAgrees oneColumn oneInAlways
    · subst id
      exact baseAgrees _ trueInBase
    · subst id
      exact recursiveAgrees _ falseInRecursive
  have controls := visible.controls.of_agrees controlsAgree

  have inputAgrees :
      AgreesOn
        (CanonicalContexts.Step.input parameters).toSchemaBundles.ids
        visible.assignment assignment := by
    apply agreesOn_of_subset _ alwaysAgrees
    intro id member
    change id ∈
      [applyPlan.occurrence, selectorPlan.occurrence,
        continuationPlan.occurrence].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨applyPlan.occurrence, by simp, ?_⟩
    change id ∈
      (PrimitivePlan.invoke applyView).occurrence.visibleIds
    exact applyView.occurrenceInputIdsSubsetVisible id member
  have inputEncoded :
      Columns.Encodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input) := by
    apply
      (CanonicalContexts.Step.input parameters
        ).toSchemaBundles.encodes_of_agrees
          (profile.family parameters) visible.assignment assignment
          (stepInputValues parameters input)
    · exact inputAgrees
    · exact visible.inputEncoded

  have finalAgrees :
      AgreesOn
        (CanonicalContexts.Step.afterHashNext parameters
          ).toSchemaBundles.ids
        visible.assignment assignment := by
    apply agreesOn_of_subset _ alwaysAgrees
    intro id member
    change id ∈
      [applyPlan.occurrence, selectorPlan.occurrence,
        continuationPlan.occurrence].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨continuationPlan.occurrence, by simp, ?_⟩
    change id ∈
      (PrimitivePlan.invoke continuationView).occurrence.visibleIds
    apply InvokePlan.occurrenceResultIdsSubsetVisible.{0, 0}
      continuationView id
    simpa [PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterHashNext] using member
  have finalEncoded :
      Columns.Encodes (profile.family parameters)
        (CanonicalContexts.Step.afterHashNext parameters) assignment
        (afterHashNextValues parameters input runningNext) := by
    apply
      (CanonicalContexts.Step.afterHashNext parameters
        ).toSchemaBundles.encodes_of_agrees
          (profile.family parameters) visible.assignment assignment
          (afterHashNextValues parameters input runningNext)
    · exact finalAgrees
    · exact visible.finalEncoded
  have resultEncoded :
      Columns.Encodes (profile.family parameters)
        (CanonicalContexts.Step.result parameters) assignment
        (resultValuesFor parameters input runningNext) := by
    have exported :=
      Columns.export_encodes
        (profile.family parameters) assignment
        (CanonicalContexts.Step.resultExports parameters)
        (CanonicalContexts.Step.resultExportsCompatible parameters)
        (CanonicalContexts.Step.afterHashNext parameters)
        (afterHashNextValues parameters input runningNext)
        finalEncoded
    simpa [CanonicalContexts.Step.result,
      CanonicalContexts.Step.resultExports] using exported

  have commonDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.common parameters)
        visible.assignment (commonValues parameters input) :=
    (CanonicalContexts.Step.common parameters
      ).toSchemaBundles.decodes_of_encodes
        (profile.family parameters) visible.assignment
        (commonValues parameters input) visible.commonEncoded
  have selectorDecodedAtVisible :
      boolCodec.decode
          [visible.assignment
            (CanonicalContexts.Step.selector parameters profile)] =
        some (decide (input.iteration = 0)) := by
    simpa [CanonicalContexts.Step.selector, commonValues,
      commonValuesWith] using
      CanonicalStepSoundness.decodedBitReference
        parameters profile visible.assignment
        (CanonicalContexts.Step.common parameters)
        (commonValues parameters input)
        (CommonRefs.iterationZero parameters)
        (CanonicalContexts.Step.commonWidths parameters profile)
        commonDecoded
  have selectorInBase :
      CanonicalContexts.Step.selector parameters profile ∈
        base.visibleIds := by
    change CanonicalContexts.Step.selector parameters profile ∈
      [ (CanonicalStepPlan.baseEqualityPlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.baseAssertionPlan
          parameters profile).occurrence,
        (CanonicalStepPlan.baseLiteralPlan
          parameters profile defaultAdmissible).occurrence
      ].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨
      (CanonicalStepPlan.baseEqualityPlan
        parameters profile recipes).occurrence,
      List.mem_cons_self, ?_⟩
    change CanonicalContexts.Step.selector parameters profile ∈
      (PrimitivePlan.invoke baseEqualityView).occurrence.visibleIds
    apply baseEqualityView.occurrenceInputIdsSubsetVisible
    exact CanonicalPrimitivePlan.bitCoordinate_mem profile
      (CommonRefs.iterationZero parameters)
      (CanonicalContexts.Step.common parameters)
      (CanonicalContexts.Step.commonWidths parameters profile)
  have selectorDecoded :
      boolCodec.decode
          [assignment
            (CanonicalContexts.Step.selector parameters profile)] =
        some (decide (input.iteration = 0)) := by
    rw [baseAgrees _ selectorInBase]
    exact selectorDecodedAtVisible

  let activation :=
    CanonicalBranchPlan.activationRecipe
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector parameters profile)
  have activationRows : Satisfies activation.rows assignment := by
    by_cases iterationZero : input.iteration = 0
    · apply activation.selected_true_complete assignment controls.1
      · simpa [activation, iterationZero] using selectorDecoded
      · have trueOne :
            assignment
                (activationColumn SourceOwners.stepBranchPath true) =
              1 := by
          simpa [iterationZero] using controls.2.1
        have falseZero :
            assignment
                (activationColumn SourceOwners.stepBranchPath false) =
              0 := by
          simpa [iterationZero] using controls.2.2
        exact ⟨by simpa [activation] using trueOne.trans controls.1.symm,
          by simpa [activation] using falseZero⟩
    · apply activation.selected_false_complete assignment controls.1
      · simpa [activation, iterationZero] using selectorDecoded
      · have trueZero :
            assignment
                (activationColumn SourceOwners.stepBranchPath true) =
              0 := by
          simpa [iterationZero] using controls.2.1
        have falseOne :
            assignment
                (activationColumn SourceOwners.stepBranchPath false) =
              1 := by
          simpa [iterationZero] using controls.2.2
        exact ⟨by simpa [activation] using trueZero,
          by simpa [activation] using falseOne.trans controls.1.symm⟩

  have baseBundleAgrees :
      AgreesOn
        (CanonicalContexts.Step.baseRunning parameters
          ).toColumnBundle.ids
        visible.assignment assignment := by
    apply agreesOn_of_subset _ baseAgrees
    intro id member
    change id ∈
      [ (CanonicalStepPlan.baseEqualityPlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.baseAssertionPlan
          parameters profile).occurrence,
        (CanonicalStepPlan.baseLiteralPlan
          parameters profile defaultAdmissible).occurrence
      ].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨
      (CanonicalStepPlan.baseLiteralPlan
        parameters profile defaultAdmissible).occurrence,
      List.mem_cons_of_mem _
        (List.mem_cons_of_mem _ List.mem_cons_self), ?_⟩
    change id ∈
      (PrimitivePlan.literal baseLiteralView).occurrence.visibleIds
    apply baseLiteralView.occurrenceOutputIdsSubsetVisible
    rw [baseLiteralView.outputExact]
    simpa [CanonicalContexts.Step.baseRunning] using member
  have recursiveBundleAgrees :
      AgreesOn
        (CanonicalContexts.Step.recursiveRunning parameters
          ).toColumnBundle.ids
        visible.assignment assignment := by
    apply agreesOn_of_subset _ recursiveAgrees
    intro id member
    change id ∈
      [ (CanonicalStepPlan.recursiveHashPlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.recursiveFreshPublicPlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.recursiveEncodePlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.recursiveEncodedEqualityPlan
          parameters profile recipes).occurrence,
        (CanonicalStepPlan.recursiveAssertionPlan
          parameters profile).occurrence,
        (CanonicalStepPlan.recursiveNifsPlan
          parameters profile recipes).occurrence
      ].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨
      (CanonicalStepPlan.recursiveNifsPlan
        parameters profile recipes).occurrence,
      List.mem_cons_of_mem _
        (List.mem_cons_of_mem _
          (List.mem_cons_of_mem _
            (List.mem_cons_of_mem _
              (List.mem_cons_of_mem _ List.mem_cons_self)))), ?_⟩
    change id ∈
      (PrimitivePlan.invoke recursiveNifsView).occurrence.visibleIds
    apply InvokePlan.occurrenceResultIdsSubsetVisible.{0, 0}
      recursiveNifsView id
    change id ∈
      (Columns.toSchemaBundles
        (HVec.append
          (instructionColumns SourceOwners.stepRecursiveNifsPath
            [Ports.committedRunning parameters])
          (CanonicalContexts.Step.afterEncodedEquality parameters))).ids
    rw [Columns.append_ids]
    apply List.mem_append_left
    rw [ReceiptScoping.singletonColumnsIds
      (instructionColumns SourceOwners.stepRecursiveNifsPath
        [Ports.committedRunning parameters])]
    exact member
  have joinedBundleAgrees :
      AgreesOn
        (HVec.head (CanonicalContexts.Step.joined parameters)
          ).toColumnBundle.ids
        visible.assignment assignment := by
    apply agreesOn_of_subset _ alwaysAgrees
    intro id member
    change id ∈
      [applyPlan.occurrence, selectorPlan.occurrence,
        continuationPlan.occurrence].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨continuationPlan.occurrence, by simp, ?_⟩
    change id ∈
      (PrimitivePlan.invoke continuationView).occurrence.visibleIds
    apply continuationView.occurrenceInputIdsSubsetVisible id
    change id ∈
      (Columns.toSchemaBundles
        ((CanonicalContexts.Step.joined parameters).append
          (CanonicalContexts.Step.common parameters))).ids
    rw [Columns.append_ids]
    apply List.mem_append_left
    exact Eq.mp
      (congrArg (fun ids => id ∈ ids)
        (ReceiptScoping.singletonColumnsIds
          (CanonicalContexts.Step.joined parameters))).symm
      member

  have baseEncodedAtVisible :
      (CanonicalContexts.Step.baseRunning parameters
        ).toColumnBundle.Encodes
          (profile.family parameters) (.data .running)
          visible.assignment (defaultRunning parameters) := by
    simpa [CanonicalContexts.Step.afterBaseLiteral,
      CanonicalContexts.Step.baseRunning] using visible.baseEncoded.1
  have baseEncoded :
      (CanonicalContexts.Step.baseRunning parameters
        ).toColumnBundle.Encodes
          (profile.family parameters) (.data .running)
          assignment (defaultRunning parameters) :=
    (CanonicalContexts.Step.baseRunning parameters
      ).toColumnBundle.encodes_of_agrees
        (profile.family parameters) (.data .running)
        visible.assignment assignment (defaultRunning parameters)
        baseBundleAgrees baseEncodedAtVisible
  have recursiveEncodedAtVisible :
      (CanonicalContexts.Step.recursiveRunning parameters
        ).toColumnBundle.Encodes
          (profile.family parameters) (.data .running)
          visible.assignment runningNext := by
    simpa [CanonicalContexts.Step.afterNifs,
      CanonicalContexts.Step.recursiveRunning] using visible.recursiveEncoded.1
  have recursiveEncoded :
      (CanonicalContexts.Step.recursiveRunning parameters
        ).toColumnBundle.Encodes
          (profile.family parameters) (.data .running)
          assignment runningNext :=
    (CanonicalContexts.Step.recursiveRunning parameters
      ).toColumnBundle.encodes_of_agrees
        (profile.family parameters) (.data .running)
        visible.assignment assignment runningNext
        recursiveBundleAgrees recursiveEncodedAtVisible
  have joinedEncodedAtVisible :
      (HVec.head (CanonicalContexts.Step.joined parameters)
        ).toColumnBundle.Encodes
          (profile.family parameters) (.data .running)
          visible.assignment runningNext := by
    exact visible.joinedEncoded.1
  have joinedEncoded :
      (HVec.head (CanonicalContexts.Step.joined parameters)
        ).toColumnBundle.Encodes
          (profile.family parameters) (.data .running)
          assignment runningNext :=
    (HVec.head (CanonicalContexts.Step.joined parameters)
      ).toColumnBundle.encodes_of_agrees
        (profile.family parameters) (.data .running)
        visible.assignment assignment runningNext
        joinedBundleAgrees joinedEncodedAtVisible

  let mux :=
    CanonicalBranchPlan.onePortJoinRecipe
      SourceOwners.stepBranchPath
      (CanonicalContexts.Step.selector parameters profile)
      (Ports.committedRunning parameters)
      (CanonicalContexts.Step.baseRunning parameters)
      (CanonicalContexts.Step.recursiveRunning parameters)
  have joinedCoordinates :
      mux.joined.values assignment =
        ((profile.family parameters).codecFor
          (.data .running)).encode runningNext := by
    simpa [mux, CanonicalBranchPlan.onePortJoinRecipe] using
      joinedEncoded.2
  have baseCoordinates :
      mux.onTrue.values assignment =
        ((profile.family parameters).codecFor
          (.data .running)).encode (defaultRunning parameters) := by
    simpa [mux, CanonicalBranchPlan.onePortJoinRecipe] using
      baseEncoded.2
  have recursiveCoordinates :
      mux.onFalse.values assignment =
        ((profile.family parameters).codecFor
          (.data .running)).encode runningNext := by
    simpa [mux, CanonicalBranchPlan.onePortJoinRecipe] using
      recursiveEncoded.2
  have muxRows : Satisfies mux.rows assignment := by
    by_cases iterationZero : input.iteration = 0
    · apply mux.selected_true_complete assignment
      · simpa [mux, iterationZero] using selectorDecoded
      · calc
          mux.joined.values assignment =
              ((profile.family parameters).codecFor
                (.data .running)).encode runningNext :=
            joinedCoordinates
          _ = ((profile.family parameters).codecFor
                (.data .running)).encode
                  (defaultRunning parameters) := by
            rw [baseRunning iterationZero]
          _ = mux.onTrue.values assignment :=
            baseCoordinates.symm
    · apply mux.selected_false_complete assignment
      · simpa [mux, iterationZero] using selectorDecoded
      · exact joinedCoordinates.trans recursiveCoordinates.symm

  have alwaysRowsExact :
      Satisfies
        (applyPlan.receipt.rows ++
          (selectorPlan.receipt.rows ++
            continuationPlan.receipt.rows))
        assignment := by
    simpa [always, CanonicalCompletionPlans.Step.always,
      ArmPlan.rows] using alwaysRows
  have splitApply :=
    (satisfies_append_iff applyPlan.receipt.rows
      (selectorPlan.receipt.rows ++ continuationPlan.receipt.rows)
      assignment).1 alwaysRowsExact
  have splitSelector :=
    (satisfies_append_iff selectorPlan.receipt.rows
      continuationPlan.receipt.rows assignment).1 splitApply.2
  have bodyRows :
      Satisfies
        (applyPlan.receipt.rows ++
          (selectorPlan.receipt.rows ++
            (activation.rows ++
              (base.rows ++
                (recursive.rows ++
                  (mux.rows ++ continuationPlan.receipt.rows))))))
        assignment := by
    apply (satisfies_append_iff applyPlan.receipt.rows _ assignment).2
    refine ⟨splitApply.1, ?_⟩
    apply (satisfies_append_iff selectorPlan.receipt.rows _ assignment).2
    refine ⟨splitSelector.1, ?_⟩
    apply (satisfies_append_iff activation.rows _ assignment).2
    refine ⟨activationRows, ?_⟩
    apply (satisfies_append_iff base.rows _ assignment).2
    refine ⟨baseRows, ?_⟩
    apply (satisfies_append_iff recursive.rows _ assignment).2
    refine ⟨recursiveRows, ?_⟩
    exact (satisfies_append_iff mux.rows
      continuationPlan.receipt.rows assignment).2
        ⟨muxRows, splitSelector.2⟩

  refine ⟨?_, inputEncoded, resultEncoded⟩
  constructor
  · rw [(CanonicalStepSoundness.encoding
      parameters profile recipes defaultAdmissible).oneExact]
    exact controls.1
  · change Satisfies
      ((CanonicalStepPlan.receipts
        parameters profile recipes defaultAdmissible
        ).flatMap fun receipt => receipt.rows) assignment
    simpa [CanonicalStepPlan.receipts,
      CanonicalStepPlan.bodyReceipts,
      always, base, recursive,
      CanonicalCompletionPlans.Step.always,
      CanonicalCompletionPlans.Step.onTrue,
      CanonicalCompletionPlans.Step.onFalse,
      ArmPlan.rows, CanonicalBranchPlan.activation_rows_conserved,
      CanonicalBranchPlan.onePortJoinReceipt,
      InputReceipts.rows_empty] using bodyRows

/-- Every accepted typed Step execution with codec-admissible values has a
physical assignment satisfying exactly the canonical emission receipts,
while encoding the original typed input and accepted result. -/
theorem physicalComplete
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (input : StepInputFor parameters)
    (output : StepOutputFor parameters)
    (accepted : Accepts parameters input output)
    (admissible :
      AdmissibleExecution parameters profile input
        (selectedRunning output)) :
    ∃ assignment : ColumnId -> Field,
      (CanonicalStepSoundness.encoding
          parameters profile recipes defaultAdmissible
        ).PhysicalSatisfies assignment ∧
        Columns.Encodes (profile.family parameters)
          (CanonicalContexts.Step.input parameters) assignment
          (stepInputValues parameters input) ∧
        Columns.Encodes (profile.family parameters)
          (CanonicalContexts.Step.result parameters) assignment
          (stepResultValues parameters output) := by
  let always :=
    CanonicalCompletionPlans.Step.always parameters profile recipes
  let base :=
    CanonicalCompletionPlans.Step.onTrue
      parameters profile recipes defaultAdmissible
  let recursive :=
    CanonicalCompletionPlans.Step.onFalse parameters profile recipes
  let applyView :=
    CanonicalStepConstructionPlans.apply parameters profile recipes
  let applyPlan :=
    CanonicalStepPlan.applyPlan parameters profile recipes
  let selectorPlan :=
    CanonicalStepPlan.selectorPlan parameters profile recipes
  let continuationPlan :=
    CanonicalStepPlan.continuationHashPlan parameters profile recipes
  rcases exists_visible parameters profile recipes defaultAdmissible input
      (selectedRunning output) admissible with
    ⟨visible⟩
  have alwaysSeparated :=
    CanonicalCompletionPlans.Step.always_separated
      parameters profile recipes
  have baseSeparated :=
    CanonicalCompletionPlans.Step.onTrue_separated
      parameters profile recipes defaultAdmissible
  have recursiveSeparated :=
    CanonicalCompletionPlans.Step.onFalse_separated
      parameters profile recipes
  have armCross :=
    CanonicalCompletionPlans.Step.arms_separated
      parameters profile recipes defaultAdmissible
  have alwaysCross :=
    CanonicalCompletionPlans.Step.always_arms_separated
      parameters profile recipes defaultAdmissible
  have oneVisible : oneColumn ∈ always.visibleIds := by
    change oneColumn ∈
      [applyPlan.occurrence, selectorPlan.occurrence,
        continuationPlan.occurrence].flatMap ArmOccurrence.visibleIds
    apply List.mem_flatMap.mpr
    refine ⟨applyPlan.occurrence, List.mem_cons_self, ?_⟩
    change oneColumn ∈
      (PrimitivePlan.invoke applyView).occurrence.visibleIds
    exact applyView.occurrenceOneMemVisible
  by_cases iterationZero : input.iteration = 0
  · have trueOne :
        visible.assignment
            (activationColumn SourceOwners.stepBranchPath true) =
          1 := by
      simpa [iterationZero] using visible.controls.2.1
    have falseZero :
        visible.assignment
            (activationColumn SourceOwners.stepBranchPath false) =
          0 := by
      simpa [iterationZero] using visible.controls.2.2
    have baseCondition :=
      baseAccepted parameters input output accepted iterationZero
    rcases CompletionSeparation.completeThreeGroups
        always base recursive laws
        visible.assignment visible.controls.1 trueOne falseZero
        (alwaysHonest parameters profile recipes input
          (selectedRunning output) visible)
        (baseHonestActive parameters profile recipes defaultAdmissible
          input (selectedRunning output) visible baseCondition.1)
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
    rcases physicalOfCompletedGroups parameters profile recipes
        defaultAdmissible input (selectedRunning output) visible assignment
        alwaysAgrees baseAgrees recursiveAgrees
        alwaysRows baseRows recursiveRows
        (fun _ =>
          baseSelectedRunning parameters input output accepted iterationZero)
        with
      ⟨physical, inputEncoded, resultEncoded⟩
    refine ⟨assignment, physical, inputEncoded, ?_⟩
    simpa only [acceptedResultValues parameters input output accepted] using
      resultEncoded
  · have trueZero :
        visible.assignment
            (activationColumn SourceOwners.stepBranchPath true) =
          0 := by
      simpa [iterationZero] using visible.controls.2.1
    have falseOne :
        visible.assignment
            (activationColumn SourceOwners.stepBranchPath false) =
          1 := by
      simpa [iterationZero] using visible.controls.2.2
    have recursiveCondition :=
      recursiveSelectedRunning parameters input output accepted iterationZero
    rcases CompletionSeparation.completeThreeGroups
        always recursive base laws
        visible.assignment visible.controls.1 falseOne trueZero
        (alwaysHonest parameters profile recipes input
          (selectedRunning output) visible)
        (recursiveHonestActive parameters profile recipes input
          (selectedRunning output) visible recursiveCondition)
        (baseHonestInactive parameters profile recipes defaultAdmissible
          input (selectedRunning output) visible)
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
    rcases physicalOfCompletedGroups parameters profile recipes
        defaultAdmissible input (selectedRunning output) visible assignment
        alwaysAgrees baseAgrees recursiveAgrees
        alwaysRows baseRows recursiveRows
        (fun zero => (iterationZero zero).elim) with
      ⟨physical, inputEncoded, resultEncoded⟩
    refine ⟨assignment, physical, inputEncoded, ?_⟩
    simpa only [acceptedResultValues parameters input output accepted] using
      resultEncoded

end CanonicalStepCompleteness

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
