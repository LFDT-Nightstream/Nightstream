import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepEvidence
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitiveRefinement

/-!
Contract: artifact-independent row soundness for the selected canonical
fixed-one Step encoding.

Owns:
- projection of the exact receipt-owned row program to each selected
  primitive occurrence;
- reconstruction of the typed Step execution from decoded input coordinates;
- agreement of every satisfying physical assignment with the canonical
  executable Step checker.

Does not own: a production codec/recipe instantiation, Rust source semantics,
numeric R1CS matrices, generated-row equality, or honest assignment
construction.

This module consumes only the selected receipt program and the certified
primitive recipes.  Generated artifacts and historical dimensions are absent.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalStepSoundness

private theorem receiptRows
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (assignment : ColumnId -> Field)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (receipt : InstructionReceipt)
    (member :
      receipt ∈
        CanonicalStepPlan.receipts
          parameters profile recipes defaultAdmissible)
    (notRecursive : receipt.owner ≠ recursiveNifsOwner) :
    Satisfies receipt.rows assignment := by
  exact evidence.ordinaryRows receipt member notRecursive

private theorem bodyReceiptMember
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (receipt : InstructionReceipt)
    (member :
      receipt ∈
        CanonicalStepPlan.bodyReceipts
          parameters profile recipes defaultAdmissible) :
    receipt ∈
      CanonicalStepPlan.receipts
        parameters profile recipes defaultAdmissible := by
  rw [CanonicalStepPlan.receipts]
  apply List.mem_cons_of_mem
  exact List.mem_append_right _ member

private theorem planRows
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (assignment : ColumnId -> Field)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    {input output : Schema (typeSystem parameters)}
    {primitive : Primitive (signature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active)
    (member :
      plan.receipt ∈
        CanonicalStepPlan.receipts
          parameters profile recipes defaultAdmissible)
    (notRecursive : path ≠ SourceOwners.stepRecursiveNifsPath) :
    Satisfies plan.receipt.rows assignment := by
  exact receiptRows parameters profile recipes defaultAdmissible
    assignment evidence plan.receipt member
      (by
        rw [PrimitivePlan.receipt_owner]
        simpa [recursiveNifsOwner] using notRecursive)

private theorem bodyPlanRows
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (assignment : ColumnId -> Field)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    {input output : Schema (typeSystem parameters)}
    {primitive : Primitive (signature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active)
    (member :
      plan.receipt ∈
        CanonicalStepPlan.bodyReceipts
          parameters profile recipes defaultAdmissible)
    (notRecursive : path ≠ SourceOwners.stepRecursiveNifsPath) :
    Satisfies plan.receipt.rows assignment := by
  apply planRows parameters profile recipes defaultAdmissible
    assignment evidence plan
  exact bodyReceiptMember parameters profile recipes defaultAdmissible
    plan.receipt member
  exact notRecursive

theorem decodedBitReference
    (parameters : Parameters)
    (profile : Profile parameters)
    (assignment : ColumnId -> Field)
    {schema : Schema (typeSystem parameters)}
    (columns : Columns schema)
    (values : Schema.Values (typeSystem parameters) schema)
    (condition : Ref (typeSystem parameters) schema .bit)
    (widths :
      SchemaWidthAgrees (profile.family parameters) schema)
    (decoded :
      Columns.Decodes (profile.family parameters)
        columns assignment values) :
    boolCodec.decode
        [assignment
          (CanonicalPrimitivePlan.bitCoordinate
            profile condition columns widths)] =
      some (condition.get values) := by
  have selected :=
    SchemaBundles.get_decodes
      (profile.family parameters) assignment
      condition columns.toSchemaBundles values decoded
  unfold ColumnBundle.Decodes at selected
  change
    boolCodec.decode
        ((columns.toSchemaBundles.get condition).values assignment) =
      some (condition.get values) at selected
  rw [ColumnBundle.values_eq_ids_map,
    CanonicalPrimitivePlan.bitReferenceIdsExact
      profile condition columns widths] at selected
  exact selected

private theorem decodedOfExactExecution
    {parameters : Parameters}
    {profile : Profile parameters}
    {input output : Schema (typeSystem parameters)}
    {primitive : Primitive (signature parameters) input output}
    {path : OwnerPath}
    {inputColumns : Columns input}
    {one active : ColumnId}
    (plan :
      PrimitivePlan parameters profile primitive path
        inputColumns one active)
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (constantOne : assignment one = 1)
    (activeOne : assignment active = 1)
    (source : Schema.Values (typeSystem parameters) input)
    (expected : Schema.Values (typeSystem parameters) output)
    (sourceDecoded :
      Columns.Decodes (profile.family parameters)
        inputColumns assignment source)
    (rowsHold : Satisfies plan.receipt.rows assignment)
    (executed : primitive.exec source = some expected) :
    Columns.Decodes (profile.family parameters)
      plan.resultColumns assignment expected := by
  rcases plan.activeSound laws assignment constantOne activeOne
      source sourceDecoded rowsHold with
    ⟨result, semantic, resultDecoded⟩
  have resultExecuted := primitive.complete source result semantic
  rw [executed] at resultExecuted
  have equal : result = expected :=
    Option.some.inj resultExecuted.symm
  subst result
  exact resultDecoded

/-- The two unconditional prefix calls of every satisfying Step assignment
decode to the exact typed common branch context. -/
private theorem commonDecodedFromEvidence
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input)) :
    Columns.Decodes (profile.family parameters)
      (CanonicalContexts.Step.common parameters) assignment
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonValues
        parameters input) := by
  let applyPlan := CanonicalStepPlan.applyPlan parameters profile recipes
  have applyRows : Satisfies applyPlan.receipt.rows assignment :=
    planRows parameters profile recipes defaultAdmissible assignment
      evidence applyPlan (by
        rw [CanonicalStepPlan.receipts]
        apply List.mem_cons_of_mem
        apply List.mem_append_right
        rw [CanonicalStepPlan.bodyReceipts]
        exact List.mem_cons_self) (by decide)
  have afterStepDecoded :=
    decodedOfExactExecution applyPlan laws assignment
      evidence.constantOne evidence.constantOne
      (stepInputValues parameters input)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterStepValues
        parameters input)
      inputDecoded applyRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.stepCall_exec
        parameters input)
  have afterStepDecoded' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.afterStep parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterStepValues
          parameters input) := by
    simpa [applyPlan, CanonicalStepPlan.applyPlan,
      PrimitivePlan.resultColumns, CanonicalContexts.Step.afterStep] using
        afterStepDecoded
  let selectorPlan :=
    CanonicalStepPlan.selectorPlan parameters profile recipes
  have selectorRows : Satisfies selectorPlan.receipt.rows assignment :=
    planRows parameters profile recipes defaultAdmissible assignment
      evidence selectorPlan (by
        rw [CanonicalStepPlan.receipts]
        apply List.mem_cons_of_mem
        apply List.mem_append_right
        rw [CanonicalStepPlan.bodyReceipts]
        exact List.mem_cons_of_mem _ List.mem_cons_self) (by decide)
  have common :=
    decodedOfExactExecution selectorPlan laws assignment
      evidence.constantOne evidence.constantOne
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterStepValues
        parameters input)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonValues
        parameters input)
      afterStepDecoded' selectorRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.iterationZeroCall_exec
        parameters input)
  simpa [selectorPlan, CanonicalStepPlan.selectorPlan,
    PrimitivePlan.resultColumns, CanonicalContexts.Step.common] using common

/-- The receipt-owned activation rows derive both arm activations from the
internally computed selector; neither activation is a prover-selected branch
premise. -/
private theorem branchControlsFromEvidence
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input)) :
    boolCodec.decode
        [assignment (CanonicalContexts.Step.selector parameters profile)] =
        some (decide (input.iteration = 0)) ∧
      assignment
          (activationColumn SourceOwners.stepBranchPath true) =
        (if input.iteration = 0 then 1 else 0) ∧
      assignment
          (activationColumn SourceOwners.stepBranchPath false) =
        (if input.iteration = 0 then 0 else 1) := by
  have common :=
    commonDecodedFromEvidence parameters profile recipes defaultAdmissible laws
      assignment input evidence inputDecoded
  have selectorDecoded :
      boolCodec.decode
          [assignment (CanonicalContexts.Step.selector parameters profile)] =
        some (decide (input.iteration = 0)) := by
    simpa [CanonicalContexts.Step.selector,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonValues,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonValuesWith]
      using
        decodedBitReference parameters profile assignment
          (CanonicalContexts.Step.common parameters)
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonValues
            parameters input)
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iterationZero
            parameters)
          (CanonicalContexts.Step.commonWidths parameters profile)
          common
  let activation :=
    CanonicalBranchPlan.activationRecipe
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector parameters profile)
  have trueRows :
      Satisfies
        (CanonicalBranchPlan.trueActivationReceipt
          SourceOwners.stepBranchPath oneColumn oneColumn
          (CanonicalContexts.Step.selector parameters profile)).rows
        assignment := by
    apply receiptRows parameters profile recipes defaultAdmissible
      assignment evidence
    apply bodyReceiptMember parameters profile recipes defaultAdmissible
    rw [CanonicalStepPlan.bodyReceipts]
    exact
      List.mem_cons_of_mem _
        (List.mem_cons_of_mem _ List.mem_cons_self)
    simp [recursiveNifsOwner]
  have falseRows :
      Satisfies
        (CanonicalBranchPlan.falseActivationReceipt
          SourceOwners.stepBranchPath oneColumn oneColumn
          (CanonicalContexts.Step.selector parameters profile)).rows
        assignment := by
    apply receiptRows parameters profile recipes defaultAdmissible
      assignment evidence
    apply bodyReceiptMember parameters profile recipes defaultAdmissible
    rw [CanonicalStepPlan.bodyReceipts]
    exact
      List.mem_cons_of_mem _
        (List.mem_cons_of_mem _
          (List.mem_cons_of_mem _ List.mem_cons_self))
    simp [recursiveNifsOwner]
  have activationRows : Satisfies activation.rows assignment := by
    rw [← CanonicalBranchPlan.activation_rows_conserved]
    exact
      (satisfies_append_iff _ _ assignment).2
        ⟨trueRows, falseRows⟩
  have parentActive : assignment activation.active = 1 := by
    simpa [activation] using evidence.constantOne
  constructor
  · exact selectorDecoded
  · by_cases iterationZero : input.iteration = 0
    · have selectedTrue :
          boolCodec.decode [assignment activation.selector] = some true := by
        simpa [activation, iterationZero] using selectorDecoded
      have selected :=
        activation.selected_true_sound assignment evidence.constantOne
          selectedTrue activationRows
      constructor
      · simpa [activation, iterationZero] using selected.1.trans parentActive
      · simpa [activation, iterationZero] using selected.2
    · have selectedFalse :
          boolCodec.decode [assignment activation.selector] = some false := by
        simpa [activation, iterationZero] using selectorDecoded
      have selected :=
        activation.selected_false_sound assignment evidence.constantOne
          selectedFalse activationRows
      constructor
      · simpa [activation, iterationZero] using selected.1
      · simpa [activation, iterationZero] using selected.2.trans parentActive

/-- Compatibility projection for the legacy all-R1CS physical encoding. -/
theorem branchControls
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (physical :
      (encoding parameters profile recipes defaultAdmissible
        ).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input)) :
    boolCodec.decode
        [assignment (CanonicalContexts.Step.selector parameters profile)] =
        some (decide (input.iteration = 0)) ∧
      assignment
          (activationColumn SourceOwners.stepBranchPath true) =
        (if input.iteration = 0 then 1 else 0) ∧
      assignment
          (activationColumn SourceOwners.stepBranchPath false) =
        (if input.iteration = 0 then 0 else 1) := by
  exact
    branchControlsFromEvidence
      parameters profile recipes defaultAdmissible laws assignment input
      (evidenceOfPhysical parameters profile recipes defaultAdmissible
        assignment physical)
      inputDecoded

private theorem baseInitial
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input))
    (iterationZero : input.iteration = 0) :
    input.z0 = input.zi ∧
      (CanonicalContexts.Step.baseRunning parameters).toColumnBundle.Decodes
        (profile.family parameters) (.data .running) assignment
        (defaultRunning parameters) := by
  have common :=
    commonDecodedFromEvidence parameters profile recipes defaultAdmissible laws
      assignment input evidence inputDecoded
  have controls :=
    branchControlsFromEvidence
      parameters profile recipes defaultAdmissible laws
      assignment input evidence inputDecoded
  have activeTrue :
      assignment
          (activationColumn SourceOwners.stepBranchPath true) = 1 := by
    simpa [iterationZero] using controls.2.1
  let equalityPlan :=
    CanonicalStepPlan.baseEqualityPlan parameters profile recipes
  have equalityRows : Satisfies equalityPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes defaultAdmissible
      assignment evidence equalityPlan (by
        dsimp [equalityPlan]
        rw [CanonicalStepPlan.bodyReceipts]
        exact
          List.mem_cons_of_mem _
            (List.mem_cons_of_mem _
              (List.mem_cons_of_mem _
                (List.mem_cons_of_mem _ List.mem_cons_self)))) (by decide)
  have afterEquality :=
    decodedOfExactExecution equalityPlan laws assignment
      evidence.constantOne activeTrue
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonValues
        parameters input)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseEqualityValues
        parameters input)
      common equalityRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.baseStateEqualCall_exec
        parameters input)
  have afterEquality' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.afterBaseEquality parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseEqualityValues
          parameters input) := by
    simpa [equalityPlan, CanonicalStepPlan.baseEqualityPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterBaseEquality] using afterEquality
  let assertionPlan :=
    CanonicalStepPlan.baseAssertionPlan parameters profile
  have assertionRows : Satisfies assertionPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes defaultAdmissible
      assignment evidence assertionPlan (by
        dsimp [assertionPlan]
        rw [CanonicalStepPlan.bodyReceipts]
        exact
          List.mem_cons_of_mem _
            (List.mem_cons_of_mem _
              (List.mem_cons_of_mem _
                (List.mem_cons_of_mem _
                  (List.mem_cons_of_mem _ List.mem_cons_self))))) (by decide)
  rcases assertionPlan.activeSound laws assignment evidence.constantOne activeTrue
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseEqualityValues
        parameters input)
      afterEquality' assertionRows with
    ⟨result, asserted, _⟩
  have accepted := asserted.1
  change stateEqual parameters input.z0 input.zi = true at accepted
  have initialState : input.z0 = input.zi := by
    simpa [stateEqual] using accepted
  let literalPlan :=
    CanonicalStepPlan.baseLiteralPlan
      parameters profile defaultAdmissible
  have literalRows : Satisfies literalPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes defaultAdmissible
      assignment evidence literalPlan (by
        dsimp [literalPlan]
        rw [CanonicalStepPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self) (by decide)
  have afterLiteral :=
    decodedOfExactExecution literalPlan laws assignment
      evidence.constantOne activeTrue
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseEqualityValues
        parameters input)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseLiteralValues
        parameters input)
      afterEquality' literalRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.baseDefaultCall_exec
        parameters input)
  have afterLiteral' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.afterBaseLiteral parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseLiteralValues
          parameters input) := by
    simpa [literalPlan, CanonicalStepPlan.baseLiteralPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterBaseLiteral] using afterLiteral
  have runningDecoded :=
    SchemaBundles.get_decodes
      (profile.family parameters) assignment
      (.here (Ports.committedRunning parameters))
      (CanonicalContexts.Step.afterBaseLiteral parameters).toSchemaBundles
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterBaseLiteralValues
        parameters input)
      afterLiteral'
  constructor
  · exact initialState
  · simpa [CanonicalContexts.Step.baseRunning,
      CanonicalContexts.Step.afterBaseLiteral,
      Columns.toSchemaBundles_get] using runningDecoded

private theorem recursiveAccepted
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input))
    (iterationNonzero : ¬input.iteration = 0) :
    ∃ folded : parameters.Running,
      parameters.machine.freshPublic input.fresh =
          parameters.machine.encodeInstance
            (parameters.machine.hash
              (Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.priorHashPreimage
                parameters.setup input)) ∧
        parameters.setup.nifs.verify
            (parameters.setup.verifierKeys Vocabulary.Step.selected)
            (input.running Vocabulary.Step.selected)
            input.fresh input.nifsProof =
          some folded ∧
        (CanonicalContexts.Step.recursiveRunning
            parameters).toColumnBundle.Decodes
          (profile.family parameters) (.data .running) assignment folded := by
  have common :=
    commonDecodedFromEvidence parameters profile recipes defaultAdmissible laws
      assignment input evidence inputDecoded
  have controls :=
    branchControlsFromEvidence
      parameters profile recipes defaultAdmissible laws
      assignment input evidence inputDecoded
  have activeFalse :
      assignment
          (activationColumn SourceOwners.stepBranchPath false) = 1 := by
    simpa [iterationNonzero] using controls.2.2
  let hashPlan :=
    CanonicalStepPlan.recursiveHashPlan parameters profile recipes
  have hashRows : Satisfies hashPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes defaultAdmissible
      assignment evidence hashPlan (by
        dsimp [hashPlan]
        rw [CanonicalStepPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self) (by decide)
  have afterHash :=
    decodedOfExactExecution hashPlan laws assignment
      evidence.constantOne activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonValues
        parameters input)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashValues
        parameters input)
      common hashRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.hashPriorCall_exec
        parameters input)
  have afterHash' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.afterHash parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashValues
          parameters input) := by
    simpa [hashPlan, CanonicalStepPlan.recursiveHashPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterHash] using afterHash
  let freshPlan :=
    CanonicalStepPlan.recursiveFreshPublicPlan parameters profile recipes
  have freshRows : Satisfies freshPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes defaultAdmissible
      assignment evidence freshPlan (by
        dsimp [freshPlan]
        rw [CanonicalStepPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self) (by decide)
  have afterFresh :=
    decodedOfExactExecution freshPlan laws assignment
      evidence.constantOne activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashValues
        parameters input)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterFreshPublicValues
        parameters input)
      afterHash' freshRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.freshPublicCall_exec
        parameters input)
  have afterFresh' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.afterFreshPublic parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterFreshPublicValues
          parameters input) := by
    simpa [freshPlan, CanonicalStepPlan.recursiveFreshPublicPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterFreshPublic] using afterFresh
  let encodePlan :=
    CanonicalStepPlan.recursiveEncodePlan parameters profile recipes
  have encodeRows : Satisfies encodePlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes defaultAdmissible
      assignment evidence encodePlan (by
        dsimp [encodePlan]
        rw [CanonicalStepPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self) (by decide)
  have afterEncode :=
    decodedOfExactExecution encodePlan laws assignment
      evidence.constantOne activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterFreshPublicValues
        parameters input)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodeValues
        parameters input)
      afterFresh' encodeRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.encodeInstanceCall_exec
        parameters input)
  have afterEncode' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.afterEncode parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodeValues
          parameters input) := by
    simpa [encodePlan, CanonicalStepPlan.recursiveEncodePlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterEncode] using afterEncode
  let equalityPlan :=
    CanonicalStepPlan.recursiveEncodedEqualityPlan
      parameters profile recipes
  have equalityRows : Satisfies equalityPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes defaultAdmissible
      assignment evidence equalityPlan (by
        dsimp [equalityPlan]
        rw [CanonicalStepPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self) (by decide)
  have afterEquality :=
    decodedOfExactExecution equalityPlan laws assignment
      evidence.constantOne activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodeValues
        parameters input)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodedEqualityValues
        parameters input)
      afterEncode' equalityRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.encodedEqualCall_exec
        parameters input)
  have afterEquality' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.afterEncodedEquality parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodedEqualityValues
          parameters input) := by
    simpa [equalityPlan,
      CanonicalStepPlan.recursiveEncodedEqualityPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterEncodedEquality] using afterEquality
  let assertionPlan :=
    CanonicalStepPlan.recursiveAssertionPlan parameters profile
  have assertionRows : Satisfies assertionPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes defaultAdmissible
      assignment evidence assertionPlan (by
        dsimp [assertionPlan]
        rw [CanonicalStepPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self) (by decide)
  rcases assertionPlan.activeSound laws assignment evidence.constantOne activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodedEqualityValues
        parameters input)
      afterEquality' assertionRows with
    ⟨assertionResult, assertionSemantic, _⟩
  have priorLinkTrue :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.priorLinkAccepted
          parameters input =
        true := by
    have accepted := assertionSemantic.1
    change
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.priorLinkAccepted
          parameters input =
        true at accepted
    exact accepted
  have priorPublic :=
    (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.priorLinkAccepted_eq_true_iff
      parameters input).mp priorLinkTrue
  let invokePlan :=
    CanonicalStepPlan.recursiveNifsInvokePlan
      parameters profile recipes
  let nifsInputs :=
    (Refs.cons
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.running
        parameters)
      (Refs.cons
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.fresh
          parameters)
        (Refs.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.nifsProof
            parameters)
          Refs.nil))).get
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodedEqualityValues
        parameters input)
  have operandsDecoded :
      invokePlan.frame.operands.Decodes
        (profile.family parameters) assignment nifsInputs := by
    rw [CallFrame.operands, invokePlan.contextExact]
    exact
      RefBundles.fromSchema_decodes
        (profile.family parameters) assignment
        (Refs.cons
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.running
            parameters)
          (Refs.cons
            (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.fresh
              parameters)
            (Refs.cons
              (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.nifsProof
                parameters)
              Refs.nil)))
        (CanonicalContexts.Step.afterEncodedEquality
          parameters).toSchemaBundles
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodedEqualityValues
          parameters input)
        afterEquality'
  have frameOne : assignment invokePlan.frame.one = 1 := by
    rw [invokePlan.oneExact]
    exact evidence.constantOne
  have frameActive : assignment invokePlan.frame.active = 1 := by
    rw [invokePlan.activeExact]
    exact activeFalse
  rcases evidence.recursiveNifs nifsInputs frameOne frameActive operandsDecoded with
    ⟨nifsOutputs, nifsEvaluated, nifsOutputsDecoded⟩
  have nifsEvaluated' :
      callEval parameters Call.nifsVerify
          ((Refs.cons
            (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.running
              parameters)
            (Refs.cons
              (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.fresh
                parameters)
              (Refs.cons
                (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.nifsProof
                  parameters)
                Refs.nil))).get
            (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterEncodedEqualityValues
              parameters input)) =
        some nifsOutputs := by
    simpa [nifsInputs, signature] using nifsEvaluated
  rw [Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.nifsCallEval
    parameters input] at nifsEvaluated'
  cases verifierResult :
      parameters.setup.nifs.verify
        (parameters.setup.verifierKeys Vocabulary.Step.selected)
        (input.running Vocabulary.Step.selected)
        input.fresh input.nifsProof with
  | none =>
      rw [verifierResult] at nifsEvaluated'
      contradiction
  | some folded =>
      rw [verifierResult] at nifsEvaluated'
      have outputsExact :
          nifsOutputs = .cons folded .nil :=
        Option.some.inj nifsEvaluated'.symm
      subst nifsOutputs
      have canonicalOutputsDecoded :
          (instructionColumns SourceOwners.stepRecursiveNifsPath
              [(Ports.committedRunning parameters)]).toSchemaBundles.Decodes
            (profile.family parameters) assignment (.cons folded .nil) := by
        have decoded := nifsOutputsDecoded
        change
          invokePlan.frame.outputs.Decodes
            (profile.family parameters) assignment (.cons folded .nil) at decoded
        rw [invokePlan.outputsExact] at decoded
        exact decoded
      have runningDecoded :
          (CanonicalContexts.Step.recursiveRunning
              parameters).toColumnBundle.Decodes
            (profile.family parameters) (.data .running)
            assignment folded := by
        simpa [CanonicalContexts.Step.recursiveRunning,
          Columns.toSchemaBundles_get] using canonicalOutputsDecoded.1
      refine ⟨folded, priorPublic, rfl, ?_⟩
      exact runningDecoded

private theorem joinRows
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (assignment : ColumnId -> Field)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment) :
    Satisfies
      (CanonicalBranchPlan.onePortJoinRecipe
        SourceOwners.stepBranchPath
        (CanonicalContexts.Step.selector parameters profile)
        (Ports.committedRunning parameters)
      (CanonicalContexts.Step.baseRunning parameters)
      (CanonicalContexts.Step.recursiveRunning parameters)).rows
      assignment := by
  have rows :
      Satisfies
        (CanonicalBranchPlan.onePortJoinReceipt
          SourceOwners.stepBranchPath
          (CanonicalContexts.Step.selector parameters profile)
          (Ports.committedRunning parameters)
          (CanonicalContexts.Step.baseRunning parameters)
          (CanonicalContexts.Step.recursiveRunning parameters)).rows
        assignment := by
    apply receiptRows parameters profile recipes defaultAdmissible
      assignment evidence
    apply bodyReceiptMember parameters profile recipes defaultAdmissible
    rw [CanonicalStepPlan.bodyReceipts]
    right
    right
    right
    right
    right
    right
    right
    right
    right
    right
    right
    right
    right
    exact List.mem_cons_self
    simp [recursiveNifsOwner]
  simpa [CanonicalBranchPlan.onePortJoinReceipt] using rows

private theorem baseJoinedDecoded
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input))
    (iterationZero : input.iteration = 0)
    (runningDecoded :
      (CanonicalContexts.Step.baseRunning parameters).toColumnBundle.Decodes
        (profile.family parameters) (.data .running) assignment
        (defaultRunning parameters)) :
    Columns.Decodes (profile.family parameters)
      (CanonicalContexts.Step.joined parameters) assignment
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedValues
        parameters (defaultRunning parameters)) := by
  have controls :=
    branchControlsFromEvidence
      parameters profile recipes defaultAdmissible laws
      assignment input evidence inputDecoded
  have selectorTrue :
      boolCodec.decode
          [assignment (CanonicalContexts.Step.selector parameters profile)] =
        some true := by
    simpa [iterationZero] using controls.1
  let mux :=
    CanonicalBranchPlan.onePortJoinRecipe
      SourceOwners.stepBranchPath
      (CanonicalContexts.Step.selector parameters profile)
      (Ports.committedRunning parameters)
      (CanonicalContexts.Step.baseRunning parameters)
      (CanonicalContexts.Step.recursiveRunning parameters)
  have selected :
      mux.joined.values assignment =
        mux.onTrue.values assignment :=
    mux.selected_true_sound assignment
      (by simpa [mux] using selectorTrue)
      (by
        simpa [mux] using
          joinRows parameters profile recipes defaultAdmissible
            assignment evidence)
  have joinedHead :
      mux.joined.Decodes (profile.family parameters) (.data .running)
        assignment (defaultRunning parameters) := by
    unfold ColumnBundle.Decodes at runningDecoded ⊢
    rw [selected]
    simpa [mux, CanonicalBranchPlan.onePortJoinRecipe] using runningDecoded
  exact ⟨
    (by
      simpa [mux, CanonicalBranchPlan.onePortJoinRecipe,
        CanonicalContexts.Step.joined] using joinedHead),
    trivial⟩

private theorem recursiveJoinedDecoded
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input))
    (iterationNonzero : ¬input.iteration = 0)
    (folded : parameters.Running)
    (runningDecoded :
      (CanonicalContexts.Step.recursiveRunning
          parameters).toColumnBundle.Decodes
        (profile.family parameters) (.data .running) assignment folded) :
    Columns.Decodes (profile.family parameters)
      (CanonicalContexts.Step.joined parameters) assignment
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedValues
        parameters folded) := by
  have controls :=
    branchControlsFromEvidence
      parameters profile recipes defaultAdmissible laws
      assignment input evidence inputDecoded
  have selectorFalse :
      boolCodec.decode
          [assignment (CanonicalContexts.Step.selector parameters profile)] =
        some false := by
    simpa [iterationNonzero] using controls.1
  let mux :=
    CanonicalBranchPlan.onePortJoinRecipe
      SourceOwners.stepBranchPath
      (CanonicalContexts.Step.selector parameters profile)
      (Ports.committedRunning parameters)
      (CanonicalContexts.Step.baseRunning parameters)
      (CanonicalContexts.Step.recursiveRunning parameters)
  have selected :
      mux.joined.values assignment =
        mux.onFalse.values assignment :=
    mux.selected_false_sound assignment
      (by simpa [mux] using selectorFalse)
      (by
        simpa [mux] using
          joinRows parameters profile recipes defaultAdmissible
            assignment evidence)
  have joinedHead :
      mux.joined.Decodes (profile.family parameters) (.data .running)
        assignment folded := by
    unfold ColumnBundle.Decodes at runningDecoded ⊢
    rw [selected]
    simpa [mux, CanonicalBranchPlan.onePortJoinRecipe] using runningDecoded
  exact ⟨
    (by
      simpa [mux, CanonicalBranchPlan.onePortJoinRecipe,
        CanonicalContexts.Step.joined] using joinedHead),
    trivial⟩

private theorem resultDecodedFromRunning
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (runningNext : parameters.Running)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input))
    (joinedDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.joined parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedValues
          parameters runningNext)) :
    Columns.Decodes (profile.family parameters)
      (CanonicalContexts.Step.result parameters) assignment
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.resultValuesFor
        parameters input runningNext) := by
  have common :=
    commonDecodedFromEvidence parameters profile recipes defaultAdmissible laws
      assignment input evidence inputDecoded
  have continuationInput :=
    Columns.append_decodes
      (profile.family parameters) assignment
      (CanonicalContexts.Step.joined parameters)
      (CanonicalContexts.Step.common parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedValues
        parameters runningNext)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.commonValues
        parameters input)
      joinedDecoded common
  let continuationPlan :=
    CanonicalStepPlan.continuationHashPlan parameters profile recipes
  have continuationRows :
      Satisfies continuationPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes defaultAdmissible
      assignment evidence continuationPlan (by
        dsimp [continuationPlan]
        rw [CanonicalStepPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        right
        right
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self) (by decide)
  have afterHashNext :=
    decodedOfExactExecution continuationPlan laws assignment
      evidence.constantOne evidence.constantOne
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.continuationInputValues
        parameters input runningNext)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashNextValues
        parameters input runningNext)
      continuationInput continuationRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.hashNextCall_exec
        parameters input runningNext)
  have afterHashNext' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.afterHashNext parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashNextValues
          parameters input runningNext) := by
    simpa [continuationPlan, CanonicalStepPlan.continuationHashPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Step.afterHashNext] using afterHashNext
  have exported :=
    Columns.export_decodes
      (profile.family parameters) assignment
      (CanonicalContexts.Step.resultExports parameters)
      (CanonicalContexts.Step.resultExportsCompatible parameters)
      (CanonicalContexts.Step.afterHashNext parameters)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.afterHashNextValues
        parameters input runningNext)
      afterHashNext'
  simpa [CanonicalContexts.Step.result,
    CanonicalContexts.Step.resultExports] using exported

/-- Any row evidence accepted by the common Step proof yields an accepted
canonical Step transition. This is the semantic boundary shared by the legacy
R1CS wrapper and the native CCS selector. -/
theorem soundFromEvidence
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input)) :
    ∃ output :
        Nightstream.HyperNova.Construction2.Paper.Output
          parameters.Digest parameters.State parameters.Running 1,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        parameters input output := by
  letI : DecidableEq parameters.State := parameters.stateDecidableEq
  letI : DecidableEq parameters.Encoded := parameters.encodedDecidableEq
  by_cases iterationZero : input.iteration = 0
  · have base :=
      baseInitial parameters profile recipes defaultAdmissible laws
        assignment input evidence inputDecoded iterationZero
    have initialState := base.1
    let output :=
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
        parameters.setup parameters.machine input
        (fun _ => defaultRunning parameters)
    refine ⟨output, ?_⟩
    apply
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_fixedOne
        parameters input output).2
    unfold
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneAccepts
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneEval
    simp only
      [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
       iterationZero, if_pos, initialState, output]
    rfl
  · rcases
      recursiveAccepted parameters profile recipes defaultAdmissible laws
        assignment input evidence inputDecoded iterationZero with
      ⟨folded, priorPublic, verifierResult, runningDecoded⟩
    let output :=
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
        parameters.setup parameters.machine input (fun _ => folded)
    refine ⟨output, ?_⟩
    apply
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_fixedOne
        parameters input output).2
    unfold
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneAccepts
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneEval
    simp only
      [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
       iterationZero, if_neg, priorPublic, if_pos]
    have verifierResultCanonical :
        parameters.setup.nifs.verify
            (parameters.setup.verifierKeys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            (input.running
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            input.fresh input.nifsProof =
          some folded := by
      simpa only
        [Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.step_selected_eq_canonical]
        using verifierResult
    rw [verifierResultCanonical]
    rfl

/-- Every satisfying selected legacy R1CS Step encoding, at an input that
decodes to typed protocol data, yields an accepted canonical Step transition.
The accepted output is derived from verifier semantics. -/
theorem physicalSound
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (physical :
      (encoding parameters profile recipes defaultAdmissible
        ).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input)) :
    ∃ output :
        Nightstream.HyperNova.Construction2.Paper.Output
          parameters.Digest parameters.State parameters.Running 1,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        parameters input output := by
  exact
    soundFromEvidence
      parameters profile recipes defaultAdmissible laws assignment input
      (evidenceOfPhysical parameters profile recipes defaultAdmissible
        assignment physical)
      inputDecoded

/-- Physical Step soundness with exact output-coordinate alignment.  The
accepted typed output is the value decoded by the selected encoding's final
export columns, so the existential result cannot float free of the physical
assignment. -/
theorem soundAlignedFromEvidence
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (evidence :
      Evidence parameters profile recipes defaultAdmissible assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input)) :
    ∃ output :
        Nightstream.HyperNova.Construction2.Paper.Output
          parameters.Digest parameters.State parameters.Running 1,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
          parameters input output ∧
        Columns.Decodes (profile.family parameters)
          (CanonicalContexts.Step.result parameters) assignment
          (stepResultValues parameters output) := by
  letI : DecidableEq parameters.State := parameters.stateDecidableEq
  letI : DecidableEq parameters.Encoded := parameters.encodedDecidableEq
  by_cases iterationZero : input.iteration = 0
  · have base :=
      baseInitial parameters profile recipes defaultAdmissible laws
        assignment input evidence inputDecoded iterationZero
    have joinedDecoded :=
      baseJoinedDecoded parameters profile recipes defaultAdmissible laws
        assignment input evidence inputDecoded iterationZero base.2
    have resultDecoded :=
      resultDecodedFromRunning
        parameters profile recipes defaultAdmissible laws assignment input
        (defaultRunning parameters) evidence inputDecoded joinedDecoded
    let output :=
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
        parameters.setup parameters.machine input
        (fun _ => defaultRunning parameters)
    refine ⟨output, ?_, ?_⟩
    · apply
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_fixedOne
          parameters input output).2
      unfold
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneAccepts
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneEval
      simp only
        [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
         iterationZero, if_pos, base.1, output]
      rfl
    · simpa [output,
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.resultValuesFor]
        using resultDecoded
  · rcases
      recursiveAccepted parameters profile recipes defaultAdmissible laws
        assignment input evidence inputDecoded iterationZero with
      ⟨folded, priorPublic, verifierResult, runningDecoded⟩
    have joinedDecoded :=
      recursiveJoinedDecoded
        parameters profile recipes defaultAdmissible laws assignment input
        evidence inputDecoded iterationZero folded runningDecoded
    have resultDecoded :=
      resultDecodedFromRunning
        parameters profile recipes defaultAdmissible laws assignment input
        folded evidence inputDecoded joinedDecoded
    let output :=
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
        parameters.setup parameters.machine input (fun _ => folded)
    refine ⟨output, ?_, ?_⟩
    · apply
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_fixedOne
          parameters input output).2
      unfold
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneAccepts
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneEval
      simp only
        [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
         iterationZero, if_neg, priorPublic, if_pos]
      have verifierResultCanonical :
          parameters.setup.nifs.verify
              (parameters.setup.verifierKeys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              (input.running
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              input.fresh input.nifsProof =
            some folded := by
        simpa only
          [Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.step_selected_eq_canonical]
          using verifierResult
      rw [verifierResultCanonical]
      rfl
    · simpa [output,
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.resultValuesFor]
        using resultDecoded

/-- Compatibility projection for the legacy all-R1CS physical encoding, with
exact output-coordinate alignment. -/
theorem physicalSoundAligned
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      (profile.family parameters).codecFor (.data .running) |>.Admissible
        (defaultRunning parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        parameters.State parameters.Witness parameters.Running
        parameters.Fresh parameters.NifsProof)
    (physical :
      (encoding parameters profile recipes defaultAdmissible
        ).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Step.input parameters) assignment
        (stepInputValues parameters input)) :
    ∃ output :
        Nightstream.HyperNova.Construction2.Paper.Output
          parameters.Digest parameters.State parameters.Running 1,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
          parameters input output ∧
        Columns.Decodes (profile.family parameters)
          (CanonicalContexts.Step.result parameters) assignment
          (stepResultValues parameters output) := by
  exact
    soundAlignedFromEvidence
      parameters profile recipes defaultAdmissible laws assignment input
      (evidenceOfPhysical parameters profile recipes defaultAdmissible
        assignment physical)
      inputDecoded

end CanonicalStepSoundness

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
