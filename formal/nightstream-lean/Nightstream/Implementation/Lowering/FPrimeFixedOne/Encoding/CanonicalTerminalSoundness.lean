import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalProgram
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PrimitiveRefinement
import Nightstream.Implementation.Lowering.Goldilocks.ReceiptSatisfaction

/-!
Contract: artifact-independent row soundness for the selected canonical
fixed-one Terminal encoding.

Owns:
- reconstruction of the verifier-derived terminal branch selector;
- semantic discharge of the active base or recursive terminal obligations;
- implication from physical row satisfaction to the canonical typed terminal
  checker.

Does not own: a production codec/recipe instance, Rust source semantics,
numeric R1CS matrices, generated-row equality, or honest assignment
construction.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalTerminalSoundness

def encoding
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    Goldilocks.Encoding
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
        parameters) :=
  (CanonicalTerminalPlan.physical parameters profile recipes).toEncoding

private theorem receiptRows
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (assignment : ColumnId -> Field)
    (physical :
      (encoding parameters profile recipes).PhysicalSatisfies assignment)
    (receipt : InstructionReceipt)
    (member :
      receipt ∈ CanonicalTerminalPlan.receipts parameters profile recipes) :
    Satisfies receipt.rows assignment := by
  apply
    (encoding parameters profile recipes).receiptSatisfies
      assignment physical receipt
  simpa [encoding, CanonicalTerminalPlan.physical] using member

private theorem bodyReceiptMember
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (receipt : InstructionReceipt)
    (member :
      receipt ∈
        CanonicalTerminalPlan.bodyReceipts parameters profile recipes) :
    receipt ∈ CanonicalTerminalPlan.receipts parameters profile recipes := by
  rw [CanonicalTerminalPlan.receipts]
  apply List.mem_cons_of_mem
  exact List.mem_append_right _ member

private theorem bodyPlanRows
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (assignment : ColumnId -> Field)
    (physical :
      (encoding parameters profile recipes).PhysicalSatisfies assignment)
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
        CanonicalTerminalPlan.bodyReceipts parameters profile recipes) :
    Satisfies plan.receipt.rows assignment := by
  apply receiptRows parameters profile recipes assignment physical
  exact bodyReceiptMember parameters profile recipes plan.receipt member

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

theorem decodedBitReference
    (parameters : Parameters)
    (profile : Profile parameters)
    (assignment : ColumnId -> Field)
    {schema : Schema (typeSystem parameters)}
    (columns : Columns schema)
    (values : Schema.Values (typeSystem parameters) schema)
    (condition : Ref (typeSystem parameters) schema .bit)
    (widths : SchemaWidthAgrees (profile.family parameters) schema)
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

/-- The unconditional selector call decodes the exact typed branch context. -/
theorem branchDecoded
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (statement :
      Nightstream.HyperNova.Construction2.Paper.TerminalStatement
        parameters.State)
    (proof :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
        parameters)
    (physical :
      (encoding parameters profile recipes).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.input parameters) assignment
        (terminalInputValues parameters statement proof)) :
    Columns.Decodes (profile.family parameters)
      (CanonicalContexts.Terminal.branchInput parameters) assignment
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchValues
        parameters (decide (statement.iteration = 0)) statement proof) := by
  let selectorPlan :=
    CanonicalTerminalPlan.selectorPlan parameters profile recipes
  have selectorRows : Satisfies selectorPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      selectorPlan (by
        dsimp [selectorPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
        exact List.mem_cons_self)
  have decoded :=
    decodedOfExactExecution selectorPlan laws assignment
      physical.1 physical.1
      (terminalInputValues parameters statement proof)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchValues
        parameters (decide (statement.iteration = 0)) statement proof)
      inputDecoded selectorRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.iterationZeroCall_exec
        parameters statement proof)
  simpa [selectorPlan, CanonicalTerminalPlan.selectorPlan,
    PrimitivePlan.resultColumns,
    CanonicalContexts.Terminal.branchInput] using decoded

/-- Terminal branch activations are derived from the internally computed
iteration-zero bit. -/
theorem branchControls
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (statement :
      Nightstream.HyperNova.Construction2.Paper.TerminalStatement
        parameters.State)
    (proof :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
        parameters)
    (physical :
      (encoding parameters profile recipes).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.input parameters) assignment
        (terminalInputValues parameters statement proof)) :
    boolCodec.decode
        [assignment
          (CanonicalContexts.Terminal.selector parameters profile)] =
        some (decide (statement.iteration = 0)) ∧
      assignment
          (activationColumn SourceOwners.terminalBranchPath true) =
        (if statement.iteration = 0 then 1 else 0) ∧
      assignment
          (activationColumn SourceOwners.terminalBranchPath false) =
        (if statement.iteration = 0 then 0 else 1) := by
  have branch :=
    branchDecoded parameters profile recipes laws assignment
      statement proof physical inputDecoded
  have selectorDecoded :
      boolCodec.decode
          [assignment
            (CanonicalContexts.Terminal.selector parameters profile)] =
        some (decide (statement.iteration = 0)) := by
    simpa [CanonicalContexts.Terminal.selector,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchValues]
      using
        decodedBitReference parameters profile assignment
          (CanonicalContexts.Terminal.branchInput parameters)
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchValues
            parameters (decide (statement.iteration = 0)) statement proof)
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.iterationZero
            parameters)
          (CanonicalContexts.Terminal.branchInputWidths parameters profile)
          branch
  let activation :=
    CanonicalBranchPlan.activationRecipe
      SourceOwners.terminalBranchPath oneColumn oneColumn
      (CanonicalContexts.Terminal.selector parameters profile)
  have trueRows :
      Satisfies
        (CanonicalBranchPlan.trueActivationReceipt
          SourceOwners.terminalBranchPath oneColumn oneColumn
          (CanonicalContexts.Terminal.selector parameters profile)).rows
        assignment := by
    apply receiptRows parameters profile recipes assignment physical
    apply bodyReceiptMember parameters profile recipes
    rw [CanonicalTerminalPlan.bodyReceipts]
    exact List.mem_cons_of_mem _ List.mem_cons_self
  have falseRows :
      Satisfies
        (CanonicalBranchPlan.falseActivationReceipt
          SourceOwners.terminalBranchPath oneColumn oneColumn
          (CanonicalContexts.Terminal.selector parameters profile)).rows
        assignment := by
    apply receiptRows parameters profile recipes assignment physical
    apply bodyReceiptMember parameters profile recipes
    rw [CanonicalTerminalPlan.bodyReceipts]
    exact
      List.mem_cons_of_mem _
        (List.mem_cons_of_mem _ List.mem_cons_self)
  have activationRows : Satisfies activation.rows assignment := by
    rw [← CanonicalBranchPlan.activation_rows_conserved]
    exact
      (satisfies_append_iff _ _ assignment).2
        ⟨trueRows, falseRows⟩
  have parentActive : assignment activation.active = 1 := by
    simpa [activation] using physical.1
  constructor
  · exact selectorDecoded
  · by_cases iterationZero : statement.iteration = 0
    · have selectedTrue :
          boolCodec.decode [assignment activation.selector] = some true := by
        simpa [activation, iterationZero] using selectorDecoded
      have selected :=
        activation.selected_true_sound assignment physical.1
          selectedTrue activationRows
      constructor
      · simpa [activation, iterationZero] using selected.1.trans parentActive
      · simpa [activation, iterationZero] using selected.2
    · have selectedFalse :
          boolCodec.decode [assignment activation.selector] = some false := by
        simpa [activation, iterationZero] using selectorDecoded
      have selected :=
        activation.selected_false_sound assignment physical.1
          selectedFalse activationRows
      constructor
      · simpa [activation, iterationZero] using selected.1
      · simpa [activation, iterationZero] using selected.2.trans parentActive

private theorem baseEndpoint
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (statement :
      Nightstream.HyperNova.Construction2.Paper.TerminalStatement
        parameters.State)
    (proof :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
        parameters)
    (physical :
      (encoding parameters profile recipes).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.input parameters) assignment
        (terminalInputValues parameters statement proof))
    (iterationZero : statement.iteration = 0) :
    statement.zi = statement.z0 := by
  have branch :=
    branchDecoded parameters profile recipes laws assignment
      statement proof physical inputDecoded
  have controls :=
    branchControls parameters profile recipes laws assignment
      statement proof physical inputDecoded
  have activeTrue :
      assignment
          (activationColumn SourceOwners.terminalBranchPath true) = 1 := by
    simpa [iterationZero] using controls.2.1
  let equalityPlan :=
    CanonicalTerminalPlan.baseEqualityPlan parameters profile recipes
  have equalityRows : Satisfies equalityPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      equalityPlan (by
        dsimp [equalityPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
        right
        right
        right
        exact List.mem_cons_self)
  have afterEquality :=
    decodedOfExactExecution equalityPlan laws assignment
      physical.1 activeTrue
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchValues
        parameters (decide (statement.iteration = 0)) statement proof)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterBaseEqualityValues
        parameters (decide (statement.iteration = 0)) statement proof)
      branch equalityRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.baseStateEqualCall_exec
        parameters (decide (statement.iteration = 0)) statement proof)
  have afterEquality' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.afterBaseEquality parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterBaseEqualityValues
          parameters (decide (statement.iteration = 0)) statement proof) := by
    simpa [equalityPlan, CanonicalTerminalPlan.baseEqualityPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterBaseEquality] using afterEquality
  let assertionPlan :=
    CanonicalTerminalPlan.baseAssertionPlan parameters profile
  have assertionRows : Satisfies assertionPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      assertionPlan (by
        dsimp [assertionPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
        right
        right
        right
        right
        exact List.mem_cons_self)
  rcases assertionPlan.activeSound laws assignment physical.1 activeTrue
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterBaseEqualityValues
        parameters (decide (statement.iteration = 0)) statement proof)
      afterEquality' assertionRows with
    ⟨result, asserted, _⟩
  have accepted := asserted.1
  change stateEqual parameters statement.zi statement.z0 = true at accepted
  simpa [stateEqual] using accepted

private theorem recursiveConditions
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (statement :
      Nightstream.HyperNova.Construction2.Paper.TerminalStatement
        parameters.State)
    (proof :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
        parameters)
    (physical :
      (encoding parameters profile recipes).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.input parameters) assignment
        (terminalInputValues parameters statement proof))
    (iterationNonzero : ¬statement.iteration = 0) :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorLinkAccepted
          parameters statement proof =
        true ∧
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.runningAcceptedValue
          parameters proof =
        true ∧
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.freshAcceptedValue
          parameters proof =
        true := by
  have branch :=
    branchDecoded parameters profile recipes laws assignment
      statement proof physical inputDecoded
  have controls :=
    branchControls parameters profile recipes laws assignment
      statement proof physical inputDecoded
  have activeFalse :
      assignment
          (activationColumn SourceOwners.terminalBranchPath false) = 1 := by
    simpa [iterationNonzero] using controls.2.2
  let hashPlan :=
    CanonicalTerminalPlan.recursiveHashPlan parameters profile recipes
  have hashRows : Satisfies hashPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      hashPlan (by
        dsimp [hashPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
        right
        right
        right
        right
        right
        exact List.mem_cons_self)
  have afterHash :=
    decodedOfExactExecution hashPlan laws assignment
      physical.1 activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchValues
        parameters (decide (statement.iteration = 0)) statement proof)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterHashValues
        parameters (decide (statement.iteration = 0)) statement proof)
      branch hashRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.hashPriorCall_exec
        parameters (decide (statement.iteration = 0)) statement proof)
  have afterHash' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.afterHash parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterHashValues
          parameters (decide (statement.iteration = 0)) statement proof) := by
    simpa [hashPlan, CanonicalTerminalPlan.recursiveHashPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterHash] using afterHash
  let freshPublicPlan :=
    CanonicalTerminalPlan.recursiveFreshPublicPlan
      parameters profile recipes
  have freshPublicRows :
      Satisfies freshPublicPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      freshPublicPlan (by
        dsimp [freshPublicPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self)
  have afterFreshPublic :=
    decodedOfExactExecution freshPublicPlan laws assignment
      physical.1 activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterHashValues
        parameters (decide (statement.iteration = 0)) statement proof)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshPublicValues
        parameters (decide (statement.iteration = 0)) statement proof)
      afterHash' freshPublicRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.freshPublicCall_exec
        parameters (decide (statement.iteration = 0)) statement proof)
  have afterFreshPublic' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.afterFreshPublic parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshPublicValues
          parameters (decide (statement.iteration = 0)) statement proof) := by
    simpa [freshPublicPlan,
      CanonicalTerminalPlan.recursiveFreshPublicPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterFreshPublic] using afterFreshPublic
  let encodePlan :=
    CanonicalTerminalPlan.recursiveEncodePlan parameters profile recipes
  have encodeRows : Satisfies encodePlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      encodePlan (by
        dsimp [encodePlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self)
  have afterEncode :=
    decodedOfExactExecution encodePlan laws assignment
      physical.1 activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshPublicValues
        parameters (decide (statement.iteration = 0)) statement proof)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodeValues
        parameters (decide (statement.iteration = 0)) statement proof)
      afterFreshPublic' encodeRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.encodeInstanceCall_exec
        parameters (decide (statement.iteration = 0)) statement proof)
  have afterEncode' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.afterEncode parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodeValues
          parameters (decide (statement.iteration = 0)) statement proof) := by
    simpa [encodePlan, CanonicalTerminalPlan.recursiveEncodePlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterEncode] using afterEncode
  let equalityPlan :=
    CanonicalTerminalPlan.recursiveEncodedEqualityPlan
      parameters profile recipes
  have equalityRows : Satisfies equalityPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      equalityPlan (by
        dsimp [equalityPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self)
  have afterEquality :=
    decodedOfExactExecution equalityPlan laws assignment
      physical.1 activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodeValues
        parameters (decide (statement.iteration = 0)) statement proof)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodedEqualityValues
        parameters (decide (statement.iteration = 0)) statement proof)
      afterEncode' equalityRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.encodedEqualCall_exec
        parameters (decide (statement.iteration = 0)) statement proof)
  have afterEquality' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.afterEncodedEquality parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodedEqualityValues
          parameters (decide (statement.iteration = 0)) statement proof) := by
    simpa [equalityPlan,
      CanonicalTerminalPlan.recursiveEncodedEqualityPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterEncodedEquality] using afterEquality
  let priorAssertionPlan :=
    CanonicalTerminalPlan.recursivePriorAssertionPlan parameters profile
  have priorAssertionRows :
      Satisfies priorAssertionPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      priorAssertionPlan (by
        dsimp [priorAssertionPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
        right
        right
        right
        right
        right
        right
        right
        right
        right
        exact List.mem_cons_self)
  rcases priorAssertionPlan.activeSound laws assignment physical.1 activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodedEqualityValues
        parameters (decide (statement.iteration = 0)) statement proof)
      afterEquality' priorAssertionRows with
    ⟨priorResult, priorSemantic, _⟩
  have priorAccepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorLinkAccepted
          parameters statement proof =
        true := by
    have accepted := priorSemantic.1
    change
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.priorLinkAccepted
          parameters statement proof =
        true at accepted
    exact accepted
  let runningPlan :=
    CanonicalTerminalPlan.recursiveRunningCheckPlan
      parameters profile recipes
  have runningRows : Satisfies runningPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      runningPlan (by
        dsimp [runningPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
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
        exact List.mem_cons_self)
  have afterRunning :=
    decodedOfExactExecution runningPlan laws assignment
      physical.1 activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterEncodedEqualityValues
        parameters (decide (statement.iteration = 0)) statement proof)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterRunningCheckValues
        parameters (decide (statement.iteration = 0)) statement proof)
      afterEquality' runningRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.runningCheckCall_exec
        parameters (decide (statement.iteration = 0)) statement proof)
  have afterRunning' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.afterRunningCheck parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterRunningCheckValues
          parameters (decide (statement.iteration = 0)) statement proof) := by
    simpa [runningPlan,
      CanonicalTerminalPlan.recursiveRunningCheckPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterRunningCheck] using afterRunning
  let runningAssertionPlan :=
    CanonicalTerminalPlan.recursiveRunningAssertionPlan parameters profile
  have runningAssertionRows :
      Satisfies runningAssertionPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      runningAssertionPlan (by
        dsimp [runningAssertionPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
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
        exact List.mem_cons_self)
  rcases runningAssertionPlan.activeSound laws assignment
      physical.1 activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterRunningCheckValues
        parameters (decide (statement.iteration = 0)) statement proof)
      afterRunning' runningAssertionRows with
    ⟨runningResult, runningSemantic, _⟩
  have runningAccepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.runningAcceptedValue
          parameters proof =
        true := by
    have accepted := runningSemantic.1
    change
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.runningAcceptedValue
          parameters proof =
        true at accepted
    exact accepted
  let freshPlan :=
    CanonicalTerminalPlan.recursiveFreshCheckPlan parameters profile recipes
  have freshRows : Satisfies freshPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      freshPlan (by
        dsimp [freshPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
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
        exact List.mem_cons_self)
  have afterFresh :=
    decodedOfExactExecution freshPlan laws assignment
      physical.1 activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterRunningCheckValues
        parameters (decide (statement.iteration = 0)) statement proof)
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshCheckValues
        parameters (decide (statement.iteration = 0)) statement proof)
      afterRunning' freshRows
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.freshCheckCall_exec
        parameters (decide (statement.iteration = 0)) statement proof)
  have afterFresh' :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.afterFreshCheck parameters) assignment
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshCheckValues
          parameters (decide (statement.iteration = 0)) statement proof) := by
    simpa [freshPlan, CanonicalTerminalPlan.recursiveFreshCheckPlan,
      PrimitivePlan.resultColumns,
      CanonicalContexts.Terminal.afterFreshCheck] using afterFresh
  let freshAssertionPlan :=
    CanonicalTerminalPlan.recursiveFreshAssertionPlan parameters profile
  have freshAssertionRows :
      Satisfies freshAssertionPlan.receipt.rows assignment :=
    bodyPlanRows parameters profile recipes assignment physical
      freshAssertionPlan (by
        dsimp [freshAssertionPlan]
        rw [CanonicalTerminalPlan.bodyReceipts]
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
        exact List.mem_cons_self)
  rcases freshAssertionPlan.activeSound laws assignment
      physical.1 activeFalse
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.afterFreshCheckValues
        parameters (decide (statement.iteration = 0)) statement proof)
      afterFresh' freshAssertionRows with
    ⟨freshResult, freshSemantic, _⟩
  have freshAccepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.freshAcceptedValue
          parameters proof =
        true := by
    have accepted := freshSemantic.1
    change
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.freshAcceptedValue
          parameters proof =
        true at accepted
    exact accepted
  exact ⟨priorAccepted, runningAccepted, freshAccepted⟩

/-- Every satisfying selected physical Terminal encoding, at an input that
decodes to typed protocol data, is accepted by the canonical typed Terminal
checker. -/
theorem physicalSound
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (laws : FieldLaws)
    (assignment : ColumnId -> Field)
    (statement :
      Nightstream.HyperNova.Construction2.Paper.TerminalStatement
        parameters.State)
    (proof :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
        parameters)
    (physical :
      (encoding parameters profile recipes).PhysicalSatisfies assignment)
    (inputDecoded :
      Columns.Decodes (profile.family parameters)
        (CanonicalContexts.Terminal.input parameters) assignment
        (terminalInputValues parameters statement proof)) :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
      parameters statement proof := by
  by_cases iterationZero : statement.iteration = 0
  · have endpoint :=
      baseEndpoint parameters profile recipes laws assignment
        statement proof physical inputDecoded iterationZero
    unfold
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
    rw [←
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
        parameters).exec_eq_some_iff_holds
          (terminalInputValues parameters statement proof) .nil]
    rw [
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program_exec_eq_reference]
    simp [
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.referenceExec,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchReferenceExec,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.baseReferenceExec,
      stateEqual, iterationZero, endpoint]
    rfl
  · rcases
      recursiveConditions parameters profile recipes laws assignment
        statement proof physical inputDecoded iterationZero with
      ⟨priorAccepted, runningAccepted, freshAccepted⟩
    unfold
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
    rw [←
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
        parameters).exec_eq_some_iff_holds
          (terminalInputValues parameters statement proof) .nil]
    rw [
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program_exec_eq_reference]
    simp [
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.referenceExec,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.branchReferenceExec,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.recursiveReferenceExec,
      iterationZero, priorAccepted, runningAccepted, freshAccepted]
    rfl

end CanonicalTerminalSoundness

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
