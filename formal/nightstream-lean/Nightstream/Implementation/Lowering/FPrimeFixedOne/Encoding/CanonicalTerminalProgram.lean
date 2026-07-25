import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalTerminalPlan

/-!
Contract: conserved and source-aligned physical receipt program for the
fixed-one Terminal verifier.

Owns:
- premise-free scoping of the exact Terminal receipt order;
- exact source-owner order and receipt-local identity uniqueness;
- construction of the conserved and source-aligned receipt program.

Does not own: primitive recipes, normal-form minimality, row satisfaction,
Rust emission, generated artifacts, or Terminal protocol semantics.

Emits constraints: no new constraints; it certifies the receipts constructed
by `CanonicalTerminalPlan`.
-/

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary

namespace CanonicalTerminalPlan

/-- The exact Terminal body is well-scoped from any allocation prefix that
contains the verifier one column and the complete typed input context. -/
theorem bodyWellScoped
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (available : List ColumnId)
    (oneAvailable : oneColumn ∈ available)
    (inputCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.input parameters) available) :
    ReceiptsWellScoped available
      (bodyReceipts parameters profile recipes) := by
  let pSelector := selectorPlan.{0} parameters profile recipes
  let pBaseEquality := baseEqualityPlan.{0} parameters profile recipes
  let pBaseAssertion := baseAssertionPlan.{0} parameters profile
  let pRecursiveHash := recursiveHashPlan.{0} parameters profile recipes
  let pRecursiveFresh :=
    recursiveFreshPublicPlan.{0} parameters profile recipes
  let pRecursiveEncode :=
    recursiveEncodePlan.{0} parameters profile recipes
  let pRecursiveEquality :=
    recursiveEncodedEqualityPlan.{0} parameters profile recipes
  let pPriorAssertion :=
    recursivePriorAssertionPlan.{0} parameters profile
  let pRunningCheck :=
    recursiveRunningCheckPlan.{0} parameters profile recipes
  let pRunningAssertion :=
    recursiveRunningAssertionPlan.{0} parameters profile
  let pFreshCheck :=
    recursiveFreshCheckPlan.{0} parameters profile recipes
  let pFreshAssertion :=
    recursiveFreshAssertionPlan.{0} parameters profile
  let trueReceipt :=
    CanonicalBranchPlan.trueActivationReceipt
      SourceOwners.terminalBranchPath oneColumn oneColumn
      (CanonicalContexts.Terminal.selector parameters profile)
  let falseReceipt :=
    CanonicalBranchPlan.falseActivationReceipt
      SourceOwners.terminalBranchPath oneColumn oneColumn
      (CanonicalContexts.Terminal.selector parameters profile)
  let joinReceipt :=
    CanonicalBranchPlan.emptyJoinReceipt
      SourceOwners.terminalBranchPath
  let afterSelector := available ++ pSelector.receipt.columnIds
  let afterTrue := afterSelector ++ trueReceipt.columnIds
  let afterFalse := afterTrue ++ falseReceipt.columnIds
  let afterBaseEquality :=
    afterFalse ++ pBaseEquality.receipt.columnIds
  let afterBaseAssertion :=
    afterBaseEquality ++ pBaseAssertion.receipt.columnIds
  let afterRecursiveHash :=
    afterBaseAssertion ++ pRecursiveHash.receipt.columnIds
  let afterRecursiveFresh :=
    afterRecursiveHash ++ pRecursiveFresh.receipt.columnIds
  let afterRecursiveEncode :=
    afterRecursiveFresh ++ pRecursiveEncode.receipt.columnIds
  let afterRecursiveEquality :=
    afterRecursiveEncode ++ pRecursiveEquality.receipt.columnIds
  let afterPriorAssertion :=
    afterRecursiveEquality ++ pPriorAssertion.receipt.columnIds
  let afterRunningCheck :=
    afterPriorAssertion ++ pRunningCheck.receipt.columnIds
  let afterRunningAssertion :=
    afterRunningCheck ++ pRunningAssertion.receipt.columnIds
  let afterFreshCheck :=
    afterRunningAssertion ++ pFreshCheck.receipt.columnIds
  let afterFreshAssertion :=
    afterFreshCheck ++ pFreshAssertion.receipt.columnIds
  change ReceiptsWellScoped available
    [pSelector.receipt,
      trueReceipt,
      falseReceipt,
      pBaseEquality.receipt,
      pBaseAssertion.receipt,
      pRecursiveHash.receipt,
      pRecursiveFresh.receipt,
      pRecursiveEncode.receipt,
      pRecursiveEquality.receipt,
      pPriorAssertion.receipt,
      pRunningCheck.receipt,
      pRunningAssertion.receipt,
      pFreshCheck.receipt,
      pFreshAssertion.receipt,
      joinReceipt]

  have selectorScoped :
      pSelector.receipt.WellScopedAfter available :=
    PrimitivePlan.wellScopedAfter pSelector available
      (ReceiptScoping.Covers.primitiveInputs
        inputCovers oneAvailable oneAvailable)
  have oneAfterSelector : oneColumn ∈ afterSelector :=
    List.mem_append_left _ oneAvailable
  have branchCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.branchInput parameters)
        afterSelector := by
    simpa only [pSelector, selectorPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Terminal.branchInput] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pSelector available inputCovers
  have selectorAvailable :
      CanonicalContexts.Terminal.selector parameters profile ∈
        afterSelector := by
    apply branchCovers
    exact CanonicalPrimitivePlan.bitCoordinate_mem profile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.BranchRef.iterationZero
        parameters)
      (CanonicalContexts.Terminal.branchInput parameters)
      (CanonicalContexts.Terminal.branchInputWidths parameters profile)
  constructor
  · exact selectorScoped

  have trueScoped :
      trueReceipt.WellScopedAfter afterSelector := by
    apply CanonicalBranchPlan.trueActivationReceipt_wellScoped
    exact ⟨oneAfterSelector, oneAfterSelector, selectorAvailable⟩
  have oneAfterTrue : oneColumn ∈ afterTrue :=
    List.mem_append_left _ oneAfterSelector
  have selectorAfterTrue :
      CanonicalContexts.Terminal.selector parameters profile ∈
        afterTrue :=
    List.mem_append_left _ selectorAvailable
  have trueAvailable :
      activationColumn SourceOwners.terminalBranchPath true ∈
        afterTrue := by
    exact ReceiptScoping.trueActivationAvailableAfter
      SourceOwners.terminalBranchPath oneColumn oneColumn
      (CanonicalContexts.Terminal.selector parameters profile)
      afterSelector
  have branchAfterTrue :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.branchInput parameters)
        afterTrue :=
    branchCovers.weaken trueReceipt.columnIds
  constructor
  · exact trueScoped

  have falseScoped :
      falseReceipt.WellScopedAfter afterTrue := by
    apply CanonicalBranchPlan.falseActivationReceipt_wellScoped
    exact ⟨oneAfterTrue, oneAfterTrue, selectorAfterTrue⟩
  have oneAfterFalse : oneColumn ∈ afterFalse :=
    List.mem_append_left _ oneAfterTrue
  have trueAfterFalse :
      activationColumn SourceOwners.terminalBranchPath true ∈
        afterFalse :=
    List.mem_append_left _ trueAvailable
  have falseAvailable :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterFalse := by
    exact ReceiptScoping.falseActivationAvailableAfter
      SourceOwners.terminalBranchPath oneColumn oneColumn
      (CanonicalContexts.Terminal.selector parameters profile)
      afterTrue
  have branchAfterFalse :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.branchInput parameters)
        afterFalse :=
    branchAfterTrue.weaken falseReceipt.columnIds
  constructor
  · exact falseScoped

  have baseEqualityScoped :
      pBaseEquality.receipt.WellScopedAfter afterFalse :=
    PrimitivePlan.wellScopedAfter pBaseEquality afterFalse
      (ReceiptScoping.Covers.primitiveInputs
        branchAfterFalse oneAfterFalse trueAfterFalse)
  have oneAfterBaseEquality : oneColumn ∈ afterBaseEquality :=
    List.mem_append_left _ oneAfterFalse
  have trueAfterBaseEquality :
      activationColumn SourceOwners.terminalBranchPath true ∈
        afterBaseEquality :=
    List.mem_append_left _ trueAfterFalse
  have falseAfterBaseEquality :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterBaseEquality :=
    List.mem_append_left _ falseAvailable
  have baseEqualityCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.afterBaseEquality parameters)
        afterBaseEquality := by
    simpa only [pBaseEquality, baseEqualityPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Terminal.afterBaseEquality] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pBaseEquality afterFalse branchAfterFalse
  have branchAfterBaseEquality :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.branchInput parameters)
        afterBaseEquality :=
    branchAfterFalse.weaken pBaseEquality.receipt.columnIds
  constructor
  · exact baseEqualityScoped

  have baseAssertionScoped :
      pBaseAssertion.receipt.WellScopedAfter afterBaseEquality :=
    PrimitivePlan.wellScopedAfter pBaseAssertion afterBaseEquality
      (ReceiptScoping.Covers.primitiveInputs
        baseEqualityCovers oneAfterBaseEquality trueAfterBaseEquality)
  have oneAfterBaseAssertion : oneColumn ∈ afterBaseAssertion :=
    List.mem_append_left _ oneAfterBaseEquality
  have falseAfterBaseAssertion :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterBaseAssertion :=
    List.mem_append_left _ falseAfterBaseEquality
  have branchAfterBaseAssertion :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.branchInput parameters)
        afterBaseAssertion :=
    branchAfterBaseEquality.weaken pBaseAssertion.receipt.columnIds
  constructor
  · exact baseAssertionScoped

  have recursiveHashScoped :
      pRecursiveHash.receipt.WellScopedAfter afterBaseAssertion :=
    PrimitivePlan.wellScopedAfter pRecursiveHash afterBaseAssertion
      (ReceiptScoping.Covers.primitiveInputs
        branchAfterBaseAssertion oneAfterBaseAssertion
          falseAfterBaseAssertion)
  have oneAfterRecursiveHash : oneColumn ∈ afterRecursiveHash :=
    List.mem_append_left _ oneAfterBaseAssertion
  have falseAfterHash :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterRecursiveHash :=
    List.mem_append_left _ falseAfterBaseAssertion
  have hashCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.afterHash parameters)
        afterRecursiveHash := by
    simpa only [pRecursiveHash, recursiveHashPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Terminal.afterHash] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pRecursiveHash afterBaseAssertion branchAfterBaseAssertion
  constructor
  · exact recursiveHashScoped

  have recursiveFreshScoped :
      pRecursiveFresh.receipt.WellScopedAfter afterRecursiveHash :=
    PrimitivePlan.wellScopedAfter pRecursiveFresh afterRecursiveHash
      (ReceiptScoping.Covers.primitiveInputs
        hashCovers oneAfterRecursiveHash falseAfterHash)
  have oneAfterRecursiveFresh : oneColumn ∈ afterRecursiveFresh :=
    List.mem_append_left _ oneAfterRecursiveHash
  have falseAfterFresh :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterRecursiveFresh :=
    List.mem_append_left _ falseAfterHash
  have freshCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.afterFreshPublic parameters)
        afterRecursiveFresh := by
    simpa only [pRecursiveFresh, recursiveFreshPublicPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Terminal.afterFreshPublic] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pRecursiveFresh afterRecursiveHash hashCovers
  constructor
  · exact recursiveFreshScoped

  have recursiveEncodeScoped :
      pRecursiveEncode.receipt.WellScopedAfter afterRecursiveFresh :=
    PrimitivePlan.wellScopedAfter pRecursiveEncode afterRecursiveFresh
      (ReceiptScoping.Covers.primitiveInputs
        freshCovers oneAfterRecursiveFresh falseAfterFresh)
  have oneAfterRecursiveEncode : oneColumn ∈ afterRecursiveEncode :=
    List.mem_append_left _ oneAfterRecursiveFresh
  have falseAfterEncode :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterRecursiveEncode :=
    List.mem_append_left _ falseAfterFresh
  have encodeCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.afterEncode parameters)
        afterRecursiveEncode := by
    simpa only [pRecursiveEncode, recursiveEncodePlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Terminal.afterEncode] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pRecursiveEncode afterRecursiveFresh freshCovers
  constructor
  · exact recursiveEncodeScoped

  have recursiveEqualityScoped :
      pRecursiveEquality.receipt.WellScopedAfter
        afterRecursiveEncode :=
    PrimitivePlan.wellScopedAfter pRecursiveEquality afterRecursiveEncode
      (ReceiptScoping.Covers.primitiveInputs
        encodeCovers oneAfterRecursiveEncode falseAfterEncode)
  have oneAfterRecursiveEquality : oneColumn ∈ afterRecursiveEquality :=
    List.mem_append_left _ oneAfterRecursiveEncode
  have falseAfterEquality :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterRecursiveEquality :=
    List.mem_append_left _ falseAfterEncode
  have equalityCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.afterEncodedEquality parameters)
        afterRecursiveEquality := by
    simpa only [pRecursiveEquality, recursiveEncodedEqualityPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Terminal.afterEncodedEquality] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pRecursiveEquality afterRecursiveEncode encodeCovers
  constructor
  · exact recursiveEqualityScoped

  have priorAssertionScoped :
      pPriorAssertion.receipt.WellScopedAfter
        afterRecursiveEquality :=
    PrimitivePlan.wellScopedAfter pPriorAssertion afterRecursiveEquality
      (ReceiptScoping.Covers.primitiveInputs
        equalityCovers oneAfterRecursiveEquality falseAfterEquality)
  have oneAfterPriorAssertion : oneColumn ∈ afterPriorAssertion :=
    List.mem_append_left _ oneAfterRecursiveEquality
  have falseAfterPriorAssertion :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterPriorAssertion :=
    List.mem_append_left _ falseAfterEquality
  have equalityAfterPriorAssertion :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.afterEncodedEquality parameters)
        afterPriorAssertion :=
    equalityCovers.weaken pPriorAssertion.receipt.columnIds
  constructor
  · exact priorAssertionScoped

  have runningCheckScoped :
      pRunningCheck.receipt.WellScopedAfter afterPriorAssertion :=
    PrimitivePlan.wellScopedAfter pRunningCheck afterPriorAssertion
      (ReceiptScoping.Covers.primitiveInputs
        equalityAfterPriorAssertion oneAfterPriorAssertion
          falseAfterPriorAssertion)
  have oneAfterRunningCheck : oneColumn ∈ afterRunningCheck :=
    List.mem_append_left _ oneAfterPriorAssertion
  have falseAfterRunningCheck :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterRunningCheck :=
    List.mem_append_left _ falseAfterPriorAssertion
  have runningCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.afterRunningCheck parameters)
        afterRunningCheck := by
    simpa only [pRunningCheck, recursiveRunningCheckPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Terminal.afterRunningCheck] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pRunningCheck afterPriorAssertion
          equalityAfterPriorAssertion
  constructor
  · exact runningCheckScoped

  have runningAssertionScoped :
      pRunningAssertion.receipt.WellScopedAfter afterRunningCheck :=
    PrimitivePlan.wellScopedAfter pRunningAssertion afterRunningCheck
      (ReceiptScoping.Covers.primitiveInputs
        runningCovers oneAfterRunningCheck falseAfterRunningCheck)
  have oneAfterRunningAssertion : oneColumn ∈ afterRunningAssertion :=
    List.mem_append_left _ oneAfterRunningCheck
  have falseAfterRunningAssertion :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterRunningAssertion :=
    List.mem_append_left _ falseAfterRunningCheck
  have runningAfterAssertion :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.afterRunningCheck parameters)
        afterRunningAssertion :=
    runningCovers.weaken pRunningAssertion.receipt.columnIds
  constructor
  · exact runningAssertionScoped

  have freshCheckScoped :
      pFreshCheck.receipt.WellScopedAfter afterRunningAssertion :=
    PrimitivePlan.wellScopedAfter pFreshCheck afterRunningAssertion
      (ReceiptScoping.Covers.primitiveInputs
        runningAfterAssertion oneAfterRunningAssertion
          falseAfterRunningAssertion)
  have oneAfterFreshCheck : oneColumn ∈ afterFreshCheck :=
    List.mem_append_left _ oneAfterRunningAssertion
  have falseAfterFreshCheck :
      activationColumn SourceOwners.terminalBranchPath false ∈
        afterFreshCheck :=
    List.mem_append_left _ falseAfterRunningAssertion
  have freshCheckCovers :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.afterFreshCheck parameters)
        afterFreshCheck := by
    simpa only [pFreshCheck, recursiveFreshCheckPlan,
      ReceiptScoping.PrimitivePlan.ResultCoveredAfter,
      PrimitivePlan.receipt,
      CanonicalContexts.Terminal.afterFreshCheck] using
      ReceiptScoping.PrimitivePlan.resultCoveredAfter
        pFreshCheck afterRunningAssertion runningAfterAssertion
  constructor
  · exact freshCheckScoped

  have freshAssertionScoped :
      pFreshAssertion.receipt.WellScopedAfter afterFreshCheck :=
    PrimitivePlan.wellScopedAfter pFreshAssertion afterFreshCheck
      (ReceiptScoping.Covers.primitiveInputs
        freshCheckCovers oneAfterFreshCheck falseAfterFreshCheck)
  constructor
  · exact freshAssertionScoped

  have joinScoped :
      joinReceipt.WellScopedAfter afterFreshAssertion :=
    CanonicalBranchPlan.emptyJoinReceipt_wellScoped
      SourceOwners.terminalBranchPath afterFreshAssertion
  constructor
  · exact joinScoped
  · trivial

/-- The constructed body receipts preserve the exact structural Terminal
owner order. -/
theorem bodyOwnersExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (bodyReceipts parameters profile recipes).map
        (fun receipt => receipt.owner) =
      SourceOwners.terminalBodyOwners := by
  simp [bodyReceipts, SourceOwners.terminalBodyOwners,
    PrimitivePlan.receipt_owner]

/-- The complete constructed receipt list has exactly the source-derived
Terminal owner skeleton. -/
theorem ownersExact
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    (receipts parameters profile recipes).map
        (fun receipt => receipt.owner) =
      SourceOwners.terminalOwners parameters := by
  simp [receipts, SourceOwners.terminalOwners,
    SourceAlignment.inputOwners, InputReceipts.owners_exact,
    bodyOwnersExact, InstructionReceipt.prelude]

/-- Structural paths and input slots in the Terminal owner skeleton are
collision-free. -/
theorem sourceOwnersNodup (parameters : Parameters) :
    (SourceOwners.terminalOwners parameters).Nodup := by
  rw [SourceOwners.terminalOwners]
  constructor
  · intro owner member equal
    subst owner
    simp [SourceAlignment.inputOwners,
      SourceOwners.terminalBodyOwners] at member
  · have tailNodup :
        (SourceAlignment.inputOwners (terminalInputSchema parameters) ++
          SourceOwners.terminalBodyOwners).Nodup := by
      rw [List.nodup_append]
      refine ⟨?_, ?_, ?_⟩
      · have inputNodup :=
          InputReceipts.ownersNodup (terminalInputSchema parameters)
        rw [InputReceipts.owners_exact] at inputNodup
        exact inputNodup
      · decide
      · simp [SourceAlignment.inputOwners,
          SourceOwners.terminalBodyOwners]
    exact tailNodup

theorem bodyLocalColumnIdsNodup
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ∀ receipt,
      receipt ∈ bodyReceipts parameters profile recipes ->
        receipt.columnIds.Nodup := by
  intro receipt member
  simp only [bodyReceipts, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl
  · exact PrimitivePlan.columnIdsNodup _
  · exact CanonicalBranchPlan.trueActivationReceipt_columnIdsNodup _ _ _ _
  · exact CanonicalBranchPlan.falseActivationReceipt_columnIdsNodup _ _ _ _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact PrimitivePlan.columnIdsNodup _
  · exact CanonicalBranchPlan.emptyJoinReceipt_columnIdsNodup _

theorem bodyLocalRowIdsNodup
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ∀ receipt,
      receipt ∈ bodyReceipts parameters profile recipes ->
        receipt.rowIds.Nodup := by
  intro receipt member
  simp only [bodyReceipts, List.mem_cons, List.not_mem_nil,
    or_false] at member
  rcases member with
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
      rfl | rfl | rfl | rfl
  · exact PrimitivePlan.rowIdsNodup _
  · exact CanonicalBranchPlan.trueActivationReceipt_rowIdsNodup _ _ _ _
  · exact CanonicalBranchPlan.falseActivationReceipt_rowIdsNodup _ _ _ _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact PrimitivePlan.rowIdsNodup _
  · exact CanonicalBranchPlan.emptyJoinReceipt_rowIdsNodup _

theorem localColumnIdsNodup
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ∀ receipt,
      receipt ∈ receipts parameters profile recipes ->
        receipt.columnIds.Nodup := by
  intro receipt member
  rw [receipts] at member
  rcases List.mem_cons.mp member with prelude | tail
  · subst receipt
    simp [InstructionReceipt.prelude, InstructionReceipt.columnIds,
      preludeColumns]
  · rcases List.mem_append.mp tail with input | body
    · exact InputReceipts.localColumnIdsNodup
        (terminalInputSchema parameters) receipt input
    · exact bodyLocalColumnIdsNodup
        parameters profile recipes receipt body

theorem localRowIdsNodup
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ∀ receipt,
      receipt ∈ receipts parameters profile recipes ->
        receipt.rowIds.Nodup := by
  intro receipt member
  rw [receipts] at member
  rcases List.mem_cons.mp member with prelude | tail
  · subst receipt
    simp [InstructionReceipt.prelude, InstructionReceipt.rowIds]
  · rcases List.mem_append.mp tail with input | body
    · exact InputReceipts.localRowIdsNodup
        (terminalInputSchema parameters) receipt input
    · exact bodyLocalRowIdsNodup
        parameters profile recipes receipt body

/-- Prelude, inputs, and the exact Terminal body form one premise-free scoped
receipt sequence. -/
theorem wellScoped
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ReceiptsWellScoped []
      (receipts parameters profile recipes) := by
  let inputPrefix :=
    InstructionReceipt.prelude ::
      InputReceipts.receipts (terminalInputSchema parameters)
  have prefixScoped : ReceiptsWellScoped [] inputPrefix := by
    simpa only [inputPrefix] using
      InputReceipts.wellScopedAfterPrelude
        (terminalInputSchema parameters)
  have oneInPrefix :
      oneColumn ∈
        inputPrefix.flatMap InstructionReceipt.columnIds := by
    simp [inputPrefix, InstructionReceipt.prelude_columnIds]
  have inputCovered :
      ReceiptScoping.Covers
        (CanonicalContexts.Terminal.input parameters)
        (inputPrefix.flatMap InstructionReceipt.columnIds) := by
    intro column member
    simp only [inputPrefix, List.flatMap_cons,
      InstructionReceipt.prelude_columnIds]
    apply List.mem_append_right [oneColumn]
    rw [InputReceipts.columnIds_exact]
    exact member
  have bodyScoped :
      ReceiptsWellScoped
        (inputPrefix.flatMap InstructionReceipt.columnIds)
        (bodyReceipts parameters profile recipes) :=
    bodyWellScoped parameters profile recipes
      (inputPrefix.flatMap InstructionReceipt.columnIds)
      oneInPrefix inputCovered
  have combined :=
    ReceiptScoping.wellScoped_append
      [] inputPrefix
      (bodyReceipts parameters profile recipes)
      prefixScoped bodyScoped
  simpa only [inputPrefix, receipts] using combined

/-- Conserved physical Terminal program: every column and row is owned by
exactly one receipt, and the receipt sequence is scoped from the empty
prefix. -/
def physical
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    ReceiptProgram
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
        parameters) where
  receipts := receipts parameters profile recipes
  preludeMember := by
    simp [receipts]
  ownersNodup := by
    rw [ownersExact]
    exact sourceOwnersNodup parameters
  localColumnIdsNodup :=
    localColumnIdsNodup parameters profile recipes
  localRowIdsNodup :=
    localRowIdsNodup parameters profile recipes
  wellScoped :=
    wellScoped parameters profile recipes

/-- The conserved physical Terminal program has exactly the owner skeleton
derived from the typed Terminal AST. -/
def aligned
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters)) :
    SourceAlignment.AlignedReceiptProgram
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.program
        parameters) where
  physical := physical parameters profile recipes
  ownersExact := by
    rw [SourceOwners.terminalProgramOwnersExact]
    exact ownersExact parameters profile recipes

end CanonicalTerminalPlan

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
