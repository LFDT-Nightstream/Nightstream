import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep
import Nightstream.Implementation.Lowering.Goldilocks.NativeCcsCompiler

/-!
Contract: finite four-matrix compiler validity for the selected native-CCS
fixed-one Step program.

Assurance tier: model-level.

Owns:
- proof that the removed activation residual suffix is not referenced by any
  surviving selected row;
- exact support of every native selector and source-row dependency by the
  surviving allocation stream;
- the premise-free `NativeCcsCompiler.Valid` value for the selected program.

Does not own: Boolean row-domain selection, Phi81 carrier padding, a
deployment application, JSON, Rust parsing, commitments, or a security
reduction.

Emits constraints: none. It certifies the exact program emitted by
`ConcreteNifsNativeCcsStep`.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsCompiler

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

section CompleteStep

variable {shape : SemanticShape}
variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {publicRingColumns verifierRows : Nat}
variable {publicFits :
  ringDegree * publicRingColumns <= shape.carrierWidth}
variable {keys : Fin 1 →
  SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows}
variable {defaultRunning :
  SelectedRunning shape publicRingColumns publicFits verifierRows}
variable {machine :
  Machine
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits
      verifierRows)
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    RunningWitness
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    FreshWitness 1}
variable {terminalChecks :
  Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
    terminalRelations}
variable {widths : Widths} {footprints : Footprints}

local notation "Selected" =>
  ConcreteNifsParameters.selected keys defaultRunning machine
    terminalRelations terminalChecks widths footprints

private abbrev StepRecipeFor
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected) :=
  CallRecipe (signature Selected)
    (application.profile.family Selected) Call.step

private abbrev DefaultAdmissibleFor
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected) :=
  ((application.profile.family Selected).codecFor (.data .running)).Admissible
    defaultRunning

private def sourceEncoding
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :=
  CanonicalStepSoundness.encoding Selected
    (ConcreteNifsNativeCcsStep.certificate
      application nifs step defaultAdmissible).baseProfile
    (ConcreteNifsNativeCcsStep.recipes
      application nifs step defaultAdmissible)
    defaultAdmissible

private def prefixReceipts
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    List InstructionReceipt :=
  InstructionReceipt.prelude ::
    InputReceipts.receipts (stepInputSchema Selected) ++
      (CanonicalStepPlan.bodyReceipts Selected
        (ConcreteNifsNativeCcsStep.certificate
          application nifs step defaultAdmissible).baseProfile
        (ConcreteNifsNativeCcsStep.recipes
          application nifs step defaultAdmissible)
        defaultAdmissible).take 12

private def tailReceipts
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    List InstructionReceipt :=
  (CanonicalStepPlan.bodyReceipts Selected
    (ConcreteNifsNativeCcsStep.certificate
      application nifs step defaultAdmissible).baseProfile
    (ConcreteNifsNativeCcsStep.recipes
      application nifs step defaultAdmissible)
    defaultAdmissible).drop 13

private theorem sourceReceipts_split
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    ConcreteNifsNativeCcsStep.sourceReceipts
        application nifs step defaultAdmissible =
      prefixReceipts application nifs step defaultAdmissible ++
        ConcreteNifsNativeCcsStep.targetReceipt
          application nifs step defaultAdmissible ::
        tailReceipts application nifs step defaultAdmissible := by
  simp [ConcreteNifsNativeCcsStep.sourceReceipts,
    ConcreteNifsNativeCcsStep.certificate,
    ConcreteNifsNativeCcsStep.recipes,
    CompleteApplicationCertification.canonicalStep,
    CanonicalEncodingRealization.step, CanonicalStepPlan.aligned,
    CanonicalStepPlan.physical, CanonicalStepPlan.receipts,
    CanonicalStepPlan.bodyReceipts, prefixReceipts, tailReceipts,
    ConcreteNifsNativeCcsStep.targetReceipt]

private theorem prefix_scoped
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    ReceiptsWellScoped []
      (prefixReceipts application nifs step defaultAdmissible) := by
  have sequenceScoped :=
    (sourceEncoding application nifs step defaultAdmissible).wellScoped
  have sourceScoped :
      ReceiptsWellScoped []
        (ConcreteNifsNativeCcsStep.sourceReceipts
          application nifs step defaultAdmissible) := by
    simpa [sourceEncoding] using sequenceScoped
  rw [sourceReceipts_split] at sourceScoped
  exact ReceiptScoping.wellScoped_prefix []
    (prefixReceipts application nifs step defaultAdmissible)
    (ConcreteNifsNativeCcsStep.targetReceipt
      application nifs step defaultAdmissible ::
      tailReceipts application nifs step defaultAdmissible)
    sourceScoped

private theorem prefix_column_not_residual
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipt : InstructionReceipt)
    (receiptMember :
      receipt ∈ prefixReceipts application nifs step defaultAdmissible)
    (row : OwnedRow)
    (rowMember : row ∈ receipt.rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∉ ConcreteNifsActivatedProgram.residuals
      application.profile nifs.operational
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame := by
  have prefixMember :
      column ∈
        (prefixReceipts application nifs step defaultAdmissible).flatMap
          InstructionReceipt.columnIds := by
    simpa using
      Nightstream.Implementation.Lowering.Goldilocks.Encoding.receipts_rows_supported []
        (prefixReceipts application nifs step defaultAdmissible)
        (prefix_scoped application nifs step defaultAdmissible)
        receipt receiptMember row rowMember column columnMember
  intro residualMember
  have targetMember :
      column ∈
        (ConcreteNifsNativeCcsStep.targetReceipt
          application nifs step defaultAdmissible).columnIds := by
    rw [
      ConcreteNifsNativeCcsStep.targetReceipt_columnIds_eq_native_append_residuals]
    exact List.mem_append_right _ residualMember
  have allNodup :=
    (sourceEncoding application nifs step defaultAdmissible).columnIdsNodup
  have sourceNodup :
      ((ConcreteNifsNativeCcsStep.sourceReceipts
        application nifs step defaultAdmissible).flatMap
          InstructionReceipt.columnIds).Nodup := by
    simpa [sourceEncoding] using allNodup
  rw [sourceReceipts_split] at sourceNodup
  simp only [List.flatMap_append, List.flatMap_cons] at sourceNodup
  have cross := (List.nodup_append.1 sourceNodup).2.2
  exact cross column prefixMember column
    (List.mem_append_left _ targetMember) rfl

private theorem residual_contradiction_of_owner
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (column : ColumnId)
    (ownerExcludes :
      column.owner ≠ ConcreteNifsNativeCcsStep.targetOwner)
    (residualMember :
      column ∈ ConcreteNifsActivatedProgram.residuals
        application.profile nifs.operational
        (ConcreteNifsNativeCcsStep.invokePlan
          application nifs step defaultAdmissible).frame) :
    False :=
  ownerExcludes
    (ConcreteNifsNativeCcsStep.residual_owner
      application nifs step defaultAdmissible column residualMember)

private theorem join_column_not_residual
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (row : OwnedRow)
    (rowMember :
      row ∈
        (CanonicalBranchPlan.onePortJoinReceipt
          SourceOwners.stepBranchPath
          (CanonicalContexts.Step.selector Selected
            (ConcreteNifsNativeCcsStep.certificate
              application nifs step defaultAdmissible).baseProfile)
          (Ports.committedRunning Selected)
          (CanonicalContexts.Step.baseRunning Selected)
          (CanonicalContexts.Step.recursiveRunning Selected)).rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∉ ConcreteNifsActivatedProgram.residuals
      application.profile nifs.operational
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame := by
  let profile :=
    (ConcreteNifsNativeCcsStep.certificate
      application nifs step defaultAdmissible).baseProfile
  let selector := CanonicalContexts.Step.selector Selected profile
  let join :=
    CanonicalBranchPlan.onePortJoinRecipe
      SourceOwners.stepBranchPath selector
      (Ports.committedRunning Selected)
      (CanonicalContexts.Step.baseRunning Selected)
      (CanonicalContexts.Step.recursiveRunning Selected)
  have rowMember' : row ∈ join.rows := by
    simpa [join, selector, profile,
      CanonicalBranchPlan.onePortJoinReceipt] using rowMember
  have supported :=
    MuxRecipe.rows_supported join row rowMember' column columnMember
  simp only [join, CanonicalBranchPlan.onePortJoinRecipe,
    List.mem_append, List.mem_singleton] at supported
  intro residualMember
  rcases supported with
    ((selectorMember | joinedMember) | baseMember) | recursiveMember
  · subst column
    have selectorInCommon :=
      CanonicalPrimitivePlan.bitCoordinate_mem profile
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iterationZero
          Selected)
        (CanonicalContexts.Step.common Selected)
        (CanonicalContexts.Step.commonWidths Selected profile)
    have ownerExcludes :=
      CanonicalPrimitivePlan.ContextExcludesOwner.id_excludes
        (CanonicalStepPlan.common_excludes Selected
          SourceOwners.stepRecursiveNifsPath (by decide) (by decide))
        selector selectorInCommon
    exact residual_contradiction_of_owner
      application nifs step defaultAdmissible selector ownerExcludes
      residualMember
  · have ownerExcludes :=
      CanonicalPrimitivePlan.ContextExcludesOwner.id_excludes
        (CanonicalPrimitivePlan.ContextExcludesOwner.branch
          SourceOwners.stepBranchPath
          SourceOwners.stepRecursiveNifsPath
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema
            Selected))
        column
        (by
          simp only [
            Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.joinedSchema]
          rw [ReceiptScoping.singletonColumnsIds]
          simpa [CanonicalContexts.Step.joined] using joinedMember)
    exact residual_contradiction_of_owner
      application nifs step defaultAdmissible column ownerExcludes
      residualMember
  · have ownerExact :=
      CanonicalPrimitivePlan.instruction_id_owner
        SourceOwners.stepBaseDefaultPath
        [Ports.committedRunning Selected] column
        (by
          rw [ReceiptScoping.singletonColumnsIds]
          simpa [CanonicalContexts.Step.baseRunning] using baseMember)
    have ownerExcludes :
        column.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
      rw [ownerExact]
      decide
    exact residual_contradiction_of_owner
      application nifs step defaultAdmissible column ownerExcludes
      residualMember
  · have outputMember :
        column ∈
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame.outputs.ids := by
      rw [
        (ConcreteNifsNativeCcsStep.invokePlan
          application nifs step defaultAdmissible).outputsExact]
      change
        column ∈
          (instructionColumns SourceOwners.stepRecursiveNifsPath
            [Ports.committedRunning Selected]).toSchemaBundles.ids
      rw [ReceiptScoping.singletonColumnsIds]
      simpa [CanonicalContexts.Step.recursiveRunning] using recursiveMember
    have visibleMember :
        column ∈
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame.visibleIds := by
      simp only [CallFrame.visibleIds]
      exact List.mem_append_right _ outputMember
    exact
      ConcreteNifsActivatedProgram.residuals_disjoint_visible
        application.profile nifs.operational
        (ConcreteNifsNativeCcsStep.invokePlan
          application nifs step defaultAdmissible).frame
        column residualMember visibleMember

private theorem continuation_column_not_residual
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (row : OwnedRow)
    (rowMember :
      row ∈
        (CanonicalStepPlan.continuationHashPlan Selected
          (ConcreteNifsNativeCcsStep.certificate
            application nifs step defaultAdmissible).baseProfile
          (ConcreteNifsNativeCcsStep.recipes
            application nifs step defaultAdmissible)).receipt.rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∉ ConcreteNifsActivatedProgram.residuals
      application.profile nifs.operational
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame := by
  let invoke :=
    CanonicalStepPlan.continuationHashInvokePlan Selected
      (ConcreteNifsNativeCcsStep.certificate
        application nifs step defaultAdmissible).baseProfile
      (ConcreteNifsNativeCcsStep.recipes
        application nifs step defaultAdmissible)
  have recipeMember : row ∈ invoke.recipe.rows invoke.frame := by
    simpa [CanonicalStepPlan.continuationHashPlan, invoke,
      PrimitivePlan.receipt, InvokePlan.receipt] using rowMember
  have supported :=
    invoke.recipe.rowsSupported invoke.frame row recipeMember
      column columnMember
  intro residualMember
  rcases List.mem_append.1 supported with visibleMember | temporaryMember
  · simp only [CallFrame.visibleIds] at visibleMember
    rcases List.mem_append.1 visibleMember with
      controlOrContext | outputMember
    · rcases List.mem_append.1 controlOrContext with
        controlMember | contextMember
      · rw [invoke.oneExact, invoke.activeExact] at controlMember
        simp only [List.mem_cons, List.not_mem_nil, or_false] at controlMember
        rcases controlMember with rfl | rfl
        · exact residual_contradiction_of_owner
            application nifs step defaultAdmissible oneColumn
            (by
              change
                PhysicalOwner.prelude ≠
                  .typed
                    (.instruction SourceOwners.stepRecursiveNifsPath)
              decide)
            residualMember
        · exact residual_contradiction_of_owner
            application nifs step defaultAdmissible oneColumn
            (by
              change
                PhysicalOwner.prelude ≠
                  .typed
                    (.instruction SourceOwners.stepRecursiveNifsPath)
              decide)
            residualMember
      · have contextMember' :
            column ∈
              (CanonicalContexts.Step.continuationInput
                Selected).toSchemaBundles.ids := by
          rw [invoke.contextExact] at contextMember
          exact contextMember
        have ownerExcludes :=
          CanonicalPrimitivePlan.ContextExcludesOwner.id_excludes
            (CanonicalStepPlan.continuationInput_excludes Selected
              SourceOwners.stepRecursiveNifsPath
              (by decide) (by decide))
            column contextMember'
        exact residual_contradiction_of_owner
          application nifs step defaultAdmissible column ownerExcludes
          residualMember
    · have outputMember' :
          column ∈
            (instructionColumns
              SourceOwners.stepContinuationHashPath
              ((signature Selected).callOutputs Call.hashNext)
            ).toSchemaBundles.ids := by
        rw [← invoke.outputsExact]
        exact outputMember
      have ownerExact :=
        CanonicalPrimitivePlan.instruction_id_owner
          SourceOwners.stepContinuationHashPath
          ((signature Selected).callOutputs Call.hashNext)
          column outputMember'
      have ownerExcludes :
          column.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
        rw [ownerExact]
        decide
      exact residual_contradiction_of_owner
        application nifs step defaultAdmissible column ownerExcludes
        residualMember
  · have temporaryMember' :
        column ∈
          (temporaryColumns
            SourceOwners.stepContinuationHashPath
            ((signature Selected).callOutputs Call.hashNext)
            ((signature Selected).callFootprint
              Call.hashNext).temporaries).toLayoutBundles.ids := by
      rw [← invoke.temporariesExact]
      exact temporaryMember
    have ownerExact :=
      CanonicalPrimitivePlan.temporary_id_owner
        SourceOwners.stepContinuationHashPath
        ((signature Selected).callOutputs Call.hashNext)
        ((signature Selected).callFootprint Call.hashNext).temporaries
        column temporaryMember'
    have ownerExcludes :
        column.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
      rw [ownerExact]
      decide
    exact residual_contradiction_of_owner
      application nifs step defaultAdmissible column ownerExcludes
      residualMember

private theorem tail_column_not_residual
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipt : InstructionReceipt)
    (receiptMember :
      receipt ∈ tailReceipts application nifs step defaultAdmissible)
    (row : OwnedRow)
    (rowMember : row ∈ receipt.rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∉ ConcreteNifsActivatedProgram.residuals
      application.profile nifs.operational
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame := by
  have choices :
      receipt =
          CanonicalBranchPlan.onePortJoinReceipt
            SourceOwners.stepBranchPath
            (CanonicalContexts.Step.selector Selected
              (ConcreteNifsNativeCcsStep.certificate
                application nifs step defaultAdmissible).baseProfile)
            (Ports.committedRunning Selected)
            (CanonicalContexts.Step.baseRunning Selected)
            (CanonicalContexts.Step.recursiveRunning Selected) ∨
        receipt =
          (CanonicalStepPlan.continuationHashPlan Selected
            (ConcreteNifsNativeCcsStep.certificate
              application nifs step defaultAdmissible).baseProfile
            (ConcreteNifsNativeCcsStep.recipes
              application nifs step defaultAdmissible)).receipt := by
    simpa [tailReceipts, CanonicalStepPlan.bodyReceipts] using receiptMember
  rcases choices with rfl | rfl
  · exact join_column_not_residual
      application nifs step defaultAdmissible
      row rowMember column columnMember
  · exact continuation_column_not_residual
      application nifs step defaultAdmissible
      row rowMember column columnMember

private theorem ordinary_column_not_residual
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipt : InstructionReceipt)
    (receiptMember :
      receipt ∈ ConcreteNifsNativeCcsStep.sourceReceipts
        application nifs step defaultAdmissible)
    (other :
      receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner)
    (row : OwnedRow)
    (rowMember : row ∈ receipt.rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∉ ConcreteNifsActivatedProgram.residuals
      application.profile nifs.operational
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame := by
  rw [sourceReceipts_split] at receiptMember
  rcases List.mem_append.1 receiptMember with prefixMember | restMember
  · exact prefix_column_not_residual
      application nifs step defaultAdmissible receipt prefixMember
      row rowMember column columnMember
  · rcases List.mem_cons.1 restMember with targetEqual | tailMember
    · subst receipt
      exact False.elim (other
        (ConcreteNifsNativeCcsStep.targetReceipt_owner
          application nifs step defaultAdmissible))
    · exact tail_column_not_residual
        application nifs step defaultAdmissible receipt tailMember
        row rowMember column columnMember

private theorem source_column_allocated
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipt : InstructionReceipt)
    (receiptMember :
      receipt ∈ ConcreteNifsNativeCcsStep.sourceReceipts
        application nifs step defaultAdmissible)
    (row : OwnedRow)
    (rowMember : row ∈ receipt.rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈
      (sourceEncoding application nifs step defaultAdmissible).columnIds :=
  Nightstream.Implementation.Lowering.Goldilocks.Encoding.rows_supported
    (sourceEncoding application nifs step defaultAdmissible)
    receipt
    (by simpa [sourceEncoding] using receiptMember)
    row rowMember column columnMember

private theorem source_column_in_receipts_of_allocated
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (column : ColumnId)
    (allocated :
      column ∈
        (sourceEncoding application nifs step defaultAdmissible).columnIds) :
    column ∈
      (ConcreteNifsNativeCcsStep.sourceReceipts
        application nifs step defaultAdmissible).flatMap
          InstructionReceipt.columnIds := by
  rw [
    Nightstream.Implementation.Lowering.Goldilocks.Encoding.columnIds_eq_receipt_columnIds
      ] at allocated
  simpa only [ConcreteNifsNativeCcsStep.sourceReceipts_encoding] using
    allocated

private theorem native_target_column_allocated
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (row : SelectedRow)
    (rowMember :
      row ∈
        (ConcreteNifsNativeCcsStep.nativeReceipt
          application nifs step defaultAdmissible).rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈
      (ConcreteNifsNativeCcsStep.program
        application nifs step defaultAdmissible).columnIds := by
  rcases List.mem_map.1 rowMember with
    ⟨source, selectedMember, rfl⟩
  have rawMember :
      source.row ∈
        ConcreteNifsRawProgram.rawRows
          application.profile nifs.operational
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame := by
    exact DirectCalls.ownRows_row_mem _ _ source
      (by simpa [ConcreteNifsNativeCcsProgram.sourceReceipt,
        ConcreteNifsRawProgram.rows] using selectedMember)
  simp only [SelectedRow.columnIds, List.mem_cons] at columnMember
  rcases columnMember with selectorEqual | sourceColumn
  · subst column
    rcases ActivatedRawProgram.selector_emitted_of_source_mem
        (ConcreteNifsNativeCcsStep.targetReceipt
          application nifs step defaultAdmissible).owner
        (ConcreteNifsNativeCcsStep.invokePlan
          application nifs step defaultAdmissible).frame.active
        (ConcreteNifsRawProgram.rawRows
          application.profile nifs.operational
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame)
        (ConcreteNifsActivatedProgram.residuals
          application.profile nifs.operational
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame)
        (ConcreteNifsActivatedProgram.residuals_length
          application.profile nifs.operational nifs.footprint
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame).symm
        source.row rawMember with
      ⟨emitted, emittedMember, activeMember⟩
    have sourceAllocated :=
      source_column_allocated application nifs step defaultAdmissible
        (ConcreteNifsNativeCcsStep.targetReceipt
          application nifs step defaultAdmissible)
        (ConcreteNifsNativeCcsStep.targetReceipt_member
          application nifs step defaultAdmissible)
        emitted
        (by simpa using emittedMember)
        (ConcreteNifsNativeCcsStep.invokePlan
          application nifs step defaultAdmissible).frame.active
        activeMember
    apply ConcreteNifsNativeCcsStep.source_column_mem_selected_of_not_residual
      application nifs step defaultAdmissible _
      (source_column_in_receipts_of_allocated
        application nifs step defaultAdmissible _ sourceAllocated)
    change
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame.active ∉
        ConcreteNifsActivatedProgram.residuals
          application.profile nifs.operational
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame
    exact ConcreteNifsActivatedProgram.active_not_residual
      application.profile nifs.operational
      (ConcreteNifsNativeCcsStep.invokePlan
        application nifs step defaultAdmissible).frame
  · rcases ActivatedRawProgram.source_column_emitted
        (ConcreteNifsNativeCcsStep.targetReceipt
          application nifs step defaultAdmissible).owner
        (ConcreteNifsNativeCcsStep.invokePlan
          application nifs step defaultAdmissible).frame.active
        (ConcreteNifsRawProgram.rawRows
          application.profile nifs.operational
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame)
        (ConcreteNifsActivatedProgram.residuals
          application.profile nifs.operational
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame)
        (ConcreteNifsActivatedProgram.residuals_length
          application.profile nifs.operational nifs.footprint
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame).symm
        source.row rawMember column sourceColumn with
      ⟨emitted, emittedMember, emittedColumn⟩
    have sourceAllocated :=
      source_column_allocated application nifs step defaultAdmissible
        (ConcreteNifsNativeCcsStep.targetReceipt
          application nifs step defaultAdmissible)
        (ConcreteNifsNativeCcsStep.targetReceipt_member
          application nifs step defaultAdmissible)
        emitted
        (by simpa using emittedMember)
        column emittedColumn
    apply ConcreteNifsNativeCcsStep.source_column_mem_selected_of_not_residual
      application nifs step defaultAdmissible column
      (source_column_in_receipts_of_allocated
        application nifs step defaultAdmissible column sourceAllocated)
    intro residualMember
    exact
      ConcreteNifsActivatedProgram.residuals_fresh
        application.profile nifs.operational nifs.footprint
        (ConcreteNifsNativeCcsStep.invokePlan
          application nifs step defaultAdmissible).frame
        column residualMember
        (List.mem_flatMap.mpr ⟨
          source.row, rawMember, sourceColumn⟩)

/-- Every selected matrix dependency has one exact surviving allocation.
This is the load-bearing finite-matrix support theorem. -/
theorem rows_supported
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (row : SelectedRow)
    (rowMember :
      row ∈
        (ConcreteNifsNativeCcsStep.program
          application nifs step defaultAdmissible).rows)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈
      (ConcreteNifsNativeCcsStep.program
        application nifs step defaultAdmissible).columnIds := by
  rcases List.mem_flatMap.1 rowMember with
    ⟨selectedReceipt, selectedReceiptMember, rowInReceipt⟩
  change
    selectedReceipt ∈
      ConcreteNifsNativeCcsStep.selectedReceipts
        application nifs step defaultAdmissible at selectedReceiptMember
  rcases List.mem_map.1 selectedReceiptMember with
    ⟨sourceReceipt, sourceReceiptMember, rfl⟩
  by_cases selected :
      sourceReceipt.owner = ConcreteNifsNativeCcsStep.targetOwner
  · simp only [ConcreteNifsNativeCcsStep.replace, selected, if_pos] at rowInReceipt
    exact native_target_column_allocated
      application nifs step defaultAdmissible
      row rowInReceipt column columnMember
  · simp only [ConcreteNifsNativeCcsStep.replace, selected] at rowInReceipt
    rcases List.mem_map.1 rowInReceipt with
      ⟨sourceRow, sourceRowMember, rfl⟩
    simp only [SelectedRow.columnIds, List.mem_cons] at columnMember
    rcases columnMember with rfl | sourceColumn
    · have sourceOne :=
        (sourceEncoding application nifs step defaultAdmissible).oneAllocated
      apply
        ConcreteNifsNativeCcsStep.source_column_mem_selected_of_not_residual
          application nifs step defaultAdmissible oneColumn
          (source_column_in_receipts_of_allocated
            application nifs step defaultAdmissible oneColumn sourceOne)
      intro residualMember
      exact
        ConcreteNifsActivatedProgram.residuals_disjoint_visible
          application.profile nifs.operational
          (ConcreteNifsNativeCcsStep.invokePlan
            application nifs step defaultAdmissible).frame
          oneColumn residualMember
          (by
            simp [CallFrame.visibleIds,
              (ConcreteNifsNativeCcsStep.invokePlan
                application nifs step defaultAdmissible).oneExact])
    · have sourceAllocated :=
        source_column_allocated application nifs step defaultAdmissible
          sourceReceipt sourceReceiptMember sourceRow sourceRowMember
          column sourceColumn
      apply
        ConcreteNifsNativeCcsStep.source_column_mem_selected_of_not_residual
          application nifs step defaultAdmissible column
          (source_column_in_receipts_of_allocated
            application nifs step defaultAdmissible column sourceAllocated)
      exact ordinary_column_not_residual
        application nifs step defaultAdmissible
        sourceReceipt sourceReceiptMember selected
        sourceRow sourceRowMember column sourceColumn

/-- The selected native Step program is a valid finite four-matrix compiler
input with no fallback index used by any row. -/
def valid
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    NativeCcsCompiler.Valid
      (ConcreteNifsNativeCcsStep.program
        application nifs step defaultAdmissible) where
  oneAllocated := by
    have sourceOne :=
      (sourceEncoding application nifs step defaultAdmissible).oneAllocated
    apply
      ConcreteNifsNativeCcsStep.source_column_mem_selected_of_not_residual
        application nifs step defaultAdmissible oneColumn
        (source_column_in_receipts_of_allocated
          application nifs step defaultAdmissible oneColumn sourceOne)
    intro residualMember
    exact
      ConcreteNifsActivatedProgram.residuals_disjoint_visible
        application.profile nifs.operational
        (ConcreteNifsNativeCcsStep.invokePlan
          application nifs step defaultAdmissible).frame
        oneColumn residualMember
        (by
          simp [CallFrame.visibleIds,
            (ConcreteNifsNativeCcsStep.invokePlan
              application nifs step defaultAdmissible).oneExact])
  columnIdsNodup :=
    ConcreteNifsNativeCcsStep.column_ids_nodup
      application nifs step defaultAdmissible
  rowsSupported :=
    rows_supported application nifs step defaultAdmissible

end CompleteStep

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsCompiler
