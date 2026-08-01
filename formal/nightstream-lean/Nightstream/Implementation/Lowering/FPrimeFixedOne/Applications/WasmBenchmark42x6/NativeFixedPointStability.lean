import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointFamily
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepRowsAggregate

/-!
Contract: close the recursive fixed point for the exact native benchmark Step
program.

Assurance tier: model-level.

Owns: equality of the native Step program before and after installation of the
four matrices compiled from that program.

Does not own: terminal R1CS lowering, Spartan, WHIR, Rust, or a security
reduction.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 500000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointStability

open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointCost
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointFamily
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointSource
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsSelector
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentFixedPoint
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4PhysicalFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4OperandFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4RawRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.CurrentM4StepPostRows

noncomputable def nativeComplete
    (setup : RelationSetup dimensions commitmentRows) :
    CompleteApplicationCertification (parameters setup) :=
  ConcreteNifsNativeCcsStep.certificate
    (deployment setup).application.phase4
    (nifsCertificate setup)
    (deployment setup).step
    (deployment setup).defaultRunningAdmissible

theorem nativeComplete_eq_current
    (setup : RelationSetup dimensions commitmentRows) :
    nativeComplete setup =
      CurrentM4PhysicalStability.certificate setup := by
  rfl

noncomputable def beforeTargetReceipts
    (setup : RelationSetup dimensions commitmentRows) :
    List InstructionReceipt :=
  ApplicationStepCostSplit.CompleteApplicationCertification.stepPrefixReceipts
      (nativeComplete setup) ++
    ApplicationStepCostSplit.CompleteApplicationCertification.applicationStepReceipt
        (nativeComplete setup) ::
      (ApplicationStepCostSplit.CompleteApplicationCertification.stepSuffixReceipts
        (nativeComplete setup)).take 11

noncomputable def afterTargetReceipts
    (setup : RelationSetup dimensions commitmentRows) :
    List InstructionReceipt :=
  (ApplicationStepCostSplit.CompleteApplicationCertification.stepSuffixReceipts
    (nativeComplete setup)).drop 12

theorem sourceReceipts_split
    (setup : RelationSetup dimensions commitmentRows) :
    ConcreteNifsNativeCcsStep.sourceReceipts
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible =
      beforeTargetReceipts setup ++
        ConcreteNifsNativeCcsStep.targetReceipt
          (deployment setup).application.phase4
          (nifsCertificate setup)
          (deployment setup).step
          (deployment setup).defaultRunningAdmissible ::
        afterTargetReceipts setup := by
  unfold ConcreteNifsNativeCcsStep.sourceReceipts
  rw [
    ApplicationStepCostSplit.CompleteApplicationCertification.stepReceipts_exact_split]
  unfold beforeTargetReceipts afterTargetReceipts nativeComplete
    ApplicationStepCostSplit.CompleteApplicationCertification.stepSuffixReceipts
    ConcreteNifsNativeCcsStep.targetReceipt ConcreteNifsNativeCcsStep.recipes
    NativeFixedPointCost.parameters ConcreteNifsPlain270Profile.selected
    CanonicalStepPlan.bodyReceipts
  simp

theorem beforeTargetRows
    (setup : RelationSetup dimensions commitmentRows) :
    (beforeTargetReceipts setup).flatMap InstructionReceipt.rows =
      prefixRows setup ++ applicationRows setup ++ preNifsRows setup := by
  unfold beforeTargetReceipts
  rw [nativeComplete_eq_current setup]
  unfold
    ApplicationStepCostSplit.CompleteApplicationCertification.stepSuffixReceipts
    prefixRows applicationRows preNifsRows selectorRows activationRows
    baseRows recursivePreNifsRows flattenRows selectedParameters
    NativeFixedPointCost.parameters ConcreteNifsPlain270Profile.selected
    CanonicalStepPlan.bodyReceipts
  simp

theorem afterTargetRows
    (setup : RelationSetup dimensions commitmentRows) :
    (afterTargetReceipts setup).flatMap InstructionReceipt.rows =
      postNifsRows setup := by
  unfold afterTargetReceipts
  rw [nativeComplete_eq_current setup]
  unfold
    ApplicationStepCostSplit.CompleteApplicationCertification.stepSuffixReceipts
    postNifsRows joinRows continuationRows selectedParameters
    NativeFixedPointCost.parameters ConcreteNifsPlain270Profile.selected
    CanonicalStepPlan.bodyReceipts
  simp

private theorem beforeTarget_no_target
    (setup : RelationSetup dimensions commitmentRows) :
    ∀ receipt, receipt ∈ beforeTargetReceipts setup →
      receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
  have ownersNodup :
      ((ConcreteNifsNativeCcsStep.sourceReceipts
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible).map
          fun receipt => receipt.owner).Nodup :=
    (nativeComplete setup).canonicalStep.program.physical.ownersNodup
  rw [sourceReceipts_split setup] at ownersNodup
  simp only [List.map_append, List.map_cons,
    ConcreteNifsNativeCcsStep.targetReceipt_owner] at ownersNodup
  have split := List.nodup_append.mp ownersNodup
  intro receipt member equal
  exact split.2.2
    ConcreteNifsNativeCcsStep.targetOwner
    (List.mem_map.mpr ⟨receipt, member, equal⟩)
    ConcreteNifsNativeCcsStep.targetOwner (by simp) rfl

private theorem afterTarget_no_target
    (setup : RelationSetup dimensions commitmentRows) :
    ∀ receipt, receipt ∈ afterTargetReceipts setup →
      receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
  have ownersNodup :
      ((ConcreteNifsNativeCcsStep.sourceReceipts
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible).map
          fun receipt => receipt.owner).Nodup :=
    (nativeComplete setup).canonicalStep.program.physical.ownersNodup
  rw [sourceReceipts_split setup] at ownersNodup
  simp only [List.map_append, List.map_cons,
    ConcreteNifsNativeCcsStep.targetReceipt_owner] at ownersNodup
  have split := List.nodup_append.mp ownersNodup
  have targetNotAfter :=
    (List.nodup_cons.mp split.2.1).1
  intro receipt member equal
  exact targetNotAfter
    (List.mem_map.mpr ⟨receipt, member, equal⟩)

private theorem replaceRows_of_noTarget
    (setup : RelationSetup dimensions commitmentRows)
    (receipts : List InstructionReceipt)
    (noTarget :
      ∀ receipt, receipt ∈ receipts →
        receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner) :
    ((receipts.map
      (ConcreteNifsNativeCcsStep.replace
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible)).flatMap
          SelectedReceipt.rows) =
      select oneColumn (receipts.flatMap InstructionReceipt.rows) := by
  induction receipts with
  | nil => rfl
  | cons head tail inductionHypothesis =>
      have headOther := noTarget head List.mem_cons_self
      have tailOther :
          ∀ receipt, receipt ∈ tail →
            receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
        intro receipt member
        exact noTarget receipt (List.mem_cons_of_mem head member)
      simp only [List.map_cons, List.flatMap_cons]
      rw [ConcreteNifsNativeCcsStep.replace_other
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible head headOther]
      rw [inductionHypothesis tailOther]
      simp only [SelectedReceipt.rows, select, List.map_append]

noncomputable def splitRows
    (setup : RelationSetup dimensions commitmentRows) :
    List SelectedRow :=
  select oneColumn
      ((beforeTargetReceipts setup).flatMap InstructionReceipt.rows) ++
    (ConcreteNifsNativeCcsStep.nativeReceipt
      (deployment setup).application.phase4
      (nifsCertificate setup)
      (deployment setup).step
      (deployment setup).defaultRunningAdmissible).rows ++
    select oneColumn
      ((afterTargetReceipts setup).flatMap InstructionReceipt.rows)

theorem rows_split
    (setup : RelationSetup dimensions commitmentRows) :
    (NativeFixedPointCost.nativeProgram setup).rows =
      splitRows setup := by
  unfold splitRows NativeFixedPointCost.nativeProgram
    ConcreteNifsNativeCcsStep.program NativeCcsProgram.Program.rows
    ConcreteNifsNativeCcsStep.selectedReceipts
  rw [sourceReceipts_split setup]
  simp only [List.map_append, List.map_cons, List.flatMap_append,
    List.flatMap_cons]
  rw [replaceRows_of_noTarget setup (beforeTargetReceipts setup)
      (beforeTarget_no_target setup),
    ConcreteNifsNativeCcsStep.replace_target,
    replaceRows_of_noTarget setup (afterTargetReceipts setup)
      (afterTarget_no_target setup)]
  exact (List.append_assoc _ _ _).symm

noncomputable def beforeTargetColumnIds
    (setup : RelationSetup dimensions commitmentRows) :
    List ColumnId :=
  (beforeTargetReceipts setup).flatMap InstructionReceipt.columnIds

noncomputable def afterTargetColumnIds
    (setup : RelationSetup dimensions commitmentRows) :
    List ColumnId :=
  (afterTargetReceipts setup).flatMap InstructionReceipt.columnIds

noncomputable def residualIds
    (setup : RelationSetup dimensions commitmentRows) :
    List ColumnId :=
  ConcreteNifsActivatedProgram.residuals
    (deployment setup).application.phase4.profile
    (nifsCertificate setup).operational
    (ConcreteNifsNativeCcsStep.invokePlan
      (deployment setup).application.phase4
      (nifsCertificate setup)
      (deployment setup).step
      (deployment setup).defaultRunningAdmissible).frame

noncomputable def sourceColumnIds
    (setup : RelationSetup dimensions commitmentRows) :
    List ColumnId :=
  (ConcreteNifsNativeCcsStep.sourceReceipts
    (deployment setup).application.phase4
    (nifsCertificate setup)
    (deployment setup).step
    (deployment setup).defaultRunningAdmissible).flatMap
      InstructionReceipt.columnIds

private theorem receiptFilter_eq_self
    (setup : RelationSetup dimensions commitmentRows)
    (receipt : InstructionReceipt)
    (other :
      receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner) :
    receipt.columnIds.filter
        (fun column => decide (column ∉ residualIds setup)) =
      receipt.columnIds := by
  apply List.filter_eq_self.mpr
  intro column member
  have owned : column.owner = receipt.owner := by
    rcases List.mem_map.1 member with ⟨ownedColumn, allocated, rfl⟩
    exact receipt.allocationsOwned ownedColumn allocated
  have notResidual : column ∉ residualIds setup := by
    intro residual
    have residualOwned :
        column.owner = ConcreteNifsNativeCcsStep.targetOwner := by
      exact ConcreteNifsNativeCcsStep.residual_owner
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible
        column residual
    exact other (owned.symm.trans residualOwned)
  simp [notResidual]

private theorem filterColumnIds_of_noTarget
    (setup : RelationSetup dimensions commitmentRows)
    (receipts : List InstructionReceipt)
    (noTarget :
      ∀ receipt, receipt ∈ receipts →
        receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner) :
    (receipts.flatMap InstructionReceipt.columnIds).filter
        (fun column => decide (column ∉ residualIds setup)) =
      receipts.flatMap InstructionReceipt.columnIds := by
  induction receipts with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      have headOther := noTarget head List.mem_cons_self
      have tailOther :
          ∀ receipt, receipt ∈ tail →
            receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
        intro receipt member
        exact noTarget receipt (List.mem_cons_of_mem head member)
      simp only [List.flatMap_cons, List.filter_append]
      rw [receiptFilter_eq_self setup head headOther,
        inductionHypothesis tailOther]

private theorem targetReceiptFilter_eq_native
    (setup : RelationSetup dimensions commitmentRows) :
    (ConcreteNifsNativeCcsStep.targetReceipt
      (deployment setup).application.phase4
      (nifsCertificate setup)
      (deployment setup).step
      (deployment setup).defaultRunningAdmissible).columnIds.filter
        (fun column => decide (column ∉ residualIds setup)) =
      (ConcreteNifsNativeCcsStep.nativeReceipt
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible).receipt.columnIds := by
  have targetNodup :
      (ConcreteNifsNativeCcsStep.targetReceipt
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible).columnIds.Nodup :=
    (nativeComplete setup).canonicalStep.program.physical.localColumnIdsNodup
      _ (ConcreteNifsNativeCcsStep.targetReceipt_member
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible)
  rw [ConcreteNifsNativeCcsStep.targetReceipt_columnIds_eq_native_append_residuals]
    at targetNodup ⊢
  rw [List.filter_append]
  have split := List.nodup_append.mp targetNodup
  have nativeKept :
      (ConcreteNifsNativeCcsStep.nativeReceipt
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible).receipt.columnIds.filter
          (fun column => decide (column ∉ residualIds setup)) =
        (ConcreteNifsNativeCcsStep.nativeReceipt
          (deployment setup).application.phase4
          (nifsCertificate setup)
          (deployment setup).step
          (deployment setup).defaultRunningAdmissible).receipt.columnIds := by
    apply List.filter_eq_self.mpr
    intro column member
    have notResidual : column ∉ residualIds setup := by
      intro residual
      exact split.2.2 column member column residual rfl
    simp [notResidual]
  have residualsRemoved :
      (residualIds setup).filter
          (fun column => decide (column ∉ residualIds setup)) =
        [] := by
    apply List.filter_eq_nil_iff.mpr
    intro column member
    simp [member]
  rw [nativeKept]
  change
    (ConcreteNifsNativeCcsStep.nativeReceipt
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible).receipt.columnIds ++
        (residualIds setup).filter
          (fun column => decide (column ∉ residualIds setup)) =
      (ConcreteNifsNativeCcsStep.nativeReceipt
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible).receipt.columnIds
  rw [residualsRemoved, List.append_nil]

private theorem replaceColumnIds_of_noTarget
    (setup : RelationSetup dimensions commitmentRows)
    (receipts : List InstructionReceipt)
    (noTarget :
      ∀ receipt, receipt ∈ receipts →
        receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner) :
    ((receipts.map
      (ConcreteNifsNativeCcsStep.replace
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible)).flatMap
          (fun receipt => receipt.receipt.columnIds)) =
      receipts.flatMap InstructionReceipt.columnIds := by
  induction receipts with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      have headOther := noTarget head List.mem_cons_self
      have tailOther :
          ∀ receipt, receipt ∈ tail →
            receipt.owner ≠ ConcreteNifsNativeCcsStep.targetOwner := by
        intro receipt member
        exact noTarget receipt (List.mem_cons_of_mem head member)
      simp only [List.map_cons, List.flatMap_cons]
      rw [ConcreteNifsNativeCcsStep.replace_other
        (deployment setup).application.phase4
        (nifsCertificate setup)
        (deployment setup).step
        (deployment setup).defaultRunningAdmissible head headOther,
        inductionHypothesis tailOther]

theorem columnIds_split
    (setup : RelationSetup dimensions commitmentRows) :
    (NativeFixedPointCost.nativeProgram setup).columnIds =
      beforeTargetColumnIds setup ++
        (ConcreteNifsNativeCcsStep.nativeReceipt
          (deployment setup).application.phase4
          (nifsCertificate setup)
          (deployment setup).step
          (deployment setup).defaultRunningAdmissible).receipt.columnIds ++
        afterTargetColumnIds setup := by
  rw [NativeCcsProgram.Program.columnIds_conserved]
  unfold beforeTargetColumnIds afterTargetColumnIds
    NativeFixedPointCost.nativeProgram ConcreteNifsNativeCcsStep.program
    ConcreteNifsNativeCcsStep.selectedReceipts
  rw [sourceReceipts_split setup]
  simp only [List.map_append, List.map_cons, List.flatMap_append,
    List.flatMap_cons]
  rw [replaceColumnIds_of_noTarget setup (beforeTargetReceipts setup)
      (beforeTarget_no_target setup),
    ConcreteNifsNativeCcsStep.replace_target,
    replaceColumnIds_of_noTarget setup (afterTargetReceipts setup)
      (afterTarget_no_target setup)]
  exact (List.append_assoc _ _ _).symm

theorem columnIds_eq_source_filter
    (setup : RelationSetup dimensions commitmentRows) :
    (NativeFixedPointCost.nativeProgram setup).columnIds =
      (sourceColumnIds setup).filter
        (fun column => decide (column ∉ residualIds setup)) := by
  rw [columnIds_split setup]
  unfold sourceColumnIds
  rw [sourceReceipts_split setup]
  simp only [List.flatMap_append, List.flatMap_cons, List.filter_append]
  rw [filterColumnIds_of_noTarget setup (beforeTargetReceipts setup)
      (beforeTarget_no_target setup),
    targetReceiptFilter_eq_native setup,
    filterColumnIds_of_noTarget setup (afterTargetReceipts setup)
      (afterTarget_no_target setup)]
  exact List.append_assoc _ _ _

theorem sourceColumnIds_eq_currentEncoding
    (setup : RelationSetup dimensions commitmentRows) :
    sourceColumnIds setup =
      (CurrentM4PhysicalStability.encoding setup).columnIds := by
  unfold sourceColumnIds
  change
    ((nativeComplete setup).canonicalStep.program.physical.receipts.flatMap
        InstructionReceipt.columnIds) =
      (CurrentM4PhysicalStability.encoding setup).columnIds
  rw [nativeComplete_eq_current setup]
  unfold CurrentM4PhysicalStability.encoding Encoding.columnIds
    Encoding.columns CurrentM4PhysicalStability.certificate
  simp only [SourceAlignment.AlignedReceiptProgram.toEncoding_receipts,
    List.map_flatMap, InstructionReceipt.columnIds]
  rfl

noncomputable def setupTemplate (template : Template) :
    SetupTemplate dimensions commitmentRows where
  verifierKey := template.verifierKey
  domainCovers := domainCovers
  rowNonempty := rowNonempty

@[simp] theorem setupTemplate_withSystem
    (template : Template)
    (system : Structure dimensions.shape) :
    (setupTemplate template).withSystem system =
      template.withSystem system := by
  rfl

private theorem nativeReceiptRows_eq_of_constraintPolynomial_eq
    (template : SetupTemplate dimensions commitmentRows)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (ConcreteNifsNativeCcsStep.nativeReceipt
      (deployment (template.withSystem left)).application.phase4
      (nifsCertificate (template.withSystem left))
      (deployment (template.withSystem left)).step
      (deployment (template.withSystem left)).defaultRunningAdmissible).rows =
    (ConcreteNifsNativeCcsStep.nativeReceipt
      (deployment (template.withSystem right)).application.phase4
      (nifsCertificate (template.withSystem right))
      (deployment (template.withSystem right)).step
      (deployment (template.withSystem right)).defaultRunningAdmissible).rows := by
  unfold ConcreteNifsNativeCcsStep.nativeReceipt
  rw [
    ConcreteNifsNativeCcsProgram.selectedReceipt_rows,
    ConcreteNifsNativeCcsProgram.selectedReceipt_rows]
  unfold ConcreteNifsRawProgram.rows
  change
    select (invokePlan (template.withSystem left)).frame.active
        (DirectCalls.ownRows
          (invokePlan (template.withSystem left)).frame.owner
          (ConcreteNifsRawProgram.rawRows
            (application (template.withSystem left))
            (operational (template.withSystem left))
            (invokePlan (template.withSystem left)).frame)) =
      select (invokePlan (template.withSystem right)).frame.active
        (DirectCalls.ownRows
          (invokePlan (template.withSystem right)).frame.owner
          (ConcreteNifsRawProgram.rawRows
            (application (template.withSystem right))
            (operational (template.withSystem right))
            (invokePlan (template.withSystem right)).frame))
  rw [
    active_eq_of_constraintPolynomial_eq
      template left right same,
    owner_eq_of_constraintPolynomial_eq
      template left right same,
    rawRows_eq_of_constraintPolynomial_eq
      template left right same]

theorem rows_eq_of_constraintPolynomial_eq
    (template : Template)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (NativeFixedPointCost.nativeProgram
      (template.withSystem left)).rows =
    (NativeFixedPointCost.nativeProgram
      (template.withSystem right)).rows := by
  rw [← setupTemplate_withSystem template left,
    ← setupTemplate_withSystem template right]
  rw [
    rows_split ((setupTemplate template).withSystem left),
    rows_split ((setupTemplate template).withSystem right)]
  unfold splitRows
  rw [
    beforeTargetRows ((setupTemplate template).withSystem left),
    beforeTargetRows ((setupTemplate template).withSystem right),
    afterTargetRows ((setupTemplate template).withSystem left),
    afterTargetRows ((setupTemplate template).withSystem right),
    prefixRows_eq_of_constraintPolynomial_eq
      (setupTemplate template) left right same,
    applicationRows_eq_of_constraintPolynomial_eq
      (setupTemplate template) left right same,
    preNifsRows_eq_of_constraintPolynomial_eq
      (setupTemplate template) left right same,
    nativeReceiptRows_eq_of_constraintPolynomial_eq
      (setupTemplate template) left right same,
    postNifsRows_eq_of_constraintPolynomial_eq
      (setupTemplate template) left right same]

theorem residualIds_eq_of_constraintPolynomial_eq
    (template : Template)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    residualIds (template.withSystem left) =
      residualIds (template.withSystem right) := by
  unfold residualIds
  change
    ConcreteNifsActivatedProgram.residuals
        (application ((setupTemplate template).withSystem left))
        (operational ((setupTemplate template).withSystem left))
        (invokePlan ((setupTemplate template).withSystem left)).frame =
      ConcreteNifsActivatedProgram.residuals
        (application ((setupTemplate template).withSystem right))
        (operational ((setupTemplate template).withSystem right))
        (invokePlan ((setupTemplate template).withSystem right)).frame
  exact CurrentM4RawRows.residuals_eq_of_constraintPolynomial_eq
    (setupTemplate template) left right same

theorem columnIds_eq_of_constraintPolynomial_eq
    (template : Template)
    (left right : Structure dimensions.shape)
    (same :
      left.constraintPolynomial = right.constraintPolynomial) :
    (NativeFixedPointCost.nativeProgram
      (template.withSystem left)).columnIds =
    (NativeFixedPointCost.nativeProgram
      (template.withSystem right)).columnIds := by
  rw [columnIds_eq_source_filter, columnIds_eq_source_filter]
  have currentColumns :
      (CurrentM4PhysicalStability.encoding
          (template.withSystem left)).columnIds =
        (CurrentM4PhysicalStability.encoding
          (template.withSystem right)).columnIds := by
    simpa only [setupTemplate_withSystem] using
      (CurrentM4PhysicalStability.columnIds_eq_of_constraintPolynomial_eq
        (setupTemplate template) left right same)
  rw [sourceColumnIds_eq_currentEncoding,
    sourceColumnIds_eq_currentEncoding, currentColumns,
    residualIds_eq_of_constraintPolynomial_eq template left right same]

theorem finalRows_eq_source
    (template : Template) :
    (NativeFixedPointCost.nativeProgram
      (finalSetup template)).rows =
      (NativeFixedPointSource.program template).rows := by
  change
    (NativeFixedPointCost.nativeProgram
      (template.withSystem (finalSystem template))).rows =
      (NativeFixedPointCost.nativeProgram
        (template.withSystem seedSystem)).rows
  exact rows_eq_of_constraintPolynomial_eq
    template (finalSystem template) seedSystem rfl

theorem finalColumnIds_eq_source
    (template : Template) :
    (NativeFixedPointCost.nativeProgram
      (finalSetup template)).columnIds =
      (NativeFixedPointSource.program template).columnIds := by
  change
    (NativeFixedPointCost.nativeProgram
      (template.withSystem (finalSystem template))).columnIds =
      (NativeFixedPointCost.nativeProgram
        (template.withSystem seedSystem)).columnIds
  exact columnIds_eq_of_constraintPolynomial_eq
    template (finalSystem template) seedSystem rfl

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Applications.WasmBenchmark42x6.NativeFixedPointStability
