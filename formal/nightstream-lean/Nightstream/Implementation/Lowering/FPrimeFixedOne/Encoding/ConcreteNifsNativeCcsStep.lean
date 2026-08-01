import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCompleteApplication
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsProgram
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepSoundness

/-!
Contract: replace the one recursive selected-NIFS R1CS activation receipt in
the complete fixed-one Step program with its native CCS receipt.

Assurance tier: model-level.

Owns:
- exact identification of the recursive NIFS receipt by its source owner;
- replacement of that receipt and no other receipt;
- selector one on all ordinary Step rows and recursive-arm activation on the
  intrinsic NIFS rows;
- receipt order, global row and column identity uniqueness, conservation, and
  receipt-derived cost for the native CCS Step program.

Does not own: a proof-free manifest, Rust matrix emission, or a deployment
application.

Emits constraints: the complete Step program with one native CCS row per
intrinsic NIFS row and no R1CS activation residual rows or columns.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
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

def certificate
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    CompleteApplicationCertification Selected :=
  ConcreteNifsCompleteApplication.complete
    application nifs step defaultAdmissible

def recipes
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :=
  (certificate application nifs step defaultAdmissible).allRecipes

def invokePlan
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :=
  CanonicalStepPlan.recursiveNifsInvokePlan Selected
    (certificate application nifs step defaultAdmissible).baseProfile
    (recipes application nifs step defaultAdmissible)

def targetOwner : PhysicalOwner :=
  .typed (.instruction SourceOwners.stepRecursiveNifsPath)

def targetReceipt
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
  (defaultAdmissible : DefaultAdmissibleFor application) :
    InstructionReceipt :=
  (CanonicalStepPlan.recursiveNifsPlan.{0} Selected
    (certificate application nifs step defaultAdmissible).baseProfile
    (recipes application nifs step defaultAdmissible)).receipt

def nativeReceipt
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    SelectedReceipt :=
  ConcreteNifsNativeCcsProgram.selectedReceipt
    application.profile nifs.operational
    (invokePlan application nifs step defaultAdmissible).frame

def replace
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipt : InstructionReceipt) :
    SelectedReceipt :=
  if receipt.owner = targetOwner then
    nativeReceipt application nifs step defaultAdmissible
  else
    { receipt := receipt, selector := oneColumn }

def sourceReceipts
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    List InstructionReceipt :=
  (certificate application nifs step defaultAdmissible
    ).canonicalStep.program.physical.receipts

@[simp] theorem sourceReceipts_encoding
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    sourceReceipts application nifs step defaultAdmissible =
      (CanonicalStepSoundness.encoding Selected
        (certificate application nifs step defaultAdmissible).baseProfile
        (recipes application nifs step defaultAdmissible)
        defaultAdmissible).receipts :=
  rfl

def selectedReceipts
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    List SelectedReceipt :=
  (sourceReceipts application nifs step defaultAdmissible).map
    (replace application nifs step defaultAdmissible)

def program
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    NativeCcsProgram.Program where
  one := oneColumn
  receipts := selectedReceipts application nifs step defaultAdmissible

@[simp] theorem nativeReceipt_owner
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (nativeReceipt application nifs step defaultAdmissible).receipt.owner =
      targetOwner :=
  rfl

@[simp] theorem targetReceipt_owner
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (targetReceipt application nifs step defaultAdmissible).owner =
      targetOwner :=
  rfl

/-- The replaced receipt is exactly the legacy activated selected-verifier
row stream. -/
@[simp] theorem targetReceipt_rows
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (targetReceipt application nifs step defaultAdmissible).rows =
      ConcreteNifsActivatedProgram.rows
        application.profile nifs.operational
        (invokePlan application nifs step defaultAdmissible).frame :=
  rfl

/-- The legacy target allocation is exactly the native allocation followed by
the removed activation-residual suffix. This is an ordered identity, not a
cardinality comparison. -/
theorem targetReceipt_columnIds_eq_native_append_residuals
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (targetReceipt application nifs step defaultAdmissible).columnIds =
      (nativeReceipt application nifs step defaultAdmissible
        ).receipt.columnIds ++
        ConcreteNifsActivatedProgram.residuals
          application.profile nifs.operational
          (invokePlan application nifs step defaultAdmissible).frame := by
  simp only [targetReceipt, nativeReceipt, invokePlan,
    CanonicalStepPlan.recursiveNifsPlan, PrimitivePlan.receipt,
    InvokePlan.receipt, InstructionReceipt.ofCall,
    ConcreteNifsNativeCcsProgram.selectedReceipt,
    ConcreteNifsNativeCcsProgram.sourceReceipt,
    ConcreteNifsNativeCcsProgram.allocations,
    ConcreteNifsNativeCcsProgram.temporaryAllocations,
    InstructionReceipt.columnIds, List.map_append, List.map_take,
    CallFrame.allocations, SchemaBundles.ids, LayoutBundles.ids,
    ConcreteNifsActivatedProgram.residuals]
  rw [List.append_assoc, List.take_append_drop]

/-- The exact owned-column suffix removed by native CCS. Unlike the residual
identifier list, this list retains each column's accounting class. -/
def removedAllocations
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    List OwnedColumn :=
  (invokePlan application nifs step defaultAdmissible).frame.temporaries.columns.drop
    (ConcreteNifsRawProgram.allocationWidth application.profile nifs.operational
      (invokePlan application nifs step defaultAdmissible).frame)

/-- The legacy target allocation is exactly the native owned allocation
followed by the removed activation-residual allocation. This preserves order,
column identity, and ownership class. -/
theorem targetReceipt_allocations_eq_native_append_removed
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (targetReceipt application nifs step defaultAdmissible).allocations =
      (nativeReceipt application nifs step defaultAdmissible
        ).receipt.allocations ++
        removedAllocations application nifs step defaultAdmissible := by
  simp only [targetReceipt, nativeReceipt, invokePlan,
    CanonicalStepPlan.recursiveNifsPlan, PrimitivePlan.receipt,
    InvokePlan.receipt, InstructionReceipt.ofCall,
    ConcreteNifsNativeCcsProgram.selectedReceipt,
    ConcreteNifsNativeCcsProgram.sourceReceipt,
    ConcreteNifsNativeCcsProgram.allocations,
    ConcreteNifsNativeCcsProgram.temporaryAllocations,
    removedAllocations, CallFrame.allocations]
  rw [List.append_assoc, List.take_append_drop]

/-- Every removed activation-residual allocation is auxiliary. Thus native
CCS removes no public or committed column. -/
theorem removedAllocation_auxiliary
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (column : OwnedColumn)
    (member :
      column ∈ removedAllocations
        application nifs step defaultAdmissible) :
    column.ownership = .auxiliaryColumn := by
  apply ConcreteNifsNativeCcsProgram.temporaryColumn_auxiliary
    application.profile nifs
    (invokePlan application nifs step defaultAdmissible).frame
  exact List.mem_of_mem_drop member

theorem targetReceipt_member
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    targetReceipt application nifs step defaultAdmissible ∈
      sourceReceipts application nifs step defaultAdmissible := by
  simp [targetReceipt, sourceReceipts, certificate,
    recipes,
    CompleteApplicationCertification.canonicalStep,
    CanonicalEncodingRealization.step, CanonicalStepPlan.aligned,
    CanonicalStepPlan.physical, CanonicalStepPlan.receipts,
    CanonicalStepPlan.bodyReceipts]

@[simp] theorem replace_target
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    replace application nifs step defaultAdmissible
        (targetReceipt application nifs step defaultAdmissible) =
      nativeReceipt application nifs step defaultAdmissible := by
  simp [replace]

theorem nativeReceipt_member
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    nativeReceipt application nifs step defaultAdmissible ∈
      selectedReceipts application nifs step defaultAdmissible := by
  rw [selectedReceipts]
  exact List.mem_map.mpr
    ⟨targetReceipt application nifs step defaultAdmissible,
      targetReceipt_member application nifs step defaultAdmissible,
      replace_target application nifs step defaultAdmissible⟩

theorem replace_other
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipt : InstructionReceipt)
    (other : receipt.owner ≠ targetOwner) :
    replace application nifs step defaultAdmissible receipt =
      { receipt := receipt, selector := oneColumn } := by
  simp [replace, other]

theorem ordinaryReceipt_member
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipt : InstructionReceipt)
    (member :
      receipt ∈ sourceReceipts application nifs step defaultAdmissible)
    (other : receipt.owner ≠ targetOwner) :
    ({ receipt := receipt, selector := oneColumn } : SelectedReceipt) ∈
      selectedReceipts application nifs step defaultAdmissible := by
  rw [selectedReceipts]
  exact List.mem_map.mpr
    ⟨receipt, member,
      replace_other application nifs step defaultAdmissible receipt other⟩

private theorem replaced_columns_of_no_target
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipts : List InstructionReceipt)
    (noTarget :
      ∀ receipt, receipt ∈ receipts →
        receipt.owner ≠ targetOwner)
    (column : ColumnId)
    (member :
      column ∈ receipts.flatMap InstructionReceipt.columnIds) :
    column ∈
      (receipts.map
        (replace application nifs step defaultAdmissible)).flatMap
          (fun receipt => receipt.receipt.columnIds) := by
  induction receipts with
  | nil =>
      simp at member
  | cons head tail inductionHypothesis =>
      have headOther := noTarget head List.mem_cons_self
      have tailOther :
          ∀ receipt, receipt ∈ tail →
            receipt.owner ≠ targetOwner := by
        intro receipt receiptMember
        exact noTarget receipt (List.mem_cons_of_mem head receiptMember)
      rcases List.mem_append.1 member with headMember | tailMember
      · apply List.mem_append_left
        simpa [replace_other application nifs step defaultAdmissible
          head headOther] using headMember
      · apply List.mem_append_right
        exact inductionHypothesis tailOther tailMember

private theorem source_column_survives_replacement
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipts : List InstructionReceipt)
    (targetMember :
      targetReceipt application nifs step defaultAdmissible ∈ receipts)
    (ownersNodup :
      (receipts.map fun receipt => receipt.owner).Nodup)
    (column : ColumnId)
    (member :
      column ∈ receipts.flatMap InstructionReceipt.columnIds)
    (notResidual :
      column ∉ ConcreteNifsActivatedProgram.residuals
        application.profile nifs.operational
        (invokePlan application nifs step defaultAdmissible).frame) :
    column ∈
      (receipts.map
        (replace application nifs step defaultAdmissible)).flatMap
          (fun receipt => receipt.receipt.columnIds) := by
  induction receipts with
  | nil =>
      simp at targetMember
  | cons head tail inductionHypothesis =>
      have ownerSplit :
          head.owner ∉ tail.map (fun receipt => receipt.owner) ∧
            (tail.map fun receipt => receipt.owner).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using ownersNodup
      rcases List.mem_cons.1 targetMember with headTarget | tailTarget
      · subst head
        have noTailTarget :
            ∀ receipt, receipt ∈ tail →
              receipt.owner ≠ targetOwner := by
          intro receipt receiptMember equal
          apply ownerSplit.1
          exact List.mem_map.mpr
            ⟨receipt, receiptMember, by
              rw [targetReceipt_owner]
              exact equal⟩
        rcases List.mem_append.1 member with targetColumn | tailColumn
        · rw [targetReceipt_columnIds_eq_native_append_residuals] at targetColumn
          rcases List.mem_append.1 targetColumn with
            nativeColumn | residualColumn
          · simp only [List.map_cons, List.flatMap_cons]
            apply List.mem_append_left
            simpa [replace_target application nifs step defaultAdmissible]
              using nativeColumn
          · exact False.elim (notResidual residualColumn)
        · simp only [List.map_cons, List.flatMap_cons]
          apply List.mem_append_right
          exact replaced_columns_of_no_target
            application nifs step defaultAdmissible tail noTailTarget
            column tailColumn
      · have headOther : head.owner ≠ targetOwner := by
          intro equal
          apply ownerSplit.1
          exact List.mem_map.mpr
            ⟨targetReceipt application nifs step defaultAdmissible,
              tailTarget, by
                rw [targetReceipt_owner]
                exact equal.symm⟩
        rcases List.mem_append.1 member with headColumn | tailColumn
        · simp only [List.map_cons, List.flatMap_cons]
          apply List.mem_append_left
          simpa [replace_other application nifs step defaultAdmissible
            head headOther] using headColumn
        · simp only [List.map_cons, List.flatMap_cons]
          apply List.mem_append_right
          exact inductionHypothesis tailTarget ownerSplit.2
            tailColumn

/-- A source allocation survives the replacement exactly when it is not one
of the removed residual columns. The result preserves the flattened receipt
order. -/
theorem source_column_mem_selected_of_not_residual
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (column : ColumnId)
    (member :
      column ∈
        (sourceReceipts application nifs step defaultAdmissible).flatMap
          InstructionReceipt.columnIds)
    (notResidual :
      column ∉ ConcreteNifsActivatedProgram.residuals
        application.profile nifs.operational
        (invokePlan application nifs step defaultAdmissible).frame) :
    column ∈
      (program application nifs step defaultAdmissible).columnIds := by
  rw [NativeCcsProgram.Program.columnIds_conserved]
  exact source_column_survives_replacement
    application nifs step defaultAdmissible
    (sourceReceipts application nifs step defaultAdmissible)
    (targetReceipt_member application nifs step defaultAdmissible)
    (certificate application nifs step defaultAdmissible
      ).canonicalStep.program.physical.ownersNodup
    column member notResidual

private theorem replace_allocations_eq_of_no_target
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipts : List InstructionReceipt)
    (noTarget :
      ∀ receipt, receipt ∈ receipts →
        receipt.owner ≠ targetOwner) :
    (receipts.map
        (replace application nifs step defaultAdmissible)).flatMap
          (fun receipt => receipt.receipt.allocations) =
      receipts.flatMap (fun receipt => receipt.allocations) := by
  induction receipts with
  | nil =>
      rfl
  | cons head tail inductionHypothesis =>
      have headOther := noTarget head List.mem_cons_self
      have tailOther :
          ∀ receipt, receipt ∈ tail →
            receipt.owner ≠ targetOwner := by
        intro receipt member
        exact noTarget receipt (List.mem_cons_of_mem head member)
      simp only [List.map_cons, List.flatMap_cons,
        replace_other application nifs step defaultAdmissible head headOther]
      rw [inductionHypothesis tailOther]

private theorem removedAllocations_filter_eq_nil
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (ownership : Ownership)
    (notAuxiliary : ownership ≠ .auxiliaryColumn) :
    (removedAllocations application nifs step defaultAdmissible).filter
        (fun column => decide (column.ownership = ownership)) =
      [] := by
  apply List.filter_eq_nil_iff.mpr
  intro column member
  have auxiliary :=
    removedAllocation_auxiliary application nifs step defaultAdmissible
      column member
  have auxiliaryNe : Ownership.auxiliaryColumn ≠ ownership := by
    intro equal
    exact notAuxiliary equal.symm
  simp [auxiliary, auxiliaryNe]

private theorem targetReceipt_filter_eq_nativeReceipt_filter
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (ownership : Ownership)
    (notAuxiliary : ownership ≠ .auxiliaryColumn) :
    (targetReceipt application nifs step defaultAdmissible
        ).allocations.filter
          (fun column => decide (column.ownership = ownership)) =
      (nativeReceipt application nifs step defaultAdmissible
        ).receipt.allocations.filter
          (fun column => decide (column.ownership = ownership)) := by
  rw [targetReceipt_allocations_eq_native_append_removed,
    List.filter_append,
    removedAllocations_filter_eq_nil
      application nifs step defaultAdmissible ownership notAuxiliary,
    List.append_nil]

private theorem source_filter_eq_selected_filter
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipts : List InstructionReceipt)
    (targetMember :
      targetReceipt application nifs step defaultAdmissible ∈ receipts)
    (ownersNodup :
      (receipts.map fun receipt => receipt.owner).Nodup)
    (ownership : Ownership)
    (notAuxiliary : ownership ≠ .auxiliaryColumn) :
    (receipts.flatMap
        (fun receipt => receipt.allocations)).filter
          (fun column => decide (column.ownership = ownership)) =
      ((receipts.map
        (replace application nifs step defaultAdmissible)).flatMap
          (fun receipt => receipt.receipt.allocations)).filter
            (fun column => decide (column.ownership = ownership)) := by
  induction receipts with
  | nil =>
      simp at targetMember
  | cons head tail inductionHypothesis =>
      have ownerSplit :
          head.owner ∉ tail.map (fun receipt => receipt.owner) ∧
            (tail.map fun receipt => receipt.owner).Nodup := by
        simpa only [List.map_cons, List.nodup_cons] using ownersNodup
      rcases List.mem_cons.1 targetMember with headTarget | tailTarget
      · subst head
        have noTailTarget :
            ∀ receipt, receipt ∈ tail →
              receipt.owner ≠ targetOwner := by
          intro receipt receiptMember equal
          apply ownerSplit.1
          exact List.mem_map.mpr
            ⟨receipt, receiptMember, by
              rw [targetReceipt_owner]
              exact equal⟩
        simp only [List.flatMap_cons, List.map_cons,
          replace_target, List.filter_append]
        rw [targetReceipt_filter_eq_nativeReceipt_filter
          application nifs step defaultAdmissible ownership notAuxiliary,
          replace_allocations_eq_of_no_target
            application nifs step defaultAdmissible tail noTailTarget]
      · have headOther : head.owner ≠ targetOwner := by
          intro equal
          apply ownerSplit.1
          exact List.mem_map.mpr
            ⟨targetReceipt application nifs step defaultAdmissible,
              tailTarget, by
                rw [targetReceipt_owner]
                exact equal.symm⟩
        simp only [List.flatMap_cons, List.map_cons,
          replace_other application nifs step defaultAdmissible
            head headOther,
          List.filter_append]
        rw [inductionHypothesis tailTarget ownerSplit.2]

/-- Native CCS preserves the exact ordered committed-column allocation from
the selected canonical Step. The theorem compares identities and ownership,
not only the committed-column count. -/
theorem committedAllocations_preserved
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (program application nifs step defaultAdmissible).allocations.filter
        (fun column =>
          decide (column.ownership = Ownership.committedColumn)) =
      ((sourceReceipts application nifs step defaultAdmissible).flatMap
        (fun receipt => receipt.allocations)).filter
          (fun column =>
            decide (column.ownership = Ownership.committedColumn)) := by
  change
    ((selectedReceipts application nifs step defaultAdmissible).flatMap
      (fun receipt => receipt.receipt.allocations)).filter
        (fun column =>
          decide (column.ownership = Ownership.committedColumn)) =
      _
  exact
    (source_filter_eq_selected_filter
      application nifs step defaultAdmissible
      (sourceReceipts application nifs step defaultAdmissible)
      (targetReceipt_member application nifs step defaultAdmissible)
      (certificate application nifs step defaultAdmissible
        ).canonicalStep.program.physical.ownersNodup
      .committedColumn (by decide)).symm

/-- Native CCS preserves the exact ordered public-column allocation from the
selected canonical Step. The theorem includes the constant-one and result
columns and makes no claim that they form a 270-column public prefix. -/
theorem publicAllocations_preserved
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (program application nifs step defaultAdmissible).allocations.filter
        (fun column =>
          decide (column.ownership = Ownership.publicColumn)) =
      ((sourceReceipts application nifs step defaultAdmissible).flatMap
        (fun receipt => receipt.allocations)).filter
          (fun column =>
            decide (column.ownership = Ownership.publicColumn)) := by
  change
    ((selectedReceipts application nifs step defaultAdmissible).flatMap
      (fun receipt => receipt.receipt.allocations)).filter
        (fun column =>
          decide (column.ownership = Ownership.publicColumn)) =
      _
  exact
    (source_filter_eq_selected_filter
      application nifs step defaultAdmissible
      (sourceReceipts application nifs step defaultAdmissible)
      (targetReceipt_member application nifs step defaultAdmissible)
      (certificate application nifs step defaultAdmissible
        ).canonicalStep.program.physical.ownersNodup
      .publicColumn (by decide)).symm

/-- Every removed activation residual belongs to the recursive NIFS
instruction. This structural owner fact lets later receipts prove that they
cannot read the removed suffix. -/
theorem residual_owner
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (column : ColumnId)
    (member :
      column ∈ ConcreteNifsActivatedProgram.residuals
        application.profile nifs.operational
        (invokePlan application nifs step defaultAdmissible).frame) :
    column.owner = targetOwner := by
  change
    column.owner =
      .typed (.instruction SourceOwners.stepRecursiveNifsPath)
  apply CanonicalPrimitivePlan.temporary_id_owner
    SourceOwners.stepRecursiveNifsPath
    ((signature Selected).callOutputs Call.nifsVerify)
    ((signature Selected).callFootprint Call.nifsVerify).temporaries
  rw [← (invokePlan application nifs step defaultAdmissible).temporariesExact]
  exact List.mem_of_mem_drop member

@[simp] theorem replace_owner
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (receipt : InstructionReceipt) :
    (replace application nifs step defaultAdmissible receipt).receipt.owner =
      receipt.owner := by
  unfold replace
  split
  · rename_i equal
    rw [nativeReceipt_owner, equal]
  · rfl

theorem selected_owner_map
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (selectedReceipts application nifs step defaultAdmissible).map
        (fun receipt => receipt.receipt.owner) =
      (sourceReceipts application nifs step defaultAdmissible).map
        (fun receipt => receipt.owner) := by
  simp [selectedReceipts, Function.comp_def]

theorem owners_nodup
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    ((selectedReceipts application nifs step defaultAdmissible).map
      fun receipt => receipt.receipt.owner).Nodup := by
  rw [selected_owner_map]
  exact
    (certificate application nifs step defaultAdmissible
      ).canonicalStep.program.physical.ownersNodup

theorem local_column_ids_nodup
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    ∀ receipt,
      receipt ∈ selectedReceipts application nifs step defaultAdmissible →
        receipt.receipt.columnIds.Nodup := by
  intro receipt member
  rcases List.mem_map.1 member with ⟨source, sourceMember, rfl⟩
  by_cases selected : source.owner = targetOwner
  · simp only [replace, selected, if_pos]
    exact
      ConcreteNifsNativeCcsProgram.sourceReceipt_columnIds_nodup
        application.profile nifs.operational
        (invokePlan application nifs step defaultAdmissible).frame
  · simp only [replace, selected, if_neg]
    exact
      (certificate application nifs step defaultAdmissible
        ).canonicalStep.program.physical.localColumnIdsNodup
          source sourceMember

theorem local_row_ids_nodup
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    ∀ receipt,
      receipt ∈ selectedReceipts application nifs step defaultAdmissible →
        receipt.receipt.rowIds.Nodup := by
  intro receipt member
  rcases List.mem_map.1 member with ⟨source, sourceMember, rfl⟩
  by_cases selected : source.owner = targetOwner
  · simp only [replace, selected, if_pos]
    exact
      ConcreteNifsNativeCcsProgram.sourceReceipt_rowIds_nodup
        application.profile nifs.operational
        (invokePlan application nifs step defaultAdmissible).frame
  · simp only [replace, selected, if_neg]
    exact
      (certificate application nifs step defaultAdmissible
        ).canonicalStep.program.physical.localRowIdsNodup
          source sourceMember

theorem column_ids_nodup
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (program application nifs step defaultAdmissible).columnIds.Nodup :=
  NativeCcsProgram.Program.columnIds_nodup
    (program application nifs step defaultAdmissible)
    (owners_nodup application nifs step defaultAdmissible)
    (local_column_ids_nodup application nifs step defaultAdmissible)

theorem row_ids_nodup
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (program application nifs step defaultAdmissible).rowIds.Nodup :=
  NativeCcsProgram.Program.rowIds_nodup
    (program application nifs step defaultAdmissible)
    (owners_nodup application nifs step defaultAdmissible)
    (local_row_ids_nodup application nifs step defaultAdmissible)

/-- The native selected program supplies exactly the row evidence consumed by
the common Step semantic proof. -/
def semanticEvidence
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (assignment : ColumnId → Field)
    (satisfied :
      (program application nifs step defaultAdmissible).Satisfies
        assignment) :
    CanonicalStepSoundness.Evidence Selected
      (certificate application nifs step defaultAdmissible).baseProfile
      (recipes application nifs step defaultAdmissible)
      defaultAdmissible assignment where
  constantOne := satisfied.1
  ordinaryRows := by
    intro receipt member other
    have sourceMember :
        receipt ∈ sourceReceipts application nifs step defaultAdmissible := by
      simpa [sourceReceipts, certificate,
        CompleteApplicationCertification.canonicalStep,
        CanonicalEncodingRealization.step, CanonicalStepPlan.aligned,
        CanonicalStepPlan.physical] using member
    have otherTarget : receipt.owner ≠ targetOwner := by
      simpa [targetOwner, CanonicalStepSoundness.recursiveNifsOwner] using
        other
    let selected : SelectedReceipt :=
      { receipt := receipt, selector := oneColumn }
    have selectedMember :
        selected ∈
          (program application nifs step defaultAdmissible).receipts := by
      change selected ∈
        selectedReceipts application nifs step defaultAdmissible
      exact
        ordinaryReceipt_member application nifs step defaultAdmissible
          receipt sourceMember otherTarget
    exact
      NativeCcsProgram.Program.source_satisfies_of_selector_one
        (program application nifs step defaultAdmissible)
        assignment satisfied selected selectedMember
        (by simpa [selected] using satisfied.1)
  recursiveNifs := by
    intro inputs constantOne activeOne decoded
    cases inputs with
    | cons running inputs =>
        cases inputs with
        | cons fresh inputs =>
            cases inputs with
            | cons proof inputs =>
                cases inputs
                have selectedSatisfied :
                    NativeCcsSelector.Satisfies
                      (nativeReceipt application nifs step
                        defaultAdmissible).rows assignment := by
                  exact
                    NativeCcsProgram.Program.receipt_satisfies
                      (program application nifs step defaultAdmissible)
                      assignment satisfied
                      (nativeReceipt application nifs step defaultAdmissible)
                      (by
                        change
                          nativeReceipt application nifs step
                              defaultAdmissible ∈
                            selectedReceipts application nifs step
                              defaultAdmissible
                        exact
                          nativeReceipt_member application nifs step
                            defaultAdmissible)
                exact
                  ConcreteNifsNativeCcsProgram.active_soundness
                    application.profile nifs
                    (invokePlan application nifs step defaultAdmissible).frame
                    assignment running fresh proof constantOne activeOne
                    decoded selectedSatisfied

/-- Native CCS Step satisfaction reaches the frozen Step relation. -/
theorem sound
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (assignment : ColumnId → Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        AppState Witness
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows))
    (satisfied :
      (program application nifs step defaultAdmissible).Satisfies assignment)
    (inputDecoded :
      Columns.Decodes
        ((certificate application nifs step defaultAdmissible
          ).baseProfile.family Selected)
        (CanonicalContexts.Step.input Selected) assignment
        (stepInputValues Selected input)) :
    ∃ output :
        Nightstream.HyperNova.Construction2.Paper.Output
          Digest AppState
          (SelectedRunning shape publicRingColumns publicFits verifierRows) 1,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        Selected input output :=
  CanonicalStepSoundness.soundFromEvidence Selected
    (certificate application nifs step defaultAdmissible).baseProfile
    (recipes application nifs step defaultAdmissible)
    defaultAdmissible
    (certificate application nifs step defaultAdmissible
      ).directProfile.fieldLaws
    assignment input
    (semanticEvidence application nifs step defaultAdmissible
      assignment satisfied)
    inputDecoded

/-- Native CCS Step satisfaction also binds the accepted value to the exact
result columns of the selected encoding. -/
theorem soundAligned
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (assignment : ColumnId → Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        AppState Witness
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows))
    (satisfied :
      (program application nifs step defaultAdmissible).Satisfies assignment)
    (inputDecoded :
      Columns.Decodes
        ((certificate application nifs step defaultAdmissible
          ).baseProfile.family Selected)
        (CanonicalContexts.Step.input Selected) assignment
        (stepInputValues Selected input)) :
    ∃ output :
        Nightstream.HyperNova.Construction2.Paper.Output
          Digest AppState
          (SelectedRunning shape publicRingColumns publicFits verifierRows) 1,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
          Selected input output ∧
        Columns.Decodes
          ((certificate application nifs step defaultAdmissible
            ).baseProfile.family Selected)
          (CanonicalContexts.Step.result Selected) assignment
          (stepResultValues Selected output) :=
  CanonicalStepSoundness.soundAlignedFromEvidence Selected
    (certificate application nifs step defaultAdmissible).baseProfile
    (recipes application nifs step defaultAdmissible)
    defaultAdmissible
    (certificate application nifs step defaultAdmissible
      ).directProfile.fieldLaws
    assignment input
    (semanticEvidence application nifs step defaultAdmissible
      assignment satisfied)
    inputDecoded

theorem rows_conserved
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (program application nifs step defaultAdmissible).rows =
      (selectedReceipts application nifs step defaultAdmissible).flatMap
        SelectedReceipt.rows :=
  rfl

theorem allocations_conserved
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (program application nifs step defaultAdmissible).allocations =
      (selectedReceipts application nifs step defaultAdmissible).flatMap
        SelectedReceipt.allocations :=
  rfl

theorem rows_length_eq_cost
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application) :
    (program application nifs step defaultAdmissible).rows.length =
      (program application nifs step defaultAdmissible).cost.recurringRows :=
  NativeCcsProgram.Program.rows_length
    (program application nifs step defaultAdmissible)

end CompleteStep

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep
