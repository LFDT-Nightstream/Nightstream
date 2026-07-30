import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalProofRecovery
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentDeployment

/-!
Contract: refine the base arm of the current Lean-emitted Step program
directly from one finite physical assignment.

Assurance tier: model-level.

Owns: recovery of every base-active semantic input, exact execution of the
common prefix, base assertion, default-running join, continuation hash, and
the frozen base Step relation.

Does not own: recursive NIFS semantics, inactive proof coordinates,
deployment application selection, terminal semantics, Rust equality, or a
security reduction.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentStepBasePhysicalRefinement

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalInputRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev ReceiptSatisfies :=
  Nightstream.Implementation.Lowering.Goldilocks.Satisfies

/-- Every value of a singleton heterogeneous schema is its head followed by
the unique empty tail. -/
private theorem hvec_singleton_eta
    {α : Type}
    {value : α → Type}
    {kind : α}
    (values : HVec value [kind]) :
    values = .cons (HVec.head values) .nil := by
  cases values with
  | cons head tail =>
      cases tail
      rfl

/-- Project one body receipt from exact whole-program physical
satisfaction. -/
private theorem bodyReceiptSatisfies
    (parameters : Parameters)
    (profile : Profile parameters)
    (recipes :
      CallRecipes (signature parameters) (profile.family parameters))
    (defaultAdmissible :
      ((profile.family parameters).codecFor
        (.data .running)).Admissible (defaultRunning parameters))
    (assignment : ColumnId → Field)
    (physical :
      (CanonicalStepSoundness.encoding
        parameters profile recipes defaultAdmissible).PhysicalSatisfies
          assignment)
    (receipt : InstructionReceipt)
    (member :
      receipt ∈
        CanonicalStepPlan.bodyReceipts
          parameters profile recipes defaultAdmissible) :
    ReceiptSatisfies receipt.rows assignment := by
  apply
    (CanonicalStepSoundness.encoding
      parameters profile recipes defaultAdmissible).receiptSatisfies
        assignment physical receipt
  simpa [CanonicalStepSoundness.encoding, CanonicalStepPlan.physical] using
    (show
      receipt ∈
        CanonicalStepPlan.receipts
          parameters profile recipes defaultAdmissible by
      rw [CanonicalStepPlan.receipts]
      apply List.mem_cons_of_mem
      exact List.mem_append_right _ member)

section

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    TerminalRelations
      (Key dimensions TranscriptState verifierRows)
      (Running dimensions verifierRows)
      RunningWitness
      (Fresh dimensions verifierRows)
      FreshWitness 1)
variable
  (terminalChecks :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
      terminalRelations)
variable (widths : Widths) (footprints : Footprints)

local notation "Selected" =>
  ConcreteNifsPlain270Profile.selected dimensions
    (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
    defaultRunning machine terminalRelations terminalChecks widths footprints

/-- The current Step selector and activation rows choose exactly one branch
from the decoded iteration. No NIFS proof value is needed to select the
branch. -/
theorem deployment_step_branch_from_physical_rows
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment :
      Fin
        ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
              ).columnIds.length →
        F)
    (accepted :
      CurrentCompiler.Accepts
        (ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
        (CurrentDeployment.deployment_step_columns_ge_270
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment)
        assignment) :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    let stable :=
      StableRows.pulledAssignment
        (EncodingRows.columnIndex
          certificate.canonicalStep.program.toEncoding) assignment
    ∃ iteration : Nat,
      ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.iteration
            Selected)).Decodes
        (deployment.application.phase4.profile.family Selected)
        (.data .nat) stable iteration ∧
      ((iteration = 0 ∧
          stable
              (activationColumn SourceOwners.stepBranchPath true) = 1 ∧
          stable
              (activationColumn SourceOwners.stepBranchPath false) = 0) ∨
        (iteration ≠ 0 ∧
          stable
              (activationColumn SourceOwners.stepBranchPath true) = 0 ∧
          stable
              (activationColumn SourceOwners.stepBranchPath false) = 1)) := by
  let certificate :=
    ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  let stable :=
    StableRows.pulledAssignment
      (EncodingRows.columnIndex
        certificate.canonicalStep.program.toEncoding) assignment
  have physical :
      certificate.canonicalStep.program.toEncoding.PhysicalSatisfies stable :=
    (CurrentCompiler.accepts_iff_physicalSatisfies
      certificate.canonicalStep.program.toEncoding
      (CurrentDeployment.deployment_step_columns_ge_270
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment)
      assignment).mp accepted
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment stable
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.iteration
            Selected)
          Codec.boundedNatCodec_exactWidthRecoverable with
    ⟨iteration, iterationDecoded⟩
  change Nat at iteration
  let selectorPlan :=
    CanonicalStepConstructionPlans.selector
      Selected certificate.baseProfile certificate.allRecipes
  have selectorRows :
      ReceiptSatisfies selectorPlan.receipt.rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical
        selectorPlan.receipt (by
          dsimp [selectorPlan]
          rw [CanonicalStepPlan.bodyReceipts]
          exact List.mem_cons_of_mem _ List.mem_cons_self)
  have selectorOperandsDecoded :
      selectorPlan.frame.operands.Decodes
        (certificate.baseProfile.family Selected) stable
        (.cons iteration .nil) := by
    rw [CallFrame.operands, selectorPlan.contextExact]
    exact
      ⟨by
        simpa [CanonicalContexts.Step.afterStep,
          Columns.toSchemaBundles_get] using iterationDecoded,
        trivial⟩
  have selectorOne : stable selectorPlan.frame.one = 1 := by
    rw [selectorPlan.oneExact]
    exact physical.1
  have selectorActive : stable selectorPlan.frame.active = 1 := by
    rw [selectorPlan.activeExact]
    exact physical.1
  rcases
      selectorPlan.recipe.activeSoundness selectorPlan.frame stable
        (.cons iteration .nil)
        selectorOne selectorActive selectorOperandsDecoded
        (by
          simpa [InvokePlan.receipt, InstructionReceipt.ofCall] using
            selectorRows) with
    ⟨selectorOutputs, selectorEvaluated, selectorOutputDecoded⟩
  let selectorValue := HVec.head selectorOutputs
  have selectorOutputsExact :
      selectorOutputs = .cons selectorValue .nil :=
    hvec_singleton_eta selectorOutputs
  rw [selectorOutputsExact] at selectorEvaluated selectorOutputDecoded
  have selectorExact :
      selectorValue = decide (iteration = 0) := by
    simp only [signature, callEval] at selectorEvaluated
    have valuesEqual := Option.some.inj selectorEvaluated
    exact (congrArg HVec.head valuesEqual).symm
  rw [selectorPlan.outputsExact] at selectorOutputDecoded
  have selectorBundleDecoded :
      ((CanonicalContexts.Step.common Selected).toSchemaBundles.get
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iterationZero
            Selected)).Decodes
        (certificate.baseProfile.family Selected) .bit stable selectorValue := by
    simpa [CanonicalContexts.Step.common,
      Columns.toSchemaBundles_get] using selectorOutputDecoded.1
  have selectorIdsExact :=
    CanonicalPrimitivePlan.bitReferenceIdsExact
      certificate.baseProfile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iterationZero
        Selected)
      (CanonicalContexts.Step.common Selected)
      (CanonicalContexts.Step.commonWidths
        Selected certificate.baseProfile)
  have selectorDecoded :
      boolCodec.decode
          [stable
            (CanonicalContexts.Step.selector
              Selected certificate.baseProfile)] =
        some selectorValue := by
    unfold ColumnBundle.Decodes at selectorBundleDecoded
    change boolCodec.decode _ = some selectorValue at selectorBundleDecoded
    rw [ColumnBundle.values_eq_ids_map, selectorIdsExact] at selectorBundleDecoded
    simpa [CanonicalContexts.Step.selector] using selectorBundleDecoded
  let activation :=
    CanonicalBranchPlan.activationRecipe
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector Selected certificate.baseProfile)
  have trueActivationRows :
      ReceiptSatisfies
        (CanonicalBranchPlan.trueActivationReceipt
          SourceOwners.stepBranchPath oneColumn oneColumn
          (CanonicalContexts.Step.selector
            Selected certificate.baseProfile)).rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical _ (by
          rw [CanonicalStepPlan.bodyReceipts]
          exact
            List.mem_cons_of_mem _
              (List.mem_cons_of_mem _ List.mem_cons_self))
  have falseActivationRows :
      ReceiptSatisfies
        (CanonicalBranchPlan.falseActivationReceipt
          SourceOwners.stepBranchPath oneColumn oneColumn
          (CanonicalContexts.Step.selector
            Selected certificate.baseProfile)).rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical _ (by
          rw [CanonicalStepPlan.bodyReceipts]
          exact
            List.mem_cons_of_mem _
              (List.mem_cons_of_mem _
                (List.mem_cons_of_mem _ List.mem_cons_self)))
  have activationRows : ReceiptSatisfies activation.rows stable := by
    rw [← CanonicalBranchPlan.activation_rows_conserved]
    exact
      (satisfies_append_iff _ _ stable).2
        ⟨trueActivationRows, falseActivationRows⟩
  refine ⟨iteration, iterationDecoded, ?_⟩
  by_cases iterationZero : iteration = 0
  · have selectedTrue :
        boolCodec.decode [stable activation.selector] = some true := by
      simpa [activation, selectorExact, iterationZero] using selectorDecoded
    have selected :=
      activation.selected_true_sound stable physical.1
        selectedTrue activationRows
    exact Or.inl
      ⟨iterationZero,
        by
          simpa [activation] using selected.1.trans physical.1,
        by simpa [activation] using selected.2⟩
  · have selectedFalse :
        boolCodec.decode [stable activation.selector] = some false := by
      simpa [activation, selectorExact, iterationZero] using selectorDecoded
    have selected :=
      activation.selected_false_sound stable physical.1
        selectedFalse activationRows
    exact Or.inr
      ⟨iterationZero,
        by simpa [activation] using selected.1,
        by
          simpa [activation] using selected.2.trans physical.1⟩

/-- **Current base Step M4 bridge.** Exact current-program acceptance and
base activation construct a complete typed Step input and output satisfying
the frozen base relation. The inactive NIFS proof coordinates are not decoded
or used as authority. -/
theorem deployment_base_step_refines_from_physical_rows
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment :
      Fin
        ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
              ).columnIds.length →
        F)
    (accepted :
      CurrentCompiler.Accepts
        (ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
        (CurrentDeployment.deployment_step_columns_ge_270
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment)
        assignment)
    (baseActive :
      let certificate :=
        ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment
      let stable :=
        StableRows.pulledAssignment
          (EncodingRows.columnIndex
            certificate.canonicalStep.program.toEncoding) assignment
      stable (activationColumn SourceOwners.stepBranchPath true) = 1) :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    let stable :=
      StableRows.pulledAssignment
        (EncodingRows.columnIndex
          certificate.canonicalStep.program.toEncoding) assignment
    ∃ input :
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
          AppState Witness
          (Running dimensions verifierRows)
          (Fresh dimensions verifierRows)
          (Proof dimensions TranscriptState verifierRows),
      ∃ output :
          Output Digest AppState (Running dimensions verifierRows) 1,
        input.iteration = 0 ∧
          Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
            Selected input output ∧
          Columns.Decodes
            (certificate.baseProfile.family Selected)
            (CanonicalContexts.Step.result Selected) stable
            (stepResultValues Selected output) := by
  dsimp only at baseActive
  let certificate :=
    ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  let stable :=
    StableRows.pulledAssignment
      (EncodingRows.columnIndex
        certificate.canonicalStep.program.toEncoding) assignment
  have baseActiveStable :
      stable (activationColumn SourceOwners.stepBranchPath true) = 1 := by
    simpa [stable, certificate] using baseActive
  have physical :
      certificate.canonicalStep.program.toEncoding.PhysicalSatisfies stable :=
    (CurrentCompiler.accepts_iff_physicalSatisfies
      certificate.canonicalStep.program.toEncoding
      (CurrentDeployment.deployment_step_columns_ge_270
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment)
      assignment).mp accepted
  rcases
      stepBaseInputs_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment stable with
    ⟨iteration, z0, zi, witness, iterationDecoded, z0Decoded, ziDecoded,
      witnessDecoded⟩
  have runningRecoverable :
      ((deployment.application.phase4.profile.family Selected).codecFor
        (.data .running)).ExactWidthRecoverable := by
    change
      deployment.application.phase4.profile.codecs.running.ExactWidthRecoverable
    rw [deployment.application.runningCodec_exact]
    exact runningCodec_exactWidthRecoverable
      (ConcreteNifsPlain270Profile.Shape dimensions)
      publicRingColumns verifierRows
      (ConcreteNifsPlain270Profile.publicFits dimensions)
  have freshRecoverable :
      ((deployment.application.phase4.profile.family Selected).codecFor
        (.data .fresh)).ExactWidthRecoverable := by
    change
      deployment.application.phase4.profile.codecs.fresh.ExactWidthRecoverable
    rw [deployment.application.freshCodec_exact]
    exact freshCodec_exactWidthRecoverable
      (ConcreteNifsPlain270Profile.Shape dimensions)
      publicRingColumns verifierRows
      (ConcreteNifsPlain270Profile.publicFits dimensions)
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment stable
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.running
            Selected)
          runningRecoverable with
    ⟨running, runningDecoded⟩
  rcases
      stepInputRef_decode_exists
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment stable
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.fresh
            Selected)
          freshRecoverable with
    ⟨fresh, freshDecoded⟩
  let proof : Proof dimensions TranscriptState verifierRows :=
    Classical.choice
      (ConcreteNifsCanonicalProofRecovery.selectedProof_nonempty
        (ConcreteNifsPlain270Profile.Shape dimensions)
        setup.system.constraintPolynomial
        publicRingColumns verifierRows
        (ConcreteNifsPlain270Profile.publicFits dimensions))
  let input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        AppState Witness
        (Running dimensions verifierRows)
        (Fresh dimensions verifierRows)
        (Proof dimensions TranscriptState verifierRows) := {
    iteration := iteration
    z0 := z0
    zi := zi
    running := fun _ => running
    fresh := fresh
    witness := witness
    nifsProof := proof
  }
  let applyPlan :=
    CanonicalStepConstructionPlans.apply
      Selected certificate.baseProfile certificate.allRecipes
  have applyRows : ReceiptSatisfies applyPlan.receipt.rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical
        applyPlan.receipt (by
          dsimp [applyPlan]
          rw [CanonicalStepPlan.bodyReceipts]
          exact List.mem_cons_self)
  have applyOperandsDecoded :
      applyPlan.frame.operands.Decodes
        (certificate.baseProfile.family Selected) stable
        (.cons zi (.cons witness .nil)) := by
    rw [CallFrame.operands, applyPlan.contextExact]
    exact
      ⟨by
        simpa [Columns.toSchemaBundles_get] using ziDecoded,
        ⟨by
          simpa [Columns.toSchemaBundles_get] using witnessDecoded,
          trivial⟩⟩
  have applyOne : stable applyPlan.frame.one = 1 := by
    rw [applyPlan.oneExact]
    exact physical.1
  have applyActive : stable applyPlan.frame.active = 1 := by
    rw [applyPlan.activeExact]
    exact physical.1
  rcases
      applyPlan.recipe.activeSoundness applyPlan.frame stable
        (.cons zi (.cons witness .nil))
        applyOne applyActive applyOperandsDecoded
        (by
          simpa [InvokePlan.receipt, InstructionReceipt.ofCall] using
            applyRows) with
    ⟨applyOutputs, applyEvaluated, applyOutputDecoded⟩
  let zNext := HVec.head applyOutputs
  have applyOutputsExact : applyOutputs = .cons zNext .nil := by
    exact hvec_singleton_eta applyOutputs
  rw [applyOutputsExact] at applyEvaluated applyOutputDecoded
  have zNextExact :
      zNext =
        machine.step
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          zi witness := by
    simp only [signature, callEval] at applyEvaluated
    have valuesEqual := Option.some.inj applyEvaluated
    exact (congrArg HVec.head valuesEqual).symm
  rw [applyPlan.outputsExact] at applyOutputDecoded
  rw [zNextExact] at applyOutputDecoded
  have zNextDecoded := applyOutputDecoded.1
  let selectorPlan :=
    CanonicalStepConstructionPlans.selector
      Selected certificate.baseProfile certificate.allRecipes
  have selectorRows : ReceiptSatisfies selectorPlan.receipt.rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical
        selectorPlan.receipt (by
          dsimp [selectorPlan]
          rw [CanonicalStepPlan.bodyReceipts]
          exact List.mem_cons_of_mem _ List.mem_cons_self)
  have selectorOperandsDecoded :
      selectorPlan.frame.operands.Decodes
        (certificate.baseProfile.family Selected) stable
        (.cons iteration .nil) := by
    rw [CallFrame.operands, selectorPlan.contextExact]
    exact
      ⟨by
        simpa [CanonicalContexts.Step.afterStep,
          Columns.toSchemaBundles_get] using iterationDecoded,
        trivial⟩
  have selectorOne : stable selectorPlan.frame.one = 1 := by
    rw [selectorPlan.oneExact]
    exact physical.1
  have selectorActive : stable selectorPlan.frame.active = 1 := by
    rw [selectorPlan.activeExact]
    exact physical.1
  rcases
      selectorPlan.recipe.activeSoundness selectorPlan.frame stable
        (.cons iteration .nil)
        selectorOne selectorActive selectorOperandsDecoded
        (by
          simpa [InvokePlan.receipt, InstructionReceipt.ofCall] using
            selectorRows) with
    ⟨selectorOutputs, selectorEvaluated, selectorOutputDecoded⟩
  let selectorValue := HVec.head selectorOutputs
  have selectorOutputsExact :
      selectorOutputs = .cons selectorValue .nil := by
    exact hvec_singleton_eta selectorOutputs
  rw [selectorOutputsExact] at selectorEvaluated selectorOutputDecoded
  have selectorExact :
      selectorValue = decide (iteration = 0) := by
    simp only [signature, callEval] at selectorEvaluated
    have valuesEqual := Option.some.inj selectorEvaluated
    exact (congrArg HVec.head valuesEqual).symm
  rw [selectorPlan.outputsExact] at selectorOutputDecoded
  have selectorBundleDecoded :
      ((CanonicalContexts.Step.common Selected).toSchemaBundles.get
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iterationZero
            Selected)).Decodes
        (certificate.baseProfile.family Selected) .bit stable selectorValue := by
    simpa [CanonicalContexts.Step.common,
      Columns.toSchemaBundles_get] using selectorOutputDecoded.1
  have selectorIdsExact :=
    CanonicalPrimitivePlan.bitReferenceIdsExact
      certificate.baseProfile
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.iterationZero
        Selected)
      (CanonicalContexts.Step.common Selected)
      (CanonicalContexts.Step.commonWidths
        Selected certificate.baseProfile)
  have selectorDecoded :
      boolCodec.decode
          [stable
            (CanonicalContexts.Step.selector
              Selected certificate.baseProfile)] =
        some selectorValue := by
    unfold ColumnBundle.Decodes at selectorBundleDecoded
    change boolCodec.decode _ = some selectorValue at selectorBundleDecoded
    rw [ColumnBundle.values_eq_ids_map, selectorIdsExact] at selectorBundleDecoded
    simpa [CanonicalContexts.Step.selector] using selectorBundleDecoded
  let activation :=
    CanonicalBranchPlan.activationRecipe
      SourceOwners.stepBranchPath oneColumn oneColumn
      (CanonicalContexts.Step.selector Selected certificate.baseProfile)
  have trueActivationRows :
      ReceiptSatisfies
        (CanonicalBranchPlan.trueActivationReceipt
          SourceOwners.stepBranchPath oneColumn oneColumn
          (CanonicalContexts.Step.selector
            Selected certificate.baseProfile)).rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical _ (by
          rw [CanonicalStepPlan.bodyReceipts]
          exact
            List.mem_cons_of_mem _
              (List.mem_cons_of_mem _ List.mem_cons_self))
  have falseActivationRows :
      ReceiptSatisfies
        (CanonicalBranchPlan.falseActivationReceipt
          SourceOwners.stepBranchPath oneColumn oneColumn
          (CanonicalContexts.Step.selector
            Selected certificate.baseProfile)).rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical _ (by
          rw [CanonicalStepPlan.bodyReceipts]
          exact
            List.mem_cons_of_mem _
              (List.mem_cons_of_mem _
                (List.mem_cons_of_mem _ List.mem_cons_self)))
  have activationRows : ReceiptSatisfies activation.rows stable := by
    rw [← CanonicalBranchPlan.activation_rows_conserved]
    exact
      (satisfies_append_iff _ _ stable).2
        ⟨trueActivationRows, falseActivationRows⟩
  have iterationZero : iteration = 0 := by
    by_contra nonzero
    have selectedFalse :
        boolCodec.decode [stable activation.selector] = some false := by
      simpa [activation, selectorExact, nonzero] using selectorDecoded
    have selected :=
      activation.selected_false_sound stable physical.1
        selectedFalse activationRows
    have baseZero :
        stable (activationColumn SourceOwners.stepBranchPath true) = 0 := by
      simpa [activation] using selected.1
    rw [baseActiveStable] at baseZero
    exact (by decide : (1 : F) ≠ 0) baseZero
  let equalityPlan :=
    CanonicalStepConstructionPlans.baseEquality
      Selected certificate.baseProfile certificate.allRecipes
  have equalityRows : ReceiptSatisfies equalityPlan.receipt.rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical
        equalityPlan.receipt (by
          dsimp [equalityPlan]
          rw [CanonicalStepPlan.bodyReceipts]
          exact
            List.mem_cons_of_mem _
              (List.mem_cons_of_mem _
                (List.mem_cons_of_mem _
                  (List.mem_cons_of_mem _ List.mem_cons_self))))
  have equalityOperandsDecoded :
      equalityPlan.frame.operands.Decodes
        (certificate.baseProfile.family Selected) stable
        (.cons z0 (.cons zi .nil)) := by
    rw [CallFrame.operands, equalityPlan.contextExact]
    exact
      ⟨by
        simpa [CanonicalContexts.Step.common,
          CanonicalContexts.Step.afterStep,
          Columns.toSchemaBundles_get] using z0Decoded,
        ⟨by
          simpa [CanonicalContexts.Step.common,
            CanonicalContexts.Step.afterStep,
            Columns.toSchemaBundles_get] using ziDecoded,
          trivial⟩⟩
  have equalityOne : stable equalityPlan.frame.one = 1 := by
    rw [equalityPlan.oneExact]
    exact physical.1
  have equalityActive : stable equalityPlan.frame.active = 1 := by
    rw [equalityPlan.activeExact]
    exact baseActive
  rcases
      equalityPlan.recipe.activeSoundness equalityPlan.frame stable
        (.cons z0 (.cons zi .nil))
        equalityOne equalityActive equalityOperandsDecoded
        (by
          simpa [InvokePlan.receipt, InstructionReceipt.ofCall] using
            equalityRows) with
    ⟨equalityOutputs, equalityEvaluated, equalityOutputDecoded⟩
  let equalValue := HVec.head equalityOutputs
  have equalityOutputsExact :
      equalityOutputs = .cons equalValue .nil := by
    exact hvec_singleton_eta equalityOutputs
  rw [equalityOutputsExact] at equalityEvaluated equalityOutputDecoded
  have equalValueExact :
      equalValue = stateEqual Selected z0 zi := by
    simp only [signature, callEval] at equalityEvaluated
    have valuesEqual := Option.some.inj equalityEvaluated
    exact (congrArg HVec.head valuesEqual).symm
  rw [equalityPlan.outputsExact] at equalityOutputDecoded
  let assertionPlan :=
    CanonicalPrimitivePlan.assertion certificate.baseProfile
      (.here (Ports.auxiliaryBit Selected))
      SourceOwners.stepBaseAssertionPath
      (CanonicalContexts.Step.afterBaseEquality Selected)
      oneColumn (activationColumn SourceOwners.stepBranchPath true)
      (CanonicalContexts.Step.afterBaseEqualityWidths
        Selected certificate.baseProfile)
  have assertionRows :
      ReceiptSatisfies assertionPlan.receipt.rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical
        assertionPlan.receipt (by
          dsimp [assertionPlan]
          rw [CanonicalStepPlan.bodyReceipts]
          exact
            List.mem_cons_of_mem _
              (List.mem_cons_of_mem _
                (List.mem_cons_of_mem _
                  (List.mem_cons_of_mem _
                    (List.mem_cons_of_mem _ List.mem_cons_self)))))
  have equalityBundleDecoded :
      ((CanonicalContexts.Step.afterBaseEquality
          Selected).toSchemaBundles.get
            (.here (Ports.auxiliaryBit Selected))).Decodes
        (certificate.baseProfile.family Selected) .bit
        stable equalValue := by
    simpa [CanonicalContexts.Step.afterBaseEquality,
      Columns.toSchemaBundles_get] using equalityOutputDecoded.1
  have equalityConditionDecoded :
      boolCodec.decode [stable assertionPlan.recipe.condition] =
        some equalValue := by
    unfold ColumnBundle.Decodes at equalityBundleDecoded
    change boolCodec.decode _ = some equalValue at equalityBundleDecoded
    rw [ColumnBundle.values_eq_ids_map] at equalityBundleDecoded
    have mappedIds :=
      congrArg (List.map stable) assertionPlan.conditionIdsExact
    exact
      (congrArg boolCodec.decode mappedIds).symm.trans
        equalityBundleDecoded
  have assertionTrue :
      boolCodec.decode [stable assertionPlan.recipe.condition] = some true :=
    (assertionPlan.recipe.active_iff_decode_true
      certificate.directProfile.fieldLaws stable
      (by
        rw [assertionPlan.oneExact]
        exact physical.1)
      (by
        rw [assertionPlan.activeExact]
        exact baseActive)).mp
      (by
        simpa [AssertPlan.receipt, InstructionReceipt.ofAssertion] using
          assertionRows)
  have equalValueTrue : equalValue = true :=
    Codec.decoded_value_unique boolCodec
      equalityConditionDecoded assertionTrue
  have initialState : z0 = zi := by
    rw [equalValueExact] at equalValueTrue
    change decide (z0 = zi) = true at equalValueTrue
    exact of_decide_eq_true equalValueTrue
  let literalPlan :=
    CanonicalStepConstructionPlans.baseLiteral
      Selected certificate.baseProfile
      certificate.defaultRunningAdmissible
  have literalRows : ReceiptSatisfies literalPlan.receipt.rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical
        literalPlan.receipt (by
          dsimp [literalPlan]
          rw [CanonicalStepPlan.bodyReceipts]
          right
          right
          right
          right
          right
          right
          exact List.mem_cons_self)
  have literalDecoded :
      literalPlan.recipe.output.Decodes
        (certificate.baseProfile.family Selected) (.data .running)
        stable defaultRunning :=
    literalPlan.recipe.decode_of_satisfies stable
      (by
        rw [literalPlan.oneExact]
        exact physical.1)
      (by
        rw [literalPlan.valueExact]
        exact certificate.defaultRunningAdmissible)
      (by
        simpa [LiteralPlan.receipt, InstructionReceipt.ofLiteral] using
          literalRows)
  let mux :=
    CanonicalBranchPlan.onePortJoinRecipe
      SourceOwners.stepBranchPath
      (CanonicalContexts.Step.selector Selected certificate.baseProfile)
      (Ports.committedRunning Selected)
      (CanonicalContexts.Step.baseRunning Selected)
      (CanonicalContexts.Step.recursiveRunning Selected)
  have joinRows :
      ReceiptSatisfies mux.rows stable := by
    have rows :
        ReceiptSatisfies
          (CanonicalBranchPlan.onePortJoinReceipt
            SourceOwners.stepBranchPath
            (CanonicalContexts.Step.selector
              Selected certificate.baseProfile)
            (Ports.committedRunning Selected)
            (CanonicalContexts.Step.baseRunning Selected)
            (CanonicalContexts.Step.recursiveRunning Selected)).rows stable :=
      bodyReceiptSatisfies
        Selected certificate.baseProfile certificate.allRecipes
          certificate.defaultRunningAdmissible stable physical _ (by
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
            exact List.mem_cons_self)
    simpa [mux, CanonicalBranchPlan.onePortJoinReceipt] using rows
  have selectorTrue :
      boolCodec.decode [stable mux.selector] = some true := by
    simpa [mux, selectorExact, iterationZero] using selectorDecoded
  have joinedValuesExact :
      mux.joined.values stable = mux.onTrue.values stable :=
    mux.selected_true_sound stable selectorTrue joinRows
  have joinedDecoded :
      mux.joined.Decodes
        (certificate.baseProfile.family Selected) (.data .running)
        stable defaultRunning := by
    unfold ColumnBundle.Decodes at literalDecoded ⊢
    rw [joinedValuesExact]
    simpa [mux, CanonicalBranchPlan.onePortJoinRecipe,
      literalPlan, CanonicalStepConstructionPlans.baseLiteral,
      CanonicalStepPlan.baseLiteralPlan] using literalDecoded
  let continuationPlan :=
    CanonicalStepConstructionPlans.continuationHash
      Selected certificate.baseProfile certificate.allRecipes
  have continuationRows :
      ReceiptSatisfies continuationPlan.receipt.rows stable :=
    bodyReceiptSatisfies
      Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible stable physical
        continuationPlan.receipt (by
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
          exact List.mem_cons_self)
  have continuationOperandsDecoded :
      continuationPlan.frame.operands.Decodes
        (certificate.baseProfile.family Selected) stable
        (.cons iteration
          (.cons z0
            (.cons
              (machine.step
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
                zi witness)
              (.cons defaultRunning .nil)))) := by
    rw [CallFrame.operands, continuationPlan.contextExact]
    exact
      ⟨by
        simpa [CanonicalContexts.Step.continuationInput,
          CanonicalContexts.Step.common,
          CanonicalContexts.Step.afterStep,
          Columns.toSchemaBundles_get] using iterationDecoded,
        ⟨by
          simpa [CanonicalContexts.Step.continuationInput,
            CanonicalContexts.Step.common,
            CanonicalContexts.Step.afterStep,
            Columns.toSchemaBundles_get] using z0Decoded,
          ⟨by
            simpa [CanonicalContexts.Step.continuationInput,
              CanonicalContexts.Step.common,
              CanonicalContexts.Step.afterStep,
              Columns.toSchemaBundles_get] using zNextDecoded,
            ⟨by
              simpa [mux, CanonicalBranchPlan.onePortJoinRecipe,
                CanonicalContexts.Step.continuationInput,
                CanonicalContexts.Step.joined,
                Columns.toSchemaBundles_get] using joinedDecoded,
              trivial⟩⟩⟩⟩
  have continuationOne : stable continuationPlan.frame.one = 1 := by
    rw [continuationPlan.oneExact]
    exact physical.1
  have continuationActive : stable continuationPlan.frame.active = 1 := by
    rw [continuationPlan.activeExact]
    exact physical.1
  rcases
      continuationPlan.recipe.activeSoundness continuationPlan.frame stable
        (.cons iteration
          (.cons z0
            (.cons
              (machine.step
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
                zi witness)
              (.cons defaultRunning .nil))))
        continuationOne continuationActive continuationOperandsDecoded
        (by
          simpa [InvokePlan.receipt, InstructionReceipt.ofCall] using
            continuationRows) with
    ⟨digestOutputs, digestEvaluated, digestOutputDecoded⟩
  let digest := HVec.head digestOutputs
  have digestOutputsExact : digestOutputs = .cons digest .nil := by
    exact hvec_singleton_eta digestOutputs
  rw [digestOutputsExact] at digestEvaluated digestOutputDecoded
  let output :=
    Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor
      (Selected).setup (Selected).machine input (fun _ => defaultRunning)
  have digestExact : digest = output.x := by
    simp only [signature, callEval] at digestEvaluated
    have valuesEqual := Option.some.inj digestEvaluated
    simpa [output, input,
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor] using
      (congrArg HVec.head valuesEqual).symm
  rw [continuationPlan.outputsExact] at digestOutputDecoded
  rw [digestExact] at digestOutputDecoded
  have outputAccepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        Selected input output := by
    apply
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.accepts_iff_fixedOne
        Selected input output).2
    unfold
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneAccepts
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.fixedOneEval
    simp only
      [Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.eval,
       input, iterationZero, if_pos, initialState, output]
    rfl
  have resultDecoded :
      Columns.Decodes
        (certificate.baseProfile.family Selected)
        (CanonicalContexts.Step.result Selected) stable
        (stepResultValues Selected output) := by
    exact
      ⟨by
        simpa [CanonicalContexts.Step.result,
          CanonicalContexts.Step.resultExports,
          Columns.toSchemaBundles_get, output, input,
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.outputFor]
          using zNextDecoded,
        ⟨by
          simpa [mux, CanonicalBranchPlan.onePortJoinRecipe,
            CanonicalContexts.Step.result,
            CanonicalContexts.Step.resultExports,
            CanonicalContexts.Step.joined,
            Columns.toSchemaBundles_get, output]
            using joinedDecoded,
          ⟨by
            simpa [CanonicalContexts.Step.result,
              CanonicalContexts.Step.resultExports,
              Columns.toSchemaBundles_get] using digestOutputDecoded.1,
            trivial⟩⟩⟩
  exact
    ⟨input, output, by simpa [input] using iterationZero,
      outputAccepted, resultDecoded⟩

end

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentStepBasePhysicalRefinement
