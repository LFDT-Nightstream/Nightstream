import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CompleteApplicationCertification
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentCompiler

/-!
Contract: close current-program M4 for each complete Lean-owned fixed-one
deployment.

Assurance tier: model-level.

Owns: the exact compiler link from finite selective-CCS assignments to the
complete Step and Terminal receipt programs; soundness into the frozen
checkers; honest reassembly back into the same finite matrices; exact
source-receipt ownership and canonical manifest evidence; and transport of
the recursive NIFS occurrence to its unchanged paper event.

Does not own: selection of one application deployment, cryptographic setup
generation, equality with Rust, or a probability bound for the named event.

Emits constraints: no new rows. It compiles the exact rows already owned by
the complete deployment.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalRunningCodec
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Protocol.FPrime.ConcretePhi81
open Nightstream.SuperNeo.Concrete.Phi81Relation.FPrimeCarrier270
open Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler

theorem canonical_running_width_ge_270
    {shape : SemanticShape}
    {verifierRows : Nat}
    (publicFits :
      ringDegree * 5 ≤ shape.carrierWidth) :
    270 ≤ (runningCodec shape 5 verifierRows publicFits).width := by
  simp [ringDegree, runningCodec, parentPayloadCodec, runningPayloadCodec,
    completePayloadCodec, Codec.pullback, Codec.product,
    Codec.finFunction, Codec.ofInjectiveEncoding,
    ConcreteNifsCanonicalCodecCore.commitmentCodec_width,
    ConcreteNifsCanonicalCodecCore.publicInputCodec_width,
    ConcreteNifsCanonicalCodecCore.pointCodec_width,
    ConcreteNifsCanonicalCodecCore.evaluationsCodec_width]
  omega

theorem step_columns_ge_running
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    parameters.widths.running ≤
      certificate.canonicalStep.program.toEncoding.columnIds.length := by
  let inputAllocations :=
    schemaOwnedColumns
      (inputColumns (stepInputSchema parameters))
  have inputLength :
      inputAllocations.length =
        parameters.widths.iteration +
          parameters.widths.state +
          parameters.widths.state +
          parameters.widths.running +
          parameters.widths.fresh +
          parameters.widths.witness +
          parameters.widths.nifsProof := by
    simp [inputAllocations, stepInputSchema, schemaOwnedColumns,
      inputColumns, allocateSchema, allocateSchemaFrom,
      bundleOwnedColumns, Ports.committedNat, Ports.committedState,
      Ports.committedRunning, Ports.committedFresh,
      Ports.committedWitness, Ports.committedNifsProof,
      dataPort, committedLayout, ownedLayout]
    omega
  have inputBound : parameters.widths.running ≤ inputAllocations.length := by
    rw [inputLength]
    omega
  have totalLength :
      certificate.canonicalStep.program.toEncoding.columnIds.length =
        1 + inputAllocations.length +
          ((CanonicalStepPlan.bodyReceipts parameters
              certificate.baseProfile certificate.allRecipes
                certificate.defaultRunningAdmissible).flatMap
                  (fun receipt => receipt.allocations)).length := by
    change
      (CanonicalStepPlan.aligned parameters certificate.baseProfile
        certificate.allRecipes certificate.defaultRunningAdmissible
          ).toEncoding.columnIds.length =
        _
    simp only [SourceAlignment.AlignedReceiptProgram.toEncoding,
      ReceiptProgram.toEncoding, Encoding.columnIds, Encoding.columns,
      CanonicalStepPlan.aligned, CanonicalStepPlan.physical,
      CanonicalStepPlan.receipts, List.flatMap_cons, List.flatMap_append,
      List.length_map, List.length_append,
      InstructionReceipt.prelude_allocations, preludeColumns,
      List.length_singleton]
    rw [InputReceipts.allocations_exact]
  rw [totalLength]
  omega

theorem terminal_columns_ge_running
    {parameters : Parameters}
    (certificate : CompleteApplicationCertification parameters) :
    parameters.widths.running ≤
      certificate.canonicalTerminal.program.toEncoding.columnIds.length := by
  let inputAllocations :=
    schemaOwnedColumns
      (inputColumns (terminalInputSchema parameters))
  have inputLength :
      inputAllocations.length =
        parameters.widths.iteration +
          parameters.widths.state +
          parameters.widths.state +
          parameters.widths.running +
          parameters.widths.runningWitness +
          parameters.widths.fresh +
          parameters.widths.freshWitness := by
    simp [inputAllocations, terminalInputSchema, schemaOwnedColumns,
      inputColumns, allocateSchema, allocateSchemaFrom,
      bundleOwnedColumns, Ports.publicNat, Ports.publicState,
      Ports.committedRunning, Ports.committedRunningWitness,
      Ports.committedFresh, Ports.committedFreshWitness,
      dataPort, publicLayout, committedLayout, ownedLayout]
    omega
  have totalLength :
      certificate.canonicalTerminal.program.toEncoding.columnIds.length =
        1 + inputAllocations.length +
          ((CanonicalTerminalPlan.bodyReceipts parameters
              certificate.baseProfile certificate.allRecipes).flatMap
                (fun receipt => receipt.allocations)).length := by
    change
      (CanonicalTerminalPlan.aligned parameters certificate.baseProfile
        certificate.allRecipes).toEncoding.columnIds.length =
        _
    simp only [SourceAlignment.AlignedReceiptProgram.toEncoding,
      ReceiptProgram.toEncoding, Encoding.columnIds, Encoding.columns,
      CanonicalTerminalPlan.aligned, CanonicalTerminalPlan.physical,
      CanonicalTerminalPlan.receipts, List.flatMap_cons, List.flatMap_append,
      List.length_map, List.length_append,
      InstructionReceipt.prelude_allocations, preludeColumns,
      List.length_singleton]
    rw [InputReceipts.allocations_exact]
  rw [totalLength, inputLength]
  omega

private abbrev TranscriptState :=
  Nightstream.Implementation.R1CS.Canonical.Poseidon2Duplex.State

section

variable {Digest AppState Witness Encoded RunningWitness FreshWitness : Type}
variable [DecidableEq AppState] [DecidableEq Encoded]
variable {dimensions : Dimensions}
variable {verifierRows : Nat}
variable (setup : RelationSetup dimensions verifierRows)
variable (defaultRunning : Running dimensions verifierRows)
variable
  (machine :
    Nightstream.HyperNova.Construction2.Paper.Machine
      (Key dimensions TranscriptState verifierRows)
      Digest AppState Witness
      (Running dimensions verifierRows)
      (Fresh dimensions verifierRows)
      Encoded 1)
variable
  (terminalRelations :
    Nightstream.HyperNova.Construction2.Paper.TerminalRelations
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

theorem deployment_running_width_ge_270
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    270 ≤ widths.running := by
  change
    270 ≤
      (ConcreteNifsPlain270Profile.selected dimensions
        (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup)
        defaultRunning machine terminalRelations terminalChecks widths
          footprints).widths.running
  rw [deployment.application.phase4.profile.widthsExact]
  change 270 ≤ deployment.application.phase4.profile.codecs.running.width
  rw [deployment.application.runningCodec_exact]
  exact canonical_running_width_ge_270
    (ConcreteNifsPlain270Profile.publicFits dimensions)

theorem deployment_step_columns_ge_270
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    270 ≤
      ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
              ).columnIds.length := by
  exact Nat.le_trans
    (deployment_running_width_ge_270 setup defaultRunning machine
      terminalRelations terminalChecks widths footprints deployment)
    (step_columns_ge_running
      (ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment))

theorem deployment_terminal_columns_ge_270
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    270 ≤
      ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalTerminal.program.toEncoding
              ).columnIds.length := by
  exact Nat.le_trans
    (deployment_running_width_ge_270 setup defaultRunning machine
      terminalRelations terminalChecks widths footprints deployment)
    (terminal_columns_ge_running
      (ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment))

theorem deployment_step_cir_sound
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment :
      Fin
        ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
              ).columnIds.length →
        Nightstream.SuperNeo.Concrete.F)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        AppState Witness
          (Running dimensions verifierRows)
          (Fresh dimensions verifierRows)
          (Proof dimensions TranscriptState verifierRows))
    (accepted :
      CurrentCompiler.Accepts
        (ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
        (deployment_step_columns_ge_270 setup defaultRunning machine
          terminalRelations terminalChecks widths footprints deployment)
        assignment)
    (inputDecoded :
      let certificate :=
        ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment
      let stable :=
        StableRows.pulledAssignment
          (EncodingRows.columnIndex
            certificate.canonicalStep.program.toEncoding) assignment
      Columns.Decodes
        (certificate.baseProfile.family Selected)
        (CanonicalContexts.Step.input Selected) stable
        (stepInputValues Selected input)) :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    let stable :=
      StableRows.pulledAssignment
        (EncodingRows.columnIndex
          certificate.canonicalStep.program.toEncoding) assignment
    ∃ output :
        Nightstream.HyperNova.Construction2.Paper.Output
          Digest AppState (Running dimensions verifierRows) 1,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
          Selected input output ∧
        Columns.Decodes
          (certificate.baseProfile.family Selected)
          (CanonicalContexts.Step.result Selected) stable
          (stepResultValues Selected output) := by
  dsimp only
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
      (deployment_step_columns_ge_270 setup defaultRunning machine
        terminalRelations terminalChecks widths footprints deployment)
      assignment).mp accepted
  exact certificate.step_sound stable input physical inputDecoded

theorem deployment_terminal_cir_sound
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment :
      Fin
        ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalTerminal.program.toEncoding
              ).columnIds.length →
        Nightstream.SuperNeo.Concrete.F)
    (statement :
      Nightstream.HyperNova.Construction2.Paper.TerminalStatement AppState)
    (proof :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
        Selected)
    (accepted :
      CurrentCompiler.Accepts
        (ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalTerminal.program.toEncoding
        (deployment_terminal_columns_ge_270 setup defaultRunning machine
          terminalRelations terminalChecks widths footprints deployment)
        assignment)
    (inputDecoded :
      let certificate :=
        ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment
      let stable :=
        StableRows.pulledAssignment
          (EncodingRows.columnIndex
            certificate.canonicalTerminal.program.toEncoding) assignment
      Columns.Decodes
        (certificate.baseProfile.family Selected)
        (CanonicalContexts.Terminal.input Selected) stable
        (terminalInputValues Selected statement proof)) :
    Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
      Selected statement proof := by
  dsimp only at inputDecoded ⊢
  let certificate :=
    ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  let stable :=
    StableRows.pulledAssignment
      (EncodingRows.columnIndex
        certificate.canonicalTerminal.program.toEncoding) assignment
  have physical :
      certificate.canonicalTerminal.program.toEncoding.PhysicalSatisfies
        stable :=
    (CurrentCompiler.accepts_iff_physicalSatisfies
      certificate.canonicalTerminal.program.toEncoding
      (deployment_terminal_columns_ge_270 setup defaultRunning machine
        terminalRelations terminalChecks widths footprints deployment)
      assignment).mp accepted
  exact certificate.terminal_sound stable statement proof physical inputDecoded

theorem deployment_step_cir_complete
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (input : CanonicalStepCompleteness.StepInputFor Selected)
    (output : CanonicalStepCompleteness.StepOutputFor Selected)
    (accepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        Selected input output)
    (admissible :
      let certificate :=
        ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment
      CanonicalStepCompleteness.AdmissibleExecution Selected
        certificate.baseProfile input
        (CanonicalStepCompleteness.selectedRunning output)) :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    ∃ assignment : ColumnId → Nightstream.SuperNeo.Concrete.F,
      CurrentCompiler.Accepts
          certificate.canonicalStep.program.toEncoding
          (deployment_step_columns_ge_270 setup defaultRunning machine
            terminalRelations terminalChecks widths footprints deployment)
          (EncodingRows.indexedAssignment
            certificate.canonicalStep.program.toEncoding assignment) ∧
        Columns.Encodes
          (certificate.baseProfile.family Selected)
          (CanonicalContexts.Step.input Selected) assignment
          (stepInputValues Selected input) ∧
        Columns.Encodes
          (certificate.baseProfile.family Selected)
          (CanonicalContexts.Step.result Selected) assignment
          (stepResultValues Selected output) := by
  dsimp only at admissible ⊢
  let certificate :=
    ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  rcases certificate.step_complete input output accepted admissible with
    ⟨assignment, physical, inputEncoded, outputEncoded⟩
  refine ⟨assignment, ?_, inputEncoded, outputEncoded⟩
  exact
    (CurrentCompiler.indexedAssignment_accepts_iff
      certificate.canonicalStep.program.toEncoding
      (deployment_step_columns_ge_270 setup defaultRunning machine
        terminalRelations terminalChecks widths footprints deployment)
      assignment).mpr physical

theorem deployment_terminal_cir_complete
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (statement :
      CanonicalTerminalCompleteness.TerminalStatementFor Selected)
    (proof : CanonicalTerminalCompleteness.TerminalProofFor Selected)
    (accepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
        Selected statement proof)
    (admissible :
      let certificate :=
        ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment
      CanonicalTerminalCompleteness.AdmissibleExecution Selected
        certificate.baseProfile statement proof) :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    ∃ assignment : ColumnId → Nightstream.SuperNeo.Concrete.F,
      CurrentCompiler.Accepts
          certificate.canonicalTerminal.program.toEncoding
          (deployment_terminal_columns_ge_270 setup defaultRunning machine
            terminalRelations terminalChecks widths footprints deployment)
          (EncodingRows.indexedAssignment
            certificate.canonicalTerminal.program.toEncoding assignment) ∧
        Columns.Encodes
          (certificate.baseProfile.family Selected)
          (CanonicalContexts.Terminal.input Selected) assignment
          (terminalInputValues Selected statement proof) := by
  dsimp only at admissible ⊢
  let certificate :=
    ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  rcases certificate.terminal_complete statement proof accepted admissible with
    ⟨assignment, physical, inputEncoded⟩
  refine ⟨assignment, ?_, inputEncoded⟩
  exact
    (CurrentCompiler.indexedAssignment_accepts_iff
      certificate.canonicalTerminal.program.toEncoding
      (deployment_terminal_columns_ge_270 setup defaultRunning machine
        terminalRelations terminalChecks widths footprints deployment)
      assignment).mpr physical

theorem deployment_recursive_nifs_refines_or_bound_event
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints)
    (assignment :
      Fin
        ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
              ).columnIds.length →
        Nightstream.SuperNeo.Concrete.F)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        AppState Witness
          (Running dimensions verifierRows)
          (Fresh dimensions verifierRows)
          (Proof dimensions TranscriptState verifierRows))
    (accepted :
      CurrentCompiler.Accepts
        (ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
        (deployment_step_columns_ge_270 setup defaultRunning machine
          terminalRelations terminalChecks widths footprints deployment)
        assignment)
    (inputDecoded :
      let certificate :=
        ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment
      let stable :=
        StableRows.pulledAssignment
          (EncodingRows.columnIndex
            certificate.canonicalStep.program.toEncoding) assignment
      Columns.Decodes
        (certificate.baseProfile.family Selected)
        (CanonicalContexts.Step.input Selected) stable
        (stepInputValues Selected input))
    (iterationNonzero : input.iteration ≠ 0) :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    let stable :=
      StableRows.pulledAssignment
        (EncodingRows.columnIndex
          certificate.canonicalStep.program.toEncoding) assignment
    let plan :=
      CanonicalStepConstructionPlans.recursiveNifs
        Selected certificate.baseProfile certificate.allRecipes
    ∃ output : Running dimensions verifierRows,
      plan.frame.outputs.Decodes
          (deployment.application.phase4.profile.family Selected)
          stable (.cons output .nil) ∧
        (ConcreteNifsPaperRefinement.PaperAcceptedAtOutput
            (ConcreteNifsParameters.context
              (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              (input.running
                Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Step.selected)
              input.fresh input.nifsProof).materialize
            input.nifsProof.certificate output ∨
          ConcreteNifsPaperRefinement.OccurrenceBoundEvent
            (ConcreteNifsParameters.context
              (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              (input.running
                Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary.Step.selected)
              input.fresh input.nifsProof).materialize
            input.nifsProof.certificate output) := by
  dsimp only at inputDecoded
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
      (deployment_step_columns_ge_270 setup defaultRunning machine
        terminalRelations terminalChecks widths footprints deployment)
      assignment).mp accepted
  exact
    ConcreteNifsCanonicalCertification.recursiveNifs_refinesPaper_or_boundEvent
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment stable input physical inputDecoded
        iterationNonzero

/-- Structural half of current-program M4 for one complete Lean-owned
deployment.  The semantic half is supplied by the five assembled theorems
above: Step and Terminal soundness/completeness, plus the recursive NIFS
paper-event refinement. -/
structure DeploymentStructuralEvidence
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) : Prop where
  stepCompiler :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    CurrentCompiler.Evidence
      certificate.canonicalStep.program
      (deployment_step_columns_ge_270 setup defaultRunning machine
        terminalRelations terminalChecks widths footprints deployment)
  terminalCompiler :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    CurrentCompiler.Evidence
      certificate.canonicalTerminal.program
      (deployment_terminal_columns_ge_270 setup defaultRunning machine
        terminalRelations terminalChecks widths footprints deployment)
  stepCanonical :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    CanonicalEncoding.Step.Claims certificate.canonicalStep
  terminalCanonical :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    CanonicalEncoding.Terminal.Claims certificate.canonicalTerminal

theorem deployment_structural_evidence
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    DeploymentStructuralEvidence setup defaultRunning machine
      terminalRelations terminalChecks widths footprints deployment where
  stepCompiler :=
    CurrentCompiler.evidence
      (ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment).canonicalStep.program
      (deployment_step_columns_ge_270 setup defaultRunning machine
        terminalRelations terminalChecks widths footprints deployment)
  terminalCompiler :=
    CurrentCompiler.evidence
      (ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment).canonicalTerminal.program
      (deployment_terminal_columns_ge_270 setup defaultRunning machine
        terminalRelations terminalChecks widths footprints deployment)
  stepCanonical :=
    (ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment).step_obligation10
  terminalCanonical :=
    (ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment).terminal_obligation10

end

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentDeployment
