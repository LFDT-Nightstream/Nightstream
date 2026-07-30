import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentStepPhysicalRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentTerminalPhysicalRefinement

/-!
Contract: assemble current Lean-owned local correspondence and recursive M4
evidence for one complete fixed-one deployment.

Assurance tier: model-level.

Owns: branch-complete Step soundness, Terminal soundness, honest Step and
Terminal completeness, exact receipt ownership and cost, canonical manifest
structure, recovery of all application-owned input codecs, and the explicit
boundary that identifies the recursive NIFS setup system with the system
compiled from the complete Step rows.

Does not own: selection of one application, application-specific rows inside
the supplied `step` recipe, construction of the recursive fixed point,
equality with Rust, or probability bounds for named security events.

Emits constraints: no new rows.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentM4

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
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

private abbrev Deployment :=
  ConcreteNifsCanonicalCertification.Deployment
    setup defaultRunning machine terminalRelations terminalChecks
      widths footprints

/-- Semantic result of one accepted current Step assignment.

The base branch yields the frozen base transition. The recursive branch also
yields the exact selected NIFS output and either its paper relation or its
unchanged occurrence-bound event. -/
def StepOutcome
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints)
    (assignment :
      Fin
        ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalStep.program.toEncoding
              ).columnIds.length →
        F) : Prop :=
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
  (∃ input :
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
          (stepResultValues Selected output)) ∨
  (∃ input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        AppState Witness
        (Running dimensions verifierRows)
        (Fresh dimensions verifierRows)
        (Proof dimensions TranscriptState verifierRows),
    ∃ output :
        Output Digest AppState (Running dimensions verifierRows) 1,
      ∃ nifsOutput : Running dimensions verifierRows,
        input.iteration ≠ 0 ∧
          Columns.Decodes
            (certificate.baseProfile.family Selected)
            (CanonicalContexts.Step.input Selected) stable
            (stepInputValues Selected input) ∧
          Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
            Selected input output ∧
          Columns.Decodes
            (certificate.baseProfile.family Selected)
            (CanonicalContexts.Step.result Selected) stable
            (stepResultValues Selected output) ∧
          callEval Selected Call.nifsVerify
              (.cons
                (input.running
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                (.cons input.fresh (.cons input.nifsProof .nil))) =
            some (.cons nifsOutput .nil) ∧
          plan.frame.outputs.Decodes
            (deployment.application.phase4.profile.family Selected)
            stable (.cons nifsOutput .nil) ∧
          (ConcreteNifsPaperRefinement.PaperAcceptedAtOutput
              (ConcreteNifsParameters.context
                (ConcreteNifsCanonicalOperationalProfile.selectedKeys
                  setup
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                (input.running
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                input.fresh input.nifsProof).materialize
              input.nifsProof.certificate nifsOutput ∨
            ConcreteNifsPaperRefinement.OccurrenceBoundEvent
              (ConcreteNifsParameters.context
                (ConcreteNifsCanonicalOperationalProfile.selectedKeys
                  setup
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                (input.running
                  Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                input.fresh input.nifsProof).materialize
              input.nifsProof.certificate nifsOutput))

/-- Current Step CIR-SOUND with no semantic input or decode premise. -/
def StepSoundness
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints) : Prop :=
  ∀ assignment,
    CurrentCompiler.Accepts
      (ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment).canonicalStep.program.toEncoding
      (CurrentDeployment.deployment_step_columns_ge_270
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment)
      assignment →
    StepOutcome setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment assignment

/-- Semantic result of one accepted current Terminal assignment. -/
def TerminalOutcome
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints)
    (assignment :
      Fin
        ((ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment).canonicalTerminal.program.toEncoding
              ).columnIds.length →
        F) : Prop :=
  let certificate :=
    ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  let stable :=
    StableRows.pulledAssignment
      (EncodingRows.columnIndex
        certificate.canonicalTerminal.program.toEncoding) assignment
  ∃ statement : TerminalStatement AppState,
    ∃ proof :
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.FixedOneTerminal.Proof
          Selected,
      Columns.Decodes
          (certificate.baseProfile.family Selected)
          (CanonicalContexts.Terminal.input Selected) stable
          (terminalInputValues Selected statement proof) ∧
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
          Selected statement proof

/-- Current Terminal CIR-SOUND with no semantic input or decode premise. -/
def TerminalSoundness
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints) : Prop :=
  ∀ assignment,
    CurrentCompiler.Accepts
      (ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment).canonicalTerminal.program.toEncoding
      (CurrentDeployment.deployment_terminal_columns_ge_270
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment)
      assignment →
    TerminalOutcome setup defaultRunning machine terminalRelations
      terminalChecks widths footprints deployment assignment

/-- Current Step CIR-COMPLETE for every admissible accepted execution. -/
def StepCompleteness
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints) : Prop :=
  ∀ (input : CanonicalStepCompleteness.StepInputFor Selected)
    (output : CanonicalStepCompleteness.StepOutputFor Selected)
    (_accepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        Selected input output)
    (_admissible :
      let certificate :=
        ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment
      CanonicalStepCompleteness.AdmissibleExecution Selected
        certificate.baseProfile input
        (CanonicalStepCompleteness.selectedRunning output)),
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    ∃ assignment : ColumnId → F,
      CurrentCompiler.Accepts
          certificate.canonicalStep.program.toEncoding
          (CurrentDeployment.deployment_step_columns_ge_270
            setup defaultRunning machine terminalRelations terminalChecks
              widths footprints deployment)
          (EncodingRows.indexedAssignment
            certificate.canonicalStep.program.toEncoding assignment) ∧
        Columns.Encodes
          (certificate.baseProfile.family Selected)
          (CanonicalContexts.Step.input Selected) assignment
          (stepInputValues Selected input) ∧
        Columns.Encodes
          (certificate.baseProfile.family Selected)
          (CanonicalContexts.Step.result Selected) assignment
          (stepResultValues Selected output)

/-- Current Terminal CIR-COMPLETE for every admissible accepted execution. -/
def TerminalCompleteness
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints) : Prop :=
  ∀ (statement :
      CanonicalTerminalCompleteness.TerminalStatementFor Selected)
    (proof : CanonicalTerminalCompleteness.TerminalProofFor Selected)
    (_accepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Terminal.Accepts
        Selected statement proof)
    (_admissible :
      let certificate :=
        ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment
      CanonicalTerminalCompleteness.AdmissibleExecution Selected
        certificate.baseProfile statement proof),
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    ∃ assignment : ColumnId → F,
      CurrentCompiler.Accepts
          certificate.canonicalTerminal.program.toEncoding
          (CurrentDeployment.deployment_terminal_columns_ge_270
            setup defaultRunning machine terminalRelations terminalChecks
              widths footprints deployment)
          (EncodingRows.indexedAssignment
            certificate.canonicalTerminal.program.toEncoding assignment) ∧
        Columns.Encodes
          (certificate.baseProfile.family Selected)
          (CanonicalContexts.Terminal.input Selected) assignment
          (terminalInputValues Selected statement proof)

/-- Exact same-system condition for recursive use of the compiled Step.

The first three equations expose the fixed-point dimensions.  The final
heterogeneous equality states that the NIFS key checks the exact
thirteen-matrix system compiled from the complete Step rows, not an arbitrary
setup relation with unrelated matrices. -/
def RecursiveSystemCoherence
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints) : Prop :=
  let certificate :=
    ConcreteNifsCanonicalCertification.complete
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  let encoding := certificate.canonicalStep.program.toEncoding
  let publicWidth :=
    CurrentDeployment.deployment_step_columns_ge_270
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  let profile := Profile.ofEncoding encoding publicWidth
  dimensions.rowVariables = profile.rowVariables ∧
    dimensions.alignedLogicalWidth = encoding.columnIds.length ∧
    dimensions.matrixCount = 13 ∧
    HEq setup.system
      (CurrentCompiler.compiledSystem encoding publicWidth)

/-- Current local correspondence evidence for one proof-carrying deployment.

Application-specific constraints are inside `deployment.step`. This structure
does not add a second generic application program or claim that every
application uses one particular canonical-opening gadget. -/
structure LocalEvidence
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints) : Prop where
  stepSound : StepSoundness setup defaultRunning machine terminalRelations
    terminalChecks widths footprints deployment
  terminalSound :
    TerminalSoundness setup defaultRunning machine terminalRelations
      terminalChecks widths footprints deployment
  stepComplete :
    StepCompleteness setup defaultRunning machine terminalRelations
      terminalChecks widths footprints deployment
  terminalComplete :
    TerminalCompleteness setup defaultRunning machine terminalRelations
      terminalChecks widths footprints deployment
  structural :
    CurrentDeployment.DeploymentStructuralEvidence
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  applicationInputsRecoverable :
    ApplicationCodecRecovery Selected
      deployment.application.phase4.profile.codecs

/-- Local Step and Terminal correspondence from the emitted programs.

It constructs every evidence field from the emitted programs and the
proof-carrying deployment. No caller supplies a semantic execution, accepted
proposition, row count, owner map, or security event. -/
theorem deployment_local_correspondence
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints) :
    LocalEvidence setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment where
  stepSound := by
    intro assignment accepted
    exact
      CurrentStepPhysicalRefinement.deployment_step_refines_from_physical_rows
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment accepted
  terminalSound := by
    intro assignment accepted
    exact
      CurrentTerminalPhysicalRefinement.deployment_terminal_refines_from_physical_rows
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment accepted
  stepComplete := by
    intro input output accepted admissible
    exact
      CurrentDeployment.deployment_step_cir_complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment input output accepted admissible
  terminalComplete := by
    intro statement proof accepted admissible
    exact
      CurrentDeployment.deployment_terminal_cir_complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment statement proof accepted admissible
  structural :=
    CurrentDeployment.deployment_structural_evidence
      setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment
  applicationInputsRecoverable := deployment.applicationCodecRecovery

/-- Complete current Lean-owned recursive M4 evidence.

Unlike local row correspondence, recursive M4 must prove that the relation
inside the selected NIFS key is the exact relation compiled from the complete
Step rows. -/
structure Evidence
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints) : Prop
    extends
      LocalEvidence setup defaultRunning machine terminalRelations
        terminalChecks widths footprints deployment where
  recursiveSystem :
    RecursiveSystemCoherence setup defaultRunning machine terminalRelations
      terminalChecks widths footprints deployment

/-- Headline current Lean-owned recursive M4 theorem.

The deployment supplies the application. The coherence proof must come from a
Lean-owned fixed-point construction for that deployment. -/
theorem deployment_m4
    (deployment : Deployment setup defaultRunning machine terminalRelations
      terminalChecks widths footprints)
    (coherence :
      RecursiveSystemCoherence setup defaultRunning machine terminalRelations
        terminalChecks widths footprints deployment) :
    Evidence setup defaultRunning machine terminalRelations terminalChecks
      widths footprints deployment where
  toLocalEvidence :=
    deployment_local_correspondence setup defaultRunning machine
      terminalRelations terminalChecks widths footprints deployment
  recursiveSystem := coherence

end

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentM4
