import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStep
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.NativeCcsActivatedBridge

/-!
Contract: honest completeness of the native selected-CCS fixed-one Step.

Assurance tier: model-level.

Owns:
- refinement of the legacy activated receipt witness to the native selector;
- reuse of every ordinary receipt witness without a new completion write;
- an honest native CCS assignment for every admissible accepted Step.

Does not own: manifests, Rust emission, or a deployment application.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStepCompleteness

open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NativeCcsProgram
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.HyperNova.Construction2.Paper
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

/-- Legacy physical satisfaction implies native selected-CCS satisfaction on
the same assignment. The old residual values can remain in the total
assignment, but the native program neither reads nor allocates them. -/
theorem satisfied_of_legacy_physical
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (assignment : ColumnId → Field)
    (physical :
      (CanonicalStepSoundness.encoding Selected
        (ConcreteNifsNativeCcsStep.certificate
          application nifs step defaultAdmissible).baseProfile
        (ConcreteNifsNativeCcsStep.recipes
          application nifs step defaultAdmissible)
        defaultAdmissible).PhysicalSatisfies assignment) :
    (ConcreteNifsNativeCcsStep.program
      application nifs step defaultAdmissible).Satisfies assignment := by
  constructor
  · simpa [ConcreteNifsNativeCcsStep.program] using physical.1
  · apply
      (NativeCcsProgram.Program.satisfies_flattened_receipts_iff
        (ConcreteNifsNativeCcsStep.selectedReceipts
          application nifs step defaultAdmissible) assignment).2
    intro selectedReceipt selectedMember
    rcases List.mem_map.1 selectedMember with
      ⟨sourceReceipt, sourceMember, rfl⟩
    by_cases isTarget :
        sourceReceipt.owner = ConcreteNifsNativeCcsStep.targetOwner
    · have sourceEqualsTarget :
          sourceReceipt =
            ConcreteNifsNativeCcsStep.targetReceipt
              application nifs step defaultAdmissible := by
        apply
          (ConcreteNifsNativeCcsStep.certificate
            application nifs step defaultAdmissible
            ).canonicalStep.program.physical.receipt_eq_of_owner_eq
        · simpa [ConcreteNifsNativeCcsStep.sourceReceipts] using sourceMember
        · exact
            ConcreteNifsNativeCcsStep.targetReceipt_member
              application nifs step defaultAdmissible
        · exact isTarget.trans
            (ConcreteNifsNativeCcsStep.targetReceipt_owner
              application nifs step defaultAdmissible).symm
      subst sourceReceipt
      simp only [ConcreteNifsNativeCcsStep.replace_target]
      have activatedSatisfied :
          Goldilocks.Satisfies
            (ConcreteNifsActivatedProgram.rows
              application.profile nifs.operational
              (ConcreteNifsNativeCcsStep.invokePlan
                application nifs step defaultAdmissible).frame)
            assignment := by
        have targetSatisfied :=
          (CanonicalStepSoundness.encoding Selected
            (ConcreteNifsNativeCcsStep.certificate
              application nifs step defaultAdmissible).baseProfile
            (ConcreteNifsNativeCcsStep.recipes
              application nifs step defaultAdmissible)
            defaultAdmissible).receiptSatisfies assignment physical
              (ConcreteNifsNativeCcsStep.targetReceipt
                application nifs step defaultAdmissible)
              (by
                simpa [
                  ConcreteNifsNativeCcsStep.sourceReceipts_encoding] using
                  ConcreteNifsNativeCcsStep.targetReceipt_member
                    application nifs step defaultAdmissible)
        simpa using targetSatisfied
      have nativeSatisfied :=
        NativeCcsActivatedBridge.selected_of_activated
          ConcreteNifsNativeCcsStep.targetOwner
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
          assignment
          (ConcreteNifsActivatedProgram.residuals_length
            application.profile nifs.operational nifs.footprint
            (ConcreteNifsNativeCcsStep.invokePlan
              application nifs step defaultAdmissible).frame).symm
          (by
            simpa [ConcreteNifsActivatedProgram.rows] using
              activatedSatisfied)
      simpa [
        ConcreteNifsNativeCcsStep.nativeReceipt,
        ConcreteNifsNativeCcsProgram.selectedReceipt,
        ConcreteNifsNativeCcsProgram.sourceReceipt,
        ConcreteNifsRawProgram.rows] using nativeSatisfied
    · simp only [ConcreteNifsNativeCcsStep.replace, isTarget, if_neg]
      have sourceSatisfied :
          Goldilocks.Satisfies sourceReceipt.rows assignment := by
        apply
          (CanonicalStepSoundness.encoding Selected
            (ConcreteNifsNativeCcsStep.certificate
              application nifs step defaultAdmissible).baseProfile
            (ConcreteNifsNativeCcsStep.recipes
              application nifs step defaultAdmissible)
            defaultAdmissible).receiptSatisfies assignment physical
              sourceReceipt
        simpa [ConcreteNifsNativeCcsStep.sourceReceipts_encoding] using
          sourceMember
      exact
        NativeCcsSelector.complete oneColumn sourceReceipt.rows assignment
          sourceSatisfied

/-- Every admissible accepted Step has one native selected-CCS assignment
that encodes the exact typed input and output. -/
theorem complete
    (application : ConcreteNifsPlain270Profile.Phase4Application Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application.profile)
    (step : StepRecipeFor application)
    (defaultAdmissible : DefaultAdmissibleFor application)
    (input : CanonicalStepCompleteness.StepInputFor Selected)
    (output : CanonicalStepCompleteness.StepOutputFor Selected)
    (accepted :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
        Selected input output)
    (admissible :
      CanonicalStepCompleteness.AdmissibleExecution Selected
        (ConcreteNifsNativeCcsStep.certificate
          application nifs step defaultAdmissible).baseProfile
        input (CanonicalStepCompleteness.selectedRunning output)) :
    ∃ assignment : ColumnId → Field,
      (ConcreteNifsNativeCcsStep.program
        application nifs step defaultAdmissible).Satisfies assignment ∧
        Columns.Encodes
          ((ConcreteNifsNativeCcsStep.certificate
            application nifs step defaultAdmissible
            ).baseProfile.family Selected)
          (CanonicalContexts.Step.input Selected) assignment
          (stepInputValues Selected input) ∧
        Columns.Encodes
          ((ConcreteNifsNativeCcsStep.certificate
            application nifs step defaultAdmissible
            ).baseProfile.family Selected)
          (CanonicalContexts.Step.result Selected) assignment
          (stepResultValues Selected output) := by
  rcases
      (ConcreteNifsNativeCcsStep.certificate
        application nifs step defaultAdmissible).step_complete
          input output accepted admissible with
    ⟨assignment, physical, inputEncoded, outputEncoded⟩
  exact ⟨assignment,
    satisfied_of_legacy_physical application nifs step defaultAdmissible
      assignment physical,
    inputEncoded, outputEncoded⟩

end CompleteStep

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNativeCcsStepCompleteness
