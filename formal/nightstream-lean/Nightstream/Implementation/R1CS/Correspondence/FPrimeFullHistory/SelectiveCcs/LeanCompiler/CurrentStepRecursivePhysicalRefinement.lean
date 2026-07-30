import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentNifsPhysicalRefinement

/-!
Contract: refine the recursive arm of the current Lean-emitted Step program
directly from one finite physical assignment.

Assurance tier: model-level.

Owns: recovery of the complete typed Step input on the active recursive arm,
current Step soundness, and the selected NIFS paper refinement for the exact
embedded occurrence.

Does not own: the base arm, derivation of branch activation, deployment
application selection, terminal semantics, Rust equality, or a probability
bound for the named event.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentStepRecursivePhysicalRefinement

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalApplicationRecovery
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPlain270Profile
open Nightstream.Implementation.R1CS
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

/-- **Current recursive Step M4 bridge.** Exact current-program acceptance
and recursive activation construct a complete typed Step input and output.
The same embedded NIFS occurrence satisfies the paper transition or produces
its unchanged occurrence-bound event.

No semantic Step input, NIFS operand, decoder equation, accepted proposition,
or event is a premise. -/
theorem deployment_recursive_step_refines_from_physical_rows
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
    (recursiveActive :
      let certificate :=
        ConcreteNifsCanonicalCertification.complete
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment
      let stable :=
        StableRows.pulledAssignment
          (EncodingRows.columnIndex
            certificate.canonicalStep.program.toEncoding) assignment
      stable (activationColumn SourceOwners.stepBranchPath false) = 1) :
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
    ∃ input :
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
          AppState Witness
          (Running dimensions verifierRows)
          (Fresh dimensions verifierRows)
          (Proof dimensions TranscriptState verifierRows),
      ∃ stepOutput :
          Output Digest AppState (Running dimensions verifierRows) 1,
        ∃ nifsOutput : Running dimensions verifierRows,
          Columns.Decodes
              (certificate.baseProfile.family Selected)
              (CanonicalContexts.Step.input Selected) stable
              (stepInputValues Selected input) ∧
            Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Accepts
              Selected input stepOutput ∧
            Columns.Decodes
              (certificate.baseProfile.family Selected)
              (CanonicalContexts.Step.result Selected) stable
              (stepResultValues Selected stepOutput) ∧
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
                input.nifsProof.certificate nifsOutput) := by
  dsimp only at recursiveActive
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
  rcases
      CurrentNifsPhysicalRefinement.deployment_recursive_nifs_refines_from_physical_rows
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment accepted recursiveActive with
    ⟨running, fresh, proof, nifsOutput, operandsDecoded, nifsEvaluated,
      nifsOutputDecoded, nifsRefinement⟩
  have operandsDecoded' := operandsDecoded
  rw [CallFrame.operands, plan.contextExact] at operandsDecoded'
  have runningDecoded :
      ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.running
            Selected)).Decodes
        (deployment.application.phase4.profile.family Selected)
        (.data .running) stable running := by
    simpa [CanonicalContexts.Step.afterEncodedEquality,
      CanonicalContexts.Step.afterEncode,
      CanonicalContexts.Step.afterFreshPublic,
      CanonicalContexts.Step.afterHash,
      CanonicalContexts.Step.common,
      CanonicalContexts.Step.afterStep,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.running,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.running,
      Columns.toSchemaBundles_get] using operandsDecoded'.1
  have freshDecoded :
      ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.fresh
            Selected)).Decodes
        (deployment.application.phase4.profile.family Selected)
        (.data .fresh) stable fresh := by
    simpa [CanonicalContexts.Step.afterEncodedEquality,
      CanonicalContexts.Step.afterEncode,
      CanonicalContexts.Step.afterFreshPublic,
      CanonicalContexts.Step.afterHash,
      CanonicalContexts.Step.common,
      CanonicalContexts.Step.afterStep,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.fresh,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.fresh,
      Columns.toSchemaBundles_get] using operandsDecoded'.2.1
  have proofDecoded :
      ((CanonicalContexts.Step.input Selected).toSchemaBundles.get
          (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.nifsProof
            Selected)).Decodes
        (deployment.application.phase4.profile.family Selected)
        (.data .nifsProof) stable proof := by
    simpa [CanonicalContexts.Step.afterEncodedEquality,
      CanonicalContexts.Step.afterEncode,
      CanonicalContexts.Step.afterFreshPublic,
      CanonicalContexts.Step.afterHash,
      CanonicalContexts.Step.common,
      CanonicalContexts.Step.afterStep,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.nifsProof,
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.nifsProof,
      Columns.toSchemaBundles_get] using operandsDecoded'.2.2.1
  rcases
      stepInput_decode_exists_of_recursiveOperands
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment stable running fresh proof
          runningDecoded freshDecoded proofDecoded with
    ⟨input, inputDecoded⟩
  have inputRunningDecoded :=
    SchemaBundles.get_decodes
      (certificate.baseProfile.family Selected) stable
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.running
        Selected)
      (CanonicalContexts.Step.input Selected).toSchemaBundles
      (stepInputValues Selected input) inputDecoded
  have inputFreshDecoded :=
    SchemaBundles.get_decodes
      (certificate.baseProfile.family Selected) stable
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.fresh
        Selected)
      (CanonicalContexts.Step.input Selected).toSchemaBundles
      (stepInputValues Selected input) inputDecoded
  have inputProofDecoded :=
    SchemaBundles.get_decodes
      (certificate.baseProfile.family Selected) stable
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.nifsProof
        Selected)
      (CanonicalContexts.Step.input Selected).toSchemaBundles
      (stepInputValues Selected input) inputDecoded
  have runningExact :
      input.running
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected =
        running :=
    Codec.decoded_value_unique _ inputRunningDecoded runningDecoded
  have freshExact : input.fresh = fresh :=
    Codec.decoded_value_unique _ inputFreshDecoded freshDecoded
  have proofExact : input.nifsProof = proof :=
    Codec.decoded_value_unique _ inputProofDecoded proofDecoded
  rcases
      CurrentDeployment.deployment_step_cir_sound
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment input accepted inputDecoded with
    ⟨stepOutput, stepAccepted, stepOutputDecoded⟩
  refine
    ⟨input, stepOutput, nifsOutput, inputDecoded, stepAccepted,
      stepOutputDecoded, ?_, nifsOutputDecoded, ?_⟩
  · simpa only [runningExact, freshExact, proofExact] using nifsEvaluated
  · rw [runningExact, freshExact, proofExact]
    exact nifsRefinement

end

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentStepRecursivePhysicalRefinement
