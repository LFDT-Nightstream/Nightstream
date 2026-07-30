import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentStepBasePhysicalRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentStepRecursivePhysicalRefinement

/-!
Contract: refine every accepted current Lean-emitted Step program, with its
branch selected by its own selector and activation rows.

Assurance tier: model-level.

Owns: branch-sensitive current Step soundness from one finite physical
assignment, including the exact recursive NIFS paper event.

Does not own: honest completeness, terminal semantics, deployment application
selection, Rust equality, or a probability bound for the named event.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentStepPhysicalRefinement

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

/-- **Current Step CIR-SOUND.** Every accepted current physical Step
assignment is either a valid frozen base transition, or a valid frozen
recursive transition whose embedded selected NIFS occurrence reaches its
paper relation or its unchanged occurrence-bound security event.

The branch, semantic inputs, output, NIFS operands, decoder equations, and
event are all conclusions. -/
theorem deployment_step_refines_from_physical_rows
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
                input.nifsProof.certificate nifsOutput)) := by
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
      CurrentStepBasePhysicalRefinement.deployment_step_branch_from_physical_rows
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment assignment accepted with
    ⟨iteration, iterationDecoded, baseBranch | recursiveBranch⟩
  · rcases baseBranch with ⟨_, baseActive, _⟩
    rcases
        CurrentStepBasePhysicalRefinement.deployment_base_step_refines_from_physical_rows
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment assignment accepted baseActive with
      ⟨input, output, iterationZero, stepAccepted, resultDecoded⟩
    exact Or.inl
      ⟨input, output, iterationZero, stepAccepted, resultDecoded⟩
  · rcases recursiveBranch with
      ⟨iterationNonzero, _, recursiveActive⟩
    rcases
        CurrentStepRecursivePhysicalRefinement.deployment_recursive_step_refines_from_physical_rows
          setup defaultRunning machine terminalRelations terminalChecks
            widths footprints deployment assignment accepted recursiveActive with
      ⟨input, output, nifsOutput, inputDecoded, stepAccepted,
        resultDecoded, nifsEvaluated, nifsOutputDecoded, refinement⟩
    have inputIterationDecoded :=
      SchemaBundles.get_decodes
        (certificate.baseProfile.family Selected) stable
        (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.iteration
          Selected)
        (CanonicalContexts.Step.input Selected).toSchemaBundles
        (stepInputValues Selected input) inputDecoded
    have iterationExact : input.iteration = iteration :=
      Codec.decoded_value_unique _ inputIterationDecoded iterationDecoded
    have inputIterationNonzero : input.iteration ≠ 0 := by
      rw [iterationExact]
      exact iterationNonzero
    exact Or.inr
      ⟨input, output, nifsOutput, inputIterationNonzero, inputDecoded,
        stepAccepted, resultDecoded, nifsEvaluated, nifsOutputDecoded,
        refinement⟩

end

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentStepPhysicalRefinement
