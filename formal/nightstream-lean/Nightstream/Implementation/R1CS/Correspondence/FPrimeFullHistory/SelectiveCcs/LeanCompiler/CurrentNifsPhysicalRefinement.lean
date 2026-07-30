import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalPhysicalSoundness
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPaperRefinement
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.SelectiveCcs.LeanCompiler.CurrentDeployment

/-!
Contract: refine the active recursive NIFS occurrence in the current
Lean-emitted Step program directly from its finite physical assignment.

Assurance tier: model-level.

Owns: recovery of the running input, fresh input, proof, and output from the
exact receipt-owned rows; the selected call equation; and transport to the
unchanged occurrence-bound paper event.

Does not own: selection of a deployment application, setup generation,
inactive proof data, terminal semantics, Rust equality, or a probability
bound for the named event.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentNifsPhysicalRefinement

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalCertification
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCanonicalKey
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
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

private theorem recursiveNifs_receipt_mem
    (deployment :
      Deployment setup defaultRunning machine terminalRelations
        terminalChecks widths footprints) :
    let certificate :=
      ConcreteNifsCanonicalCertification.complete
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment
    let plan :=
      CanonicalStepConstructionPlans.recursiveNifs
        Selected certificate.baseProfile certificate.allRecipes
    plan.receipt ∈
      CanonicalStepPlan.receipts
        Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible := by
  dsimp
  rw [CanonicalStepPlan.receipts]
  apply List.mem_cons_of_mem
  apply List.mem_append_right
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
  exact List.mem_cons_self

/-- **Current recursive NIFS M4 bridge.** Exact current-program acceptance
and physical activation construct every semantic operand of the selected
NIFS call. The same occurrence then satisfies the fixed-active paper
transition or produces its unchanged occurrence-bound event.

No semantic Step input, operand value, decoder equation, accepted
proposition, or event is a premise. -/
theorem deployment_recursive_nifs_refines_from_physical_rows
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
    ∃ running : Running dimensions verifierRows,
      ∃ fresh : Fresh dimensions verifierRows,
        ∃ proof : Proof dimensions TranscriptState verifierRows,
          ∃ output : Running dimensions verifierRows,
            plan.frame.operands.Decodes
                (deployment.application.phase4.profile.family Selected)
                stable
                (.cons running (.cons fresh (.cons proof .nil))) ∧
              callEval Selected Call.nifsVerify
                  (.cons running (.cons fresh (.cons proof .nil))) =
                some (.cons output .nil) ∧
              plan.frame.outputs.Decodes
                (deployment.application.phase4.profile.family Selected)
                stable (.cons output .nil) ∧
              (ConcreteNifsPaperRefinement.PaperAcceptedAtOutput
                  (ConcreteNifsParameters.context
                    (ConcreteNifsCanonicalOperationalProfile.selectedKeys
                      setup
                      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                    running fresh proof).materialize
                  proof.certificate output ∨
                ConcreteNifsPaperRefinement.OccurrenceBoundEvent
                  (ConcreteNifsParameters.context
                    (ConcreteNifsCanonicalOperationalProfile.selectedKeys
                      setup
                      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
                    running fresh proof).materialize
                  proof.certificate output) := by
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
  have physical :
      certificate.canonicalStep.program.toEncoding.PhysicalSatisfies stable :=
    (CurrentCompiler.accepts_iff_physicalSatisfies
      certificate.canonicalStep.program.toEncoding
      (CurrentDeployment.deployment_step_columns_ge_270
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment)
      assignment).mp accepted
  have constantOne : stable plan.frame.one = 1 := by
    rw [plan.oneExact]
    exact physical.1
  have activeOne : stable plan.frame.active = 1 := by
    rw [plan.activeExact]
    exact recursiveActive
  have receiptRows :
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        plan.receipt.rows stable := by
    apply
      (CanonicalStepSoundness.encoding
        Selected certificate.baseProfile certificate.allRecipes
        certificate.defaultRunningAdmissible).receiptSatisfies
        stable physical plan.receipt
    exact recursiveNifs_receipt_mem
      setup defaultRunning machine terminalRelations terminalChecks
        widths footprints deployment
  have selectedRows :
      Nightstream.Implementation.Lowering.Goldilocks.Satisfies
        (ConcreteNifsActivatedProgram.rows
          deployment.application.phase4.profile
          (ConcreteNifsCanonicalOperationalProfile.operational
            setup defaultRunning machine terminalRelations terminalChecks
              widths footprints deployment.application)
          plan.frame)
        stable := by
    simpa [plan, certificate,
      CanonicalStepConstructionPlans.recursiveNifs,
      CanonicalStepPlan.recursiveNifsPlan] using receiptRows
  rcases
      ConcreteNifsCanonicalPhysicalSoundness.active_soundness
        setup defaultRunning machine terminalRelations terminalChecks
          widths footprints deployment.application deployment.footprintExact
          plan.frame stable constantOne activeOne selectedRows with
    ⟨running, fresh, proof, output, decodedInputs, evaluated,
      decodedOutput⟩
  have exactResult :=
    (ConcreteNifsSelectedCallFrame.call_result_exact
      running fresh proof output).mp evaluated
  have refinement :=
    ConcreteNifsPaperRefinement.accepted_refinesPaper_or_boundEvent
      GoldilocksField.goldilocks_euclidPrime
      (ConcreteNifsParameters.context
        (ConcreteNifsCanonicalOperationalProfile.selectedKeys setup
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
        running fresh proof).materialize
      proof.certificate output exactResult.1 exactResult.2
  exact
    ⟨running, fresh, proof, output, decodedInputs, evaluated,
      decodedOutput, refinement⟩

end

end Nightstream.Implementation.R1CS.FPrimeFullHistorySelectiveCcs.LeanCompiler.CurrentNifsPhysicalRefinement
