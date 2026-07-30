import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.CanonicalStepConstructionPlans
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCompleteApplication
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPaperRefinement

/-!
Contract: transport the selected NIFS paper refinement through its exact
recursive Step occurrence.

The theorem below starts from satisfaction of the complete canonical Step
program. It projects the receipt-owned `nifsVerify` rows, derives the recursive
activation from the internally computed iteration selector, and decodes the
three NIFS operands from the authoritative Step input columns. It then applies
the occurrence-bound paper refinement.

The application `step` remains a proof-carrying HyperNova setup input. No Rust
row, generated artifact, accepted proposition, source-authority premise, or
caller-selected event enters this theorem.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2400000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsStepPaperRefinement

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

section SelectedFrame

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

private abbrev FamilyFor
    (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

/-- The exact recursive NIFS receipt is a member of the complete Step
receipt list. -/
private theorem recursiveNifs_receipt_mem
    (application : Poseidon23ApplicationProfile Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application)
    (step :
      CallRecipe (signature Selected) (FamilyFor application) Call.step)
    (defaultRunningAdmissible :
      ((FamilyFor application).codecFor (.data .running)).Admissible
        defaultRunning) :
    let complete :=
      ConcreteNifsCompleteApplication.complete application nifs step
        defaultRunningAdmissible
    let plan :=
      CanonicalStepConstructionPlans.recursiveNifs
        Selected complete.baseProfile complete.allRecipes
    plan.receipt ∈
      CanonicalStepPlan.receipts
        Selected complete.baseProfile complete.allRecipes
        defaultRunningAdmissible := by
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

/-- **Headline Step/NIFS refinement.** On the recursive Step arm, complete
physical Step satisfaction yields the output decoded by the exact embedded
`nifsVerify` occurrence and either the unchanged fixed-active paper
transition or one occurrence-bound named event. -/
theorem recursiveNifs_refinesPaper_or_boundEvent
    (application : Poseidon23ApplicationProfile Selected)
    (nifs : ConcreteNifsVerifyCallRecipe.Certification application)
    (step :
      CallRecipe (signature Selected) (FamilyFor application) Call.step)
    (defaultRunningAdmissible :
      ((FamilyFor application).codecFor (.data .running)).Admissible
        defaultRunning)
    (assignment : ColumnId → Field)
    (input :
      Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.Canonical.Input
        AppState Witness
        (SelectedRunning shape publicRingColumns publicFits verifierRows)
        (SelectedFresh shape publicRingColumns publicFits verifierRows)
        (SelectedProof shape TranscriptState publicRingColumns publicFits
          verifierRows))
    (physical :
      let complete :=
        ConcreteNifsCompleteApplication.complete application nifs step
          defaultRunningAdmissible
      (CanonicalStepSoundness.encoding
        Selected complete.baseProfile complete.allRecipes
        defaultRunningAdmissible).PhysicalSatisfies assignment)
    (inputDecoded :
      let complete :=
        ConcreteNifsCompleteApplication.complete application nifs step
          defaultRunningAdmissible
      Columns.Decodes
        (complete.baseProfile.family Selected)
        (CanonicalContexts.Step.input Selected) assignment
        (stepInputValues Selected input))
    (iterationNonzero : input.iteration ≠ 0) :
    let complete :=
      ConcreteNifsCompleteApplication.complete application nifs step
        defaultRunningAdmissible
    let plan :=
      CanonicalStepConstructionPlans.recursiveNifs
        Selected complete.baseProfile complete.allRecipes
    ∃ output :
        SelectedRunning shape publicRingColumns publicFits verifierRows,
      plan.frame.outputs.Decodes (FamilyFor application) assignment
          (.cons output .nil) ∧
        (ConcreteNifsPaperRefinement.PaperAcceptedAtOutput
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              (input.running Vocabulary.Step.selected)
              input.fresh input.nifsProof).materialize
            input.nifsProof.certificate output ∨
          ConcreteNifsPaperRefinement.OccurrenceBoundEvent
            (ConcreteNifsParameters.context
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
              (input.running Vocabulary.Step.selected)
              input.fresh input.nifsProof).materialize
            input.nifsProof.certificate output) := by
  dsimp at physical inputDecoded ⊢
  let complete :=
    ConcreteNifsCompleteApplication.complete application nifs step
      defaultRunningAdmissible
  let plan :=
    CanonicalStepConstructionPlans.recursiveNifs
      Selected complete.baseProfile complete.allRecipes
  have controls :=
    CanonicalStepSoundness.branchControls
      Selected complete.baseProfile complete.allRecipes
      defaultRunningAdmissible complete.directProfile.fieldLaws assignment
      input physical inputDecoded
  have recursiveActive :
      assignment
          (activationColumn SourceOwners.stepBranchPath false) = 1 := by
    simpa [iterationNonzero] using controls.2.2
  have constantOne : assignment plan.frame.one = 1 := by
    rw [plan.oneExact]
    exact physical.1
  have activeOne : assignment plan.frame.active = 1 := by
    rw [plan.activeExact]
    exact recursiveActive
  have receiptRows : Satisfies plan.receipt.rows assignment := by
    apply
      (CanonicalStepSoundness.encoding
        Selected complete.baseProfile complete.allRecipes
        defaultRunningAdmissible).receiptSatisfies
        assignment physical plan.receipt
    exact recursiveNifs_receipt_mem application nifs step
      defaultRunningAdmissible
  have runningDecoded :=
    SchemaBundles.get_decodes
      (FamilyFor application) assignment
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.running
        Selected)
      (CanonicalContexts.Step.input Selected).toSchemaBundles
      (stepInputValues Selected input) inputDecoded
  have freshDecoded :=
    SchemaBundles.get_decodes
      (FamilyFor application) assignment
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.fresh
        Selected)
      (CanonicalContexts.Step.input Selected).toSchemaBundles
      (stepInputValues Selected input) inputDecoded
  have proofDecoded :=
    SchemaBundles.get_decodes
      (FamilyFor application) assignment
      (Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.InputRefs.nifsProof
        Selected)
      (CanonicalContexts.Step.input Selected).toSchemaBundles
      (stepInputValues Selected input) inputDecoded
  have operandsDecoded :
      plan.frame.operands.Decodes (FamilyFor application) assignment
        (.cons (input.running Vocabulary.Step.selected)
          (.cons input.fresh (.cons input.nifsProof .nil))) := by
    rw [CallFrame.operands, plan.contextExact]
    exact ⟨by
      simpa [CanonicalContexts.Step.afterEncodedEquality,
        CanonicalContexts.Step.afterEncode,
        CanonicalContexts.Step.afterFreshPublic,
        CanonicalContexts.Step.afterHash,
        CanonicalContexts.Step.common,
        CanonicalContexts.Step.afterStep,
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.running,
        Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.running,
        Columns.toSchemaBundles_get] using runningDecoded,
      by
        constructor
        · simpa [CanonicalContexts.Step.afterEncodedEquality,
            CanonicalContexts.Step.afterEncode,
            CanonicalContexts.Step.afterFreshPublic,
            CanonicalContexts.Step.afterHash,
            CanonicalContexts.Step.common,
            CanonicalContexts.Step.afterStep,
            Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.fresh,
            Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.fresh,
            Columns.toSchemaBundles_get] using freshDecoded
        · exact ⟨by
            simpa [CanonicalContexts.Step.afterEncodedEquality,
              CanonicalContexts.Step.afterEncode,
              CanonicalContexts.Step.afterFreshPublic,
              CanonicalContexts.Step.afterHash,
              CanonicalContexts.Step.common,
              CanonicalContexts.Step.afterStep,
              Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.RecursiveRefs.nifsProof,
              Nightstream.Implementation.Lowering.FPrimeFixedOne.Step.CommonRefs.nifsProof,
              Columns.toSchemaBundles_get] using proofDecoded,
            trivial⟩⟩
  have selectedRows :
      Satisfies
        (ConcreteNifsActivatedProgram.rows
          application nifs.operational plan.frame)
        assignment := by
    simpa [plan, complete,
      CanonicalStepConstructionPlans.recursiveNifs,
      CanonicalStepPlan.recursiveNifsPlan] using receiptRows
  exact
    ConcreteNifsPaperRefinement.selectedNifs_refinesPaper_or_boundEvent
      application nifs plan.frame assignment
      (input.running Vocabulary.Step.selected) input.fresh input.nifsProof
      constantOne activeOne operandsDecoded selectedRows

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsStepPaperRefinement
