import NightstreamFPrime.Export.Stage1.ActualContextSecurity
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputWitnessConsumer

/-!
Connect the arbitrary terminal opening to the actual PiDEC parent witness.
The terminal supplies the child witnesses; decoded step and preimage matching
supply their exact NIFS output link. Interior extraction and probability
bounds remain separate obligations.
-/

namespace NightstreamFPrime.Export.Stage1.ActualTerminalSecurity

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper
open NightstreamFPrime.Spec.HyperNova.NonInteractiveMultiFold
open ActualContextSecurity

/-- Actual terminal acceptance either comes from the base step, supplies the
valid recomposed PiDEC parent of the decoded recursive proof, or exhibits the
named state-hash collision. No output-match or child-opening premise is added. -/
theorem terminal_implies_parentOrBaseOrCollision
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)
    (statement : TerminalStatement AppState) (payload : TerminalPayload application)
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload)) :
    let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
    let input := ActualStep.input application fits assignment
      (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment)
    let relation := PerApplicationFixedPoint.relation application fits
    let ajtai := PerApplicationCanonicalPackage.commitmentKey commitmentSetup
    let key := ProductionKey.key relation ajtai
    input.iteration = 0 ∨
      (0 < input.iteration ∧ ∃ attempt,
        key.piDecAttempt (input.running functionIndex) input.fresh input.nifsProof = some attempt ∧
        CE.Holds (semantics ajtai) productionGlobalParams attempt.parent
          ((PaperAlgebra.piDecAlgebra ajtai).recomposeAssignment
            (payload.runningWitness functionIndex))) ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        (terminalPreimage application fits commitmentSetup statement payload) := by
  let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
  let input := ActualStep.input application fits assignment
    (ActualStep.decodedFresh application assignment)
    (ActualPiDECMessages.proof application fits assignment)
  let output := ActualStep.output application assignment
    (stateHash (terminalPreimage application fits commitmentSetup statement payload))
  let relation := PerApplicationFixedPoint.relation application fits
  let ajtai := PerApplicationCanonicalPackage.commitmentKey commitmentSetup
  let key := ProductionKey.key relation ajtai
  rcases terminal_implies_matchingStepOrCollision application fits commitmentSetup
    statement payload terminal with ⟨step, same⟩ | collision
  · change FixedAugmentedTransition
      (Lifecycle.setup relation ajtai
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup))
      (Lifecycle.machineFor (PerApplicationFixedPoint.publicFits application) application)
      functionIndex input output at step
    rcases step.2.2.2 with base | recursive
    · exact Or.inl base.1
    · rcases recursive with ⟨priorPcValid, positive, _priorPublic, selectedNifs, _unchanged⟩
      have selected : selectedIndex priorPcValid = functionIndex := by
        apply Fin.ext
        have bound := (selectedIndex priorPcValid).isLt
        change (selectedIndex priorPcValid).val < 1 at bound
        change (selectedIndex priorPcValid).val = 0
        omega
      rw [selected] at selectedNifs
      have outputSame : output.runningNext functionIndex = payload.running functionIndex := by
        exact congrArg (fun preimage => preimage.running functionIndex) same
      have accepted : Nifs.PaperNonInteractive.verify key
          (input.running functionIndex) input.fresh input.nifsProof =
            some (payload.running functionIndex) := by
        have checked : Nifs.PaperNonInteractive.verify key
            (input.running functionIndex) input.fresh input.nifsProof =
              some (output.runningNext functionIndex) := by
          simpa [Accepts, Lifecycle.setup, Lifecycle.nifsVerifier, key] using selectedNifs
        exact checked.trans (congrArg some outputSame)
      have checks := (Nifs.PaperNonInteractive.verify_eq_some_iff key
        (input.running functionIndex) input.fresh input.nifsProof
        (payload.running functionIndex)).mp accepted
      rcases (Nifs.PaperNonInteractive.piDecCheck_eq_true_iff key
        (input.running functionIndex) input.fresh input.nifsProof).mp checks.2.1 with
        ⟨attempt, attemptEq, _attemptAccepted⟩
      rcases (Stage1.Terminal.holdsFor_recursive_iff relation ajtai
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
        application statement payload).mp terminal with
        ⟨_valid, _pcValid, _positive, _publicLink, runningValid, freshValid⟩
      exact Or.inr (Or.inl ⟨positive, attempt, attemptEq,
        PiDEC.v1_1.OutputWitnessConsumer.terminalHolds_extracts_parent relation ajtai
          (input.running functionIndex) input.fresh input.nifsProof
          (payload.running functionIndex) attempt attemptEq accepted
          (payload.runningWitness functionIndex) payload.fresh payload.freshWitness
          ⟨runningValid functionIndex, freshValid⟩⟩)
  · exact Or.inr (Or.inr collision)

end NightstreamFPrime.Export.Stage1.ActualTerminalSecurity
