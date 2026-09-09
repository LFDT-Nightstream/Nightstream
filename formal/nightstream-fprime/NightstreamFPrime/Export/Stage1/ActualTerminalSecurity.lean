import NightstreamFPrime.Export.Stage1.ActualContextSecurity
import NightstreamFPrime.Export.Stage1.PerApplicationSecurity
import NightstreamFPrime.Lifecycle.PiDEC.v1_1.OutputWitnessConsumer

/-!
Connect the arbitrary terminal opening to the authenticated NIFS inputs and
the actual PiDEC parent witness. The terminal supplies the child witnesses;
decoded step and preimage matching supply their exact NIFS output link.
Interior extraction and probability bounds remain separate obligations.
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

/-- The accepted terminal opening supplies the selected context, the complete
prior-state public input and the digest absorbed by PiCCS, and the exact NIFS
output. The base branch performs no NIFS call. No input-authentication or
output-match premise is added at this boundary. -/
theorem terminal_implies_nifsOrBaseOrCollision
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
    let context := PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup
    let prior := priorHashPreimage (Lifecycle.setup relation ajtai context) input
    (ActualStep.contextKey application assignment = context ∧
      (input.iteration = 0 ∨
        (0 < input.iteration ∧
          input.fresh.publicInputs ⟨0, by decide⟩ = encHash (stateHash prior) ∧
          ProductionKey.priorDigest input.fresh = stateHash prior ∧
          Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
            (input.running functionIndex) input.fresh input.nifsProof =
              some (payload.running functionIndex)))) ∨
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
  let context := PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup
  let prior := priorHashPreimage (Lifecycle.setup relation ajtai context) input
  rcases terminal_implies_matchingStepOrCollision application fits commitmentSetup
    statement payload terminal with ⟨step, same⟩ | collision
  · apply Or.inl
    refine ⟨congrArg (fun preimage => preimage.verifierKeys functionIndex) same, ?_⟩
    change FixedAugmentedTransition (Lifecycle.setup relation ajtai context)
      (Lifecycle.machineFor (PerApplicationFixedPoint.publicFits application) application)
      functionIndex input output at step
    rcases step.2.2.2 with base | recursive
    · exact Or.inl base.1
    · rcases recursive with ⟨priorPcValid, positive, priorPublic, selectedNifs, _unchanged⟩
      have selected : selectedIndex priorPcValid = functionIndex := by
        apply Fin.ext
        have bound := (selectedIndex priorPcValid).isLt
        change (selectedIndex priorPcValid).val < 1 at bound
        change (selectedIndex priorPcValid).val = 0
        omega
      rw [selected] at selectedNifs
      change input.fresh.publicInputs ⟨0, by decide⟩ = encHash (stateHash prior) at priorPublic
      have digest : ProductionKey.priorDigest input.fresh = stateHash prior := by
        unfold ProductionKey.priorDigest
        rw [priorPublic]
        exact decodeHash_encHash _ (StateEncoding.stateHash_length prior)
      have checked : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
          (input.running functionIndex) input.fresh input.nifsProof =
            some (output.runningNext functionIndex) := by
        simpa [Accepts, Lifecycle.setup, Lifecycle.nifsVerifier] using selectedNifs
      have outputSame : output.runningNext functionIndex = payload.running functionIndex :=
        congrArg (fun preimage => preimage.running functionIndex) same
      exact Or.inr ⟨positive, priorPublic, digest, checked.trans (congrArg some outputSame)⟩
  · exact Or.inr collision

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
  let relation := PerApplicationFixedPoint.relation application fits
  let ajtai := PerApplicationCanonicalPackage.commitmentKey commitmentSetup
  let key := ProductionKey.key relation ajtai
  rcases terminal_implies_nifsOrBaseOrCollision application fits commitmentSetup
    statement payload terminal with ⟨_context, base | recursive⟩ | collision
  · exact Or.inl base
  · rcases recursive with ⟨positive, _priorPublic, _priorDigest, accepted⟩
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

/-- The authenticated terminal boundary consumes the selected NIFS security
result on the actual decoded input and proof. Its cryptographic failure
events retain their existing meaning; this is a deterministic connection. -/
theorem terminal_implies_securityOrCollision
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)
    (statement : TerminalStatement AppState) (payload : TerminalPayload application)
    (lowNorm : Spec.Phi81StrongSet.LowNormInvertibility)
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload)) :
    let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
    let input := ActualStep.input application fits assignment
      (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment)
    (ActualStep.contextKey application assignment =
        PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup ∧
      (input.iteration = 0 ∨
        (0 < input.iteration ∧
          Nifs.PaperSecurityComposition.SecurityOutcome
            (PerApplicationSecurity.canonicalKey fits commitmentSetup)
            (input.running functionIndex) input.fresh input.nifsProof
            (PerApplicationSecurity.productionExtractionAlgebra fits commitmentSetup)
            (PerApplicationSecurity.productionStrongSet fits commitmentSetup lowNorm)))) ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        (terminalPreimage application fits commitmentSetup statement payload) := by
  rcases terminal_implies_nifsOrBaseOrCollision application fits commitmentSetup
    statement payload terminal with ⟨context, base | recursive⟩ | collision
  · exact Or.inl ⟨context, Or.inl base⟩
  · rcases recursive with ⟨positive, _priorPublic, _priorDigest, accepted⟩
    exact Or.inl ⟨context, Or.inr ⟨positive,
      Nifs.PaperSecurityComposition.accepted_implies_securityOutcome
        (PerApplicationSecurity.canonicalKey fits commitmentSetup) _ _ _
        (payload.running functionIndex)
        (PerApplicationSecurity.productionExtractionAlgebra fits commitmentSetup)
        (PerApplicationSecurity.productionStrongSet fits commitmentSetup lowNorm) accepted⟩⟩
  · exact Or.inr collision

end NightstreamFPrime.Export.Stage1.ActualTerminalSecurity
