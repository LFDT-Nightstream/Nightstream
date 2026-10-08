import NightstreamFPrime.Export.Stage1.ActualContextSecurity
import NightstreamFPrime.Export.Stage1.PerApplicationSecurity
import NightstreamFPrime.Lifecycle.PiDEC.v1_2.OutputWitnessConsumer

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
prior-state public input (all words, so stronger than the decoded digest), the
prior-state link of the transcript coverage contract (the digest absorbed by
PiCCS, the selected verifier context, the running vector read by NIFS, and a
well-formed prior preimage), and the exact NIFS output. The base branch
performs no NIFS call. No input-authentication or output-match premise is added
at this boundary. -/
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
          PiCCSSecurity.PriorLink prior (input.running functionIndex) input.fresh context ∧
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
  · have contextEqual : ActualStep.contextKey application assignment = context :=
      congrArg (fun preimage => preimage.verifierKeys functionIndex) same
    apply Or.inl
    refine ⟨contextEqual, ?_⟩
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
      dsimp only [Lifecycle.machineFor, Lifecycle.machine] at priorPublic
      have digest : ProductionKey.priorDigest input.fresh = stateHash prior := by
        unfold ProductionKey.priorDigest
        rw [priorPublic]
        exact decodeHash_encHash _ (StateEncoding.stateHash_length prior)
      have wellFormed : StateEncoding.WellFormed prior := by
        show StateEncoding.WellFormed
          (priorHashPreimage (Lifecycle.setup relation ajtai context) input)
        rw [← contextEqual, ActualStep.priorHashPreimage_eq_prior]
        exact StateDecoder.preimage_wellFormed _ _ _
      have checked : Nifs.PaperNonInteractive.verify (ProductionKey.key relation ajtai)
          (input.running functionIndex) input.fresh input.nifsProof =
            some (output.runningNext functionIndex) := by
        simpa [Accepts, Lifecycle.setup, Lifecycle.nifsVerifier] using selectedNifs
      have outputSame : output.runningNext functionIndex = payload.running functionIndex :=
        congrArg (fun preimage => preimage.running functionIndex) same
      exact Or.inr ⟨positive, priorPublic, ⟨digest, rfl, rfl, wellFormed⟩,
        checked.trans (congrArg some outputSame)⟩
  · exact Or.inr collision

/-- Transcript coverage for accepted terminals. If two accepted recursive
terminals present equal prover-dependent transcript inputs before a challenge,
they have the same prior preimage, the same NIFS running input, and agree on
everything the transcript absorbs directly. Otherwise one of them is the base
step or a state hash collides. Acceptance supplies the prior-state link. -/
theorem terminal_calls_identify_view_or_collision
    (application : Lifecycle.Stage1.Application.Program)
    (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
    (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)
    (statement statement' : TerminalStatement AppState)
    (payload payload' : TerminalPayload application)
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload))
    (terminal' : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement' (.recursive payload'))
    (challenge : TranscriptCoverage.Challenge) :
    let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
    let assignment' := ProductionRelation.Plan.logicalAssignment payload'.freshWitness
    let input := ActualStep.input application fits assignment
      (ActualStep.decodedFresh application assignment)
      (ActualPiDECMessages.proof application fits assignment)
    let input' := ActualStep.input application fits assignment'
      (ActualStep.decodedFresh application assignment')
      (ActualPiDECMessages.proof application fits assignment')
    let relation := PerApplicationFixedPoint.relation application fits
    let ajtai := PerApplicationCanonicalPackage.commitmentKey commitmentSetup
    let context := PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup
    let prior := priorHashPreimage (Lifecycle.setup relation ajtai context) input
    let prior' := priorHashPreimage (Lifecycle.setup relation ajtai context) input'
    TranscriptCoverage.proverCalls input.fresh input.nifsProof challenge =
        TranscriptCoverage.proverCalls input'.fresh input'.nifsProof challenge →
      input.iteration = 0 ∨ input'.iteration = 0 ∨
        (prior = prior' ∧ input.running functionIndex = input'.running functionIndex ∧
          TranscriptCoverage.AgreeOnAbsorbed input.fresh input'.fresh
            input.nifsProof input'.nifsProof challenge) ∨
        PiCCSSecurity.StateHashCollision prior prior' ∨
        PiCCSSecurity.StateHashCollision (decodedNext application assignment)
          (terminalPreimage application fits commitmentSetup statement payload) ∨
        PiCCSSecurity.StateHashCollision (decodedNext application assignment')
          (terminalPreimage application fits commitmentSetup statement' payload') := by
  intro assignment assignment' input input' relation ajtai context prior prior' same
  rcases terminal_implies_nifsOrBaseOrCollision application fits commitmentSetup
    statement payload terminal with ⟨_, base | ⟨_, _, link, _⟩⟩ | collision
  · exact Or.inl base
  · rcases terminal_implies_nifsOrBaseOrCollision application fits commitmentSetup
      statement' payload' terminal' with ⟨_, base' | ⟨_, _, link', _⟩⟩ | collision'
    · exact Or.inr (Or.inl base')
    · rcases PiCCSSecurity.calls_identify_view_or_collision challenge link link' same with
        ⟨priorEqual, _, runningEqual, agree⟩ | collision
      · exact Or.inr (Or.inr (Or.inl ⟨priorEqual, runningEqual, agree⟩))
      · exact Or.inr (Or.inr (Or.inr (Or.inl collision)))
    · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inr collision'))))
  · exact Or.inr (Or.inr (Or.inr (Or.inr (Or.inl collision))))

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
  · rcases recursive with ⟨positive, _priorPublic, _link, accepted⟩
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
      PiDEC.v1_2.OutputWitnessConsumer.terminalHolds_extracts_parent relation ajtai
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
            (PerApplicationSecurity.productionStrongSet fits commitmentSetup)))) ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        (terminalPreimage application fits commitmentSetup statement payload) := by
  rcases terminal_implies_nifsOrBaseOrCollision application fits commitmentSetup
    statement payload terminal with ⟨context, base | recursive⟩ | collision
  · exact Or.inl ⟨context, Or.inl base⟩
  · rcases recursive with ⟨positive, _priorPublic, _link, accepted⟩
    exact Or.inl ⟨context, Or.inr ⟨positive,
      Nifs.PaperSecurityComposition.accepted_implies_securityOutcome
        (PerApplicationSecurity.canonicalKey fits commitmentSetup) _ _ _
        (payload.running functionIndex)
        (PerApplicationSecurity.productionExtractionAlgebra fits commitmentSetup)
        (PerApplicationSecurity.productionStrongSet fits commitmentSetup) accepted⟩⟩
  · exact Or.inr collision

end NightstreamFPrime.Export.Stage1.ActualTerminalSecurity
