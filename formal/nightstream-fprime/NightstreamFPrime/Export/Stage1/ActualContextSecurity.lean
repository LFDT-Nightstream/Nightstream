import NightstreamFPrime.Export.Stage1.ActualPiDECOutput
import NightstreamFPrime.Export.Stage1.PerApplicationCanonicalPackage
import NightstreamFPrime.Layout.Stage1.PiCCSSecurity
import NightstreamFPrime.Layout.ProductionRelation.AcceptedOpening
import NightstreamFPrime.Lifecycle.Stage1.Terminal

/-!
Bind the arbitrary-assignment step to the canonical verifier context through
the concrete terminal public-input check. Unequal contexts give the existing
named state-hash collision. No representation or context equality is assumed.
The terminal's actual CCS opening supplies the rows and public input.
Interior history extraction remains separate.
-/

namespace NightstreamFPrime.Export.Stage1.ActualContextSecurity

open NightstreamFPrime.Circuit
open NightstreamFPrime.Layout
open NightstreamFPrime.Layout.Stage1
open NightstreamFPrime.Lifecycle
open NightstreamFPrime.Lifecycle.PaperAlgebra
open NightstreamFPrime.Spec
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint
open NightstreamFPrime.Spec.Folding.PiCCS.PaperJoint.PaperLinearAlgebra
open NightstreamFPrime.Spec.HyperNova.Construction2.Paper

variable (application : Lifecycle.Stage1.Application.Program)
  (fits : PerApplicationFixedPoint.FitsTwoPow28 application)
  (commitmentSetup : PerApplicationCanonicalPackage.CommitmentSetup application)

/-- The exact next preimage whose hash is constrained by the selected rows. -/
def decodedNext
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application)) :
    HashPreimage (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application) :=
  ActualHashSlots.nextPreimage (PerApplicationFixedPoint.logicalWidth application)
    (PerApplicationFixedPoint.publicFits application)
    (ActualStep.priorState application assignment)
    (ActualStep.outputState application assignment)

/-- The checked public hash, not an assumed context equality, selects the
canonical context of an arbitrary accepted assignment. -/
theorem selectedRowsAndCheckedPublic_imply_contextOrCollision
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest) (fixed : digest.length = 4)
    (claimedNext : HashPreimage
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (checkedDigest : digest = stateHash
      (publicFits := PerApplicationFixedPoint.publicFits application)
        { claimedNext with verifierKeys := fun _ =>
          PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup })
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    ActualStep.contextKey application assignment =
        PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        { claimedNext with verifierKeys := fun _ =>
          PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup } := by
  let actual := decodedNext application assignment
  let expected := { claimedNext with verifierKeys := fun _ =>
    PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup }
  change actual.verifierKeys functionIndex = expected.verifierKeys functionIndex ∨
    PiCCSSecurity.StateHashCollision actual expected
  have outputHash := ActualHashSlots.selectedRowsAndPublic_imply_outputHash
    application fits assignment digest fixed publicEqual rows
  change digest = stateHash actual at outputHash
  have hashes : stateHash actual = stateHash expected :=
    outputHash.symm.trans checkedDigest
  have keyLength : (actual.verifierKeys functionIndex).length =
      (expected.verifierKeys functionIndex).length := by
    simp [actual, decodedNext, ActualHashSlots.nextPreimage, expected,
      PerApplicationCanonicalPackage.verifierContextDigest,
      PilotProduction.digestWords, PilotValues.digestWords]
  by_cases same : actual.verifierKeys functionIndex = expected.verifierKeys functionIndex
  · exact Or.inl same
  · apply Or.inr
    refine ⟨?_, hashes⟩
    intro encodedEqual
    exact same (StateEncoding.serializePreimage_eq_implies_context_eq
      actual expected keyLength encodedEqual)

/-- Compose context identification with the complete arbitrary-row step.
The relation, application and commitment key are the same selected objects. -/
theorem selectedRowsAndCheckedPublic_imply_stepOrCollision
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (digest : Digest) (fixed : digest.length = 4)
    (claimedNext : HashPreimage
      (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application))
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        encHash (publicFits := PerApplicationFixedPoint.publicFits application) digest)
    (checkedDigest : digest = stateHash
      (publicFits := PerApplicationFixedPoint.publicFits application)
        { claimedNext with verifierKeys := fun _ =>
          PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup })
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment) :
    StepHoldsFor (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup) application
        (ActualStep.input application fits assignment (ActualStep.decodedFresh application assignment)
          (ActualPiDECMessages.proof application fits assignment))
        (ActualStep.output application assignment digest) ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        { claimedNext with verifierKeys := fun _ =>
          PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup } := by
  rcases selectedRowsAndCheckedPublic_imply_contextOrCollision application fits
    commitmentSetup assignment digest fixed claimedNext publicEqual checkedDigest rows with
    context | collision
  · apply Or.inl
    have step := ActualPiDECOutput.selectedRowsAndPublic_imply_step application fits
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      assignment digest publicEqual rows fixed
    simpa only [context] using step
  · exact Or.inr collision

abbrev TerminalPayload := TerminalProof
  (Running (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
    (publicFits := PerApplicationFixedPoint.publicFits application))
  (Stage1.Terminal.RunningWitness (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
    (publicFits := PerApplicationFixedPoint.publicFits application))
  (Fresh (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
    (publicFits := PerApplicationFixedPoint.publicFits application))
  (Stage1.Terminal.FreshWitness (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
    (publicFits := PerApplicationFixedPoint.publicFits application)) slotCount

/-- The actual preimage in the selected recursive terminal public check. -/
def terminalPreimage (statement : TerminalStatement AppState)
    (payload : TerminalPayload application) :
    HashPreimage (logicalWidth := PerApplicationFixedPoint.logicalWidth application)
      (publicFits := PerApplicationFixedPoint.publicFits application) where
  verifierKeys := fun _ => PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup
  iteration := statement.iteration
  z0 := statement.z0
  current := statement.zi
  running := payload.running
  pc := payload.pc

/-- The concrete terminal verifier supplies the checked hash. The assignment
has the actual fresh public input; the caller supplies no context premise,
canonical encoder or digest/preimage equality as an additional assumption. -/
theorem selectedRowsAndTerminal_imply_stepOrCollision
    (assignment : Assignment F (PerApplicationFixedPoint.logicalWidth application))
    (statement : TerminalStatement AppState) (payload : TerminalPayload application)
    (publicEqual : Phi81Relation.projectPublicInput
      (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
        (PerApplicationFixedPoint.publicFits application))
      (Phi81CarrierLayout.extendAssignment 0 assignment) =
        payload.fresh.publicInputs ⟨0, by decide⟩)
    (rows : (PerApplicationFixedPoint.structuralPlan application fits).RowsZero assignment)
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload)) :
    StepHoldsFor (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup) application
        (ActualStep.input application fits assignment (ActualStep.decodedFresh application assignment)
          (ActualPiDECMessages.proof application fits assignment))
        (ActualStep.output application assignment
          (stateHash (terminalPreimage application fits commitmentSetup statement payload))) ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        (terminalPreimage application fits commitmentSetup statement payload) := by
  rcases (Stage1.Terminal.holdsFor_recursive_iff
    (PerApplicationFixedPoint.relation application fits)
    (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
    application statement payload).mp terminal with
    ⟨_statementValid, _pcValid, _positive, publicLink, _runningValid, _freshValid⟩
  let expected := terminalPreimage application fits commitmentSetup statement payload
  change payload.fresh.publicInputs ⟨0, by decide⟩ = encHash (stateHash expected) at publicLink
  have fixed : (stateHash expected).length = 4 := by
    exact StateEncoding.stateHash_length expected
  exact selectedRowsAndCheckedPublic_imply_stepOrCollision application fits commitmentSetup
    assignment (stateHash expected) fixed expected (publicEqual.trans publicLink) rfl rows

/-- Decode the exact terminal fresh opening into the selected logical rows
and public input, including arbitrary values in the carrier padding. -/
theorem terminal_implies_rowsAndPublic
    (statement : TerminalStatement AppState) (payload : TerminalPayload application)
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload)) :
    (PerApplicationFixedPoint.structuralPlan application fits).RowsZero
        (ProductionRelation.Plan.logicalAssignment payload.freshWitness) ∧
      Phi81Relation.projectPublicInput
        (shape := FullShape (PerApplicationFixedPoint.logicalWidth application)
          (PerApplicationFixedPoint.publicFits application))
        (Phi81CarrierLayout.extendAssignment 0
          (ProductionRelation.Plan.logicalAssignment payload.freshWitness)) =
            payload.fresh.publicInputs ⟨0, by decide⟩ := by
  rcases (Stage1.Terminal.holdsFor_recursive_iff
    (PerApplicationFixedPoint.relation application fits)
    (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
    application statement payload).mp terminal with
    ⟨_statementValid, _pcValid, _positive, _publicLink, _runningValid, freshValid⟩
  have publicLogicalFits : ringDegree * publicRingColumns ≤
      PerApplicationFixedPoint.logicalWidth application := by
    unfold PerApplicationFixedPoint.logicalWidth
    rw [ApplicationRetainedGeometry.completeLogicalWidth_eq]
    norm_num [ringDegree, publicRingColumns]
    omega
  exact ProductionRelation.Plan.freshHolds_implies_rowsAndPublic
    (PerApplicationFixedPoint.structuralPlan application fits) fits.carrier
    (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    payload.fresh payload.freshWitness publicLogicalFits freshValid

/-- The terminal's fresh opening supplies the arbitrary assignment, its rows
and its public input. No extra row, padding, encoder, representation, context
or checked-hash premise is needed at this concrete verifier boundary. -/
theorem terminal_implies_stepOrCollision
    (statement : TerminalStatement AppState) (payload : TerminalPayload application)
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload)) :
    let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
    StepHoldsFor (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup) application
        (ActualStep.input application fits assignment (ActualStep.decodedFresh application assignment)
          (ActualPiDECMessages.proof application fits assignment))
        (ActualStep.output application assignment
          (stateHash (terminalPreimage application fits commitmentSetup statement payload))) ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        (terminalPreimage application fits commitmentSetup statement payload) := by
  have accepted := terminal_implies_rowsAndPublic application fits commitmentSetup
    statement payload terminal
  exact selectedRowsAndTerminal_imply_stepOrCollision application fits commitmentSetup
    (ProductionRelation.Plan.logicalAssignment payload.freshWitness) statement payload
    accepted.2 accepted.1 terminal

/-- The accepted terminal opening identifies the complete next preimage,
including its natural iteration, or gives the exact state-hash collision. -/
theorem terminal_implies_preimageOrCollision
    (statement : TerminalStatement AppState) (payload : TerminalPayload application)
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload)) :
    decodedNext application (ProductionRelation.Plan.logicalAssignment payload.freshWitness) =
        terminalPreimage application fits commitmentSetup statement payload ∨
      PiCCSSecurity.StateHashCollision
        (decodedNext application (ProductionRelation.Plan.logicalAssignment payload.freshWitness))
        (terminalPreimage application fits commitmentSetup statement payload) := by
  rcases (Stage1.Terminal.holdsFor_recursive_iff
    (PerApplicationFixedPoint.relation application fits)
    (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
    (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
    application statement payload).mp terminal with
    ⟨valid, pcValid, positive, publicLink, _runningValid, _freshValid⟩
  let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
  let actual := decodedNext application assignment
  let expected := terminalPreimage application fits commitmentSetup statement payload
  have accepted := terminal_implies_rowsAndPublic application fits commitmentSetup
    statement payload terminal
  change payload.fresh.publicInputs ⟨0, by decide⟩ = encHash (stateHash expected) at publicLink
  have hashes := ActualHashSlots.selectedRowsAndPublic_imply_outputHash
    application fits assignment (stateHash expected) (StateEncoding.stateHash_length expected)
    (accepted.2.trans publicLink) accepted.1
  change stateHash expected = stateHash actual at hashes
  change actual = expected ∨ PiCCSSecurity.StateHashCollision actual expected
  by_cases encodedEqual : serializePreimage actual = serializePreimage expected
  · apply Or.inl
    have keyLength : (actual.verifierKeys functionIndex).length =
        (expected.verifierKeys functionIndex).length := by
      simp [actual, decodedNext, ActualHashSlots.nextPreimage, expected, terminalPreimage,
        PerApplicationCanonicalPackage.verifierContextDigest,
        PilotProduction.digestWords, PilotValues.digestWords]
    have counterWord := StateEncoding.serializePreimage_eq_implies_iteration_word_eq
      actual expected keyLength encodedEqual
    have priorBound := (StateDecoder.preimage_wellFormed
      (PerApplicationFixedPoint.logicalWidth application)
      (PerApplicationFixedPoint.publicFits application)
      (ActualStep.priorState application assignment)).2.1
    have iteration := StateEncoding.natWord_successor_eq_below_modulus
      (StateDecoder.iteration (ActualStep.priorState application assignment))
      statement.iteration priorBound positive valid.1 counterWord
    have actualWellFormed : StateEncoding.WellFormed actual := by
      refine ⟨?_, ?_, rfl⟩
      · simp [PilotProduction.FixedPreimage, actual, decodedNext, ActualHashSlots.nextPreimage,
          StateDecoder.preimage, PilotProduction.digestWords, PilotValues.digestWords,
          Stage1.Application.stateWordCount]
      · change StateDecoder.iteration (ActualStep.priorState application assignment) + 1 <
          goldilocksModulus
        rw [iteration]
        exact valid.1
    have expectedWellFormed : StateEncoding.WellFormed expected := by
      refine ⟨⟨?_, valid.2.1, valid.2.2⟩, valid.1, ?_⟩
      · simp [expected, terminalPreimage, PerApplicationCanonicalPackage.verifierContextDigest,
          PilotProduction.digestWords, PilotValues.digestWords]
      · change payload.pc = 1
        change 1 ≤ payload.pc ∧ payload.pc ≤ 1 at pcValid
        omega
    exact StateEncoding.serializePreimage_injective actualWellFormed expectedWellFormed encodedEqual
  · exact Or.inr ⟨encodedEqual, hashes.symm⟩

/-- The complete decoded step reaches the advertised terminal preimage.
This supplies the local step and exact predecessor counter for history extraction. -/
theorem terminal_implies_matchingStepOrCollision
    (statement : TerminalStatement AppState) (payload : TerminalPayload application)
    (terminal : Stage1.Terminal.HoldsFor (PerApplicationFixedPoint.relation application fits)
      (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
      (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup)
      application statement (.recursive payload)) :
    let assignment := ProductionRelation.Plan.logicalAssignment payload.freshWitness
    (StepHoldsFor (PerApplicationFixedPoint.relation application fits)
        (PerApplicationCanonicalPackage.commitmentKey commitmentSetup)
        (PerApplicationCanonicalPackage.verifierContextDigest fits commitmentSetup) application
        (ActualStep.input application fits assignment (ActualStep.decodedFresh application assignment)
          (ActualPiDECMessages.proof application fits assignment))
        (ActualStep.output application assignment
          (stateHash (terminalPreimage application fits commitmentSetup statement payload))) ∧
      decodedNext application assignment =
        terminalPreimage application fits commitmentSetup statement payload) ∨
      PiCCSSecurity.StateHashCollision (decodedNext application assignment)
        (terminalPreimage application fits commitmentSetup statement payload) := by
  rcases terminal_implies_preimageOrCollision application fits commitmentSetup
    statement payload terminal with same | collision
  · rcases terminal_implies_stepOrCollision application fits commitmentSetup
      statement payload terminal with step | collision
    · exact Or.inl ⟨step, same⟩
    · exact False.elim (collision.1 (congrArg serializePreimage same))
  · exact Or.inr collision

end NightstreamFPrime.Export.Stage1.ActualContextSecurity
