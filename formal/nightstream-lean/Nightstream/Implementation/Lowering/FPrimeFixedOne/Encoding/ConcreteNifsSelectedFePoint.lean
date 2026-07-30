import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelected

/-!
Contract: identify the FE row point carried by the Lean-owned operational
transcript with the exact point computed by the selected ConcretePhi81
verifier.

The equality is derived from whole-frame decoding and satisfaction of the
operational PiCCS rows.  It does not accept a challenge point or semantic
verifier result from the caller.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedFePoint

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane

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

private theorem feChallengePoint_transport
    {State : Type}
    {domain : FlatNcDomain}
    {leftInput rightInput : PublicInput shape}
    {leftMachine rightMachine : Transcript.Fe.Machine State}
    {leftState rightState : State}
    {leftCertificate : SumCheck.Fe.Certificate leftInput domain}
    {rightCertificate : SumCheck.Fe.Certificate rightInput domain}
    (machineEqual : leftMachine = rightMachine)
    (stateEqual : leftState = rightState)
    (inputEqual : leftInput = rightInput)
    (certificateEqual : HEq leftCertificate rightCertificate) :
    (Transcript.Fe.derive
      leftMachine leftState leftCertificate).challengePoint =
      (Transcript.Fe.derive
        rightMachine rightState rightCertificate).challengePoint := by
  subst rightInput
  have certificateEqual' : leftCertificate = rightCertificate :=
    eq_of_heq certificateEqual
  subst rightCertificate
  subst rightMachine
  subst rightState
  rfl

/-- The physical FE challenge columns decode to the exact row/lane point
computed by the selected verifier execution. -/
theorem selectedFePoint_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {schema : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) schema (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) schema (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) schema (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (constantWire : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (satisfied :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    KSplitNcTranscriptPhases.decodedFePoint
        (numericAssignment (columnMap frame) assignment)
        (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
          (ConcreteNifsOperationalOccurrence.transcriptInput
            application profile frame)) =
      (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate).piCcs.fePoint := by
  let restoredInput :=
    KSplitNcStaticInput.withDynamicClaims
      profile.constraintPolynomial proof.piCcsInput
  let restoredTranscript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame)
  let restoredRows :=
    KSplitNcStaticInput.retarget proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.input application profile frame)
  let numeric := numericAssignment (columnMap frame) assignment
  let operationalSchedule :=
    KSplitNcTranscriptSemantics.valueSchedule profile.constants numeric
      restoredTranscript
  let selectedSchedule :=
    keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
      |>.template.piCcsSchedule
  let selectedProfile :=
    keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
      |>.template.profile
  let operationalPre :=
    derivePreSumcheck operationalSchedule
      (KSplitNcTranscriptSemantics.priorState numeric restoredTranscript)
      KSplitNcTranscriptSemantics.unitStatement
  let selectedStatement :=
    (ConcreteNifsParameters.context
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
      running fresh proof).materialize.piCcsStatement
  let selectedPre :=
    derivePreSumcheck selectedSchedule proof.priorState selectedStatement
  let operationalCertificate :=
    KSplitNcOperational.certificate numeric restoredTranscript
      proof.certificate.piCcs.output
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have admissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have polynomialSelected :=
    profile.proofAdmissiblePolynomial proof admissible
  have inputEqual : restoredInput = proof.piCcsInput := by
    exact KSplitNcStaticInput.withDynamicClaims_eq
      profile.constraintPolynomial proof.piCcsInput polynomialSelected
  have restoredSatisfied :
      Satisfies
        (KSplitNcOperationalRows.rows profile.constants restoredRows)
        numeric := by
    simpa [restoredRows, numeric,
      ConcreteNifsOperationalOccurrence.rows,
      KSplitNcStaticInput.rows_retarget] using satisfied
  have transcriptSatisfied :
      Satisfies
        (KSplitNcOperationalRows.transcriptRows
          profile.constants restoredRows)
        numeric :=
    KSplitNcOperationalRows.satisfies_group
      profile.constants restoredRows numeric restoredSatisfied _ (by
        simp [KSplitNcOperationalRows.rowGroups])
  have endpointSatisfied :
      Satisfies (KSplitNcOperationalRows.endpointRows restoredRows) numeric :=
    KSplitNcOperationalRows.satisfies_group
      profile.constants restoredRows numeric restoredSatisfied _ (by
        simp [KSplitNcOperationalRows.rowGroups])
  have transcriptValid :
      SymbolicDuplexSemantics.Valid
        restoredTranscript.transcriptBase profile.constants numeric
        (KSplitNcTranscript.outputBuilder restoredTranscript) := by
    exact SymbolicDuplexSemantics.valid_of_satisfied
      restoredTranscript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder restoredTranscript)
      numeric (numericAssignment_canonical (columnMap frame) assignment)
      (ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
        application frame assignment constantWire)
      transcriptSatisfied
  have authority :
      KSplitNcEndpoints.DecodedAuthority
        (KSplitNcOperationalRows.endpointInput restoredRows)
        numeric proof.certificate.piCcs.output := by
    constructor
    · intro coordinate
      exact ConcreteNifsOperationalFrame.decodedView
        (FamilyFor application) frame
        (profile.endpointViews.priorPoint coordinate)
        assignment proof proofDecoded
    · intro runningIndex matrix lane
      exact ConcreteNifsOperationalFrame.decodedView
        (FamilyFor application) frame
        (profile.endpointViews.claimedYRing runningIndex matrix lane)
        assignment proof proofDecoded
    · intro source matrix lane
      exact ConcreteNifsOperationalFrame.decodedView
        (FamilyFor application) frame
        (profile.endpointViews.outputYRing source matrix lane)
        assignment proof proofDecoded
    · intro source lane
      exact ConcreteNifsOperationalFrame.decodedView
        (FamilyFor application) frame
        (profile.endpointViews.outputYZcol source lane)
        assignment proof proofDecoded
  have endpoints :=
    KSplitNcEndpoints.endpointAgrees_of_rows
      selectedProfile profile.constants numeric
      (ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
        application frame assignment constantWire)
      (KSplitNcOperationalRows.endpointInput restoredRows)
      proof.certificate.piCcs.output transcriptValid authority
      endpointSatisfied
  have feReplay :=
    KSplitNcTranscriptPhases.decoded_fe
      selectedProfile profile.constants numeric
      (ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
        application frame assignment constantWire)
      restoredTranscript transcriptValid endpoints.feInitial
  have preEqual : operationalPre = selectedPre :=
    ConcreteNifsOperationalSelected.selectedPreSumcheck_eq
      application profile frame assignment running fresh proof
      constantWire decoded
  have preStateEqual :
      operationalPre.state = selectedPre.state :=
    congrArg PreSumcheck.state preEqual
  have feCoinsEqual :
      operationalPre.challenges.feCoins =
        selectedPre.challenges.feCoins :=
    congrArg (fun prepared => prepared.challenges.feCoins) preEqual
  have initialClaimEqual :
      Polynomial.Fe.initial selectedProfile restoredInput
          operationalPre.challenges.feCoins =
        Polynomial.Fe.initial selectedProfile proof.piCcsInput
          selectedPre.challenges.feCoins := by
    rw [inputEqual, feCoinsEqual]
  have machineEqual :
      feMachine operationalSchedule
          (Polynomial.Fe.initial selectedProfile restoredInput
            operationalPre.challenges.feCoins) =
        feMachine selectedSchedule
          (Polynomial.Fe.initial selectedProfile proof.piCcsInput
            selectedPre.challenges.feCoins) := by
    have scheduleMachine :
        feMachine operationalSchedule
            (Polynomial.Fe.initial selectedProfile restoredInput
              operationalPre.challenges.feCoins) =
          feMachine selectedSchedule
            (Polynomial.Fe.initial selectedProfile restoredInput
              operationalPre.challenges.feCoins) := by
      dsimp [operationalSchedule, selectedSchedule]
      rw [profile.selectedSchedule]
      rfl
    exact scheduleMachine.trans
      (congrArg (feMachine selectedSchedule) initialClaimEqual)
  have certificateEqual :
      HEq operationalCertificate.fe proof.certificate.piCcs.fe :=
    ConcreteNifsOperationalOccurrenceSemantics.feCertificate_heq
      application profile frame assignment running fresh proof decoded
  have pointEqual :=
    feChallengePoint_transport machineEqual preStateEqual inputEqual
      certificateEqual
  exact feReplay.point.trans pointEqual

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedFePoint
