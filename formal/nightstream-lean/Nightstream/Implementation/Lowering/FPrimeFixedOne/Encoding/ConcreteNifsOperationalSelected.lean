import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrenceSemantics

/-!
Contract: connect the frame-static Lean-owned Split-NC rows to the exact
selected operational `PiCCS` occurrence consumed by `ConcreteNifsParameters`.

Whole-frame decoding is the sole source of semantic values.  The selected
constraint polynomial is recovered from codec admissibility, and row
satisfaction supplies the deterministic verifier equations.  No challenge,
certificate equality, acceptance result, or paper event is a premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelected

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrenceSemantics
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

private abbrev FamilyFor (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private theorem blockLaneCertificate_heq_of_input_eq
    {leftInput rightInput : PublicInput shape}
    {domains : Domains}
    (left : Protocol.BlockLane.Certificate leftInput domains)
    (right : Protocol.BlockLane.Certificate rightInput domains)
    (inputEqual : leftInput = rightInput)
    (feEqual : HEq left.fe right.fe)
    (ncEqual : left.nc = right.nc)
    (outputEqual : left.output = right.output) :
    HEq left right := by
  subst rightInput
  apply heq_of_eq
  cases left
  cases right
  simp only at feEqual ncEqual outputEqual
  congr
  exact eq_of_heq feEqual

private theorem feAccepted_transport
    {State : Type}
    {domain : FlatNcDomain}
    {leftInput rightInput : PublicInput shape}
    {leftMachine rightMachine : Transcript.Fe.Machine State}
    {leftState rightState : State}
    {profile : Polynomial.Fe.SupportedProfile shape domain}
    {leftCoins rightCoins : Polynomial.Fe.Coins shape domain}
    {message : OutputMessage shape}
    {leftCertificate : SumCheck.Fe.Certificate leftInput domain}
    {rightCertificate : SumCheck.Fe.Certificate rightInput domain}
    (machineEqual : leftMachine = rightMachine)
    (stateEqual : leftState = rightState)
    (inputEqual : leftInput = rightInput)
    (coinsEqual : leftCoins = rightCoins)
    (certificateEqual : HEq leftCertificate rightCertificate)
    (accepted :
      Fe.Accepted leftMachine leftState profile leftInput leftCoins
        message leftCertificate) :
    Fe.Accepted rightMachine rightState profile rightInput rightCoins
      message rightCertificate := by
  subst rightInput
  have certificateEqual' : leftCertificate = rightCertificate :=
    eq_of_heq certificateEqual
  subst rightCertificate
  subst rightMachine
  subst rightState
  subst rightCoins
  exact accepted

private theorem feFinalState_transport
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
    (Transcript.Fe.derive leftMachine leftState leftCertificate).finalState =
      (Transcript.Fe.derive rightMachine rightState
        rightCertificate).finalState := by
  subst rightInput
  have certificateEqual' : leftCertificate = rightCertificate :=
    eq_of_heq certificateEqual
  subst rightCertificate
  subst rightMachine
  subst rightState
  rfl

theorem feDerived_transport_heq
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
    Transcript.Fe.derive leftMachine leftState leftCertificate =
      Transcript.Fe.derive rightMachine rightState rightCertificate := by
  subst rightInput
  have certificateEqual' : leftCertificate = rightCertificate :=
    eq_of_heq certificateEqual
  subst rightCertificate
  subst rightMachine
  subst rightState
  rfl

private theorem boundaryValue_transport_heq
    {domain : FlatNcDomain}
    {leftInput rightInput : PublicInput shape}
    (leftInitial rightInitial : K)
    (leftPoint rightPoint : Polynomial.Fe.Point shape domain)
    (leftCertificate : SumCheck.Fe.Certificate leftInput domain)
    (rightCertificate : SumCheck.Fe.Certificate rightInput domain)
    (initialEqual : leftInitial = rightInitial)
    (pointEqual : leftPoint = rightPoint)
    (inputEqual : leftInput = rightInput)
    (certificateEqual : HEq leftCertificate rightCertificate) :
    KSplitNcFeRows.boundaryValue leftInitial leftPoint leftCertificate =
      KSplitNcFeRows.boundaryValue rightInitial rightPoint
        rightCertificate := by
  subst rightInput
  have certificateEqual' : leftCertificate = rightCertificate :=
    eq_of_heq certificateEqual
  subst rightCertificate
  subst rightInitial
  subst rightPoint
  rfl

private theorem ncAccepted_transport
    {State : Type}
    {domain : BlockNcDomain}
    {leftMachine rightMachine : Transcript.Nc.Machine State}
    {leftState rightState : State}
    {leftCoins rightCoins : Polynomial.Nc.BlockLane.Mixing.Coins domain}
    {message : OutputMessage shape}
    {leftCertificate rightCertificate :
      Transcript.Nc.BlockLane.Certificate domain}
    (machineEqual : leftMachine = rightMachine)
    (stateEqual : leftState = rightState)
    (coinsEqual : leftCoins = rightCoins)
    (certificateEqual : leftCertificate = rightCertificate)
    (accepted :
      Nc.BlockLane.Accepted leftMachine leftState leftCoins message
        leftCertificate) :
    Nc.BlockLane.Accepted rightMachine rightState rightCoins message
      rightCertificate := by
  subst rightMachine
  subst rightState
  subst rightCoins
  subst rightCertificate
  exact accepted

private theorem ncFinalState_transport
    {State : Type}
    {domain : BlockNcDomain}
    {leftMachine rightMachine : Transcript.Nc.Machine State}
    {leftState rightState : State}
    {leftCertificate rightCertificate :
      Transcript.Nc.BlockLane.Certificate domain}
    (machineEqual : leftMachine = rightMachine)
    (stateEqual : leftState = rightState)
    (certificateEqual : leftCertificate = rightCertificate) :
    (Transcript.Nc.BlockLane.derive
        leftMachine leftState leftCertificate).finalState =
      (Transcript.Nc.BlockLane.derive
        rightMachine rightState rightCertificate).finalState := by
  subst rightMachine
  subst rightState
  subst rightCertificate
  rfl

/-- Equality transport for the complete NC transcript-derived execution. -/
theorem ncDerived_transport
    {State : Type}
    {domain : BlockNcDomain}
    {leftMachine rightMachine : Transcript.Nc.Machine State}
    {leftState rightState : State}
    {leftCertificate rightCertificate :
      Transcript.Nc.BlockLane.Certificate domain}
    (machineEqual : leftMachine = rightMachine)
    (stateEqual : leftState = rightState)
    (certificateEqual : leftCertificate = rightCertificate) :
    Transcript.Nc.BlockLane.derive
        leftMachine leftState leftCertificate =
      Transcript.Nc.BlockLane.derive
        rightMachine rightState rightCertificate := by
  subst rightMachine
  subst rightState
  subst rightCertificate
  rfl

theorem selectedFeMachine_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (transcriptInput :
      KSplitNcTranscript.Input polynomialInput domains)
    (initialClaim : K) :
    feMachine
        (KSplitNcTranscriptSemantics.valueSchedule profile.constants
          assignment transcriptInput)
        initialClaim =
      feMachine
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.piCcsSchedule)
        initialClaim := by
  rw [profile.selectedSchedule]
  rfl

theorem selectedNcMachine_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (assignment : Nat → Nat)
    (transcriptInput :
      KSplitNcTranscript.Input polynomialInput domains) :
    ncMachine
        (KSplitNcTranscriptSemantics.valueSchedule profile.constants
          assignment transcriptInput) =
      ncMachine
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.piCcsSchedule) := by
  rw [profile.selectedSchedule]
  rfl

/-- Satisfaction of the frame-static rows reaches the exact operational
Split-NC relation after restoring the public claims decoded from the proof.
This theorem deliberately stops before changing the verifier-key/input
carrier from the row program's unit serialization to the selected typed
serialization. -/
theorem retargetedAccepted_of_rows
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
        (rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    let restoredInput :=
      KSplitNcStaticInput.withDynamicClaims
        profile.constraintPolynomial proof.piCcsInput
    let restoredRows :=
      KSplitNcStaticInput.retarget proof.piCcsInput
        (input application profile frame)
    Protocol.BlockLane.Accepted
      (fun _ : Unit => restoredInput)
      (KSplitNcTranscriptSemantics.valueSchedule profile.constants
        (numericAssignment (columnMap frame) assignment)
        restoredRows.transcript)
      (KSplitNcTranscriptSemantics.priorState
        (numericAssignment (columnMap frame) assignment)
        restoredRows.transcript)
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      KSplitNcTranscriptSemantics.unitStatement
      (KSplitNcOperational.certificate
        (numericAssignment (columnMap frame) assignment)
        restoredRows.transcript proof.certificate.piCcs.output) := by
  dsimp only
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have admissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have selected :=
    profile.proofAdmissiblePolynomial proof admissible
  have restored :=
    KSplitNcStaticInput.withDynamicClaims_eq
      profile.constraintPolynomial proof.piCcsInput selected
  let restoredRows :=
    KSplitNcStaticInput.retarget proof.piCcsInput
      (input application profile frame)
  have authority :
      KSplitNcEndpoints.DecodedAuthority
        (KSplitNcOperationalRows.endpointInput restoredRows)
        (numericAssignment (columnMap frame) assignment)
        proof.certificate.piCcs.output := by
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
  apply KSplitNcOperationalRows.accepted_of_rows
    (keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
      |>.template.profile)
    profile.constants restoredRows proof.certificate.piCcs.output
    (numericAssignment (columnMap frame) assignment)
  · exact fun column =>
      numericAssignment_canonical (columnMap frame) assignment column
  · exact numericConstantWire application frame assignment constantWire
  · exact authority
  · simpa only [
      ConcreteNifsOperationalOccurrence.rows,
      KSplitNcStaticInput.rows_retarget] using satisfied

/-- The certificate decoded from the physical row columns is the exact raw
certificate carried by the selected proof.  The heterogeneous equality is
only the dependent transport from the restored static input to the proof's
identical codec-bound input. -/
theorem selectedCertificate_heq
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
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    HEq
      (KSplitNcOperational.certificate
        (numericAssignment (columnMap frame) assignment)
        (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
          (transcriptInput application profile frame))
        proof.certificate.piCcs.output)
      proof.certificate.piCcs := by
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have admissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have selected :=
    profile.proofAdmissiblePolynomial proof admissible
  have restored :=
    KSplitNcStaticInput.withDynamicClaims_eq
      profile.constraintPolynomial proof.piCcsInput selected
  apply blockLaneCertificate_heq_of_input_eq _ _ restored
  · exact feCertificate_heq application profile frame assignment
      running fresh proof decoded
  · exact ncCertificate_eq application profile frame assignment
      running fresh proof decoded
  · rfl

/-- The transcript preparation computed by the row occurrence is the exact
selected verifier preparation.  The only cross-carrier fact required is
equality of the complete serialized statement. -/
theorem selectedPreSumcheck_eq
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
        (.cons running (.cons fresh (.cons proof .nil)))) :
    derivePreSumcheck
        (KSplitNcTranscriptSemantics.valueSchedule profile.constants
          (numericAssignment (columnMap frame) assignment)
          (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
            (transcriptInput application profile frame)))
        (KSplitNcTranscriptSemantics.priorState
          (numericAssignment (columnMap frame) assignment)
          (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
            (transcriptInput application profile frame)))
        KSplitNcTranscriptSemantics.unitStatement =
      derivePreSumcheck
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.piCcsSchedule)
        proof.priorState
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize.piCcsStatement := by
  have prior :
      KSplitNcTranscriptSemantics.priorState
          (numericAssignment (columnMap frame) assignment)
          (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
            (transcriptInput application profile frame)) =
        proof.priorState := by
    simpa only [KSplitNcStaticInput.retargetTranscript] using
      priorState_eq application profile frame assignment
        running fresh proof decoded
  have statement :
      KSplitNcTranscriptSemantics.fieldValues
          (numericAssignment (columnMap frame) assignment)
          (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
            (transcriptInput application profile frame)).statementFields =
        profile.serialization.statementFields
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize.piCcsStatement := by
    simpa only [KSplitNcStaticInput.retargetTranscript] using
      statementFields_eq application profile frame assignment
        running fresh proof constantWire decoded
  rw [profile.selectedSchedule, prior]
  unfold derivePreSumcheck
  simp only [
    KSplitNcTranscriptSemantics.valueSchedule,
    KSplitNcTranscriptSemantics.valueSerialization,
    KSplitNcTranscriptSemantics.unitStatement,
    KSplitNcPoseidonSchedule.schedule_bindStatement,
    KSplitNcPoseidonSchedule.schedule_deriveCore,
    KSplitNcPoseidonSchedule.schedule_enterDelayedDomain]
  rw [statement]
  rfl

/-- The deterministic FE row/lane boundary computed by the frame-static
operational view is the boundary computed by the selected verifier.  This is
an equality of verifier computations; no carried boundary column or row
satisfaction is a premise. -/
theorem selectedBoundaryValue_eq
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
        (.cons running (.cons fresh (.cons proof .nil)))) :
    let numeric := numericAssignment (columnMap frame) assignment
    let transcript :=
      KSplitNcStaticInput.retargetTranscript proof.piCcsInput
        (transcriptInput application profile frame)
    let selectedPre :=
      derivePreSumcheck
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.piCcsSchedule)
        proof.priorState
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize.piCcsStatement
    KSplitNcFeRows.boundaryValue
        (KSplitNcTranscriptPhases.semanticFeInitial
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          profile.constants numeric transcript)
        (KSplitNcTranscriptPhases.semanticFeExecution
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          profile.constants numeric transcript).challengePoint
        (KSplitNcTranscriptPhases.feCertificate numeric transcript) =
      KSplitNcFeRows.boundaryValue
        (Polynomial.Fe.initial
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          proof.piCcsInput selectedPre.challenges.feCoins)
        (Transcript.Fe.derive
          (feMachine
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
              |>.template.piCcsSchedule)
            (Polynomial.Fe.initial
              (keys
                Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
                |>.template.profile)
              proof.piCcsInput selectedPre.challenges.feCoins))
          selectedPre.state proof.certificate.piCcs.fe).challengePoint
        proof.certificate.piCcs.fe := by
  dsimp only
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have admissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have polynomialSelected :=
    profile.proofAdmissiblePolynomial proof admissible
  have inputEqual :
      KSplitNcStaticInput.withDynamicClaims
          profile.constraintPolynomial proof.piCcsInput =
        proof.piCcsInput :=
    KSplitNcStaticInput.withDynamicClaims_eq
      profile.constraintPolynomial proof.piCcsInput polynomialSelected
  have preEqual :=
    selectedPreSumcheck_eq application profile frame assignment
      running fresh proof constantWire decoded
  have certificateHeq :
      HEq
        (KSplitNcTranscriptPhases.feCertificate
          (numericAssignment (columnMap frame) assignment)
          (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
            (transcriptInput application profile frame)))
        proof.certificate.piCcs.fe :=
    ConcreteNifsOperationalOccurrenceSemantics.feCertificate_heq
      application profile frame assignment running fresh proof decoded
  let numeric := numericAssignment (columnMap frame) assignment
  let transcript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (transcriptInput application profile frame)
  let operationalPre :=
    KSplitNcTranscriptPhases.semanticPre profile.constants numeric transcript
  let selectedPre :=
    derivePreSumcheck
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.piCcsSchedule)
      proof.priorState
      (ConcreteNifsParameters.context
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
        running fresh proof).materialize.piCcsStatement
  let operationalInitial :=
    KSplitNcTranscriptPhases.semanticFeInitial
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      profile.constants numeric transcript
  let selectedInitial :=
    Polynomial.Fe.initial
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      proof.piCcsInput selectedPre.challenges.feCoins
  have initialEqual : operationalInitial = selectedInitial := by
    change
      Polynomial.Fe.initial
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          (KSplitNcStaticInput.withDynamicClaims
            profile.constraintPolynomial proof.piCcsInput)
          operationalPre.challenges.feCoins =
        Polynomial.Fe.initial
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          proof.piCcsInput selectedPre.challenges.feCoins
    rw [show operationalPre = selectedPre by exact preEqual, inputEqual]
  have machineEqual :
      KSplitNcTranscriptPhases.semanticFeMachine
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          profile.constants numeric transcript =
        feMachine
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.piCcsSchedule)
          selectedInitial := by
    unfold KSplitNcTranscriptPhases.semanticFeMachine
    rw [selectedFeMachine_eq application profile numeric transcript]
    exact congrArg
      (feMachine
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.piCcsSchedule))
      initialEqual
  have stateEqual :
      operationalPre.state = selectedPre.state :=
    congrArg PreSumcheck.state preEqual
  have executionEqual :
      KSplitNcTranscriptPhases.semanticFeExecution
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          profile.constants numeric transcript =
        Transcript.Fe.derive
          (feMachine
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
              |>.template.piCcsSchedule)
            selectedInitial)
          selectedPre.state proof.certificate.piCcs.fe := by
    unfold KSplitNcTranscriptPhases.semanticFeExecution
    exact feDerived_transport_heq machineEqual stateEqual inputEqual
      certificateHeq
  apply boundaryValue_transport_heq
    operationalInitial selectedInitial
    (KSplitNcTranscriptPhases.semanticFeExecution
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      profile.constants numeric transcript).challengePoint
    (Transcript.Fe.derive
      (feMachine
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.piCcsSchedule)
        selectedInitial)
      selectedPre.state proof.certificate.piCcs.fe).challengePoint
    (KSplitNcTranscriptPhases.feCertificate numeric transcript)
    proof.certificate.piCcs.fe
    initialEqual
    (congrArg Transcript.Fe.Derived.challengePoint executionEqual)
    inputEqual
    certificateHeq

/-- Exact selected ΠCCS acceptance transports back to the frame-static
operational carrier. This is the completeness-side inverse of the transport
used by `selectedPiCcsAccepted_of_rows`: it changes no verifier equation and
introduces no row-satisfaction premise. -/
theorem retargetedAccepted_of_selected
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
    (selectedAccepted :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsAccepted
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate) :
    let restoredInput :=
      KSplitNcStaticInput.withDynamicClaims
        profile.constraintPolynomial proof.piCcsInput
    let restoredTranscript :=
      KSplitNcStaticInput.retargetTranscript proof.piCcsInput
        (transcriptInput application profile frame)
    Protocol.BlockLane.Accepted
      (fun _ : Unit => restoredInput)
      (KSplitNcTranscriptSemantics.valueSchedule profile.constants
        (numericAssignment (columnMap frame) assignment)
        restoredTranscript)
      (KSplitNcTranscriptSemantics.priorState
        (numericAssignment (columnMap frame) assignment)
        restoredTranscript)
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      KSplitNcTranscriptSemantics.unitStatement
      (KSplitNcOperational.certificate
        (numericAssignment (columnMap frame) assignment)
        restoredTranscript proof.certificate.piCcs.output) := by
  dsimp only
  let restoredInput :=
    KSplitNcStaticInput.withDynamicClaims
      profile.constraintPolynomial proof.piCcsInput
  let restoredTranscript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (transcriptInput application profile frame)
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
  let selectedStatement :=
    (ConcreteNifsParameters.context
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
      running fresh proof).materialize.piCcsStatement
  let operationalPre :=
    derivePreSumcheck operationalSchedule
      (KSplitNcTranscriptSemantics.priorState numeric restoredTranscript)
      KSplitNcTranscriptSemantics.unitStatement
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
  have preEqual : operationalPre = selectedPre := by
    exact selectedPreSumcheck_eq application profile frame assignment
      running fresh proof constantWire decoded
  have feCertificateEqual :
      HEq operationalCertificate.fe proof.certificate.piCcs.fe := by
    exact feCertificate_heq application profile frame assignment
      running fresh proof decoded
  have ncCertificateEqual :
      operationalCertificate.nc = proof.certificate.piCcs.nc := by
    exact ncCertificate_eq application profile frame assignment
      running fresh proof decoded
  have feCoinsEqual :
      operationalPre.challenges.feCoins =
        selectedPre.challenges.feCoins :=
    congrArg (fun prepared => prepared.challenges.feCoins) preEqual
  have ncCoinsEqual :
      operationalPre.challenges.ncCoins =
        selectedPre.challenges.ncCoins :=
    congrArg (fun prepared => prepared.challenges.ncCoins) preEqual
  have preStateEqual : operationalPre.state = selectedPre.state :=
    congrArg PreSumcheck.state preEqual
  have initialClaimEqual :
      Polynomial.Fe.initial selectedProfile restoredInput
          operationalPre.challenges.feCoins =
        Polynomial.Fe.initial selectedProfile proof.piCcsInput
          selectedPre.challenges.feCoins := by
    rw [inputEqual, feCoinsEqual]
  have feMachineEqual :
      feMachine operationalSchedule
          (Polynomial.Fe.initial selectedProfile restoredInput
            operationalPre.challenges.feCoins) =
        feMachine selectedSchedule
          (Polynomial.Fe.initial selectedProfile proof.piCcsInput
            selectedPre.challenges.feCoins) := by
    exact
      (selectedFeMachine_eq application profile numeric restoredTranscript
        (Polynomial.Fe.initial selectedProfile restoredInput
          operationalPre.challenges.feCoins)).trans
        (congrArg (feMachine selectedSchedule) initialClaimEqual)
  have ncMachineEqual :
      ncMachine operationalSchedule = ncMachine selectedSchedule :=
    selectedNcMachine_eq application profile numeric restoredTranscript
  unfold
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsAccepted
      at selectedAccepted
  unfold Protocol.BlockLane.Accepted at selectedAccepted ⊢
  rcases selectedAccepted with ⟨feAccepted, ncAccepted⟩
  constructor
  · exact feAccepted_transport feMachineEqual.symm preStateEqual.symm
      inputEqual.symm feCoinsEqual.symm feCertificateEqual.symm feAccepted
  · have feFinalStateEqual :=
      feFinalState_transport feMachineEqual.symm preStateEqual.symm
        inputEqual.symm feCertificateEqual.symm
    exact ncAccepted_transport ncMachineEqual.symm feFinalStateEqual
      ncCoinsEqual.symm ncCertificateEqual.symm ncAccepted

/-- The frame-static physical rows enforce the exact ΠCCS acceptance
component used by the selected deterministic NIFS verifier.  The proof
transports only values already recovered from the decoded call frame:
the public polynomial input, transcript preparation, FE/NC machines, and
the two raw certificates. -/
theorem selectedPiCcsAccepted_of_rows
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
        (rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsAccepted
      (ConcreteNifsParameters.context
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
        running fresh proof).materialize
      proof.certificate := by
  let restoredInput :=
    KSplitNcStaticInput.withDynamicClaims
      profile.constraintPolynomial proof.piCcsInput
  let restoredTranscript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (transcriptInput application profile frame)
  let numeric := numericAssignment (columnMap frame) assignment
  let operationalSchedule :=
    KSplitNcTranscriptSemantics.valueSchedule profile.constants numeric
      restoredTranscript
  let selectedSchedule :=
    keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
      |>.template.piCcsSchedule
  let selectedStatement :=
    (ConcreteNifsParameters.context
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
      running fresh proof).materialize.piCcsStatement
  let operationalPre :=
    derivePreSumcheck operationalSchedule
      (KSplitNcTranscriptSemantics.priorState numeric restoredTranscript)
      KSplitNcTranscriptSemantics.unitStatement
  let selectedPre :=
    derivePreSumcheck selectedSchedule proof.priorState selectedStatement
  let operationalCertificate :=
    KSplitNcOperational.certificate numeric restoredTranscript
      proof.certificate.piCcs.output
  have operational :=
    retargetedAccepted_of_rows application profile frame assignment
      running fresh proof constantWire decoded satisfied
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
  have preEqual : operationalPre = selectedPre := by
    exact selectedPreSumcheck_eq application profile frame assignment
      running fresh proof constantWire decoded
  have feCertificateEqual :
      HEq operationalCertificate.fe proof.certificate.piCcs.fe := by
    exact feCertificate_heq application profile frame assignment
      running fresh proof decoded
  have ncCertificateEqual :
      operationalCertificate.nc = proof.certificate.piCcs.nc := by
    exact ncCertificate_eq application profile frame assignment
      running fresh proof decoded
  have feCoinsEqual :
      operationalPre.challenges.feCoins =
        selectedPre.challenges.feCoins :=
    congrArg (fun prepared => prepared.challenges.feCoins) preEqual
  have ncCoinsEqual :
      operationalPre.challenges.ncCoins =
        selectedPre.challenges.ncCoins :=
    congrArg (fun prepared => prepared.challenges.ncCoins) preEqual
  have preStateEqual : operationalPre.state = selectedPre.state :=
    congrArg PreSumcheck.state preEqual
  have initialClaimEqual :
      Polynomial.Fe.initial
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          restoredInput operationalPre.challenges.feCoins =
        Polynomial.Fe.initial
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          proof.piCcsInput selectedPre.challenges.feCoins := by
    rw [inputEqual, feCoinsEqual]
  have feMachineEqual :
      feMachine operationalSchedule
          (Polynomial.Fe.initial
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
              |>.template.profile)
            restoredInput operationalPre.challenges.feCoins) =
        feMachine selectedSchedule
          (Polynomial.Fe.initial
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
              |>.template.profile)
            proof.piCcsInput selectedPre.challenges.feCoins) := by
    exact
      (selectedFeMachine_eq application profile numeric restoredTranscript
        (Polynomial.Fe.initial
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          restoredInput operationalPre.challenges.feCoins)).trans
        (congrArg (feMachine selectedSchedule) initialClaimEqual)
  have ncMachineEqual :
      ncMachine operationalSchedule = ncMachine selectedSchedule := by
    exact selectedNcMachine_eq application profile numeric restoredTranscript
  unfold Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsAccepted
  unfold Protocol.BlockLane.Accepted at operational ⊢
  rcases operational with ⟨feAccepted, ncAccepted⟩
  constructor
  · exact feAccepted_transport feMachineEqual preStateEqual inputEqual
      feCoinsEqual feCertificateEqual feAccepted
  · have feFinalStateEqual :=
      feFinalState_transport feMachineEqual preStateEqual inputEqual
        feCertificateEqual
    exact ncAccepted_transport ncMachineEqual feFinalStateEqual
      ncCoinsEqual ncCertificateEqual ncAccepted

/-- The physical post-output ΠCCS builder decodes to the exact state entering
the selected ΠRLC sampler.  This is a deterministic state refinement, not an
acceptance premise. -/
theorem selectedPiRlcInitialState_eq
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
        (rows application profile frame)
        (numericAssignment (columnMap frame) assignment)) :
    SymbolicDuplexSemantics.decodedBuilder
        (numericAssignment (columnMap frame) assignment)
        (KSplitNcTranscript.outputBuilder
          (transcriptInput application profile frame)) =
      (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate).piRlcInitialState := by
  let restoredInput :=
    KSplitNcStaticInput.withDynamicClaims
      profile.constraintPolynomial proof.piCcsInput
  let restoredTranscript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (transcriptInput application profile frame)
  let restoredRows :=
    KSplitNcStaticInput.retarget proof.piCcsInput
      (input application profile frame)
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
    simpa [restoredRows, numeric, rows,
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
  have canonical : ∀ column, numeric column < goldilocksP :=
    numericAssignment_canonical (columnMap frame) assignment
  have wire : numeric 0 = 1 :=
    numericConstantWire application frame assignment constantWire
  have transcriptValid :
      SymbolicDuplexSemantics.Valid
        restoredTranscript.transcriptBase profile.constants numeric
        (KSplitNcTranscript.outputBuilder restoredTranscript) := by
    exact SymbolicDuplexSemantics.valid_of_satisfied
      restoredTranscript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder restoredTranscript)
      numeric canonical wire transcriptSatisfied
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
      selectedProfile profile.constants numeric wire
      (KSplitNcOperationalRows.endpointInput restoredRows)
      proof.certificate.piCcs.output transcriptValid authority
      endpointSatisfied
  have decodedOutput :=
    KSplitNcTranscriptPhases.decoded_output
      selectedProfile profile.constants numeric wire restoredTranscript
      transcriptValid endpoints.feInitial
  have preEqual : operationalPre = selectedPre := by
    exact selectedPreSumcheck_eq application profile frame assignment
      running fresh proof constantWire decoded
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
  have feMachineEqual :
      feMachine operationalSchedule
          (Polynomial.Fe.initial selectedProfile restoredInput
            operationalPre.challenges.feCoins) =
        feMachine selectedSchedule
          (Polynomial.Fe.initial selectedProfile proof.piCcsInput
            selectedPre.challenges.feCoins) := by
    exact
      (selectedFeMachine_eq application profile numeric restoredTranscript
        (Polynomial.Fe.initial selectedProfile restoredInput
          operationalPre.challenges.feCoins)).trans
        (congrArg (feMachine selectedSchedule) initialClaimEqual)
  have feCertificateEqual :
      HEq operationalCertificate.fe proof.certificate.piCcs.fe := by
    exact feCertificate_heq application profile frame assignment
      running fresh proof decoded
  have feFinalStateEqual :
      (KSplitNcTranscriptPhases.semanticFeExecution
        selectedProfile profile.constants numeric restoredTranscript).finalState =
        (Transcript.Fe.derive
          (feMachine selectedSchedule
            (Polynomial.Fe.initial selectedProfile proof.piCcsInput
              selectedPre.challenges.feCoins))
          selectedPre.state proof.certificate.piCcs.fe).finalState := by
    simpa [KSplitNcTranscriptPhases.semanticFeExecution,
      KSplitNcTranscriptPhases.semanticFeMachine,
      KSplitNcTranscriptPhases.semanticFeInitial,
      KSplitNcTranscriptPhases.semanticPre,
      operationalSchedule, operationalPre, operationalCertificate] using
      (feFinalState_transport feMachineEqual preStateEqual inputEqual
        feCertificateEqual)
  have ncMachineEqual :
      ncMachine operationalSchedule = ncMachine selectedSchedule := by
    exact selectedNcMachine_eq application profile numeric restoredTranscript
  have ncCertificateEqual :
      operationalCertificate.nc = proof.certificate.piCcs.nc := by
    exact ncCertificate_eq application profile frame assignment
      running fresh proof decoded
  have ncFinalStateEqual :
      (KSplitNcTranscriptPhases.semanticNcExecution
        selectedProfile profile.constants numeric restoredTranscript).finalState =
        (Transcript.Nc.BlockLane.derive
          (ncMachine selectedSchedule)
          (Transcript.Fe.derive
            (feMachine selectedSchedule
              (Polynomial.Fe.initial selectedProfile proof.piCcsInput
                selectedPre.challenges.feCoins))
            selectedPre.state proof.certificate.piCcs.fe).finalState
          proof.certificate.piCcs.nc).finalState := by
    simpa [KSplitNcTranscriptPhases.semanticNcExecution,
      operationalSchedule, operationalCertificate] using
      (ncFinalState_transport ncMachineEqual feFinalStateEqual
        ncCertificateEqual)
  have outputFields :
      KSplitNcTranscriptSemantics.fieldValues numeric
          restoredTranscript.outputFields =
        profile.serialization.outputFields
          proof.certificate.piCcs.output := by
    simpa [restoredTranscript, numeric] using
      outputFields_eq application profile frame assignment
        running fresh proof constantWire decoded
  have absorbedOutputEqual :
      operationalSchedule.absorbOutput
          (KSplitNcTranscriptPhases.semanticNcExecution
            selectedProfile profile.constants numeric
            restoredTranscript).finalState
          (KSplitNcTranscriptSemantics.zeroOutput shape) =
        selectedSchedule.absorbOutput
          (Transcript.Nc.BlockLane.derive
            (ncMachine selectedSchedule)
            (Transcript.Fe.derive
              (feMachine selectedSchedule
                (Polynomial.Fe.initial selectedProfile proof.piCcsInput
                  selectedPre.challenges.feCoins))
              selectedPre.state proof.certificate.piCcs.fe).finalState
            proof.certificate.piCcs.nc).finalState
          proof.certificate.piCcs.output := by
    dsimp [operationalSchedule, selectedSchedule]
    rw [profile.selectedSchedule]
    simp only [KSplitNcTranscriptSemantics.valueSchedule,
      KSplitNcPoseidonSchedule.schedule_absorbOutput,
      KSplitNcTranscriptSemantics.valueSerialization]
    rw [outputFields, ncFinalStateEqual]
    simp [selectedSchedule, profile.selectedSchedule]
  calc
    SymbolicDuplexSemantics.decodedBuilder numeric
          (KSplitNcTranscript.outputBuilder restoredTranscript) =
        operationalSchedule.absorbOutput
          (KSplitNcTranscriptPhases.semanticNcExecution
            selectedProfile profile.constants numeric
            restoredTranscript).finalState
          (KSplitNcTranscriptSemantics.zeroOutput shape) := decodedOutput
    _ = selectedSchedule.absorbOutput
          (Transcript.Nc.BlockLane.derive
            (ncMachine selectedSchedule)
            (Transcript.Fe.derive
              (feMachine selectedSchedule
                (Polynomial.Fe.initial selectedProfile proof.piCcsInput
                  selectedPre.challenges.feCoins))
              selectedPre.state proof.certificate.piCcs.fe).finalState
            proof.certificate.piCcs.nc).finalState
          proof.certificate.piCcs.output := absorbedOutputEqual
    _ = (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
          (ConcreteNifsParameters.context
            (keys
              Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
            running fresh proof).materialize
          proof.certificate).piRlcInitialState := by
      rfl

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelected
