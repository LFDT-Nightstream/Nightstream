import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedHonest

/-!
Contract: close honest completeness for the selected operational ΠCCS row
occurrence.

The selected verifier supplies the three fixed-phase chains and four endpoint
relations.  This module transports those facts through the numeric witness,
decodes endpoint authority from the actual proof codec, and constructs the
final satisfying assignment for the unchanged Lean-owned row program.

No row equation, verifier result, source-authority record, Rust artifact, or
generated measurement is accepted as a premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedComplete

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

private abbrev FrameFor
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor application) Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

private theorem transcriptBase_le_numericBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame).transcriptBase ≤
      KSplitNcOperationalRows.numericBase
        (ConcreteNifsOperationalOccurrence.input
          application profile frame) := by
  unfold KSplitNcOperationalRows.numericBase
    ConcreteNifsOperationalOccurrence.input
  exact Nat.le_add_right _ _

private theorem decodedTemporary_afterNumeric
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : Nat → Nat)
    (index : Nat) (bounded : index < 5) :
    KSplitNcTranscriptSemantics.decodedColumns
        (ConcreteNifsOperationalHonest.afterNumeric
          application profile frame initial)
        (ConcreteNifsOperationalOccurrence.temporaryK
          (FamilyFor application) frame index) =
      KSplitNcTranscriptSemantics.decodedColumns
        (ConcreteNifsOperationalHonest.afterTranscript
          application profile frame initial)
        (ConcreteNifsOperationalOccurrence.temporaryK
          (FamilyFor application) frame index) := by
  have below :=
    ConcreteNifsOperationalConservation.temporaryK_below_numericBase
      application profile frame index bounded
  unfold KSplitNcTranscriptSemantics.decodedColumns
    Nightstream.Implementation.R1CS.ProjectionProgram.KColumns.value
    Nightstream.Implementation.R1CS.ProjectionProgram.baseAt
  rw [
    ConcreteNifsOperationalHonest.afterNumeric_preserves_before
      application profile frame initial
      (ConcreteNifsOperationalOccurrence.temporaryK
        (FamilyFor application) frame index).c0 below.1,
    ConcreteNifsOperationalHonest.afterNumeric_preserves_before
      application profile frame initial
      (ConcreteNifsOperationalOccurrence.temporaryK
        (FamilyFor application) frame index).c1 below.2]

/-- Installing the numeric fixed-phase witnesses preserves all four exact
selected endpoint relations. -/
theorem selectedEndpoints_afterNumeric
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (encoded :
      frame.operands.Encodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (constantWire : assignment frame.one = 1) :
    let initial :=
      ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
        application profile frame assignment running fresh proof
    let input :=
      KSplitNcStaticInput.retarget proof.piCcsInput
        (ConcreteNifsOperationalOccurrence.input application profile frame)
    KSplitNcOperational.EndpointAgrees
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      profile.constants
      (ConcreteNifsOperationalHonest.afterNumeric
        application profile frame initial)
      input.transcript proof.certificate.piCcs.output := by
  dsimp only
  let initial :=
    ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
      application profile frame assignment running fresh proof
  let input :=
    KSplitNcStaticInput.retarget proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.input application profile frame)
  let transcript := input.transcript
  let transcriptAssignment :=
    ConcreteNifsOperationalHonest.afterTranscript
      application profile frame initial
  let numericAssignment :=
    ConcreteNifsOperationalHonest.afterNumeric
      application profile frame initial
  let selectedProfile :=
    (keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
      |>.template.profile)
  have endpoints :
      KSplitNcOperational.EndpointAgrees selectedProfile profile.constants
        transcriptAssignment transcript proof.certificate.piCcs.output := by
    simpa only [initial, input, transcript, transcriptAssignment,
      selectedProfile] using
      ConcreteNifsOperationalSelectedHonest.selectedEndpoints_afterTranscript
        application profile frame assignment running fresh proof fits encoded
        constantWire
  have sources :=
    ConcreteNifsOperationalSelectedHonest.retargetedTranscript_inPrefix
      application profile frame proof
  have agree :
      ∀ column, column < transcript.transcriptBase →
        transcriptAssignment column = numericAssignment column := by
    intro column below
    symm
    apply ConcreteNifsOperationalHonest.afterNumeric_preserves_before
      application profile frame initial column
    exact Nat.lt_of_lt_of_le below
      (transcriptBase_le_numericBase application profile frame)
  have preEqual :=
    KSplitNcTranscriptAssignmentInvariant.semanticPre_eq
      profile.constants transcriptAssignment numericAssignment transcript
      sources agree
  have feEqual :=
    KSplitNcTranscriptAssignmentInvariant.semanticFeExecution_eq
      selectedProfile profile.constants transcriptAssignment numericAssignment
      transcript sources agree
  have ncEqual :=
    KSplitNcTranscriptAssignmentInvariant.semanticNcExecution_eq
      selectedProfile profile.constants transcriptAssignment numericAssignment
      transcript sources agree
  have initialEqual :
      KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
          profile.constants transcriptAssignment transcript =
        KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
          profile.constants numericAssignment transcript := by
    unfold KSplitNcTranscriptPhases.semanticFeInitial
    rw [preEqual]
  have temporary0 :=
    decodedTemporary_afterNumeric application profile frame initial 0
      (by omega)
  have temporary2 :=
    decodedTemporary_afterNumeric application profile frame initial 2
      (by omega)
  have temporary3 :=
    decodedTemporary_afterNumeric application profile frame initial 3
      (by omega)
  have temporary4 :=
    decodedTemporary_afterNumeric application profile frame initial 4
      (by omega)
  refine {
    feInitial := ?_
    feTerminal := ?_
    ncInitial := ?_
    ncTerminal := ?_
  }
  · calc
      KSplitNcTranscriptSemantics.decodedColumns numericAssignment
          transcript.fe.initial =
        KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
          transcript.fe.initial := by
            simpa only [input, transcript,
              KSplitNcStaticInput.retarget,
              KSplitNcStaticInput.retargetTranscript,
              ConcreteNifsOperationalOccurrence.input,
              ConcreteNifsOperationalOccurrence.transcriptInput] using
              temporary0
      _ =
        KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
          profile.constants transcriptAssignment transcript :=
        endpoints.feInitial
      _ =
        KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
          profile.constants numericAssignment transcript :=
        initialEqual
  · calc
      KSplitNcTranscriptSemantics.decodedColumns numericAssignment
          transcript.fe.terminal =
        KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
          transcript.fe.terminal := by
            simpa only [input, transcript,
              KSplitNcStaticInput.retarget,
              KSplitNcStaticInput.retargetTranscript,
              ConcreteNifsOperationalOccurrence.input,
              ConcreteNifsOperationalOccurrence.transcriptInput] using
              temporary2
      _ =
        Polynomial.Fe.terminalFromMessage selectedProfile
          (KSplitNcStaticInput.withDynamicClaims
            profile.constraintPolynomial proof.piCcsInput)
          (KSplitNcTranscriptPhases.semanticPre profile.constants
            transcriptAssignment transcript).challenges.feCoins
          (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
            profile.constants transcriptAssignment transcript).challengePoint
          proof.certificate.piCcs.output :=
        endpoints.feTerminal
      _ =
        Polynomial.Fe.terminalFromMessage selectedProfile
          (KSplitNcStaticInput.withDynamicClaims
            profile.constraintPolynomial proof.piCcsInput)
          (KSplitNcTranscriptPhases.semanticPre profile.constants
            numericAssignment transcript).challenges.feCoins
          (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
            profile.constants numericAssignment transcript).challengePoint
          proof.certificate.piCcs.output := by
            rw [preEqual, feEqual]
  · calc
      KSplitNcTranscriptSemantics.decodedColumns numericAssignment
          transcript.nc.initial =
        KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
          transcript.nc.initial := by
            simpa only [input, transcript,
              KSplitNcStaticInput.retarget,
              KSplitNcStaticInput.retargetTranscript,
              ConcreteNifsOperationalOccurrence.input,
              ConcreteNifsOperationalOccurrence.transcriptInput] using
              temporary3
      _ = Polynomial.Nc.BlockLane.InitialSum.claimedInitial :=
        endpoints.ncInitial
  · calc
      KSplitNcTranscriptSemantics.decodedColumns numericAssignment
          transcript.nc.terminal =
        KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
          transcript.nc.terminal := by
            simpa only [input, transcript,
              KSplitNcStaticInput.retarget,
              KSplitNcStaticInput.retargetTranscript,
              ConcreteNifsOperationalOccurrence.input,
              ConcreteNifsOperationalOccurrence.transcriptInput] using
              temporary4
      _ =
        Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          proof.certificate.piCcs.output
          (KSplitNcTranscriptPhases.semanticPre profile.constants
            transcriptAssignment transcript).challenges.ncCoins
          (KSplitNcTranscriptPhases.semanticNcExecution selectedProfile
            profile.constants transcriptAssignment transcript).challengePoint :=
        endpoints.ncTerminal
      _ =
        Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          proof.certificate.piCcs.output
          (KSplitNcTranscriptPhases.semanticPre profile.constants
            numericAssignment transcript).challenges.ncCoins
          (KSplitNcTranscriptPhases.semanticNcExecution selectedProfile
            profile.constants numericAssignment transcript).challengePoint := by
            rw [preEqual, ncEqual]

/-- The final numeric witness still decodes every endpoint authority directly
from the selected proof operand. -/
theorem selectedAuthority_afterNumeric
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (encoded :
      frame.operands.Encodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    let initial :=
      ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
        application profile frame assignment running fresh proof
    let input :=
      KSplitNcStaticInput.retarget proof.piCcsInput
        (ConcreteNifsOperationalOccurrence.input application profile frame)
    KSplitNcEndpoints.DecodedAuthority
      (KSplitNcOperationalRows.endpointInput input)
      (ConcreteNifsOperationalHonest.afterNumeric
        application profile frame initial)
      proof.certificate.piCcs.output := by
  dsimp only
  let physical :=
    ConcreteNifsOperationalSelectedHonest.seededAssignment
      application profile frame assignment running fresh proof
  let initial :=
    ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
      application profile frame assignment running fresh proof
  let input :=
    KSplitNcStaticInput.retarget proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.input application profile frame)
  let numericAssignment :=
    ConcreteNifsOperationalHonest.afterNumeric
      application profile frame initial
  have physicalEncoded :
      frame.operands.Encodes (FamilyFor application) physical
        (.cons running (.cons fresh (.cons proof .nil))) := by
    exact ConcreteNifsOperationalSelectedHonest.seeded_encodes
      application profile frame assignment running fresh proof fits encoded
  have physicalDecoded :
      frame.operands.Decodes (FamilyFor application) physical
        (.cons running (.cons fresh (.cons proof .nil))) := by
    exact frame.operands.decodes_of_encodes
      (FamilyFor application) physical
      (.cons running (.cons fresh (.cons proof .nil))) physicalEncoded
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame physical running fresh proof
      physicalDecoded
  have authorityInitial :
      KSplitNcEndpoints.DecodedAuthority
        (KSplitNcOperationalRows.endpointInput input)
        initial proof.certificate.piCcs.output := by
    refine {
      priorPoint := ?_
      claimedYRing := ?_
      outputYRing := ?_
      outputYZcol := ?_
    }
    · intro coordinate
      simpa only [physical, initial, input,
        ConcreteNifsOperationalSelectedHonest.seededNumericAssignment,
        KSplitNcOperationalRows.endpointInput,
        KSplitNcStaticInput.retarget,
        ConcreteNifsOperationalOccurrence.input,
        ConcreteNifsOperationalFrame.authorityColumns] using
        ConcreteNifsOperationalFrame.decodedView
          (FamilyFor application) frame
          (profile.endpointViews.priorPoint coordinate)
          physical proof proofDecoded
    · intro runningIndex matrix lane
      simpa only [physical, initial, input,
        ConcreteNifsOperationalSelectedHonest.seededNumericAssignment,
        KSplitNcOperationalRows.endpointInput,
        KSplitNcStaticInput.retarget,
        ConcreteNifsOperationalOccurrence.input,
        ConcreteNifsOperationalFrame.authorityColumns] using
        ConcreteNifsOperationalFrame.decodedView
          (FamilyFor application) frame
          (profile.endpointViews.claimedYRing runningIndex matrix lane)
          physical proof proofDecoded
    · intro source matrix lane
      simpa only [physical, initial, input,
        ConcreteNifsOperationalSelectedHonest.seededNumericAssignment,
        KSplitNcOperationalRows.endpointInput,
        KSplitNcStaticInput.retarget,
        ConcreteNifsOperationalOccurrence.input,
        ConcreteNifsOperationalFrame.authorityColumns] using
        ConcreteNifsOperationalFrame.decodedView
          (FamilyFor application) frame
          (profile.endpointViews.outputYRing source matrix lane)
          physical proof proofDecoded
    · intro source lane
      simpa only [physical, initial, input,
        ConcreteNifsOperationalSelectedHonest.seededNumericAssignment,
        KSplitNcOperationalRows.endpointInput,
        KSplitNcStaticInput.retarget,
        ConcreteNifsOperationalOccurrence.input,
        ConcreteNifsOperationalFrame.authorityColumns] using
        ConcreteNifsOperationalFrame.decodedView
          (FamilyFor application) frame
          (profile.endpointViews.outputYZcol source lane)
          physical proof proofDecoded
  have preserved :
      ∀ column,
        column <
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame).transcriptBase →
          numericAssignment column = initial column := by
    intro column below
    change
      ConcreteNifsOperationalHonest.afterNumeric
          application profile frame initial column =
        initial column
    rw [ConcreteNifsOperationalHonest.afterNumeric_preserves_before
      application profile frame initial column
      (Nat.lt_of_lt_of_le below
        (transcriptBase_le_numericBase application profile frame))]
    exact ConcreteNifsOperationalHonest.afterTranscript_preserves_before
      application profile frame initial column below
  have transportView
      {value :
        SelectedProof shape TranscriptState publicRingColumns publicFits
            verifierRows →
          Nightstream.SuperNeo.Concrete.K}
      (view :
        PaperNifsCodecProjection.KView
          ((FamilyFor application).codecFor (.data .nifsProof)) value) :
      KPointEquality.decoded numericAssignment
          (ConcreteNifsOperationalFrame.proofLocation
            (FamilyFor application) frame view).carried =
        KPointEquality.decoded initial
          (ConcreteNifsOperationalFrame.proofLocation
            (FamilyFor application) frame view).carried := by
    exact KSplitNcEndpointsSemanticHonest.decoded_eq_of_preserved
      initial numericAssignment
      (ConcreteNifsOperationalFrame.proofLocation
        (FamilyFor application) frame view).carried
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame).transcriptBase
      (ConcreteNifsEndpointConservation.proofView_below_transcriptBase
        application profile frame view)
      preserved
  refine {
    priorPoint := ?_
    claimedYRing := ?_
    outputYRing := ?_
    outputYZcol := ?_
  }
  · intro coordinate
    simpa only [input, numericAssignment,
      KSplitNcOperationalRows.endpointInput,
      KSplitNcStaticInput.retarget,
      ConcreteNifsOperationalOccurrence.input,
      ConcreteNifsOperationalFrame.authorityColumns] using
      (transportView (profile.endpointViews.priorPoint coordinate)).trans
        (authorityInitial.priorPoint coordinate)
  · intro runningIndex matrix lane
    simpa only [input, numericAssignment,
      KSplitNcOperationalRows.endpointInput,
      KSplitNcStaticInput.retarget,
      ConcreteNifsOperationalOccurrence.input,
      ConcreteNifsOperationalFrame.authorityColumns] using
      (transportView
        (profile.endpointViews.claimedYRing
          runningIndex matrix lane)).trans
        (authorityInitial.claimedYRing runningIndex matrix lane)
  · intro source matrix lane
    simpa only [input, numericAssignment,
      KSplitNcOperationalRows.endpointInput,
      KSplitNcStaticInput.retarget,
      ConcreteNifsOperationalOccurrence.input,
      ConcreteNifsOperationalFrame.authorityColumns] using
      (transportView
        (profile.endpointViews.outputYRing source matrix lane)).trans
        (authorityInitial.outputYRing source matrix lane)
  · intro source lane
    simpa only [input, numericAssignment,
      KSplitNcOperationalRows.endpointInput,
      KSplitNcStaticInput.retarget,
      ConcreteNifsOperationalOccurrence.input,
      ConcreteNifsOperationalFrame.authorityColumns] using
      (transportView
        (profile.endpointViews.outputYZcol source lane)).trans
        (authorityInitial.outputYZcol source lane)

/-- One accepted selected ΠCCS proof has a satisfying assignment for the
complete Lean-owned operational row occurrence. -/
theorem selectedRows_honest
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (assignment : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (encoded :
      frame.operands.Encodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil))))
    (constantWire : assignment frame.one = 1)
    (selectedAccepted :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsAccepted
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate) :
    let initial :=
      ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
        application profile frame assignment running fresh proof
    Satisfies
      (ConcreteNifsOperationalOccurrence.rows application profile frame)
      (ConcreteNifsOperationalHonest.retargetedWitness
        application profile frame proof.piCcsInput initial) := by
  dsimp only
  let initial :=
    ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
      application profile frame assignment running fresh proof
  have chains :=
    ConcreteNifsOperationalSelectedHonest.selectedChains_afterTranscript
      application profile frame assignment running fresh proof fits encoded
      constantWire selectedAccepted
  have authority :=
    selectedAuthority_afterNumeric application profile frame assignment
      running fresh proof fits encoded
  have endpoints :=
    selectedEndpoints_afterNumeric application profile frame assignment
      running fresh proof fits encoded constantWire
  exact
    ConcreteNifsOperationalHonest.rows_honest_retargeted_of_semantics
      application profile frame proof.piCcsInput initial
      proof.certificate.piCcs.output
      (ConcreteNifsOperationalSelectedHonest.seededNumericAssignment_residues
        application profile frame assignment running fresh proof)
      (ConcreteNifsOperationalSelectedHonest.seededNumericAssignment_constantWire
        application profile frame assignment running fresh proof fits
        constantWire)
      chains.1 chains.2.1 chains.2.2 authority endpoints

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedComplete
