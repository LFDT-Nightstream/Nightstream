import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsClaimedValuesHonest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalHonest
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelected
import Nightstream.Implementation.R1CS.Canonical.KSplitNcTranscriptAssignmentInvariant

/-!
Contract: construct the selected ΠCCS operational witness from one honest
physical call frame.

The five leading extension-field temporaries are computed from the selected
verifier itself.  In particular the FE row/lane boundary is the deterministic
prefix result, not a caller-provided claim.  The write preserves the complete
visible call frame before the operational transcript and arithmetic witnesses
are installed.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedHonest

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
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc.Verifier.Protocol.TranscriptAuthority.BlockLane
open Nightstream.SuperNeo.SumCheck.Finite

private abbrev TranscriptState := Poseidon2Duplex.State
private abbrev ops := ConcreteCarrier.extensionOps.toOps

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

/-- Exact selected verifier values for the five claimed-chain temporaries. -/
def values
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    ConcreteNifsClaimedValuesHonest.Values :=
  let selectedKeys :=
    keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
  let context :=
    ConcreteNifsParameters.context selectedKeys running fresh proof
  let prepared :=
    derivePreSumcheck selectedKeys.template.piCcsSchedule proof.priorState
      context.materialize.piCcsStatement
  let feInitial :=
    Polynomial.Fe.initial selectedKeys.template.profile proof.piCcsInput
      prepared.challenges.feCoins
  let feExecution :=
    Transcript.Fe.derive
      (feMachine selectedKeys.template.piCcsSchedule feInitial)
      prepared.state proof.certificate.piCcs.fe
  let ncExecution :=
    Transcript.Nc.BlockLane.derive
      (ncMachine selectedKeys.template.piCcsSchedule)
      feExecution.finalState proof.certificate.piCcs.nc
  {
    feInitial := feInitial
    feBoundary :=
      KSplitNcFeRows.boundaryValue feInitial feExecution.challengePoint
        proof.certificate.piCcs.fe
    feTerminal :=
      Polynomial.Fe.terminalFromMessage selectedKeys.template.profile
        proof.piCcsInput prepared.challenges.feCoins
        feExecution.challengePoint proof.certificate.piCcs.output
    ncInitial := Polynomial.Nc.BlockLane.InitialSum.claimedInitial
    ncTerminal :=
      Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
        proof.certificate.piCcs.output prepared.challenges.ncCoins
        ncExecution.challengePoint
  }

/-- Exact agreement between the five claimed-value columns and the
verifier-derived operational computation they abbreviate. -/
structure ClaimedValuesAgree
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (selectedProfile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (assignment : Nat → Nat)
    (transcript : KSplitNcTranscript.Input polynomialInput domains)
    (message : OutputMessage shape)
    (claims : ConcreteNifsClaimedValuesHonest.Values) : Prop where
  feInitial :
    KSplitNcTranscriptPhases.semanticFeInitial selectedProfile constants
        assignment transcript =
      claims.feInitial
  feBoundary :
    KSplitNcFeRows.boundaryValue
        (KSplitNcTranscriptPhases.semanticFeInitial selectedProfile constants
          assignment transcript)
        (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile constants
          assignment transcript).challengePoint
        (KSplitNcTranscriptPhases.feCertificate assignment transcript) =
      claims.feBoundary
  feTerminal :
    Polynomial.Fe.terminalFromMessage selectedProfile polynomialInput
        (KSplitNcTranscriptPhases.semanticPre constants assignment
          transcript).challenges.feCoins
        (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile constants
          assignment transcript).challengePoint
        message =
      claims.feTerminal
  ncInitial :
    Polynomial.Nc.BlockLane.InitialSum.claimedInitial = claims.ncInitial
  ncTerminal :
    Polynomial.Nc.BlockLane.Terminal.terminalFromMessage message
        (KSplitNcTranscriptPhases.semanticPre constants assignment
          transcript).challenges.ncCoins
        (KSplitNcTranscriptPhases.semanticNcExecution selectedProfile constants
          assignment transcript).challengePoint =
      claims.ncTerminal

/-- Claimed-value agreement is invariant under assignment extensions that
preserve every transcript source. -/
theorem claimedValuesAgree_of_agree
    {polynomialInput : PublicInput shape}
    {domains : Domains}
    (selectedProfile : Polynomial.Fe.SupportedProfile shape domains.fe)
    (constants : Poseidon2Schedule.Constants)
    (left right : Nat → Nat)
    (transcript : KSplitNcTranscript.Input polynomialInput domains)
    (sources : KSplitNcTranscriptPlacement.InputInPrefix transcript)
    (agree :
      ∀ column, column < transcript.transcriptBase →
        left column = right column)
    (message : OutputMessage shape)
    (claims : ConcreteNifsClaimedValuesHonest.Values)
    (holds :
      ClaimedValuesAgree selectedProfile constants left transcript
        message claims) :
    ClaimedValuesAgree selectedProfile constants right transcript
      message claims := by
  have preEqual :=
    KSplitNcTranscriptAssignmentInvariant.semanticPre_eq
      constants left right transcript sources agree
  have feEqual :=
    KSplitNcTranscriptAssignmentInvariant.semanticFeExecution_eq
      selectedProfile constants left right transcript sources agree
  have ncEqual :=
    KSplitNcTranscriptAssignmentInvariant.semanticNcExecution_eq
      selectedProfile constants left right transcript sources agree
  have certificateEqual :=
    KSplitNcTranscriptAssignmentInvariant.feCertificate_eq
      left right transcript sources agree
  have initialEqual :
      KSplitNcTranscriptPhases.semanticFeInitial selectedProfile constants
          left transcript =
        KSplitNcTranscriptPhases.semanticFeInitial selectedProfile constants
          right transcript := by
    unfold KSplitNcTranscriptPhases.semanticFeInitial
    rw [preEqual]
  refine {
    feInitial := ?_
    feBoundary := ?_
    feTerminal := ?_
    ncInitial := holds.ncInitial
    ncTerminal := ?_
  }
  · calc
      KSplitNcTranscriptPhases.semanticFeInitial selectedProfile constants
          right transcript =
        KSplitNcTranscriptPhases.semanticFeInitial selectedProfile constants
          left transcript := initialEqual.symm
      _ = claims.feInitial := holds.feInitial
  · calc
      KSplitNcFeRows.boundaryValue
          (KSplitNcTranscriptPhases.semanticFeInitial selectedProfile constants
            right transcript)
          (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile constants
            right transcript).challengePoint
          (KSplitNcTranscriptPhases.feCertificate right transcript) =
        KSplitNcFeRows.boundaryValue
          (KSplitNcTranscriptPhases.semanticFeInitial selectedProfile constants
            left transcript)
          (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile constants
            left transcript).challengePoint
          (KSplitNcTranscriptPhases.feCertificate left transcript) := by
            rw [← initialEqual, ← feEqual, ← certificateEqual]
      _ = claims.feBoundary := holds.feBoundary
  · rw [← holds.feTerminal, preEqual, feEqual]
  · rw [← holds.ncTerminal, preEqual, ncEqual]

/-- Selected claimed values installed into the actual physical prefix. -/
def seededAssignment
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
        verifierRows) : ColumnId → Field :=
  ConcreteNifsClaimedValuesHonest.seed application profile frame assignment
    (values (keys := keys) running fresh proof)

/-- Canonical numeric view of the selected physical seed. -/
def seededNumericAssignment
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
        verifierRows) : Nat → Nat :=
  numericAssignment (columnMap frame)
    (seededAssignment application profile frame assignment
      running fresh proof)

private theorem operands_subset_visible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ∀ column, column ∈ frame.operands.ids → column ∈ frame.visibleIds := by
  intro column member
  have contextMember : column ∈ frame.contextBundles.ids := by
    exact RefBundles.fromSchema_ids_subset _ _ column member
  simp [CallFrame.visibleIds, contextMember]

/-- Installing the selected claims preserves the exact honest operand
encoding, including every codec admissibility proof. -/
theorem seeded_encodes
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
    frame.operands.Encodes (FamilyFor application)
      (seededAssignment application profile frame assignment
        running fresh proof)
      (.cons running (.cons fresh (.cons proof .nil))) := by
  apply RefBundles.encodes_of_agrees
    (FamilyFor application) assignment
    (seededAssignment application profile frame assignment
      running fresh proof)
    frame.operands
    (.cons running (.cons fresh (.cons proof .nil)))
  · apply agreesOn_of_subset
      (operands_subset_visible application frame)
    simpa [seededAssignment] using
      (ConcreteNifsClaimedValuesHonest.seed_agrees_visible
        application profile frame assignment
        (values (keys := keys) running fresh proof) fits)
  · exact encoded

theorem seededNumericAssignment_residues
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
        verifierRows) :
    ∀ column,
      seededNumericAssignment application profile frame assignment
        running fresh proof column < goldilocksP := by
  intro column
  exact numericAssignment_canonical (columnMap frame)
    (seededAssignment application profile frame assignment
      running fresh proof) column

theorem seededNumericAssignment_constantWire
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
    (constantWire : assignment frame.one = 1) :
    seededNumericAssignment application profile frame assignment
        running fresh proof 0 = 1 := by
  apply ConcreteNifsOperationalOccurrenceSemantics.numericConstantWire
    application frame
  have oneVisible : frame.one ∈ frame.visibleIds := by
    simp [CallFrame.visibleIds]
  calc
    seededAssignment application profile frame assignment
        running fresh proof frame.one =
      assignment frame.one := by
        exact ConcreteNifsClaimedValuesHonest.seed_agrees_visible
          application profile frame assignment
          (values (keys := keys) running fresh proof) fits
          frame.one oneVisible
    _ = 1 := constantWire

/-- Restoring the selected dynamic public input changes no physical transcript
source or placement fact. -/
theorem retargetedTranscript_inPrefix
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) :
    KSplitNcTranscriptPlacement.InputInPrefix
      (KSplitNcStaticInput.retargetTranscript proof.piCcsInput
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)) := by
  let staticSources :=
    ConcreteNifsOperationalConservation.transcriptInput_inPrefix
      application profile frame
  exact {
    positive := staticSources.positive
    prior := staticSources.prior
    statement := staticSources.statement
    output := staticSources.output
    feInitial := staticSources.feInitial
    feRow := staticSources.feRow
    feLane := staticSources.feLane
    ncBlock := staticSources.ncBlock
    ncLane := staticSources.ncLane
  }

/-- Honest replay of the selected transcript is valid and leaves every
semantic transcript source unchanged. -/
theorem selectedTranscript_valid
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
    (constantWire : assignment frame.one = 1) :
    let initial :=
      seededNumericAssignment application profile frame assignment
        running fresh proof
    let transcript :=
      KSplitNcStaticInput.retargetTranscript proof.piCcsInput
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)
    SymbolicDuplexSemantics.Valid transcript.transcriptBase
      profile.constants
      (ConcreteNifsOperationalHonest.afterTranscript
        application profile frame initial)
      (KSplitNcTranscript.outputBuilder transcript) := by
  dsimp only
  let initial :=
    seededNumericAssignment application profile frame assignment
      running fresh proof
  let transcript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame)
  have sources :=
    retargetedTranscript_inPrefix application profile frame proof
  have placed :=
    (KSplitNcTranscriptPlacement.outputBuilder_invariant
      transcript sources).1
  have satisfied :
      Satisfies
        (SymbolicDuplex.rows transcript.transcriptBase profile.constants
          (KSplitNcTranscript.outputBuilder transcript))
        (ConcreteNifsOperationalHonest.afterTranscript
          application profile frame initial) := by
    apply SymbolicDuplexHonest.rows_honest
      transcript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder transcript) initial
      placed sources.positive
    · exact seededNumericAssignment_residues application profile frame
        assignment running fresh proof
    · exact seededNumericAssignment_constantWire application profile frame
        assignment running fresh proof fits constantWire
  apply SymbolicDuplexSemantics.valid_of_satisfied
    transcript.transcriptBase profile.constants
    (KSplitNcTranscript.outputBuilder transcript)
    (ConcreteNifsOperationalHonest.afterTranscript
      application profile frame initial)
  · exact SymbolicDuplexHonest.witnesses_residues
      transcript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder transcript).entries initial
      (seededNumericAssignment_residues application profile frame
        assignment running fresh proof)
  · rw [ConcreteNifsOperationalHonest.afterTranscript_preserves_before
      application profile frame initial 0 sources.positive]
    exact seededNumericAssignment_constantWire application profile frame
      assignment running fresh proof fits constantWire
  · exact satisfied

/-- The honest transcript witness preserves every seeded claimed-value pair
below its allocation base. -/
theorem seededTemporaryK_afterTranscript
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
    (index : Nat) (indexLt : index < 5) :
    let initial :=
      seededNumericAssignment application profile frame assignment
        running fresh proof
    let columns :=
      ConcreteNifsOperationalOccurrence.temporaryK
        (FamilyFor application) frame index
    KSplitNcTranscriptSemantics.decodedColumns
        (ConcreteNifsOperationalHonest.afterTranscript
          application profile frame initial)
        columns =
      (values (keys := keys) running fresh proof).get index := by
  dsimp only
  let initial :=
    seededNumericAssignment application profile frame assignment
      running fresh proof
  let columns :=
    ConcreteNifsOperationalOccurrence.temporaryK
      (FamilyFor application) frame index
  have before :=
    ConcreteNifsOperationalConservation.temporaryK_below_transcriptBase
      application profile frame index indexLt
  have low :
      ConcreteNifsOperationalHonest.afterTranscript
          application profile frame initial columns.c0 =
        initial columns.c0 :=
    ConcreteNifsOperationalHonest.afterTranscript_preserves_before
      application profile frame initial columns.c0 before.1
  have high :
      ConcreteNifsOperationalHonest.afterTranscript
          application profile frame initial columns.c1 =
        initial columns.c1 :=
    ConcreteNifsOperationalHonest.afterTranscript_preserves_before
      application profile frame initial columns.c1 before.2
  have decodedPreserved :
      KSplitNcTranscriptSemantics.decodedColumns
          (ConcreteNifsOperationalHonest.afterTranscript
            application profile frame initial)
          columns =
        KSplitNcTranscriptSemantics.decodedColumns initial columns := by
    unfold KSplitNcTranscriptSemantics.decodedColumns
      Nightstream.Implementation.R1CS.ProjectionProgram.KColumns.value
      Nightstream.Implementation.R1CS.ProjectionProgram.baseAt
    rw [low, high]
  rw [decodedPreserved]
  exact ConcreteNifsClaimedValuesHonest.seed_temporaryK
    application profile frame assignment
    (values (keys := keys) running fresh proof) fits index indexLt

/-- The seeded values are the selected verifier's own operational
computations, before any row witness is installed. -/
theorem seededValues_agree
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
    let physical :=
      seededAssignment application profile frame assignment running fresh proof
    let initial :=
      seededNumericAssignment application profile frame assignment
        running fresh proof
    let transcript :=
      KSplitNcStaticInput.retargetTranscript proof.piCcsInput
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)
    ClaimedValuesAgree
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      profile.constants initial transcript proof.certificate.piCcs.output
      (values (keys := keys) running fresh proof) := by
  dsimp only
  let physical :=
    seededAssignment application profile frame assignment running fresh proof
  let initial :=
    seededNumericAssignment application profile frame assignment
      running fresh proof
  let transcript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame)
  let selectedKey :=
    keys Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
  let selectedContext :=
    ConcreteNifsParameters.context selectedKey running fresh proof
  let selectedPre :=
    derivePreSumcheck selectedKey.template.piCcsSchedule proof.priorState
      selectedContext.materialize.piCcsStatement
  let selectedInitial :=
    Polynomial.Fe.initial selectedKey.template.profile proof.piCcsInput
      selectedPre.challenges.feCoins
  let selectedFe :=
    Transcript.Fe.derive
      (feMachine selectedKey.template.piCcsSchedule selectedInitial)
      selectedPre.state proof.certificate.piCcs.fe
  let selectedNc :=
    Transcript.Nc.BlockLane.derive
      (ncMachine selectedKey.template.piCcsSchedule)
      selectedFe.finalState proof.certificate.piCcs.nc
  let operationalPre :=
    KSplitNcTranscriptPhases.semanticPre profile.constants initial transcript
  let operationalInitial :=
    KSplitNcTranscriptPhases.semanticFeInitial selectedKey.template.profile
      profile.constants initial transcript
  let operationalFe :=
    KSplitNcTranscriptPhases.semanticFeExecution selectedKey.template.profile
      profile.constants initial transcript
  let operationalNc :=
    KSplitNcTranscriptPhases.semanticNcExecution selectedKey.template.profile
      profile.constants initial transcript
  have physicalEncoded :
      frame.operands.Encodes (FamilyFor application) physical
        (.cons running (.cons fresh (.cons proof .nil))) :=
    seeded_encodes application profile frame assignment running fresh proof
      fits encoded
  have physicalDecoded :
      frame.operands.Decodes (FamilyFor application) physical
        (.cons running (.cons fresh (.cons proof .nil))) :=
    frame.operands.decodes_of_encodes (FamilyFor application) physical
      (.cons running (.cons fresh (.cons proof .nil))) physicalEncoded
  have physicalWire : physical frame.one = 1 := by
    have oneVisible : frame.one ∈ frame.visibleIds := by
      simp [CallFrame.visibleIds]
    calc
      physical frame.one = assignment frame.one := by
        exact ConcreteNifsClaimedValuesHonest.seed_agrees_visible
          application profile frame assignment
          (values (keys := keys) running fresh proof) fits
          frame.one oneVisible
      _ = 1 := constantWire
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame physical running fresh proof
      physicalDecoded
  have admissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have inputEqual :
      KSplitNcStaticInput.withDynamicClaims
          profile.constraintPolynomial proof.piCcsInput =
        proof.piCcsInput :=
    KSplitNcStaticInput.withDynamicClaims_eq
      profile.constraintPolynomial proof.piCcsInput
      (profile.proofAdmissiblePolynomial proof admissible)
  have preEqual : operationalPre = selectedPre := by
    exact ConcreteNifsOperationalSelected.selectedPreSumcheck_eq
      application profile frame physical running fresh proof physicalWire
      physicalDecoded
  have feCertificateEqual :
      HEq
        (KSplitNcTranscriptPhases.feCertificate initial transcript)
        proof.certificate.piCcs.fe :=
    ConcreteNifsOperationalOccurrenceSemantics.feCertificate_heq
      application profile frame physical running fresh proof physicalDecoded
  have ncCertificateEqual :
      KSplitNcTranscriptPhases.ncCertificate initial transcript =
        proof.certificate.piCcs.nc :=
    ConcreteNifsOperationalOccurrenceSemantics.ncCertificate_eq
      application profile frame physical running fresh proof physicalDecoded
  have initialEqual : operationalInitial = selectedInitial := by
    change
      Polynomial.Fe.initial selectedKey.template.profile
          (KSplitNcStaticInput.withDynamicClaims
            profile.constraintPolynomial proof.piCcsInput)
          operationalPre.challenges.feCoins =
        Polynomial.Fe.initial selectedKey.template.profile proof.piCcsInput
          selectedPre.challenges.feCoins
    rw [inputEqual, preEqual]
  have feMachineEqual :
      KSplitNcTranscriptPhases.semanticFeMachine
          selectedKey.template.profile profile.constants initial transcript =
        feMachine selectedKey.template.piCcsSchedule selectedInitial := by
    exact
      (ConcreteNifsOperationalSelected.selectedFeMachine_eq
        application profile initial transcript operationalInitial).trans
        (congrArg (feMachine selectedKey.template.piCcsSchedule)
          initialEqual)
  have feExecutionEqual : operationalFe = selectedFe := by
    exact ConcreteNifsOperationalSelected.feDerived_transport_heq
      feMachineEqual
      (congrArg PreSumcheck.state preEqual)
      inputEqual
      feCertificateEqual
  have ncMachineEqual :
      ncMachine
          (KSplitNcTranscriptSemantics.valueSchedule
            profile.constants initial transcript) =
        ncMachine selectedKey.template.piCcsSchedule :=
    ConcreteNifsOperationalSelected.selectedNcMachine_eq
      application profile initial transcript
  have ncExecutionEqual : operationalNc = selectedNc := by
    exact ConcreteNifsOperationalSelected.ncDerived_transport
      ncMachineEqual
      (congrArg Transcript.Fe.Derived.finalState feExecutionEqual)
      ncCertificateEqual
  refine {
    feInitial := initialEqual
    feBoundary := ?_
    feTerminal := ?_
    ncInitial := rfl
    ncTerminal := ?_
  }
  · simpa only [physical, initial, transcript, selectedKey, selectedContext,
      selectedPre, selectedInitial, selectedFe, values,
      operationalInitial, operationalFe] using
      ConcreteNifsOperationalSelected.selectedBoundaryValue_eq
        application profile frame physical running fresh proof physicalWire
        physicalDecoded
  · change
      Polynomial.Fe.terminalFromMessage selectedKey.template.profile
          (KSplitNcStaticInput.withDynamicClaims
            profile.constraintPolynomial proof.piCcsInput)
          operationalPre.challenges.feCoins
          operationalFe.challengePoint proof.certificate.piCcs.output =
        Polynomial.Fe.terminalFromMessage selectedKey.template.profile
          proof.piCcsInput selectedPre.challenges.feCoins
          selectedFe.challengePoint proof.certificate.piCcs.output
    rw [inputEqual, preEqual, feExecutionEqual]
  · change
      Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          proof.certificate.piCcs.output
          operationalPre.challenges.ncCoins
          operationalNc.challengePoint =
        Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          proof.certificate.piCcs.output
          selectedPre.challenges.ncCoins
          selectedNc.challengePoint
    rw [preEqual, ncExecutionEqual]

/-- Selected ΠCCS acceptance is preserved when the honest transcript replay
installs only columns above the caller-owned transcript prefix. -/
theorem selectedAccepted_afterTranscript
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
      seededNumericAssignment application profile frame assignment
        running fresh proof
    let transcript :=
      KSplitNcStaticInput.retargetTranscript proof.piCcsInput
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)
    Protocol.BlockLane.Accepted
      (fun _ : Unit =>
        KSplitNcStaticInput.withDynamicClaims
          profile.constraintPolynomial proof.piCcsInput)
      (KSplitNcTranscriptSemantics.valueSchedule profile.constants
        (ConcreteNifsOperationalHonest.afterTranscript
          application profile frame initial)
        transcript)
      (KSplitNcTranscriptSemantics.priorState
        (ConcreteNifsOperationalHonest.afterTranscript
          application profile frame initial)
        transcript)
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      KSplitNcTranscriptSemantics.unitStatement
      (KSplitNcOperational.certificate
        (ConcreteNifsOperationalHonest.afterTranscript
          application profile frame initial)
        transcript proof.certificate.piCcs.output) := by
  dsimp only
  let physical :=
    seededAssignment application profile frame assignment running fresh proof
  let initial :=
    seededNumericAssignment application profile frame assignment
      running fresh proof
  let transcript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame)
  have physicalEncoded :
      frame.operands.Encodes (FamilyFor application) physical
        (.cons running (.cons fresh (.cons proof .nil))) := by
    exact seeded_encodes application profile frame assignment running fresh
      proof fits encoded
  have physicalDecoded :
      frame.operands.Decodes (FamilyFor application) physical
        (.cons running (.cons fresh (.cons proof .nil))) := by
    exact frame.operands.decodes_of_encodes (FamilyFor application) physical
      (.cons running (.cons fresh (.cons proof .nil))) physicalEncoded
  have physicalWire : physical frame.one = 1 := by
    have oneVisible : frame.one ∈ frame.visibleIds := by
      simp [CallFrame.visibleIds]
    calc
      physical frame.one = assignment frame.one := by
        exact ConcreteNifsClaimedValuesHonest.seed_agrees_visible
          application profile frame assignment
          (values (keys := keys) running fresh proof) fits
          frame.one oneVisible
      _ = 1 := constantWire
  have acceptedInitial :
      Protocol.BlockLane.Accepted
        (fun _ : Unit =>
          KSplitNcStaticInput.withDynamicClaims
            profile.constraintPolynomial proof.piCcsInput)
        (KSplitNcTranscriptSemantics.valueSchedule profile.constants
          initial transcript)
        (KSplitNcTranscriptSemantics.priorState initial transcript)
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.profile)
        KSplitNcTranscriptSemantics.unitStatement
        (KSplitNcOperational.certificate initial transcript
          proof.certificate.piCcs.output) := by
    simpa only [physical, initial, transcript,
      seededNumericAssignment] using
      ConcreteNifsOperationalSelected.retargetedAccepted_of_selected
        application profile frame physical running fresh proof physicalWire
        physicalDecoded selectedAccepted
  apply
    (KSplitNcTranscriptAssignmentInvariant.accepted_iff
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      profile.constants initial
      (ConcreteNifsOperationalHonest.afterTranscript
        application profile frame initial)
      transcript
      (retargetedTranscript_inPrefix application profile frame proof)
      (fun column below => by
        symm
        exact ConcreteNifsOperationalHonest.afterTranscript_preserves_before
          application profile frame initial column
          (by
            simpa only [transcript, KSplitNcStaticInput.retargetTranscript]
              using below))
      proof.certificate.piCcs.output).mp
  exact acceptedInitial

/-- The honest selected frame binds all four physical claimed-chain
endpoints to the verifier-derived values after transcript replay. -/
theorem selectedEndpoints_afterTranscript
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
      seededNumericAssignment application profile frame assignment
        running fresh proof
    let transcript :=
      KSplitNcStaticInput.retargetTranscript proof.piCcsInput
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)
    KSplitNcOperational.EndpointAgrees
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      profile.constants
      (ConcreteNifsOperationalHonest.afterTranscript
        application profile frame initial)
      transcript proof.certificate.piCcs.output := by
  dsimp only
  let initial :=
    seededNumericAssignment application profile frame assignment
      running fresh proof
  let transcript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame)
  let transcriptAssignment :=
    ConcreteNifsOperationalHonest.afterTranscript
      application profile frame initial
  let selectedProfile :=
    (keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
      |>.template.profile)
  let claims := values (keys := keys) running fresh proof
  have claimsInitial :
      ClaimedValuesAgree selectedProfile profile.constants initial transcript
        proof.certificate.piCcs.output claims := by
    simpa only [initial, transcript, selectedProfile, claims] using
      seededValues_agree application profile frame assignment running fresh
        proof fits encoded constantWire
  have claimsAfter :
      ClaimedValuesAgree selectedProfile profile.constants
        transcriptAssignment transcript proof.certificate.piCcs.output
        claims := by
    exact claimedValuesAgree_of_agree
      selectedProfile profile.constants initial transcriptAssignment transcript
      (retargetedTranscript_inPrefix application profile frame proof)
      (fun column below => by
        symm
        exact
          ConcreteNifsOperationalHonest.afterTranscript_preserves_before
            application profile frame initial column
            (by
              simpa only [transcript, KSplitNcStaticInput.retargetTranscript]
                using below))
      proof.certificate.piCcs.output claims claimsInitial
  have temporary0 :=
    seededTemporaryK_afterTranscript application profile frame assignment
      running fresh proof fits 0 (by omega)
  have temporary1 :=
    seededTemporaryK_afterTranscript application profile frame assignment
      running fresh proof fits 1 (by omega)
  have temporary2 :=
    seededTemporaryK_afterTranscript application profile frame assignment
      running fresh proof fits 2 (by omega)
  have temporary3 :=
    seededTemporaryK_afterTranscript application profile frame assignment
      running fresh proof fits 3 (by omega)
  have temporary4 :=
    seededTemporaryK_afterTranscript application profile frame assignment
      running fresh proof fits 4 (by omega)
  refine {
    feInitial := ?_
    feTerminal := ?_
    ncInitial := ?_
    ncTerminal := ?_
  }
  · calc
      KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
          transcript.fe.initial =
        claims.feInitial := by
          simpa only [transcript, transcriptAssignment,
            KSplitNcStaticInput.retargetTranscript,
            ConcreteNifsOperationalOccurrence.transcriptInput,
            ConcreteNifsClaimedValuesHonest.Values.get] using temporary0
      _ =
        KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
          profile.constants transcriptAssignment transcript :=
        claimsAfter.feInitial.symm
  · calc
      KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
          transcript.fe.terminal =
        claims.feTerminal := by
          simpa only [transcript, transcriptAssignment,
            KSplitNcStaticInput.retargetTranscript,
            ConcreteNifsOperationalOccurrence.transcriptInput,
            ConcreteNifsClaimedValuesHonest.Values.get] using temporary2
      _ =
        Polynomial.Fe.terminalFromMessage selectedProfile
          (KSplitNcStaticInput.withDynamicClaims
            profile.constraintPolynomial proof.piCcsInput)
          (KSplitNcTranscriptPhases.semanticPre profile.constants
            transcriptAssignment transcript).challenges.feCoins
          (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
            profile.constants transcriptAssignment transcript).challengePoint
          proof.certificate.piCcs.output :=
        claimsAfter.feTerminal.symm
  · calc
      KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
          transcript.nc.initial =
        claims.ncInitial := by
          simpa only [transcript, transcriptAssignment,
            KSplitNcStaticInput.retargetTranscript,
            ConcreteNifsOperationalOccurrence.transcriptInput,
            ConcreteNifsClaimedValuesHonest.Values.get] using temporary3
      _ = Polynomial.Nc.BlockLane.InitialSum.claimedInitial :=
        claimsAfter.ncInitial.symm
  · calc
      KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
          transcript.nc.terminal =
        claims.ncTerminal := by
          simpa only [transcript, transcriptAssignment,
            KSplitNcStaticInput.retargetTranscript,
            ConcreteNifsOperationalOccurrence.transcriptInput,
            ConcreteNifsClaimedValuesHonest.Values.get] using temporary4
      _ =
        Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          proof.certificate.piCcs.output
          (KSplitNcTranscriptPhases.semanticPre profile.constants
            transcriptAssignment transcript).challenges.ncCoins
          (KSplitNcTranscriptPhases.semanticNcExecution selectedProfile
            profile.constants transcriptAssignment transcript).challengePoint :=
        claimsAfter.ncTerminal.symm

/-- The shared physical FE row/lane column carries the deterministic
verifier-derived boundary value after transcript replay. -/
theorem selectedBoundary_afterTranscript
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
      seededNumericAssignment application profile frame assignment
        running fresh proof
    let transcript :=
      KSplitNcStaticInput.retargetTranscript proof.piCcsInput
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame)
    let transcriptAssignment :=
      ConcreteNifsOperationalHonest.afterTranscript
        application profile frame initial
    KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
        transcript.fe.boundary =
      KSplitNcFeRows.boundaryValue
        (KSplitNcTranscriptPhases.semanticFeInitial
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          profile.constants transcriptAssignment transcript)
        (KSplitNcTranscriptPhases.semanticFeExecution
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
            |>.template.profile)
          profile.constants transcriptAssignment transcript).challengePoint
        (KSplitNcTranscriptPhases.feCertificate transcriptAssignment
          transcript) := by
  dsimp only
  let initial :=
    seededNumericAssignment application profile frame assignment
      running fresh proof
  let transcript :=
    KSplitNcStaticInput.retargetTranscript proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame)
  let transcriptAssignment :=
    ConcreteNifsOperationalHonest.afterTranscript
      application profile frame initial
  let selectedProfile :=
    (keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
      |>.template.profile)
  let claims := values (keys := keys) running fresh proof
  have claimsInitial :
      ClaimedValuesAgree selectedProfile profile.constants initial transcript
        proof.certificate.piCcs.output claims := by
    simpa only [initial, transcript, selectedProfile, claims] using
      seededValues_agree application profile frame assignment running fresh
        proof fits encoded constantWire
  have claimsAfter :
      ClaimedValuesAgree selectedProfile profile.constants
        transcriptAssignment transcript proof.certificate.piCcs.output
        claims := by
    exact claimedValuesAgree_of_agree
      selectedProfile profile.constants initial transcriptAssignment transcript
      (retargetedTranscript_inPrefix application profile frame proof)
      (fun column below => by
        symm
        exact
          ConcreteNifsOperationalHonest.afterTranscript_preserves_before
            application profile frame initial column
            (by
              simpa only [transcript, KSplitNcStaticInput.retargetTranscript]
                using below))
      proof.certificate.piCcs.output claims claimsInitial
  have temporary1 :=
    seededTemporaryK_afterTranscript application profile frame assignment
      running fresh proof fits 1 (by omega)
  calc
    KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
        transcript.fe.boundary =
      claims.feBoundary := by
        simpa only [transcript, transcriptAssignment,
          KSplitNcStaticInput.retargetTranscript,
          ConcreteNifsOperationalOccurrence.transcriptInput,
          ConcreteNifsClaimedValuesHonest.Values.get] using temporary1
    _ =
      KSplitNcFeRows.boundaryValue
        (KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
          profile.constants transcriptAssignment transcript)
        (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
          profile.constants transcriptAssignment transcript).challengePoint
        (KSplitNcTranscriptPhases.feCertificate transcriptAssignment
          transcript) :=
      claimsAfter.feBoundary.symm

/-- Selected verifier acceptance supplies the exact three physical claimed
chains consumed by the operational honest witness. -/
theorem selectedChains_afterTranscript
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
      seededNumericAssignment application profile frame assignment
        running fresh proof
    let input :=
      KSplitNcStaticInput.retarget proof.piCcsInput
        (ConcreteNifsOperationalOccurrence.input application profile frame)
    let transcriptAssignment :=
      ConcreteNifsOperationalHonest.afterTranscript
        application profile frame initial
    let columns := KSplitNcTranscript.numericColumns input.transcript
    FixedPhase.Chain ops
        (columns.fe.rowSource.paperCurrent transcriptAssignment)
        (columns.fe.rowSource.paperRounds transcriptAssignment)
        (columns.fe.rowSource.paperChallenges transcriptAssignment)
        (columns.fe.rowSource.paperTerminal transcriptAssignment) ∧
      FixedPhase.Chain ops
        (columns.fe.laneSource.paperCurrent transcriptAssignment)
        (columns.fe.laneSource.paperRounds transcriptAssignment)
        (columns.fe.laneSource.paperChallenges transcriptAssignment)
        (columns.fe.laneSource.paperTerminal transcriptAssignment) ∧
      FixedPhase.Chain ops
        (columns.nc.paperCurrent transcriptAssignment)
        (columns.nc.paperRounds transcriptAssignment)
        (columns.nc.paperChallenges transcriptAssignment)
        (columns.nc.paperTerminal transcriptAssignment) := by
  dsimp only
  let initial :=
    seededNumericAssignment application profile frame assignment
      running fresh proof
  let input :=
    KSplitNcStaticInput.retarget proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.input application profile frame)
  let transcript := input.transcript
  let transcriptAssignment :=
    ConcreteNifsOperationalHonest.afterTranscript
      application profile frame initial
  let columns := KSplitNcTranscript.numericColumns transcript
  let selectedProfile :=
    (keys
      Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
      |>.template.profile)
  have accepted :
      Protocol.BlockLane.Accepted
        (fun _ : Unit =>
          KSplitNcStaticInput.withDynamicClaims
            profile.constraintPolynomial proof.piCcsInput)
        (KSplitNcTranscriptSemantics.valueSchedule profile.constants
          transcriptAssignment transcript)
        (KSplitNcTranscriptSemantics.priorState
          transcriptAssignment transcript)
        selectedProfile KSplitNcTranscriptSemantics.unitStatement
        (KSplitNcOperational.certificate transcriptAssignment transcript
          proof.certificate.piCcs.output) := by
    simpa only [initial, input, transcript, transcriptAssignment,
      KSplitNcStaticInput.retarget] using
      selectedAccepted_afterTranscript application profile frame assignment
        running fresh proof fits encoded constantWire selectedAccepted
  have endpoints :
      KSplitNcOperational.EndpointAgrees selectedProfile profile.constants
        transcriptAssignment transcript proof.certificate.piCcs.output := by
    simpa only [initial, input, transcript, transcriptAssignment,
      selectedProfile, KSplitNcStaticInput.retarget] using
      selectedEndpoints_afterTranscript application profile frame assignment
        running fresh proof fits encoded constantWire
  have boundary :
      KSplitNcTranscriptSemantics.decodedColumns transcriptAssignment
          transcript.fe.boundary =
        KSplitNcFeRows.boundaryValue
          (KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
            profile.constants transcriptAssignment transcript)
          (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
            profile.constants transcriptAssignment transcript).challengePoint
          (KSplitNcTranscriptPhases.feCertificate transcriptAssignment
            transcript) := by
    simpa only [initial, input, transcript, transcriptAssignment,
      selectedProfile, KSplitNcStaticInput.retarget] using
      selectedBoundary_afterTranscript application profile frame assignment
        running fresh proof fits encoded constantWire
  have feAccepted :
      SumCheck.Fe.Accepted
        (KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
          profile.constants transcriptAssignment transcript)
        (Polynomial.Fe.terminalFromMessage selectedProfile
          (KSplitNcStaticInput.withDynamicClaims
            profile.constraintPolynomial proof.piCcsInput)
          (KSplitNcTranscriptPhases.semanticPre profile.constants
            transcriptAssignment transcript).challenges.feCoins
          (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
            profile.constants transcriptAssignment transcript).challengePoint
          proof.certificate.piCcs.output)
        (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
          profile.constants transcriptAssignment transcript).challengePoint
        (KSplitNcTranscriptPhases.feCertificate transcriptAssignment
          transcript) := by
    exact accepted.1
  have ncAccepted :
      SumCheck.Nc.Accepted
        Polynomial.Nc.BlockLane.InitialSum.claimedInitial
        (KSplitNcTranscriptPhases.semanticNcExecution selectedProfile
          profile.constants transcriptAssignment transcript).challengePoint.coordinates
        (Polynomial.Nc.BlockLane.Terminal.terminalFromMessage
          proof.certificate.piCcs.output
          (KSplitNcTranscriptPhases.semanticPre profile.constants
            transcriptAssignment transcript).challenges.ncCoins
          (KSplitNcTranscriptPhases.semanticNcExecution selectedProfile
            profile.constants transcriptAssignment transcript).challengePoint)
        (KSplitNcTranscriptPhases.ncCertificate transcriptAssignment
          transcript).toSumCheck := by
    exact accepted.2
  have feSplit :=
    KSplitNcFeRows.accepted_splits_at_boundaryValue
      (KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
        profile.constants transcriptAssignment transcript)
      (Polynomial.Fe.terminalFromMessage selectedProfile
        (KSplitNcStaticInput.withDynamicClaims
          profile.constraintPolynomial proof.piCcsInput)
        (KSplitNcTranscriptPhases.semanticPre profile.constants
          transcriptAssignment transcript).challenges.feCoins
        (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
          profile.constants transcriptAssignment transcript).challengePoint
        proof.certificate.piCcs.output)
      (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
        profile.constants transcriptAssignment transcript).challengePoint
      (KSplitNcTranscriptPhases.feCertificate transcriptAssignment transcript)
      feAccepted
  have feAgrees :=
    KSplitNcTranscriptPhases.feAgrees transcriptAssignment transcript
  have ncAgrees :=
    KSplitNcTranscriptPhases.ncAgrees transcriptAssignment transcript
  have transcriptValid :
      SymbolicDuplexSemantics.Valid transcript.transcriptBase
        profile.constants transcriptAssignment
        (KSplitNcTranscript.outputBuilder transcript) := by
    simpa only [initial, input, transcript, transcriptAssignment,
      KSplitNcStaticInput.retarget] using
      selectedTranscript_valid application profile frame assignment running
        fresh proof fits constantWire
  have transcriptWire : transcriptAssignment 0 = 1 := by
    change
      ConcreteNifsOperationalHonest.afterTranscript
          application profile frame initial 0 =
        1
    rw [ConcreteNifsOperationalHonest.afterTranscript_preserves_before
      application profile frame initial 0
      (retargetedTranscript_inPrefix application profile frame proof).positive]
    exact seededNumericAssignment_constantWire application profile frame
      assignment running fresh proof fits constantWire
  have feReplay :=
    KSplitNcTranscriptPhases.decoded_fe selectedProfile profile.constants
      transcriptAssignment transcriptWire transcript transcriptValid
      endpoints.feInitial
  have ncReplay :=
    KSplitNcTranscriptPhases.decoded_nc selectedProfile profile.constants
      transcriptAssignment transcriptWire transcript transcriptValid
      endpoints.feInitial
  have rowBoundary :
      columns.fe.rowSource.paperTerminal transcriptAssignment =
        KSplitNcFeRows.boundaryValue
          (KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
            profile.constants transcriptAssignment transcript)
          (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
            profile.constants transcriptAssignment transcript).challengePoint
          (KSplitNcTranscriptPhases.feCertificate transcriptAssignment
            transcript) := by
    exact boundary
  have laneBoundary :
      columns.fe.laneSource.paperCurrent transcriptAssignment =
        KSplitNcFeRows.boundaryValue
          (KSplitNcTranscriptPhases.semanticFeInitial selectedProfile
            profile.constants transcriptAssignment transcript)
          (KSplitNcTranscriptPhases.semanticFeExecution selectedProfile
            profile.constants transcriptAssignment transcript).challengePoint
          (KSplitNcTranscriptPhases.feCertificate transcriptAssignment
            transcript) := by
    exact boundary
  have rowChain := feSplit.1
  rw [← rowBoundary] at rowChain
  rw [← endpoints.feInitial, ← feAgrees.initial] at rowChain
  rw [← feReplay.point] at rowChain
  rw [← feAgrees.rowRounds, ← feAgrees.rowChallenges] at rowChain
  have laneChain := feSplit.2
  rw [← laneBoundary, ← endpoints.feTerminal, ← feAgrees.terminal] at laneChain
  rw [← feReplay.point] at laneChain
  rw [← feAgrees.laneRounds, ← feAgrees.laneChallenges] at laneChain
  have ncChain := ncAccepted
  unfold SumCheck.Nc.Accepted at ncChain
  rw [← endpoints.ncInitial, ← ncAgrees.initial,
    ← endpoints.ncTerminal, ← ncAgrees.terminal] at ncChain
  rw [← ncReplay.point] at ncChain
  rw [← ncAgrees.challenges, ← ncAgrees.rounds] at ncChain
  exact ⟨rowChain, laneChain, ncChain⟩

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedHonest
