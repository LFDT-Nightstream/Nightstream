import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsEndpointConservation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalConservation
import Nightstream.Implementation.R1CS.Canonical.KSplitNcBlockLaneHonest
import Nightstream.Implementation.R1CS.Canonical.KSplitNcEndpointsSemanticHonest

/-!
Contract: sequential honest completion of the selected operational Split-NC
row occurrence.

The assignment is extended in verifier order: exact Poseidon2 transcript
replay, three claimed-chain Horner programs, then the verifier-owned endpoint
computations.  Placement theorems prove that each later witness preserves all
earlier rows.

The semantic premises are the unchanged three fixed-phase chains and
`EndpointAgrees`; this module does not accept row equations or an operational
acceptance conclusion.  A selected-proof bridge must construct those semantic
premises before this theorem is promoted to `nifsVerify` honest completeness.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalHonest

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
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

/-- Assignment after installing every Poseidon2 transcript call witness. -/
def afterTranscript
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : Nat → Nat) : Nat → Nat :=
  let transcript :=
    ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame
  SymbolicDuplexHonest.witnesses transcript.transcriptBase profile.constants
    (KSplitNcTranscript.outputBuilder transcript).entries initial

/-- Transcript replay changes no caller-owned column below its physical
allocation base. -/
theorem afterTranscript_preserves_before
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
    (column : Nat)
    (below :
      column <
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame).transcriptBase) :
    afterTranscript application profile frame initial column =
      initial column := by
  unfold afterTranscript
  apply SymbolicDuplexHonest.witnesses_preserve_before
    (ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame).transcriptBase
    profile.constants
    (KSplitNcTranscript.outputBuilder
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame)).entries
    initial
    (ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame).transcriptBase
    column
  · intro entry member
    unfold SymbolicDuplexHonest.outputBase
      SymbolicDuplexHonest.callBase
    exact Nat.le_add_right _ _
  · exact below

/-- Assignment after the FE-row, FE-lane, and block×lane NC chains. -/
def afterNumeric
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : Nat → Nat) : Nat → Nat :=
  let input :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  KSplitNcBlockLaneHonest.witness
    (KSplitNcTranscript.numericColumns input.transcript)
    (KSplitNcOperationalRows.numericBase input)
    (afterTranscript application profile frame initial)

/-- Numeric claimed-chain completion preserves every earlier transcript and
caller-owned column. -/
theorem afterNumeric_preserves_before
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
    (column : Nat)
    (below :
      column <
        KSplitNcOperationalRows.numericBase
          (ConcreteNifsOperationalOccurrence.input
            application profile frame)) :
    afterNumeric application profile frame initial column =
      afterTranscript application profile frame initial column := by
  unfold afterNumeric
  exact KSplitNcBlockLaneHonest.witness_off_block
    (KSplitNcTranscript.numericColumns
      (ConcreteNifsOperationalOccurrence.input
        application profile frame).transcript)
    (KSplitNcOperationalRows.numericBase
      (ConcreteNifsOperationalOccurrence.input
        application profile frame))
    (afterTranscript application profile frame initial)
    column below

/-- Final operational witness, after the three endpoint programs. -/
def witness
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : Nat → Nat) : Nat → Nat :=
  let input :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  KSplitNcEndpointsHonest.witness
    (KSplitNcOperationalRows.endpointInput input)
    (afterNumeric application profile frame initial)

/-- Final operational witness after restoring the dynamic public claims
carried by the selected proof.  Retargeting changes no row or column, but the
endpoint arithmetic must evaluate the restored values rather than the static
layout placeholders. -/
def retargetedWitness
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (publicInput : PublicInput shape)
    (initial : Nat → Nat) : Nat → Nat :=
  let input :=
    KSplitNcStaticInput.retarget publicInput
      (ConcreteNifsOperationalOccurrence.input application profile frame)
  KSplitNcEndpointsHonest.witness
    (KSplitNcOperationalRows.endpointInput input)
    (afterNumeric application profile frame initial)

private theorem satisfies_append
    {left right : List Nightstream.Implementation.R1CS.Row}
    {assignment : Nat → Nat}
    (leftSatisfied : Satisfies left assignment)
    (rightSatisfied : Satisfies right assignment) :
    Satisfies (left ++ right) assignment := by
  intro row member
  exact (List.mem_append.1 member).elim
    (leftSatisfied row) (rightSatisfied row)

/-- Model-proved honest completion of the complete operational ΠCCS row
occurrence.  The caller supplies only the frozen semantic chains and endpoint
relation; every auxiliary assignment and every row equation is constructed
here. -/
theorem rows_honest_of_semantics
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
    (message : OutputMessage shape)
    (initialResidues : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    (rowChain :
      let input :=
        ConcreteNifsOperationalOccurrence.input application profile frame
      let transcriptAssignment :=
        afterTranscript application profile frame initial
      let source :=
        (KSplitNcTranscript.numericColumns input.transcript).fe.rowSource
      FixedPhase.Chain ops
        (source.paperCurrent transcriptAssignment)
        (source.paperRounds transcriptAssignment)
        (source.paperChallenges transcriptAssignment)
        (source.paperTerminal transcriptAssignment))
    (laneChain :
      let input :=
        ConcreteNifsOperationalOccurrence.input application profile frame
      let transcriptAssignment :=
        afterTranscript application profile frame initial
      let source :=
        (KSplitNcTranscript.numericColumns input.transcript).fe.laneSource
      FixedPhase.Chain ops
        (source.paperCurrent transcriptAssignment)
        (source.paperRounds transcriptAssignment)
        (source.paperChallenges transcriptAssignment)
        (source.paperTerminal transcriptAssignment))
    (ncChain :
      let input :=
        ConcreteNifsOperationalOccurrence.input application profile frame
      let transcriptAssignment :=
        afterTranscript application profile frame initial
      let source :=
        (KSplitNcTranscript.numericColumns input.transcript).nc
      FixedPhase.Chain ops
        (source.paperCurrent transcriptAssignment)
        (source.paperRounds transcriptAssignment)
        (source.paperChallenges transcriptAssignment)
        (source.paperTerminal transcriptAssignment))
    (authority :
      let input :=
        ConcreteNifsOperationalOccurrence.input application profile frame
      KSplitNcEndpoints.DecodedAuthority
        (KSplitNcOperationalRows.endpointInput input)
        (afterNumeric application profile frame initial) message)
    (endpoints :
      let input :=
        ConcreteNifsOperationalOccurrence.input application profile frame
      KSplitNcOperational.EndpointAgrees
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.profile)
        profile.constants
        (afterNumeric application profile frame initial)
        input.transcript message) :
    Satisfies
      (ConcreteNifsOperationalOccurrence.rows application profile frame)
      (witness application profile frame initial) := by
  let input :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  let transcript := input.transcript
  let transcriptAssignment :=
    afterTranscript application profile frame initial
  let numericAssignment :=
    afterNumeric application profile frame initial
  let finalAssignment :=
    witness application profile frame initial
  have transcriptSources :=
    ConcreteNifsOperationalConservation.transcriptInput_inPrefix
      application profile frame
  have transcriptPositive : 0 < transcript.transcriptBase :=
    transcriptSources.positive
  have placed :=
    (KSplitNcTranscriptPlacement.outputBuilder_invariant
      transcript transcriptSources).1
  have transcriptSatisfied :
      Satisfies
        (KSplitNcOperationalRows.transcriptRows profile.constants input)
        transcriptAssignment := by
    apply SymbolicDuplexHonest.rows_honest
      transcript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder transcript) initial
      placed transcriptPositive initialResidues constantWire
  have transcriptResidues :
      ∀ column, transcriptAssignment column < goldilocksP := by
    exact SymbolicDuplexHonest.witnesses_residues
      transcript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder transcript).entries initial
      initialResidues
  have transcriptWire : transcriptAssignment 0 = 1 := by
    rw [show transcriptAssignment 0 = initial 0 by
      exact SymbolicDuplexHonest.witnesses_constantWire
        transcript.transcriptBase profile.constants transcriptPositive
        (KSplitNcTranscript.outputBuilder transcript).entries initial]
    exact constantWire
  have numericPositive : 0 < KSplitNcOperationalRows.numericBase input := by
    exact Nat.lt_of_lt_of_le transcriptPositive (Nat.le_add_right _ _)
  have numericSatisfied :
      Satisfies (KSplitNcOperationalRows.numericRows input)
        numericAssignment := by
    exact KSplitNcBlockLaneHonest.rows_honest
      (KSplitNcTranscript.numericColumns transcript)
      (KSplitNcOperationalRows.numericBase input)
      transcriptAssignment numericPositive transcriptWire
      (ConcreteNifsOperationalConservation.feRowSource_below
        application profile frame)
      (ConcreteNifsOperationalConservation.feLaneSource_below
        application profile frame)
      (ConcreteNifsOperationalConservation.ncSource_below
        application profile frame)
      rowChain laneChain ncChain
  have numericResidues :
      ∀ column, numericAssignment column < goldilocksP := by
    exact KSplitNcBlockLaneHonest.witness_residues
      (KSplitNcTranscript.numericColumns transcript)
      (KSplitNcOperationalRows.numericBase input)
      transcriptAssignment transcriptResidues
  have numericWire : numericAssignment 0 = 1 := by
    rw [show numericAssignment 0 = transcriptAssignment 0 by
      exact KSplitNcBlockLaneHonest.witness_off_block
        (KSplitNcTranscript.numericColumns transcript)
        (KSplitNcOperationalRows.numericBase input)
        transcriptAssignment 0 numericPositive]
    exact transcriptWire
  have transcriptAtNumeric :
      Satisfies
        (KSplitNcOperationalRows.transcriptRows profile.constants input)
        numericAssignment := by
    apply KHornerSupport.satisfies_extend _
      transcriptAssignment numericAssignment
    · intro row member column mentioned
      exact
        (KSplitNcBlockLaneHonest.witness_off_block
          (KSplitNcTranscript.numericColumns transcript)
          (KSplitNcOperationalRows.numericBase input)
          transcriptAssignment column
          (ConcreteNifsOperationalConservation.transcriptRows_below_numericBase
            application profile frame row member column mentioned)).symm
    · exact transcriptSatisfied
  have transcriptValid :
      SymbolicDuplexSemantics.Valid transcript.transcriptBase
        profile.constants numericAssignment
        (KSplitNcTranscript.outputBuilder transcript) := by
    exact SymbolicDuplexSemantics.valid_of_satisfied
      transcript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder transcript)
      numericAssignment numericResidues numericWire transcriptAtNumeric
  have endpointPositive :
      0 < KSplitNcOperationalRows.endpointBase input := by
    unfold KSplitNcOperationalRows.endpointBase
    omega
  have endpointSatisfied :
      Satisfies (KSplitNcOperationalRows.endpointRows input)
        finalAssignment := by
    exact KSplitNcEndpointsSemanticHonest.rows_honest_of_endpointAgrees
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      profile.constants numericAssignment numericWire
      (KSplitNcOperationalRows.endpointInput input) message
      transcriptValid authority
      (ConcreteNifsEndpointConservation.endpointInputs_below
        application profile frame)
      endpoints endpointPositive
  have transcriptAtFinal :
      Satisfies
        (KSplitNcOperationalRows.transcriptRows profile.constants input)
        finalAssignment := by
    apply KHornerSupport.satisfies_extend _
      numericAssignment finalAssignment
    · intro row member column mentioned
      exact
        (KSplitNcEndpointsHonest.witness_off_source
          (KSplitNcOperationalRows.endpointInput input)
          numericAssignment column
          (Nat.lt_of_lt_of_le
            (ConcreteNifsOperationalConservation.transcriptRows_below_numericBase
              application profile frame row member column mentioned)
            (by
              change
                KSplitNcOperationalRows.numericBase input ≤
                  KSplitNcOperationalRows.endpointBase input
              exact Nat.le_add_right _ _))).symm
    · exact transcriptAtNumeric
  have numericAtFinal :
      Satisfies (KSplitNcOperationalRows.numericRows input)
        finalAssignment := by
    apply KHornerSupport.satisfies_extend _
      numericAssignment finalAssignment
    · intro row member column mentioned
      exact
        (KSplitNcEndpointsHonest.witness_off_source
          (KSplitNcOperationalRows.endpointInput input)
          numericAssignment column
          (ConcreteNifsOperationalConservation.numericRows_below_endpointBase
            application profile frame row member column mentioned)).symm
    · exact numericSatisfied
  have combined :=
    satisfies_append (satisfies_append transcriptAtFinal numericAtFinal)
      endpointSatisfied
  simpa [ConcreteNifsOperationalOccurrence.rows,
    KSplitNcOperationalRows.rows, KSplitNcOperationalRows.rowGroups,
    KSplitNcOperationalRows.transcriptRows,
    KSplitNcOperationalRows.numericRows,
    KSplitNcOperationalRows.endpointRows,
    List.append_assoc] using combined

/-- Honest completion after restoring the exact dynamic public claims.  The
emitted row list is definitionally unchanged by retargeting, while the
endpoint witness and semantic premises are indexed by the authoritative
selected input. -/
theorem rows_honest_retargeted_of_semantics
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (publicInput : PublicInput shape)
    (initial : Nat → Nat)
    (message : OutputMessage shape)
    (initialResidues : ∀ column, initial column < goldilocksP)
    (constantWire : initial 0 = 1)
    (rowChain :
      let input :=
        KSplitNcStaticInput.retarget publicInput
          (ConcreteNifsOperationalOccurrence.input application profile frame)
      let transcriptAssignment :=
        afterTranscript application profile frame initial
      let source :=
        (KSplitNcTranscript.numericColumns input.transcript).fe.rowSource
      FixedPhase.Chain ops
        (source.paperCurrent transcriptAssignment)
        (source.paperRounds transcriptAssignment)
        (source.paperChallenges transcriptAssignment)
        (source.paperTerminal transcriptAssignment))
    (laneChain :
      let input :=
        KSplitNcStaticInput.retarget publicInput
          (ConcreteNifsOperationalOccurrence.input application profile frame)
      let transcriptAssignment :=
        afterTranscript application profile frame initial
      let source :=
        (KSplitNcTranscript.numericColumns input.transcript).fe.laneSource
      FixedPhase.Chain ops
        (source.paperCurrent transcriptAssignment)
        (source.paperRounds transcriptAssignment)
        (source.paperChallenges transcriptAssignment)
        (source.paperTerminal transcriptAssignment))
    (ncChain :
      let input :=
        KSplitNcStaticInput.retarget publicInput
          (ConcreteNifsOperationalOccurrence.input application profile frame)
      let transcriptAssignment :=
        afterTranscript application profile frame initial
      let source :=
        (KSplitNcTranscript.numericColumns input.transcript).nc
      FixedPhase.Chain ops
        (source.paperCurrent transcriptAssignment)
        (source.paperRounds transcriptAssignment)
        (source.paperChallenges transcriptAssignment)
        (source.paperTerminal transcriptAssignment))
    (authority :
      let input :=
        KSplitNcStaticInput.retarget publicInput
          (ConcreteNifsOperationalOccurrence.input application profile frame)
      KSplitNcEndpoints.DecodedAuthority
        (KSplitNcOperationalRows.endpointInput input)
        (afterNumeric application profile frame initial) message)
    (endpoints :
      let input :=
        KSplitNcStaticInput.retarget publicInput
          (ConcreteNifsOperationalOccurrence.input application profile frame)
      KSplitNcOperational.EndpointAgrees
        (keys
          Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
          |>.template.profile)
        profile.constants
        (afterNumeric application profile frame initial)
        input.transcript message) :
    Satisfies
      (ConcreteNifsOperationalOccurrence.rows application profile frame)
      (retargetedWitness application profile frame publicInput initial) := by
  let input :=
    KSplitNcStaticInput.retarget publicInput
      (ConcreteNifsOperationalOccurrence.input application profile frame)
  let transcript := input.transcript
  let transcriptAssignment :=
    afterTranscript application profile frame initial
  let numericAssignment :=
    afterNumeric application profile frame initial
  let finalAssignment :=
    retargetedWitness application profile frame publicInput initial
  have transcriptSources :=
    ConcreteNifsOperationalConservation.transcriptInput_inPrefix
      application profile frame
  have retargetedTranscriptSources :
      KSplitNcTranscriptPlacement.InputInPrefix transcript := by
    exact {
      positive := transcriptSources.positive
      prior := transcriptSources.prior
      statement := transcriptSources.statement
      output := transcriptSources.output
      feInitial := transcriptSources.feInitial
      feRow := transcriptSources.feRow
      feLane := transcriptSources.feLane
      ncBlock := transcriptSources.ncBlock
      ncLane := transcriptSources.ncLane
    }
  have transcriptPositive : 0 < transcript.transcriptBase :=
    retargetedTranscriptSources.positive
  have placed :=
    (KSplitNcTranscriptPlacement.outputBuilder_invariant
      transcript retargetedTranscriptSources).1
  have transcriptSatisfied :
      Satisfies
        (KSplitNcOperationalRows.transcriptRows profile.constants input)
        transcriptAssignment := by
    apply SymbolicDuplexHonest.rows_honest
      transcript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder transcript) initial
      placed transcriptPositive initialResidues constantWire
  have transcriptResidues :
      ∀ column, transcriptAssignment column < goldilocksP := by
    exact SymbolicDuplexHonest.witnesses_residues
      transcript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder transcript).entries initial
      initialResidues
  have transcriptWire : transcriptAssignment 0 = 1 := by
    rw [show transcriptAssignment 0 = initial 0 by
      exact SymbolicDuplexHonest.witnesses_constantWire
        transcript.transcriptBase profile.constants transcriptPositive
        (KSplitNcTranscript.outputBuilder transcript).entries initial]
    exact constantWire
  have numericPositive : 0 < KSplitNcOperationalRows.numericBase input := by
    exact Nat.lt_of_lt_of_le transcriptPositive (Nat.le_add_right _ _)
  have numericSatisfied :
      Satisfies (KSplitNcOperationalRows.numericRows input)
        numericAssignment := by
    exact KSplitNcBlockLaneHonest.rows_honest
      (KSplitNcTranscript.numericColumns transcript)
      (KSplitNcOperationalRows.numericBase input)
      transcriptAssignment numericPositive transcriptWire
      (ConcreteNifsOperationalConservation.feRowSource_below
        application profile frame)
      (ConcreteNifsOperationalConservation.feLaneSource_below
        application profile frame)
      (ConcreteNifsOperationalConservation.ncSource_below
        application profile frame)
      rowChain laneChain ncChain
  have numericResidues :
      ∀ column, numericAssignment column < goldilocksP := by
    exact KSplitNcBlockLaneHonest.witness_residues
      (KSplitNcTranscript.numericColumns transcript)
      (KSplitNcOperationalRows.numericBase input)
      transcriptAssignment transcriptResidues
  have numericWire : numericAssignment 0 = 1 := by
    rw [show numericAssignment 0 = transcriptAssignment 0 by
      exact KSplitNcBlockLaneHonest.witness_off_block
        (KSplitNcTranscript.numericColumns transcript)
        (KSplitNcOperationalRows.numericBase input)
        transcriptAssignment 0 numericPositive]
    exact transcriptWire
  have transcriptAtNumeric :
      Satisfies
        (KSplitNcOperationalRows.transcriptRows profile.constants input)
        numericAssignment := by
    apply KHornerSupport.satisfies_extend _
      transcriptAssignment numericAssignment
    · intro row member column mentioned
      exact
        (KSplitNcBlockLaneHonest.witness_off_block
          (KSplitNcTranscript.numericColumns transcript)
          (KSplitNcOperationalRows.numericBase input)
          transcriptAssignment column
          (ConcreteNifsOperationalConservation.transcriptRows_below_numericBase
            application profile frame row member column mentioned)).symm
    · exact transcriptSatisfied
  have transcriptValid :
      SymbolicDuplexSemantics.Valid transcript.transcriptBase
        profile.constants numericAssignment
        (KSplitNcTranscript.outputBuilder transcript) := by
    exact SymbolicDuplexSemantics.valid_of_satisfied
      transcript.transcriptBase profile.constants
      (KSplitNcTranscript.outputBuilder transcript)
      numericAssignment numericResidues numericWire transcriptAtNumeric
  have endpointPositive :
      0 < KSplitNcOperationalRows.endpointBase input := by
    unfold KSplitNcOperationalRows.endpointBase
    omega
  have endpointSatisfied :
      Satisfies (KSplitNcOperationalRows.endpointRows input)
        finalAssignment := by
    have endpointInputs :
        KSplitNcEndpointsSupport.InputsBelow
          (KSplitNcOperationalRows.endpointInput input) := by
      let staticInputs :=
        ConcreteNifsEndpointConservation.endpointInputs_below
          application profile frame
      exact {
        feInitialGamma := staticInputs.feInitialGamma
        feInitialAlpha := staticInputs.feInitialAlpha
        feInitialClaims := staticInputs.feInitialClaims
        feInitialEndpoint := staticInputs.feInitialEndpoint
        feTerminalGamma := staticInputs.feTerminalGamma
        feTerminalAlpha := staticInputs.feTerminalAlpha
        feTerminalBetaA := staticInputs.feTerminalBetaA
        feTerminalBetaR := staticInputs.feTerminalBetaR
        feTerminalPointLane := staticInputs.feTerminalPointLane
        feTerminalPointRow := staticInputs.feTerminalPointRow
        feTerminalPriorPoint := staticInputs.feTerminalPriorPoint
        feTerminalMessage := staticInputs.feTerminalMessage
        feTerminalEndpoint := staticInputs.feTerminalEndpoint
        ncGamma := staticInputs.ncGamma
        ncBetaBlock := staticInputs.ncBetaBlock
        ncBetaA := staticInputs.ncBetaA
        ncPointBlock := staticInputs.ncPointBlock
        ncPointLane := staticInputs.ncPointLane
        ncMessage := staticInputs.ncMessage
        ncInitialEndpoint := staticInputs.ncInitialEndpoint
        ncTerminalEndpoint := staticInputs.ncTerminalEndpoint
      }
    exact KSplitNcEndpointsSemanticHonest.rows_honest_of_endpointAgrees
      (keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected
        |>.template.profile)
      profile.constants numericAssignment numericWire
      (KSplitNcOperationalRows.endpointInput input) message
      transcriptValid authority
      endpointInputs
      endpoints endpointPositive
  have transcriptAtFinal :
      Satisfies
        (KSplitNcOperationalRows.transcriptRows profile.constants input)
        finalAssignment := by
    apply KHornerSupport.satisfies_extend _
      numericAssignment finalAssignment
    · intro row member column mentioned
      exact
        (KSplitNcEndpointsHonest.witness_off_source
          (KSplitNcOperationalRows.endpointInput input)
          numericAssignment column
          (Nat.lt_of_lt_of_le
            (ConcreteNifsOperationalConservation.transcriptRows_below_numericBase
              application profile frame row member column mentioned)
            (by
              change
                KSplitNcOperationalRows.numericBase input ≤
                  KSplitNcOperationalRows.endpointBase input
              exact Nat.le_add_right _ _))).symm
    · exact transcriptAtNumeric
  have numericAtFinal :
      Satisfies (KSplitNcOperationalRows.numericRows input)
        finalAssignment := by
    apply KHornerSupport.satisfies_extend _
      numericAssignment finalAssignment
    · intro row member column mentioned
      exact
        (KSplitNcEndpointsHonest.witness_off_source
          (KSplitNcOperationalRows.endpointInput input)
          numericAssignment column
          (ConcreteNifsOperationalConservation.numericRows_below_endpointBase
            application profile frame row member column mentioned)).symm
    · exact numericSatisfied
  have combined :=
    satisfies_append (satisfies_append transcriptAtFinal numericAtFinal)
      endpointSatisfied
  simpa [ConcreteNifsOperationalOccurrence.rows,
    KSplitNcStaticInput.rows_retarget,
    KSplitNcOperationalRows.rows, KSplitNcOperationalRows.rowGroups,
    KSplitNcOperationalRows.transcriptRows,
    KSplitNcOperationalRows.numericRows,
    KSplitNcOperationalRows.endpointRows,
    List.append_assoc] using combined

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalHonest
