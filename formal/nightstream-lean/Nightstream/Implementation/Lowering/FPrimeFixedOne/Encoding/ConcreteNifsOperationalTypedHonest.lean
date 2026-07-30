import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsNumericCompletion
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerSelected
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSelectedComplete

/-!
Contract: pull the selected operational ΠCCS witness into the exact typed
`nifsVerify` call frame.

The numeric witness is canonicalized, installed only on the frame's declared
temporary suffix, and proved to preserve the authoritative operand encoding
and constant-one wire.  Satisfaction is transported through the exact global
column map, so the resulting physical assignment decodes to the selected
ΠRLC sampler state.

No sampler challenge, verifier result, source-authority record, Rust layout,
or generated artifact is accepted as a premise.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1800000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalTypedHonest

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

/-- Canonical numeric representative of the selected operational witness. -/
def operationalNumeric
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
  ConcreteNifsNumericCompletion.canonicalize
    (ConcreteNifsOperationalHonest.retargetedWitness
      application profile frame proof.piCcsInput
      (ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
        application profile frame assignment running fresh proof))

/-- Typed assignment obtained by installing the canonical operational witness
on the call frame's declared temporary suffix. -/
def assignment
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows) : ColumnId → Field :=
  ConcreteNifsNumericCompletion.complete frame
    (ConcreteNifsOperationalSelectedHonest.seededAssignment
      application profile frame initial running fresh proof)
    (operationalNumeric application profile frame initial running fresh proof)

/-- The operational witness completion preserves every visible call
coordinate, including the output bundle supplied by honest completeness. -/
theorem assignment_agrees_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    AgreesOn frame.visibleIds initial
      (assignment application profile frame initial running fresh proof) := by
  apply agreesOn_trans
  · simpa [ConcreteNifsOperationalSelectedHonest.seededAssignment] using
      (ConcreteNifsClaimedValuesHonest.seed_agrees_visible
        application profile frame initial
        (ConcreteNifsOperationalSelectedHonest.values
          (keys := keys) running fresh proof) fits)
  · simpa [assignment] using
      (ConcreteNifsNumericCompletion.complete_agrees_visible frame
        (ConcreteNifsOperationalSelectedHonest.seededAssignment
          application profile frame initial running fresh proof)
        (operationalNumeric application profile frame initial
          running fresh proof))

/-- The operational witness writes only the call's declared temporaries. -/
theorem assignment_changesOnly
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    ChangesOnly frame.temporaries.ids initial
      (assignment application profile frame initial running fresh proof) := by
  intro column notTemporary
  change
    ConcreteNifsNumericCompletion.complete frame
        (ConcreteNifsOperationalSelectedHonest.seededAssignment
          application profile frame initial running fresh proof)
        (operationalNumeric application profile frame initial running fresh proof)
        column =
      initial column
  rw [ConcreteNifsNumericCompletion.complete_changesOnly frame
    (ConcreteNifsOperationalSelectedHonest.seededAssignment
      application profile frame initial running fresh proof)
    (operationalNumeric application profile frame initial running fresh proof)
    column notTemporary]
  exact ConcreteNifsClaimedValuesHonest.seed_changesOnly
    application profile frame initial
    (ConcreteNifsOperationalSelectedHonest.values
      (keys := keys) running fresh proof) fits column notTemporary

private theorem temporaryBase_lt_transcriptBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    temporaryBase frame <
      (ConcreteNifsOperationalOccurrence.transcriptInput
        application profile frame).transcriptBase := by
  change temporaryBase frame < temporaryBase frame + 10
  omega

theorem temporaryBase_le_samplerBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    temporaryBase frame ≤
      ConcreteNifsOperationalSampler.samplerBase
        application profile frame := by
  change temporaryBase frame ≤
    temporaryBase frame + 10 +
      KSplitNcOperationalRows.allocationWidth
        (ConcreteNifsOperationalOccurrence.input application profile frame)
  omega

theorem samplerBase_le_orderedLength
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame) :
    ConcreteNifsOperationalSampler.samplerBase application profile frame ≤
      (orderedIds frame).length := by
  rw [orderedIds_eq_visible_append_temporaries, List.length_append]
  change
    ConcreteNifsOperationalSampler.samplerBase
        application profile frame ≤
      temporaryBase frame + frame.temporaries.ids.length
  have rawBound :
      ConcreteNifsRawProgram.allocationWidth application profile frame ≤
        frame.temporaries.ids.length := fits
  change temporaryBase frame + 10 +
      KSplitNcOperationalRows.allocationWidth
        (ConcreteNifsOperationalOccurrence.input application profile frame) ≤
    temporaryBase frame + frame.temporaries.ids.length
  have prefixBound :
      10 +
          KSplitNcOperationalRows.allocationWidth
            (ConcreteNifsOperationalOccurrence.input
              application profile frame) ≤
        ConcreteNifsRawProgram.allocationWidth
          application profile frame := by
    simp [ConcreteNifsRawProgram.allocationWidth,
      ConcreteNifsOperationalSampler.cost,
      ConcreteNifsOperationalSampler.challengeCost,
      KSplitNcOperationalRows.allocationWidth_eq_cost]
    omega
  omega

/-- Canonicalization changes no selected visible input source. -/
theorem operationalNumeric_before_temporaryBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (source : Nat) (before : source < temporaryBase frame) :
    operationalNumeric application profile frame initial running fresh proof
        source =
      ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
        application profile frame initial running fresh proof source := by
  let seeded :=
    ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
      application profile frame initial running fresh proof
  let input :=
    KSplitNcStaticInput.retarget proof.piCcsInput
      (ConcreteNifsOperationalOccurrence.input application profile frame)
  have beforeTranscript :
      source <
        (ConcreteNifsOperationalOccurrence.transcriptInput
          application profile frame).transcriptBase :=
    Nat.lt_trans before
      (temporaryBase_lt_transcriptBase application profile frame)
  have beforeNumeric :
      source <
        KSplitNcOperationalRows.numericBase
          (ConcreteNifsOperationalOccurrence.input
            application profile frame) := by
    unfold KSplitNcOperationalRows.numericBase
    exact Nat.lt_of_lt_of_le beforeTranscript (Nat.le_add_right _ _)
  have beforeNumericRetarget :
      source < KSplitNcOperationalRows.numericBase input := by
    simpa [input, KSplitNcStaticInput.retarget,
      ConcreteNifsOperationalOccurrence.input] using beforeNumeric
  have beforeEndpoint :
      source <
        (KSplitNcOperationalRows.endpointInput input).frameBase := by
    change source < KSplitNcOperationalRows.endpointBase input
    unfold KSplitNcOperationalRows.endpointBase
    exact Nat.lt_of_lt_of_le beforeNumericRetarget (Nat.le_add_right _ _)
  unfold operationalNumeric ConcreteNifsNumericCompletion.canonicalize
  change
    (KSplitNcEndpointsHonest.witness
        (KSplitNcOperationalRows.endpointInput input)
        (ConcreteNifsOperationalHonest.afterNumeric
          application profile frame seeded)
        source) %
        goldilocksP =
      ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
        application profile frame initial running fresh proof source
  rw [KSplitNcEndpointsHonest.witness_off_source
      (KSplitNcOperationalRows.endpointInput input)
      (ConcreteNifsOperationalHonest.afterNumeric
        application profile frame seeded)
      source beforeEndpoint]
  rw [ConcreteNifsOperationalHonest.afterNumeric_preserves_before
    application profile frame seeded source beforeNumeric]
  rw [ConcreteNifsOperationalHonest.afterTranscript_preserves_before
    application profile frame seeded source beforeTranscript]
  exact Nat.mod_eq_of_lt
    (ConcreteNifsOperationalSelectedHonest.seededNumericAssignment_residues
      application profile frame initial running fresh proof source)

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
  have contextMember : column ∈ frame.contextBundles.ids :=
    RefBundles.fromSchema_ids_subset _ _ column member
  simp [CallFrame.visibleIds, contextMember]

/-- The typed completion preserves the exact encoded selected operands. -/
theorem assignment_encodes
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (encoded :
      frame.operands.Encodes (FamilyFor application) initial
        (.cons running (.cons fresh (.cons proof .nil)))) :
    frame.operands.Encodes (FamilyFor application)
      (assignment application profile frame initial running fresh proof)
      (.cons running (.cons fresh (.cons proof .nil))) := by
  apply RefBundles.encodes_of_agrees
    (FamilyFor application)
    (ConcreteNifsOperationalSelectedHonest.seededAssignment
      application profile frame initial running fresh proof)
    (assignment application profile frame initial running fresh proof)
    frame.operands
    (.cons running (.cons fresh (.cons proof .nil)))
  · exact agreesOn_of_subset (operands_subset_visible application frame)
      (ConcreteNifsNumericCompletion.complete_agrees_visible frame
        (ConcreteNifsOperationalSelectedHonest.seededAssignment
          application profile frame initial running fresh proof)
        (operationalNumeric application profile frame initial
          running fresh proof))
  · exact ConcreteNifsOperationalSelectedHonest.seeded_encodes
      application profile frame initial running fresh proof fits encoded

/-- The typed completion preserves the selected call's constant-one wire. -/
theorem assignment_constantWire
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (constantWire : initial frame.one = 1) :
    assignment application profile frame initial running fresh proof
        frame.one = 1 := by
  have preserved :=
    ConcreteNifsNumericCompletion.complete_agrees_visible frame
      (ConcreteNifsOperationalSelectedHonest.seededAssignment
        application profile frame initial running fresh proof)
      (operationalNumeric application profile frame initial
        running fresh proof)
  have onePreserved :=
    preserved frame.one (by simp [CallFrame.visibleIds])
  have seedOne :
      ConcreteNifsOperationalSelectedHonest.seededAssignment
          application profile frame initial running fresh proof frame.one =
        1 := by
    exact
      (ConcreteNifsClaimedValuesHonest.seed_agrees_visible
        application profile frame initial
        (ConcreteNifsOperationalSelectedHonest.values
          (keys := keys) running fresh proof)
        fits frame.one (by simp [CallFrame.visibleIds])).trans constantWire
  exact onePreserved.trans seedOne

/-- The typed numeric assignment agrees with the canonical operational witness
through the complete selected operational allocation. -/
theorem numericAssignment_eq_operationalNumeric
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (source : Nat) (sourceBound : source < (orderedIds frame).length) :
    numericAssignment (columnMap frame)
        (assignment application profile frame initial running fresh proof)
        source =
      operationalNumeric application profile frame initial running fresh proof
        source := by
  exact ConcreteNifsNumericCompletion.numericAssignment_complete_of_lt
    frame
    (ConcreteNifsOperationalSelectedHonest.seededAssignment
      application profile frame initial running fresh proof)
    (operationalNumeric application profile frame initial running fresh proof)
    (ConcreteNifsNumericCompletion.canonicalize_lt _)
    (operationalNumeric_before_temporaryBase
      application profile frame initial running fresh proof)
    source sourceBound

/-- One accepted selected ΠCCS proof yields a typed physical assignment
satisfying the exact operational row prefix. -/
theorem assignment_satisfies
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (encoded :
      frame.operands.Encodes (FamilyFor application) initial
        (.cons running (.cons fresh (.cons proof .nil))))
    (constantWire : initial frame.one = 1)
    (selectedAccepted :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsAccepted
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate) :
    Satisfies
      (ConcreteNifsOperationalOccurrence.rows application profile frame)
      (numericAssignment (columnMap frame)
        (assignment application profile frame initial running fresh proof)) := by
  let witness :=
    ConcreteNifsOperationalHonest.retargetedWitness
      application profile frame proof.piCcsInput
      (ConcreteNifsOperationalSelectedHonest.seededNumericAssignment
        application profile frame initial running fresh proof)
  let canonical :=
    operationalNumeric application profile frame initial running fresh proof
  have satisfiedWitness :=
    ConcreteNifsOperationalSelectedComplete.selectedRows_honest
      application profile frame initial running fresh proof fits encoded
      constantWire selectedAccepted
  have satisfiedCanonical :
      Satisfies
        (ConcreteNifsOperationalOccurrence.rows application profile frame)
        canonical := by
    exact ConcreteNifsNumericCompletion.satisfies_canonicalize
      (ConcreteNifsOperationalOccurrence.rows application profile frame)
      witness satisfiedWitness
  apply KHornerSupport.satisfies_extend _
    canonical
    (numericAssignment (columnMap frame)
      (assignment application profile frame initial running fresh proof))
  · intro row member column mentioned
    have belowSampler :=
      ConcreteNifsEndpointConservation.operationalRows_below_afterAllocation
        application profile frame row member column mentioned
    exact (numericAssignment_eq_operationalNumeric
      application profile frame initial running fresh proof column
      (Nat.lt_of_lt_of_le belowSampler
        (samplerBase_le_orderedLength application profile frame fits))).symm
  · exact satisfiedCanonical

/-- The typed honest assignment decodes to the selected ΠRLC sampler state. -/
theorem decodedSamplerInitial_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (initial : ColumnId → Field)
    (running :
      SelectedRunning shape publicRingColumns publicFits verifierRows)
    (fresh :
      SelectedFresh shape publicRingColumns publicFits verifierRows)
    (proof :
      SelectedProof shape TranscriptState publicRingColumns publicFits
        verifierRows)
    (fits : ConcreteNifsRawProgram.FrameFits application profile frame)
    (encoded :
      frame.operands.Encodes (FamilyFor application) initial
        (.cons running (.cons fresh (.cons proof .nil))))
    (constantWire : initial frame.one = 1)
    (selectedAccepted :
      Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.PiCcsAccepted
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate) :
    SymbolicDuplexSemantics.decodedBuilder
        (numericAssignment (columnMap frame)
          (assignment application profile frame initial running fresh proof))
        (PiRlcCanonicalSymbolicMachineHonest.initialBuilder
          (ConcreteNifsOperationalSampler.samplerLanes
            application profile frame)) =
      (Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.derive
        (ConcreteNifsParameters.context
          (keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected)
          running fresh proof).materialize
        proof.certificate).piRlcInitialState := by
  let completed :=
    assignment application profile frame initial running fresh proof
  have encodedCompleted :=
    assignment_encodes application profile frame initial running fresh proof
      fits encoded
  have decodedCompleted :=
    frame.operands.decodes_of_encodes
      (FamilyFor application) completed
      (.cons running (.cons fresh (.cons proof .nil))) encodedCompleted
  have constantCompleted :=
    assignment_constantWire application profile frame initial running fresh
      proof fits constantWire
  have satisfiedCompleted :=
    assignment_satisfies application profile frame initial running fresh proof
      fits encoded constantWire selectedAccepted
  exact
    (ConcreteNifsOperationalSamplerSelected.decodedSamplerInitial_eq
      application profile frame
      (numericAssignment (columnMap frame) completed)).trans
      (ConcreteNifsOperationalSelected.selectedPiRlcInitialState_eq
        application profile frame completed running fresh proof
        constantCompleted decodedCompleted satisfiedCompleted)

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalTypedHonest
