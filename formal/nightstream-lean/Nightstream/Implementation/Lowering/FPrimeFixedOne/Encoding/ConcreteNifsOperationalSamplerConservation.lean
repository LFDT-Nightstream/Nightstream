import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsEndpointConservation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawProgram
import Nightstream.Implementation.R1CS.Canonical.PiRlcCanonicalSamplerProgramConservation

/-!
Contract: exact numeric-column conservation for the selected operational
ΠCCS plus fixed-active ΠRLC sampler prefix.

The proof connects three independently constructed pieces: the selected
operational ΠCCS allocation, the generic 105,930-row sampler allocation, and
the direct proof-codec challenge bindings.  Every operand stays below
`ConcreteNifsRawProgram.actionBase`; no allocation length is used as a proxy
for a value bound.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerConservation

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.Poseidon2Core
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexHonest
open Nightstream.Implementation.R1CS.Canonical.SymbolicDuplexPlacement
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

private theorem transcriptBase_le_samplerBase
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
      ConcreteNifsOperationalSampler.samplerBase
        application profile frame := by
  change
    (ConcreteNifsOperationalOccurrence.input
        application profile frame).transcript.transcriptBase ≤
      KSplitNcOperationalRows.afterAllocation
        (ConcreteNifsOperationalOccurrence.input application profile frame)
  unfold KSplitNcOperationalRows.afterAllocation
    KSplitNcOperationalRows.allocationWidth
  omega

/-- The selected ΠCCS output state is a genuine caller-owned prefix for the
downstream fixed-active sampler. -/
theorem samplerLanes_inPrefix
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ∀ lane : Fin width,
      ValueInPrefix
        (ConcreteNifsOperationalSampler.samplerBase
          application profile frame)
        (ConcreteNifsOperationalSampler.samplerLanes
          application profile frame lane) := by
  let transcriptInput :=
    ConcreteNifsOperationalOccurrence.transcriptInput
      application profile frame
  have invariant :=
    KSplitNcTranscriptPlacement.outputBuilder_invariant transcriptInput
      (ConcreteNifsOperationalConservation.transcriptInput_inPrefix
        application profile frame)
  intro lane column mentioned
  have belowNumeric :=
    invariant.1.lanesBefore lane column
      (by simpa [transcriptInput,
          ConcreteNifsOperationalSampler.samplerLanes] using mentioned)
  let input :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  have inputTranscript : input.transcript = transcriptInput := rfl
  have belowInputNumeric :
      column < KSplitNcOperationalRows.numericBase input := by
    rw [KSplitNcOperationalRows.numericBase, inputTranscript]
    simpa [SymbolicDuplexHonest.outputBase,
      SymbolicDuplexHonest.callBase, KSplitNcTranscript.replay] using
        belowNumeric
  have numericToSampler :
      KSplitNcOperationalRows.numericBase
          input ≤
      ConcreteNifsOperationalSampler.samplerBase
          application profile frame := by
    change
      KSplitNcOperationalRows.numericBase input ≤
        KSplitNcOperationalRows.afterAllocation input
    unfold KSplitNcOperationalRows.afterAllocation
      KSplitNcOperationalRows.allocationWidth
      KSplitNcOperationalRows.numericBase
    omega
  exact Nat.lt_of_lt_of_le belowInputNumeric numericToSampler

theorem samplerBase_positive
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    0 <
      ConcreteNifsOperationalSampler.samplerBase
        application profile frame := by
  have sourcePositive :=
    (ConcreteNifsOperationalConservation.transcriptInput_inPrefix
      application profile frame).positive
  exact Nat.lt_of_lt_of_le sourcePositive
    (transcriptBase_le_samplerBase application profile frame)

/-- Every fixed-active sampler row lies inside its exact 99,885-column
allocation after the operational ΠCCS prefix. -/
theorem samplerRows_below_actionBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RowsBelow
      (ConcreteNifsOperationalSampler.samplerRows
        application profile frame)
      (ConcreteNifsRawProgram.actionBase application profile frame) := by
  intro row member column mentioned
  let base :=
    ConcreteNifsOperationalSampler.samplerBase application profile frame
  have classified :=
    PiRlcCanonicalSamplerProgramConservation.rows_conservation
      base profile.constants
      (ConcreteNifsOperationalSampler.samplerLanes
        application profile frame)
      (by simpa [base] using
        samplerBase_positive application profile frame)
      (by simpa [base] using
        samplerLanes_inPrefix application profile frame)
      row (by simpa [ConcreteNifsOperationalSampler.samplerRows, base]
        using member)
      column mentioned
  rcases classified with before | allocated
  · unfold ConcreteNifsRawProgram.actionBase
    exact Nat.lt_of_lt_of_le before (Nat.le_add_right _ _)
  · simpa [ConcreteNifsRawProgram.actionBase, base] using
      PiRlcCanonicalSamplerProgram.allocation_lt_end
        base column allocated

private theorem challengeLocation_below_temporaryBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total)
    (position : Fin ringDegree) :
    (ConcreteNifsOperationalSampler.challengeLocation
      application profile frame coordinate position).numeric <
        temporaryBase frame := by
  simpa [ConcreteNifsOperationalSampler.challengeLocation,
    ConcreteNifsOperationalOccurrence.proofFieldLocation,
    ConcreteNifsCarrierFrame.proofFLocation] using
    ConcreteNifsCarrierFrame.proofFLocation_numeric_lt
      (FamilyFor application) frame
      (profile.samplerViews.challenge coordinate position)

private theorem selectorOutput_below_actionBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (coordinate :
      Fin
        Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity.total)
    (position : Fin ringDegree) :
    PiRlcCanonicalSelector.outputColumn
        (PiRlcCanonicalSamplerProgram.selectorBase
          (ConcreteNifsOperationalSampler.samplerBase
            application profile frame))
        (ConcreteNifsOperationalSampler.samplerCoordinate coordinate)
        (ConcreteNifsOperationalSampler.samplerPosition position) <
      ConcreteNifsRawProgram.actionBase application profile frame := by
  have coordinateLt := coordinate.isLt
  have positionLt := position.isLt
  unfold ConcreteNifsRawProgram.actionBase
    PiRlcCanonicalSelector.outputColumn
    PiRlcCanonicalSelector.positionBase
    PiRlcCanonicalSelector.scalarBase
    PiRlcCanonicalSamplerProgram.selectorBase
    PiRlcCanonicalSelectorBatchHonest.candidateEnd
    PiRlcCanonicalSamplerProgram.candidateBase
    PiRlcCanonicalCandidatesBatchHonest.u64End
    PiRlcCanonicalSamplerProgram.u64Base
  simp only [ConcreteNifsOperationalSampler.samplerCoordinate,
    ConcreteNifsOperationalSampler.samplerPosition,
    PiRlcCanonicalSamplerProgram.coordinateCount,
    PiRlcCanonicalSamplerProgram.transcriptCalls,
    SymbolicDuplex.stride,
    PiRlcCanonicalU64.lanesPerScalar,
    CanonicalU64Recipe.auxiliaryCount,
    PiRlcCanonicalCandidates.candidatesPerScalar,
    PiRlcCanonicalCandidate.auxiliaryCount,
    PiRlcCanonicalSelector.scalarAuxiliaryCount,
    PiRlcCanonicalSelector.outputCount,
    PiRlcCanonicalSelector.positionAuxiliaryCount,
    PiRlcCanonicalSamplerProgram.cost,
    Nightstream.SuperNeo.Concrete.ringDegree,
    Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.arity_total]
      at *
  omega

/-- The direct challenge bindings read one allocated selector output and one
authoritative proof-codec coordinate; both precede the action allocation. -/
theorem challengeRows_below_actionBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RowsBelow
      (ConcreteNifsOperationalSampler.challengeRows
        application profile frame)
      (ConcreteNifsRawProgram.actionBase application profile frame) := by
  intro row member column mentioned
  unfold ConcreteNifsOperationalSampler.challengeRows at member
  rcases List.mem_flatten.1 member with ⟨group, groupMember, rowMember⟩
  rcases List.mem_ofFn.1 groupMember with ⟨coordinate, rfl⟩
  rcases List.mem_ofFn.1 rowMember with ⟨position, rfl⟩
  simp only [ConcreteNifsOperationalSampler.challengeRow,
    KEquality.equalityRow] at mentioned
  rcases mentioned with inOutput | inOne | inProof
  · have same :
      column =
        PiRlcCanonicalSelector.outputColumn
          (PiRlcCanonicalSamplerProgram.selectorBase
            (ConcreteNifsOperationalSampler.samplerBase
              application profile frame))
          (ConcreteNifsOperationalSampler.samplerCoordinate coordinate)
          (ConcreteNifsOperationalSampler.samplerPosition position) := by
      simpa [Mentions] using inOutput
    rw [same]
    exact selectorOutput_below_actionBase
      application profile frame coordinate position
  · have same : column = 0 := by
      simpa [Mentions] using inOne
    rw [same]
    exact Nat.lt_of_lt_of_le
      (samplerBase_positive application profile frame)
      (by unfold ConcreteNifsRawProgram.actionBase; omega)
  · have same :
      column =
        (ConcreteNifsOperationalSampler.challengeLocation
          application profile frame coordinate position).numeric := by
      simpa [FLocation.carried, Mentions] using inProof
    rw [same]
    have visible :=
      challengeLocation_below_temporaryBase
        application profile frame coordinate position
    have visibleToSampler :
        temporaryBase frame <
          ConcreteNifsOperationalSampler.samplerBase
            application profile frame := by
      have transcriptBound :=
        transcriptBase_le_samplerBase application profile frame
      have source :
          temporaryBase frame <
            (ConcreteNifsOperationalOccurrence.transcriptInput
              application profile frame).transcriptBase := by
        change temporaryBase frame < temporarySource frame 10
        unfold temporarySource
        omega
      exact Nat.lt_of_lt_of_le source transcriptBound
    unfold ConcreteNifsRawProgram.actionBase
    exact Nat.lt_of_lt_of_le visible
      (Nat.le_trans (Nat.le_of_lt visibleToSampler)
        (Nat.le_add_right _ _))

/-- The complete operational ΠCCS plus ΠRLC prefix stays below the first
action coordinate. -/
theorem rows_below_actionBase
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RowsBelow
      (ConcreteNifsOperationalSampler.rows application profile frame)
      (ConcreteNifsRawProgram.actionBase application profile frame) := by
  intro row member column mentioned
  unfold ConcreteNifsOperationalSampler.rows at member
  rcases List.mem_append.1 member with inPrefix | inChallenge
  · rcases List.mem_append.1 inPrefix with inOperational | inSampler
    · have below :=
        ConcreteNifsEndpointConservation.operationalRows_below_afterAllocation
          application profile frame row inOperational column mentioned
      exact Nat.lt_of_lt_of_le below
        (by
          unfold ConcreteNifsRawProgram.actionBase
            ConcreteNifsOperationalSampler.samplerBase
          exact Nat.le_add_right _ _)
    · exact samplerRows_below_actionBase application profile frame
        row inSampler column mentioned
  · exact challengeRows_below_actionBase application profile frame
      row inChallenge column mentioned

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerConservation
