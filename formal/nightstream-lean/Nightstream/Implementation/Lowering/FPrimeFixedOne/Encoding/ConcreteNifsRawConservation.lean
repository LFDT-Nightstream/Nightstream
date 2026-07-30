import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsAllocationCoverage
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalSamplerConservation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecAudit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionConservation
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcPointAudit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputAudit
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityAudit

/-!
Contract: exact column conservation for the complete selected raw
`nifsVerify` program.

The raw verifier owns the temporary prefix declared by
`ConcreteNifsRawProgram.allocation`.  Activation residuals occupy the
following suffix.  This module proves that every raw row reads only the
authoritative visible coordinates or that exact prefix; it does not infer
freshness from row/allocation counts or from the enclosing frame's larger
temporary bundle.

Emits constraints: none.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawConservation

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
open Nightstream.Implementation.R1CS.Canonical.KCompositeRowSupport
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc

private abbrev TranscriptState := Poseidon2Duplex.State

/-- Every operand of every numeric row lies strictly before one source
boundary. -/
def NumericRowsBelow
    (boundary : Nat)
    (rows : List Nightstream.Implementation.R1CS.Row) : Prop :=
  ∀ row, row ∈ rows → ∀ column,
    Mentions row.a column ∨ Mentions row.b column ∨ Mentions row.c column →
      column < boundary

private theorem fCarried_below
    {columnMap : Nat → ColumnId}
    {typed : PaperNifsCodecProjection.FColumnId}
    (location : PaperNifsCallColumnMap.FLocation columnMap typed)
    (boundary column : Nat)
    (locationBelow : location.numeric < boundary)
    (mentioned : Mentions location.carried column) :
    column < boundary := by
  have same : column = location.numeric := by
    simpa [PaperNifsCallColumnMap.FLocation.carried, Mentions] using
      mentioned
  omega

private theorem kCarriedLow_below
    {columnMap : Nat → ColumnId}
    {typed : PaperNifsCodecProjection.KColumnIds}
    (location : PaperNifsCallColumnMap.KLocation columnMap typed)
    (boundary column : Nat)
    (locationBelow : location.numeric.c0 < boundary)
    (mentioned : Mentions location.carried.low column) :
    column < boundary := by
  have same : column = location.numeric.c0 := by
    simpa [PaperNifsCallColumnMap.KLocation.carried, Mentions] using
      mentioned
  omega

private theorem kCarriedHigh_below
    {columnMap : Nat → ColumnId}
    {typed : PaperNifsCodecProjection.KColumnIds}
    (location : PaperNifsCallColumnMap.KLocation columnMap typed)
    (boundary column : Nat)
    (locationBelow : location.numeric.c1 < boundary)
    (mentioned : Mentions location.carried.high column) :
    column < boundary := by
  have same : column = location.numeric.c1 := by
    simpa [PaperNifsCallColumnMap.KLocation.carried, Mentions] using
      mentioned
  omega

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

private theorem outputAuthoritative_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (column : Nat)
    (mentioned :
      ConcreteNifsOutputAudit.AuthoritativeMention
        application profile frame column) :
    column < temporaryBase frame := by
  rcases mentioned with
      ⟨child, row, lane, output | proof⟩
    | ⟨child, publicCoordinate, output | proof⟩
    | ⟨child, coordinate, outputLow | outputHigh
        | parentLow | parentHigh⟩
    | ⟨child, matrix, lane, outputLow | outputHigh
        | proofLow | proofHigh⟩
  · unfold ConcreteNifsOutputRows.outputChildCommitment at output
    exact fCarried_below _ _ _
      (ConcreteNifsCarrierFrame.outputFLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childCommitment child row lane))
      output
  · unfold ConcreteNifsOutputRows.proofChildCommitment at proof
    exact fCarried_below _ _ _
      (ConcreteNifsCarrierFrame.proofFLocation_numeric_lt
        (FamilyFor application) frame
        (profile.payloadViews.commitment child row lane))
      proof
  · unfold ConcreteNifsOutputRows.outputChildPublic at output
    exact fCarried_below _ _ _
      (ConcreteNifsCarrierFrame.outputFLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childPublic child publicCoordinate))
      output
  · unfold ConcreteNifsOutputRows.proofChildPublic at proof
    exact fCarried_below _ _ _
      (ConcreteNifsCarrierFrame.proofFLocation_numeric_lt
        (FamilyFor application) frame
        (profile.payloadViews.publicInput child publicCoordinate))
      proof
  · unfold ConcreteNifsOutputRows.outputChildPoint at outputLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childPoint child coordinate)).1
      outputLow
  · unfold ConcreteNifsOutputRows.outputChildPoint at outputHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childPoint child coordinate)).2
      outputHigh
  · unfold ConcreteNifsOutputRows.outputParentPoint at parentLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.parentPoint coordinate)).1
      parentLow
  · unfold ConcreteNifsOutputRows.outputParentPoint at parentHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.parentPoint coordinate)).2
      parentHigh
  · unfold ConcreteNifsOutputRows.outputChildEvaluation at outputLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childEvaluation child matrix lane)).1
      outputLow
  · unfold ConcreteNifsOutputRows.outputChildEvaluation at outputHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childEvaluation child matrix lane)).2
      outputHigh
  · unfold ConcreteNifsOutputRows.proofChildEvaluation at proofLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.proofKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.payloadViews.evaluation child matrix lane)).1
      proofLow
  · unfold ConcreteNifsOutputRows.proofChildEvaluation at proofHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.proofKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.payloadViews.evaluation child matrix lane)).2
      proofHigh

/-- Output-materialization rows use only authoritative visible coordinates
and the constant wire. -/
theorem outputRows_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    NumericRowsBelow
      (temporaryBase frame)
      (ConcreteNifsOutputRows.rows application profile frame) := by
  intro row member column mentioned
  rcases ConcreteNifsOutputAudit.rows_conservation
      application profile frame row member column mentioned with
    rfl | authoritative
  · have positive : 0 < temporaryBase frame := by
      unfold temporaryBase visibleIds
      simp
    exact positive
  · exact outputAuthoritative_below
      application profile frame column authoritative

private theorem piDecFCoordinate_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (coordinate : Phi81RadixRows.FCoordinate)
    (member :
      coordinate ∈
        ConcreteNifsPiDecRows.fCoordinates application profile frame)
    (column : Nat)
    (mentioned :
      (∃ child, Mentions (coordinate.children child) column)
        ∨ Mentions coordinate.parent column) :
    column < temporaryBase frame := by
  unfold ConcreteNifsPiDecRows.fCoordinates at member
  rcases List.mem_append.1 member with inCommitment | inPublic
  · unfold ConcreteNifsPiDecRows.commitmentCoordinates at inCommitment
    rcases List.mem_flatMap.1 inCommitment with
      ⟨row, _, inLanes⟩
    rcases List.mem_ofFn.1 inLanes with ⟨lane, rfl⟩
    rcases mentioned with ⟨child, childMention⟩ | parentMention
    · unfold ConcreteNifsPiDecRows.commitmentCoordinate at childMention
      exact fCarried_below _ _ _
        (ConcreteNifsCarrierFrame.proofFLocation_numeric_lt
          (FamilyFor application) frame
          (profile.payloadViews.commitment child row lane))
        childMention
    · unfold ConcreteNifsPiDecRows.commitmentCoordinate at parentMention
      exact fCarried_below _ _ _
        (ConcreteNifsCarrierFrame.outputFLocation_numeric_lt
          (FamilyFor application) frame
          (profile.runningViews.parentCommitment row lane))
        parentMention
  · unfold ConcreteNifsPiDecRows.publicCoordinates at inPublic
    rcases List.mem_ofFn.1 inPublic with ⟨publicCoordinate, rfl⟩
    rcases mentioned with ⟨child, childMention⟩ | parentMention
    · unfold ConcreteNifsPiDecRows.publicCoordinate at childMention
      exact fCarried_below _ _ _
        (ConcreteNifsCarrierFrame.proofFLocation_numeric_lt
          (FamilyFor application) frame
          (profile.payloadViews.publicInput child publicCoordinate))
        childMention
    · unfold ConcreteNifsPiDecRows.publicCoordinate at parentMention
      exact fCarried_below _ _ _
        (ConcreteNifsCarrierFrame.outputFLocation_numeric_lt
          (FamilyFor application) frame
          (profile.runningViews.parentPublic publicCoordinate))
        parentMention

private theorem piDecKCoordinate_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (coordinate : Phi81RadixRows.KCoordinate)
    (member :
      coordinate ∈
        ConcreteNifsPiDecRows.evaluationCoordinates
          application profile frame)
    (column : Nat)
    (mentioned :
      (∃ child,
        Mentions (coordinate.children child).low column
          ∨ Mentions (coordinate.children child).high column)
        ∨ Mentions coordinate.parent.low column
        ∨ Mentions coordinate.parent.high column) :
    column < temporaryBase frame := by
  unfold ConcreteNifsPiDecRows.evaluationCoordinates at member
  rcases List.mem_flatMap.1 member with
    ⟨matrix, _, inLanes⟩
  rcases List.mem_ofFn.1 inLanes with ⟨lane, rfl⟩
  rcases mentioned with
    ⟨child, childLow | childHigh⟩ | parentLow | parentHigh
  · unfold ConcreteNifsPiDecRows.evaluationCoordinate at childLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.proofKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.payloadViews.evaluation child matrix lane)).1
      childLow
  · unfold ConcreteNifsPiDecRows.evaluationCoordinate at childHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.proofKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.payloadViews.evaluation child matrix lane)).2
      childHigh
  · unfold ConcreteNifsPiDecRows.evaluationCoordinate at parentLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.parentEvaluation matrix lane)).1
      parentLow
  · unfold ConcreteNifsPiDecRows.evaluationCoordinate at parentHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.parentEvaluation matrix lane)).2
      parentHigh

/-- Outgoing ΠDEC rows use only authoritative proof/output coordinates and
the constant wire. -/
theorem piDecRows_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    NumericRowsBelow
      (temporaryBase frame)
      (ConcreteNifsPiDecRows.rows application profile frame) := by
  intro row member column mentioned
  rcases ConcreteNifsPiDecAudit.rows_conservation
      application profile frame row member column mentioned with
    rfl | ⟨coordinate, coordinateMember, support⟩
      | ⟨coordinate, coordinateMember, support⟩
  · have positive : 0 < temporaryBase frame := by
      unfold temporaryBase visibleIds
      simp
    exact positive
  · exact piDecFCoordinate_below application profile frame
      coordinate coordinateMember column support
  · exact piDecKCoordinate_below application profile frame
      coordinate coordinateMember column support

private theorem runningPointPair_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (pair :
      Nightstream.Implementation.R1CS.Canonical.KMul.Carried ×
        Nightstream.Implementation.R1CS.Canonical.KMul.Carried)
    (member :
      pair ∈
        ConcreteNifsRunningAuthorityRows.pointPairs
          application profile frame)
    (column : Nat)
    (mentioned :
      Mentions pair.1.low column ∨ Mentions pair.1.high column
        ∨ Mentions pair.2.low column ∨ Mentions pair.2.high column) :
    column < temporaryBase frame := by
  unfold ConcreteNifsRunningAuthorityRows.pointPairs at member
  rcases List.mem_flatMap.1 member with
    ⟨child, _, inCoordinates⟩
  rcases List.mem_ofFn.1 inCoordinates with ⟨coordinate, rfl⟩
  rcases mentioned with childLow | childHigh | parentLow | parentHigh
  · unfold ConcreteNifsRunningAuthorityRows.pointPair at childLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.runningKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childPoint child coordinate)).1
      childLow
  · unfold ConcreteNifsRunningAuthorityRows.pointPair at childHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.runningKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childPoint child coordinate)).2
      childHigh
  · unfold ConcreteNifsRunningAuthorityRows.pointPair at parentLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.runningKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.parentPoint coordinate)).1
      parentLow
  · unfold ConcreteNifsRunningAuthorityRows.pointPair at parentHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.runningKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.parentPoint coordinate)).2
      parentHigh

private theorem runningFCoordinate_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (coordinate : Phi81RadixRows.FCoordinate)
    (member :
      coordinate ∈
        ConcreteNifsRunningAuthorityRows.fCoordinates
          application profile frame)
    (column : Nat)
    (mentioned :
      (∃ child, Mentions (coordinate.children child) column)
        ∨ Mentions coordinate.parent column) :
    column < temporaryBase frame := by
  unfold ConcreteNifsRunningAuthorityRows.fCoordinates at member
  rcases List.mem_append.1 member with inCommitment | inPublic
  · unfold ConcreteNifsRunningAuthorityRows.commitmentCoordinates at inCommitment
    rcases List.mem_flatMap.1 inCommitment with
      ⟨row, _, inLanes⟩
    rcases List.mem_ofFn.1 inLanes with ⟨lane, rfl⟩
    rcases mentioned with ⟨child, childMention⟩ | parentMention
    · unfold ConcreteNifsRunningAuthorityRows.commitmentCoordinate at childMention
      exact fCarried_below _ _ _
        (ConcreteNifsCarrierFrame.runningFLocation_numeric_lt
          (FamilyFor application) frame
          (profile.runningViews.childCommitment child row lane))
        childMention
    · unfold ConcreteNifsRunningAuthorityRows.commitmentCoordinate at parentMention
      exact fCarried_below _ _ _
        (ConcreteNifsCarrierFrame.runningFLocation_numeric_lt
          (FamilyFor application) frame
          (profile.runningViews.parentCommitment row lane))
        parentMention
  · unfold ConcreteNifsRunningAuthorityRows.publicCoordinates at inPublic
    rcases List.mem_ofFn.1 inPublic with ⟨publicCoordinate, rfl⟩
    rcases mentioned with ⟨child, childMention⟩ | parentMention
    · unfold ConcreteNifsRunningAuthorityRows.publicCoordinate at childMention
      exact fCarried_below _ _ _
        (ConcreteNifsCarrierFrame.runningFLocation_numeric_lt
          (FamilyFor application) frame
          (profile.runningViews.childPublic child publicCoordinate))
        childMention
    · unfold ConcreteNifsRunningAuthorityRows.publicCoordinate at parentMention
      exact fCarried_below _ _ _
        (ConcreteNifsCarrierFrame.runningFLocation_numeric_lt
          (FamilyFor application) frame
          (profile.runningViews.parentPublic publicCoordinate))
        parentMention

private theorem runningKCoordinate_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (coordinate : Phi81RadixRows.KCoordinate)
    (member :
      coordinate ∈
        ConcreteNifsRunningAuthorityRows.evaluationCoordinates
          application profile frame)
    (column : Nat)
    (mentioned :
      (∃ child,
        Mentions (coordinate.children child).low column
          ∨ Mentions (coordinate.children child).high column)
        ∨ Mentions coordinate.parent.low column
        ∨ Mentions coordinate.parent.high column) :
    column < temporaryBase frame := by
  unfold ConcreteNifsRunningAuthorityRows.evaluationCoordinates at member
  rcases List.mem_flatMap.1 member with
    ⟨matrix, _, inLanes⟩
  rcases List.mem_ofFn.1 inLanes with ⟨lane, rfl⟩
  rcases mentioned with
    ⟨child, childLow | childHigh⟩ | parentLow | parentHigh
  · unfold ConcreteNifsRunningAuthorityRows.evaluationCoordinate at childLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.runningKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childEvaluation child matrix lane)).1
      childLow
  · unfold ConcreteNifsRunningAuthorityRows.evaluationCoordinate at childHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.runningKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childEvaluation child matrix lane)).2
      childHigh
  · unfold ConcreteNifsRunningAuthorityRows.evaluationCoordinate at parentLow
    exact kCarriedLow_below _ _ _
      (ConcreteNifsCarrierFrame.runningKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.parentEvaluation matrix lane)).1
      parentLow
  · unfold ConcreteNifsRunningAuthorityRows.evaluationCoordinate at parentHigh
    exact kCarriedHigh_below _ _ _
      (ConcreteNifsCarrierFrame.runningKLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.parentEvaluation matrix lane)).2
      parentHigh

/-- Incoming running-authority rows read only the authoritative running
operand and the constant wire. -/
theorem runningAuthorityRows_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    NumericRowsBelow
      (temporaryBase frame)
      (ConcreteNifsRunningAuthorityRows.rows application profile frame) := by
  intro row member column mentioned
  rcases ConcreteNifsRunningAuthorityAudit.rows_conservation
      application profile frame row member column mentioned with
    rfl | ⟨pair, pairMember, support⟩
      | ⟨coordinate, coordinateMember, support⟩
      | ⟨coordinate, coordinateMember, support⟩
  · have positive : 0 < temporaryBase frame := by
      unfold temporaryBase visibleIds
      simp
    exact positive
  · exact runningPointPair_below application profile frame
      pair pairMember column support
  · exact runningFCoordinate_below application profile frame
      coordinate coordinateMember column support
  · exact runningKCoordinate_below application profile frame
      coordinate coordinateMember column support

private theorem actionEnd_eq_rawEnd
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ConcreteNifsRawProgram.actionBase application profile frame +
        (ConcreteNifsPiRlcActionRows.cost
          shape publicRingColumns verifierRows).auxiliaryColumns =
      temporaryBase frame +
        ConcreteNifsRawProgram.allocationWidth
          application profile frame := by
  rw [ConcreteNifsAllocationCoverage.actionBase_eq_temporarySource
    application profile frame]
  unfold ConcreteNifsRawProgram.allocationWidth temporarySource
  omega

private theorem temporaryBase_le_rawEnd
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
      temporaryBase frame +
        ConcreteNifsRawProgram.allocationWidth
          application profile frame := by
  exact Nat.le_add_right _ _

private theorem actionBase_le_rawEnd
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    ConcreteNifsRawProgram.actionBase application profile frame ≤
      temporaryBase frame +
        ConcreteNifsRawProgram.allocationWidth
          application profile frame := by
  have endEqual := actionEnd_eq_rawEnd application profile frame
  omega

private theorem endpointBase_le_rawEnd
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    KSplitNcOperationalRows.endpointBase
        (ConcreteNifsOperationalOccurrence.input
          application profile frame) ≤
      temporaryBase frame +
        ConcreteNifsRawProgram.allocationWidth
          application profile frame := by
  let input :=
    ConcreteNifsOperationalOccurrence.input application profile frame
  have endpointToOperational :
      KSplitNcOperationalRows.endpointBase input ≤
        KSplitNcOperationalRows.afterAllocation input := by
    rw [KSplitNcOperationalRows.afterAllocation_eq_endpoint_end]
    exact Nat.le_add_right _ _
  have operationalToAction :
      KSplitNcOperationalRows.afterAllocation input ≤
        ConcreteNifsRawProgram.actionBase
          application profile frame := by
    unfold ConcreteNifsRawProgram.actionBase
      ConcreteNifsOperationalSampler.samplerBase
    exact Nat.le_add_right _ _
  exact Nat.le_trans endpointToOperational
    (Nat.le_trans operationalToAction
      (actionBase_le_rawEnd application profile frame))

/-- The direct ΠRLC point rows consume only the selected FE-row transcript
challenge, the authoritative output point, and the constant wire.  Each is
strictly before the exact end of the raw allocation. -/
theorem piRlcPointRows_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    NumericRowsBelow
      (temporaryBase frame +
        ConcreteNifsRawProgram.allocationWidth application profile frame)
      (ConcreteNifsPiRlcPointRows.rows application profile frame) := by
  intro row member column mentioned
  rcases ConcreteNifsPiRlcPointAudit.rows_conservation
      application profile frame row member column mentioned with
    rfl | ⟨coordinate, transcriptLow | transcriptHigh⟩
      | ⟨coordinate, outputLow | outputHigh⟩
  · have positive : 0 < temporaryBase frame := by
      unfold temporaryBase visibleIds
      simp
    exact Nat.lt_of_lt_of_le positive
      (temporaryBase_le_rawEnd application profile frame)
  · have inputs :=
      ConcreteNifsEndpointConservation.endpointInputs_below
        application profile frame
    have below :
        CarriedBelow
          (ConcreteNifsPiRlcPointRows.transcriptCoordinate
            application profile frame coordinate)
          (KSplitNcOperationalRows.endpointBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame)) := by
      simpa [ConcreteNifsPiRlcPointRows.transcriptCoordinate,
        ConcreteNifsPiRlcPointRows.endpointInput,
        KSplitNcEndpoints.feTerminalInput,
        KSplitNcOperationalRows.endpointInput] using
          inputs.feTerminalPointRow coordinate
    exact Nat.lt_of_lt_of_le (below.1 column transcriptLow)
      (endpointBase_le_rawEnd application profile frame)
  · have inputs :=
      ConcreteNifsEndpointConservation.endpointInputs_below
        application profile frame
    have below :
        CarriedBelow
          (ConcreteNifsPiRlcPointRows.transcriptCoordinate
            application profile frame coordinate)
          (KSplitNcOperationalRows.endpointBase
            (ConcreteNifsOperationalOccurrence.input
              application profile frame)) := by
      simpa [ConcreteNifsPiRlcPointRows.transcriptCoordinate,
        ConcreteNifsPiRlcPointRows.endpointInput,
        KSplitNcEndpoints.feTerminalInput,
        KSplitNcOperationalRows.endpointInput] using
          inputs.feTerminalPointRow coordinate
    exact Nat.lt_of_lt_of_le (below.2 column transcriptHigh)
      (endpointBase_le_rawEnd application profile frame)
  · unfold ConcreteNifsPiRlcPointRows.outputCoordinate at outputLow
    exact Nat.lt_of_lt_of_le
      (kCarriedLow_below _ _ _
        (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
          (FamilyFor application) frame
          (profile.runningViews.parentPoint coordinate)).1
        outputLow)
      (temporaryBase_le_rawEnd application profile frame)
  · unfold ConcreteNifsPiRlcPointRows.outputCoordinate at outputHigh
    exact Nat.lt_of_lt_of_le
      (kCarriedHigh_below _ _ _
        (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
          (FamilyFor application) frame
          (profile.runningViews.parentPoint coordinate)).2
        outputHigh)
      (temporaryBase_le_rawEnd application profile frame)

/-- Numeric sources before the end of the raw allocation translate into the
call's visible set or the exact declared raw prefix. -/
theorem columnMap_before_raw_end
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source : Nat)
    (before :
      source <
        temporaryBase frame +
          ConcreteNifsRawProgram.allocationWidth
            application profile frame) :
    columnMap frame source ∈
      frame.visibleIds ++
        ConcreteNifsRawProgram.allocation application profile frame := by
  by_cases inVisible : source < temporaryBase frame
  · have sourceBound : source < (orderedIds frame).length := by
      rw [orderedIds_eq_visible_append_temporaries, List.length_append]
      unfold temporaryBase at inVisible
      omega
    have visibleBound : source < (visibleIds frame).length := by
      simpa [temporaryBase] using inVisible
    have sourceBoundAppend :
        source <
          (visibleIds frame ++ frame.temporaries.ids).length := by
      simpa only [orderedIds_eq_visible_append_temporaries] using
        sourceBound
    have selected :
        (orderedIds frame)[source]'sourceBound =
          (visibleIds frame)[source]'visibleBound := by
      change
        (visibleIds frame ++ frame.temporaries.ids)[
            source]'sourceBoundAppend =
          (visibleIds frame)[source]'visibleBound
      rw [List.getElem_append_left]
    have mapped :
        columnMap frame source =
          (orderedIds frame)[source]'sourceBound := by
      unfold columnMap
      rw [List.getD_eq_getElem?_getD,
        List.getElem?_eq_getElem sourceBound]
      rfl
    apply List.mem_append_left
    rw [mapped, selected]
    exact visibleIds_supported frame (List.getElem_mem visibleBound)
  · let offset := source - temporaryBase frame
    have sourceEq : source = temporarySource frame offset := by
      unfold offset temporarySource
      omega
    have offsetLt :
        offset <
          ConcreteNifsRawProgram.allocationWidth
            application profile frame := by
      unfold offset
      omega
    apply List.mem_append_right
    unfold ConcreteNifsRawProgram.allocation
      ConcreteNifsRawProgram.allocationSources
    apply List.mem_map.2
    refine ⟨temporarySource frame offset, ?_, ?_⟩
    · exact List.mem_map.2
        ⟨offset, List.mem_range.2 offsetLt, rfl⟩
    · rw [← sourceEq]

private theorem mentions_of_term_member
    (row : Nightstream.Implementation.R1CS.Row)
    (term : Nat × Nat)
    (member : term ∈ row.a ++ row.b ++ row.c) :
    Mentions row.a term.1 ∨ Mentions row.b term.1 ∨
      Mentions row.c term.1 := by
  rcases List.mem_append.1 member with inLeft | inC
  · rcases List.mem_append.1 inLeft with inA | inB
    · exact Or.inl
        (List.mem_map.2 ⟨term, inA, rfl⟩)
    · exact Or.inr (Or.inl
        (List.mem_map.2 ⟨term, inB, rfl⟩))
  · exact Or.inr (Or.inr
      (List.mem_map.2 ⟨term, inC, rfl⟩))

/-- A numeric source-bound theorem transports to exact typed conservation
through the sole call-frame map. -/
theorem translate_supported_of_below
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (sourceRows : List Nightstream.Implementation.R1CS.Row)
    (bounded :
      NumericRowsBelow
        (temporaryBase frame +
          ConcreteNifsRawProgram.allocationWidth application profile frame)
        sourceRows) :
    RawRowsSupportedBy
      (frame.visibleIds ++
        ConcreteNifsRawProgram.allocation application profile frame)
      (ConcreteNifsRawProgram.translate application frame sourceRows) := by
  intro row rowMember column columnMember
  rcases List.mem_map.1 rowMember with
    ⟨numericRow, numericMember, rfl⟩
  rw [NumericRowBridge.row_columnIds] at columnMember
  rcases List.mem_map.1 columnMember with
    ⟨term, termMember, rfl⟩
  exact columnMap_before_raw_end application profile frame term.1
    (bounded numericRow numericMember term.1
      (mentions_of_term_member numericRow term termMember))

/-- The direct action slice is supported by authoritative visible columns
and the action products that occupy the final interval of the raw
allocation. -/
theorem actionRows_supported_exact
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RawRowsSupportedBy
      (frame.visibleIds ++
        ConcreteNifsRawProgram.allocation application profile frame)
      (ConcreteNifsPiRlcActionRows.rows application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame)) := by
  intro row rowMember column columnMember
  rcases ConcreteNifsPiRlcActionAudit.rows_conservation
      application profile frame
      (ConcreteNifsRawProgram.actionBase application profile frame)
      row rowMember column columnMember with visible | product
  · exact List.mem_append_left _ visible
  · rcases ConcreteNifsPiRlcActionConservation.columns_before_end
        application profile frame
        (ConcreteNifsRawProgram.actionBase application profile frame)
        column product with
      ⟨source, rfl, before⟩
    apply columnMap_before_raw_end application profile frame source
    rw [← actionEnd_eq_rawEnd application profile frame]
    exact before

private theorem raw_supported_append
    (allowed : List ColumnId)
    (left right : List Nightstream.Implementation.Lowering.Goldilocks.Row)
    (leftSupported : RawRowsSupportedBy allowed left)
    (rightSupported : RawRowsSupportedBy allowed right) :
    RawRowsSupportedBy allowed (left ++ right) := by
  intro row rowMember column columnMember
  rcases List.mem_append.1 rowMember with inLeft | inRight
  · exact leftSupported row inLeft column columnMember
  · exact rightSupported row inRight column columnMember

/-- Exact raw-program support.  Unlike the earlier enclosing-frame theorem,
this result does not admit unused call-frame temporaries: every dependency is
either an authoritative visible coordinate or one member of the raw
program's own dense allocation. -/
theorem rawRows_supported_exact
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RawRowsSupportedBy
      (frame.visibleIds ++
        ConcreteNifsRawProgram.allocation application profile frame)
      (ConcreteNifsRawProgram.rawRows application profile frame) := by
  let rawEnd :=
    temporaryBase frame +
      ConcreteNifsRawProgram.allocationWidth application profile frame
  have visibleToEnd : temporaryBase frame ≤ rawEnd := by
    exact temporaryBase_le_rawEnd application profile frame
  have runningBelow :
      NumericRowsBelow rawEnd
        (ConcreteNifsRunningAuthorityRows.rows
          application profile frame) := by
    intro row member column mentioned
    exact Nat.lt_of_lt_of_le
      (runningAuthorityRows_below application profile frame
        row member column mentioned)
      visibleToEnd
  have operationalBelow :
      NumericRowsBelow rawEnd
        (ConcreteNifsOperationalSampler.rows
          application profile frame) := by
    intro row member column mentioned
    exact Nat.lt_of_lt_of_le
      (ConcreteNifsOperationalSamplerConservation.rows_below_actionBase
        application profile frame row member column mentioned)
      (actionBase_le_rawEnd application profile frame)
  have pointBelow :
      NumericRowsBelow rawEnd
        (ConcreteNifsPiRlcPointRows.rows application profile frame) := by
    exact piRlcPointRows_below application profile frame
  have piDecBelow :
      NumericRowsBelow rawEnd
        (ConcreteNifsPiDecRows.rows application profile frame) := by
    intro row member column mentioned
    exact Nat.lt_of_lt_of_le
      (piDecRows_below application profile frame
        row member column mentioned)
      visibleToEnd
  have outputBelow :
      NumericRowsBelow rawEnd
        (ConcreteNifsOutputRows.rows application profile frame) := by
    intro row member column mentioned
    exact Nat.lt_of_lt_of_le
      (outputRows_below application profile frame
        row member column mentioned)
      visibleToEnd
  unfold ConcreteNifsRawProgram.rawRows
  apply raw_supported_append
  · apply raw_supported_append
    · apply raw_supported_append
      · apply raw_supported_append
        · apply raw_supported_append
          · apply raw_supported_append
            · intro row rowMember column columnMember
              exact List.mem_append_left _
                (ConcreteNifsProofCanonicalityRows.rows_supported
                  application profile frame row rowMember column columnMember)
            · exact translate_supported_of_below
                application profile frame _ runningBelow
          · exact translate_supported_of_below
              application profile frame _ operationalBelow
        · exact translate_supported_of_below
            application profile frame _ pointBelow
      · exact actionRows_supported_exact application profile frame
    · exact translate_supported_of_below
        application profile frame _ piDecBelow
  · exact translate_supported_of_below
      application profile frame _ outputBelow

/-- Stable owned rows inherit the exact raw-program conservation theorem. -/
theorem rows_supported_exact
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : OwnedRow)
    (member :
      row ∈ ConcreteNifsRawProgram.rows application profile frame)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈
      frame.visibleIds ++
        ConcreteNifsRawProgram.allocation application profile frame :=
  DirectCalls.ownRows_supported frame.owner
    (ConcreteNifsRawProgram.rawRows application profile frame)
    (frame.visibleIds ++
      ConcreteNifsRawProgram.allocation application profile frame)
    (rawRows_supported_exact application profile frame)
    row member column columnMember

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRawConservation
