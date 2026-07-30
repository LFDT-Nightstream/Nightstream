import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionRows
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingActionAudit

/-!
Contract: whole-program column support for the selected public-parent Phi81
actions inside one `nifsVerify` call frame.

Every authoritative read is a direct `columnMap` image of a decoded operand or
output coordinate.  Every product is a direct image of the exact action
allocation.  This module does not own action semantics, activation, or the
surrounding NIFS program.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionAudit

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionRows
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.Nifs
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81
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

private def Allowed
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : List ColumnId :=
  frame.visibleIds ++ frame.temporaries.ids

private def CarriedSupported
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (value : CarriedRing) : Prop :=
  ∀ lane term, term ∈ value lane →
    term.column ∈ Allowed application frame

def CarriedVisible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (value : CarriedRing) : Prop :=
  ∀ lane term, term ∈ value lane →
    term.column ∈ frame.visibleIds

private theorem carriedF_source_visible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source : List (Nat × Nat))
    (sourceBelow :
      ∀ sourceTerm ∈ source,
        sourceTerm.1 < temporaryBase frame)
    (term : Term)
    (member :
      term ∈ ConcreteNifsPiRlcActionRows.carriedF
        application frame source) :
    term.column ∈ frame.visibleIds := by
  unfold ConcreteNifsPiRlcActionRows.carriedF at member
  unfold NumericRowBridge.terms at member
  rcases List.mem_map.1 member with ⟨sourceTerm, sourceMember, rfl⟩
  exact columnMap_before_temporaryBase frame sourceTerm.1
    (sourceBelow sourceTerm sourceMember)

private theorem fLocation_visible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {typed : PaperNifsCodecProjection.FColumnId}
    (location : FLocation (columnMap frame) typed)
    (below : location.numeric < temporaryBase frame)
    (term : Term)
    (member :
      term ∈ ConcreteNifsPiRlcActionRows.carriedF
        application frame location.carried) :
    term.column ∈ frame.visibleIds := by
  apply carriedF_source_visible application frame location.carried
    _ term member
  intro sourceTerm sourceMember
  have same : sourceTerm = (location.numeric, 1) := by
    simpa [FLocation.carried] using sourceMember
  subst sourceTerm
  exact below

private theorem kLocationLow_visible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {typed : PaperNifsCodecProjection.KColumnIds}
    (location : KLocation (columnMap frame) typed)
    (below : location.numeric.c0 < temporaryBase frame)
    (term : Term)
    (member :
      term ∈ ConcreteNifsPiRlcActionRows.carriedF
        application frame location.carried.low) :
    term.column ∈ frame.visibleIds := by
  apply carriedF_source_visible application frame location.carried.low
    _ term member
  intro sourceTerm sourceMember
  have same : sourceTerm = (location.numeric.c0, 1) := by
    simpa [KLocation.carried] using sourceMember
  subst sourceTerm
  exact below

private theorem kLocationHigh_visible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    {typed : PaperNifsCodecProjection.KColumnIds}
    (location : KLocation (columnMap frame) typed)
    (below : location.numeric.c1 < temporaryBase frame)
    (term : Term)
    (member :
      term ∈ ConcreteNifsPiRlcActionRows.carriedF
        application frame location.carried.high) :
    term.column ∈ frame.visibleIds := by
  apply carriedF_source_visible application frame location.carried.high
    _ term member
  intro sourceTerm sourceMember
  have same : sourceTerm = (location.numeric.c1, 1) := by
    simpa [KLocation.carried] using sourceMember
  subst sourceTerm
  exact below

private theorem carriedF_supported
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source : List (Nat × Nat))
    (term : Term)
    (member :
      term ∈ ConcreteNifsPiRlcActionRows.carriedF application frame source) :
    term.column ∈ Allowed application frame := by
  unfold ConcreteNifsPiRlcActionRows.carriedF at member
  unfold NumericRowBridge.terms at member
  rcases List.mem_map.1 member with ⟨sourceTerm, _, rfl⟩
  exact columnMap_supported frame sourceTerm.1

private theorem challenge_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source : Fin FixedActive.arity.total) :
    CarriedSupported application frame
      (ConcreteNifsPiRlcActionRows.challenge application profile frame source) := by
  intro lane term member
  exact carriedF_supported application frame _ term member

private theorem commitmentValue_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Fin verifierRows)
    (source : Fin FixedActive.arity.total) :
    CarriedSupported application frame
      (ConcreteNifsPiRlcActionRows.commitmentValue
        application profile frame row source) := by
  refine Fin.addCases ?_ ?_ source
  · intro freshIndex lane term member
    simp only [ConcreteNifsPiRlcActionRows.commitmentValue,
      Fin.addCases_left] at member
    exact carriedF_supported application frame _ term member
  · intro child lane term member
    simp only [ConcreteNifsPiRlcActionRows.commitmentValue,
      Fin.addCases_right] at member
    exact carriedF_supported application frame _ term member

private theorem publicValue_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (block : Fin publicRingColumns)
    (source : Fin FixedActive.arity.total) :
    CarriedSupported application frame
      (ConcreteNifsPiRlcActionRows.publicValue
        application profile frame block source) := by
  refine Fin.addCases ?_ ?_ source
  · intro freshIndex lane term member
    simp only [ConcreteNifsPiRlcActionRows.publicValue,
      Fin.addCases_left] at member
    exact carriedF_supported application frame _ term member
  · intro child lane term member
    simp only [ConcreteNifsPiRlcActionRows.publicValue,
      Fin.addCases_right] at member
    exact carriedF_supported application frame _ term member

private theorem evaluationValueLow_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount)
    (source : Fin FixedActive.arity.total) :
    CarriedSupported application frame
      (ConcreteNifsPiRlcActionRows.evaluationValueLow
        application profile frame matrix source) := by
  intro lane term member
  exact carriedF_supported application frame _ term member

private theorem evaluationValueHigh_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount)
    (source : Fin FixedActive.arity.total) :
    CarriedSupported application frame
      (ConcreteNifsPiRlcActionRows.evaluationValueHigh
        application profile frame matrix source) := by
  intro lane term member
  exact carriedF_supported application frame _ term member

private theorem commitmentOutput_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Fin verifierRows) :
    CarriedSupported application frame
      (ConcreteNifsPiRlcActionRows.commitmentOutput
        application profile frame row) := by
  intro lane term member
  exact carriedF_supported application frame _ term member

private theorem publicOutput_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (block : Fin publicRingColumns) :
    CarriedSupported application frame
      (ConcreteNifsPiRlcActionRows.publicOutput
        application profile frame block) := by
  intro lane term member
  exact carriedF_supported application frame _ term member

private theorem evaluationOutputLow_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount) :
    CarriedSupported application frame
      (ConcreteNifsPiRlcActionRows.evaluationOutputLow
        application profile frame matrix) := by
  intro lane term member
  exact carriedF_supported application frame _ term member

private theorem evaluationOutputHigh_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount) :
    CarriedSupported application frame
      (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
        application profile frame matrix) := by
  intro lane term member
  exact carriedF_supported application frame _ term member

theorem challenge_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source : Fin FixedActive.arity.total) :
    CarriedVisible application frame
      (ConcreteNifsPiRlcActionRows.challenge application profile frame source) := by
  intro lane term member
  let location :=
    proofFLocation (FamilyFor application) frame
      (profile.samplerViews.challenge source lane)
  exact fLocation_visible application frame location
    (ConcreteNifsCarrierFrame.proofFLocation_numeric_lt
      (FamilyFor application) frame
      (profile.samplerViews.challenge source lane))
    term member

theorem commitmentValue_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Fin verifierRows)
    (source : Fin FixedActive.arity.total) :
    CarriedVisible application frame
      (ConcreteNifsPiRlcActionRows.commitmentValue
        application profile frame row source) := by
  refine Fin.addCases ?_ ?_ source
  · intro freshIndex lane term member
    simp only [ConcreteNifsPiRlcActionRows.commitmentValue,
      Fin.addCases_left] at member
    exact fLocation_visible application frame
      (freshFLocation (FamilyFor application) frame
        (profile.freshViews.commitment row lane))
      (ConcreteNifsCarrierFrame.freshFLocation_numeric_lt
        (FamilyFor application) frame
        (profile.freshViews.commitment row lane))
      term member
  · intro child lane term member
    simp only [ConcreteNifsPiRlcActionRows.commitmentValue,
      Fin.addCases_right] at member
    exact fLocation_visible application frame
      (runningFLocation (FamilyFor application) frame
        (profile.runningViews.childCommitment child row lane))
      (ConcreteNifsCarrierFrame.runningFLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childCommitment child row lane))
      term member

theorem publicValue_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (block : Fin publicRingColumns)
    (source : Fin FixedActive.arity.total) :
    CarriedVisible application frame
      (ConcreteNifsPiRlcActionRows.publicValue
        application profile frame block source) := by
  refine Fin.addCases ?_ ?_ source
  · intro freshIndex lane term member
    simp only [ConcreteNifsPiRlcActionRows.publicValue,
      Fin.addCases_left] at member
    exact fLocation_visible application frame
      (freshFLocation (FamilyFor application) frame
        (profile.freshViews.publicInput
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane)))
      (ConcreteNifsCarrierFrame.freshFLocation_numeric_lt
        (FamilyFor application) frame
        (profile.freshViews.publicInput
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane)))
      term member
  · intro child lane term member
    simp only [ConcreteNifsPiRlcActionRows.publicValue,
      Fin.addCases_right] at member
    exact fLocation_visible application frame
      (runningFLocation (FamilyFor application) frame
        (profile.runningViews.childPublic child
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane)))
      (ConcreteNifsCarrierFrame.runningFLocation_numeric_lt
        (FamilyFor application) frame
        (profile.runningViews.childPublic child
          (ConcreteNifsPiRlcActionRows.publicCoordinate block lane)))
      term member

theorem evaluationValueLow_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount)
    (source : Fin FixedActive.arity.total) :
    CarriedVisible application frame
      (ConcreteNifsPiRlcActionRows.evaluationValueLow
        application profile frame matrix source) := by
  intro lane term member
  let view :=
    profile.endpointViews.outputYRing
      ((keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.alignment.semanticIndex
          source)
      matrix lane
  exact kLocationLow_visible application frame
    (proofKLocation (FamilyFor application) frame view)
    (ConcreteNifsCarrierFrame.proofKLocation_numeric_lt
      (FamilyFor application) frame view).1
    term member

theorem evaluationValueHigh_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount)
    (source : Fin FixedActive.arity.total) :
    CarriedVisible application frame
      (ConcreteNifsPiRlcActionRows.evaluationValueHigh
        application profile frame matrix source) := by
  intro lane term member
  let view :=
    profile.endpointViews.outputYRing
      ((keys
        Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.alignment.semanticIndex
          source)
      matrix lane
  exact kLocationHigh_visible application frame
    (proofKLocation (FamilyFor application) frame view)
    (ConcreteNifsCarrierFrame.proofKLocation_numeric_lt
      (FamilyFor application) frame view).2
    term member

theorem commitmentOutput_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Fin verifierRows) :
    CarriedVisible application frame
      (ConcreteNifsPiRlcActionRows.commitmentOutput
        application profile frame row) := by
  intro lane term member
  exact fLocation_visible application frame
    (outputFLocation (FamilyFor application) frame
      (profile.runningViews.parentCommitment row lane))
    (ConcreteNifsCarrierFrame.outputFLocation_numeric_lt
      (FamilyFor application) frame
      (profile.runningViews.parentCommitment row lane))
    term member

theorem publicOutput_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (block : Fin publicRingColumns) :
    CarriedVisible application frame
      (ConcreteNifsPiRlcActionRows.publicOutput
        application profile frame block) := by
  intro lane term member
  exact fLocation_visible application frame
    (outputFLocation (FamilyFor application) frame
      (profile.runningViews.parentPublic
        (ConcreteNifsPiRlcActionRows.publicCoordinate block lane)))
    (ConcreteNifsCarrierFrame.outputFLocation_numeric_lt
      (FamilyFor application) frame
      (profile.runningViews.parentPublic
        (ConcreteNifsPiRlcActionRows.publicCoordinate block lane)))
    term member

theorem evaluationOutputLow_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount) :
    CarriedVisible application frame
      (ConcreteNifsPiRlcActionRows.evaluationOutputLow
        application profile frame matrix) := by
  intro lane term member
  let view := profile.runningViews.parentEvaluation matrix lane
  exact kLocationLow_visible application frame
    (outputKLocation (FamilyFor application) frame view)
    (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
      (FamilyFor application) frame view).1
    term member

theorem evaluationOutputHigh_visible
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount) :
    CarriedVisible application frame
      (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
        application profile frame matrix) := by
  intro lane term member
  let view := profile.runningViews.parentEvaluation matrix lane
  exact kLocationHigh_visible application frame
    (outputKLocation (FamilyFor application) frame view)
    (ConcreteNifsCarrierFrame.outputKLocation_numeric_lt
      (FamilyFor application) frame view).2
    term member

private theorem familyIds_supported
    {count : Nat}
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (values : Fin count → CarriedRing)
    (supported : ∀ source, CarriedSupported application frame (values source))
    (column : ColumnId)
    (member : column ∈ Phi81RingAction.familyIds values) :
    column ∈ Allowed application frame := by
  rcases List.mem_flatMap.1 member with ⟨source, _, sourceMember⟩
  rcases List.mem_flatMap.1 sourceMember with ⟨lane, _, laneMember⟩
  rcases List.mem_map.1 laneMember with ⟨term, termMember, rfl⟩
  exact supported source lane term termMember

private theorem carriedIds_supported
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (value : CarriedRing)
    (supported : CarriedSupported application frame value)
    (column : ColumnId)
    (member : column ∈ Phi81RingAction.carriedIds value) :
    column ∈ Allowed application frame := by
  rcases List.mem_flatMap.1 member with ⟨lane, _, laneMember⟩
  rcases List.mem_map.1 laneMember with ⟨term, termMember, rfl⟩
  exact supported lane term termMember

private theorem familyIds_visible
    {count : Nat}
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (values : Fin count → CarriedRing)
    (visible : ∀ source, CarriedVisible application frame (values source))
    (column : ColumnId)
    (member : column ∈ Phi81RingAction.familyIds values) :
    column ∈ frame.visibleIds := by
  rcases List.mem_flatMap.1 member with ⟨source, _, sourceMember⟩
  rcases List.mem_flatMap.1 sourceMember with ⟨lane, _, laneMember⟩
  rcases List.mem_map.1 laneMember with ⟨term, termMember, rfl⟩
  exact visible source lane term termMember

private theorem carriedIds_visible
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (value : CarriedRing)
    (visible : CarriedVisible application frame value)
    (column : ColumnId)
    (member : column ∈ Phi81RingAction.carriedIds value) :
    column ∈ frame.visibleIds := by
  rcases List.mem_flatMap.1 member with ⟨lane, _, laneMember⟩
  rcases List.mem_map.1 laneMember with ⟨term, termMember, rfl⟩
  exact visible lane term termMember

private theorem actionFrame_conservation
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase target : Nat)
    (values : Fin FixedActive.arity.total → CarriedRing)
    (output : CarriedRing)
    (challenges : Fin FixedActive.arity.total → CarriedRing)
    (challengesVisible :
      ∀ source, CarriedVisible application frame (challenges source))
    (valuesVisible :
      ∀ source, CarriedVisible application frame (values source))
    (outputVisible : CarriedVisible application frame output)
    (row : Row)
    (rowMember :
      row ∈ Phi81RingAction.rawRows
        {
          owner := frame.owner
          firstOrdinal :=
            target * Phi81RingAction.rowCount FixedActive.arity.total
          one := frame.one
          challenges := challenges
          values := values
          output := output
          productColumn := fun source left right =>
            columnMap frame
              (ConcreteNifsPiRlcActionRows.productSource
                productBase target source left right)
        })
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ frame.visibleIds
      ∨ column ∈
        Phi81RingAction.productIds
          {
            owner := frame.owner
            firstOrdinal :=
              target * Phi81RingAction.rowCount FixedActive.arity.total
            one := frame.one
            challenges := challenges
            values := values
            output := output
            productColumn := fun source left right =>
              columnMap frame
                (ConcreteNifsPiRlcActionRows.productSource
                  productBase target source left right)
          } := by
  let actionFrame : Phi81RingAction.Frame FixedActive.arity.total := {
    owner := frame.owner
    firstOrdinal :=
      target * Phi81RingAction.rowCount FixedActive.arity.total
    one := frame.one
    challenges := challenges
    values := values
    output := output
    productColumn := fun source left right =>
      columnMap frame
        (ConcreteNifsPiRlcActionRows.productSource
          productBase target source left right)
  }
  have allowed :=
    Phi81RingAction.rawRows_supported actionFrame row rowMember
      column columnMember
  rcases List.mem_append.1 allowed with inVisible | inProducts
  · unfold Phi81RingAction.visibleIds at inVisible
    rcases List.mem_cons.1 inVisible with isOne | inFamilies
    · left
      subst column
      change frame.one ∈ frame.visibleIds
      simp [CallFrame.visibleIds]
    · rcases List.mem_append.1 inFamilies with inInputs | inOutput
      · rcases List.mem_append.1 inInputs with inChallenge | inValue
        · exact Or.inl
            (familyIds_visible application frame challenges
              challengesVisible column inChallenge)
        · exact Or.inl
            (familyIds_visible application frame values
              valuesVisible column inValue)
      · exact Or.inl
          (carriedIds_visible application frame output outputVisible
            column inOutput)
  · exact Or.inr inProducts

private theorem actionFrame_supported
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase target : Nat)
    (values : Fin FixedActive.arity.total → CarriedRing)
    (output : CarriedRing)
    (challenges : Fin FixedActive.arity.total → CarriedRing)
    (challengesSupported :
      ∀ source, CarriedSupported application frame (challenges source))
    (valuesSupported :
      ∀ source, CarriedSupported application frame (values source))
    (outputSupported : CarriedSupported application frame output)
    (row : Row)
    (rowMember :
      row ∈ Phi81RingAction.rawRows
        {
          owner := frame.owner
          firstOrdinal :=
            target * Phi81RingAction.rowCount FixedActive.arity.total
          one := frame.one
          challenges := challenges
          values := values
          output := output
          productColumn := fun source left right =>
            columnMap frame
              (ConcreteNifsPiRlcActionRows.productSource
                productBase target source left right)
        })
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ Allowed application frame := by
  let actionFrame : Phi81RingAction.Frame FixedActive.arity.total := {
    owner := frame.owner
    firstOrdinal :=
      target * Phi81RingAction.rowCount FixedActive.arity.total
    one := frame.one
    challenges := challenges
    values := values
    output := output
    productColumn := fun source left right =>
      columnMap frame
        (ConcreteNifsPiRlcActionRows.productSource
          productBase target source left right)
  }
  have allowed :=
    Phi81RingAction.rawRows_supported actionFrame row rowMember
      column columnMember
  rcases List.mem_append.1 allowed with inVisible | inProducts
  · unfold Phi81RingAction.visibleIds at inVisible
    rcases List.mem_cons.1 inVisible with isOne | inFamilies
    · subst column
      change frame.one ∈ Allowed application frame
      unfold Allowed
      apply List.mem_append.2
      left
      simp [CallFrame.visibleIds]
    · rcases List.mem_append.1 inFamilies with inInputs | inOutput
      · rcases List.mem_append.1 inInputs with inChallenge | inValue
        · exact familyIds_supported application frame challenges
            challengesSupported column inChallenge
        · exact familyIds_supported application frame values
            valuesSupported column inValue
      · exact carriedIds_supported application frame output outputSupported
          column inOutput
  · unfold Phi81RingAction.productIds at inProducts
    rcases List.mem_flatMap.1 inProducts with
      ⟨source, _, sourceMember⟩
    rcases List.mem_flatMap.1 sourceMember with
      ⟨left, _, leftMember⟩
    rcases List.mem_map.1 leftMember with ⟨right, _, rfl⟩
    exact columnMap_supported frame
      (ConcreteNifsPiRlcActionRows.productSource
        productBase target source left right)

/-- Every selected action row dependency belongs to the enclosing call's
authoritative visible set or exact temporary bundle. -/
theorem rows_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat) :
    RawRowsSupportedBy
      (frame.visibleIds ++ frame.temporaries.ids)
      (ConcreteNifsPiRlcActionRows.rows
        application profile frame productBase) := by
  intro row rowMember column columnMember
  rcases List.mem_flatMap.1 rowMember with
    ⟨action, actionMember, rowInAction⟩
  unfold ConcreteNifsPiRlcActionRows.frames at actionMember
  rcases List.mem_append.1 actionMember with firstThree | inHigh
  rcases List.mem_append.1 firstThree with firstTwo | inLow
  rcases List.mem_append.1 firstTwo with inCommitment | inPublic
  · rcases List.mem_ofFn.1 inCommitment with ⟨target, actionEq⟩
    rw [← actionEq] at rowInAction
    exact actionFrame_supported application frame productBase target.val
      (ConcreteNifsPiRlcActionRows.commitmentValue
        application profile frame target)
      (ConcreteNifsPiRlcActionRows.commitmentOutput
        application profile frame target)
      (ConcreteNifsPiRlcActionRows.challenge application profile frame)
      (challenge_supported application profile frame)
      (commitmentValue_supported application profile frame target)
      (commitmentOutput_supported application profile frame target)
      row rowInAction column columnMember
  · rcases List.mem_ofFn.1 inPublic with ⟨block, actionEq⟩
    rw [← actionEq] at rowInAction
    exact actionFrame_supported application frame productBase
      (verifierRows + block.val)
      (ConcreteNifsPiRlcActionRows.publicValue
        application profile frame block)
      (ConcreteNifsPiRlcActionRows.publicOutput
        application profile frame block)
      (ConcreteNifsPiRlcActionRows.challenge application profile frame)
      (challenge_supported application profile frame)
      (publicValue_supported application profile frame block)
      (publicOutput_supported application profile frame block)
      row rowInAction column columnMember
  · rcases List.mem_ofFn.1 inLow with ⟨matrix, actionEq⟩
    rw [← actionEq] at rowInAction
    exact actionFrame_supported application frame productBase
      (verifierRows + publicRingColumns + matrix.val)
      (ConcreteNifsPiRlcActionRows.evaluationValueLow
        application profile frame matrix)
      (ConcreteNifsPiRlcActionRows.evaluationOutputLow
        application profile frame matrix)
      (ConcreteNifsPiRlcActionRows.challenge application profile frame)
      (challenge_supported application profile frame)
      (evaluationValueLow_supported application profile frame matrix)
      (evaluationOutputLow_supported application profile frame matrix)
      row rowInAction column columnMember
  · rcases List.mem_ofFn.1 inHigh with ⟨matrix, actionEq⟩
    rw [← actionEq] at rowInAction
    exact actionFrame_supported application frame productBase
      (verifierRows + publicRingColumns + shape.matrixCount + matrix.val)
      (ConcreteNifsPiRlcActionRows.evaluationValueHigh
        application profile frame matrix)
      (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
        application profile frame matrix)
      (ConcreteNifsPiRlcActionRows.challenge application profile frame)
      (challenge_supported application profile frame)
      (evaluationValueHigh_supported application profile frame matrix)
      (evaluationOutputHigh_supported application profile frame matrix)
      row rowInAction column columnMember

/-- Exact action conservation: every non-product dependency is authoritative
and visible before the occurrence, while every remaining dependency belongs
to the action program's explicit product allocation. -/
theorem rows_conservation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat)
    (row : Row)
    (rowMember :
      row ∈ ConcreteNifsPiRlcActionRows.rows
        application profile frame productBase)
    (column : ColumnId)
    (columnMember : column ∈ row.columnIds) :
    column ∈ frame.visibleIds
      ∨ column ∈ ConcreteNifsPiRlcActionRows.columns
        application profile frame productBase := by
  rcases List.mem_flatMap.1 rowMember with
    ⟨action, actionMember, rowInAction⟩
  unfold ConcreteNifsPiRlcActionRows.frames at actionMember
  rcases List.mem_append.1 actionMember with firstThree | inHigh
  rcases List.mem_append.1 firstThree with firstTwo | inLow
  rcases List.mem_append.1 firstTwo with inCommitment | inPublic
  · rcases List.mem_ofFn.1 inCommitment with ⟨target, actionEq⟩
    rw [← actionEq] at rowInAction
    rcases actionFrame_conservation application frame productBase target.val
        (ConcreteNifsPiRlcActionRows.commitmentValue
          application profile frame target)
        (ConcreteNifsPiRlcActionRows.commitmentOutput
          application profile frame target)
        (ConcreteNifsPiRlcActionRows.challenge application profile frame)
        (challenge_visible application profile frame)
        (commitmentValue_visible application profile frame target)
        (commitmentOutput_visible application profile frame target)
        row rowInAction column columnMember with visible | product
    · exact Or.inl visible
    · apply Or.inr
      unfold ConcreteNifsPiRlcActionRows.columns
      apply List.mem_flatMap.2
      refine
        ⟨ConcreteNifsPiRlcActionRows.commitmentFrame
            application profile frame productBase target, ?_, product⟩
      · unfold ConcreteNifsPiRlcActionRows.frames
        exact List.mem_append_left _
          (List.mem_append_left _
            (List.mem_append_left _
              (List.mem_ofFn.2 ⟨target, rfl⟩)))
  · rcases List.mem_ofFn.1 inPublic with ⟨block, actionEq⟩
    rw [← actionEq] at rowInAction
    rcases actionFrame_conservation application frame productBase
        (verifierRows + block.val)
        (ConcreteNifsPiRlcActionRows.publicValue
          application profile frame block)
        (ConcreteNifsPiRlcActionRows.publicOutput
          application profile frame block)
        (ConcreteNifsPiRlcActionRows.challenge application profile frame)
        (challenge_visible application profile frame)
        (publicValue_visible application profile frame block)
        (publicOutput_visible application profile frame block)
        row rowInAction column columnMember with visible | product
    · exact Or.inl visible
    · apply Or.inr
      unfold ConcreteNifsPiRlcActionRows.columns
      apply List.mem_flatMap.2
      refine
        ⟨ConcreteNifsPiRlcActionRows.publicFrame
            application profile frame productBase block, ?_, product⟩
      · unfold ConcreteNifsPiRlcActionRows.frames
        exact List.mem_append_left _
          (List.mem_append_left _
            (List.mem_append_right _
              (List.mem_ofFn.2 ⟨block, rfl⟩)))
  · rcases List.mem_ofFn.1 inLow with ⟨matrix, actionEq⟩
    rw [← actionEq] at rowInAction
    rcases actionFrame_conservation application frame productBase
        (verifierRows + publicRingColumns + matrix.val)
        (ConcreteNifsPiRlcActionRows.evaluationValueLow
          application profile frame matrix)
        (ConcreteNifsPiRlcActionRows.evaluationOutputLow
          application profile frame matrix)
        (ConcreteNifsPiRlcActionRows.challenge application profile frame)
        (challenge_visible application profile frame)
        (evaluationValueLow_visible application profile frame matrix)
        (evaluationOutputLow_visible application profile frame matrix)
        row rowInAction column columnMember with visible | product
    · exact Or.inl visible
    · apply Or.inr
      unfold ConcreteNifsPiRlcActionRows.columns
      apply List.mem_flatMap.2
      refine
        ⟨ConcreteNifsPiRlcActionRows.evaluationLowFrame
            application profile frame productBase matrix, ?_, product⟩
      · unfold ConcreteNifsPiRlcActionRows.frames
        exact List.mem_append_left _
          (List.mem_append_right _
            (List.mem_ofFn.2 ⟨matrix, rfl⟩))
  · rcases List.mem_ofFn.1 inHigh with ⟨matrix, actionEq⟩
    rw [← actionEq] at rowInAction
    rcases actionFrame_conservation application frame productBase
        (verifierRows + publicRingColumns + shape.matrixCount + matrix.val)
        (ConcreteNifsPiRlcActionRows.evaluationValueHigh
          application profile frame matrix)
        (ConcreteNifsPiRlcActionRows.evaluationOutputHigh
          application profile frame matrix)
        (ConcreteNifsPiRlcActionRows.challenge application profile frame)
        (challenge_visible application profile frame)
        (evaluationValueHigh_visible application profile frame matrix)
        (evaluationOutputHigh_visible application profile frame matrix)
        row rowInAction column columnMember with visible | product
    · exact Or.inl visible
    · apply Or.inr
      unfold ConcreteNifsPiRlcActionRows.columns
      apply List.mem_flatMap.2
      refine
        ⟨ConcreteNifsPiRlcActionRows.evaluationHighFrame
            application profile frame productBase matrix, ?_, product⟩
      · unfold ConcreteNifsPiRlcActionRows.frames
        exact List.mem_append_right _
          (List.mem_ofFn.2 ⟨matrix, rfl⟩)

/-- Every product coordinate counted by the selected action program is
mentioned by the corresponding emitted product row. -/
theorem columns_written
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat)
    (column : ColumnId)
    (member :
      column ∈ ConcreteNifsPiRlcActionRows.columns
        application profile frame productBase) :
    ∃ row ∈ ConcreteNifsPiRlcActionRows.rows
        application profile frame productBase,
      column ∈ row.columnIds := by
  unfold ConcreteNifsPiRlcActionRows.columns at member
  rcases List.mem_flatMap.1 member with
    ⟨action, actionMember, columnMember⟩
  rcases Phi81RingAction.productIds_written
      action column columnMember with
    ⟨row, rowMember, written⟩
  exact ⟨row,
    List.mem_flatMap.2 ⟨action, actionMember, rowMember⟩,
    written⟩

private theorem frame_for_target
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase target : Nat)
    (targetLt :
      target <
        ConcreteNifsPiRlcActionRows.targetCount
          shape publicRingColumns verifierRows) :
    ∃ action ∈ ConcreteNifsPiRlcActionRows.frames
        application profile frame productBase,
      ∀ source left right,
        action.productColumn source left right =
          columnMap frame
            (ConcreteNifsPiRlcActionRows.productSource
              productBase target source left right) := by
  by_cases inCommitment : target < verifierRows
  · let row : Fin verifierRows := ⟨target, inCommitment⟩
    refine
      ⟨ConcreteNifsPiRlcActionRows.commitmentFrame
          application profile frame productBase row,
        ?_, ?_⟩
    · unfold ConcreteNifsPiRlcActionRows.frames
      apply List.mem_append_left
      apply List.mem_append_left
      apply List.mem_append_left
      exact List.mem_ofFn.2 ⟨row, rfl⟩
    · intro source left right
      rfl
  · by_cases inPublic : target < verifierRows + publicRingColumns
    · let block : Fin publicRingColumns :=
        ⟨target - verifierRows, by omega⟩
      refine
        ⟨ConcreteNifsPiRlcActionRows.publicFrame
            application profile frame productBase block,
          ?_, ?_⟩
      · unfold ConcreteNifsPiRlcActionRows.frames
        apply List.mem_append_left
        apply List.mem_append_left
        apply List.mem_append_right
        exact List.mem_ofFn.2 ⟨block, rfl⟩
      · intro source left right
        simp only [ConcreteNifsPiRlcActionRows.publicFrame,
          ConcreteNifsPiRlcActionRows.actionFrame]
        congr 2
        dsimp [block]
        omega
    · by_cases inLow :
        target <
          verifierRows + publicRingColumns + shape.matrixCount
      · let matrix : Fin shape.matrixCount :=
          ⟨target - (verifierRows + publicRingColumns), by omega⟩
        refine
          ⟨ConcreteNifsPiRlcActionRows.evaluationLowFrame
              application profile frame productBase matrix,
            ?_, ?_⟩
        · unfold ConcreteNifsPiRlcActionRows.frames
          apply List.mem_append_left
          apply List.mem_append_right
          exact List.mem_ofFn.2 ⟨matrix, rfl⟩
        · intro source left right
          simp only [ConcreteNifsPiRlcActionRows.evaluationLowFrame,
            ConcreteNifsPiRlcActionRows.actionFrame]
          congr 2
          dsimp [matrix]
          omega
      · let matrix : Fin shape.matrixCount :=
          ⟨target -
              (verifierRows + publicRingColumns + shape.matrixCount), by
            unfold ConcreteNifsPiRlcActionRows.targetCount at targetLt
            omega⟩
        refine
          ⟨ConcreteNifsPiRlcActionRows.evaluationHighFrame
              application profile frame productBase matrix,
            ?_, ?_⟩
        · unfold ConcreteNifsPiRlcActionRows.frames
          apply List.mem_append_right
          exact List.mem_ofFn.2 ⟨matrix, rfl⟩
        · intro source left right
          simp only [ConcreteNifsPiRlcActionRows.evaluationHighFrame,
            ConcreteNifsPiRlcActionRows.actionFrame]
          congr 2
          dsimp [matrix]
          omega

/-- Every coordinate of the action program's declared dense product interval
is one of its explicitly enumerated product columns. -/
theorem dense_column_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase offset : Nat)
    (offsetLt :
      offset <
        (ConcreteNifsPiRlcActionRows.cost
          shape publicRingColumns verifierRows).auxiliaryColumns) :
    columnMap frame (productBase + offset) ∈
      ConcreteNifsPiRlcActionRows.columns
        application profile frame productBase := by
  let width :=
    Phi81RingAction.productWidth FixedActive.arity.total
  have widthPositive : 0 < width := by
    simp [width, Phi81RingAction.productWidth, ringDegree]
  let target := offset / width
  let within := offset % width
  have targetLt :
      target <
        ConcreteNifsPiRlcActionRows.targetCount
          shape publicRingColumns verifierRows := by
    apply (Nat.div_lt_iff_lt_mul widthPositive).2
    simpa [target, width, ConcreteNifsPiRlcActionRows.cost] using offsetLt
  have withinLt : within < width :=
    Nat.mod_lt offset widthPositive
  let coefficientWidth := ringDegree * ringDegree
  have coefficientWidthPositive : 0 < coefficientWidth := by
    simp [coefficientWidth, ringDegree]
  let source : Fin FixedActive.arity.total :=
    ⟨within / coefficientWidth, by
      apply (Nat.div_lt_iff_lt_mul coefficientWidthPositive).2
      simpa [width, coefficientWidth, Phi81RingAction.productWidth,
        FixedActive.arity_total, ringDegree, Nat.mul_assoc] using withinLt⟩
  let coefficient := within % coefficientWidth
  have coefficientLt : coefficient < coefficientWidth :=
    Nat.mod_lt within coefficientWidthPositive
  let left : Fin ringDegree :=
    ⟨coefficient / ringDegree, by
      apply (Nat.div_lt_iff_lt_mul (by simp [ringDegree])).2
      simpa [coefficientWidth, Nat.mul_comm] using coefficientLt⟩
  let right : Fin ringDegree :=
    ⟨coefficient % ringDegree,
      Nat.mod_lt coefficient (by simp [ringDegree])⟩
  have withinSplit :
      within =
        source.val * coefficientWidth + coefficient := by
    have split := Nat.div_add_mod within coefficientWidth
    rw [Nat.mul_comm coefficientWidth (within / coefficientWidth)] at split
    simpa [source, coefficient] using split.symm
  have coefficientSplit :
      coefficient =
        left.val * ringDegree + right.val := by
    have split := Nat.div_add_mod coefficient ringDegree
    rw [Nat.mul_comm ringDegree (coefficient / ringDegree)] at split
    simpa [left, right] using split.symm
  have offsetSplit :
      offset = target * width + within := by
    have split := Nat.div_add_mod offset width
    rw [Nat.mul_comm width (offset / width)] at split
    simpa [target, within] using split.symm
  have productOffsetEq :
      Phi81RingAction.productOffset source.val left.val right.val =
        within := by
    calc
      Phi81RingAction.productOffset source.val left.val right.val =
          source.val * (ringDegree * ringDegree) +
            (left.val * ringDegree + right.val) := by
        unfold Phi81RingAction.productOffset
        rw [Nat.add_mul, Nat.mul_assoc, Nat.add_assoc]
      _ = source.val * coefficientWidth + coefficient := by
        rw [coefficientSplit]
      _ = within := withinSplit.symm
  rcases frame_for_target application profile frame productBase target
      targetLt with
    ⟨action, actionMember, productColumnEq⟩
  unfold ConcreteNifsPiRlcActionRows.columns
  apply List.mem_flatMap.2
  refine ⟨action, actionMember, ?_⟩
  unfold Phi81RingAction.productIds
  apply List.mem_flatMap.2
  refine ⟨source.val, List.mem_range.2 source.isLt, ?_⟩
  apply List.mem_flatMap.2
  refine ⟨left.val, List.mem_range.2 left.isLt, ?_⟩
  apply List.mem_map.2
  refine ⟨right.val, List.mem_range.2 right.isLt, ?_⟩
  rw [productColumnEq]
  apply congrArg (columnMap frame)
  unfold ConcreteNifsPiRlcActionRows.productSource
  rw [productOffsetEq]
  change
    offset =
      target * Phi81RingAction.productWidth FixedActive.arity.total + within
    at offsetSplit
  rw [Nat.add_assoc, ← offsetSplit]

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionAudit
