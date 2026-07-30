import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
import Nightstream.Implementation.R1CS.Canonical.KEquality

/-!
Contract: Lean-owned materialization rows for the selected fixed-active NIFS
output children.

For every one of the fourteen children, these rows bind the output-running
codec to exactly the three prover payload fields retained by
`PiDecChildPayload` and to the parent point inherited by
`PiDecChildPayload.materialize`:

* commitment = proof payload commitment;
* public input = proof payload public input;
* point = output parent point;
* evaluations = proof payload evaluations.

The rows read only authoritative proof/output codec columns and the constant
wire. They allocate no column. Parent construction, activation, other NIFS
phases, Rust, and generated artifacts are outside this module.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1600000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputRows

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.Implementation.R1CS.Canonical.KMul
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

def outputChildCommitment
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (row : Fin verifierRows) (lane : Fin ringDegree) : LinComb :=
  (ConcreteNifsCarrierFrame.outputFLocation
    (FamilyFor application) frame
    (profile.runningViews.childCommitment child row lane)).carried

def proofChildCommitment
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (row : Fin verifierRows) (lane : Fin ringDegree) : LinComb :=
  (ConcreteNifsCarrierFrame.proofFLocation
    (FamilyFor application) frame
    (profile.payloadViews.commitment child row lane)).carried

def outputChildPublic
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (column : Fin (ringDegree * publicRingColumns)) : LinComb :=
  (ConcreteNifsCarrierFrame.outputFLocation
    (FamilyFor application) frame
    (profile.runningViews.childPublic child column)).carried

def proofChildPublic
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (column : Fin (ringDegree * publicRingColumns)) : LinComb :=
  (ConcreteNifsCarrierFrame.proofFLocation
    (FamilyFor application) frame
    (profile.payloadViews.publicInput child column)).carried

def outputChildPoint
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin shape.rowVariables) : Carried :=
  (ConcreteNifsCarrierFrame.outputKLocation
    (FamilyFor application) frame
    (profile.runningViews.childPoint child coordinate)).carried

def outputParentPoint
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (coordinate : Fin shape.rowVariables) : Carried :=
  (ConcreteNifsCarrierFrame.outputKLocation
    (FamilyFor application) frame
    (profile.runningViews.parentPoint coordinate)).carried

def outputChildEvaluation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) : Carried :=
  (ConcreteNifsCarrierFrame.outputKLocation
    (FamilyFor application) frame
    (profile.runningViews.childEvaluation child matrix lane)).carried

def proofChildEvaluation
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) : Carried :=
  (ConcreteNifsCarrierFrame.proofKLocation
    (FamilyFor application) frame
    (profile.payloadViews.evaluation child matrix lane)).carried

def commitmentRows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    List Nightstream.Implementation.R1CS.Row :=
  (List.ofFn fun row : Fin verifierRows => row).flatMap fun row =>
    (List.ofFn fun lane : Fin ringDegree => lane).map fun lane =>
      KEquality.equalityRow
        (outputChildCommitment application profile frame child row lane)
        (proofChildCommitment application profile frame child row lane)

def publicRows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    List Nightstream.Implementation.R1CS.Row :=
  (List.ofFn fun column : Fin (ringDegree * publicRingColumns) => column).map
    fun column =>
      KEquality.equalityRow
        (outputChildPublic application profile frame child column)
        (proofChildPublic application profile frame child column)

def pointRows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    List Nightstream.Implementation.R1CS.Row :=
  (List.ofFn fun coordinate : Fin shape.rowVariables => coordinate).flatMap
    fun coordinate =>
      KEquality.rows
        (outputChildPoint application profile frame child coordinate)
        (outputParentPoint application profile frame coordinate)

def evaluationRows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    List Nightstream.Implementation.R1CS.Row :=
  (List.ofFn fun matrix : Fin shape.matrixCount => matrix).flatMap fun matrix =>
    (List.ofFn fun lane : Fin ringDegree => lane).flatMap fun lane =>
      KEquality.rows
        (outputChildEvaluation application profile frame child matrix lane)
        (proofChildEvaluation application profile frame child matrix lane)

def childRows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    List Nightstream.Implementation.R1CS.Row :=
  commitmentRows application profile frame child ++
    publicRows application profile frame child ++
    pointRows application profile frame child ++
    evaluationRows application profile frame child

/-- Complete output-child materialization program, in child-major order. -/
def rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    List Nightstream.Implementation.R1CS.Row :=
  (List.ofFn fun child : Fin productionGlobalParams.k => child).flatMap
    (childRows application profile frame)

private theorem child_member
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (row : Nightstream.Implementation.R1CS.Row)
    (member : row ∈ childRows application profile frame child) :
    row ∈ rows application profile frame :=
  List.mem_flatMap.2
    ⟨child, List.mem_ofFn.2 ⟨child, rfl⟩, member⟩

theorem commitmentRow_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (row : Fin verifierRows) (lane : Fin ringDegree) :
    KEquality.equalityRow
        (outputChildCommitment application profile frame child row lane)
        (proofChildCommitment application profile frame child row lane) ∈
      rows application profile frame := by
  apply child_member application profile frame child
  simp only [childRows, List.mem_append]
  apply Or.inl
  apply Or.inl
  apply Or.inl
  apply List.mem_flatMap.2
  refine ⟨row, List.mem_ofFn.2 ⟨row, rfl⟩, ?_⟩
  exact List.mem_map.2 ⟨lane, List.mem_ofFn.2 ⟨lane, rfl⟩, rfl⟩

theorem publicRow_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (column : Fin (ringDegree * publicRingColumns)) :
    KEquality.equalityRow
        (outputChildPublic application profile frame child column)
        (proofChildPublic application profile frame child column) ∈
      rows application profile frame := by
  apply child_member application profile frame child
  simp only [childRows, List.mem_append]
  apply Or.inl
  apply Or.inl
  apply Or.inr
  exact List.mem_map.2
    ⟨column, List.mem_ofFn.2 ⟨column, rfl⟩, rfl⟩

theorem pointRow_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin shape.rowVariables)
    (row : Nightstream.Implementation.R1CS.Row)
    (member :
      row ∈ KEquality.rows
        (outputChildPoint application profile frame child coordinate)
        (outputParentPoint application profile frame coordinate)) :
    row ∈ rows application profile frame := by
  apply child_member application profile frame child
  simp only [childRows, List.mem_append]
  apply Or.inl
  apply Or.inr
  exact List.mem_flatMap.2
    ⟨coordinate, List.mem_ofFn.2 ⟨coordinate, rfl⟩, member⟩

theorem evaluationRow_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k)
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree)
    (row : Nightstream.Implementation.R1CS.Row)
    (member :
      row ∈ KEquality.rows
        (outputChildEvaluation application profile frame child matrix lane)
        (proofChildEvaluation application profile frame child matrix lane)) :
    row ∈ rows application profile frame := by
  apply child_member application profile frame child
  simp only [childRows, List.mem_append]
  apply Or.inr
  exact List.mem_flatMap.2
    ⟨matrix, List.mem_ofFn.2 ⟨matrix, rfl⟩,
      List.mem_flatMap.2
        ⟨lane, List.mem_ofFn.2 ⟨lane, rfl⟩, member⟩⟩

private theorem flatMap_const_length
    {α β : Type} (values : List α) (program : α → List β) (count : Nat)
    (each : ∀ value, (program value).length = count) :
    (values.flatMap program).length = count * values.length := by
  induction values with
  | nil => simp
  | cons head tail inductionHypothesis =>
      simp only [List.flatMap_cons, List.length_append, List.length_cons]
      rw [each head, inductionHypothesis, Nat.mul_succ]
      exact Nat.add_comm _ _

theorem commitmentRows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    (commitmentRows application profile frame child).length =
      ringDegree * verifierRows := by
  unfold commitmentRows
  rw [flatMap_const_length]
  · rw [List.length_ofFn]
  · intro row
    rw [List.length_map, List.length_ofFn]

theorem publicRows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    (publicRows application profile frame child).length =
      ringDegree * publicRingColumns := by
  simp [publicRows]

theorem pointRows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    (pointRows application profile frame child).length =
      2 * shape.rowVariables := by
  unfold pointRows
  rw [flatMap_const_length]
  · rw [List.length_ofFn]
  · intro coordinate
    exact KEquality.rows_length _ _

theorem evaluationRows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    (evaluationRows application profile frame child).length =
      (2 * ringDegree) * shape.matrixCount := by
  unfold evaluationRows
  rw [flatMap_const_length]
  · rw [List.length_ofFn]
  · intro matrix
    rw [flatMap_const_length]
    · rw [List.length_ofFn]
    · intro lane
      exact KEquality.rows_length _ _

def childRowCount
    (shape : SemanticShape) (publicRingColumns verifierRows : Nat) : Nat :=
  ringDegree * verifierRows +
    ringDegree * publicRingColumns +
    2 * shape.rowVariables +
    (2 * ringDegree) * shape.matrixCount

theorem childRows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (child : Fin productionGlobalParams.k) :
    (childRows application profile frame child).length =
      childRowCount shape publicRingColumns verifierRows := by
  simp only [childRows, List.length_append, childRowCount]
  rw [commitmentRows_length, publicRows_length, pointRows_length,
    evaluationRows_length]

/-- Exact receipt-derived row count for all fourteen output children. -/
theorem rows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (rows application profile frame).length =
      childRowCount shape publicRingColumns verifierRows *
        productionGlobalParams.k := by
  unfold rows
  rw [flatMap_const_length]
  · rw [List.length_ofFn]
  · intro child
    exact childRows_length application profile frame child

/-- Output binding reads existing columns and allocates none. -/
def columns
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) : List Nat :=
  []

theorem columns_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (columns application profile frame).length = 0 := rfl

def cost
    (shape : SemanticShape) (publicRingColumns verifierRows : Nat) :
    Typed.Cost where
  recurringRows :=
    childRowCount shape publicRingColumns verifierRows *
      productionGlobalParams.k
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem cost_rows_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (cost shape publicRingColumns verifierRows).recurringRows =
      (rows application profile frame).length := by
  simpa [cost] using (rows_length application profile frame).symm

theorem cost_columns_eq
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (cost shape publicRingColumns verifierRows).auxiliaryColumns =
      (columns application profile frame).length := rfl

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOutputRows
