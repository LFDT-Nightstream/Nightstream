import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrenceSemantics
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsSelectedCallFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RadixRows
import Nightstream.Implementation.R1CS.Canonical.KConsistency
import Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81.FixedActive.Canonical.RunningAuthority

/-!
Contract: Lean-owned physical rows for the complete incoming running-authority
relation of the selected fixed-active NIFS.

Every operand is a direct codec projection in the sole call-frame column
namespace.  The program checks every child point coordinate and every
commitment, public-input, and evaluation coordinate.  The radix coefficients
are the verifier-owned production `2^i`, `i < 14`, weights.

The checks allocate no columns.  This module owns the exact row construction,
soundness to `RunningAuthority.Equations`, honest completeness, and the
receipt-derived cost.  It does not own activation, outgoing PiDEC, outputs,
the operational PiCCS proof, or Rust.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityRows

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
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierViews
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.KHorner
open Nightstream.Implementation.R1CS.Canonical.KMul
open Nightstream.Implementation.R1CS.Canonical.LinCombNormal
open Nightstream.SuperNeo
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.Folding.PiCCS.SplitNc
open Nightstream.SuperNeo.Folding.Nifs.ConcretePhi81

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

/-! ## Direct coordinate programs -/

def commitmentCoordinate
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (row : Fin verifierRows) (lane : Fin ringDegree) :
    Phi81RadixRows.FCoordinate where
  children := fun child =>
    (ConcreteNifsCarrierFrame.runningFLocation
      (FamilyFor application) frame
      (profile.runningViews.childCommitment child row lane)).carried
  parent :=
    (ConcreteNifsCarrierFrame.runningFLocation
      (FamilyFor application) frame
      (profile.runningViews.parentCommitment row lane)).carried

def commitmentCoordinates
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List Phi81RadixRows.FCoordinate :=
  (List.ofFn fun row : Fin verifierRows => row).flatMap fun row =>
    List.ofFn fun lane : Fin ringDegree =>
      commitmentCoordinate application profile frame row lane

def publicCoordinate
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (column : Fin (ringDegree * publicRingColumns)) :
    Phi81RadixRows.FCoordinate where
  children := fun child =>
    (ConcreteNifsCarrierFrame.runningFLocation
      (FamilyFor application) frame
      (profile.runningViews.childPublic child column)).carried
  parent :=
    (ConcreteNifsCarrierFrame.runningFLocation
      (FamilyFor application) frame
      (profile.runningViews.parentPublic column)).carried

def publicCoordinates
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List Phi81RadixRows.FCoordinate :=
  List.ofFn fun column : Fin (ringDegree * publicRingColumns) =>
    publicCoordinate application profile frame column

def evaluationCoordinate
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    Phi81RadixRows.KCoordinate where
  children := fun child =>
    (ConcreteNifsCarrierFrame.runningKLocation
      (FamilyFor application) frame
      (profile.runningViews.childEvaluation child matrix lane)).carried
  parent :=
    (ConcreteNifsCarrierFrame.runningKLocation
      (FamilyFor application) frame
      (profile.runningViews.parentEvaluation matrix lane)).carried

def evaluationCoordinates
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List Phi81RadixRows.KCoordinate :=
  (List.ofFn fun matrix : Fin shape.matrixCount => matrix).flatMap fun matrix =>
    List.ofFn fun lane : Fin ringDegree =>
      evaluationCoordinate application profile frame matrix lane

def pointPair
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin shape.rowVariables) : Carried × Carried :=
  ((ConcreteNifsCarrierFrame.runningKLocation
      (FamilyFor application) frame
      (profile.runningViews.childPoint child coordinate)).carried,
    (ConcreteNifsCarrierFrame.runningKLocation
      (FamilyFor application) frame
      (profile.runningViews.parentPoint coordinate)).carried)

def pointPairs
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List (Carried × Carried) :=
  (List.ofFn fun child : Fin productionGlobalParams.k => child).flatMap
    fun child =>
      List.ofFn fun coordinate : Fin shape.rowVariables =>
        pointPair application profile frame child coordinate

def fCoordinates
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List Phi81RadixRows.FCoordinate :=
  commitmentCoordinates application profile frame ++
    publicCoordinates application profile frame

def rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    List Nightstream.Implementation.R1CS.Row :=
  KConsistency.consistencyRows (pointPairs application profile frame) ++
    Phi81RadixRows.rows
      (fCoordinates application profile frame)
      (evaluationCoordinates application profile frame)

/-! ## Exact receipt -/

private theorem sum_ofFn_const (count value : Nat) :
    (List.ofFn fun _ : Fin count => value).sum = count * value := by
  induction count with
  | zero => simp
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ, List.sum_cons, inductionHypothesis, Nat.succ_mul]
      omega

theorem commitmentCoordinates_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (commitmentCoordinates application profile frame).length =
      verifierRows * ringDegree := by
  unfold commitmentCoordinates
  rw [List.length_flatMap]
  rw [List.map_ofFn]
  simp only [List.length_ofFn]
  exact sum_ofFn_const verifierRows ringDegree

theorem evaluationCoordinates_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (evaluationCoordinates application profile frame).length =
      shape.matrixCount * ringDegree := by
  unfold evaluationCoordinates
  rw [List.length_flatMap]
  rw [List.map_ofFn]
  simp only [List.length_ofFn]
  exact sum_ofFn_const shape.matrixCount ringDegree

theorem pointPairs_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (pointPairs application profile frame).length =
      productionGlobalParams.k * shape.rowVariables := by
  unfold pointPairs
  rw [List.length_flatMap]
  rw [List.map_ofFn]
  simp only [List.length_ofFn]
  exact sum_ofFn_const productionGlobalParams.k shape.rowVariables

theorem rows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (rows application profile frame).length =
      2 * productionGlobalParams.k * shape.rowVariables +
        verifierRows * ringDegree +
        ringDegree * publicRingColumns +
        2 * (shape.matrixCount * ringDegree) := by
  unfold rows fCoordinates publicCoordinates
  rw [List.length_append,
    KConsistency.consistencyRows_length_eq,
    Phi81RadixRows.rows_length,
    List.length_append, List.length_ofFn,
    pointPairs_length, commitmentCoordinates_length,
    evaluationCoordinates_length]
  rw [Nat.mul_assoc 2 productionGlobalParams.k shape.rowVariables]
  omega

/-! ## Cost and allocation receipt -/

/-- The running-authority slice only reads authoritative call-frame columns. -/
def columns : List Nat := []

theorem columns_length : columns.length = 0 := rfl

theorem columns_nodup : columns.Nodup := List.nodup_nil

/-- Exact intrinsic cost of the selected running-authority relation. -/
def cost
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat) : Cost where
  recurringRows :=
    2 * productionGlobalParams.k * shape.rowVariables +
      verifierRows * ringDegree +
      ringDegree * publicRingColumns +
      2 * (shape.matrixCount * ringDegree)
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns := 0

theorem cost_rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil)))) :
    (rows application profile frame).length =
      (cost shape publicRingColumns verifierRows).recurringRows :=
  rows_length application profile frame

theorem cost_columns :
    columns.length =
      (cost shape publicRingColumns verifierRows).auxiliaryColumns :=
  columns_length

/-! ## Membership receipts used by soundness -/

theorem commitmentCoordinate_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (row : Fin verifierRows) (lane : Fin ringDegree) :
    commitmentCoordinate application profile frame row lane ∈
      fCoordinates application profile frame := by
  apply List.mem_append_left
  exact List.mem_flatMap.2
    ⟨row, List.mem_ofFn.2 ⟨row, rfl⟩,
      List.mem_ofFn.2 ⟨lane, rfl⟩⟩

theorem publicCoordinate_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (column : Fin (ringDegree * publicRingColumns)) :
    publicCoordinate application profile frame column ∈
      fCoordinates application profile frame := by
  apply List.mem_append_right
  exact List.mem_ofFn.2 ⟨column, rfl⟩

theorem evaluationCoordinate_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (matrix : Fin shape.matrixCount) (lane : Fin ringDegree) :
    evaluationCoordinate application profile frame matrix lane ∈
      evaluationCoordinates application profile frame :=
  List.mem_flatMap.2
    ⟨matrix, List.mem_ofFn.2 ⟨matrix, rfl⟩,
      List.mem_ofFn.2 ⟨lane, rfl⟩⟩

theorem pointPair_mem
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame :
      CallFrame (signature := signature Selected)
        (FamilyFor application) Call.nifsVerify
        (Refs.cons runningRef
          (Refs.cons freshRef (Refs.cons proofRef .nil))))
    (child : Fin productionGlobalParams.k)
    (coordinate : Fin shape.rowVariables) :
    pointPair application profile frame child coordinate ∈
      pointPairs application profile frame :=
  List.mem_flatMap.2
    ⟨child, List.mem_ofFn.2 ⟨child, rfl⟩,
      List.mem_ofFn.2 ⟨coordinate, rfl⟩⟩

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsRunningAuthorityRows
