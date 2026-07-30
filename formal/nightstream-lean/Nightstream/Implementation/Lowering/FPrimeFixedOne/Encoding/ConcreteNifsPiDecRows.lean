import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RadixRows

/-!
Contract: Lean-owned physical rows for the selected outgoing `Pi_DEC`.

The verifier retains exactly three public recomposition families:
commitment, public input, and evaluations.  Every child coordinate is read
from the authoritative proof payload codec and every parent coordinate is
read from the authoritative output-running codec.  The weights are the
verifier-owned production radix weights from `Phi81RadixRows`.

This slice allocates no columns.  It does not own the derived parent theorem,
activation, the other NIFS phases, or Rust.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecRows

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
    (SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows)
    Digest AppState Witness
    (SelectedRunning shape publicRingColumns publicFits verifierRows)
    (SelectedFresh shape publicRingColumns publicFits verifierRows)
    Encoded 1}
variable {terminalRelations :
  TerminalRelations
    (SelectedKey shape TranscriptState publicRingColumns publicFits verifierRows)
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
    (ConcreteNifsCarrierFrame.proofFLocation
      (FamilyFor application) frame
      (profile.payloadViews.commitment child row lane)).carried
  parent :=
    (ConcreteNifsCarrierFrame.outputFLocation
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
    (ConcreteNifsCarrierFrame.proofFLocation
      (FamilyFor application) frame
      (profile.payloadViews.publicInput child column)).carried
  parent :=
    (ConcreteNifsCarrierFrame.outputFLocation
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
    (ConcreteNifsCarrierFrame.proofKLocation
      (FamilyFor application) frame
      (profile.payloadViews.evaluation child matrix lane)).carried
  parent :=
    (ConcreteNifsCarrierFrame.outputKLocation
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

/-- The complete selected outgoing `Pi_DEC` row slice. -/
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
  Phi81RadixRows.rows
    (fCoordinates application profile frame)
    (evaluationCoordinates application profile frame)

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
  rw [List.length_flatMap, List.map_ofFn]
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
  rw [List.length_flatMap, List.map_ofFn]
  simp only [List.length_ofFn]
  exact sum_ofFn_const shape.matrixCount ringDegree

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
      verifierRows * ringDegree +
        ringDegree * publicRingColumns +
        2 * (shape.matrixCount * ringDegree) := by
  unfold rows fCoordinates publicCoordinates
  rw [Phi81RadixRows.rows_length, List.length_append, List.length_ofFn,
    commitmentCoordinates_length, evaluationCoordinates_length]

/-- The selected outgoing `Pi_DEC` slice only reads call-frame columns. -/
def columns : List Nat := []

theorem columns_length : columns.length = 0 := rfl

theorem columns_nodup : columns.Nodup := List.nodup_nil

def cost
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat) : Cost where
  recurringRows :=
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

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiDecRows
