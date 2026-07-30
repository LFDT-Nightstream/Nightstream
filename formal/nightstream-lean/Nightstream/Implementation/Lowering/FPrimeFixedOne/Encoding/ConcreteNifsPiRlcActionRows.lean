import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierFrame
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.Phi81RingAction

/-!
Contract: Lean-owned physical Phi81 actions for the selected fixed-active
`Pi_RLC` parent.

Every challenge is read directly from the proof codec.  Commitment and public
input sources are the canonical one-fresh-plus-fourteen-running call operands;
evaluation sources are the `Pi_CCS` output ring stored in the proof.  Every
action writes directly to the parent coordinates of the `nifsVerify` output
bundle.

The caller supplies only the first numeric source reserved for product cells.
All product sources are then derived by one injective arithmetic layout and
translated through the sole global call-column map.

This module owns rows, allocation order, and exact cost.  It does not own
activation, the surrounding `Pi_CCS`/sampler rows, outgoing radix
recomposition, output-child materialization, or semantic acceptance.
-/

set_option autoImplicit false
set_option maxRecDepth 100000
set_option maxHeartbeats 1200000

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionRows

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.Goldilocks.NumericRowBridge
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsCarrierFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
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

private abbrev FamilyFor (application : Poseidon23ApplicationProfile Selected) :=
  application.family Selected

private abbrev Profile
    (application : Poseidon23ApplicationProfile Selected) :=
  ConcreteNifsOperationalProfile.Profile application

private abbrev FrameFor
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)} :=
  CallFrame (signature := signature Selected)
    (FamilyFor application) Call.nifsVerify
    (Refs.cons runningRef
      (Refs.cons freshRef (Refs.cons proofRef .nil)))

/-- Translate one located numeric base-field expression through the sole
global call-column map. -/
def carriedF
    (application : Poseidon23ApplicationProfile Selected)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (carried : List (Nat × Nat)) : LinearCombination :=
  terms (columnMap frame) carried

/-- Exact public-input coordinate of one Phi81 block and lane. -/
def publicCoordinate
    (block : Fin publicRingColumns)
    (lane : Fin ringDegree) :
    Fin (ringDegree * publicRingColumns) :=
  ⟨block.val * ringDegree + lane.val, by
    have beforeNext :
        block.val * ringDegree + lane.val <
          (block.val + 1) * ringDegree := by
      rw [Nat.add_mul, Nat.one_mul]
      exact Nat.add_lt_add_left lane.isLt _
    have nextWithin :
        (block.val + 1) * ringDegree <=
          publicRingColumns * ringDegree :=
      Nat.mul_le_mul_right ringDegree (Nat.succ_le_of_lt block.isLt)
    exact Nat.lt_of_lt_of_le beforeNext
      (by simpa [Nat.mul_comm] using nextWithin)⟩

/-- Complete carried challenge ring, read from the proof codec. -/
def challenge
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (source : Fin FixedActive.arity.total) :
    CarriedRing :=
  fun lane =>
    carriedF application frame
      (proofFLocation (FamilyFor application) frame
        (profile.samplerViews.challenge source lane)).carried

/-- One fresh or running commitment ring in canonical source order. -/
def commitmentValue
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Fin verifierRows)
    (source : Fin FixedActive.arity.total) :
    CarriedRing :=
  Fin.addCases
    (fun _ lane =>
      carriedF application frame
        (freshFLocation (FamilyFor application) frame
          (profile.freshViews.commitment row lane)).carried)
    (fun child lane =>
      carriedF application frame
        (runningFLocation (FamilyFor application) frame
          (profile.runningViews.childCommitment child row lane)).carried)
    source

/-- One fresh or running public-input ring in canonical source order. -/
def publicValue
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (block : Fin publicRingColumns)
    (source : Fin FixedActive.arity.total) :
    CarriedRing :=
  Fin.addCases
    (fun _ lane =>
      carriedF application frame
        (freshFLocation (FamilyFor application) frame
          (profile.freshViews.publicInput
            (publicCoordinate block lane))).carried)
    (fun child lane =>
      carriedF application frame
        (runningFLocation (FamilyFor application) frame
          (profile.runningViews.childPublic child
            (publicCoordinate block lane))).carried)
    source

/-- Low extension coordinate of one `Pi_CCS` output evaluation ring. -/
def evaluationValueLow
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount)
    (source : Fin FixedActive.arity.total) :
    CarriedRing :=
  fun lane =>
    carriedF application frame
      (proofKLocation (FamilyFor application) frame
        (profile.endpointViews.outputYRing
          ((keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.alignment.semanticIndex
              source)
          matrix lane)).carried.low

/-- High extension coordinate of one `Pi_CCS` output evaluation ring. -/
def evaluationValueHigh
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount)
    (source : Fin FixedActive.arity.total) :
    CarriedRing :=
  fun lane =>
    carriedF application frame
      (proofKLocation (FamilyFor application) frame
        (profile.endpointViews.outputYRing
          ((keys
            Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne.selected).template.alignment.semanticIndex
              source)
          matrix lane)).carried.high

/-- Output-parent commitment ring. -/
def commitmentOutput
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (row : Fin verifierRows) :
    CarriedRing :=
  fun lane =>
    carriedF application frame
      (outputFLocation (FamilyFor application) frame
        (profile.runningViews.parentCommitment row lane)).carried

/-- Output-parent public-input ring. -/
def publicOutput
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (block : Fin publicRingColumns) :
    CarriedRing :=
  fun lane =>
    carriedF application frame
      (outputFLocation (FamilyFor application) frame
        (profile.runningViews.parentPublic
          (publicCoordinate block lane))).carried

/-- Low output-parent evaluation ring. -/
def evaluationOutputLow
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount) :
    CarriedRing :=
  fun lane =>
    carriedF application frame
      (outputKLocation (FamilyFor application) frame
        (profile.runningViews.parentEvaluation matrix lane)).carried.low

/-- High output-parent evaluation ring. -/
def evaluationOutputHigh
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (matrix : Fin shape.matrixCount) :
    CarriedRing :=
  fun lane =>
    carriedF application frame
      (outputKLocation (FamilyFor application) frame
        (profile.runningViews.parentEvaluation matrix lane)).carried.high

/-- Number of independent RingF actions needed by the full public parent. -/
def targetCount
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat) : Nat :=
  verifierRows + publicRingColumns + 2 * shape.matrixCount

/-- Numeric product-cell source for one action target. -/
def productSource
    (productBase target source left right : Nat) : Nat :=
  productBase +
    target * Phi81RingAction.productWidth FixedActive.arity.total +
      Phi81RingAction.productOffset source left right

/-- One concrete action frame. -/
def actionFrame
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase target : Nat)
    (values : Fin FixedActive.arity.total → CarriedRing)
    (output : CarriedRing) :
    Phi81RingAction.Frame FixedActive.arity.total where
  owner := frame.owner
  firstOrdinal :=
    target * Phi81RingAction.rowCount FixedActive.arity.total
  one := frame.one
  challenges := challenge application profile frame
  values := values
  output := output
  productColumn := fun source left right =>
    columnMap frame (productSource productBase target source left right)

/-- Commitment actions occupy the first target interval. -/
def commitmentFrame
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat)
    (row : Fin verifierRows) :
    Phi81RingAction.Frame FixedActive.arity.total :=
  actionFrame application profile frame productBase row.val
    (commitmentValue application profile frame row)
    (commitmentOutput application profile frame row)

/-- Public-input actions follow commitment actions. -/
def publicFrame
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat)
    (block : Fin publicRingColumns) :
    Phi81RingAction.Frame FixedActive.arity.total :=
  actionFrame application profile frame productBase
    (verifierRows + block.val)
    (publicValue application profile frame block)
    (publicOutput application profile frame block)

/-- Low evaluation actions follow the complete public-input interval. -/
def evaluationLowFrame
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat)
    (matrix : Fin shape.matrixCount) :
    Phi81RingAction.Frame FixedActive.arity.total :=
  actionFrame application profile frame productBase
    (verifierRows + publicRingColumns + matrix.val)
    (evaluationValueLow application profile frame matrix)
    (evaluationOutputLow application profile frame matrix)

/-- High evaluation actions are the final target interval. -/
def evaluationHighFrame
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat)
    (matrix : Fin shape.matrixCount) :
    Phi81RingAction.Frame FixedActive.arity.total :=
  actionFrame application profile frame productBase
    (verifierRows + publicRingColumns + shape.matrixCount + matrix.val)
    (evaluationValueHigh application profile frame matrix)
    (evaluationOutputHigh application profile frame matrix)

/-- Every independent public-parent action, in carrier order. -/
def frames
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat) :
    List (Phi81RingAction.Frame FixedActive.arity.total) :=
  List.ofFn (commitmentFrame application profile frame productBase) ++
    List.ofFn (publicFrame application profile frame productBase) ++
      List.ofFn (evaluationLowFrame application profile frame productBase) ++
        List.ofFn (evaluationHighFrame application profile frame productBase)

@[simp] theorem frames_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat) :
    (frames application profile frame productBase).length =
      targetCount shape publicRingColumns verifierRows := by
  simp only [frames, List.length_append, List.length_ofFn, targetCount]
  omega

private theorem flatMap_length_eq_mul
    {α β : Type}
    (items : List α)
    (emit : α → List β)
    (width : Nat)
    (exact : ∀ item, (emit item).length = width) :
    (items.flatMap emit).length = items.length * width := by
  induction items with
  | nil =>
      simp
  | cons head tail ih =>
      simp only [List.flatMap_cons, List.length_append, exact, ih,
        List.length_cons]
      rw [Nat.succ_mul]
      omega

/-- Complete ungated Phi81-action row list. -/
def rows
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat) :
    List Row :=
  (frames application profile frame productBase).flatMap
    Phi81RingAction.rawRows

/-- Exact product-cell allocation in the same target order. -/
def columns
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat) :
    List ColumnId :=
  (frames application profile frame productBase).flatMap
    Phi81RingAction.productIds

/-- Program-derived cost of all public-parent actions. -/
def cost
    (shape : SemanticShape)
    (publicRingColumns verifierRows : Nat) : Cost where
  recurringRows :=
    targetCount shape publicRingColumns verifierRows *
      Phi81RingAction.rowCount FixedActive.arity.total
  committedColumns := 0
  publicColumns := 0
  auxiliaryColumns :=
    targetCount shape publicRingColumns verifierRows *
      Phi81RingAction.productWidth FixedActive.arity.total

theorem rows_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat) :
    (rows application profile frame productBase).length =
      (cost shape publicRingColumns verifierRows).recurringRows := by
  rw [rows, flatMap_length_eq_mul
    (frames application profile frame productBase)
    Phi81RingAction.rawRows
    (Phi81RingAction.rowCount FixedActive.arity.total)
    (fun item => Phi81RingAction.rawRows_length item)]
  rw [frames_length]
  rfl

theorem columns_length
    (application : Poseidon23ApplicationProfile Selected)
    (profile : Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef :
      Ref (typeSystem Selected) context (.data .running)}
    {freshRef :
      Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef :
      Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (productBase : Nat) :
    (columns application profile frame productBase).length =
      (cost shape publicRingColumns verifierRows).auxiliaryColumns := by
  rw [columns, flatMap_length_eq_mul
    (frames application profile frame productBase)
    Phi81RingAction.productIds
    (Phi81RingAction.productWidth FixedActive.arity.total)
    (fun item => Phi81RingAction.productIds_length item)]
  rw [frames_length]
  rfl

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsPiRlcActionRows
