import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalOccurrence
import Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile

/-!
Contract: physical canonicality rows for the selected NIFS proof's initial
Poseidon2 duplex state.

The proof codec omits no lane coordinate, but its admissible domain requires
the initial duplex state to be the verifier-owned empty state. These eight
rows make that semantic restriction physical: each selected prior-lane
coordinate is multiplied by the constant wire and forced to zero.

Owns: construction, exact row and column cost, soundness, honest
completeness, and support for this eight-row slice.

Does not own: the remaining proof decoder, transcript replay, verifier
acceptance, activation, Rust, or generated artifacts.

Emits constraints: eight rows and no columns.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsProofCanonicalityRows

open Nightstream.HyperNova.Construction2.Paper
open Nightstream.Implementation.Lowering.Typed
open Nightstream.Implementation.Lowering.Goldilocks
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Vocabulary
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.DirectCalls
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsCallFrame
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.PaperNifsGlobalColumnMap
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsParameters
open Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsOperationalProfile
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

/-- Exact physical coordinate selected for one initial duplex lane. -/
def priorLaneColumn
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (lane : Fin 8) : ColumnId :=
  ((profile.priorLane lane).column
    (proofOperand frame.operands) (proof_widthsAgree frame)).column

/-- One row `priorLane * 1 = 0`. -/
def laneRow
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef))
    (lane : Fin 8) : Row where
  a := singleton (priorLaneColumn application profile frame lane) 1
  b := singleton frame.one 1
  c := []

/-- The complete proof-canonicality slice, in lane order. -/
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
    List Row :=
  List.ofFn (laneRow application profile frame)

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
    (rows application profile frame).length = 8 := by
  simp [rows]

private theorem rawSatisfies_member
    (source : List Row)
    (assignment : ColumnId → Field)
    (satisfied : RawSatisfies source assignment) :
    ∀ row, row ∈ source → row.Holds assignment := by
  induction source with
  | nil =>
      intro row member
      simp at member
  | cons head tail inductionHypothesis =>
      intro row member
      rcases List.mem_cons.1 member with rfl | inTail
      · exact satisfied.1
      · exact inductionHypothesis satisfied.2 row inTail

private theorem rawSatisfies_ofFn
    {count : Nat}
    (row : Fin count → Row)
    (assignment : ColumnId → Field)
    (holds : ∀ index, (row index).Holds assignment) :
    RawSatisfies (List.ofFn row) assignment := by
  induction count with
  | zero =>
      simp
  | succ count inductionHypothesis =>
      rw [List.ofFn_succ]
      exact
        ⟨holds 0,
          inductionHypothesis (fun index => row index.succ)
            (fun index => holds index.succ)⟩

private theorem singleton_eval
    (assignment : ColumnId → Field)
    (column : ColumnId) :
    (Goldilocks.singleton column 1).eval assignment =
      assignment column := by
  simp only [Goldilocks.singleton, LinearCombination.eval, Fin.one_mul,
    Fin.add_zero]

/-- Physical satisfaction forces every selected prior lane to zero. -/
theorem rows_sound
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
    (constantWire : assignment frame.one = 1)
    (satisfied : RawSatisfies (rows application profile frame) assignment)
    (lane : Fin 8) :
    assignment (priorLaneColumn application profile frame lane) = 0 := by
  have member :
      laneRow application profile frame lane ∈
        rows application profile frame := by
    exact List.mem_ofFn.2 ⟨lane, rfl⟩
  have holds :=
    rawSatisfies_member
      (rows application profile frame) assignment satisfied
      (laneRow application profile frame lane) member
  change
    (Goldilocks.singleton
        (priorLaneColumn application profile frame lane) 1).eval assignment *
      (Goldilocks.singleton frame.one 1).eval assignment =
        LinearCombination.eval assignment [] at holds
  rw [singleton_eval, singleton_eval, constantWire,
    LinearCombination.eval_nil, Fin.mul_one] at holds
  exact holds

/-- An admissible decoded proof satisfies all eight canonicality rows. -/
theorem rows_honest
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
    (constantWire : assignment frame.one = 1)
    (decoded :
      frame.operands.Decodes (FamilyFor application) assignment
        (.cons running (.cons fresh (.cons proof .nil)))) :
    RawSatisfies (rows application profile frame) assignment := by
  have proofDecoded :=
    ConcreteNifsSelectedCallFrame.proof_decodes_of_frame_decodes
      (FamilyFor application) frame assignment running fresh proof decoded
  have proofAdmissible :
      ((FamilyFor application).codecFor (.data .nifsProof)).Admissible proof :=
    Codec.admissible_of_decode
      ((FamilyFor application).codecFor (.data .nifsProof)) proofDecoded
  have priorState :=
    profile.proofAdmissiblePriorState proof proofAdmissible
  unfold rows
  apply rawSatisfies_ofFn
  intro lane
  have laneValue :=
    (profile.priorLane lane).value_eq_of_bundle_decodes
      (FamilyFor application) (.data .nifsProof)
      (proofOperand frame.operands) (proof_widthsAgree frame)
      assignment proof proofDecoded
  have selectedZero :
      assignment (priorLaneColumn application profile frame lane) = 0 := by
    change
      assignment (priorLaneColumn application profile frame lane) =
        NumericRowBridge.residue (proof.priorState.lanes lane) at laneValue
    simpa only [priorState, Poseidon2Duplex.empty,
      NumericRowBridge.residue] using laneValue
  change
    (Goldilocks.singleton
        (priorLaneColumn application profile frame lane) 1).eval assignment *
      (Goldilocks.singleton frame.one 1).eval assignment =
        LinearCombination.eval assignment []
  rw [singleton_eval, singleton_eval, selectedZero, constantWire,
    LinearCombination.eval_nil, Fin.zero_mul]

/-- Every row reads only the proof bundle and the constant wire. -/
theorem rows_supported
    (application : Poseidon23ApplicationProfile Selected)
    (profile : ConcreteNifsOperationalProfile.Profile application)
    {context : Schema (typeSystem Selected)}
    {runningRef : Ref (typeSystem Selected) context (.data .running)}
    {freshRef : Ref (typeSystem Selected) context (.data .fresh)}
    {proofRef : Ref (typeSystem Selected) context (.data .nifsProof)}
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    RawRowsSupportedBy frame.visibleIds
      (rows application profile frame) := by
  intro row rowMember column columnMember
  rcases List.mem_ofFn.1 rowMember with ⟨lane, rfl⟩
  simp only [laneRow, Row.columnIds, Goldilocks.singleton,
    List.append_nil, List.map_append, List.map_cons, List.map_nil,
    List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at columnMember
  rcases columnMember with rfl | rfl
  · apply visibleIds_supported frame
    exact proofOperand_mem_visible frame
      ((profile.priorLane lane).column_mem
        (proofOperand frame.operands) (proof_widthsAgree frame))
  · simp [CallFrame.visibleIds]

/-- This slice allocates no columns. -/
def columns : List ColumnId := []

theorem columns_length : columns.length = 0 := rfl

/-- Receipt-derived intrinsic cost. -/
def cost : Cost where
  recurringRows := 8
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
    (frame : FrameFor application
      (context := context) (runningRef := runningRef)
      (freshRef := freshRef) (proofRef := proofRef)) :
    (rows application profile frame).length = cost.recurringRows :=
  rows_length application profile frame

theorem cost_columns :
    columns.length = cost.auxiliaryColumns :=
  rfl

end SelectedFrame

end Nightstream.Implementation.Lowering.FPrimeFixedOne.Encoding.ConcreteNifsProofCanonicalityRows
