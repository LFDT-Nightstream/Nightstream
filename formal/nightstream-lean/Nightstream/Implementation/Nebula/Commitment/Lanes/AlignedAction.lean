import Nightstream.Protocol.Nebula.LaneLayout
import Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism.PiRLCFinite
import Nightstream.SuperNeo.Concrete.Phi81Relation.PiDECAlgebra.Radix

/-!
Contract: exact whole-ring lane projections for the concrete Phi81 folding
actions used by the V2 product commitment.

Assurance tier: implementation-to-algebra bridge.

Owns one aligned slice between complete Phi81 carriers; its exact block
embedding; and proofs that the slice commutes with assignment addition,
PiRLC's `RingF` action and finite combination, and PiDEC split and
recomposition.

Does not own commitment keys, Ajtai arithmetic, bundle combination, NIFS
transcripts, R1CS rows, Rust, or cryptographic binding.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 100000

namespace Nightstream.Implementation.Nebula.AlignedLaneAction

open Nightstream.Protocol.Nebula
open Nightstream.SuperNeo.Concrete
open Nightstream.SuperNeo.Concrete.Phi81Relation
open Nightstream.SuperNeo.Concrete.Phi81Relation.EvaluationHomomorphism
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint

/-- One complete inner carrier at an aligned interval of a complete outer
carrier. -/
structure Slice (outer inner : Phi81Relation.Shape) where
  start : Nat
  startAligned : LaneLayout.Aligned start
  within : start + inner.carrierWidth ≤ outer.carrierWidth

namespace Slice

variable {outer inner : Phi81Relation.Shape}

/-- The start expressed as a complete Phi81 block offset. -/
def blockOffset (slice : Slice outer inner) : Nat :=
  slice.start / ringDegree

theorem start_eq_blockOffset_mul (slice : Slice outer inner) :
    slice.start = slice.blockOffset * ringDegree := by
  have divides : ringDegree ∣ slice.start :=
    Nat.dvd_of_mod_eq_zero slice.startAligned
  exact (Nat.div_mul_cancel divides).symm

private theorem carrierWidth_eq_blocks (shape : Phi81Relation.Shape) :
    shape.carrierWidth =
      Phi81ColumnLayout.blockCount shape.carrierWidth * ringDegree := by
  unfold Phi81Relation.Shape.carrierWidth
  rw [Phi81CarrierLayout.blockCount_carrierWidth]
  rfl

/-- Embed one inner carrier block at the exact aligned outer block offset. -/
def embedBlock (slice : Slice outer inner)
    (block : Fin (Phi81ColumnLayout.blockCount inner.carrierWidth)) :
    Fin (Phi81ColumnLayout.blockCount outer.carrierWidth) :=
  ⟨slice.blockOffset + block.val, by
    have startExact := slice.start_eq_blockOffset_mul
    have innerExact := carrierWidth_eq_blocks inner
    have outerExact := carrierWidth_eq_blocks outer
    have blockLt := block.isLt
    have within := slice.within
    simp only [ringDegree] at startExact innerExact outerExact
    omega⟩

/-- Read the exact inner interval from one outer assignment. -/
def project (slice : Slice outer inner)
    (assignment : Assignment outer) : Assignment inner :=
  fun column =>
    assignment ⟨slice.start + column.val, by
      have bound := column.isLt
      have within := slice.within
      omega⟩

@[simp] theorem project_apply (slice : Slice outer inner)
    (assignment : Assignment outer) (column : Fin inner.carrierWidth) :
    slice.project assignment column =
      assignment ⟨slice.start + column.val, by
        have bound := column.isLt
        have within := slice.within
        omega⟩ :=
  rfl

/-- The local block/lane coordinate is the outer block-offset coordinate. -/
theorem project_carrierColumn (slice : Slice outer inner)
    (assignment : Assignment outer)
    (block : Fin (Phi81ColumnLayout.blockCount inner.carrierWidth))
    (lane : Fin ringDegree) :
    slice.project assignment
        (Phi81CarrierLayout.carrierColumn block lane) =
      assignment
        (Phi81CarrierLayout.carrierColumn (slice.embedBlock block) lane) := by
  unfold project Phi81CarrierLayout.carrierColumn
    Phi81ColumnLayout.flatIndex embedBlock
  apply congrArg assignment
  apply Fin.ext
  have startExact := slice.start_eq_blockOffset_mul
  simp only [ringDegree] at startExact ⊢
  omega

/-- Every inner block read is exactly the corresponding embedded outer block
read. -/
theorem assignmentBlock_project (slice : Slice outer inner)
    (assignment : Assignment outer)
    (block : Fin (Phi81ColumnLayout.blockCount inner.carrierWidth)) :
    CarrierAction.assignmentBlock (slice.project assignment) block =
      CarrierAction.assignmentBlock assignment (slice.embedBlock block) := by
  funext lane
  exact slice.project_carrierColumn assignment block lane

private theorem carrierColumn_decode (column : Fin inner.carrierWidth) :
    Phi81CarrierLayout.carrierColumn
        (Phi81ColumnLayout.decode column).1
        (Phi81ColumnLayout.decode column).2 = column := by
  apply Fin.ext
  exact Phi81ColumnLayout.flatIndex_decode column

@[simp] theorem project_zero (slice : Slice outer inner) :
    slice.project (BaseLinear.assignmentZero : Assignment outer) =
      (BaseLinear.assignmentZero : Assignment inner) :=
  rfl

@[simp] theorem project_add (slice : Slice outer inner)
    (left right : Assignment outer) :
    slice.project (BaseLinear.assignmentAdd left right) =
      BaseLinear.assignmentAdd (slice.project left) (slice.project right) :=
  rfl

/-- Whole-ring alignment is the necessary fact: an arbitrary field-coordinate
slice does not commute with the Phi81 ring action. -/
theorem project_act (slice : Slice outer inner) (challenge : RingF)
    (assignment : Assignment outer) :
    slice.project (CarrierAction.act challenge assignment) =
      CarrierAction.act challenge (slice.project assignment) := by
  funext column
  let decoded := Phi81ColumnLayout.decode column
  have columnExact :
      Phi81CarrierLayout.carrierColumn decoded.1 decoded.2 = column := by
    exact carrierColumn_decode column
  rw [← columnExact]
  rw [slice.project_carrierColumn]
  unfold CarrierAction.act
  rw [CarrierAction.decode_carrierColumn,
    CarrierAction.decode_carrierColumn]
  change
    ringFMul challenge
        (CarrierAction.assignmentBlock assignment
          (slice.embedBlock decoded.1)) decoded.2 =
      ringFMul challenge
        (CarrierAction.assignmentBlock (slice.project assignment)
          decoded.1) decoded.2
  rw [slice.assignmentBlock_project]

/-- The exact head-first PiRLC assignment fold projects with the same source
order and challenge vector. -/
theorem project_combineAssignments (slice : Slice outer inner)
    {count : Nat} (challenges : Fin count → RingF)
    (assignments : Fin count → Assignment outer) :
    slice.project (PiRLCFinite.combineAssignments challenges assignments) =
      PiRLCFinite.combineAssignments challenges
        (fun index => slice.project (assignments index)) := by
  induction count with
  | zero => rfl
  | succ count inductionHypothesis =>
      rw [PiRLCFinite.combineAssignments,
        PiRLCFinite.combineAssignments, slice.project_add,
        slice.project_act]
      rw [inductionHypothesis
        (fun index => challenges index.succ)
        (fun index => assignments index.succ)]

/-- PiDEC digit splitting is coordinatewise and therefore preserves an exact
slice. -/
theorem project_splitAssignment (slice : Slice outer inner)
    (assignment : Assignment outer)
    (child : PiDECAlgebra.Radix.ChildIndex) :
    slice.project (PiDECAlgebra.Radix.splitAssignment assignment child) =
      PiDECAlgebra.Radix.splitAssignment (slice.project assignment) child :=
  rfl

/-- PiDEC recomposition is coordinatewise and preserves the same child order
and radix powers across an exact slice. -/
theorem project_recomposeAssignment (slice : Slice outer inner)
    (assignments : PiDECAlgebra.Radix.ChildIndex → Assignment outer) :
    slice.project (PiDECAlgebra.Radix.recomposeAssignment assignments) =
      PiDECAlgebra.Radix.recomposeAssignment
        (fun child => slice.project (assignments child)) := by
  funext column
  simp only [project, PiDECAlgebra.Radix.recomposeAssignment_apply]

end Slice

/-! ## Exact lane-layout adapters -/

variable {fullShape operationsShape snapshotShape : Phi81Relation.Shape}

def operationsSlice
    (layout : LaneLayout.Layout fullShape.carrierWidth
      operationsShape.carrierWidth snapshotShape.carrierWidth) :
    Slice fullShape operationsShape where
  start := layout.operationsStart
  startAligned := layout.operationsStartAligned
  within := layout.operationsWithin

def initialSnapshotSlice
    (layout : LaneLayout.Layout fullShape.carrierWidth
      operationsShape.carrierWidth snapshotShape.carrierWidth) :
    Slice fullShape snapshotShape where
  start := layout.initialSnapshotStart
  startAligned := layout.initialSnapshotStartAligned
  within := layout.initialSnapshotWithin

def finalSnapshotSlice
    (layout : LaneLayout.Layout fullShape.carrierWidth
      operationsShape.carrierWidth snapshotShape.carrierWidth) :
    Slice fullShape snapshotShape where
  start := layout.finalSnapshotStart
  startAligned := layout.finalSnapshotStartAligned
  within := layout.finalSnapshotWithin

end Nightstream.Implementation.Nebula.AlignedLaneAction
