import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsMetadataAccumulator
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplaySchema

/-!
Contract: compact schema for the Rust claim-coordinate overlay relation.

Assurance tier: artifact schema.

Owns one shared six-range state layout, one no-op arm, and one active-arm
descriptor for each claim chunk. Each active descriptor reconstructs the
exact coordinate calls, normalized update or carry rows, chunk-zero pins,
and source-field links.

Does not own generated data, Rust conformance, phase selection, accumulator
completion, lifecycle semantics, or security reduction.

Emits constraints: no. It describes existing constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay.Artifact

open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataAccumulator
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsMetadataCoordinateMaps
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

/-- First columns of the six 108-coordinate state ranges. -/
structure StateBases where
  beforeStatementFresh : Nat
  afterStatementFresh : Nat
  beforeRunningCommitments : Nat
  afterRunningCommitments : Nat
  beforeRunningPublic : Nat
  afterRunningPublic : Nat
deriving DecidableEq, Repr, Inhabited

def StateBases.beforeBase (layout : StateBases) : MapKind → Nat
  | .statementFresh => layout.beforeStatementFresh
  | .runningCommitments => layout.beforeRunningCommitments
  | .runningPublic => layout.beforeRunningPublic

def StateBases.afterBase (layout : StateBases) : MapKind → Nat
  | .statementFresh => layout.afterStatementFresh
  | .runningCommitments => layout.afterRunningCommitments
  | .runningPublic => layout.afterRunningPublic

def StateBases.beforeColumn
    (layout : StateBases) (kind : MapKind) (output : Fin outputWidth) : Nat :=
  layout.beforeBase kind + output.val

def StateBases.afterColumn
    (layout : StateBases) (kind : MapKind) (output : Fin outputWidth) : Nat :=
  layout.afterBase kind + output.val

/-- One active overlay kind. `phaseState` and `phaseChunkBase` reconstruct
the exact links into the selected claim-replay base arm. -/
structure RawActiveArm where
  overlayKind : Nat
  phaseKind : Nat
  chunkIndex : Nat
  rowCount : Nat
  columnCount : Nat
  phaseState : StateBases
  phaseChunkBase : Nat
  coordinateCalls : List CoordinateCall
  linkCallIndices : List Nat
deriving DecidableEq, Repr

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  noopRowCount : Nat
  noopColumnCount : Nat
  overlayState : StateBases
  activeArms : List RawActiveArm
deriving DecidableEq, Repr

def mapOrder : List MapKind :=
  [.statementFresh, .runningCommitments, .runningPublic]

def RawActiveArm.chunk (arm : RawActiveArm) : Fin claimChunkCount :=
  ⟨arm.chunkIndex % claimChunkCount, Nat.mod_lt _ (by decide)⟩

def RawActiveArm.callFor
    (arm : RawActiveArm) (kind : MapKind) : Option CoordinateCall :=
  arm.coordinateCalls.find? fun call => call.mapKind = kind

/-- Exact normalized Rust row for `after = before + partial`. -/
def updateRow (layout : StateBases) (call : CoordinateCall)
    (output : Fin outputWidth) : Row :=
  ⟨[(layout.beforeColumn call.mapKind output, goldilocksP - 1),
      (layout.afterColumn call.mapKind output, 1),
      (call.outputColumn output, goldilocksP - 1)],
    [(0, 1)], []⟩

def updateRows (layout : StateBases) (call : CoordinateCall) : List Row :=
  List.ofFn (updateRow layout call)

/-- Exact normalized Rust row for an inactive-map carry. -/
def carryRow (layout : StateBases) (kind : MapKind)
    (output : Fin outputWidth) : Row :=
  ⟨[(layout.beforeColumn kind output, goldilocksP - 1),
      (layout.afterColumn kind output, 1)],
    [(0, 1)], []⟩

def carryRows (layout : StateBases) (kind : MapKind) : List Row :=
  List.ofFn (carryRow layout kind)

def initialRowsFor (layout : StateBases) (kind : MapKind) : List Row :=
  List.ofFn fun output : Fin outputWidth =>
    ⟨[(layout.beforeColumn kind output, 1)], [(0, 1)], []⟩

def initialRows (layout : StateBases) : List Row :=
  mapOrder.flatMap (initialRowsFor layout)

def mapRows (layout : StateBases) (arm : RawActiveArm)
    (kind : MapKind) : List Row :=
  match arm.callFor kind with
  | some call => call.rows ++ updateRows layout call
  | none => carryRows layout kind

/-- Complete explicit-plus-compact source row program for one active arm. -/
def RawActiveArm.rows
    (layout : StateBases) (arm : RawActiveArm) : List Row :=
  mapOrder.flatMap (mapRows layout arm) ++
    if arm.chunkIndex = 0 then initialRows layout else []

def RawActiveArm.Satisfied
    (layout : StateBases) (arm : RawActiveArm)
    (assignment : Nat → Nat) : Prop :=
  Satisfies (arm.rows layout) assignment

/-- Exact no-op source program: `1 * 1 = 1`. -/
def noopRows : List Row :=
  [⟨[(0, 1)], [(0, 1)], [(0, 1)]⟩]

structure FieldLink where
  phaseField : Nat
  overlayField : Nat
deriving DecidableEq, Repr, Inhabited

def stateLinksAt
    (phase overlay : StateBases) (output : Fin outputWidth) : List FieldLink :=
  [ ⟨phase.beforeColumn .statementFresh output,
      overlay.beforeColumn .statementFresh output⟩
  , ⟨phase.afterColumn .statementFresh output,
      overlay.afterColumn .statementFresh output⟩
  , ⟨phase.beforeColumn .runningCommitments output,
      overlay.beforeColumn .runningCommitments output⟩
  , ⟨phase.afterColumn .runningCommitments output,
      overlay.afterColumn .runningCommitments output⟩
  , ⟨phase.beforeColumn .runningPublic output,
      overlay.beforeColumn .runningPublic output⟩
  , ⟨phase.afterColumn .runningPublic output,
      overlay.afterColumn .runningPublic output⟩ ]

def RawActiveArm.stateLinks
    (overlay : StateBases) (arm : RawActiveArm) : List FieldLink :=
  (List.finRange outputWidth).flatMap fun output =>
    stateLinksAt arm.phaseState overlay output

def RawActiveArm.callChunkLinks
    (arm : RawActiveArm) (call : CoordinateCall) : List FieldLink :=
  call.activeFields.map fun field =>
    ⟨arm.phaseChunkBase + (call.mapKind.claimChunkOffset field).val,
      call.fieldColumn field⟩

def RawActiveArm.chunkLinks (arm : RawActiveArm) : List FieldLink :=
  arm.linkCallIndices.flatMap fun index =>
    arm.callChunkLinks (arm.coordinateCalls.getD index default)

/-- Complete exact link order used by the joint selective composer. -/
def RawActiveArm.links
    (overlay : StateBases) (arm : RawActiveArm) : List FieldLink :=
  arm.stateLinks overlay ++ arm.chunkLinks

/-- Explicit trust-boundary premise for all active map fields in one selected
overlay arm. The base replay and exact physical links must discharge it from
the same verifier-owned frame. -/
def RawActiveArm.FrameLinked
    (arm : RawActiveArm) (frame : ClaimFrame)
    (assignment : Nat → Nat) : Prop :=
  ∀ call ∈ arm.coordinateCalls,
    ∀ field ∈ call.activeFields,
      assignment (call.fieldColumn field) =
        (frame ⟨call.mapKind.framePosition field,
          call.mapKind.framePosition_lt field⟩).val

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateOverlay.Artifact
