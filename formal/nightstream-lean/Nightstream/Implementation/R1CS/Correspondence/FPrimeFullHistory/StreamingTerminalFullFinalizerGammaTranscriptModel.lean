import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
import Nightstream.Implementation.R1CS.Core.SevenBytePacking
import Nightstream.Implementation.R1CS.Correspondence.PiRlcChallenge.Transcript.ColumnReplay

/-!
Contract: independent operation schedule for the terminal Nebula gamma
transcript.

Owns the ten labelled field appends, their authoritative source columns, and
the two labelled two-field challenges. It does not own generated pins,
Poseidon2 calls, row satisfaction, output muxes, or lifecycle closure.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptModel

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.SevenBytePacking
open Nightstream.Implementation.R1CS.PiRlcChallenge.Transcript
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer

def asciiBytes (text : String) : List Nat :=
  text.toList.map Char.toNat

def pinnedWords (values : List Nat) : List ColumnReplay.Operation :=
  values.map ColumnReplay.Operation.pinned

def externalFields (columns : List Nat) : List ColumnReplay.Operation :=
  columns.map ColumnReplay.Operation.external

/-- Exact TranscriptGadget append-fields framing. -/
def appendFields
    (label : String) (columns : List Nat) : List ColumnReplay.Operation :=
  [.pinned 2] ++
    pinnedWords (packBytesAsNats (asciiBytes label)) ++
    [.pinned columns.length] ++
    externalFields columns

/-- Exact TranscriptGadget append-message framing. -/
def appendMessage
    (label message : String) : List ColumnReplay.Operation :=
  [.pinned 1] ++
    pinnedWords (packBytesAsNats (asciiBytes label)) ++
    pinnedWords (packBytesAsNats (asciiBytes message))

/-- Exact two-field challenge, including squeeze and query binding. -/
def challengeFields (label : String) : List ColumnReplay.Operation :=
  appendMessage "chal/label" label ++
    [.digest, .pinned 257, .pinned 2]

def decodedDPreColumns : List Nat :=
  (List.range dPreWordCount).map rawArtifact.dPreWordColumn

def vkFsAppend : List ColumnReplay.Operation :=
  appendFields "nebula/vk_fs" rawArtifact.vkFsColumns

def boundaryAppend : List ColumnReplay.Operation :=
  appendFields "nebula/z_i" rawArtifact.boundaryColumns

def accumulatorAppend : List ColumnReplay.Operation :=
  appendFields "nebula/acc_digest" rawArtifact.accumulatorColumns

def stagedLaneAppend : List ColumnReplay.Operation :=
  appendFields "nebula/lane" rawArtifact.stagedDigestOutputColumns

def programBindingAppend : List ColumnReplay.Operation :=
  appendFields "nebula/program_binding"
    (rawArtifact.laneColumns.take digestFields)

def segmentIndexAppend : List ColumnReplay.Operation :=
  appendFields "nebula/seg_idx"
    [rawArtifact.laneSegmentIndexColumn]

def timestampAppend : List ColumnReplay.Operation :=
  appendFields "nebula/ts" [rawArtifact.laneColumn 7]

def dPreOpsAppend : List ColumnReplay.Operation :=
  appendFields "nebula/d_pre_ops" (decodedDPreColumns.take digestFields)

def dPreIsAppend : List ColumnReplay.Operation :=
  appendFields "nebula/d_pre_is"
    ((decodedDPreColumns.drop digestFields).take digestFields)

def dPreFsAppend : List ColumnReplay.Operation :=
  appendFields "nebula/d_pre_fs"
    ((decodedDPreColumns.drop (2 * digestFields)).take digestFields)

def gamma1Challenge : List ColumnReplay.Operation :=
  challengeFields "nebula/gamma1"

def gamma2Challenge : List ColumnReplay.Operation :=
  challengeFields "nebula/gamma2"

/-- Twelve semantic phases in exact production order. -/
def operationSegments : List (List ColumnReplay.Operation) :=
  [vkFsAppend, boundaryAppend, accumulatorAppend, stagedLaneAppend,
    programBindingAppend, segmentIndexAppend, timestampAppend,
    dPreOpsAppend, dPreIsAppend, dPreFsAppend,
    gamma1Challenge, gamma2Challenge]

def operations : List ColumnReplay.Operation :=
  operationSegments.flatten

def externalColumn? : ColumnReplay.Operation → Option Nat
  | .external column => some column
  | _ => none

/-- Complete transcript-input authority boundary, derived from the
handwritten operation schedule in exact absorb order. -/
def externalColumns : List Nat :=
  operations.filterMap externalColumn?

theorem operation_segment_count : operationSegments.length = 12 := by
  rfl

theorem decoded_dPre_column_count : decodedDPreColumns.length = 12 := by
  rfl

theorem operation_count : operations.length = 110 := by
  rfl

theorem external_column_count : externalColumns.length = 34 := by
  rfl

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerGammaTranscriptModel
