/-!
Wire schema for the compact Rust streaming F-prime program artifact.

Owns proof-free phase runs and executable geometry checks.

Does not own phase meanings, phase-local constraints, lifecycle semantics,
relation dimensions, Rust conformance, or security reduction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact

structure RawRun where
  phaseCode : Nat
  firstIndex : Nat
  count : Nat
  deriving DecidableEq, Inhabited, Repr

/-- Compact source-field link contract for one non-no-op claim-coordinate
overlay. The active fields occupy one contiguous chunk range. -/
structure RawOverlayLinkRun where
  overlayKind : Nat
  phaseKind : Nat
  chunkIndex : Nat
  activeOffsetStart : Nat
  activeFieldCount : Nat
  deriving DecidableEq, Inhabited, Repr

/-- Compact rectangular link run shared by all PiRLC family overlays. Each
outer item links `fieldCount` consecutive body and overlay fields. -/
structure RawFieldLinkRun where
  phaseFieldStart : Nat
  overlayFieldStart : Nat
  outerCount : Nat
  phaseStride : Nat
  overlayStride : Nat
  fieldCount : Nat
  deriving DecidableEq, Inhabited, Repr

def RawFieldLinkRun.linkCount (run : RawFieldLinkRun) : Nat :=
  run.outerCount * run.fieldCount

def RawFieldLinkRun.valid
    (phaseColumns overlayColumns : Nat) (run : RawFieldLinkRun) : Bool :=
  0 < run.outerCount && 0 < run.fieldCount &&
    run.phaseFieldStart + (run.outerCount - 1) * run.phaseStride +
      run.fieldCount ≤ phaseColumns &&
    run.overlayFieldStart + (run.outerCount - 1) * run.overlayStride +
      run.fieldCount ≤ overlayColumns

def RawRun.expand (run : RawRun) : List (Nat × Nat) :=
  (List.range run.count).map fun offset =>
    (run.phaseCode, run.firstIndex + offset)

structure RawProgram where
  schemaVersion : Nat
  stateChunkFields : Nat
  priorStateFrameFields : Nat
  priorStateChunks : Nat
  claimFrameFields : Nat
  claimChunkFields : Nat
  claimChunks : Nat
  piCcsRounds : Nat
  piRlcFamilies : Nat
  firstPiRlcFamilyProgramCursor : Nat
  successorPrefixFrameFields : Nat
  successorPrefixChunks : Nat
  workItemCount : Nat
  lifecycleGroupCount : Nat
  circuitKindCount : Nat
  claimCoordinateOverlayKindCount : Nat
  combinedOverlayKindCount : Nat
  piRlcFamilyFirstOverlayKind : Nat
  piRlcFamilyEvenPhaseKind : Nat
  piRlcFamilyOddPhaseKind : Nat
  piRlcFamilyBodySourceRows : Nat
  piRlcFamilyBodyEvenSourceRows : Nat
  piRlcFamilyBodyOddSourceRows : Nat
  piRlcFamilyBodyEvenRows : Nat
  piRlcFamilyBodyOddRows : Nat
  piRlcFamilyBodyEvenColumns : Nat
  piRlcFamilyBodyOddColumns : Nat
  piRlcFamilyOverlayRows : Nat
  piRlcFamilyOverlayColumns : Nat
  piRlcFamilyLinkFieldCount : Nat
  piRlcFamilyTotalLinkFieldCount : Nat
  phasePublicLogicalColumns : Nat
  phasePublicColumns : Nat
  afterStateDigestStart : Nat
  afterStateDigestEnd : Nat
  beforeStateDigestStart : Nat
  beforeStateDigestEnd : Nat
  beforeCursorStart : Nat
  beforeCursorEnd : Nat
  afterCursorStart : Nat
  afterCursorEnd : Nat
  phasePublicPaddingStart : Nat
  phasePublicPaddingEnd : Nat
  lifecycleGroupMap : List Nat
  circuitKindMap : List Nat
  claimCoordinateOverlayKindMap : List Nat
  piRlcFamilyOverlayKindMap : List Nat
  claimCoordinateOverlayLinkRuns : List RawOverlayLinkRun
  piRlcFamilyOverlayLinkRuns : List RawFieldLinkRun
  runs : List RawRun
  deriving DecidableEq, Repr

def RawProgram.expanded (raw : RawProgram) : List (Nat × Nat) :=
  raw.runs.flatMap RawRun.expand

def RawRun.valid (run : RawRun) : Bool :=
  run.phaseCode < 19 && 0 < run.count

def RawOverlayLinkRun.valid (raw : RawProgram)
    (run : RawOverlayLinkRun) : Bool :=
  0 < run.overlayKind &&
    run.overlayKind < raw.claimCoordinateOverlayKindCount &&
    run.phaseKind < raw.circuitKindCount &&
    run.chunkIndex < raw.claimChunks &&
    run.activeOffsetStart + run.activeFieldCount <= raw.claimChunkFields

def ProgramValid (raw : RawProgram) : Prop :=
  raw.schemaVersion = 8 /\
    0 < raw.stateChunkFields /\
    0 < raw.priorStateChunks /\
    (raw.priorStateChunks - 1) * raw.stateChunkFields <
      raw.priorStateFrameFields /\
    raw.priorStateFrameFields <=
      raw.priorStateChunks * raw.stateChunkFields /\
    0 < raw.claimChunkFields /\
    0 < raw.claimChunks /\
    (raw.claimChunks - 1) * raw.claimChunkFields < raw.claimFrameFields /\
    raw.claimFrameFields <= raw.claimChunks * raw.claimChunkFields /\
    0 < raw.successorPrefixChunks /\
    (raw.successorPrefixChunks - 1) * raw.stateChunkFields <
      raw.successorPrefixFrameFields /\
    raw.successorPrefixFrameFields <=
      raw.successorPrefixChunks * raw.stateChunkFields /\
    raw.runs.all RawRun.valid = true /\
    raw.expanded.length = raw.workItemCount /\
    raw.firstPiRlcFamilyProgramCursor + raw.piRlcFamilies ≤
      raw.workItemCount /\
    0 < raw.lifecycleGroupCount /\
    0 < raw.circuitKindCount /\
    0 < raw.claimCoordinateOverlayKindCount /\
    raw.piRlcFamilyFirstOverlayKind =
      raw.claimCoordinateOverlayKindCount /\
    raw.combinedOverlayKindCount =
      raw.piRlcFamilyFirstOverlayKind + raw.piRlcFamilies /\
    raw.piRlcFamilyEvenPhaseKind < raw.circuitKindCount /\
    raw.piRlcFamilyOddPhaseKind < raw.circuitKindCount /\
    0 < raw.piRlcFamilyBodySourceRows /\
    raw.piRlcFamilyBodySourceRows ≤ raw.piRlcFamilyBodyEvenSourceRows /\
    raw.piRlcFamilyBodySourceRows ≤ raw.piRlcFamilyBodyOddSourceRows /\
    raw.piRlcFamilyBodyEvenSourceRows ≤ raw.piRlcFamilyBodyEvenRows /\
    raw.piRlcFamilyBodyOddSourceRows ≤ raw.piRlcFamilyBodyOddRows /\
    0 < raw.piRlcFamilyBodyEvenColumns /\
    raw.piRlcFamilyBodyEvenColumns ≤ raw.piRlcFamilyBodyOddColumns /\
    0 < raw.piRlcFamilyOverlayRows /\
    0 < raw.piRlcFamilyOverlayColumns /\
    0 < raw.piRlcFamilyLinkFieldCount /\
    raw.piRlcFamilyTotalLinkFieldCount =
      raw.piRlcFamilies * raw.piRlcFamilyLinkFieldCount /\
    raw.afterStateDigestStart = 1 /\
    raw.afterStateDigestEnd = raw.afterStateDigestStart + 256 /\
    raw.beforeStateDigestStart = raw.afterStateDigestEnd /\
    raw.beforeStateDigestEnd = raw.beforeStateDigestStart + 256 /\
    raw.beforeCursorStart = raw.beforeStateDigestEnd /\
    raw.beforeCursorEnd = raw.beforeCursorStart + 64 /\
    raw.afterCursorStart = raw.beforeCursorEnd /\
    raw.afterCursorEnd = raw.afterCursorStart + 64 /\
    raw.phasePublicLogicalColumns = raw.afterCursorEnd /\
    raw.phasePublicPaddingStart = raw.phasePublicLogicalColumns /\
    raw.phasePublicPaddingEnd = raw.phasePublicColumns /\
    raw.phasePublicLogicalColumns ≤ raw.phasePublicColumns /\
    raw.phasePublicColumns % 54 = 0 /\
    raw.lifecycleGroupMap.length = raw.workItemCount /\
    raw.circuitKindMap.length = raw.workItemCount /\
    raw.claimCoordinateOverlayKindMap.length = raw.workItemCount /\
    raw.piRlcFamilyOverlayKindMap.length = raw.workItemCount /\
    raw.claimCoordinateOverlayLinkRuns.length + 1 =
      raw.claimCoordinateOverlayKindCount /\
    raw.claimCoordinateOverlayLinkRuns.map (fun run => run.overlayKind) =
      (List.range raw.claimCoordinateOverlayKindCount).drop 1 /\
    raw.lifecycleGroupMap.all (fun group => group < raw.lifecycleGroupCount) = true /\
    raw.circuitKindMap.all (fun kind => kind < raw.circuitKindCount) = true /\
    raw.claimCoordinateOverlayKindMap.all
      (fun kind => kind < raw.claimCoordinateOverlayKindCount) = true /\
    raw.piRlcFamilyOverlayKindMap.all
      (fun kind => kind < raw.combinedOverlayKindCount) = true /\
    raw.claimCoordinateOverlayLinkRuns.all
      (RawOverlayLinkRun.valid raw) = true /\
    raw.piRlcFamilyOverlayLinkRuns.all
      (RawFieldLinkRun.valid raw.piRlcFamilyBodyEvenColumns
        raw.piRlcFamilyOverlayColumns) = true /\
    (raw.piRlcFamilyOverlayLinkRuns.map RawFieldLinkRun.linkCount).sum =
      raw.piRlcFamilyLinkFieldCount

instance programValidDecidable (raw : RawProgram) :
    Decidable (ProgramValid raw) := by
  unfold ProgramValid
  infer_instance

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingProgram.Artifact
