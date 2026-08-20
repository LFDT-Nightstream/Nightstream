/-!
Contract: compact schema for the exact Rust terminal source-to-final profile.

The artifact records lifecycle scope, artifact identities, canonical column
layout, source-stage ownership, and source-slice to final-row bindings. It does
not copy the complete relation, prove row satisfaction, or make a digest an
authority.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfile.Artifact

structure Range where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

abbrev Range.ValidWithin (range : Range) (bound : Nat) : Prop :=
  range.start <= range.stop /\ range.stop <= bound

structure FinalRun where
  family : String
  rows : Range
deriving DecidableEq, Repr

inductive FinalRunsWithin (bound : Nat) : List FinalRun -> Prop where
  | nil : FinalRunsWithin bound []
  | cons {run : FinalRun} {runs : List FinalRun}
      (valid : run.rows.ValidWithin bound)
      (tail : FinalRunsWithin bound runs) :
      FinalRunsWithin bound (run :: runs)

structure ColumnLayout where
  publicColumns : Range
  lifecyclePrivate : Range
  phasePrivate : Range
  scheduleSelectorRuns : List Range
  scheduledRingPadding : Range
  overlayPrivate : Range
  overlaySelectorRuns : List Range
  finalRingPadding : Range
deriving DecidableEq, Repr

structure SliceBinding where
  sourceFields : Range
  sourceDomain : String
  finalLinkRows : Range
deriving DecidableEq, Repr

structure SourceStageBinding where
  occurrence : Nat
  path : String
  sourceRows : Range
  sourceFieldRuns : List Range
  finalRuns : List FinalRun
deriving DecidableEq, Repr

inductive SourceStageBindingsWithin
    (sourceRowBound finalRowBound : Nat) : List SourceStageBinding -> Prop where
  | nil : SourceStageBindingsWithin sourceRowBound finalRowBound []
  | cons {binding : SourceStageBinding} {bindings : List SourceStageBinding}
      (sourceRows : binding.sourceRows.ValidWithin sourceRowBound)
      (finalRuns : FinalRunsWithin finalRowBound binding.finalRuns)
      (tail : SourceStageBindingsWithin sourceRowBound finalRowBound bindings) :
      SourceStageBindingsWithin sourceRowBound finalRowBound
        (binding :: bindings)

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  lifecycleScope : String
  sourceArtifactIdentity : String
  finalArtifactIdentity : String
  acceptedWorkItems : Nat
  terminalArm : Nat
  lifecycleGroup : Nat
  phaseKind : Nat
  scheduleSelectorColumn : Nat
  lifecycleSelectorColumn : Nat
  phaseSelectorColumn : Nat
  sourceRows : Nat
  sourceColumns : Nat
  sourcePublicColumns : Nat
  finalRows : Nat
  finalColumns : Nat
  finalPublicColumns : Nat
  columnLayout : ColumnLayout
  sourceStageOccurrence : Nat
  sourceStagePath : String
  sourceStageRows : Range
  finalStageRuns : List FinalRun
  afterXOutSourceFields : List Nat
  afterNebulaLaneSourceFields : List Nat
  afterLocalStateDigest : SliceBinding
  afterDelayedPayload : SliceBinding
  sourceStageBindings : List SourceStageBinding
deriving DecidableEq, Repr

structure RawArtifact.Valid (artifact : RawArtifact) : Prop where
  schemaVersion : artifact.schemaVersion = 1
  profileId : artifact.profileId =
    "nightstream/goldilocks/streaming-terminal-slice/v1"
  lifecycleScope : artifact.lifecycleScope = "recursive-terminal-arm-435"
  sourceArtifactIdentity : artifact.sourceArtifactIdentity =
    "rust:nightstream/streaming-lifecycle-recursive/source-rows/v1"
  finalArtifactIdentity : artifact.finalArtifactIdentity =
    "rust:nightstream/streaming-selective-ccs/final-rows/v1"
  acceptedWorkItems : artifact.acceptedWorkItems = 436
  terminalArm : artifact.terminalArm + 1 = artifact.acceptedWorkItems
  sourcePublicInside : artifact.sourcePublicColumns <= artifact.sourceColumns
  finalPublicInside : artifact.finalPublicColumns <= artifact.finalColumns
  selectorsInside :
    artifact.scheduleSelectorColumn < artifact.finalColumns /\
      artifact.lifecycleSelectorColumn < artifact.finalColumns /\
      artifact.phaseSelectorColumn < artifact.finalColumns
  sourceStageRows : artifact.sourceStageRows.ValidWithin artifact.sourceRows
  finalStageRuns : FinalRunsWithin artifact.finalRows artifact.finalStageRuns
  xOutLength : artifact.afterXOutSourceFields.length = 32
  nebulaLaneLength : artifact.afterNebulaLaneSourceFields.length = 50
  localStateLength :
    artifact.afterLocalStateDigest.sourceFields.stop -
      artifact.afterLocalStateDigest.sourceFields.start = 4
  localStateRows :
    artifact.afterLocalStateDigest.finalLinkRows.ValidWithin artifact.finalRows
  delayedPayloadRows :
    artifact.afterDelayedPayload.finalLinkRows.ValidWithin artifact.finalRows
  sourceStageBindings : SourceStageBindingsWithin artifact.sourceRows
    artifact.finalRows artifact.sourceStageBindings

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalProfile.Artifact
