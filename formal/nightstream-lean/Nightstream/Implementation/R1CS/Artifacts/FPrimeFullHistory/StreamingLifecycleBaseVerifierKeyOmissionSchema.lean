import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingLifecycleRecursiveVerifierKeySchema

/-!
Schema for the compact base verifier-key omission audit.

Rust exhaustively projects one changed column from the exact source rows and
replays the complete candidate. Lean owns the mutation and target-failure
argument. The projection is not a digest and does not replace the source rows.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission.Artifact

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey.Artifact

inductive MatrixSide where
  | a
  | b
  | c
deriving DecidableEq, Repr

structure Occurrence where
  sourceRow : Nat
  side : MatrixSide
  coefficient : Nat
  family : String
deriving DecidableEq, Repr

inductive OccurrenceOwnership (family : String) : List Occurrence -> Prop where
  | nil : OccurrenceOwnership family []
  | cons {occurrence : Occurrence} {occurrences : List Occurrence}
      (owned : occurrence.family = family)
      (tail : OccurrenceOwnership family occurrences) :
      OccurrenceOwnership family (occurrence :: occurrences)

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  lifecycleScope : String
  sourceArtifactIdentity : String
  finalArtifactIdentity : String
  family : String
  stagePath : String
  occurrence : Nat
  sourceRows : Range
  sourceColumns : Range
  sourceRowCount : Nat
  selectedRowCount : Nat
  retainedRowCount : Nat
  columnCount : Nat
  constantOneColumn : Nat
  changedColumn : Nat
  baselineValue : Nat
  candidateValue : Nat
  finalRowCount : Nat
  sourceRuns : List SourceRun
  finalRuns : List FinalRun
  occurrences : List Occurrence
deriving DecidableEq, Repr

def RetainedRowsIgnoreChangedColumn (artifact : RawArtifact) : Prop :=
  OccurrenceOwnership artifact.family artifact.occurrences

structure RawArtifact.Valid (artifact : RawArtifact) : Prop where
  schemaVersion : artifact.schemaVersion = 1
  profileId : artifact.profileId =
    "nightstream/goldilocks/streaming-base-authority-attack/v1"
  lifecycleScope : artifact.lifecycleScope = "base"
  family : artifact.family = "fprime.base.verifier_key"
  stagePath : artifact.stagePath = artifact.family
  sourceRowsOrdered : artifact.sourceRows.start <= artifact.sourceRows.stop
  sourceColumnsOrdered :
    artifact.sourceColumns.start <= artifact.sourceColumns.stop
  rowPartition :
    artifact.selectedRowCount + artifact.retainedRowCount =
      artifact.sourceRowCount
  constantOne : artifact.constantOneColumn = 0
  changedColumnInBounds : artifact.changedColumn < artifact.columnCount
  baselineCanonical : artifact.baselineValue < goldilocksP
  candidateCanonical : artifact.candidateValue < goldilocksP
  targetFails : artifact.candidateValue ≠ artifact.baselineValue
  sourceRunsCover : SourceRunChain artifact.sourceRows.start
    artifact.sourceRuns artifact.sourceRows.stop
  finalRunsInside : FinalRunsWithin artifact.finalRowCount artifact.finalRuns
  retainedRowsIgnore : RetainedRowsIgnoreChangedColumn artifact

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleBaseVerifierKeyOmission.Artifact
