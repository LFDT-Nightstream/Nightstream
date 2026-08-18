import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingArtifactLeafSchema

/-!
Contract: schema for the compact exact Rust prior-state replay source arms.

Owns source-coordinate recipe calls, residual rows, the public-prefix column
permutation, semantic source slices, and normalized ownership ranges. It owns
no generated data, lifecycle target authority, or permission to remove a row.

Emits constraints: no. It describes existing constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

inductive ArmKind where
  | full
  | final
deriving DecidableEq, Repr

structure Range where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

structure ColumnBinding where
  source : Nat
  normalized : Nat
deriving DecidableEq, Repr

structure ColumnLayout where
  constantOne : ColumnBinding
  publicColumns : List ColumnBinding
  normalizedPrivateStart : Nat
deriving DecidableEq, Repr

structure PhysicalStage where
  path : String
  sourceRows : Range
  normalizedPrivateColumns : Range
deriving DecidableEq, Repr

structure NamedRange where
  name : String
  range : Range
deriving DecidableEq, Repr

structure SemanticColumns where
  beforeReplayState : List ColumnBinding
  afterReplayState : List ColumnBinding
  chunk : List ColumnBinding
  targetDigest : List ColumnBinding
  beforeLocalStateDigest : List ColumnBinding
  afterLocalStateDigest : List ColumnBinding
  beforeProgramCursor : ColumnBinding
  afterProgramCursor : ColumnBinding
  afterXOutBits : List ColumnBinding
  beforeXOutBits : List ColumnBinding
  beforeProgramCursorBits : List ColumnBinding
  afterProgramCursorBits : List ColumnBinding
  beforeXOutPreimage : List ColumnBinding
  afterXOutPreimage : List ColumnBinding
  beforeBoundary : List ColumnBinding
  afterBoundary : List ColumnBinding
  beforeAccumulator : List ColumnBinding
  afterAccumulator : List ColumnBinding
  delayedNebulaPayload : List ColumnBinding
deriving DecidableEq, Repr

/-- Compact exact source artifact. Identity fields are non-authoritative review
metadata. Exactness comes from the represented recipe calls and residual rows. -/
structure RawArm where
  schemaVersion : Nat
  profileId : String
  branchScope : String
  lifecycleScope : String
  armKind : ArmKind
  sourcePath : String
  sourceHashSchema : String
  sourceArtifactIdentity : String
  finalTargetBindingStatus : String
  sourceRowCount : Nat
  sourceColumnCount : Nat
  normalizedColumnCount : Nat
  publicColumnCount : Nat
  columnLayout : ColumnLayout
  semanticColumns : SemanticColumns
  physicalStages : List PhysicalStage
  rowFamilies : List NamedRange
  columnFamilies : List NamedRange
  poseidon2Calls : List Poseidon2Call.Call
  canonicalU64Calls : List CanonicalCall
  residualRows : List IndexedRow
deriving DecidableEq, Repr

/-- Satisfaction of all exact source-row families represented by one arm. The
constant-one and canonical-residue conditions stay explicit hypotheses. -/
def RawArm.Satisfied (arm : RawArm) (assignment : Nat → Nat) : Prop :=
  (∀ call ∈ arm.canonicalU64Calls, call.Satisfied assignment) ∧
    (∀ call ∈ arm.poseidon2Calls, Satisfies call.rows assignment) ∧
    ∀ indexed ∈ arm.residualRows, RowHolds assignment indexed.row

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPriorStateReplaySource.Artifact
