import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingArtifactLeafSchema

/-!
Contract: schema for the compact exact Rust Prelude source artifact.

Owns source-coordinate recipe calls, residual rows, the public-prefix column
permutation, and normalized ownership ranges. It owns no generated data,
semantic proof, or permission to remove a source row.

Emits constraints: no. It describes existing constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact

structure Range where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

structure ColumnBinding where
  source : Nat
  normalized : Nat
deriving DecidableEq, Repr

/-- Exact canonical public-prefix layout. Column zero remains the constant-one
column. Listed source columns move, in order, to normalized columns `1, 2, ...`.
All unlisted source columns retain source order after that public prefix. -/
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
  initialReplayState : List ColumnBinding
  beforeLocalStateDigest : List ColumnBinding
  afterLocalStateDigest : List ColumnBinding
  beforeProgramCursor : ColumnBinding
  afterProgramCursor : ColumnBinding
deriving DecidableEq, Repr

/-- Compact exact source artifact. The identity is non-authoritative review
metadata. Exactness comes from the represented recipe calls and residual rows. -/
structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  branchScope : String
  lifecycleScope : String
  sourcePath : String
  sourceArtifactIdentity : String
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

/-- Satisfaction of every exact source-row family represented by the compact
artifact. The constant-one and canonical-residue conditions remain explicit
hypotheses of correspondence theorems. -/
def RawArtifact.Satisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  (∀ call ∈ artifact.canonicalU64Calls,
      call.Satisfied assignment) ∧
    (∀ call ∈ artifact.poseidon2Calls,
      Satisfies call.rows assignment) ∧
    ∀ indexed ∈ artifact.residualRows,
      RowHolds assignment indexed.row

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Artifact
