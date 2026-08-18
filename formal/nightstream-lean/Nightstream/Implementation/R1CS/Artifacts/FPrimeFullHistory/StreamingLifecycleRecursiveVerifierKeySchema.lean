import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeSchema

/-!
Schema for the exact recursive lifecycle verifier-key source-stage artifact.

The generated data is non-authoritative structure. A handwritten leaf theorem
must prove that the represented rows imply the typed verifier-key relation.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey.Artifact

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact

structure Range where
  start : Nat
  stop : Nat
deriving DecidableEq, Repr

structure SourceRun where
  sourceRows : Range
  disposition : String
  emittedStart : Option Nat
deriving DecidableEq, Repr

structure FinalRun where
  family : String
  rows : Range
  rewriteId : Option Nat
deriving DecidableEq, Repr

structure HashBlock where
  sourceRows : Range
  recipe : VariableHashRecipe
deriving DecidableEq, Repr

structure DigestBinding where
  sourceRows : Range
  leftColumns : List Nat
  rightColumns : List Nat
deriving DecidableEq, Repr

def DigestBinding.row (binding : DigestBinding) (lane : Nat) : Row :=
  ⟨[(binding.leftColumns.getD lane 0, 1),
      (binding.rightColumns.getD lane 0, goldilocksP - 1)],
    [(0, 1)], []⟩

def DigestBinding.rows (binding : DigestBinding) : List Row :=
  (List.range 4).map binding.row

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceArtifactIdentity : String
  finalArtifactIdentity : String
  stagePath : String
  occurrence : Nat
  sourceRows : Range
  sourceColumns : Range
  structureDigestColumns : Range
  ajtaiPpDigestColumns : Range
  initialSemanticStateDigestColumns : Range
  baseVerifierKeyHash : HashBlock
  policyVerifierKeyHash : HashBlock
  policyDigestBinding : DigestBinding
  initialBoundaryHash : HashBlock
  initialBoundaryBinding : DigestBinding
  publicTraceBinding : DigestBinding
  finalRowCount : Nat
  sourceRuns : List SourceRun
  finalRuns : List FinalRun
deriving DecidableEq, Repr

/-- Explicit source-run partition. Each constructor checks one small range;
the certificate never evaluates the complete generated list. -/
inductive SourceRunChain : Nat -> List SourceRun -> Nat -> Prop where
  | nil (cursor : Nat) : SourceRunChain cursor [] cursor
  | cons {cursor finalCursor : Nat} {run : SourceRun} {runs : List SourceRun}
      (starts : run.sourceRows.start = cursor)
      (ordered : run.sourceRows.start <= run.sourceRows.stop)
      (tail : SourceRunChain run.sourceRows.stop runs finalCursor) :
      SourceRunChain cursor (run :: runs) finalCursor

/-- Every final run is a valid interval inside the final row domain. -/
inductive FinalRunsWithin (bound : Nat) : List FinalRun -> Prop where
  | nil : FinalRunsWithin bound []
  | cons {run : FinalRun} {runs : List FinalRun}
      (ordered : run.rows.start <= run.rows.stop)
      (inside : run.rows.stop <= bound)
      (tail : FinalRunsWithin bound runs) :
      FinalRunsWithin bound (run :: runs)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingLifecycleRecursiveVerifierKey.Artifact
