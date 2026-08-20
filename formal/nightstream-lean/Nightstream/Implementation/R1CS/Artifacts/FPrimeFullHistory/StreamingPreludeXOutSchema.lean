import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPreludeSourceSchema
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingVariableHashRecipeSchema

/-!
Contract: compact exact-row schema for the two Prelude XOut hashes.

Each block owns one 32-field Poseidon2 recipe in original Rust source
coordinates and the source-to-normalized bindings for its inputs and output.
Compact public spans define a total pullback from the normalized assignment.
Rust checks every represented source row and its pullback against the source
and normalized matrices.

Does not own lifecycle input authority, public-input acceptance, or collision
resistance.

Emits constraints: no. It describes Rust-emitted constraints.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOut.Artifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Program
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeSource.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingVariableHashRecipe.Artifact

structure HashBlock where
  sourceRows : Range
  preimageColumns : List ColumnBinding
  digestColumns : List ColumnBinding
  canonicalCalls : List CanonicalCall
  normalizedBitBase : Nat
  recipe : VariableHashRecipe
deriving DecidableEq, Repr

def HashBlock.SourceSatisfied
    (block : HashBlock) (assignment : Nat → Nat) : Prop :=
  Satisfies block.recipe.trace.rows assignment ∧
    ∀ call ∈ block.canonicalCalls, call.Satisfied assignment

structure ColumnSpan where
  sourceStart : Nat
  normalizedStart : Nat
  length : Nat
deriving DecidableEq, Repr

def ColumnSpan.publicBefore (span : ColumnSpan) (source : Nat) : Nat :=
  min span.length (source - span.sourceStart)

def ColumnSpan.mapColumn
    (span : ColumnSpan) (source fallback : Nat) : Nat :=
  if span.sourceStart ≤ source ∧ source < span.sourceStart + span.length then
    span.normalizedStart + (source - span.sourceStart)
  else
    fallback

structure RawArtifact where
  schemaVersion : Nat
  profileId : String
  sourceArtifactIdentity : String
  branchScope : String
  lifecycleScope : String
  stagePath : String
  sourceRowCount : Nat
  sourceColumnCount : Nat
  normalizedColumnCount : Nat
  publicSpans : List ColumnSpan
  afterXOut : HashBlock
  beforeXOut : HashBlock
deriving DecidableEq, Repr

def RawArtifact.normalizedPrivateStart (artifact : RawArtifact) : Nat :=
  1 + (artifact.publicSpans.map ColumnSpan.length).sum

def RawArtifact.publicBefore
    (artifact : RawArtifact) (source : Nat) : Nat :=
  (artifact.publicSpans.map fun span => span.publicBefore source).sum

def RawArtifact.normalizedColumn
    (artifact : RawArtifact) (source : Nat) : Nat :=
  if source = 0 then
    0
  else
    artifact.publicSpans.foldr
      (fun span fallback => span.mapColumn source fallback)
      (artifact.normalizedPrivateStart + (source - 1 - artifact.publicBefore source))

def RawArtifact.sourceAssignment
    (artifact : RawArtifact) (assignment : Nat → Nat) : Nat → Nat :=
  fun source => assignment (artifact.normalizedColumn source)

def RawArtifact.Satisfied
    (artifact : RawArtifact) (assignment : Nat → Nat) : Prop :=
  artifact.afterXOut.SourceSatisfied (artifact.sourceAssignment assignment) ∧
    artifact.beforeXOut.SourceSatisfied (artifact.sourceAssignment assignment)

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPreludeXOut.Artifact
