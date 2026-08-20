import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerIsFsLeafRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafRowSound
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionSound

/-!
Contract: bind all three terminal leaf digests to their complete Rust-owned
972-field source slices and checked commitment/hash rows.

This module does not give the source slices semantic fresh-witness authority.
A lifecycle parent must bind all three slices from the same opened witness.

Emits constraints: no.
-/

set_option autoImplicit false
set_option maxRecDepth 65536

namespace Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge

open Nightstream.SuperNeo.Concrete
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalFinalizer.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalFullFinalizer
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRelation
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionRowSound
open Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerTransitionSound

private abbrev artifact :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalFullFinalizer.rawArtifact

abbrev OpsSatisfied (assignment : Nat → Nat) : Prop :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafRowSound.OpsLeafSatisfied
    assignment

abbrev IsSatisfied (assignment : Nat → Nat) : Prop :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerIsFsLeafRowSound.IsLeafSatisfied
    assignment

abbrev FsSatisfied (assignment : Nat → Nat) : Prop :=
  Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerIsFsLeafRowSound.FsLeafSatisfied
    assignment

def leafArtifact (lane : Fin 3) : LeafHashArtifact :=
  [artifact.opsLeaf, artifact.isLeaf, artifact.fsLeaf].getD lane.val
    artifact.opsLeaf

def authoritativeColumns (lane : Fin 3) : List Nat :=
  [artifact.opsColumns, artifact.isColumns, artifact.fsColumns].getD lane.val []

/-- Complete ordered commitment-field sources for operations, initial-state,
and final-state leaves. -/
def inputValues (assignment : Nat → Nat) : Fin 3 → List Nat := fun lane =>
  (authoritativeColumns lane).map assignment

theorem input_length (assignment : Nat → Nat) (lane : Fin 3) :
    (inputValues assignment lane).length = commitmentDataFields := by
  fin_cases lane <;> rfl

def digestField (value : Nat) : F :=
  ⟨value % goldilocksModulus, Nat.mod_lt _ (by decide)⟩

/-- The three leaf digests computed from exact canonical openings, both
seeded maps, and the final Poseidon2 envelopes. -/
def computedLeaves (assignment : Nat → Nat) : LeafDigests :=
  fun lane output =>
    digestField
      (Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound.computedDigest
        (leafArtifact lane) assignment output)

private theorem leaf_digest_feeds_chain (lane : Fin 3) (output : Fin 4) :
    (advanceChainLink lane).recipe.payloadColumns.getD output.val 0 =
      ((leafArtifact lane).envelopeRecipe.outputColumns.getD output.val 0) := by
  fin_cases lane <;> fin_cases output <;> rfl

private theorem assigned_digest_field_eq_computed
    (leaf : LeafHashArtifact)
    (assignment : Nat → Nat)
    (hash :
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound.assignedDigest
          leaf assignment =
        Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound.computedDigest
          leaf assignment)
    (output : Fin 4) :
    fieldValue assignment
        (leaf.envelopeRecipe.outputColumns.getD output.val 0) =
      digestField
        (Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound.computedDigest
          leaf assignment output) := by
  apply Fin.ext
  exact congrArg (fun value => value % goldilocksModulus)
    (congrFun hash output)

/-- The three retained leaf row families make the chain consume exactly the
digests computed from all three authoritative source slices. -/
theorem rows_bind_assignedLeaves
    (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (opsSatisfied : OpsSatisfied assignment)
    (isSatisfied : IsSatisfied assignment)
    (fsSatisfied : FsSatisfied assignment) :
    assignedLeaves assignment = computedLeaves assignment := by
  have opsSound :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafRowSound.rows_sound
      assignment canonical one opsSatisfied
  have isSound :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerIsFsLeafRowSound.isLeaf_rows_sound
      assignment canonical one isSatisfied
  have fsSound :=
    Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerIsFsLeafRowSound.fsLeaf_rows_sound
      assignment canonical one fsSatisfied
  have opsHash :
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound.assignedDigest
          artifact.opsLeaf assignment =
        Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound.computedDigest
          artifact.opsLeaf assignment := by
    simpa [
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafEnvelopeRowSound.assignedDigest,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerOpsLeafEnvelopeRowSound.computedDigest,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound.assignedDigest,
      Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafRowSound.computedDigest]
      using opsSound.envelope.hash
  funext lane output
  change fieldValue assignment
      ((advanceChainLink lane).recipe.payloadColumns.getD output.val 0) =
    computedLeaves assignment lane output
  rw [leaf_digest_feeds_chain lane output]
  fin_cases lane
  · simpa [computedLeaves, leafArtifact] using
      assigned_digest_field_eq_computed artifact.opsLeaf assignment opsHash
        output
  · simpa [computedLeaves, leafArtifact] using
      assigned_digest_field_eq_computed artifact.isLeaf assignment
        isSound.envelope.hash output
  · simpa [computedLeaves, leafArtifact] using
      assigned_digest_field_eq_computed artifact.fsLeaf assignment
        fsSound.envelope.hash output

end Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingTerminalFullFinalizerLeafSourceBridge
