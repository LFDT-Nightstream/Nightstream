import Nightstream.Implementation.Nebula.Production.Carrier.StreamingStateBinding
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplay
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact Rust artifact boundary for one bounded claim-replay step.

Assurance tier: Rust-conformant for property
`FPRIME-STREAMING-CLAIM-REPLAY-ROWS-V1`.

Owns the exact field-native and low-norm dimensions, complete row ownership,
and transport of every canonical-u64 and Poseidon2 leaf to its existing Lean
soundness theorem.

Does not own the interpretation of glue columns as a replay state, the
86-step frame theorem, recursive lifecycle integration, or the Poseidon2
collision reduction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplay
open Nightstream.Implementation.R1CS.Program

theorem artifact_valid : rawArtifact.Valid :=
  rawArtifact_valid

theorem exact_shape :
    rawArtifact.full.rowCount = 156384 /\
      rawArtifact.full.columnCount = 157305 /\
      rawArtifact.finalChunk.rowCount = 149837 /\
      rawArtifact.finalChunk.columnCount = 150705 /\
      rawArtifact.lowNormRows = 51338 /\
      rawArtifact.lowNormColumns = 536112 /\
      rawArtifact.lowNormPublicColumns = 2592 /\
      rawArtifact.lowNormTotalCoordinates = 536086 := by
  native_decide

theorem exact_leaf_counts :
    rawArtifact.full.canonicalCalls.length = 40 /\
      rawArtifact.finalChunk.canonicalCalls.length = 40 /\
      rawArtifact.full.poseidon2Calls.length = 256 /\
      rawArtifact.finalChunk.poseidon2Calls.length = 245 /\
      rawArtifact.full.glueRows.length = 24 /\
      rawArtifact.finalChunk.glueRows.length = 77 := by
  native_decide

/-- The width census identifies the next architecture target. Almost all
branch-private coordinates belong to Poseidon2 call traces, not carried
protocol state. -/
theorem poseidon2_width_attribution_exact :
    rawArtifact.lowNormFullBranchCoordinates =
        rawArtifact.lowNormFullPoseidon2Coordinates + 943 /\
      rawArtifact.lowNormFinalBranchCoordinates =
        rawArtifact.lowNormFinalPoseidon2Coordinates /\
      rawArtifact.lowNormSharedPrivateCoordinates = 1103 := by
  native_decide

/-- Every canonical-word leaf in a satisfying arm has the unique canonical
64-bit value specified by the Lean-owned recipe. -/
theorem canonical_call_refines
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied assignment)
    (call : CanonicalCall) (member : call ∈ arm.canonicalCalls) :
    Refines assignment call.layout := by
  apply CanonicalU64RecipeSound.sound goldilocks_euclidPrime canonical one
  exact satisfied.1 call member

/-- Every Poseidon2 leaf in a satisfying arm agrees with the exact production
permutation interpreter under its Rust-emitted column renaming. -/
theorem poseidon2_call_refines
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied assignment)
    (call : Poseidon2Call.Call) (member : call ∈ arm.poseidon2Calls) :
    AgreeOn
      (Poseidon2PermutationSound.interpret
        (pullAssignment assignment call.columnMap))
      (pullAssignment assignment call.columnMap)
      (knownAfter Poseidon2Permutation.inputColumns
        Poseidon2Permutation.definitions) := by
  apply Poseidon2PermutationSound.poseidon2Permutation_renamed_sound
    call.columnMap call.columnMap_zero canonical one
  simpa [Poseidon2Call.Call.rows] using satisfied.2.1 call member

/-- Glue rows are stored exactly and remain enforced after the repeated leaf
programs are compressed into call certificates. -/
theorem glue_row_holds
    (arm : RawArm) (assignment : Nat → Nat)
    (satisfied : arm.Satisfied assignment)
    (indexed : IndexedRow) (member : indexed ∈ arm.glueRows) :
    RowHolds assignment indexed.row := by
  exact satisfied.2.2 indexed member

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
