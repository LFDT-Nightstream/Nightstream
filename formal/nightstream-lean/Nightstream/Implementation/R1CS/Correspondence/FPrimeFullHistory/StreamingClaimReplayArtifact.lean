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
    rawArtifact.full.rowCount = 201959 /\
      rawArtifact.full.columnCount = 202866 /\
      rawArtifact.finalChunk.rowCount = 188705 /\
      rawArtifact.finalChunk.columnCount = 189771 /\
      rawArtifact.lowNormRows = 61034 /\
      rawArtifact.lowNormColumns = 673866 /\
      rawArtifact.lowNormPublicColumns = 648 /\
      rawArtifact.lowNormTotalCoordinates = 673865 := by
  native_decide

theorem exact_leaf_counts :
    rawArtifact.full.canonicalCalls.length = 10 /\
      rawArtifact.finalChunk.canonicalCalls.length = 10 /\
      rawArtifact.full.poseidon2Calls.length = 324 /\
      rawArtifact.finalChunk.poseidon2Calls.length = 313 /\
      rawArtifact.full.coordinateCalls.length = 1 /\
      rawArtifact.finalChunk.coordinateCalls.length = 0 /\
      rawArtifact.full.glueRows.length = 270 /\
      rawArtifact.finalChunk.glueRows.length = 215 := by
  native_decide

/-- The normalized public prefix is eight digest lanes followed by the before
and after program cursors. Both physical arms use the same word roles. -/
theorem exact_public_word_layout :
    rawArtifact.full.publicColumnCount = 641 /\
      rawArtifact.finalChunk.publicColumnCount = 641 /\
      rawArtifact.full.publicWordCallIndices =
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1] /\
      rawArtifact.finalChunk.publicWordCallIndices =
        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1] /\
      rawArtifact.lowNormPublicColumns = 648 := by
  native_decide

/-- Both arms use two 128-field digest preimages. Each preimage contains the
20 replay words followed by the 108 carried commitment coordinates. -/
theorem exact_state_word_layout :
    rawArtifact.full.stateWordColumns.length = 256 /\
      rawArtifact.full.stateWordColumns.take 19 = List.range' 1 19 /\
      rawArtifact.full.stateWordColumns[19]? = some 128 /\
      (rawArtifact.full.stateWordColumns.drop 20).take 108 =
        List.range' 20 108 /\
      (rawArtifact.full.stateWordColumns.drop 128).take 19 =
        List.range' 195 19 /\
      rawArtifact.full.stateWordColumns[147]? = some 322 /\
      rawArtifact.full.stateWordColumns.drop 148 = List.range' 214 108 /\
      rawArtifact.finalChunk.stateWordColumns =
        rawArtifact.full.stateWordColumns := by
  native_decide

/-- The width census identifies the next architecture target. Almost all
branch-private coordinates belong to Poseidon2 call traces, not carried
protocol state. -/
theorem poseidon2_width_attribution_exact :
    rawArtifact.lowNormFullBranchCoordinates =
        rawArtifact.lowNormFullPoseidon2Coordinates + 25761 /\
      rawArtifact.lowNormFinalBranchCoordinates =
        rawArtifact.lowNormFinalPoseidon2Coordinates + 22732 /\
      rawArtifact.lowNormSharedPrivateCoordinates = 260 := by
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

/-- Every selected coordinate call enforces its compact source, shape, and
seeded output rows. -/
theorem coordinate_call_holds
    (arm : RawArm) (assignment : Nat → Nat)
    (satisfied : arm.Satisfied assignment)
    (call : CoordinateCall) (member : call ∈ arm.coordinateCalls) :
    Satisfies call.rows assignment := by
  exact satisfied.2.2.1 call member

/-- Glue rows are stored exactly and remain enforced after the repeated leaf
programs are compressed into call certificates. -/
theorem glue_row_holds
    (arm : RawArm) (assignment : Nat → Nat)
    (satisfied : arm.Satisfied assignment)
    (indexed : IndexedRow) (member : indexed ∈ arm.glueRows) :
    RowHolds assignment indexed.row := by
  exact satisfied.2.2.2 indexed member

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayArtifact
