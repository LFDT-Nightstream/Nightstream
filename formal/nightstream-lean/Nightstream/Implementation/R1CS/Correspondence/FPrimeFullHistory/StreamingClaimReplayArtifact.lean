import Nightstream.Implementation.Nebula.Production.Carrier.StreamingStateBinding
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplay
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField

/-!
Contract: exact Rust artifact boundary for one bounded claim-replay step.

Assurance tier: Rust-conformant for property
`FPRIME-STREAMING-CLAIM-REPLAY-ROWS-V4`.

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
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCanonicalCallCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayCoordinateCallCertificates
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFinalGlueRowCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFinalPoseidon2CallCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFullGlueRowCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayFullPoseidon2CallCertificate
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayStateWordLayoutCertificate
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingClaimReplay
open Nightstream.Implementation.R1CS.Program

theorem artifact_valid : rawArtifact.Valid :=
  rawArtifact_valid

theorem exact_shape :
    rawArtifact.full.rowCount = 307762 /\
      rawArtifact.full.columnCount = 307491 /\
      rawArtifact.finalChunk.rowCount = 343256 /\
      rawArtifact.finalChunk.columnCount = 342464 /\
      rawArtifact.lowNormRows = 167491 /\
      rawArtifact.lowNormColumns = 808110 /\
      rawArtifact.lowNormPublicColumns = 648 /\
      rawArtifact.lowNormTotalCoordinates = 808068 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem exact_leaf_counts :
    rawArtifact.full.canonicalCalls.length = 10 /\
      rawArtifact.finalChunk.canonicalCalls.length = 10 /\
      rawArtifact.full.poseidon2Calls.length = 378 /\
      rawArtifact.finalChunk.poseidon2Calls.length = 367 /\
      rawArtifact.full.coordinateCalls.length = 2 /\
      rawArtifact.finalChunk.coordinateCalls.length = 1 /\
      rawArtifact.full.glueRows.length = 486 /\
      rawArtifact.finalChunk.glueRows.length = 323 := by
  change fullArm.canonicalCalls.length = 10 /\
    finalArm.canonicalCalls.length = 10 /\
    fullArm.poseidon2Calls.length = 378 /\
    finalArm.poseidon2Calls.length = 367 /\
    fullArm.coordinateCalls.length = 2 /\
    finalArm.coordinateCalls.length = 1 /\
    fullArm.glueRows.length = 486 /\
    finalArm.glueRows.length = 323
  exact ⟨fullArm_canonicalCalls_valid.1,
    finalArm_canonicalCalls_valid.1,
    fullArm_poseidon2Calls_length,
    finalArm_poseidon2Calls_length,
    by simpa only [fullArm_coordinateCalls_exact, List.length_cons,
      List.length_nil],
    by simpa only [finalArm_coordinateCalls_exact, List.length_cons,
      List.length_nil],
    fullArm_glueRows_length,
    finalArm_glueRows_length⟩

/-- The full arm owns both coordinate maps. The final arm owns the remaining
statement-and-fresh slice and carries the completed running-metadata map. -/
theorem exact_coordinate_map_kinds :
    rawArtifact.full.coordinateCalls.map CoordinateCall.mapKind =
        [.statementFresh, .runningMetadata] /\
      rawArtifact.finalChunk.coordinateCalls.map CoordinateCall.mapKind =
        [.statementFresh] := by
  change fullArm.coordinateCalls.map CoordinateCall.mapKind =
      [.statementFresh, .runningMetadata] /\
    finalArm.coordinateCalls.map CoordinateCall.mapKind = [.statementFresh]
  constructor
  · rw [fullArm_coordinateCalls_exact]
    rfl
  · rw [finalArm_coordinateCalls_exact]
    rfl

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
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- Each side of the transition has 20 replay words followed by both
108-coordinate commitments. Both physical arms use the same state roles. -/
theorem exact_state_word_layout :
    rawArtifact.full.stateWordColumns.length = 472 /\
      rawArtifact.full.stateWordColumns.take 19 = List.range' 1 19 /\
      rawArtifact.full.stateWordColumns[19]? = some 236 /\
      (rawArtifact.full.stateWordColumns.drop 20).take 216 =
        List.range' 20 216 /\
      (rawArtifact.full.stateWordColumns.drop 236).take 19 =
        List.range' 303 19 /\
      rawArtifact.full.stateWordColumns[255]? = some 538 /\
      rawArtifact.full.stateWordColumns.drop 256 = List.range' 322 216 /\
      rawArtifact.finalChunk.stateWordColumns =
        rawArtifact.full.stateWordColumns := by
  change fullArm.stateWordColumns.length = 472 /\
    fullArm.stateWordColumns.take 19 = List.range' 1 19 /\
    fullArm.stateWordColumns[19]? = some 236 /\
    (fullArm.stateWordColumns.drop 20).take 216 = List.range' 20 216 /\
    (fullArm.stateWordColumns.drop 236).take 19 = List.range' 303 19 /\
    fullArm.stateWordColumns[255]? = some 538 /\
    fullArm.stateWordColumns.drop 256 = List.range' 322 216 /\
    finalArm.stateWordColumns = fullArm.stateWordColumns
  exact arms_stateWordLayout_exact

/-- The width census identifies the next architecture target. Almost all
branch-private coordinates belong to Poseidon2 call traces, not carried
protocol state. -/
theorem poseidon2_width_attribution_exact :
    rawArtifact.lowNormFullBranchCoordinates =
        rawArtifact.lowNormFullPoseidon2Coordinates + 48184 /\
      rawArtifact.lowNormFinalBranchCoordinates =
        rawArtifact.lowNormFinalPoseidon2Coordinates + 60196 /\
      rawArtifact.lowNormSharedPrivateCoordinates = 476 := by
  exact ⟨rfl, rfl, rfl⟩

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
