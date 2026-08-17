import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublic
import Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
import Nightstream.Implementation.R1CS.Canonical.GoldilocksField
import Nightstream.Implementation.R1CS.Core.Poseidon2Call

/-!
Contract: exact Rust artifact boundary for the PiRLC family public suffix.

Assurance tier: Rust-conformant for property
`FPRIME-STREAMING-PIRLC-FAMILY-FULL-XOUT-ROWS-V3`.

Owns both parity shapes, the exact suffix boundary outside one delegated
phase-envelope range, the 1045-field local states, both 32-field full XOut
preimages and four-field outputs, and transport of every owned canonical-u64
and Poseidon2 leaf to its Lean soundness theorem.

Does not own the delegated phase-envelope rows, interpretation of the state
columns as `FamilyState`, the state-digest transcript, the source-prefix
relation, selective lowering, or recursive lifecycle integration.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64RecipeSound
open Nightstream.Implementation.R1CS.Canonical.GoldilocksField
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublic.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyPublic
open Nightstream.Implementation.R1CS.Program

inductive ArmKind where
  | even
  | odd
deriving DecidableEq, Repr

def armFor : ArmKind → RawArm
  | .even => evenArm
  | .odd => oddArm

theorem artifact_valid : rawArtifact.Valid :=
  rawArtifact_valid

theorem exact_shape :
    rawArtifact.even.sourceRowCount = 310646 /\
      rawArtifact.even.rowCount = 1300897 /\
      rawArtifact.even.columnCount = 1301126 /\
      rawArtifact.odd.sourceRowCount = 311846 /\
      rawArtifact.odd.rowCount = 1302097 /\
      rawArtifact.odd.columnCount = 1302326 /\
      rawArtifact.lowNormRows = 491046 /\
      rawArtifact.lowNormColumns = 8858862 /\
      rawArtifact.lowNormPublicColumns = 648 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

theorem exact_leaf_counts :
    rawArtifact.even.canonicalCalls.length = 11 /\
      rawArtifact.odd.canonicalCalls.length = 11 /\
      rawArtifact.even.poseidon2Calls.length = 544 /\
      rawArtifact.odd.poseidon2Calls.length = 544 /\
      rawArtifact.even.glueRows.length = 121 /\
      rawArtifact.odd.glueRows.length = 121 := by
  exact
    ⟨by
      simpa [rawArtifact] using
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCanonicalCallCertificate.evenArm_canonicalCalls_valid.1,
    by
      simpa [rawArtifact] using
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicCanonicalCallCertificate.oddArm_canonicalCalls_valid.1,
    by
      simpa [rawArtifact] using
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenPoseidon2CallCertificate.evenArm_poseidon2Calls_length,
    by
      simpa [rawArtifact] using
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddPoseidon2CallCertificate.oddArm_poseidon2Calls_length,
    by
      simpa [rawArtifact] using
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenGlueRowCertificate.evenArm_glueRows_length,
    by
      simpa [rawArtifact] using
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddGlueRowCertificate.oddArm_glueRows_length⟩

/-- The public-state artifact delegates one exact contiguous range per parity
arm to the phase-envelope artifact. -/
theorem exact_phase_envelope_ranges :
    rawArtifact.even.phaseEnvelopeRowStart = 626420 /\
      rawArtifact.even.phaseEnvelopeRowEnd = 1289391 /\
      rawArtifact.odd.phaseEnvelopeRowStart = 627620 /\
      rawArtifact.odd.phaseEnvelopeRowEnd = 1290591 := by
  exact ⟨rfl, rfl, rfl, rfl⟩

/-- The shared public prefix is the after full XOut digest, the before full
XOut digest, the before global cursor, and the after global cursor. -/
theorem exact_public_word_layout :
    rawArtifact.even.publicColumnCount = 641 /\
      rawArtifact.odd.publicColumnCount = 641 /\
      rawArtifact.even.publicWordCallIndices =
        [3, 4, 5, 6, 7, 8, 9, 10, 0, 1] /\
      rawArtifact.odd.publicWordCallIndices =
        [3, 4, 5, 6, 7, 8, 9, 10, 0, 1] /\
      rawArtifact.lowNormPublicColumns = 648 := by
  exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- Each serialized side has exactly the complete 1045-field `FamilyState`.
Its last source column is the family cursor carried by the source rows. -/
theorem exact_state_column_shape (kind : ArmKind) :
    (armFor kind).beforeStateColumns.length = 1045 /\
      (armFor kind).afterStateColumns.length = 1045 /\
      (armFor kind).beforeStateColumns.getD 1044 0 =
      (armFor kind).beforeFamilyCursorColumn /\
      (armFor kind).afterStateColumns.getD 1044 0 =
        (armFor kind).afterFamilyCursorColumn := by
  cases kind with
  | even =>
      exact
        ⟨Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate.evenArm_stateColumnLayout_valid.1.1,
          Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate.evenArm_stateColumnLayout_valid.2.1,
          Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate.evenArm_beforeState_last_is_cursor,
          Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate.evenArm_afterState_last_is_cursor⟩
  | odd =>
      exact
        ⟨Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate.oddArm_stateColumnLayout_valid.1.1,
          Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate.oddArm_stateColumnLayout_valid.2.1,
          Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate.oddArm_beforeState_last_is_cursor,
          Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicStateColumnLayoutCertificate.oddArm_afterState_last_is_cursor⟩

/-- Each side carries the complete 32-field full-state preimage and its
four-field Poseidon2 output. -/
theorem exact_x_out_column_shape (kind : ArmKind) :
    (armFor kind).afterXOutPreimageColumns.length = 32 /\
      (armFor kind).beforeXOutPreimageColumns.length = 32 /\
      (armFor kind).afterXOutDigestColumns.length = 4 /\
      (armFor kind).beforeXOutDigestColumns.length = 4 := by
  cases kind with
  | even =>
      have valid :=
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicSmallLayoutCertificate.evenArm_xOutColumnLayout_valid
      exact ⟨valid.1.1, valid.2.1.1, valid.2.2.1.1, valid.2.2.2.1⟩
  | odd =>
      have valid :=
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicSmallLayoutCertificate.oddArm_xOutColumnLayout_valid
      exact ⟨valid.1.1, valid.2.1.1, valid.2.2.1.1, valid.2.2.2.1⟩

/-- The generated coordinate chain covers every suffix row once. The
phase-envelope marker delegates its rows and does not prove them here. -/
theorem exact_suffix_owner_chain (kind : ArmKind) :
    exactOwnerChainFrom (armFor kind) (armFor kind).sourceRowCount
      (armFor kind).owners = true := by
  cases kind with
  | even =>
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenOwnershipTailCertificate.evenArm_ownership_valid.2.2.2.2
  | odd =>
      exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddOwnershipTailCertificate.oddArm_ownership_valid.2.2.2.2

/-- Every canonical-word leaf in a satisfying suffix has the unique
canonical 64-bit value specified by the Lean-owned recipe. -/
theorem canonical_call_refines
    (arm : RawArm) (assignment : Nat → Nat)
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (satisfied : arm.Satisfied assignment)
    (call :
      FPrimeFullHistoryStreamingClaimReplay.Artifact.CanonicalCall)
    (member : call ∈ arm.canonicalCalls) :
    Refines assignment call.layout := by
  apply CanonicalU64RecipeSound.sound goldilocks_euclidPrime canonical one
  exact satisfied.1 call member

/-- Every suffix Poseidon2 leaf agrees with the exact production permutation
under its Rust-emitted column renaming. -/
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

/-- The compact artifact stores each non-template suffix row exactly. -/
theorem glue_row_holds
    (arm : RawArm) (assignment : Nat → Nat)
    (satisfied : arm.Satisfied assignment)
    (indexed : FPrimeFullHistoryStreamingClaimReplay.Artifact.IndexedRow)
    (member : indexed ∈ arm.glueRows) :
    RowHolds assignment indexed.row := by
  exact satisfied.2.2 indexed member

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
