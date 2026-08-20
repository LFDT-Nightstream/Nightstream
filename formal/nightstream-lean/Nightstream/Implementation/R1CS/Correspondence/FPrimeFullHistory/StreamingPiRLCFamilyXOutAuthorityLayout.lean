import Nightstream.Implementation.Nebula.FPrime.State.OutputAuthorityRows
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyXOutPreimage

/-!
Contract: exact adapter from one PiRLC family XOut preimage to the shared
32-field state-output authority layout.

Owns only column roles. It reuses the existing cursor and program-counter
bit witnesses and emits no rows or columns. It does not give authority to a
payload value or a Nebula digest.

Assurance tier: artifact-checked column layout for property
`FPRIME-STREAMING-PIRLC-FAMILY-XOUT-PREIMAGE-V1`, Nightstream b2/k16.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutAuthorityLayout

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.Canonical
open Nightstream.Implementation.R1CS.Canonical.CanonicalU64Recipe
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplay.Artifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyDigest
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicState
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutArtifact
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutPreimage
open Nightstream.Implementation.R1CS.Program

/-- Exact physical columns for the fixed 32-field XOut frame. -/
def frameLayout (kind : ArmKind) (side : StateSide) :
    StateOutputFrameRows.Layout where
  domainColumn := xOutPreimageColumn kind side 0
  vkFsDigestColumn := fun lane =>
    xOutPreimageColumn kind side (1 + lane.val)
  piCcsHeaderColumn := fun lane =>
    xOutPreimageColumn kind side (5 + lane.val)
  chunkCountHalfColumn := fun half =>
    xOutPreimageColumn kind side (9 + half.val)
  stepCountHalfColumn := fun half =>
    xOutPreimageColumn kind side (11 + half.val)
  pcHalfColumn := fun half =>
    xOutPreimageColumn kind side (13 + half.val)
  currentBoundaryColumn := fun lane =>
    xOutPreimageColumn kind side (15 + lane.val)
  semanticStateColumn := fun lane =>
    xOutPreimageColumn kind side (19 + lane.val)
  accumulatorDigestColumn := fun lane =>
    xOutPreimageColumn kind side (23 + lane.val)
  nebulaMarkerColumn := xOutPreimageColumn kind side 27
  nebulaDigestColumn := fun lane =>
    xOutPreimageColumn kind side (28 + lane.val)
  carryDigestOutputColumn := fun lane =>
    xOutPreimageColumn kind side (28 + lane.val)

/-- Shared authority layout on the exact PiRLC columns. The two cursor words
reuse one canonical-u64 bit decomposition because the physical rows require
their low and high halves to be equal. -/
def authorityLayout (kind : ArmKind) (side : StateSide) :
    StateOutputAuthorityRows.Layout where
  frame := frameLayout kind side
  vkFsDigestColumn := fun lane =>
    xOutPreimageColumn kind side (1 + lane.val)
  piCcsHeaderColumn := fun lane =>
    xOutPreimageColumn kind side (5 + lane.val)
  chunkCount := {
    lowColumn := xOutPreimageColumn kind side 9
    highColumn := xOutPreimageColumn kind side 10
    bitStart := (cursorCall kind side).bitBase }
  stepCount := {
    lowColumn := xOutPreimageColumn kind side 11
    highColumn := xOutPreimageColumn kind side 12
    bitStart := (cursorCall kind side).bitBase }
  pc := {
    lowColumn := xOutPreimageColumn kind side 13
    highColumn := xOutPreimageColumn kind side 14
    bitStart := (pcCall kind).bitBase }
  currentBoundaryColumn := fun lane =>
    xOutPreimageColumn kind side (15 + lane.val)
  semanticStateColumn := fun lane =>
    xOutPreimageColumn kind side (19 + lane.val)
  accumulatorDigestColumn := fun lane =>
    xOutPreimageColumn kind side (23 + lane.val)

/-- The adapter preserves every exact field role of the shared authority
layout. This proof reduces only the small adapter record. -/
theorem authorityLayout_valid (kind : ArmKind) (side : StateSide) :
    (authorityLayout kind side).Valid := by
  exact {
    exactVkFsDigestColumns := rfl
    exactPiCcsHeaderColumns := rfl
    exactChunkCountColumns := rfl
    exactStepCountColumns := rfl
    exactPcColumns := rfl
    exactCurrentBoundaryColumns := rfl
    exactSemanticStateColumns := rfl
    exactAccumulatorDigestColumns := rfl }

/-- The exact Rust hash input is the same ordered 32-column message as the
shared state-output frame layout. The hash-layout leaf certificates own the
generated-list equality; this adapter owns only the fixed field roles. -/
theorem hash_input_columns_exact (kind : ArmKind) (side : StateSide) :
    (xOutHashFor kind side).inputColumns =
      StateOutputFrameRows.inputColumns (frameLayout kind side) := by
  have generatedExact :
      (xOutHashFor kind side).inputColumns =
        match side with
        | .after => (armFor kind).afterXOutPreimageColumns
        | .before => (armFor kind).beforeXOutPreimageColumns := by
    cases kind <;> cases side
    · exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenHashLayoutCertificate.evenArm_hashLayout_valid.1.2.2.2.1
    · exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicEvenHashLayoutCertificate.evenArm_hashLayout_valid.2.2.2.2.1
    · exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddHashLayoutCertificate.oddArm_hashLayout_valid.1.2.2.2.1
    · exact
        Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyPublicOddHashLayoutCertificate.oddArm_hashLayout_valid.2.2.2.2.1
  rw [generatedExact]
  cases kind <;> cases side <;> rfl

private theorem halves_rows_satisfied
    (assignment : Nat → Nat)
    (call : CanonicalCall)
    (lowColumn highColumn : Nat)
    (canonicalRows :
      Satisfies (CanonicalU64Recipe.rows call.layout) assignment)
    (lowHolds : RowHolds assignment
      (builderLinearRow lowColumn (lowTerms call.layout)))
    (highHolds : RowHolds assignment
      (builderLinearRow highColumn (highTerms call.layout))) :
    Satisfies (U64HalvesRows.rows {
      lowColumn := lowColumn
      highColumn := highColumn
      bitStart := call.bitBase }) assignment := by
  intro row member
  rw [U64HalvesRows.rows, List.mem_append] at member
  rcases member with lowMember | highMember
  · rw [BoundedWordRows.rows, List.mem_append] at lowMember
    rcases lowMember with bitMember | recompositionMember
    · apply canonicalRows row
      rw [CanonicalU64Recipe.rows, List.mem_append]
      apply Or.inl
      rcases List.mem_map.mp bitMember with ⟨offset, offsetMember, rfl⟩
      refine List.mem_map.mpr ⟨offset, ?_, ?_⟩
      · have offsetBound : offset < 32 := by
          simpa [U64HalvesRows.Layout.lowWord] using
            (List.mem_range.mp offsetMember)
        exact List.mem_range.mpr (by omega)
      · simp [U64HalvesRows.Layout.lowWord,
          BoundedWordRows.Layout.bitColumn,
          CanonicalU64Recipe.bitColumn, CanonicalCall.layout]
    · simp only [List.mem_singleton] at recompositionMember
      subst row
      simpa [U64HalvesRows.Layout.lowWord,
        BoundedWordRows.Layout.recompositionRow,
        BoundedWordRows.Layout.terms, CanonicalU64Recipe.lowTerms,
        CanonicalU64Recipe.bitColumn, CanonicalCall.layout] using lowHolds
  · rw [BoundedWordRows.rows, List.mem_append] at highMember
    rcases highMember with bitMember | recompositionMember
    · apply canonicalRows row
      rw [CanonicalU64Recipe.rows, List.mem_append]
      apply Or.inl
      rcases List.mem_map.mp bitMember with ⟨offset, offsetMember, rfl⟩
      refine List.mem_map.mpr ⟨32 + offset, ?_, ?_⟩
      · apply List.mem_range.mpr
        have offsetBound : offset < 32 := by
          simpa [U64HalvesRows.Layout.highWord] using
            (List.mem_range.mp offsetMember)
        omega
      · simp [U64HalvesRows.Layout.highWord,
          BoundedWordRows.Layout.bitColumn,
          CanonicalU64Recipe.bitColumn, CanonicalCall.layout,
          Nat.add_assoc]
    · simp only [List.mem_singleton] at recompositionMember
      subst row
      simpa [U64HalvesRows.Layout.highWord,
        BoundedWordRows.Layout.recompositionRow,
        BoundedWordRows.Layout.terms, CanonicalU64Recipe.highTerms,
        CanonicalU64Recipe.bitColumn, CanonicalCall.layout,
        Nat.add_assoc] using highHolds

/-- One accepted PiRLC suffix satisfies all 198 shared authority rows by
reusing its canonical-u64 and certified structural rows. -/
theorem authority_rows_satisfied
    (kind : ArmKind) (side : StateSide)
    (assignment : Nat → Nat)
    (satisfied : (armFor kind).Satisfied assignment) :
    Satisfies (StateOutputAuthorityRows.rows (authorityLayout kind side))
      assignment := by
  have cursorRows := cursor_call_rows_satisfied kind side assignment satisfied
  have pcRows := pc_call_rows_satisfied kind assignment satisfied
  have cursorLow := builder_structural_row_holds kind side assignment satisfied
    (xOutPreimageColumn kind side 9) (lowTerms (cursorCall kind side).layout)
    (by simp [rawStructuralRows])
  have cursorHigh := builder_structural_row_holds kind side assignment satisfied
    (xOutPreimageColumn kind side 10) (highTerms (cursorCall kind side).layout)
    (by simp [rawStructuralRows])
  have stepLow := builder_structural_row_holds kind side assignment satisfied
    (xOutPreimageColumn kind side 11) (lowTerms (cursorCall kind side).layout)
    (by simp [rawStructuralRows])
  have stepHigh := builder_structural_row_holds kind side assignment satisfied
    (xOutPreimageColumn kind side 12) (highTerms (cursorCall kind side).layout)
    (by simp [rawStructuralRows])
  have pcLow := builder_structural_row_holds kind side assignment satisfied
    (xOutPreimageColumn kind side 13) (lowTerms (pcCall kind).layout)
    (by simp [rawStructuralRows])
  have pcHigh := builder_structural_row_holds kind side assignment satisfied
    (xOutPreimageColumn kind side 14) (highTerms (pcCall kind).layout)
    (by simp [rawStructuralRows])
  rw [StateOutputAuthorityRows.rows]
  intro row member
  rw [List.mem_append] at member
  rcases member with chunkOrStepMember | pcMember
  · rw [List.mem_append] at chunkOrStepMember
    rcases chunkOrStepMember with chunkMember | stepMember
    · exact halves_rows_satisfied assignment (cursorCall kind side)
        _ _ cursorRows cursorLow cursorHigh row chunkMember
    · exact halves_rows_satisfied assignment (cursorCall kind side)
        _ _ cursorRows stepLow stepHigh row stepMember
  · exact halves_rows_satisfied assignment (pcCall kind)
      _ _ pcRows pcLow pcHigh row pcMember

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyXOutAuthorityLayout
