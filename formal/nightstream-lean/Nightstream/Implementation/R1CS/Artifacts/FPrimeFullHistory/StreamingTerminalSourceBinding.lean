import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding

/-! Structural validation of the compact exact terminal source-binding artifact. -/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalSourceBinding

open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingTerminalSourceBinding.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingTerminalSourceBinding

theorem rawArtifact_valid : rawArtifact.Valid := by
  refine {
    schemaVersion := rfl
    profileId := rfl
    sourceArtifactIdentity := rfl
    finalArtifactIdentity := rfl
    lifecycleScope := rfl
    rowFamily := rfl
    rowCount := by decide
    decodedCount := by decide
    finalAssignmentWithin := by decide
    decodedWithin := by decide
    groupsValid := by decide
    groupsCanonical := by
      intro group member
      change group ∈ decoderGroups at member
      simp only [decoderGroups, List.mem_cons, List.not_mem_nil, or_false] at member
      rcases member with
        rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
        rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
      all_goals decide }

theorem rawArtifact_rows_length :
    rawArtifact.rows.length = rawArtifact.rowStop - rawArtifact.rowStart := by
  exact rawArtifact.rows_length.trans rawArtifact_valid.rowCount.symm

/-- The exact Rust-emitted decoder block that owns the delayed terminal
payload. This compact leaf certificate avoids reducing the 2,255 decoded
rows in downstream proofs. -/
def delayedPayloadBlock : DecoderBlock :=
  { owner := "delayed_payload"
    sourceFields := { start := 30400389, stop := 30402558 }
    decodedColumns := { start := 28041985, stop := 28044154 }
    finalColumns := { start := 22126657, stop := 22128826 }
    width := 1
    radix := 2
    scale := 1 }

def delayedOpenOffset : Nat := 1400

def delayedOpenSourceColumn : Nat := 22128057

/-- Exact Rust-emitted source decoders for the 50-field post-phase Nebula
lane. Their decoded ranges are consecutive and follow `Lane` field order. -/
def nebulaProgramDigestBlock : DecoderBlock :=
  { owner := "nebula_lane"
    sourceFields := { start := 672, stop := 676 }
    decodedColumns := { start := 28041931, stop := 28041935 }
    finalColumns := { start := 1815, stop := 1979 }
    width := 41
    radix := 3
    scale := 1 }

def nebulaOpenBlock : DecoderBlock :=
  { owner := "nebula_lane"
    sourceFields := { start := 19341627, stop := 19341628 }
    decodedColumns := { start := 28041935, stop := 28041936 }
    finalColumns := { start := 19949237, stop := 19949238 }
    width := 1
    radix := 2
    scale := 1 }

def nebulaCountersBlock : DecoderBlock :=
  { owner := "nebula_lane"
    sourceFields := { start := 19341628, stop := 19341630 }
    decodedColumns := { start := 28041936, stop := 28041938 }
    finalColumns := { start := 19949238, stop := 19949320 }
    width := 41
    radix := 3
    scale := 1 }

def nebulaTimestampBlock : DecoderBlock :=
  { owner := "nebula_lane"
    sourceFields := { start := 19231123, stop := 19231124 }
    decodedColumns := { start := 28041938, stop := 28041939 }
    finalColumns := { start := 19526378, stop := 19526422 }
    width := 44
    radix := 2
    scale := 1 }

def nebulaStateBlock : DecoderBlock :=
  { owner := "nebula_lane"
    sourceFields := { start := 19341630, stop := 19341672 }
    decodedColumns := { start := 28041939, stop := 28041981 }
    finalColumns := { start := 19949320, stop := 19951042 }
    width := 41
    radix := 3
    scale := 1 }

theorem delayedPayloadBlock_member :
    DecoderGroup.block delayedPayloadBlock ∈ rawArtifact.decoderGroups := by
  simp [rawArtifact, decoderGroups, delayedPayloadBlock]

theorem nebulaProgramDigestBlock_member :
    DecoderGroup.block nebulaProgramDigestBlock ∈ rawArtifact.decoderGroups := by
  simp [rawArtifact, decoderGroups, nebulaProgramDigestBlock]

theorem nebulaOpenBlock_member :
    DecoderGroup.block nebulaOpenBlock ∈ rawArtifact.decoderGroups := by
  simp [rawArtifact, decoderGroups, nebulaOpenBlock]

theorem nebulaCountersBlock_member :
    DecoderGroup.block nebulaCountersBlock ∈ rawArtifact.decoderGroups := by
  simp [rawArtifact, decoderGroups, nebulaCountersBlock]

theorem nebulaTimestampBlock_member :
    DecoderGroup.block nebulaTimestampBlock ∈ rawArtifact.decoderGroups := by
  simp [rawArtifact, decoderGroups, nebulaTimestampBlock]

theorem nebulaStateBlock_member :
    DecoderGroup.block nebulaStateBlock ∈ rawArtifact.decoderGroups := by
  simp [rawArtifact, decoderGroups, nebulaStateBlock]

theorem delayedOpenOffset_bound :
    delayedOpenOffset < delayedPayloadBlock.count := by
  norm_num [delayedOpenOffset, delayedPayloadBlock, DecoderBlock.count,
    Range.length]

theorem delayedOpenDecodedColumn :
    delayedPayloadBlock.decodedColumns.start + delayedOpenOffset = 28043385 := by
  norm_num [delayedPayloadBlock, delayedOpenOffset]

theorem delayedOpenTerms :
    delayedPayloadBlock.termsAt delayedOpenOffset =
      [(delayedOpenSourceColumn, 1)] := by
  norm_num [DecoderBlock.termsAt, decoderTerms, delayedPayloadBlock,
    delayedOpenOffset, delayedOpenSourceColumn, goldilocksP]

theorem delayedPayloadDecodedColumn (index : Nat) :
    delayedPayloadBlock.decodedColumns.start + index = 28041985 + index := by
  rfl

theorem delayedPayloadTermsAt (index : Nat) :
    delayedPayloadBlock.termsAt index = [(22126657 + index, 1)] := by
  simp [DecoderBlock.termsAt, decoderTerms, delayedPayloadBlock, goldilocksP]

end Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingTerminalSourceBinding
