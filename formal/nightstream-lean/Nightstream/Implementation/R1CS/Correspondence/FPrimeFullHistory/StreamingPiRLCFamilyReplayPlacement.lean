import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder
import Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger
import Nightstream.Implementation.R1CS.Correspondence.FPrimeFullHistory.StreamingPiRLCFamilyReplayArtifact

/-!
Contract: exact placement of the normalized production PiRLC replay calls.

Assurance tier: Rust-conformant artifact composition.

Owns the structural join between each isolated replay call chain, the first
Poseidon2 rewrite batch for its parity body, and the matching 600-column
source-to-final decoder-template batch.

Does not own Poseidon2 leaf semantics, final-row satisfaction, selector
authority, family-phase semantics, recursive orchestration, or cryptographic
security.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayPlacement

open Nightstream.Implementation.R1CS
open Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplay.Artifact
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoderSchema
open Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedgerSchema

private abbrev replayEven :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyReplay.evenArm

private abbrev replayOdd :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyReplay.oddArm

private abbrev bodyLedger :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyRowLedger.ledger

private abbrev bodyDecoderEven :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.evenArm

private abbrev bodyDecoderOdd :=
  Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.StreamingPiRLCFamilyBodyDecoder.oddArm

/-- Structural meaning of the compact replay chain. It records only the two
affine coordinates that the normalized placement join needs. -/
def CallsPlacedFrom : Nat → Nat → List Poseidon2Call.Call → Prop
  | _, _, [] => True
  | row, column, call :: rest =>
      call.rowStart = row ∧
        call.rowEnd = row + 600 ∧
        call.firstAllocatedColumn = column ∧
        CallsPlacedFrom (row + 600) (column + 600) rest

/-- The Boolean chain checker implies its structural meaning without reducing
any generated call list. -/
theorem exactCallChainFrom_sound
    (row column : Nat) (calls : List Poseidon2Call.Call)
    (checked : exactCallChainFrom row column calls = true) :
    CallsPlacedFrom row column calls := by
  induction calls generalizing row column with
  | nil => trivial
  | cons call rest inductionHypothesis =>
      simp only [exactCallChainFrom, Bool.and_eq_true, beq_iff_eq] at checked
      exact ⟨checked.1.1.1, checked.1.1.2, checked.1.2,
        inductionHypothesis (row + 600) (column + 600) checked.2⟩

/-- Structural indexing of a checked affine call chain. This proof does not
reduce a generated call list. -/
theorem callsPlacedFrom_get
    {calls : List Poseidon2Call.Call} {call : Poseidon2Call.Call}
    {row column index : Nat}
    (placed : CallsPlacedFrom row column calls)
    (selected : calls[index]? = some call) :
    call.rowStart = row + index * 600 ∧
      call.rowEnd = row + index * 600 + 600 ∧
      call.firstAllocatedColumn = column + index * 600 := by
  induction calls generalizing row column index with
  | nil => simp at selected
  | cons head tail inductionHypothesis =>
      simp only [CallsPlacedFrom] at placed
      rcases placed with ⟨headStart, headEnd, headColumn, tailPlaced⟩
      cases index with
      | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at selected
          subst call
          omega
      | succ index =>
          simp only [List.getElem?_cons_succ] at selected
          have tailCoordinates := inductionHypothesis
            tailPlaced selected
          omega

private theorem arm_valid_implies_placement
    (arm :
      Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplay.Artifact.RawArm)
    (beforeAbsorbed afterAbsorbed inputCalls outputCalls : Nat)
    (valid : arm.Valid beforeAbsorbed afterAbsorbed inputCalls outputCalls) :
    arm.poseidon2Calls.length = inputCalls + outputCalls ∧
      CallsPlacedFrom 0 165680 arm.poseidon2Calls := by
  rcases valid with
    ⟨_, _, _, _, _, _, length, _, _, _, _, _, _, _, chain⟩
  exact ⟨length, exactCallChainFrom_sound 0 165680 _ chain.1⟩

/-- First normalized rewrite batch owned by the even replay. -/
def evenReplayRewriteBatch : RawRewriteBatch where
  rewriteStart := 0
  count := 242
  rewriteStride := 1
  arm := 0
  kind := .poseidon2
  sourceStart := 165446
  sourceStride := 600
  sourceWidth := 600
  emittedStart := 74375
  emittedStride := 86
  emittedWidth := 86

/-- First normalized rewrite batch owned by the odd replay. -/
def oddReplayRewriteBatch : RawRewriteBatch where
  rewriteStart := 7318
  count := 244
  rewriteStride := 1
  arm := 1
  kind := .poseidon2
  sourceStart := 165446
  sourceStride := 600
  sourceWidth := 600
  emittedStart := 309886
  emittedStride := 86
  emittedWidth := 86

/-- Source-to-final decoder batch for all even replay-local 600-column
Poseidon2 allocations. -/
def evenReplayDecoderBatch : RawTemplateInstances where
  sourceStart := 166320
  count := 242
  sourceStride := 600
  finalStart := 2218425
  finalStride := 3526
  referenceStart := 0
  referenceStride := 0
  referenceFinalStart := 0
  referenceFinalStride := 0

/-- Source-to-final decoder batch for all odd replay-local 600-column
Poseidon2 allocations. -/
def oddReplayDecoderBatch : RawTemplateInstances where
  sourceStart := 166320
  count := 244
  sourceStride := 600
  finalStart := 2218425
  finalStride := 3526
  referenceStart := 0
  referenceStride := 0
  referenceFinalStart := 0
  referenceFinalStride := 0

/-- Exact Rust-emitted even replay rewrite leaf. This unfolds only the first
ledger entry. -/
theorem evenReplayRewriteBatch_exact :
    bodyLedger.rewriteBatches[0]? = some evenReplayRewriteBatch := by
  rfl

/-- Exact Rust-emitted odd replay rewrite leaf. This unfolds only the bounded
ledger prefix through its first odd-arm entry. -/
theorem oddReplayRewriteBatch_exact :
    bodyLedger.rewriteBatches[20]? = some oddReplayRewriteBatch := by
  rfl

/-- Exact Rust-emitted even decoder leaf. Projection keeps the shared rule
list opaque and checks only its width and first affine instance batch. -/
theorem evenReplayDecoderBatch_exact :
    (bodyDecoderEven.templates[2]?).map (fun template =>
      (template.sourceWidth, template.instances[0]?)) =
      some (600, some evenReplayDecoderBatch) := by
  rfl

/-- Exact Rust-emitted odd decoder leaf. -/
theorem oddReplayDecoderBatch_exact :
    (bodyDecoderOdd.templates[2]?).map (fun template =>
      (template.sourceWidth, template.instances[0]?)) =
      some (600, some oddReplayDecoderBatch) := by
  rfl

/-- Compact placement join. Input columns remain the exact call-owned list;
adding `sourceColumnOffset` gives their coordinates in the body source arm.
The rewrite and decoder records give the affine source-row, emitted-row, and
final-slot coordinates for the same call index. -/
structure Placement where
  calls : List Poseidon2Call.Call
  rewrite : RawRewriteBatch
  decoder : RawTemplateInstances
  sourceColumnOffset : Nat

structure Placement.Valid (placement : Placement) : Prop where
  callCount : placement.rewrite.count = placement.calls.length
  decoderCount : placement.decoder.count = placement.calls.length
  poseidonRewrite : placement.rewrite.kind = .poseidon2
  sourceRows : placement.rewrite.sourceStart = 165446
  sourceRowStride : placement.rewrite.sourceStride = 600
  sourceRowWidth : placement.rewrite.sourceWidth = 600
  emittedRowStride : placement.rewrite.emittedStride = 86
  emittedRowWidth : placement.rewrite.emittedWidth = 86
  sourceColumns : placement.decoder.sourceStart = 165680 + placement.sourceColumnOffset
  sourceColumnStride : placement.decoder.sourceStride = 600
  finalSlots : placement.decoder.finalStart = 2218425
  finalSlotStride : placement.decoder.finalStride = 86 * 41
  callsPlaced : CallsPlacedFrom 0 165680 placement.calls

/-- Exact source and final ownership for one same-index replay call. -/
structure Placement.IndexedOwnership
    (placement : Placement) (index : Nat)
    (call : Poseidon2Call.Call) : Prop where
  callRowStart : call.rowStart = index * 600
  callRowEnd : call.rowEnd = index * 600 + 600
  callColumnStart : call.firstAllocatedColumn = 165680 + index * 600
  rewriteSourceStart :
    placement.rewrite.sourceStart + index * placement.rewrite.sourceStride =
      165446 + index * 600
  rewriteEmittedStart :
    placement.rewrite.emittedStart + index * placement.rewrite.emittedStride =
      placement.rewrite.emittedStart + index * 86
  decoderSourceStart :
    placement.decoder.sourceStart + index * placement.decoder.sourceStride =
      165680 + placement.sourceColumnOffset + index * 600
  decoderFinalStart :
    placement.decoder.finalStart + index * placement.decoder.finalStride =
      2218425 + index * (86 * 41)

theorem Placement.Valid.indexedOwnership
    {placement : Placement} (valid : placement.Valid)
    {index : Nat} {call : Poseidon2Call.Call}
    (selected : placement.calls[index]? = some call) :
    placement.IndexedOwnership index call := by
  have callCoordinates := callsPlacedFrom_get valid.callsPlaced selected
  exact {
    callRowStart := by omega
    callRowEnd := by omega
    callColumnStart := by omega
    rewriteSourceStart := by
      rw [valid.sourceRows, valid.sourceRowStride]
    rewriteEmittedStart := by rw [valid.emittedRowStride]
    decoderSourceStart := by
      rw [valid.sourceColumns, valid.sourceColumnStride]
    decoderFinalStart := by
      rw [valid.finalSlots, valid.finalSlotStride] }

def evenPlacement : Placement where
  calls := replayEven.poseidon2Calls
  rewrite := evenReplayRewriteBatch
  decoder := evenReplayDecoderBatch
  sourceColumnOffset := 640

def oddPlacement : Placement where
  calls := replayOdd.poseidon2Calls
  rewrite := oddReplayRewriteBatch
  decoder := oddReplayDecoderBatch
  sourceColumnOffset := 640

/-- The exact even replay call chain, normalized rewrite batch, and decoder
batch are one same-index placement. -/
theorem evenPlacement_valid : evenPlacement.Valid := by
  have facts := arm_valid_implies_placement replayEven 0 2 229 13
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay.evenArm_valid
  refine {
    callCount := ?_
    decoderCount := ?_
    poseidonRewrite := rfl
    sourceRows := rfl
    sourceRowStride := rfl
    sourceRowWidth := rfl
    emittedRowStride := rfl
    emittedRowWidth := rfl
    sourceColumns := rfl
    sourceColumnStride := rfl
    finalSlots := rfl
    finalSlotStride := rfl
    callsPlaced := facts.2 }
  · change 242 = replayEven.poseidon2Calls.length
    exact
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay.evenArm_poseidon2Calls_length.symm
  · change 242 = replayEven.poseidon2Calls.length
    exact
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay.evenArm_poseidon2Calls_length.symm

/-- The exact odd replay call chain, normalized rewrite batch, and decoder
batch are one same-index placement. -/
theorem oddPlacement_valid : oddPlacement.Valid := by
  have facts := arm_valid_implies_placement replayOdd 2 0 230 14
    Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay.oddArm_valid
  refine {
    callCount := ?_
    decoderCount := ?_
    poseidonRewrite := rfl
    sourceRows := rfl
    sourceRowStride := rfl
    sourceRowWidth := rfl
    emittedRowStride := rfl
    emittedRowWidth := rfl
    sourceColumns := rfl
    sourceColumnStride := rfl
    finalSlots := rfl
    finalSlotStride := rfl
    callsPlaced := facts.2 }
  · change 244 = replayOdd.poseidon2Calls.length
    exact
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay.oddArm_poseidon2Calls_length.symm
  · change 244 = replayOdd.poseidon2Calls.length
    exact
      Nightstream.Implementation.R1CS.Artifacts.FPrimeFullHistory.Generated.FPrimeFullHistoryStreamingPiRLCFamilyReplay.oddArm_poseidon2Calls_length.symm

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingPiRLCFamilyReplayPlacement
