import Nightstream.Implementation.Nebula.Production.Carrier.StreamingFPrimeProgram
import Nightstream.Implementation.Nebula.Production.Carrier.StreamingPiCcsCoordinateBindingClaimSchedule

/-!
Contract: exact physical overlay schedule for the three-map PiCCS claim binding.

Assurance tier: model-level schedule contract.

Owns the 99 overlay kinds, the exact kind selected by each claim work item,
and the 98 compact source-link runs. The runs cover every authoritative
claim-frame field from offset 383 through 99,902 and both carried
states of all three 108-coordinate commitments.

Does not own coordinate-map row soundness, Rust artifact conformance,
accumulator terminal equality, lifecycle semantics, or security reduction.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayOverlaySchedule

open Nightstream.Implementation.Nebula.ProductionStreamingFPrimeProgram
open Nightstream.Implementation.Nebula.ProductionStreamingPiCcsCoordinateBindingClaimSchedule

/-- Kind zero is the no-op. Each of the 98 claim chunks owns one distinct
nonzero physical overlay kind. -/
def overlayKindCodeAt (chunk : Fin claimChunkCount) : Nat :=
  chunk.val + 1

theorem overlayKindCodeAt_lt (chunk : Fin claimChunkCount) :
    overlayKindCodeAt chunk < 99 := by
  have bound := chunk.isLt
  unfold claimChunkCount at bound
  simp [overlayKindCodeAt]
  omega

def overlayKindAt (chunk : Fin claimChunkCount) : Fin 99 :=
  ⟨overlayKindCodeAt chunk, overlayKindCodeAt_lt chunk⟩

/-- Invalid claim indices map to no-op. The production program has no such
index. -/
def overlayKindForWorkItem (item : WorkItem) : Fin 99 :=
  if item.phase = .claimReplay then
    if bound : item.index < claimChunkCount then
      overlayKindAt ⟨item.index, bound⟩
    else
      0
  else
    0

def productionOverlayKindMap : List Nat :=
  (program productionConfig).map fun item =>
    (overlayKindForWorkItem item).val

/-- Exact compact source-field link run for one claim chunk. Each tuple is
`(overlay kind, phase kind, chunk index, active offset, active count)`. -/
def productionOverlayLinkRunAt
    (chunk : Fin claimChunkCount) : Nat × Nat × Nat × Nat × Nat :=
  ( chunk.val + 1
  , if chunk.val = 97 then 4 else 3
  , chunk.val
  , if chunk.val = 0 then 383 else 0
  , if chunk.val = 0 then 641 else if chunk.val = 97 then 575 else 1024 )

def productionOverlayLinkRuns : List (Nat × Nat × Nat × Nat × Nat) :=
  List.ofFn productionOverlayLinkRunAt

def productionOverlayLinkRunChunk :
    List (Nat × Nat × Nat × Nat × Nat) :=
  productionOverlayLinkRuns.take 64

def productionOverlayLinkRunTail :
    List (Nat × Nat × Nat × Nat × Nat) :=
  productionOverlayLinkRuns.drop 64

private theorem length_of_take_drop
    {α : Type} {items : List α} {count headLength tailLength : Nat}
    (head : (items.take count).length = headLength)
    (tail : (items.drop count).length = tailLength) :
    items.length = headLength + tailLength := by
  have split := congrArg List.length (List.take_append_drop count items)
  simpa only [List.length_append, head, tail] using split.symm

private theorem mapSum_of_take_drop
    {α : Type} (items : List α) (value : α → Nat)
    (count headSum tailSum : Nat)
    (head : ((items.take count).map value).sum = headSum)
    (tail : ((items.drop count).map value).sum = tailSum) :
    (items.map value).sum = headSum + tailSum := by
  rw [← List.take_append_drop count items, List.map_append,
    List.sum_append, head, tail]

theorem productionOverlayLinkRunChunk_checked :
    productionOverlayLinkRunChunk.length = 64 ∧
      (productionOverlayLinkRunChunk.map fun run => run.2.2.2.2).sum =
        65_153 ∧
      (productionOverlayLinkRunChunk.map fun run => 648 + run.2.2.2.2).sum =
        106_625 := by
  exact ⟨rfl, rfl, rfl⟩

theorem productionOverlayLinkRunTail_checked :
    productionOverlayLinkRunTail.length = 34 ∧
      (productionOverlayLinkRunTail.map fun run => run.2.2.2.2).sum =
        34_367 ∧
      (productionOverlayLinkRunTail.map fun run => 648 + run.2.2.2.2).sum =
        56_399 := by
  exact ⟨rfl, rfl, rfl⟩

@[simp] theorem productionOverlayKindMap_length :
    productionOverlayKindMap.length = 436 := by
  unfold productionOverlayKindMap
  rw [List.length_map, production_program_length]

@[simp] theorem productionOverlayLinkRuns_length :
    productionOverlayLinkRuns.length = 98 := by
  exact length_of_take_drop
    productionOverlayLinkRunChunk_checked.1
    productionOverlayLinkRunTail_checked.1

theorem productionOverlayLinkRuns_census :
      (productionOverlayLinkRuns.map fun run => run.2.2.2.2).sum = 99_520 /\
      (productionOverlayLinkRuns.map fun run => 648 + run.2.2.2.2).sum =
        163_024 := by
  constructor
  · have total := mapSum_of_take_drop productionOverlayLinkRuns
      (fun run => run.2.2.2.2) 64 65_153 34_367
      productionOverlayLinkRunChunk_checked.2.1
      productionOverlayLinkRunTail_checked.2.1
    norm_num at total
    exact total
  · have total := mapSum_of_take_drop productionOverlayLinkRuns
      (fun run => 648 + run.2.2.2.2) 64 106_625 56_399
      productionOverlayLinkRunChunk_checked.2.2
      productionOverlayLinkRunTail_checked.2.2
    norm_num at total
    exact total

@[simp] theorem overlayKindAt_zero :
    (overlayKindAt ⟨0, by decide⟩).val = 1 := by
  rfl

@[simp] theorem overlayKindAt_final :
    (overlayKindAt ⟨97, by decide⟩).val = 98 := by
  rfl

end Nightstream.Implementation.R1CS.FPrimeFullHistoryStreamingClaimReplayOverlaySchedule
