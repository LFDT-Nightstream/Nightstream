import Nightstream.Implementation.NebulaV2.NIFS.Running.FullClaimDecoder

/-!
Contract: countermodel for the retired V2 paper-input projection that omitted
the memory suffix.

The legacy projection kept the running claim, the product commitment bundle,
and the 270-coordinate state carrier. It discarded the 4,980-bit memory
suffix before it called the paper NIFS verifier. This file proves that two
different canonical full envelopes can therefore select the same paper input.

This is negative evidence. No release relation may use `legacyPaperInput`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.LegacyMemorySuffixAlias

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.ProductFullClaimDecoder
open Nightstream.SuperNeo.Concrete.Phi81Relation

/-- Replace only the memory suffix of a complete envelope. -/
def withMemory (value : Value widths) (memory : MemoryClaimCodec.Claim) :
    Value widths :=
  { value with memory := memory }

/-- The exact paper input selected by the retired decoder. The definition
shows the defect: neither component reads `value.memory`. -/
noncomputable def legacyPaperInput
    {fullShape : Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (value : Value widths) :
    ProductFullClaimDecoder.Running fullShape ×
      ProductFullClaimDecoder.Fresh fullShape :=
  (runningOfValue contract value, freshOfValue contract value)

theorem legacyPaperInput_ignores_memory
    {fullShape : Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (value : Value widths)
    (left right : MemoryClaimCodec.Claim) :
    legacyPaperInput contract (withMemory value left) =
      legacyPaperInput contract (withMemory value right) := by
  rfl

theorem withMemory_canonical
    {value : Value widths} (canonical : value.Canonical)
    {memory : MemoryClaimCodec.Claim} (memoryCanonical : memory.Canonical) :
    (withMemory value memory).Canonical where
  profileExact := canonical.profileExact
  memoryCanonical := memoryCanonical

/-- Two unequal canonical memory suffixes give different complete claim
blocks but the same paper NIFS input. This is the deterministic alias that
the memory-digest carrier must remove. -/
theorem distinct_blocks_same_legacy_paper_input
    {fullShape : Shape}
    (contract : ProductNifsCodec.FullShapeContract fullShape)
    (value : Value widths) (canonical : value.Canonical)
    (left right : MemoryClaimCodec.Claim)
    (leftCanonical : left.Canonical)
    (rightCanonical : right.Canonical)
    (different : left ≠ right) :
    (withMemory value left).block ≠ (withMemory value right).block ∧
      legacyPaperInput contract (withMemory value left) =
        legacyPaperInput contract (withMemory value right) := by
  constructor
  · intro blocksEqual
    have encodingsEqual :
        (withMemory value left).encode = (withMemory value right).encode :=
      congrArg Subtype.val blocksEqual
    have valuesEqual := Value.encode_injective_on_canonical
      (withMemory_canonical canonical leftCanonical)
      (withMemory_canonical canonical rightCanonical) encodingsEqual
    exact different (congrArg Value.memory valuesEqual)
  · exact legacyPaperInput_ignores_memory contract value left right

end Nightstream.Implementation.NebulaV2.LegacyMemorySuffixAlias
