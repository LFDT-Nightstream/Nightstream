import Nightstream.Implementation.Nebula.FPrime.State.PriorLinkRows

/-! Focused gates for the exact incoming-state recursive link. -/

set_option autoImplicit false

namespace tests.NebulaPriorStateLinkRows

open Nightstream.Implementation.Nebula
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula

theorem selected_carrier_geometry :
    PriorStateLinkRows.ccsPublicBitCount = 540 ∧
      PriorStateLinkRows.digestBitCount = 256 ∧
      PriorStateLinkRows.memoryDigestBitCount = 256 ∧
      PriorStateLinkRows.paddingBitCount = 27 := by
  decide

theorem exact_row_count
    {widths : FullClaimEnvelope.CompilerWidths}
    {layout : PriorStateLinkRows.Layout widths}
    (valid : layout.Valid) :
    (PriorStateLinkRows.rows layout).length = 40090 :=
  PriorStateLinkRows.rows_length_exact valid

theorem satisfying_rows_derive_complete_public_carrier
    {widths : FullClaimEnvelope.CompilerWidths}
    {layout : PriorStateLinkRows.Layout widths}
    (valid : layout.Valid) {assignment : Nat → Nat}
    {claim : FullClaimEnvelope.Value widths}
    {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input)
    (memoryParsed : MemoryClaimRows.ParsedColumnsMatch
      layout.memoryDigest.frame.claim assignment claim.memory)
    (holds : Satisfies (PriorStateLinkRows.rows layout) assignment) :
    PriorStateLinkRows.CcsPublicExact valid assignment claim canonical :=
  PriorStateLinkRows.claimCcsPublicExact valid canonical one placed memoryParsed
    holds

theorem satisfying_rows_derive_one_exact_public_word
    {widths : FullClaimEnvelope.CompilerWidths}
    {layout : PriorStateLinkRows.Layout widths}
    (valid : layout.Valid) {assignment : Nat → Nat}
    {claim : FullClaimEnvelope.Value widths}
    {input : FixedBits.Word widths.totalBits}
    (canonical : ∀ column, assignment column < goldilocksP)
    (one : assignment 0 = 1)
    (placed : FullClaimEnvelopeRows.Placed layout.fullClaim assignment claim input)
    (memoryParsed : MemoryClaimRows.ParsedColumnsMatch
      layout.memoryDigest.frame.claim assignment claim.memory)
    (holds : Satisfies (PriorStateLinkRows.rows layout) assignment) :
    claim.ccsPublic = PriorStateLinkRows.CcsPublicExact.typedWord valid
      (PriorStateLinkRows.outputDigest layout assignment canonical) claim.memory :=
  PriorStateLinkRows.CcsPublicExact.ccsPublic_eq_ccsPublicWord
    (PriorStateLinkRows.claimCcsPublicExact valid canonical one placed
      memoryParsed holds)

end tests.NebulaPriorStateLinkRows
