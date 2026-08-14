import Mathlib.Algebra.Field.TransferInstance
import Nightstream.Implementation.Nebula.FPrime.Claim.NifsReceipt
import Nightstream.Protocol.Nebula.GlobalFPrime
import Nightstream.Implementation.Nebula.Core.ConcreteField

/-!
Contract: lifetime F-prime chain over exact selected-verifier V2 receipts.

Assurance tier: implementation model.

Owns the specialization of the global delayed schedule to full-envelope NIFS
receipts, exact claim count, exact acceptance of every complete input block,
and exact trailing-receipt ownership.

Does not own generated recursive rows, placeholder/swap refinement, NIFS
soundness, recursive-size closure, or terminal backend verification.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.FullClaimGlobalFPrime

open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.SuperNeo.Concrete

/-- The executable coefficient pair is transported through the proved exact
equivalence to the mathematical Goldilocks quadratic field. -/
noncomputable local instance concreteKField : Field K :=
  Nightstream.Implementation.Nebula.ConcreteField.superNeoEquiv.field

def verifiedReceipts {widths : CompilerWidths}
    {selected : SelectedVerifier widths}
    (receipts : List (Receipt selected)) :
    List (FullClaim.Verified
      (protocolSchema widths (PackedProof selected)) Digest.Value
      (Challenges K) (State K) (VerifyClaim selected)) :=
  receipts.map Receipt.toVerified

theorem verifiedReceipts_length {widths : CompilerWidths}
    {selected : SelectedVerifier widths}
    (receipts : List (Receipt selected)) :
    (verifiedReceipts receipts).length = receipts.length := by
  simp [verifiedReceipts]

theorem verifiedReceipts_get {widths : CompilerWidths}
    {selected : SelectedVerifier widths}
    (receipts : List (Receipt selected)) (index : Fin receipts.length) :
    (verifiedReceipts receipts).get
        ⟨index.val, by simpa [verifiedReceipts] using index.isLt⟩ =
      (receipts.get index).toVerified := by
  simp [verifiedReceipts]

/-- One exact global chain. `receipts` is the sole lifetime claim owner. -/
structure Chain {widths : CompilerWidths}
    (selected : SelectedVerifier widths)
    (initial final : ClosedCarry Digest.Value) (segmentCount : Nat) where
  receipts : List (Receipt selected)
  model : GlobalFPrime.Chain
    (protocolSchema widths (PackedProof selected)) Digest.Value
    (VerifyClaim selected) initial (verifiedReceipts receipts) final segmentCount

namespace Chain

theorem exactClaimCount
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {initial final : ClosedCarry Digest.Value} {segmentCount : Nat}
    (chain : Chain selected initial final segmentCount) :
    chain.receipts.length = segmentCount * Lifecycle.claimsPerSegment := by
  have exact := chain.model.exactClaimCount
  simpa [verifiedReceipts_length] using exact

theorem everyReceiptAcceptedOnExactBlock
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {initial final : ClosedCarry Digest.Value} {segmentCount : Nat}
    (chain : Chain selected initial final segmentCount) :
    ∀ receipt ∈ chain.receipts,
      selected.verify receipt.proof receipt.envelope.block receipt.output = true := by
  intro receipt _member
  exact receipt.verifier_accepts_exact_block

theorem everyDelayedClaimAccepted
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {initial final : ClosedCarry Digest.Value} {segmentCount : Nat}
    (chain : Chain selected initial final segmentCount) :
    ∀ verified ∈ verifiedReceipts chain.receipts,
      VerifyClaim selected verified.proof verified.claim :=
  chain.model.everyClaimAccepted

theorem completeDelayedSchedule
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {initial final : ClosedCarry Digest.Value} {segmentCount : Nat}
    (chain : Chain selected initial final segmentCount)
    (positiveSegments : 0 < segmentCount) :
    Lifecycle.CompleteSchedule chain.receipts.length := by
  have schedule := chain.model.completeDelayedSchedule positiveSegments
  simpa [verifiedReceipts_length] using schedule

/-- Terminal handling consumes the exact verified form of the trailing receipt,
not another claim with the same memory suffix. -/
theorem terminalConsumesExactTrailingReceipt
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {initial final : ClosedCarry Digest.Value} {segmentCount : Nat}
    (chain : Chain selected initial final segmentCount)
    (positive : 0 < chain.receipts.length) :
    GlobalFPrime.consumedClaimAt (verifiedReceipts chain.receipts)
        (Lifecycle.terminalIndex (verifiedReceipts chain.receipts).length) =
      some ((chain.receipts.get
        ⟨chain.receipts.length - 1, by omega⟩).toVerified) := by
  have mappedPositive : 0 < (verifiedReceipts chain.receipts).length := by
    simpa [verifiedReceipts_length] using positive
  have terminal := GlobalFPrime.terminal_consumes_exact_trailing_claim
    (claims := verifiedReceipts chain.receipts) mappedPositive
  rw [terminal]
  congr 1
  let index : Fin chain.receipts.length :=
    ⟨chain.receipts.length - 1, by omega⟩
  have getExact := verifiedReceipts_get chain.receipts index
  simpa [index, verifiedReceipts_length] using getExact

end Chain

end Nightstream.Implementation.Nebula.FullClaimGlobalFPrime
