import Nightstream.Implementation.Nebula.FPrime.Claim.EnvelopeRows
import Nightstream.Implementation.Nebula.FPrime.Claim.NifsReceipt

/-!
Contract: row-linked call to the setup-selected V2 NIFS verifier.

Assurance tier: implementation schema.

Owns the conversion from a satisfying full-envelope link block and an accepted
selected-verifier call to one exact delayed full-claim receipt.

Does not own the selected verifier's internal R1CS rows, final generated row
inclusion, NIFS soundness, or recursive-size closure.

Emits constraints: through the contained `FullClaimEnvelopeRows.CallSite`.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.Nebula.FullClaimNifsCall

open Nightstream.Implementation.Nebula.FullClaimEnvelope
open Nightstream.Implementation.Nebula.FullClaimNifsReceipt
open Nightstream.Implementation.R1CS
open Nightstream.Protocol.Nebula
open Nightstream.Protocol.Nebula.FPrime
open Nightstream.Protocol.Nebula.ProductState
open Nightstream.SuperNeo.Concrete

/-- A generated recursive call must instantiate this schema with its actual
rows, assignment, absolute columns, and selected NIFS verifier. -/
structure CircuitCall {widths : CompilerWidths}
    (selected : SelectedVerifier widths)
    (programRows : List Row) (assignment : Nat → Nat) where
  claim : Claim selected
  proof : selected.Proof
  output : selected.Output
  input : FixedBits.Word widths.totalBits
  claimCanonical : (Value.ofProtocolClaim claim).Canonical
  link : FullClaimEnvelopeRows.CallSite programRows assignment
    (Value.ofProtocolClaim claim) input
  verifierAccepted : selected.verify proof input output = true

namespace CircuitCall

def envelope {widths : CompilerWidths}
    {selected : SelectedVerifier widths} {programRows : List Row}
    {assignment : Nat → Nat}
    (call : CircuitCall selected programRows assignment) : Value widths :=
  Value.ofProtocolClaim call.claim

theorem input_is_exact_full_claim
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {programRows : List Row} {assignment : Nat → Nat}
    (call : CircuitCall selected programRows assignment)
    (satisfies : Satisfies programRows assignment) :
    call.input = call.envelope.block :=
  call.link.sound satisfies

/-- The row link turns the circuit call into the exact receipt used by the
delayed F-prime model. -/
def toReceipt
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {programRows : List Row} {assignment : Nat → Nat}
    (call : CircuitCall selected programRows assignment)
    (satisfies : Satisfies programRows assignment) : Receipt selected where
  claim := call.claim
  proof := call.proof
  output := call.output
  accepted := by
    constructor
    · exact call.claimCanonical
    · have inputExact := call.input_is_exact_full_claim satisfies
      change selected.verify call.proof call.envelope.block call.output = true
      rw [← inputExact]
      exact call.verifierAccepted

theorem receipt_uses_call_claim
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {programRows : List Row} {assignment : Nat → Nat}
    (call : CircuitCall selected programRows assignment)
    (satisfies : Satisfies programRows assignment) :
    (call.toReceipt satisfies).claim = call.claim ∧
      (call.toReceipt satisfies).envelope = call.envelope :=
  ⟨rfl, rfl⟩

end CircuitCall

/-- If F-prime consumes the receipt made by this call, it consumes the memory
field of the exact full claim accepted on the row-linked input. -/
theorem satisfying_call_and_transition_bind_exact_claim
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {programRows : List Row} {assignment : Nat → Nat}
    {balanced : State K → Prop}
    {before after : Carry Digest.Value (Challenges K) (State K)}
    (call : CircuitCall selected programRows assignment)
    (satisfies : Satisfies programRows assignment)
    (transition : FullClaimNifsReceipt.Transition selected balanced before
      (call.toReceipt satisfies) after) :
    selected.verify call.proof call.input call.output = true ∧
      call.input = call.envelope.block ∧
      Consumes balanced before call.claim.memory after := by
  exact ⟨call.verifierAccepted, call.input_is_exact_full_claim satisfies,
    transition.consumes⟩

end Nightstream.Implementation.Nebula.FullClaimNifsCall
