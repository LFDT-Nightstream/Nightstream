import Nightstream.Implementation.NebulaV2.FPrime.Claim.Envelope

/-!
Contract: setup-selected NIFS receipt for the exact V2 full-claim envelope.

Assurance tier: implementation model and explicit cryptographic boundary.

Owns one verifier-key-selected Boolean verifier, its exact full-claim bit input,
the accepted proof and output, conversion to the delayed full-claim model, and
the proof that delayed memory consumption uses the same accepted full claim.

Does not prove NIFS knowledge soundness, fold extraction, generated verifier
rows, or verifier-key digest binding. A verifier that always returns `true`
still satisfies this local interface and is not sound.

Emits constraints: no.
-/

set_option autoImplicit false

namespace Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Protocol.NebulaV2
open Nightstream.Protocol.NebulaV2.FPrime
open Nightstream.Protocol.NebulaV2.ProductState
open Nightstream.SuperNeo.Concrete

/-- One setup-owned NIFS verifier. The width index prevents a receipt for one
compiled relation from being used with another relation shape. -/
structure SelectedVerifier (widths : CompilerWidths) where
  Proof : Type
  Output : Type
  verifierKeyDigest : Digest.Value
  relationManifestDigest : Digest.Value
  profile : Profile.Identity
  profileExact : profile = Profile.v2
  verify : Proof → FixedBits.Word widths.totalBits → Output → Bool

abbrev PackedProof {widths : CompilerWidths}
    (selected : SelectedVerifier widths) := selected.Proof × selected.Output

abbrev Claim {widths : CompilerWidths}
    (selected : SelectedVerifier widths) :=
  FullClaim.Claim (protocolSchema widths (PackedProof selected)) Digest.Value
    (Challenges K) (State K)

/-- The exact adapter predicate used by the delayed full-claim model. It
reconstructs the complete envelope from the claim and passes that complete bit
block to the one setup-selected verifier. -/
def VerifyClaim {widths : CompilerWidths}
    (selected : SelectedVerifier widths) :
    PackedProof selected → Claim selected → Prop :=
  fun proofAndOutput claim =>
    let envelope := Value.ofProtocolClaim claim
    envelope.Canonical ∧
      selected.verify proofAndOutput.1 envelope.block proofAndOutput.2 = true

/-- A typed receipt cannot carry a separately selected memory suffix or
commitment bundle. Both are fields of `claim`, which determines the complete
verifier input block. -/
structure Receipt {widths : CompilerWidths}
    (selected : SelectedVerifier widths) where
  claim : Claim selected
  proof : selected.Proof
  output : selected.Output
  accepted : VerifyClaim selected (proof, output) claim

namespace Receipt

def envelope {widths : CompilerWidths} {selected : SelectedVerifier widths}
    (receipt : Receipt selected) : Value widths :=
  Value.ofProtocolClaim receipt.claim

theorem canonical {widths : CompilerWidths}
    {selected : SelectedVerifier widths} (receipt : Receipt selected) :
    receipt.envelope.Canonical := by
  exact receipt.accepted.1

theorem verifier_accepts_exact_block {widths : CompilerWidths}
    {selected : SelectedVerifier widths} (receipt : Receipt selected) :
    selected.verify receipt.proof receipt.envelope.block receipt.output = true :=
  receipt.accepted.2

theorem claim_eq_envelope {widths : CompilerWidths}
    {selected : SelectedVerifier widths} (receipt : Receipt selected) :
    receipt.envelope.toProtocolClaim = receipt.claim :=
  Value.to_ofProtocolClaim receipt.claim

def toVerified {widths : CompilerWidths}
    {selected : SelectedVerifier widths} (receipt : Receipt selected) :
    FullClaim.Verified
      (protocolSchema widths (PackedProof selected)) Digest.Value
      (Challenges K) (State K) (VerifyClaim selected) where
  claim := receipt.claim
  proof := (receipt.proof, receipt.output)
  profileExact := receipt.canonical.profileExact
  accepted := receipt.accepted

theorem verified_claim_is_exact {widths : CompilerWidths}
    {selected : SelectedVerifier widths} (receipt : Receipt selected) :
    receipt.toVerified.claim = receipt.claim ∧
      receipt.toVerified.proof = (receipt.proof, receipt.output) :=
  ⟨rfl, rfl⟩

end Receipt

/-- Delayed transition specialized to a selected full-envelope verifier. -/
abbrev Transition {widths : CompilerWidths}
    (selected : SelectedVerifier widths)
    (balanced : State K → Prop)
    (before : Carry Digest.Value (Challenges K) (State K))
    (receipt : Receipt selected)
    (after : Carry Digest.Value (Challenges K) (State K)) :=
  FullClaim.Transition (VerifyClaim selected) balanced before
    (schema := protocolSchema widths (PackedProof selected))
    (Digest := Digest.Value) (Challenge := Challenges K) (Products := State K)
    receipt.toVerified after

/-- One accepted transition binds both facts needed by the authority chain:
the selected NIFS verifier accepted the complete claim block, and F-prime
consumed the memory field of that exact typed claim. -/
theorem transition_accepts_and_consumes_same_full_claim
    {widths : CompilerWidths} {selected : SelectedVerifier widths}
    {balanced : State K → Prop}
    {before after : Carry Digest.Value (Challenges K) (State K)}
    {receipt : Receipt selected}
    (transition : Transition selected balanced before receipt after) :
    selected.verify receipt.proof receipt.envelope.block receipt.output = true ∧
      Consumes balanced before receipt.claim.memory after := by
  exact ⟨receipt.verifier_accepts_exact_block, transition.consumes⟩

end Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
