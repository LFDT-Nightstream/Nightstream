import Nightstream.Implementation.NebulaV2.FullClaimNifsCall

set_option autoImplicit false

namespace tests.NebulaV2FullClaimEnvelope

open Nightstream.Implementation.NebulaV2.FullClaimEnvelope
open Nightstream.Implementation.NebulaV2.FullClaimNifsReceipt
open Nightstream.Protocol.NebulaV2

#check Value.encode_slice
#check Value.encode_injective_on_canonical
#check Nightstream.Implementation.NebulaV2.FullClaimEnvelopeRows.input_eq_block
#check Nightstream.Implementation.NebulaV2.FullClaimNifsCall.satisfying_call_and_transition_bind_exact_claim

/-- This verifier is a countermodel to any claim that the local receipt type
alone proves NIFS soundness. It accepts every proof and every full claim. -/
def alwaysAcceptingVerifier (widths : CompilerWidths)
    (verifierKeyDigest relationManifestDigest : Digest.Value) :
    SelectedVerifier widths where
  Proof := Unit
  Output := Unit
  verifierKeyDigest := verifierKeyDigest
  relationManifestDigest := relationManifestDigest
  profile := Profile.v2
  profileExact := rfl
  verify := fun _proof _claim _output => true

def impossibleRelation {widths : CompilerWidths}
    (selected : SelectedVerifier widths) (_claim : Claim selected)
    (_output : selected.Output) : Prop := False

/-- Exact envelope linkage is necessary but does not replace the separate NIFS
cryptographic reduction. -/
theorem always_accepting_verifier_has_receipt_without_semantic_relation
    {widths : CompilerWidths}
    (verifierKeyDigest relationManifestDigest : Digest.Value)
    (value : Value widths) (canonical : value.Canonical) :
    ∃ receipt : Receipt
        (alwaysAcceptingVerifier widths verifierKeyDigest relationManifestDigest),
      ¬ impossibleRelation
        (alwaysAcceptingVerifier widths verifierKeyDigest relationManifestDigest)
        receipt.claim receipt.output := by
  let selected :=
    alwaysAcceptingVerifier widths verifierKeyDigest relationManifestDigest
  let receipt : Receipt selected :=
    { claim := value.toProtocolClaim
      proof := ()
      output := ()
      accepted := by
        change value.Canonical ∧ true = true
        exact ⟨canonical, rfl⟩ }
  exact ⟨receipt, by simp [impossibleRelation]⟩

end tests.NebulaV2FullClaimEnvelope
