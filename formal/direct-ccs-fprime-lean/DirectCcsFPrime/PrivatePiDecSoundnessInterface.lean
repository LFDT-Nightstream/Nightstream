import DirectCcsFPrime.PrivatePiDecSoundness

/-!
Typed interface for private `Pi_DEC` verifier soundness.

Spec: `specs/PrivatePiDecSoundness.spec.md`
-/

namespace DirectCcsFPrime

namespace PrivatePiDecSoundnessInterface

abbrev PrivatePiDecProofSound
    {n : Nat}
    {Source Proof : Type}
    (Verify :
      Source →
        (Fin n → Nat) →
        SuperNeoBridge.AjtaiCEChildBundle 14 n →
        Proof →
        Prop)
    (source : Source)
    (parentResidues : Fin n → Nat) : Prop :=
  PrivatePiDecSoundness.PrivatePiDecProofSound
    Verify
    source
    parentResidues

abbrev AcceptedPrivatePiDec
    {n : Nat}
    {Source Proof : Type}
    (SourceBindsParent : Source → (Fin n → Nat) → Prop)
    (Verify :
      Source →
        (Fin n → Nat) →
        SuperNeoBridge.AjtaiCEChildBundle 14 n →
        Proof →
        Prop)
    (source : Source)
    (parentResidues : Fin n → Nat)
    (bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n)
    (proof : Proof) : Prop :=
  PrivatePiDecSoundness.AcceptedPrivatePiDec
    SourceBindsParent
    Verify
    source
    parentResidues
    bundle
    proof

abbrev same_private_pidec_next_inputs :=
  @PrivatePiDecSoundness.same_private_pidec_next_inputs

abbrev same_private_pidec_witness_digits :=
  @PrivatePiDecSoundness.same_private_pidec_witness_digits

abbrev same_opened_parentCEB_digest_source_private_pidec_inputs_of_ajtaiCEOpening :=
  @PrivatePiDecSoundness.same_opened_parentCEB_digest_source_private_pidec_inputs_of_ajtaiCEOpening

end PrivatePiDecSoundnessInterface

end DirectCcsFPrime
