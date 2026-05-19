import DirectCcsFPrime.SuperNeoBridge

/-!
Typed interface for the direct CCS bridge into imported SuperNeo CE/Ajtai and
stage theorem surfaces.

Spec: `specs/SuperNeoBridge.spec.md`
-/

namespace DirectCcsFPrime

namespace SuperNeoBridgeInterface

open DecDigitUniqueness

abbrev AjtaiCEChildBundle (k n : Nat) :=
  SuperNeoBridge.AjtaiCEChildBundle k n

abbrev AcceptedSuperNeoReducedHandle
    {n : Nat}
    {Source Proof : Type}
    (SourceBindsParent : Source → (Fin n → Nat) → Prop)
    (Verify :
      Source →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop)
    (source : Source)
    (parentResidues : Fin n → Nat)
    (bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n)
    (proof : Proof) : Prop :=
  SuperNeoBridge.AcceptedSuperNeoReducedHandle
    SourceBindsParent
    Verify
    source
    parentResidues
    bundle
    proof

abbrev GoldilocksChildTableProofSound
    {n : Nat}
    {Source Proof : Type}
    (Verify :
      Source →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop)
    (source : Source)
    (parentResidues : Fin n → Nat) : Prop :=
  GoldilocksChildTableAuthorization.GoldilocksChildTableProofSound
    Verify
    source
    parentResidues

abbrev same_opened_parentCEB_digest_source_superneo_next_inputs_of_statementCommitment_and_ajtaiCEOpening :=
  @SuperNeoBridge.same_opened_parentCEB_digest_source_superneo_next_inputs_of_statementCommitment_and_ajtaiCEOpening

abbrev same_opened_parentCEB_digest_source_superneo_challenge_and_inputs_of_statementCommitment_and_ajtaiCEOpening :=
  @SuperNeoBridge.same_opened_parentCEB_digest_source_superneo_challenge_and_inputs_of_statementCommitment_and_ajtaiCEOpening

abbrev ReusedStageAuthority :=
  SuperNeoBridge.ReusedStageAuthority

abbrev piCCSStrong :=
  @SuperNeoBridge.ReusedStageAuthority.piCCSStrong

abbrev piRLCWeak :=
  @SuperNeoBridge.ReusedStageAuthority.piRLCWeak

abbrev piDECKnowledge :=
  @SuperNeoBridge.ReusedStageAuthority.piDECKnowledge

end SuperNeoBridgeInterface

end DirectCcsFPrime
