import DirectCcsFPrime.ProofSystem.PrivatePiDec.Impl.GoldilocksNoWrap
import DirectCcsFPrime.Bridge.SuperNeoBridge

/-!
Private `Pi_DEC` verifier soundness over real SuperNeo child bundles.

This module removes a remaining abstraction leak: the reduced-handle bridge
must not authorize a standalone digit table that is later merely associated
with CE children. The verifier acceptance is stated over the actual
SuperNeo-backed child bundle, whose fields include `CE.Holds` witnesses and the
identity between the checked digit table and the witness-derived table.
-/

namespace DirectCcsFPrime

namespace PrivatePiDecSoundness

open DecDigitUniqueness
open BinaryChildTableAuthorization
open GoldilocksChildTableAuthorization
open SuperNeoBridge

/--
Soundness boundary for a private terminal `Pi_DEC` verifier over an actual
SuperNeo child bundle.

Verification must imply exactly the arithmetic facts needed to make the hidden
child table unique for a parent:

* binary digits,
* exact `k_dec = 14` length,
* Goldilocks modular recomposition to the parent residues.

Full child CE membership is carried by `AjtaiCEChildBundle.ceHolds`; witness
table identity is carried by `AjtaiCEChildBundle.digitTableMatchesWitnesses`.
-/
def PrivatePiDecProofSound
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
  ∀ bundle proof,
    Verify source parentResidues bundle proof →
      binaryColumnDigits bundle.digitTable ∧
      fixedColumnLength 14 bundle.digitTable ∧
      (∀ j,
        recomposeNatDigits (bundle.digitTable j) %
            SuperNeo.Goldilocks.q =
          parentResidues j % SuperNeo.Goldilocks.q)

/--
Accepted private `Pi_DEC` proof over the actual SuperNeo child bundle.
-/
structure AcceptedPrivatePiDec
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
    (proof : Proof) : Prop where
  sourceBound : SourceBindsParent source parentResidues
  proofVerified : Verify source parentResidues bundle proof
  wireIdentity : bundle.nextPiCCSInputs = bundle.digitTable

/--
Accepted private `Pi_DEC` proofs for the same functionally parent-bound source
cannot feed different next `Pi_CCS` child inputs.
-/
theorem same_private_pidec_next_inputs
    {n : Nat}
    {Source Proof : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source →
        (Fin n → Nat) →
        SuperNeoBridge.AjtaiCEChildBundle 14 n →
        Proof →
        Prop}
    {source : Source}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : SuperNeoBridge.AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hBind :
      SourceBindsParentFunctionally SourceBindsParent)
    (hSound :
      PrivatePiDecProofSound Verify source parentA)
    (hA :
      AcceptedPrivatePiDec
        SourceBindsParent
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedPrivatePiDec
        SourceBindsParent
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  have hParent : parentA = parentB :=
    hBind source parentA parentB hA.sourceBound hB.sourceBound
  subst parentB
  have hAProof := hSound bundleA proofA hA.proofVerified
  have hBProof := hSound bundleB proofB hB.proofVerified
  have hMod :
      ∀ j,
        recomposeNatDigits (bundleA.digitTable j) % SuperNeo.Goldilocks.q =
        recomposeNatDigits (bundleB.digitTable j) % SuperNeo.Goldilocks.q := by
    intro j
    calc
      recomposeNatDigits (bundleA.digitTable j) %
            SuperNeo.Goldilocks.q =
          parentA j % SuperNeo.Goldilocks.q := hAProof.2.2 j
      _ = recomposeNatDigits (bundleB.digitTable j) %
            SuperNeo.Goldilocks.q := (hBProof.2.2 j).symm
  have hChildren : bundleA.digitTable = bundleB.digitTable :=
    GoldilocksNoWrap.binary_column_length14_unique_of_goldilocks_mod_eq
      hAProof.2.1
      hBProof.2.1
      hAProof.1
      hBProof.1
      hMod
  calc
    bundleA.nextPiCCSInputs = bundleA.digitTable := hA.wireIdentity
    _ = bundleB.digitTable := hChildren
    _ = bundleB.nextPiCCSInputs := hB.wireIdentity.symm

/--
Same source also fixes the CE witness-derived digit table, because each
accepted bundle requires `digitTable` to be extracted from the CE witnesses.
-/
theorem same_private_pidec_witness_digits
    {n : Nat}
    {Source Proof : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source →
        (Fin n → Nat) →
        SuperNeoBridge.AjtaiCEChildBundle 14 n →
        Proof →
        Prop}
    {source : Source}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : SuperNeoBridge.AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hBind :
      SourceBindsParentFunctionally SourceBindsParent)
    (hSound :
      PrivatePiDecProofSound Verify source parentA)
    (hA :
      AcceptedPrivatePiDec
        SourceBindsParent
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedPrivatePiDec
        SourceBindsParent
        Verify
        source
        parentB
        bundleB
        proofB) :
    SuperNeoBridge.childWitnessDigitTable
        (k := 14)
        (n := n)
        bundleA.witness =
      SuperNeoBridge.childWitnessDigitTable
        (k := 14)
        (n := n)
        bundleB.witness := by
  have hNext :
      bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs :=
    same_private_pidec_next_inputs hBind hSound hA hB
  calc
    SuperNeoBridge.childWitnessDigitTable
          (k := 14)
          (n := n)
          bundleA.witness =
        bundleA.digitTable := bundleA.digitTableMatchesWitnesses.symm
    _ = bundleA.nextPiCCSInputs := hA.wireIdentity.symm
    _ = bundleB.nextPiCCSInputs := hNext
    _ = bundleB.digitTable := hB.wireIdentity
    _ = SuperNeoBridge.childWitnessDigitTable
          (k := 14)
          (n := n)
          bundleB.witness := bundleB.digitTableMatchesWitnesses

/--
Full reduced-handle theorem using the private `Pi_DEC` verifier over real
SuperNeo child bundles and the local Ajtai CE-opening adapter.
-/
theorem same_opened_parentCEB_digest_source_private_pidec_inputs_of_ajtaiCEOpening
    {n : Nat}
    {Digest Proof : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        SuperNeoBridge.AjtaiCEChildBundle 14 n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : SuperNeoBridge.AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hSound :
      PrivatePiDecProofSound Verify source parentA)
    (hA :
      AcceptedPrivatePiDec
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedPrivatePiDec
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_private_pidec_next_inputs
      (ParentOpeningAuthorization.bindsOpenedParentCEBResiduesFor_functionally
        hDigest
        (AjtaiResidueBinding.encodedParentCEBOpeningResiduesFunctionalFor_of_noAjtaiBindingCollision
          hEncoding
          hNoCollision
          adapter))
      hSound
      hA
      hB

end PrivatePiDecSoundness

end DirectCcsFPrime
