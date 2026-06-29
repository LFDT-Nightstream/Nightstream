import DirectCcsFPrime.ProofSystem.PrivatePiDec.Security.PrivatePiDecSoundness

/-!
Canonical private `Pi_DEC` verifier relation.

This module instantiates the private `Pi_DEC` soundness boundary with the exact
terminal relation needed by the reduced `CE(B)^1` handle strategy. It does not
model a separate sumcheck protocol. The verifier relation is the set of
constraints the terminal proof must enforce over the actual SuperNeo child
bundle.
-/

namespace DirectCcsFPrime

namespace CanonicalPrivatePiDecVerifier

open DecDigitUniqueness
open BinaryChildTableAuthorization
open GoldilocksChildTableAuthorization
open PrivatePiDecSoundness
open SuperNeoBridge

/--
Canonical terminal private `Pi_DEC` verifier relation.

The source is present because the public challenge source chooses the parent
handle; the actual relation checks the hidden child bundle against the opened
parent residues.
-/
def Verify
    {n : Nat}
    {Source : Type}
    (_source : Source)
    (parentResidues : Fin n → Nat)
    (bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n)
    (_proof : Unit) : Prop :=
  binaryColumnDigits bundle.digitTable ∧
  fixedColumnLength 14 bundle.digitTable ∧
  (∀ j,
    recomposeNatDigits (bundle.digitTable j) %
        SuperNeo.Goldilocks.q =
      parentResidues j % SuperNeo.Goldilocks.q)

/--
The canonical verifier relation satisfies the private `Pi_DEC` soundness
boundary by definition.
-/
theorem verify_sound
    {n : Nat}
    {Source : Type}
    {source : Source}
    {parentResidues : Fin n → Nat} :
    PrivatePiDecProofSound
      (Verify (n := n) (Source := Source))
      source
      parentResidues := by
  intro bundle proof hVerified
  exact hVerified

/--
Accepted canonical private `Pi_DEC` proofs for the same functionally
parent-bound source cannot feed different next `Pi_CCS` child inputs.
-/
theorem same_canonical_private_pidec_next_inputs
    {n : Nat}
    {Source : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {source : Source}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : SuperNeoBridge.AjtaiCEChildBundle 14 n}
    {proofA proofB : Unit}
    (hBind :
      SourceBindsParentFunctionally SourceBindsParent)
    (hA :
      AcceptedPrivatePiDec
        SourceBindsParent
        (Verify (n := n) (Source := Source))
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedPrivatePiDec
        SourceBindsParent
        (Verify (n := n) (Source := Source))
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_private_pidec_next_inputs
      hBind
      (verify_sound (n := n) (Source := Source) (source := source)
        (parentResidues := parentA))
      hA
      hB

/--
Full reduced-handle theorem for the canonical private `Pi_DEC` relation.

This is the no-extra-verifier-assumption version of the local theorem: the
private verifier is the concrete terminal relation itself.
-/
theorem same_opened_parentCEB_digest_source_canonical_private_pidec_inputs_of_ajtaiCEOpening
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : SuperNeoBridge.AjtaiCEChildBundle 14 n}
    {proofA proofB : Unit}
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
    (hA :
      AcceptedPrivatePiDec
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        (Verify (n := n) (Source := DigestParentBinding.Source Digest))
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
        (Verify (n := n) (Source := DigestParentBinding.Source Digest))
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    PrivatePiDecSoundness.same_opened_parentCEB_digest_source_private_pidec_inputs_of_ajtaiCEOpening
      hDigest
      hEncoding
      hNoCollision
      adapter
      (verify_sound
        (n := n)
        (Source := DigestParentBinding.Source Digest)
        (source := source)
        (parentResidues := parentA))
      hA
      hB

/--
Implementation-shaped canonical private `Pi_DEC` theorem.

The parent statement encoder is the deterministic commitment encoder
`StatementEncodesByCommitment`, so statement-encoding consistency is discharged
inside Lean instead of supplied by the caller.
-/
theorem same_opened_parentCEB_digest_source_canonical_private_pidec_inputs_of_statementCommitment_and_ajtaiCEOpening
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : SuperNeoBridge.AjtaiCEChildBundle 14 n}
    {proofA proofB : Unit}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      AcceptedPrivatePiDec
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        (Verify (n := n) (Source := DigestParentBinding.Source Digest))
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
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        (Verify (n := n) (Source := DigestParentBinding.Source Digest))
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_opened_parentCEB_digest_source_canonical_private_pidec_inputs_of_ajtaiCEOpening
      hDigest
      ParentOpeningAuthorization.statementEncodesByCommitment_functional
      hNoCollision
      adapter
      hA
      hB

/--
Existential implementation-facing authorization relation.

This is the shape used by an actual terminal proof witness: for a compact
digest-bound parent source, the prover supplies some opened parent residues,
some SuperNeo child bundle, and the canonical private `Pi_DEC` relation proves
that the bundle authorizes exactly `nextInputs` for the next `Pi_CCS`.
-/
def AuthorizedNextPiCCSInputs
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    (source : DigestParentBinding.Source Digest)
    (nextInputs : ColumnDigits n) : Prop :=
  ∃
    (parentResidues : Fin n → Nat)
    (bundle : SuperNeoBridge.AjtaiCEChildBundle 14 n)
    (proof : Unit),
      AcceptedPrivatePiDec
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        (Verify (n := n) (Source := DigestParentBinding.Source Digest))
        source
        parentResidues
        bundle
        proof ∧
      bundle.ce = ce ∧
      bundle.ajtaiParams = params ∧
      nextInputs = bundle.nextPiCCSInputs

/--
For one compact parent source, the existential private `Pi_DEC`
authorization relation is functional.

This is the theorem needed to use `Hash CE(B)^1 + private Pi_DEC` as the
challenge/transition boundary: a prover may choose witnesses, but under parent
encoding binding, Ajtai-backed parent-opening binding, canonical binary DEC,
and child-wire identity, those witnesses cannot authorize two different next
`Pi_CCS` child accumulators for the same source.
-/
theorem authorized_nextPiCCSInputs_functional_of_ajtaiCEOpening
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextA nextB : ColumnDigits n}
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
    (hA :
      AuthorizedNextPiCCSInputs
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextA)
    (hB :
      AuthorizedNextPiCCSInputs
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)
        source
        nextB) :
    nextA = nextB := by
  rcases hA with
    ⟨parentA, bundleA, proofA, hAcceptedA, _hCEA, _hParamsA, hNextA⟩
  rcases hB with
    ⟨parentB, bundleB, proofB, hAcceptedB, _hCEB, _hParamsB, hNextB⟩
  have hBundleInputs :
      bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs :=
    same_opened_parentCEB_digest_source_canonical_private_pidec_inputs_of_ajtaiCEOpening
      hDigest
      hEncoding
      hNoCollision
      adapter
      hAcceptedA
      hAcceptedB
  calc
    nextA = bundleA.nextPiCCSInputs := hNextA
    _ = bundleB.nextPiCCSInputs := hBundleInputs
    _ = nextB := hNextB.symm

/--
Implementation-shaped functionality for existential private `Pi_DEC`
authorization with deterministic parent-statement commitment encoding.
-/
theorem authorized_nextPiCCSInputs_functional_of_statementCommitment_and_ajtaiCEOpening
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {source : DigestParentBinding.Source Digest}
    {nextA nextB : ColumnDigits n}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hA :
      AuthorizedNextPiCCSInputs
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)
        source
        nextA)
    (hB :
      AuthorizedNextPiCCSInputs
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)
        source
        nextB) :
    nextA = nextB :=
  authorized_nextPiCCSInputs_functional_of_ajtaiCEOpening
    hDigest
    ParentOpeningAuthorization.statementEncodesByCommitment_functional
    hNoCollision
    adapter
    hA
    hB

end CanonicalPrivatePiDecVerifier

end DirectCcsFPrime
