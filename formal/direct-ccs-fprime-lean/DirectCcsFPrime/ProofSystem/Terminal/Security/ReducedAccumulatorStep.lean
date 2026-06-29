import DirectCcsFPrime.ProofSystem.PrivatePiDec.Impl.CanonicalPrivatePiDecVerifier
import DirectCcsFPrime.Core.Construction2DirectFPrime

/-!
Reduced-handle accumulator step for direct CCS F'.

This module instantiates the accumulator-update side of the direct
Construction-2 transition with the reduced parent `CE(B)` handle strategy. It
does not prove how `Pi_CCS` and `Pi_RLC` compute the parent source; that remains
the `ParentSourceStep` obligation. It proves that once that source is fixed,
private `Pi_DEC` authorization cannot choose different next `Pi_CCS` children.
-/

namespace DirectCcsFPrime

namespace ReducedAccumulatorStep

open DecDigitUniqueness

/--
Accumulator handle exposed to the direct F' public image.

`parentSource` is the compact parent `CE(B)` source for the latest update.
`nextPiCCSInputs` is the authorized `CE(b)^k` child table consumed by the next
`Pi_CCS` step.
-/
structure AccumulatorHandle (Source : Type) (n : Nat) where
  parentSource : Source
  nextPiCCSInputs : ColumnDigits n

/--
An authorization relation is functional when one source cannot authorize two
different next accumulator input tables.
-/
def AuthorizedFunctional
    {Source NextInputs : Type}
    (Authorized : Source → NextInputs → Prop) : Prop :=
  ∀ source nextA nextB,
    Authorized source nextA →
    Authorized source nextB →
      nextA = nextB

/--
The parent-source derivation is functional when the same prior accumulator
state and step index cannot yield two different compact parent sources.

For the concrete protocol this is the theorem obligation for the
`Pi_CCS -> Pi_RLC` parent `CE(B)` source computation.
-/
def ParentSourceFunctional
    {Source : Type}
    {n : Nat}
    (ParentSourceStep :
      Nat → AccumulatorHandle Source n → Source → Prop) : Prop :=
  ∀ i prior sourceA sourceB,
    ParentSourceStep i prior sourceA →
    ParentSourceStep i prior sourceB →
      sourceA = sourceB

/--
Reduced accumulator update:

1. derive the compact parent source from the prior accumulator, and
2. prove that private `Pi_DEC` authorizes the next `Pi_CCS` child inputs from
   that source.
-/
def Step
    {Source : Type}
    {n : Nat}
    (ParentSourceStep :
      Nat → AccumulatorHandle Source n → Source → Prop)
    (Authorized : Source → ColumnDigits n → Prop)
    (i : Nat)
    (prior next : AccumulatorHandle Source n) : Prop :=
  ParentSourceStep i prior next.parentSource ∧
  Authorized next.parentSource next.nextPiCCSInputs

/--
If parent-source derivation and private authorization are both functional, then
the reduced accumulator update is functional in the fields that matter for the
next `Pi_CCS`.
-/
theorem step_fields_functional
    {Source : Type}
    {n : Nat}
    {ParentSourceStep :
      Nat → AccumulatorHandle Source n → Source → Prop}
    {Authorized : Source → ColumnDigits n → Prop}
    {i : Nat}
    {prior nextA nextB : AccumulatorHandle Source n}
    (hParent : ParentSourceFunctional ParentSourceStep)
    (hAuthorized : AuthorizedFunctional Authorized)
    (hA : Step ParentSourceStep Authorized i prior nextA)
    (hB : Step ParentSourceStep Authorized i prior nextB) :
    nextA.parentSource = nextB.parentSource ∧
      nextA.nextPiCCSInputs = nextB.nextPiCCSInputs := by
  rcases hA with ⟨hParentA, hAuthorizedA⟩
  rcases hB with ⟨hParentB, hAuthorizedB⟩
  have hSource :
      nextA.parentSource = nextB.parentSource :=
    hParent i prior nextA.parentSource nextB.parentSource hParentA hParentB
  constructor
  · exact hSource
  · rw [hSource] at hAuthorizedA
    exact
      hAuthorized
        nextB.parentSource
        nextA.nextPiCCSInputs
        nextB.nextPiCCSInputs
        hAuthorizedA
        hAuthorizedB

/--
Construction-2 transition specialization: if two accepted latest direct F'
transitions start from the same prior image and use the reduced accumulator
step, their accumulator-update fields agree.
-/
theorem transition_accumulator_fields_functional
    {Digest Boundary Source : Type}
    {n : Nat}
    {BoundaryStep :
      Nat → Boundary → Boundary → Prop}
    {ParentSourceStep :
      Nat → AccumulatorHandle Source n → Source → Prop}
    {Authorized : Source → ColumnDigits n → Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccumulatorHandle Source n)}
    (hParent : ParentSourceFunctional ParentSourceStep)
    (hAuthorized : AuthorizedFunctional Authorized)
    (hA :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (Step ParentSourceStep Authorized)
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (Step ParentSourceStep Authorized)
        i
        prior
        nextB) :
    nextA.accumulator.parentSource = nextB.accumulator.parentSource ∧
      nextA.accumulator.nextPiCCSInputs =
        nextB.accumulator.nextPiCCSInputs := by
  rcases hA with
    ⟨_hPriorA, _hNextA, _hVkA, _hInitialA, _hPriorPcA,
      _hNextPcA, _hBoundaryA, hAccA⟩
  rcases hB with
    ⟨_hPriorB, _hNextB, _hVkB, _hInitialB, _hPriorPcB,
      _hNextPcB, _hBoundaryB, hAccB⟩
  exact step_fields_functional hParent hAuthorized hAccA hAccB

/--
Canonical private `Pi_DEC` authorization is functional for the reduced parent
handle under the concrete binding assumptions.
-/
theorem canonical_authorized_functional_of_ajtaiCEOpening
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
        ce) :
    AuthorizedFunctional
      (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes := StatementEncodes)) := by
  intro source nextA nextB hA hB
  exact
    CanonicalPrivatePiDecVerifier.authorized_nextPiCCSInputs_functional_of_ajtaiCEOpening
      hDigest
      hEncoding
      hNoCollision
      adapter
      hA
      hB

/--
Canonical private `Pi_DEC` authorization is functional for the deterministic
parent-statement commitment encoder used by the direct terminal path.
-/
theorem canonical_authorized_functional_of_statementCommitment_and_ajtaiCEOpening
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
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce) :
    AuthorizedFunctional
      (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
        (n := n)
        (hashEncoded := hashEncoded)
        (params := params)
        (ce := ce)
        (StatementEncodes :=
          ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent)) := by
  intro source nextA nextB hA hB
  exact
    CanonicalPrivatePiDecVerifier.authorized_nextPiCCSInputs_functional_of_statementCommitment_and_ajtaiCEOpening
      hDigest
      hNoCollision
      adapter
      hA
      hB

/--
Construction-2 transition theorem for the canonical reduced-handle private
`Pi_DEC` authorization.
-/
theorem transition_accumulator_fields_functional_of_ajtaiCEOpening
    {Digest Boundary : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep :
      Nat → Boundary → Boundary → Prop}
    {ParentSourceStep :
      Nat →
        AccumulatorHandle (DigestParentBinding.Source Digest) n →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccumulatorHandle (DigestParentBinding.Source Digest) n)}
    (hParent : ParentSourceFunctional ParentSourceStep)
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
      Construction2DirectFPrime.Transition
        BoundaryStep
        (Step
          ParentSourceStep
          (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
            (n := n)
            (hashEncoded := hashEncoded)
            (params := params)
            (ce := ce)
            (StatementEncodes := StatementEncodes)))
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (Step
          ParentSourceStep
          (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
            (n := n)
            (hashEncoded := hashEncoded)
            (params := params)
            (ce := ce)
            (StatementEncodes := StatementEncodes)))
        i
        prior
        nextB) :
    nextA.accumulator.parentSource = nextB.accumulator.parentSource ∧
      nextA.accumulator.nextPiCCSInputs =
        nextB.accumulator.nextPiCCSInputs := by
  exact
    transition_accumulator_fields_functional
      hParent
      (canonical_authorized_functional_of_ajtaiCEOpening
        hDigest
        hEncoding
        hNoCollision
        adapter)
      hA
      hB

/--
Construction-2 transition theorem for the deterministic parent-statement
commitment encoder used by the direct terminal path.
-/
theorem transition_accumulator_fields_functional_of_statementCommitment_and_ajtaiCEOpening
    {Digest Boundary : Type}
    {n : Nat}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {BoundaryStep :
      Nat → Boundary → Boundary → Prop}
    {ParentSourceStep :
      Nat →
        AccumulatorHandle (DigestParentBinding.Source Digest) n →
        DigestParentBinding.Source Digest →
          Prop}
    {i : Nat}
    {prior nextA nextB :
      Construction2DirectFPrime.PublicImage
        Digest
        Boundary
        (AccumulatorHandle (DigestParentBinding.Source Digest) n)}
    (hParent : ParentSourceFunctional ParentSourceStep)
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
      Construction2DirectFPrime.Transition
        BoundaryStep
        (Step
          ParentSourceStep
          (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
            (n := n)
            (hashEncoded := hashEncoded)
            (params := params)
            (ce := ce)
            (StatementEncodes :=
              ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)))
        i
        prior
        nextA)
    (hB :
      Construction2DirectFPrime.Transition
        BoundaryStep
        (Step
          ParentSourceStep
          (CanonicalPrivatePiDecVerifier.AuthorizedNextPiCCSInputs
            (n := n)
            (hashEncoded := hashEncoded)
            (params := params)
            (ce := ce)
            (StatementEncodes :=
              ParentOpeningAuthorization.StatementEncodesByCommitment
                commitmentOfParent)))
        i
        prior
        nextB) :
    nextA.accumulator.parentSource = nextB.accumulator.parentSource ∧
      nextA.accumulator.nextPiCCSInputs =
        nextB.accumulator.nextPiCCSInputs := by
  exact
    transition_accumulator_fields_functional
      hParent
      (canonical_authorized_functional_of_statementCommitment_and_ajtaiCEOpening
        hDigest
        hNoCollision
        adapter)
      hA
      hB

end ReducedAccumulatorStep

end DirectCcsFPrime
