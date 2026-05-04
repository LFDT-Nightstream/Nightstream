import DirectCcsFPrime.DecProofAuthorization

/-!
Reduced transcript authorization for parent-bound DEC children.

This module proves the equality fact behind the reduced Fiat-Shamir input:
under a sound DEC proof verifier and a uniqueness theorem for the child
decomposition, the same reduced public source cannot authorize two different
hidden child accumulators for the next `Pi_CCS` verifier.
-/

namespace DirectCcsFPrime

namespace TranscriptAuthorization

open DecAuthorizationInterface
open DecProofAuthorization

universe u

/--
An accepted reduced transcript source.

`source` is the public data that the challenge function sees. Examples include
a parent `CE(B)` handle plus a child-table commitment or proof commitment.
`SourceBindsParent` is the proof-visible statement that this source binds the
parent used by the DEC proof verifier.
-/
structure AcceptedReducedSource
    {k : Nat}
    {Source Claim Proof : Type u}
    (SourceBindsParent : Source → Claim → Prop)
    (VerifyDecProof : Source → Claim → Children k Claim → Proof → Prop)
    (source : Source)
    (parent : Claim)
    (children nextInputs : Children k Claim)
    (proof : Proof) : Prop where
  sourceBound : SourceBindsParent source parent
  proofVerified : VerifyDecProof source parent children proof
  wireIdentity : WireIdentity nextInputs children

/-- Soundness for a proof verifier whose public input is the reduced source. -/
def ReducedDecProofSound
    {k : Nat}
    {Source Claim Proof : Type u}
    (VerifyDecProof : Source → Claim → Children k Claim → Proof → Prop)
    (DecRecompose : Claim → Children k Claim → Prop)
    (ChildCEMembership : Children k Claim → Prop) : Prop :=
  ∀ source parent children proof,
    VerifyDecProof source parent children proof →
      DecRecompose parent children ∧ ChildCEMembership children

/--
Convert a reduced-source authorization into the existing proof-level
authorization by treating the reduced source as the digest/public handle.
-/
theorem accepted_reduced_source_to_dec_proof
    {k : Nat}
    {Source Claim Proof : Type u}
    {SourceBindsParent : Source → Claim → Prop}
    {VerifyDecProof : Source → Claim → Children k Claim → Proof → Prop}
    {source : Source}
    {parent : Claim}
    {children nextInputs : Children k Claim}
    {proof : Proof}
    (h :
      AcceptedReducedSource
        SourceBindsParent
        VerifyDecProof
        source
        parent
        children
        nextInputs
        proof) :
    AcceptedByDecProof
      SourceBindsParent
      VerifyDecProof
      parent
      source
      children
      nextInputs
      proof :=
  { hashBound := h.sourceBound
    proofVerified := h.proofVerified
    wireIdentity := h.wireIdentity }

theorem reduced_sound_to_dec_proof_sound
    {k : Nat}
    {Source Claim Proof : Type u}
    {VerifyDecProof : Source → Claim → Children k Claim → Proof → Prop}
    {DecRecompose : Claim → Children k Claim → Prop}
    {ChildCEMembership : Children k Claim → Prop}
    (hSound :
      ReducedDecProofSound
        VerifyDecProof
        DecRecompose
        ChildCEMembership) :
    DecProofSound
      VerifyDecProof
      DecRecompose
      ChildCEMembership := by
  intro source parent children proof hVerify
  exact hSound source parent children proof hVerify

/--
The main reduced-transcript safety fact.

The same reduced source and parent cannot authorize two different next incoming
accumulators once the DEC proof verifier is sound and the child decomposition is
unique.
-/
theorem same_reduced_source_unique_next_inputs
    {k : Nat}
    {Source Claim Proof : Type u}
    {SourceBindsParent : Source → Claim → Prop}
    {VerifyDecProof : Source → Claim → Children k Claim → Proof → Prop}
    {DecRecompose : Claim → Children k Claim → Prop}
    {ChildCEMembership : Children k Claim → Prop}
    {source : Source}
    {parent : Claim}
    {childrenA childrenB nextA nextB : Children k Claim}
    {proofA proofB : Proof}
    (hSound :
      ReducedDecProofSound
        VerifyDecProof
        DecRecompose
        ChildCEMembership)
    (hUnique :
      UniqueDecAuthorization DecRecompose ChildCEMembership parent)
    (hA :
      AcceptedReducedSource
        SourceBindsParent
        VerifyDecProof
        source
        parent
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedReducedSource
        SourceBindsParent
        VerifyDecProof
        source
        parent
        childrenB
        nextB
        proofB) :
    nextA = nextB := by
  exact
    accepted_by_dec_proof_unique_next_inputs
      (reduced_sound_to_dec_proof_sound hSound)
      hUnique
      (accepted_reduced_source_to_dec_proof hA)
      (accepted_reduced_source_to_dec_proof hB)

/--
Any deterministic challenge function over the reduced source is insensitive to
attempted hidden-child changes, because accepted hidden children are unique.

This theorem is intentionally not a random-oracle theorem. It is the wiring
claim needed before any Fiat-Shamir hash instantiation is meaningful.
-/
theorem same_reduced_source_same_challenge_and_inputs
    {k : Nat}
    {Source Claim Proof Challenge : Type u}
    {SourceBindsParent : Source → Claim → Prop}
    {VerifyDecProof : Source → Claim → Children k Claim → Proof → Prop}
    {DecRecompose : Claim → Children k Claim → Prop}
    {ChildCEMembership : Children k Claim → Prop}
    {source : Source}
    {parent : Claim}
    {childrenA childrenB nextA nextB : Children k Claim}
    {proofA proofB : Proof}
    (challenge : Source → Challenge)
    (hSound :
      ReducedDecProofSound
        VerifyDecProof
        DecRecompose
        ChildCEMembership)
    (hUnique :
      UniqueDecAuthorization DecRecompose ChildCEMembership parent)
    (hA :
      AcceptedReducedSource
        SourceBindsParent
        VerifyDecProof
        source
        parent
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedReducedSource
        SourceBindsParent
        VerifyDecProof
        source
        parent
        childrenB
        nextB
        proofB) :
    challenge source = challenge source ∧ nextA = nextB := by
  exact
    ⟨rfl,
      same_reduced_source_unique_next_inputs
        hSound
        hUnique
        hA
        hB⟩

end TranscriptAuthorization

end DirectCcsFPrime
