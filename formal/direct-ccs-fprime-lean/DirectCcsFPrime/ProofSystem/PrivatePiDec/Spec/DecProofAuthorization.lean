import DirectCcsFPrime.ProofSystem.PrivatePiDec.Spec.DecAuthorization

/-!
Proof-system boundary for compact DEC authorization.

This module is intentionally proof-system agnostic. A concrete sumcheck-like
protocol may instantiate `VerifyDecProof`, but it is sound for this use only if
verification implies the DEC recomposition and child-membership predicates over
the same private child wires.
-/

namespace DirectCcsFPrime

namespace DecProofAuthorization

open DecAuthorizationInterface

universe u

/--
Soundness obligation for a proof verifier used to authorize private DEC
children from a compact parent handle.

`VerifyDecProof digest parent children proof` may be implemented by a
sumcheck-like proof, but this predicate is accepted by the IVC argument only if
it implies both arithmetic obligations needed by DEC authorization.
-/
def DecProofSound
    {k : Nat}
    {Claim Digest Proof : Type u}
    (VerifyDecProof : Digest → Claim → Children k Claim → Proof → Prop)
    (DecRecompose : Claim → Children k Claim → Prop)
    (ChildCEMembership : Children k Claim → Prop) : Prop :=
  ∀ digest parent children proof,
    VerifyDecProof digest parent children proof →
      DecRecompose parent children ∧ ChildCEMembership children

/--
Accepted proof-level authorization.

The next-round inputs are accepted only when the parent handle is bound, the
proof verifier accepts the same child wires, and those child wires are exactly
the next `Pi_CCS` incoming accumulator.
-/
structure AcceptedByDecProof
    {k : Nat}
    {Claim Digest Proof : Type u}
    (HashBindsParent : Digest → Claim → Prop)
    (VerifyDecProof : Digest → Claim → Children k Claim → Proof → Prop)
    (parent : Claim)
    (digest : Digest)
    (children nextInputs : Children k Claim)
    (proof : Proof) : Prop where
  hashBound : HashBindsParent digest parent
  proofVerified : VerifyDecProof digest parent children proof
  wireIdentity : WireIdentity nextInputs children

/--
A sound DEC proof verifier can be converted into the lower-level accepted-input
authorization predicate.
-/
theorem accepted_by_dec_proof_to_accepted_next_inputs
    {k : Nat}
    {Claim Digest Proof : Type u}
    {HashBindsParent : Digest → Claim → Prop}
    {VerifyDecProof : Digest → Claim → Children k Claim → Proof → Prop}
    {DecRecompose : Claim → Children k Claim → Prop}
    {ChildCEMembership : Children k Claim → Prop}
    {parent : Claim}
    {digest : Digest}
    {children nextInputs : Children k Claim}
    {proof : Proof}
    (hSound : DecProofSound VerifyDecProof DecRecompose ChildCEMembership)
    (h :
      AcceptedByDecProof
        HashBindsParent
        VerifyDecProof
        parent
        digest
        children
        nextInputs
        proof) :
    AcceptedNextInputs
      HashBindsParent
      DecRecompose
      ChildCEMembership
      parent
      digest
      children
      nextInputs := by
  have hProof := hSound digest parent children proof h.proofVerified
  exact
    { hashBound := h.hashBound
      recompose := hProof.1
      membership := hProof.2
      wireIdentity := h.wireIdentity }

/--
If the proof verifier is sound and the underlying DEC authorization is unique,
then two accepted proof authorizations of the same parent cannot lead to
different next-round inputs.
-/
theorem accepted_by_dec_proof_unique_next_inputs
    {k : Nat}
    {Claim Digest Proof : Type u}
    {HashBindsParent : Digest → Claim → Prop}
    {VerifyDecProof : Digest → Claim → Children k Claim → Proof → Prop}
    {DecRecompose : Claim → Children k Claim → Prop}
    {ChildCEMembership : Children k Claim → Prop}
    {parent : Claim}
    {digest : Digest}
    {childrenA childrenB nextA nextB : Children k Claim}
    {proofA proofB : Proof}
    (hSound : DecProofSound VerifyDecProof DecRecompose ChildCEMembership)
    (hUnique :
      UniqueDecAuthorization DecRecompose ChildCEMembership parent)
    (hA :
      AcceptedByDecProof
        HashBindsParent
        VerifyDecProof
        parent
        digest
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedByDecProof
        HashBindsParent
        VerifyDecProof
        parent
        digest
        childrenB
        nextB
        proofB) :
    nextA = nextB := by
  exact
    DecAuthorization.accepted_next_inputs_unique_of_unique_dec
      hUnique
      (accepted_by_dec_proof_to_accepted_next_inputs hSound hA)
      (accepted_by_dec_proof_to_accepted_next_inputs hSound hB)

end DecProofAuthorization

end DirectCcsFPrime
