import DirectCcsFPrime.ProofSystem.PrivatePiDec.Impl.DecBase2Authorization
import DirectCcsFPrime.ProofSystem.PrivatePiDec.Spec.TranscriptAuthorization

/-!
Concrete base-2 reduced transcript authorization.

This module discharges the abstract reduced-transcript uniqueness obligation
using the canonical SuperNeo base-2 split surface.
-/

namespace DirectCcsFPrime

namespace Base2TranscriptAuthorization

open DecAuthorizationInterface
open DecBase2Authorization
open TranscriptAuthorization

/--
Local proof soundness for a fixed reduced source and base-2 parent.

This models a sumcheck-like proof verifier that is checked by `F'`. Verification
is sufficient only if it implies both DEC recomposition and canonical child
membership for the same child wires that feed the next `Pi_CCS`.
-/
def LocalBase2DecProofSound
    {k : Nat}
    {Source Proof : Type}
    (VerifyBase2DecProof :
      Source → Base2Claim → Base2Children k → Proof → Prop)
    (source : Source)
    (parent : Base2Claim) : Prop :=
  ∀ children proof,
    VerifyBase2DecProof source parent children proof →
      Base2DecRecompose parent children ∧
      CanonicalChildMembership parent children

/--
Same reduced source and parent imply same next accumulator inputs for canonical
base-2 DEC authorization.
-/
theorem same_base2_reduced_source_unique_next_inputs
    {k : Nat}
    {Source Proof : Type}
    {SourceBindsParent : Source → Base2Claim → Prop}
    {VerifyBase2DecProof :
      Source → Base2Claim → Base2Children k → Proof → Prop}
    {source : Source}
    {parent : Base2Claim}
    {childrenA childrenB nextA nextB : Base2Children k}
    {proofA proofB : Proof}
    (hSound :
      LocalBase2DecProofSound
        VerifyBase2DecProof
        source
        parent)
    (hA :
      AcceptedReducedSource
        SourceBindsParent
        VerifyBase2DecProof
        source
        parent
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedReducedSource
        SourceBindsParent
        VerifyBase2DecProof
        source
        parent
        childrenB
        nextB
        proofB) :
    nextA = nextB := by
  have hAProof := hSound childrenA proofA hA.proofVerified
  have hBProof := hSound childrenB proofB hB.proofVerified
  have hChildren :
      childrenA = childrenB :=
    (canonical_child_membership_unique parent k)
      childrenA
      childrenB
      hAProof.1
      hAProof.2
      hBProof.1
      hBProof.2
  calc
    nextA = childrenA := hA.wireIdentity
    _ = childrenB := hChildren
    _ = nextB := hB.wireIdentity.symm

/--
For a deterministic challenge function over the reduced source, accepted
canonical base-2 authorizations with the same source and parent have both the
same public challenge and the same hidden next accumulator.

This theorem is a wiring/uniqueness theorem, not a random-oracle theorem.
-/
theorem same_base2_reduced_source_same_challenge_and_inputs
    {k : Nat}
    {Source Proof Challenge : Type}
    {SourceBindsParent : Source → Base2Claim → Prop}
    {VerifyBase2DecProof :
      Source → Base2Claim → Base2Children k → Proof → Prop}
    {source : Source}
    {parent : Base2Claim}
    {childrenA childrenB nextA nextB : Base2Children k}
    {proofA proofB : Proof}
    (challenge : Source → Challenge)
    (hSound :
      LocalBase2DecProofSound
        VerifyBase2DecProof
        source
        parent)
    (hA :
      AcceptedReducedSource
        SourceBindsParent
        VerifyBase2DecProof
        source
        parent
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedReducedSource
        SourceBindsParent
        VerifyBase2DecProof
        source
        parent
        childrenB
        nextB
        proofB) :
    challenge source = challenge source ∧ nextA = nextB := by
  exact
    ⟨rfl,
      same_base2_reduced_source_unique_next_inputs
        hSound
        hA
        hB⟩

end Base2TranscriptAuthorization

end DirectCcsFPrime
