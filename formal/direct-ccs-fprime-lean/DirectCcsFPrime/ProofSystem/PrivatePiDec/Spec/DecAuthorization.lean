import DirectCcsFPrime.ProofSystem.PrivatePiDec.Spec.DecAuthorizationInterface

/-!
Parent-bound accumulator authorization theorems for direct CCS F'.

This module proves the protocol wiring facts that prevent a prover from binding
a `CE(B)` parent while feeding different private `CE(b)^k` children to the next
`Pi_CCS`.
-/

namespace DirectCcsFPrime

namespace DecAuthorization

open DecAuthorizationInterface

universe u

variable {k : Nat}
variable {Claim Digest : Type u}

variable {HashBindsParent : Digest → Claim → Prop}
variable {DecRecompose : Claim → Children k Claim → Prop}
variable {ChildCEMembership : Children k Claim → Prop}

theorem accepted_next_inputs_are_authorized_children
    {parent : Claim}
    {digest : Digest}
    {children nextInputs : Children k Claim}
    (h :
      AcceptedNextInputs
        HashBindsParent
        DecRecompose
        ChildCEMembership
        parent
        digest
        children
        nextInputs) :
    nextInputs = children :=
  h.wireIdentity

theorem different_next_inputs_violate_wire_identity
    {parent : Claim}
    {digest : Digest}
    {children nextInputs altNext : Children k Claim}
    (h :
      AcceptedNextInputs
        HashBindsParent
        DecRecompose
        ChildCEMembership
        parent
        digest
        children
        nextInputs)
    (hDiff : altNext ≠ nextInputs) :
    ¬ WireIdentity altNext children := by
  intro hWire
  apply hDiff
  calc
    altNext = children := hWire
    _ = nextInputs := h.wireIdentity.symm

theorem accepted_next_inputs_unique_of_unique_dec
    {parent : Claim}
    {digest : Digest}
    {childrenA childrenB nextA nextB : Children k Claim}
    (hUnique :
      UniqueDecAuthorization
        DecRecompose
        ChildCEMembership
        parent)
    (hA :
      AcceptedNextInputs
        HashBindsParent
        DecRecompose
        ChildCEMembership
        parent
        digest
        childrenA
        nextA)
    (hB :
      AcceptedNextInputs
        HashBindsParent
        DecRecompose
        ChildCEMembership
        parent
        digest
        childrenB
        nextB) :
    nextA = nextB := by
  have hChildren :
      childrenA = childrenB :=
    hUnique
      childrenA
      childrenB
      hA.recompose
      hA.membership
      hB.recompose
      hB.membership
  calc
    nextA = childrenA := hA.wireIdentity
    _ = childrenB := hChildren
    _ = nextB := hB.wireIdentity.symm

theorem different_next_inputs_must_violate_authorization
    {parent : Claim}
    {digest : Digest}
    {children nextInputs altChildren altNext : Children k Claim}
    (hUnique :
      UniqueDecAuthorization
        DecRecompose
        ChildCEMembership
        parent)
    (h :
      AcceptedNextInputs
        HashBindsParent
        DecRecompose
        ChildCEMembership
        parent
        digest
        children
        nextInputs)
    (hDiff : altNext ≠ nextInputs) :
    ¬
      (DecRecompose parent altChildren ∧
       ChildCEMembership altChildren ∧
       WireIdentity altNext altChildren) := by
  intro hAlt
  have hChildren :
      children = altChildren :=
    hUnique
      children
      altChildren
      h.recompose
      h.membership
      hAlt.1
      hAlt.2.1
  apply hDiff
  calc
    altNext = altChildren := hAlt.2.2
    _ = children := hChildren.symm
    _ = nextInputs := h.wireIdentity.symm

end DecAuthorization

end DirectCcsFPrime
