/-!
Contract boundary for parent-bound direct CCS F' accumulator authorization.

Spec: `specs/PrivatePiDec/Spec/DecAuthorization.spec.md`
-/

namespace DirectCcsFPrime

namespace DecAuthorizationInterface

universe u

/--
Abstract CE surface. The fields are intentionally generic because this module
proves wiring/authorization obligations, not field arithmetic or commitment
security.
-/
structure CE (C X R Y Z : Type u) where
  c : C
  x : X
  r : R
  y : Y
  z : Z

abbrev Children (k : Nat) (Claim : Type u) :=
  Fin k → Claim

/-- The next `Pi_CCS` input vector must be exactly the authorized child vector. -/
def WireIdentity {k : Nat} {Claim : Type u}
    (nextInputs children : Children k Claim) : Prop :=
  nextInputs = children

/--
All checks that authorize private `CE(b)^k` children from a bound `CE(B)` parent.

`HashBindsParent` represents the proof-visible binding of the parent handle.
`DecRecompose` represents `Pi_DEC` recomposition of parent fields from children.
`ChildCEMembership` represents full child CE relation membership and low-norm
checks.
-/
structure AcceptedNextInputs
    {k : Nat}
    {Claim Digest : Type u}
    (HashBindsParent : Digest → Claim → Prop)
    (DecRecompose : Claim → Children k Claim → Prop)
    (ChildCEMembership : Children k Claim → Prop)
    (parent : Claim)
    (digest : Digest)
    (children nextInputs : Children k Claim) : Prop where
  hashBound : HashBindsParent digest parent
  recompose : DecRecompose parent children
  membership : ChildCEMembership children
  wireIdentity : WireIdentity nextInputs children

/--
The low-norm DEC decomposition of a fixed parent is unique under the modeled
membership checks.

This is an explicit obligation. A concrete arithmetic instantiation must prove
it from the base-`b` digit bounds, commitment binding assumptions, and CE
membership relation.
-/
def UniqueDecAuthorization
    {k : Nat}
    {Claim : Type u}
    (DecRecompose : Claim → Children k Claim → Prop)
    (ChildCEMembership : Children k Claim → Prop)
    (parent : Claim) : Prop :=
  ∀ a b : Children k Claim,
    DecRecompose parent a →
    ChildCEMembership a →
    DecRecompose parent b →
    ChildCEMembership b →
    a = b

end DecAuthorizationInterface

end DirectCcsFPrime
