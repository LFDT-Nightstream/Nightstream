import DirectCcsFPrime.DecAuthorization
import SuperNeo.DecompInterface

/-!
Concrete base-2 DEC authorization bridge.

This module reuses the existing SuperNeo base-2 decomposition surface to
instantiate the abstract `UniqueDecAuthorization` obligation for canonical
`splitBase2Coeffs` children.
-/

namespace DirectCcsFPrime

namespace DecBase2Authorization

open DecAuthorizationInterface

/-- Minimal concrete claim surface for the DEC uniqueness bridge. -/
structure Base2Claim where
  z : SuperNeo.Coeffs
deriving DecidableEq

def Base2Children (k : Nat) :=
  Children k Base2Claim

/-- Canonical private children obtained from SuperNeo's base-2 split. -/
def canonicalChildren (parent : Base2Claim) (k : Nat) : Base2Children k :=
  fun i => { z := (SuperNeo.splitBase2Coeffs parent.z k)[i.1]! }

/-- Convert child function form back to the row array expected by SuperNeo DEC helpers. -/
def childrenRows {k : Nat} (children : Base2Children k) : Array SuperNeo.Coeffs :=
  Array.ofFn (fun i : Fin k => (children i).z)

/-- Recompose child witnesses with SuperNeo's base-2 recomposition helper. -/
def Base2DecRecompose (parent : Base2Claim) {k : Nat} (children : Base2Children k) : Prop :=
  SuperNeo.recomposeBase2Coeffs (childrenRows children) = parent.z

/--
Concrete child membership for this bridge: children are exactly the canonical
SuperNeo base-2 split rows of the parent witness.

The full production CE membership theorem should refine this predicate by
deriving canonical children from low-norm digit bounds, commitment binding, and
evaluation consistency. This predicate is the sufficient canonical split
instantiation used to discharge the abstract wiring theorem today.
-/
def CanonicalChildMembership (parent : Base2Claim) {k : Nat} (children : Base2Children k) : Prop :=
  children = canonicalChildren parent k

theorem canonical_child_membership_unique
    (parent : Base2Claim)
    (k : Nat) :
    UniqueDecAuthorization
      (Claim := Base2Claim)
      (k := k)
      (fun p cs => Base2DecRecompose (k := k) p cs)
      (CanonicalChildMembership parent)
      parent := by
  intro a b _haRec haMem _hbRec hbMem
  exact haMem.trans hbMem.symm

theorem canonical_children_recompose_of_parent_bound
    (parent : Base2Claim)
    (k : Nat)
    (hk : 0 < k)
    (hLt : ∀ j : Fin parent.z.size, parent.z[j.1].val < 2 ^ k) :
    Base2DecRecompose parent (canonicalChildren parent k) := by
  unfold Base2DecRecompose
  have hRows :
      childrenRows (canonicalChildren parent k)
        =
      SuperNeo.splitBase2Coeffs parent.z k := by
    simp [childrenRows, canonicalChildren, SuperNeo.splitBase2Coeffs]
  rw [hRows]
  exact
    SuperNeo.DecompInterface.recomposeBase2Coeffs_splitBase2Coeffs_eq_of_val_lt_pow
      parent.z
      k
      hk
      hLt

theorem accepted_next_inputs_unique_for_canonical_base2
    {Digest : Type}
    {HashBindsParent : Digest → Base2Claim → Prop}
    {parent : Base2Claim}
    {digest : Digest}
    {k : Nat}
    {childrenA childrenB nextA nextB : Base2Children k}
    (hA :
      AcceptedNextInputs
        HashBindsParent
        (fun p cs => Base2DecRecompose (k := k) p cs)
        (CanonicalChildMembership parent)
        parent
        digest
        childrenA
        nextA)
    (hB :
      AcceptedNextInputs
        HashBindsParent
        (fun p cs => Base2DecRecompose (k := k) p cs)
        (CanonicalChildMembership parent)
        parent
        digest
        childrenB
        nextB) :
    nextA = nextB :=
  DecAuthorization.accepted_next_inputs_unique_of_unique_dec
    (canonical_child_membership_unique parent k)
    hA
    hB

end DecBase2Authorization

end DirectCcsFPrime
