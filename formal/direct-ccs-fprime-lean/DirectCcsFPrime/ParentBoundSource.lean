import DirectCcsFPrime.GoldilocksChildTableAuthorization

/-!
Concrete parent-bound reduced source.

The reduced-source theorems are only useful if the implementation can point to
a concrete source shape whose parent binding is functional. This module gives
the minimal model: the source carries the parent residues directly. A real
implementation may commit/hash this source, but the parent
residue data must remain verifier-bound rather than self-consistent advice.
-/

namespace DirectCcsFPrime

namespace ParentBoundSource

open DecDigitUniqueness
open GoldilocksChildTableAuthorization

/-- Minimal reduced source that explicitly carries its parent residues. -/
structure Source (n : Nat) where
  parentResidues : Fin n → Nat

/-- The source authorizes exactly the parent residues it carries. -/
def BindsParent {n : Nat} (source : Source n) (parent : Fin n → Nat) : Prop :=
  source.parentResidues = parent

/-- Concrete parent binding is functional. -/
theorem binds_parent_functionally {n : Nat} :
    SourceBindsParentFunctionally (BindsParent (n := n)) := by
  intro source parentA parentB hA hB
  exact hA.symm.trans hB

/--
For the concrete parent-bound source, accepted Goldilocks child-table
authorizations for the same source have equal next `Pi_CCS` inputs.
-/
theorem same_source_next_inputs
    {n : Nat}
    {Proof : Type}
    {Verify :
      Source n → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source n}
    {parentA parentB : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedGoldilocksChildTable
        (BindsParent (n := n))
        Verify
        source
        parentA
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedGoldilocksChildTable
        (BindsParent (n := n))
        Verify
        source
        parentB
        childrenB
        nextB
        proofB) :
    nextA = nextB := by
  exact
    same_functionally_bound_source_next_inputs
      binds_parent_functionally
      hSound
      hA
      hB

/--
For deterministic challenges over the concrete parent-bound source, accepted
authorizations for the same source have equal challenge and equal next inputs.
-/
theorem same_source_challenge_and_inputs
    {n : Nat}
    {Proof Challenge : Type}
    {Verify :
      Source n → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source n}
    {parentA parentB : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (challenge : Source n → Challenge)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedGoldilocksChildTable
        (BindsParent (n := n))
        Verify
        source
        parentA
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedGoldilocksChildTable
        (BindsParent (n := n))
        Verify
        source
        parentB
        childrenB
        nextB
        proofB) :
    challenge source = challenge source ∧ nextA = nextB := by
  exact
    same_functionally_bound_source_challenge_and_inputs
      challenge
      binds_parent_functionally
      hSound
      hA
      hB

end ParentBoundSource

end DirectCcsFPrime
