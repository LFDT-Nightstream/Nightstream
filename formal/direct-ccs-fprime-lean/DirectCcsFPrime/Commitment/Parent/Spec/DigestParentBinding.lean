import DirectCcsFPrime.ProofSystem.PrivatePiDec.Security.GoldilocksChildTableAuthorization

/-!
Digest parent binding.

This module models the next step after `ParentBoundSource`: replacing an
explicit parent-residue source with a compact digest source. The digest source
is sound only under a parent-digest binding assumption.

The module proves the positive theorem under that assumption and a concrete
negative theorem for an unbinding digest.
-/

namespace DirectCcsFPrime

namespace DigestParentBinding

open DecDigitUniqueness
open GoldilocksChildTableAuthorization

/-- Minimal digest-only source. -/
structure Source (Digest : Type) where
  digest : Digest

/--
The digest source authorizes parent residues whose hash equals the carried
digest.
-/
def BindsParent
    {n : Nat}
    {Digest : Type}
    (hashParent : (Fin n → Nat) → Digest)
    (source : Source Digest)
    (parent : Fin n → Nat) : Prop :=
  source.digest = hashParent parent

/--
The parent-digest binding assumption needed for digest-only sources.

This is the symbolic form of the cryptographic collision-resistance/binding
obligation used by the reduced-source proof. It does not claim that a finite
digest is globally injective as a mathematical function; it is the exact
assumption the protocol needs for accepted parent encodings: the same digest
source cannot authorize two different parent residue vectors.
-/
def ParentDigestBinding
    {n : Nat}
    {Digest : Type}
    (hashParent : (Fin n → Nat) → Digest) : Prop :=
  ∀ parentA parentB,
    hashParent parentA = hashParent parentB →
      parentA = parentB

/-- A digest source binds parents functionally under parent-digest binding. -/
theorem binds_parent_functionally_of_digest_binding
    {n : Nat}
    {Digest : Type}
    {hashParent : (Fin n → Nat) → Digest}
    (hBinding : ParentDigestBinding hashParent) :
    SourceBindsParentFunctionally (BindsParent hashParent) := by
  intro source parentA parentB hA hB
  apply hBinding
  exact hA.symm.trans hB

/--
Accepted Goldilocks child-table authorizations for the same digest source have
equal next inputs under parent-digest binding.
-/
theorem same_digest_source_next_inputs
    {n : Nat}
    {Digest Proof : Type}
    {hashParent : (Fin n → Nat) → Digest}
    {Verify :
      Source Digest → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source Digest}
    {parentA parentB : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (hBinding : ParentDigestBinding hashParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedGoldilocksChildTable
        (BindsParent hashParent)
        Verify
        source
        parentA
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedGoldilocksChildTable
        (BindsParent hashParent)
        Verify
        source
        parentB
        childrenB
        nextB
        proofB) :
    nextA = nextB := by
  exact
    same_functionally_bound_source_next_inputs
      (binds_parent_functionally_of_digest_binding hBinding)
      hSound
      hA
      hB

/--
Deterministic challenges over the same digest source cannot authorize different
next inputs under parent-digest binding.
-/
theorem same_digest_source_challenge_and_inputs
    {n : Nat}
    {Digest Proof Challenge : Type}
    {hashParent : (Fin n → Nat) → Digest}
    {Verify :
      Source Digest → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source Digest}
    {parentA parentB : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (challenge : Source Digest → Challenge)
    (hBinding : ParentDigestBinding hashParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedGoldilocksChildTable
        (BindsParent hashParent)
        Verify
        source
        parentA
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedGoldilocksChildTable
        (BindsParent hashParent)
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
      (binds_parent_functionally_of_digest_binding hBinding)
      hSound
      hA
      hB

private def parentZero : Fin 1 → Nat :=
  fun _ => 0

private def parentOne : Fin 1 → Nat :=
  fun _ => 1

private def constantHash (_parent : Fin 1 → Nat) : Unit :=
  ()

private def constantSource : Source Unit :=
  { digest := () }

/--
An unbinding digest cannot be treated as a functional parent binding.

This is the digest-only analogue of the reduced-source counterexample: a
constant digest authorizes both parent `0` and parent `1`.
-/
theorem constant_digest_parent_binding_not_functional :
    ¬ SourceBindsParentFunctionally (BindsParent constantHash) := by
  intro hFunctional
  have hParents :
      parentZero = parentOne :=
    hFunctional
      constantSource
      parentZero
      parentOne
      rfl
      rfl
  have hCoeff := congrFun hParents ⟨0, by decide⟩
  simp [parentZero, parentOne] at hCoeff

end DigestParentBinding

end DirectCcsFPrime
