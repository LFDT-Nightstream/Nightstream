import DirectCcsFPrime.ProofSystem.PrivatePiDec.Security.BinaryChildTableAuthorization
import DirectCcsFPrime.ProofSystem.PrivatePiDec.Impl.GoldilocksNoWrap

/-!
Goldilocks child-table authorization.

This module models the implementation-facing DEC proof boundary where the
verifier checks recomposition as equality modulo the Goldilocks field. It proves
that this is enough to authorize a unique hidden child table only when the proof
also establishes exact-length binary columns for the concrete `k_dec = 14`
profile.
-/

namespace DirectCcsFPrime

namespace GoldilocksChildTableAuthorization

open DecDigitUniqueness
open BinaryChildTableAuthorization
open GoldilocksNoWrap

/--
Soundness boundary for a Goldilocks-field child-table verifier.

Verification must imply:

- binary child digits,
- exact `k_dec = 14` columns,
- recomposition to the parent residues modulo the Goldilocks field.

The exact length and bitness are what let `GoldilocksNoWrap` lift modular
recomposition equality back to integer child-table equality.
-/
def GoldilocksChildTableProofSound
    {n : Nat}
    {Source Proof : Type}
    (Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop)
    (source : Source)
    (parentResidues : Fin n → Nat) : Prop :=
  ∀ children proof,
    Verify source parentResidues children proof →
      binaryColumnDigits children ∧
      fixedColumnLength 14 children ∧
      (∀ j,
        recomposeNatDigits (children j) % SuperNeo.Goldilocks.q =
        parentResidues j % SuperNeo.Goldilocks.q)

/--
Accepted authorization for a reduced source and hidden Goldilocks child table.
-/
structure AcceptedGoldilocksChildTable
    {n : Nat}
    {Source Proof : Type}
    (SourceBindsParent : Source → (Fin n → Nat) → Prop)
    (Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop)
    (source : Source)
    (parentResidues : Fin n → Nat)
    (children nextInputs : ColumnDigits n)
    (proof : Proof) : Prop where
  sourceBound : SourceBindsParent source parentResidues
  proofVerified : Verify source parentResidues children proof
  wireIdentity : nextInputs = children

/--
The reduced source binds parent residues functionally.

This is the positive form of the parent-binding requirement: for a fixed source
there is at most one parent-residue vector it can authorize.
-/
def SourceBindsParentFunctionally
    {n : Nat}
    {Source : Type}
    (SourceBindsParent : Source → (Fin n → Nat) → Prop) : Prop :=
  ∀ source parentA parentB,
    SourceBindsParent source parentA →
    SourceBindsParent source parentB →
      parentA = parentB

/--
Same reduced source and parent residues imply the same next child table when
the verifier proves exact-length binary Goldilocks modular recomposition.
-/
theorem same_goldilocks_child_table_next_inputs
    {n : Nat}
    {Source Proof : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source}
    {parentResidues : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (hSound :
      GoldilocksChildTableProofSound Verify source parentResidues)
    (hA :
      AcceptedGoldilocksChildTable
        SourceBindsParent
        Verify
        source
        parentResidues
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedGoldilocksChildTable
        SourceBindsParent
        Verify
        source
        parentResidues
        childrenB
        nextB
        proofB) :
    nextA = nextB := by
  have hAProof := hSound childrenA proofA hA.proofVerified
  have hBProof := hSound childrenB proofB hB.proofVerified
  have hMod :
      ∀ j,
        recomposeNatDigits (childrenA j) % SuperNeo.Goldilocks.q =
        recomposeNatDigits (childrenB j) % SuperNeo.Goldilocks.q := by
    intro j
    calc
      recomposeNatDigits (childrenA j) % SuperNeo.Goldilocks.q =
          parentResidues j % SuperNeo.Goldilocks.q := hAProof.2.2 j
      _ = recomposeNatDigits (childrenB j) % SuperNeo.Goldilocks.q :=
          (hBProof.2.2 j).symm
  have hChildren :
      childrenA = childrenB :=
    binary_column_length14_unique_of_goldilocks_mod_eq
      hAProof.2.1
      hBProof.2.1
      hAProof.1
      hBProof.1
      hMod
  calc
    nextA = childrenA := hA.wireIdentity
    _ = childrenB := hChildren
    _ = nextB := hB.wireIdentity.symm

/--
For deterministic challenges over the reduced source, accepted Goldilocks
child-table authorizations with the same source and parent residues have the
same challenge and the same hidden next accumulator.

This is a wiring/uniqueness theorem. It does not assert random-oracle security.
-/
theorem same_goldilocks_child_table_challenge_and_inputs
    {n : Nat}
    {Source Proof Challenge : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source}
    {parentResidues : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (challenge : Source → Challenge)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentResidues)
    (hA :
      AcceptedGoldilocksChildTable
        SourceBindsParent
        Verify
        source
        parentResidues
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedGoldilocksChildTable
        SourceBindsParent
        Verify
        source
        parentResidues
        childrenB
        nextB
        proofB) :
    challenge source = challenge source ∧ nextA = nextB := by
  exact
    ⟨rfl,
      same_goldilocks_child_table_next_inputs
        hSound
        hA
        hB⟩

/--
Same reduced source implies the same next child table when:

- the source binds parent residues functionally;
- the verifier proves exact-length binary Goldilocks modular recomposition;
- the next `Pi_CCS` inputs are wire-identical to the proved children.

This is the implementation-facing reduced-source theorem: callers do not need
to separately assume equal parent residues if the source binding is functional.
-/
theorem same_functionally_bound_source_next_inputs
    {n : Nat}
    {Source Proof : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source}
    {parentA parentB : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (hBind :
      SourceBindsParentFunctionally SourceBindsParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedGoldilocksChildTable
        SourceBindsParent
        Verify
        source
        parentA
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedGoldilocksChildTable
        SourceBindsParent
        Verify
        source
        parentB
        childrenB
        nextB
        proofB) :
    nextA = nextB := by
  have hParent : parentA = parentB :=
    hBind source parentA parentB hA.sourceBound hB.sourceBound
  subst parentB
  exact same_goldilocks_child_table_next_inputs hSound hA hB

/--
For deterministic challenges over a functionally parent-bound source, accepted
Goldilocks child-table authorizations with the same source have both the same
challenge and the same hidden next accumulator.

This is still a wiring/uniqueness theorem, not a random-oracle theorem.
-/
theorem same_functionally_bound_source_challenge_and_inputs
    {n : Nat}
    {Source Proof Challenge : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source}
    {parentA parentB : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (challenge : Source → Challenge)
    (hBind :
      SourceBindsParentFunctionally SourceBindsParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedGoldilocksChildTable
        SourceBindsParent
        Verify
        source
        parentA
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedGoldilocksChildTable
        SourceBindsParent
        Verify
        source
        parentB
        childrenB
        nextB
        proofB) :
    challenge source = challenge source ∧ nextA = nextB := by
  exact
    ⟨rfl,
      same_functionally_bound_source_next_inputs
        hBind
        hSound
        hA
        hB⟩

end GoldilocksChildTableAuthorization

end DirectCcsFPrime
