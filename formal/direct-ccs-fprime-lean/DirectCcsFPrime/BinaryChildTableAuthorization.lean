import DirectCcsFPrime.DecDigitUniqueness

/-!
Binary child-table authorization.

This module gives a lower-level arithmetic target for the reduced-handle
strategy: a verifier may authorize hidden children by proving a fixed-length
binary digit table that recomposes to the public parent coefficient vector.
-/

namespace DirectCcsFPrime

namespace BinaryChildTableAuthorization

open DecDigitUniqueness

/-- All coefficient columns have the same fixed digit length. -/
def fixedColumnLength {n : Nat} (k : Nat) (cols : ColumnDigits n) : Prop :=
  ∀ j, (cols j).length = k

/--
Soundness boundary for a proof verifier over a binary child table.

Verification must imply binary digits, fixed length, and column-wise
recomposition to the parent. This is the concrete arithmetic shape a
sumcheck/table proof would need to establish before the reduced transcript
source can replace hashing full `CE(b)^k` children.
-/
def BinaryChildTableProofSound
    {n : Nat}
    {Source Proof : Type}
    (Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop)
    (k : Nat)
    (source : Source)
    (parent : Fin n → Nat) : Prop :=
  ∀ children proof,
    Verify source parent children proof →
      binaryColumnDigits children ∧
      fixedColumnLength k children ∧
      recomposeColumns children = parent

/-- Accepted authorization for a reduced source and hidden binary child table. -/
structure AcceptedBinaryChildTable
    {n : Nat}
    {Source Proof : Type}
    (SourceBindsParent : Source → (Fin n → Nat) → Prop)
    (Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop)
    (source : Source)
    (parent : Fin n → Nat)
    (children nextInputs : ColumnDigits n)
    (proof : Proof) : Prop where
  sourceBound : SourceBindsParent source parent
  proofVerified : Verify source parent children proof
  wireIdentity : nextInputs = children

/--
Same reduced source and parent imply same next child table when the verifier
forces a fixed-length binary recomposition table.
-/
theorem same_binary_child_table_next_inputs
    {n : Nat}
    {Source Proof : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {k : Nat}
    {source : Source}
    {parent : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (hSound :
      BinaryChildTableProofSound Verify k source parent)
    (hA :
      AcceptedBinaryChildTable
        SourceBindsParent
        Verify
        source
        parent
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedBinaryChildTable
        SourceBindsParent
        Verify
        source
        parent
        childrenB
        nextB
        proofB) :
    nextA = nextB := by
  have hAProof := hSound childrenA proofA hA.proofVerified
  have hBProof := hSound childrenB proofB hB.proofVerified
  have hLen : sameColumnLengths childrenA childrenB := by
    intro j
    calc
      (childrenA j).length = k := hAProof.2.1 j
      _ = (childrenB j).length := (hBProof.2.1 j).symm
  have hChildren :
      childrenA = childrenB :=
    binary_column_authorization_unique
      hLen
      hAProof.1
      hBProof.1
      hAProof.2.2
      hBProof.2.2
  calc
    nextA = childrenA := hA.wireIdentity
    _ = childrenB := hChildren
    _ = nextB := hB.wireIdentity.symm

/--
The deterministic challenge over a reduced source is not affected by attempted
hidden-child changes once accepted hidden child tables are unique.
-/
theorem same_binary_child_table_challenge_and_inputs
    {n : Nat}
    {Source Proof Challenge : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {k : Nat}
    {source : Source}
    {parent : Fin n → Nat}
    {childrenA childrenB nextA nextB : ColumnDigits n}
    {proofA proofB : Proof}
    (challenge : Source → Challenge)
    (hSound :
      BinaryChildTableProofSound Verify k source parent)
    (hA :
      AcceptedBinaryChildTable
        SourceBindsParent
        Verify
        source
        parent
        childrenA
        nextA
        proofA)
    (hB :
      AcceptedBinaryChildTable
        SourceBindsParent
        Verify
        source
        parent
        childrenB
        nextB
        proofB) :
    challenge source = challenge source ∧ nextA = nextB := by
  exact
    ⟨rfl,
      same_binary_child_table_next_inputs
        hSound
        hA
        hB⟩

end BinaryChildTableAuthorization

end DirectCcsFPrime
