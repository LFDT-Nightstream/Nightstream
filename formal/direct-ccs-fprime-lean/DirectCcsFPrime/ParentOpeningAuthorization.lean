import DirectCcsFPrime.ParentEncoding
import SuperNeo.ProofSystem.ConstraintSystem.CCS

/-!
Parent opening authorization for reduced `CE(B)` handles.

Hashing one parent `CE(B)` is not enough by itself to authorize DEC child
residues. The residues used by `Pi_DEC` must come from an opening witness for
that same encoded parent handle. This module isolates that obligation.
-/

namespace DirectCcsFPrime

namespace ParentOpeningAuthorization

/-- Extract one canonical field residue from a parent CE witness. -/
def witnessResidue
    (wit : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness)
    (j : Nat) : Nat :=
  (SuperNeo.coeffAt wit.assignment j).val

/-- Extract the parent residue vector checked by private `Pi_DEC`. -/
def witnessResidues
    {n : Nat}
    (wit : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness) :
    Fin n → Nat :=
  fun j => witnessResidue wit j.1

/-- Extract residues directly from a CE assignment vector. -/
def assignmentResidues {n : Nat} (assignment : SuperNeo.Coeffs) :
    Fin n → Nat :=
  fun j => (SuperNeo.coeffAt assignment j.1).val

theorem witnessResidues_eq_assignmentResidues
    {n : Nat}
    (wit : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness) :
    witnessResidues (n := n) wit = assignmentResidues wit.assignment := by
  rfl

/--
The implementation-facing relation saying a public SuperNeo CE statement is
the statement encoded into a flattened parent `CE(B)` handle.

Commitments and ring/extension elements are abstract in the imported SuperNeo
Lean facade, so the concrete serializer is intentionally supplied as a
predicate. The protocol proof may use it only together with canonical parent
encoding and opening-residue functional binding below.
-/
abbrev StatementEncodesParentCEB (Commitment : Type) :=
  SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment →
    ParentEncoding.SomeParentCEB →
      Prop

/--
Encoding consistency for parent `CE(B)` statements.

If two public CE statements encode the same parent handle, they must at least
carry the same commitment. A concrete serializer should normally prove equality
of the whole public tuple, but commitment equality is the part needed for the
Ajtai/opening binding theorem below.
-/
def StatementEncodingCommitmentFunctional
    {Commitment : Type}
    (StatementEncodes : StatementEncodesParentCEB Commitment) : Prop :=
  ∀
    (parent : ParentEncoding.SomeParentCEB)
    (stmtA stmtB :
      SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment),
      StatementEncodes stmtA parent →
      StatementEncodes stmtB parent →
        stmtA.commitment = stmtB.commitment

/--
Deterministic commitment-facing parent statement encoding.

This is the minimal concrete serializer property needed by the reduced
`CE(B)` theorem: for a fixed encoded parent handle, every accepted CE statement
must carry the same canonical commitment extracted from that parent handle.
-/
def StatementEncodesByCommitment
    {Commitment : Type}
    (commitmentOfParent : ParentEncoding.SomeParentCEB → Commitment) :
    StatementEncodesParentCEB Commitment :=
  fun stmt parent => stmt.commitment = commitmentOfParent parent

/--
The deterministic commitment-facing parent statement encoding satisfies the
serializer consistency premise used by the Ajtai/opening binding theorem.
-/
theorem statementEncodesByCommitment_functional
    {Commitment : Type}
    {commitmentOfParent : ParentEncoding.SomeParentCEB → Commitment} :
    StatementEncodingCommitmentFunctional
      (StatementEncodesByCommitment commitmentOfParent) := by
  intro parent stmtA stmtB hA hB
  exact hA.trans hB.symm

/--
Commitment-level opening binding for DEC parent residues.

This is the theorem-facing Ajtai binding obligation needed by the reduced
handle strategy. If two CE openings satisfy possibly different CE relations but
produce the same commitment, then the witness coefficients consumed by DEC are
the same.
-/
def CommitmentOpeningResiduesFunctional
    {n : Nat}
    {Commitment : Type} : Prop :=
  ∀
    (ceA ceB : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment)
    (stmtA stmtB :
      SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment)
    (witA witB : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness),
      stmtA.commitment = stmtB.commitment →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ceA stmtA witA →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ceB stmtB witB →
        witnessResidues (n := n) witA =
          witnessResidues (n := n) witB

/--
Commitment-map binding at exactly the residue level consumed by `Pi_DEC`.

This is the theorem-facing form of Ajtai binding needed for the reduced
parent-handle argument. It deliberately binds only the residue projection
needed by `Pi_DEC`; a stronger concrete Ajtai theorem may prove full witness
equality and imply this predicate.
-/
def CommitMapResiduesFunctional
    {n : Nat}
    {Commitment : Type}
    (commitMap : SuperNeo.Coeffs → Commitment) : Prop :=
  ∀ assignmentA assignmentB,
    commitMap assignmentA = commitMap assignmentB →
      assignmentResidues (n := n) assignmentA =
        assignmentResidues (n := n) assignmentB

/--
Opening-residue binding for a fixed CE relation.

This is the correct local theory shape for a parent `CE(B)`: both openings are
for the same CE relation/commitment map.
-/
def FixedCEOpeningResiduesFunctional
    {n : Nat}
    {Commitment : Type}
    (ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment) : Prop :=
  ∀
    (stmtA stmtB :
      SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment)
    (witA witB : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness),
      stmtA.commitment = stmtB.commitment →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmtA witA →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmtB witB →
        witnessResidues (n := n) witA =
          witnessResidues (n := n) witB

/--
Commitment-map residue binding implies opening-residue binding for a fixed CE
relation.
-/
theorem fixedCEOpeningResiduesFunctional_of_commitMapBinding
    {n : Nat}
    {Commitment : Type}
    (ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment)
    (hCommitMap :
      CommitMapResiduesFunctional
        (n := n)
        ce.commitMap) :
    FixedCEOpeningResiduesFunctional (n := n) ce := by
  intro stmtA stmtB witA witB hCommitment hHoldsA hHoldsB
  have hMap :
      ce.commitMap witA.assignment = ce.commitMap witB.assignment := by
    calc
      ce.commitMap witA.assignment = stmtA.commitment := hHoldsA.1.symm
      _ = stmtB.commitment := hCommitment
      _ = ce.commitMap witB.assignment := hHoldsB.1
  exact hCommitMap witA.assignment witB.assignment hMap

/--
Opening-residue functional binding for a fixed encoded parent `CE(B)` relation.
-/
def EncodedParentCEBOpeningResiduesFunctionalFor
    {n : Nat}
    {Commitment : Type}
    (ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment)
    (StatementEncodes : StatementEncodesParentCEB Commitment) : Prop :=
  ∀
    (parent : ParentEncoding.SomeParentCEB)
    (stmtA stmtB :
      SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment)
    (witA witB : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness),
      StatementEncodes stmtA parent →
      StatementEncodes stmtB parent →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmtA witA →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmtB witB →
        witnessResidues (n := n) witA =
          witnessResidues (n := n) witB

/--
Statement encoding consistency plus fixed-CE opening binding gives the precise
encoded-parent opening-residue theorem used by the reduced-handle argument.
-/
theorem encodedParentCEBOpeningResiduesFunctionalFor_of_fixedCEBinding
    {n : Nat}
    {Commitment : Type}
    {ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment}
    {StatementEncodes : StatementEncodesParentCEB Commitment}
    (hEncoding : StatementEncodingCommitmentFunctional StatementEncodes)
    (hFixed : FixedCEOpeningResiduesFunctional (n := n) ce) :
    EncodedParentCEBOpeningResiduesFunctionalFor
      (n := n)
      ce
      StatementEncodes := by
  intro parent stmtA stmtB witA witB hEncodeA hEncodeB hHoldsA hHoldsB
  exact
    hFixed
      stmtA
      stmtB
      witA
      witB
      (hEncoding parent stmtA stmtB hEncodeA hEncodeB)
      hHoldsA
      hHoldsB

/--
Opening-residue functional binding for one encoded parent `CE(B)` handle.

This is the exact mathematical obligation behind using one parent digest as the
Fiat-Shamir source: once the encoded parent handle is fixed, any accepted parent
CE opening must yield the same DEC parent residues. A concrete instantiation
should derive this from Ajtai commitment binding plus the CE relation.
-/
def EncodedParentCEBOpeningResiduesFunctional
    {n : Nat}
    {Commitment : Type}
    (StatementEncodes : StatementEncodesParentCEB Commitment) : Prop :=
  ∀
    (parent : ParentEncoding.SomeParentCEB)
    (ceA ceB : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment)
    (stmtA stmtB :
      SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment)
    (witA witB : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness),
      StatementEncodes stmtA parent →
      StatementEncodes stmtB parent →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ceA stmtA witA →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ceB stmtB witB →
        witnessResidues (n := n) witA =
          witnessResidues (n := n) witB

/--
Statement encoding consistency plus commitment-level opening binding implies
opening-residue functional binding for encoded parent handles.

This theorem removes one layer of opacity from the reduced-handle proof: a
concrete implementation may discharge `CommitmentOpeningResiduesFunctional`
from Ajtai binding, while `StatementEncodingCommitmentFunctional` is a
serializer/encoding conformance obligation.
-/
theorem encodedParentCEBOpeningResiduesFunctional_of_commitmentBinding
    {n : Nat}
    {Commitment : Type}
    {StatementEncodes : StatementEncodesParentCEB Commitment}
    (hEncoding : StatementEncodingCommitmentFunctional StatementEncodes)
    (hCommitmentBinding :
      CommitmentOpeningResiduesFunctional
        (n := n)
        (Commitment := Commitment)) :
    EncodedParentCEBOpeningResiduesFunctional
      (n := n)
      StatementEncodes := by
  intro parent ceA ceB stmtA stmtB witA witB
    hEncodeA hEncodeB hHoldsA hHoldsB
  exact
    hCommitmentBinding
      ceA
      ceB
      stmtA
      stmtB
      witA
      witB
      (hEncoding parent stmtA stmtB hEncodeA hEncodeB)
      hHoldsA
      hHoldsB

/--
A digest source authorizes parent residues through a full encoded parent
`CE(B)` handle and an accepted CE opening for that handle.

This rules out treating the residue vector as independent advice: the residues
must be the coefficients extracted from the opened parent witness.
-/
def BindsOpenedParentCEBResidues
    {n : Nat}
    {Digest Commitment : Type}
    (hashEncoded : List Nat → Digest)
    (StatementEncodes : StatementEncodesParentCEB Commitment)
    (source : DigestParentBinding.Source Digest)
    (parentResidues : Fin n → Nat) : Prop :=
  ∃
    (parent : ParentEncoding.SomeParentCEB)
    (ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment)
    (stmt : SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment)
    (wit : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness),
      source.digest =
        ParentEncoding.hashEncodedSomeParentCEB hashEncoded parent ∧
      StatementEncodes stmt parent ∧
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmt wit ∧
      witnessResidues (n := n) wit = parentResidues

/--
Fixed-CE version of `BindsOpenedParentCEBResidues`.

This is the preferred source relation for the parent `CE(B)` reduced-handle
theory: the parent opening is checked against the one fixed CE relation of the
program/fold step.
-/
def BindsOpenedParentCEBResiduesFor
    {n : Nat}
    {Digest Commitment : Type}
    (hashEncoded : List Nat → Digest)
    (ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment)
    (StatementEncodes : StatementEncodesParentCEB Commitment)
    (source : DigestParentBinding.Source Digest)
    (parentResidues : Fin n → Nat) : Prop :=
  ∃
    (parent : ParentEncoding.SomeParentCEB)
    (stmt : SuperNeo.ProofSystem.ConstraintSystem.CE.Statement Commitment)
    (wit : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness),
      source.digest =
        ParentEncoding.hashEncodedSomeParentCEB hashEncoded parent ∧
      StatementEncodes stmt parent ∧
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds ce stmt wit ∧
      witnessResidues (n := n) wit = parentResidues

/--
Canonical full-parent digest binding plus opening-residue functional binding
makes the authorized DEC parent residues functional for a fixed digest source.
-/
theorem bindsOpenedParentCEBResidues_functionally
    {n : Nat}
    {Digest Commitment : Type}
    {hashEncoded : List Nat → Digest}
    {StatementEncodes : StatementEncodesParentCEB Commitment}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hOpening :
      EncodedParentCEBOpeningResiduesFunctional
        (n := n)
        StatementEncodes) :
    GoldilocksChildTableAuthorization.SourceBindsParentFunctionally
      (BindsOpenedParentCEBResidues
        (n := n)
        hashEncoded
        StatementEncodes) := by
  intro source parentResiduesA parentResiduesB hA hB
  rcases hA with
    ⟨parentA, ceA, stmtA, witA, hDigestA, hEncodeA, hHoldsA, hResiduesA⟩
  rcases hB with
    ⟨parentB, ceB, stmtB, witB, hDigestB, hEncodeB, hHoldsB, hResiduesB⟩
  have hHash :
      ParentEncoding.hashEncodedSomeParentCEB hashEncoded parentA =
        ParentEncoding.hashEncodedSomeParentCEB hashEncoded parentB :=
    hDigestA.symm.trans hDigestB
  have hParent : parentA = parentB :=
    ParentEncoding.same_parentCEB_of_encoded_digest_binding hDigest hHash
  subst parentB
  calc
    parentResiduesA = witnessResidues (n := n) witA := hResiduesA.symm
    _ = witnessResidues (n := n) witB :=
      hOpening parentA ceA ceB stmtA stmtB witA witB
        hEncodeA
        hEncodeB
        hHoldsA
        hHoldsB
    _ = parentResiduesB := hResiduesB

/--
Canonical full-parent digest binding plus fixed-CE opening binding makes the
authorized DEC parent residues functional for a fixed digest source.
-/
theorem bindsOpenedParentCEBResiduesFor_functionally
    {n : Nat}
    {Digest Commitment : Type}
    {hashEncoded : List Nat → Digest}
    {ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment}
    {StatementEncodes : StatementEncodesParentCEB Commitment}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hOpening :
      EncodedParentCEBOpeningResiduesFunctionalFor
        (n := n)
        ce
        StatementEncodes) :
    GoldilocksChildTableAuthorization.SourceBindsParentFunctionally
      (BindsOpenedParentCEBResiduesFor
        (n := n)
        hashEncoded
        ce
        StatementEncodes) := by
  intro source parentResiduesA parentResiduesB hA hB
  rcases hA with
    ⟨parentA, stmtA, witA, hDigestA, hEncodeA, hHoldsA, hResiduesA⟩
  rcases hB with
    ⟨parentB, stmtB, witB, hDigestB, hEncodeB, hHoldsB, hResiduesB⟩
  have hHash :
      ParentEncoding.hashEncodedSomeParentCEB hashEncoded parentA =
        ParentEncoding.hashEncodedSomeParentCEB hashEncoded parentB :=
    hDigestA.symm.trans hDigestB
  have hParent : parentA = parentB :=
    ParentEncoding.same_parentCEB_of_encoded_digest_binding hDigest hHash
  subst parentB
  calc
    parentResiduesA = witnessResidues (n := n) witA := hResiduesA.symm
    _ = witnessResidues (n := n) witB :=
      hOpening parentA stmtA stmtB witA witB
        hEncodeA
        hEncodeB
        hHoldsA
        hHoldsB
    _ = parentResiduesB := hResiduesB

/--
End-to-end local parent-opening source binding from explicit components:
encoded parent digest binding, statement commitment encoding, and fixed CE
commitment-map residue binding.
-/
theorem bindsOpenedParentCEBResiduesFor_functionally_of_commitMapBinding
    {n : Nat}
    {Digest Commitment : Type}
    {hashEncoded : List Nat → Digest}
    {ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment}
    {StatementEncodes : StatementEncodesParentCEB Commitment}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      StatementEncodingCommitmentFunctional StatementEncodes)
    (hCommitMap :
      CommitMapResiduesFunctional
        (n := n)
        ce.commitMap) :
    GoldilocksChildTableAuthorization.SourceBindsParentFunctionally
      (BindsOpenedParentCEBResiduesFor
        (n := n)
        hashEncoded
        ce
        StatementEncodes) := by
  exact
    bindsOpenedParentCEBResiduesFor_functionally
      hDigest
      (encodedParentCEBOpeningResiduesFunctionalFor_of_fixedCEBinding
        hEncoding
        (fixedCEOpeningResiduesFunctional_of_commitMapBinding ce hCommitMap))

end ParentOpeningAuthorization

end DirectCcsFPrime
