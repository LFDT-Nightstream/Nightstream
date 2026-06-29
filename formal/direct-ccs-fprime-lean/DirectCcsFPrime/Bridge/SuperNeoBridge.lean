import DirectCcsFPrime.Commitment.Parent.Impl.AjtaiResidueBinding
import DirectCcsFPrime.ProofSystem.PrivatePiDec.Security.GoldilocksChildTableAuthorization
import DirectCcsFPrime.Commitment.Parent.Spec.ParentEncoding
import DirectCcsFPrime.Commitment.Parent.Security.ParentOpeningAuthorization
import SuperNeo.FoldingProtocol.PiCCSInterface
import SuperNeo.FoldingProtocol.PiDECInterface
import SuperNeo.FoldingProtocol.PiRLCInterface
import SuperNeo.ProofSystem.ConstraintSystem.CCS
import SuperNeo.ProofSystem.Lattice

/-!
Bridge from the direct-CCS reduced-handle proof to the existing SuperNeo Lean
surfaces.

This file intentionally does not redefine CE, Ajtai commitments, `Π_CCS`,
`Π_RLC`, or `Π_DEC`. It packages the local reduced-handle theorem around the
existing SuperNeo relation and stage interfaces.
-/

namespace DirectCcsFPrime

namespace SuperNeoBridge

open DecDigitUniqueness
open BinaryChildTableAuthorization
open GoldilocksChildTableAuthorization

/-- Convert a SuperNeo base-field element to its canonical natural residue. -/
def fieldDigit (x : SuperNeo.F) : Nat :=
  x.val

/-- Extract one flattened coefficient from a SuperNeo CE witness assignment. -/
def witnessCoeffDigit
    (wit : SuperNeo.ProofSystem.ConstraintSystem.CE.Witness)
    (j : Nat) : Nat :=
  fieldDigit (SuperNeo.coeffAt wit.assignment j)

/--
Extract the child digit table from `k` SuperNeo CE witnesses.

Rows are the `k` low-norm DEC children. Columns are flattened witness
coefficient positions. This is the concrete wire table that the reduced-handle
proof must authorize before the next `Π_CCS` transcript consumes it.
-/
def childWitnessDigitTable
    {k n : Nat}
    (witness :
      Fin k → SuperNeo.ProofSystem.ConstraintSystem.CE.Witness) :
    ColumnDigits n :=
  fun j =>
    List.ofFn
      (fun i : Fin k => witnessCoeffDigit (witness i) j.1)

/--
One child accumulator bundle backed by the existing SuperNeo CE relation and
Ajtai opening relation.

`digitTable` is the implementation-facing binary child table proved by the
reduced-handle DEC authorization. `nextPiCCSInputs` are the exact child wires
consumed by the next `Π_CCS` verifier. The bridge theorem below requires those
two objects to be wire-identical through the accepted proof object.
-/
structure AjtaiCEChildBundle (k n : Nat) where
  ce :
    SuperNeo.ProofSystem.ConstraintSystem.CE
      SuperNeo.ProofSystem.Commitment
  ajtaiParams : SuperNeo.ProofSystem.AjtaiParams
  statement :
    Fin k →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Statement
        SuperNeo.ProofSystem.Commitment
  witness :
    Fin k →
      SuperNeo.ProofSystem.ConstraintSystem.CE.Witness
  opening : Fin k → SuperNeo.ProofSystem.Opening
  ceHolds :
    ∀ i,
      SuperNeo.ProofSystem.ConstraintSystem.CE.Holds
        ce
        (statement i)
        (witness i)
  ajtaiOpens :
    ∀ i,
      SuperNeo.ProofSystem.opensTo
        ajtaiParams
        (statement i).commitment
        (opening i)
  /--
  Child table checked by the reduced-handle DEC proof.

  This table is not free advice: `digitTableMatchesWitnesses` below ties it to
  the concrete extraction from the imported SuperNeo CE witnesses.
  -/
  digitTable : ColumnDigits n
  digitTableMatchesWitnesses :
    digitTable = childWitnessDigitTable (k := k) (n := n) witness
  nextPiCCSInputs : ColumnDigits n

namespace AjtaiCEChildBundle

/-- Recover commitment consistency from the imported SuperNeo CE relation. -/
theorem commitment_eq_commitMap
    {k n : Nat}
    (bundle : AjtaiCEChildBundle k n)
    (i : Fin k) :
    (bundle.statement i).commitment =
      bundle.ce.commitMap (bundle.witness i).assignment := by
  exact (bundle.ceHolds i).1

/-- Recover public input projection consistency from the imported CE relation. -/
theorem publicInput_eq_projector
    {k n : Nat}
    (bundle : AjtaiCEChildBundle k n)
    (i : Fin k) :
    (bundle.statement i).publicInput =
      bundle.ce.inputProjector (bundle.witness i).assignment := by
  exact (bundle.ceHolds i).2.1

/-- Recover the low-norm witness bound from the imported CE relation. -/
theorem witness_norm_bound
    {k n : Nat}
    (bundle : AjtaiCEChildBundle k n)
    (i : Fin k) :
    SuperNeo.normInfCoeffs (bundle.witness i).assignment <
      bundle.ce.normBound := by
  exact (bundle.ceHolds i).2.2.1

/-- Recover evaluation consistency from the imported CE relation. -/
theorem evaluations_eq
    {k n : Nat}
    (bundle : AjtaiCEChildBundle k n)
    (i : Fin k) :
    (bundle.statement i).evaluations =
      bundle.ce.shape.evaluationFamily
        (bundle.witness i).assignment
        (bundle.statement i).point := by
  exact (bundle.ceHolds i).2.2.2

/-- The reduced-handle child table is extracted from CE witnesses. -/
theorem digitTable_eq_childWitnessDigitTable
    {k n : Nat}
    (bundle : AjtaiCEChildBundle k n) :
    bundle.digitTable =
      childWitnessDigitTable (k := k) (n := n) bundle.witness :=
  bundle.digitTableMatchesWitnesses

/-- Column `j` of the witness-derived table contains exactly `k` child digits. -/
theorem childWitnessDigitTable_length
    {k n : Nat}
    (bundle : AjtaiCEChildBundle k n)
    (j : Fin n) :
    ((childWitnessDigitTable (k := k) (n := n) bundle.witness) j).length = k := by
  simp [childWitnessDigitTable]

end AjtaiCEChildBundle

/--
Accepted reduced-handle authorization for a SuperNeo-backed child CE bundle.

The proof verifier is still abstract here: this bridge only says what its
accepted proof must connect. Its soundness is supplied separately as
`GoldilocksChildTableProofSound`, which is the existing implementation-facing
Goldilocks/no-wrap theorem boundary.
-/
structure AcceptedSuperNeoReducedHandle
    {n : Nat}
    {Source Proof : Type}
    (SourceBindsParent : Source → (Fin n → Nat) → Prop)
    (Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop)
    (source : Source)
    (parentResidues : Fin n → Nat)
    (bundle : AjtaiCEChildBundle 14 n)
    (proof : Proof) : Prop where
  acceptedChildTable :
    AcceptedGoldilocksChildTable
      SourceBindsParent
      Verify
      source
      parentResidues
      bundle.digitTable
      bundle.nextPiCCSInputs
      proof

/--
Bridge theorem for the reduced CE(B)-handle strategy.

If the reduced source functionally binds parent residues, and the checked proof
establishes the Goldilocks child-table obligations, then two accepted
SuperNeo-backed child CE bundles for the same source must feed identical next
`Π_CCS` child inputs.

The imported CE/Ajtai fields in `AjtaiCEChildBundle` make this theorem a bridge
over the existing SuperNeo formal surfaces rather than a parallel CE model.
-/
theorem same_source_superneo_next_inputs
    {n : Nat}
    {Source Proof : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hBind :
      SourceBindsParentFunctionally SourceBindsParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        SourceBindsParent
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        SourceBindsParent
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_functionally_bound_source_next_inputs
      hBind
      hSound
      hA.acceptedChildTable
      hB.acceptedChildTable

/--
Same source also forces equality of the child tables extracted from real CE
witnesses, not only equality of a standalone `nextPiCCSInputs` field.
-/
theorem same_source_superneo_witness_digits
    {n : Nat}
    {Source Proof : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hBind :
      SourceBindsParentFunctionally SourceBindsParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        SourceBindsParent
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        SourceBindsParent
        Verify
        source
        parentB
        bundleB
        proofB) :
    childWitnessDigitTable (k := 14) (n := n) bundleA.witness =
      childWitnessDigitTable (k := 14) (n := n) bundleB.witness := by
  have hNext :
      bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs :=
    same_source_superneo_next_inputs hBind hSound hA hB
  calc
    childWitnessDigitTable (k := 14) (n := n) bundleA.witness =
        bundleA.digitTable :=
      bundleA.digitTableMatchesWitnesses.symm
    _ = bundleA.nextPiCCSInputs :=
      hA.acceptedChildTable.wireIdentity.symm
    _ = bundleB.nextPiCCSInputs :=
      hNext
    _ = bundleB.digitTable :=
      hB.acceptedChildTable.wireIdentity
    _ = childWitnessDigitTable (k := 14) (n := n) bundleB.witness :=
      bundleB.digitTableMatchesWitnesses

/--
Accepted reduced-handle proof obligations transfer to the actual CE-witness
digit table.
-/
theorem accepted_witness_digit_table_sound
    {n : Nat}
    {Source Proof : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source}
    {parent : Fin n → Nat}
    {bundle : AjtaiCEChildBundle 14 n}
    {proof : Proof}
    (hSound :
      GoldilocksChildTableProofSound Verify source parent)
    (h :
      AcceptedSuperNeoReducedHandle
        SourceBindsParent
        Verify
        source
        parent
        bundle
        proof) :
    binaryColumnDigits
        (childWitnessDigitTable (k := 14) (n := n) bundle.witness) ∧
      fixedColumnLength
        14
        (childWitnessDigitTable (k := 14) (n := n) bundle.witness) ∧
      (∀ j,
        recomposeNatDigits
            ((childWitnessDigitTable (k := 14) (n := n) bundle.witness) j) %
            SuperNeo.Goldilocks.q =
          parent j % SuperNeo.Goldilocks.q) := by
  have hProof := hSound bundle.digitTable proof h.acceptedChildTable.proofVerified
  simpa [h.acceptedChildTable.wireIdentity, bundle.digitTableMatchesWitnesses] using hProof

/--
Same result with deterministic challenge equality included.

This is a wiring theorem, not a random-oracle theorem: the challenge function is
deterministic over the reduced source, and the hidden children accepted for
that source are unique under the proof obligations.
-/
theorem same_source_superneo_challenge_and_inputs
    {n : Nat}
    {Source Proof Challenge : Type}
    {SourceBindsParent : Source → (Fin n → Nat) → Prop}
    {Verify :
      Source → (Fin n → Nat) → ColumnDigits n → Proof → Prop}
    {source : Source}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (challenge : Source → Challenge)
    (hBind :
      SourceBindsParentFunctionally SourceBindsParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        SourceBindsParent
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        SourceBindsParent
        Verify
        source
        parentB
        bundleB
        proofB) :
    challenge source = challenge source ∧
      bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    ⟨rfl,
      same_source_superneo_next_inputs
        hBind
        hSound
        hA
        hB⟩

/--
Digest-source specialization of `same_source_superneo_next_inputs`.

This theorem is deliberately conditional: a digest can stand in for the parent
residue source only under an explicit parent-digest binding assumption at this
formal boundary. This is the symbolic form of the hash
collision-resistance/binding assumption for accepted parent encodings.
-/
theorem same_digest_source_superneo_next_inputs
    {n : Nat}
    {Digest Proof : Type}
    {hashParent : (Fin n → Nat) → Digest}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hBinding : DigestParentBinding.ParentDigestBinding hashParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (DigestParentBinding.BindsParent hashParent)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (DigestParentBinding.BindsParent hashParent)
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_source_superneo_next_inputs
      (DigestParentBinding.binds_parent_functionally_of_digest_binding hBinding)
      hSound
      hA
      hB

/--
Digest-source specialization for actual CE witness digits.

Under parent-digest binding and a sound child-table proof, two accepted
SuperNeo-backed bundles for the same digest source extract the same child
digits from their CE witnesses.
-/
theorem same_digest_source_superneo_witness_digits
    {n : Nat}
    {Digest Proof : Type}
    {hashParent : (Fin n → Nat) → Digest}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hBinding : DigestParentBinding.ParentDigestBinding hashParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (DigestParentBinding.BindsParent hashParent)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (DigestParentBinding.BindsParent hashParent)
        Verify
        source
        parentB
        bundleB
        proofB) :
    childWitnessDigitTable (k := 14) (n := n) bundleA.witness =
      childWitnessDigitTable (k := 14) (n := n) bundleB.witness := by
  exact
    same_source_superneo_witness_digits
      (DigestParentBinding.binds_parent_functionally_of_digest_binding hBinding)
      hSound
      hA
      hB

/--
Digest-source specialization with deterministic challenge equality.

The challenge is deterministic over the digest source. The theorem says that,
under parent-digest binding and proof-checked DEC authorization, the hidden
children wired into next `Π_CCS` cannot vary while keeping the same challenge
source.
-/
theorem same_digest_source_superneo_challenge_and_inputs
    {n : Nat}
    {Digest Proof Challenge : Type}
    {hashParent : (Fin n → Nat) → Digest}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (challenge : DigestParentBinding.Source Digest → Challenge)
    (hBinding : DigestParentBinding.ParentDigestBinding hashParent)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (DigestParentBinding.BindsParent hashParent)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (DigestParentBinding.BindsParent hashParent)
        Verify
        source
        parentB
        bundleB
        proofB) :
    challenge source = challenge source ∧
      bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_source_superneo_challenge_and_inputs
      challenge
      (DigestParentBinding.binds_parent_functionally_of_digest_binding hBinding)
      hSound
      hA
      hB

/--
Encoded-parent specialization for the reduced handle used by direct CCS.

This is the preferred theorem shape for a hash-backed implementation:
prove/assume binding over the canonical encoded parent list, then use the
encoding injectivity theorem to satisfy the digest-source parent binding
required by the SuperNeo bridge.
-/
theorem same_encoded_digest_source_superneo_next_inputs
    {n : Nat}
    {Digest Proof : Type}
    {hashEncoded : List Nat → Digest}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hBinding :
      ParentEncoding.EncodedParentResidueDigestBinding
        (n := n)
        hashEncoded)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (DigestParentBinding.BindsParent
          (ParentEncoding.hashEncodedParentResidues
            (n := n)
            hashEncoded))
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (DigestParentBinding.BindsParent
          (ParentEncoding.hashEncodedParentResidues
            (n := n)
            hashEncoded))
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_digest_source_superneo_next_inputs
      (ParentEncoding.parentDigestBinding_of_encodedParentResidueDigestBinding
        hBinding)
      hSound
      hA
      hB

/--
Full-parent `CE(B)` encoded-digest specialization.

This is the theorem shape for the reduced-handle implementation strategy:
the Fiat-Shamir/public source binds one canonical encoded parent `CE(B)` handle;
a deterministic projection extracts the parent residues checked by the private
`Pi_DEC` child-table proof; and the accepted children are wire-identical to the
next `Π_CCS` accumulator inputs.
-/
theorem same_encoded_parentCEB_digest_source_superneo_next_inputs
    {n : Nat}
    {Digest Proof : Type}
    {hashEncoded : List Nat → Digest}
    {project : ParentEncoding.SomeParentCEB → (Fin n → Nat)}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hBinding :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentEncoding.BindsProjectedParentCEBResidues
          (n := n)
          hashEncoded
          project)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentEncoding.BindsProjectedParentCEBResidues
          (n := n)
          hashEncoded
          project)
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_source_superneo_next_inputs
      (ParentEncoding.bindsProjectedParentCEBResidues_functionally_of_encodedDigestBinding
        hBinding)
      hSound
        hA
        hB

/--
Full-parent `CE(B)` encoded-digest specialization for the actual child CE
witness digits.

This is the stronger observable conclusion behind the next-input theorem: the
accepted private child witnesses themselves expose the same binary digit table
when the digest-bound parent handle, projected residues, checked child table,
and next `Π_CCS` wires are all connected.
-/
theorem same_encoded_parentCEB_digest_source_superneo_witness_digits
    {n : Nat}
    {Digest Proof : Type}
    {hashEncoded : List Nat → Digest}
    {project : ParentEncoding.SomeParentCEB → (Fin n → Nat)}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hBinding :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentEncoding.BindsProjectedParentCEBResidues
          (n := n)
          hashEncoded
          project)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentEncoding.BindsProjectedParentCEBResidues
          (n := n)
          hashEncoded
          project)
        Verify
        source
        parentB
        bundleB
        proofB) :
    childWitnessDigitTable (k := 14) (n := n) bundleA.witness =
      childWitnessDigitTable (k := 14) (n := n) bundleB.witness := by
  exact
    same_source_superneo_witness_digits
      (ParentEncoding.bindsProjectedParentCEBResidues_functionally_of_encodedDigestBinding
        hBinding)
      hSound
      hA
      hB

/--
Full-parent `CE(B)` encoded-digest specialization with deterministic challenge
equality.

The challenge source is the compact digest source, while the child accumulator
fed to the next `Π_CCS` is authorized by the projected parent and private
checked `Pi_DEC` table.
-/
theorem same_encoded_parentCEB_digest_source_superneo_challenge_and_inputs
    {n : Nat}
    {Digest Proof Challenge : Type}
    {hashEncoded : List Nat → Digest}
    {project : ParentEncoding.SomeParentCEB → (Fin n → Nat)}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (challenge : DigestParentBinding.Source Digest → Challenge)
    (hBinding :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentEncoding.BindsProjectedParentCEBResidues
          (n := n)
          hashEncoded
          project)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentEncoding.BindsProjectedParentCEBResidues
          (n := n)
          hashEncoded
          project)
        Verify
        source
        parentB
        bundleB
        proofB) :
    challenge source = challenge source ∧
      bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    ⟨rfl,
      same_encoded_parentCEB_digest_source_superneo_next_inputs
        hBinding
        hSound
        hA
        hB⟩

/--
Opening-authorized full-parent `CE(B)` encoded-digest specialization.

This is the stricter theorem shape needed for the real reduced-handle design:
the digest binds the encoded parent handle, and the DEC parent residues are
extracted from an accepted opening witness for that same parent handle.
-/
theorem same_opened_parentCEB_digest_source_superneo_next_inputs
    {n : Nat}
    {Digest Proof Commitment : Type}
    {hashEncoded : List Nat → Digest}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hOpening :
      ParentOpeningAuthorization.EncodedParentCEBOpeningResiduesFunctional
        (n := n)
        StatementEncodes)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResidues
          (n := n)
          hashEncoded
          StatementEncodes)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResidues
          (n := n)
          hashEncoded
          StatementEncodes)
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_source_superneo_next_inputs
      (ParentOpeningAuthorization.bindsOpenedParentCEBResidues_functionally
        hDigest
        hOpening)
      hSound
      hA
      hB

/--
Opening-authorized full-parent `CE(B)` encoded-digest specialization for
actual child CE witness digits.
-/
theorem same_opened_parentCEB_digest_source_superneo_witness_digits
    {n : Nat}
    {Digest Proof Commitment : Type}
    {hashEncoded : List Nat → Digest}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hOpening :
      ParentOpeningAuthorization.EncodedParentCEBOpeningResiduesFunctional
        (n := n)
        StatementEncodes)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResidues
          (n := n)
          hashEncoded
          StatementEncodes)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResidues
          (n := n)
          hashEncoded
          StatementEncodes)
        Verify
        source
        parentB
        bundleB
        proofB) :
    childWitnessDigitTable (k := 14) (n := n) bundleA.witness =
      childWitnessDigitTable (k := 14) (n := n) bundleB.witness := by
  exact
    same_source_superneo_witness_digits
      (ParentOpeningAuthorization.bindsOpenedParentCEBResidues_functionally
        hDigest
        hOpening)
      hSound
      hA
      hB

/--
Opening-authorized full-parent `CE(B)` encoded-digest specialization with
deterministic challenge equality.
-/
theorem same_opened_parentCEB_digest_source_superneo_challenge_and_inputs
    {n : Nat}
    {Digest Proof Commitment Challenge : Type}
    {hashEncoded : List Nat → Digest}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (challenge : DigestParentBinding.Source Digest → Challenge)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hOpening :
      ParentOpeningAuthorization.EncodedParentCEBOpeningResiduesFunctional
        (n := n)
        StatementEncodes)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResidues
          (n := n)
          hashEncoded
          StatementEncodes)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResidues
          (n := n)
          hashEncoded
          StatementEncodes)
        Verify
        source
        parentB
        bundleB
        proofB) :
    challenge source = challenge source ∧
      bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    ⟨rfl,
      same_opened_parentCEB_digest_source_superneo_next_inputs
        hDigest
        hOpening
        hSound
        hA
        hB⟩

/--
Fixed-CE opening-authorized full-parent `CE(B)` encoded-digest specialization.

This is the preferred reduced-handle theorem for a direct CCS fold step: one
fixed parent CE relation, one encoded parent digest source, and private
checked children wired into the next `Π_CCS`.
-/
theorem same_opened_parentCEB_digest_source_superneo_next_inputs_for_ce
    {n : Nat}
    {Digest Proof Commitment : Type}
    {hashEncoded : List Nat → Digest}
    {ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hOpening :
      ParentOpeningAuthorization.EncodedParentCEBOpeningResiduesFunctionalFor
        (n := n)
        ce
        StatementEncodes)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_source_superneo_next_inputs
      (ParentOpeningAuthorization.bindsOpenedParentCEBResiduesFor_functionally
        hDigest
        hOpening)
      hSound
      hA
      hB

/--
Fixed-CE opening-authorized theorem discharged from explicit commitment-map
residue binding.
-/
theorem same_opened_parentCEB_digest_source_superneo_next_inputs_of_commitMapBinding
    {n : Nat}
    {Digest Proof Commitment : Type}
    {hashEncoded : List Nat → Digest}
    {ce : SuperNeo.ProofSystem.ConstraintSystem.CE Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hCommitMap :
      ParentOpeningAuthorization.CommitMapResiduesFunctional
        (n := n)
        ce.commitMap)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_source_superneo_next_inputs
      (ParentOpeningAuthorization.bindsOpenedParentCEBResiduesFor_functionally_of_commitMapBinding
        hDigest
        hEncoding
        hCommitMap)
      hSound
      hA
      hB

/--
Fixed-CE opening-authorized theorem discharged from concrete Ajtai
no-collision plus an adapter from the CE commitment map to `opensTo`.

This is the proof-critical implementation shape for the reduced `CE(B)^1`
source: the parent residues are not arbitrary private advice; they are forced
by bounded Ajtai openings of the same parent commitment.
-/
theorem same_opened_parentCEB_digest_source_superneo_next_inputs_of_ajtaiBinding
    {n : Nat}
    {Digest Proof : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.AssignmentOpeningAdapter
        n
        params
        ce.commitMap)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_opened_parentCEB_digest_source_superneo_next_inputs_of_commitMapBinding
      hDigest
      hEncoding
      (AjtaiResidueBinding.commitMapResiduesFunctional_of_noAjtaiBindingCollision
        hNoCollision
        adapter)
      hSound
      hA
      hB

/--
Preferred concrete Ajtai-backed theorem for implementation use.

This variant only requires an Ajtai adapter for CE openings that actually
satisfy `CE.Holds`, rather than a global commitment-map binding theorem over
all possible assignments.
-/
theorem same_opened_parentCEB_digest_source_superneo_next_inputs_of_ajtaiCEOpening
    {n : Nat}
    {Digest Proof : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {StatementEncodes :
      ParentOpeningAuthorization.StatementEncodesParentCEB
        SuperNeo.ProofSystem.Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hEncoding :
      ParentOpeningAuthorization.StatementEncodingCommitmentFunctional
        StatementEncodes)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          StatementEncodes)
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_opened_parentCEB_digest_source_superneo_next_inputs_for_ce
      hDigest
      (AjtaiResidueBinding.encodedParentCEBOpeningResiduesFunctionalFor_of_noAjtaiBindingCollision
        hEncoding
        hNoCollision
        adapter)
      hSound
      hA
      hB

/--
Implementation-shaped fixed-CE opened-parent theorem.

This is the direct reduced-handle bridge used by the terminal proof path: the
parent statement encoder is the deterministic commitment encoder
`StatementEncodesByCommitment`, so serializer consistency is discharged inside
Lean rather than passed as an extra premise.
-/
theorem same_opened_parentCEB_digest_source_superneo_next_inputs_of_statementCommitment_and_ajtaiCEOpening
    {n : Nat}
    {Digest Proof : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        Verify
        source
        parentB
        bundleB
        proofB) :
    bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    same_opened_parentCEB_digest_source_superneo_next_inputs_of_ajtaiCEOpening
      hDigest
      ParentOpeningAuthorization.statementEncodesByCommitment_functional
      hNoCollision
      adapter
      hSound
      hA
      hB

/--
Implementation-shaped fixed-CE opened-parent theorem with deterministic
challenge equality.

The challenge is deterministic over the compact digest source. The nontrivial
part is that the accepted private children wired into next `Π_CCS` are fixed by
the opened parent handle and Ajtai opening binding.
-/
theorem same_opened_parentCEB_digest_source_superneo_challenge_and_inputs_of_statementCommitment_and_ajtaiCEOpening
    {n : Nat}
    {Digest Proof Challenge : Type}
    {hashEncoded : List Nat → Digest}
    {params : SuperNeo.ProofSystem.AjtaiParams}
    {ce :
      SuperNeo.ProofSystem.ConstraintSystem.CE
        SuperNeo.ProofSystem.Commitment}
    {commitmentOfParent :
      ParentEncoding.SomeParentCEB →
        SuperNeo.ProofSystem.Commitment}
    {Verify :
      DigestParentBinding.Source Digest →
        (Fin n → Nat) →
        ColumnDigits n →
        Proof →
        Prop}
    {source : DigestParentBinding.Source Digest}
    {parentA parentB : Fin n → Nat}
    {bundleA bundleB : AjtaiCEChildBundle 14 n}
    {proofA proofB : Proof}
    (challenge : DigestParentBinding.Source Digest → Challenge)
    (hDigest :
      ParentEncoding.EncodedParentCEBDigestBinding hashEncoded)
    (hNoCollision :
      AjtaiResidueBinding.NoAjtaiBindingCollision params)
    (adapter :
      AjtaiResidueBinding.CEOpeningAdapter
        n
        params
        ce)
    (hSound :
      GoldilocksChildTableProofSound Verify source parentA)
    (hA :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        Verify
        source
        parentA
        bundleA
        proofA)
    (hB :
      AcceptedSuperNeoReducedHandle
        (ParentOpeningAuthorization.BindsOpenedParentCEBResiduesFor
          (n := n)
          hashEncoded
          ce
          (ParentOpeningAuthorization.StatementEncodesByCommitment
            commitmentOfParent))
        Verify
        source
        parentB
        bundleB
        proofB) :
    challenge source = challenge source ∧
      bundleA.nextPiCCSInputs = bundleB.nextPiCCSInputs := by
  exact
    ⟨rfl,
      same_opened_parentCEB_digest_source_superneo_next_inputs_of_statementCommitment_and_ajtaiCEOpening
        hDigest
        hNoCollision
        adapter
        hSound
        hA
        hB⟩

/--
Existing SuperNeo stage authority reused by direct CCS.

This is the narrow adapter for the already-formalized protocol reductions: a
single `ceRelation ctx` can be handed to the imported theorem surfaces for
`Π_CCS`, `Π_RLC`, and `Π_DEC`.
-/
structure ReusedStageAuthority (ctx : SuperNeo.ProtocolTargetContext) : Prop where
  ceRelation : SuperNeo.ceRelation ctx

namespace ReusedStageAuthority

/-- Reuse the existing `Π_CCS` theorem surface. -/
theorem piCCSStrong
    {ctx : SuperNeo.ProtocolTargetContext}
    (h : ReusedStageAuthority ctx) :
    SuperNeo.PiCCSInterface.piCCSStrongStatement ctx :=
  SuperNeo.PiCCSInterface.piCCSStrong_of_ce h.ceRelation

/-- Reuse the existing `Π_RLC` theorem surface. -/
theorem piRLCWeak
    {ctx : SuperNeo.ProtocolTargetContext}
    (h : ReusedStageAuthority ctx) :
    SuperNeo.PiRLCInterface.piRLCWeakStatement ctx :=
  SuperNeo.PiRLCInterface.piRLCWeak_of_ce h.ceRelation

/-- Reuse the existing `Π_DEC` theorem surface. -/
theorem piDECKnowledge
    {ctx : SuperNeo.ProtocolTargetContext}
    (h : ReusedStageAuthority ctx) :
    SuperNeo.PiDECInterface.piDECKnowledgeStatement ctx :=
  SuperNeo.PiDECInterface.piDEC_of_ce h.ceRelation

end ReusedStageAuthority

end SuperNeoBridge

end DirectCcsFPrime
