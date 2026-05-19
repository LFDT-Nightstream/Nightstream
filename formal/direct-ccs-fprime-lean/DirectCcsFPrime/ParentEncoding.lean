import DirectCcsFPrime.DigestParentBinding

/-!
Canonical parent encodings for reduced-handle binding.

This module owns only encoding injectivity. It does not assume or prove any
hash security. The digest-binding modules may instantiate their parent hash as
`hashEncoded (encodeParent...)` and rely on a separate hash-binding
assumption over the encoded lists.
-/

namespace DirectCcsFPrime

namespace ParentEncoding

/-- Encode a fixed-length vector as its entries in index order. -/
def encodeVector {n : Nat} (v : Fin n → Nat) : List Nat :=
  List.ofFn v

/-- `List.ofFn` is injective for fixed-length vectors. -/
theorem encodeVector_injective {n : Nat} :
    Function.Injective (encodeVector (n := n)) := by
  intro a b h
  funext i
  have hget := congrArg (fun xs : List Nat => xs[i.val]?) h
  simp [encodeVector, i.isLt] at hget
  exact hget

/-- Split equal lists whose first segment is a fixed-length encoded vector. -/
theorem encodeVector_append_inj
    {n : Nat}
    {headA headB : Fin n → Nat}
    {tailA tailB : List Nat}
    (h : encodeVector headA ++ tailA = encodeVector headB ++ tailB) :
    headA = headB ∧ tailA = tailB := by
  have hSplit :
      encodeVector headA = encodeVector headB ∧ tailA = tailB :=
    List.append_inj h (by simp [encodeVector])
  exact ⟨encodeVector_injective hSplit.1, hSplit.2⟩

/--
Canonical encoding for the parent residue vector used by the reduced-handle DEC
theorem.

The leading length tag prevents silent shape erasure at the byte/field
encoding boundary.
-/
def encodeParentResidues {n : Nat} (parent : Fin n → Nat) : List Nat :=
  n :: encodeVector parent

/-- Parent residue encoding is injective for a fixed shape. -/
theorem encodeParentResidues_injective {n : Nat} :
    Function.Injective (encodeParentResidues (n := n)) := by
  intro parentA parentB h
  have hTail : encodeVector parentA = encodeVector parentB := by
    simpa [encodeParentResidues] using h
  exact encodeVector_injective hTail

/-- Hash a parent through its canonical residue encoding. -/
def hashEncodedParentResidues
    {n : Nat}
    {Digest : Type}
    (hashEncoded : List Nat → Digest)
    (parent : Fin n → Nat) : Digest :=
  hashEncoded (encodeParentResidues parent)

/--
Hash binding over canonical parent residue encodings.

This is the symbolic form of the collision-resistance/binding
assumption restricted to accepted parent encodings. It says a feasible accepted
collision cannot even change the encoded parent list.
-/
def EncodedParentResidueDigestBinding
    {n : Nat}
    {Digest : Type}
    (hashEncoded : List Nat → Digest) : Prop :=
  ∀ (parentA parentB : Fin n → Nat),
    hashEncoded (encodeParentResidues (n := n) parentA) =
      hashEncoded (encodeParentResidues (n := n) parentB) →
        encodeParentResidues (n := n) parentA =
          encodeParentResidues (n := n) parentB

/--
Encoding-level digest binding implies the parent-binding assumption consumed by
`DigestParentBinding`.
-/
theorem parentDigestBinding_of_encodedParentResidueDigestBinding
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    (hBinding : EncodedParentResidueDigestBinding (n := n) hashEncoded) :
    DigestParentBinding.ParentDigestBinding
      (hashEncodedParentResidues (n := n) hashEncoded) := by
  intro parentA parentB hHash
  apply encodeParentResidues_injective
  exact hBinding parentA parentB hHash

/--
Shape of a flattened terminal parent `CE(B)` handle.

The direct CCS runtime fixes these lengths from the SuperNeo profile and the
program shape. The encoding below includes them as explicit tags so the digest
source is domain and shape separated.
-/
structure ParentCEBShape where
  commitmentLen : Nat
  publicInputLen : Nat
  pointLen : Nat
  evaluationLen : Nat
  auxiliaryLen : Nat
deriving DecidableEq

/-- Flattened parent `CE(B)` data that may be bound before proving `Pi_DEC`. -/
structure ParentCEB (shape : ParentCEBShape) where
  domainTag : Nat
  profileTag : Nat
  relationTag : Nat
  paramsTag : Nat
  commitment : Fin shape.commitmentLen → Nat
  publicInput : Fin shape.publicInputLen → Nat
  point : Fin shape.pointLen → Nat
  evaluations : Fin shape.evaluationLen → Nat
  auxiliary : Fin shape.auxiliaryLen → Nat

/-- Shape tags included in the parent `CE(B)` encoding. -/
def encodeShape (shape : ParentCEBShape) : List Nat :=
  [ shape.commitmentLen
  , shape.publicInputLen
  , shape.pointLen
  , shape.evaluationLen
  , shape.auxiliaryLen
  ]

/-- Canonical same-shape encoding for a flattened parent `CE(B)` handle. -/
def encodeParentCEB {shape : ParentCEBShape} (parent : ParentCEB shape) :
    List Nat :=
  [ parent.domainTag
  , parent.profileTag
  , parent.relationTag
  , parent.paramsTag
  ] ++
  encodeShape shape ++
  encodeVector parent.commitment ++
  encodeVector parent.publicInput ++
  encodeVector parent.point ++
  encodeVector parent.evaluations ++
  encodeVector parent.auxiliary

/--
Same-shape parent `CE(B)` encoding is injective.

This proves there is no encoding ambiguity once the direct CCS program has
fixed the parent handle shape. The cryptographic digest assumption is still
separate: this theorem only says equal encoded field lists imply equal parent
objects.
-/
theorem encodeParentCEB_injective {shape : ParentCEBShape} :
    Function.Injective (encodeParentCEB (shape := shape)) := by
  intro parentA parentB h
  cases parentA
  cases parentB
  simp [encodeParentCEB, encodeShape] at h ⊢
  rcases h with ⟨hDomain, hProfile, hRelation, hParams, hTail⟩
  have hCommitment := encodeVector_append_inj hTail
  rcases hCommitment with ⟨hCommitment, hTail⟩
  have hPublicInput := encodeVector_append_inj hTail
  rcases hPublicInput with ⟨hPublicInput, hTail⟩
  have hPoint := encodeVector_append_inj hTail
  rcases hPoint with ⟨hPoint, hTail⟩
  have hEvaluations := encodeVector_append_inj hTail
  rcases hEvaluations with ⟨hEvaluations, hAuxiliaryEncoded⟩
  have hAuxiliary := encodeVector_injective hAuxiliaryEncoded
  exact
    ⟨ hDomain
    , hProfile
    , hRelation
    , hParams
    , hCommitment
    , hPublicInput
    , hPoint
    , hEvaluations
    , hAuxiliary
    ⟩

/-- Equal parent `CE(B)` encodings recover the encoded shape tags. -/
theorem encodeParentCEB_shape_eq
    {shapeA shapeB : ParentCEBShape}
    (parentA : ParentCEB shapeA)
    (parentB : ParentCEB shapeB)
    (h : encodeParentCEB parentA = encodeParentCEB parentB) :
    shapeA = shapeB := by
  cases shapeA with
  | mk commitmentLenA publicInputLenA pointLenA evaluationLenA auxiliaryLenA =>
  cases shapeB with
  | mk commitmentLenB publicInputLenB pointLenB evaluationLenB auxiliaryLenB =>
  cases parentA
  cases parentB
  simp [encodeParentCEB, encodeShape] at h ⊢
  exact
    ⟨ h.2.2.2.2.1
    , h.2.2.2.2.2.1
    , h.2.2.2.2.2.2.1
    , h.2.2.2.2.2.2.2.1
    , h.2.2.2.2.2.2.2.2.1
    ⟩

/-- Parent `CE(B)` value bundled with its shape. -/
structure SomeParentCEB where
  shape : ParentCEBShape
  parent : ParentCEB shape

/-- Canonical encoding for a shape-indexed parent `CE(B)` value. -/
def encodeSomeParentCEB (parent : SomeParentCEB) : List Nat :=
  encodeParentCEB parent.parent

/--
Shape-independent parent `CE(B)` encoding is injective.

The shape tags in `encodeParentCEB` are sufficient to recover the parent shape,
then the same-shape theorem recovers the parent fields.
-/
theorem encodeSomeParentCEB_injective :
    Function.Injective encodeSomeParentCEB := by
  intro parentA parentB h
  cases parentA with
  | mk shapeA parentA =>
  cases parentB with
  | mk shapeB parentB =>
  have hShape : shapeA = shapeB := encodeParentCEB_shape_eq parentA parentB h
  subst shapeB
  have hParent : parentA = parentB := encodeParentCEB_injective h
  subst hParent
  rfl

/-- Hash a shape-indexed parent `CE(B)` through its canonical encoding. -/
def hashEncodedSomeParentCEB
    {Digest : Type}
    (hashEncoded : List Nat → Digest)
    (parent : SomeParentCEB) : Digest :=
  hashEncoded (encodeSomeParentCEB parent)

/--
Hash binding over canonical shape-indexed parent `CE(B)` encodings.

This is the direct formal boundary for the protocol hash source:
`hash(domain || encodeSomeParentCEB(parent))`.
-/
def EncodedParentCEBDigestBinding
    {Digest : Type}
    (hashEncoded : List Nat → Digest) : Prop :=
  ∀ parentA parentB,
    hashEncoded (encodeSomeParentCEB parentA) =
      hashEncoded (encodeSomeParentCEB parentB) →
        encodeSomeParentCEB parentA = encodeSomeParentCEB parentB

/--
Binding over canonical parent `CE(B)` encodings recovers the exact parent
handle, including its shape.
-/
theorem same_parentCEB_of_encoded_digest_binding
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    (hBinding : EncodedParentCEBDigestBinding hashEncoded)
    {parentA parentB : SomeParentCEB}
    (hHash :
      hashEncodedSomeParentCEB hashEncoded parentA =
        hashEncodedSomeParentCEB hashEncoded parentB) :
    parentA = parentB := by
  apply encodeSomeParentCEB_injective
  exact hBinding parentA parentB hHash

/--
A digest-only source authorizes a parent-residue vector through a full encoded
parent `CE(B)` handle and a deterministic projection from that handle.

This is the intended reduced-handle source shape: the transcript binds one
parent `CE(B)`, while the private `Pi_DEC` authorization proves children
against the projected parent coefficient/residue vector.
-/
def BindsProjectedParentCEBResidues
    {n : Nat}
    {Digest : Type}
    (hashEncoded : List Nat → Digest)
    (project : SomeParentCEB → (Fin n → Nat))
    (source : DigestParentBinding.Source Digest)
    (parentResidues : Fin n → Nat) : Prop :=
  ∃ parent : SomeParentCEB,
    source.digest = hashEncodedSomeParentCEB hashEncoded parent ∧
      project parent = parentResidues

/--
Binding a canonical full parent `CE(B)` encoding functionally binds every
deterministic residue projection from that parent.

This theorem is the formal link between the implementation idea
"hash one parent `CE(B)`" and the arithmetic DEC theorem, which reasons about
the parent residue vector.
-/
theorem bindsProjectedParentCEBResidues_functionally_of_encodedDigestBinding
    {n : Nat}
    {Digest : Type}
    {hashEncoded : List Nat → Digest}
    {project : SomeParentCEB → (Fin n → Nat)}
    (hBinding : EncodedParentCEBDigestBinding hashEncoded) :
    GoldilocksChildTableAuthorization.SourceBindsParentFunctionally
      (BindsProjectedParentCEBResidues
        (n := n)
        hashEncoded
        project) := by
  intro source parentResiduesA parentResiduesB hA hB
  rcases hA with ⟨parentA, hDigestA, hProjectA⟩
  rcases hB with ⟨parentB, hDigestB, hProjectB⟩
  have hHash :
      hashEncodedSomeParentCEB hashEncoded parentA =
        hashEncodedSomeParentCEB hashEncoded parentB :=
    hDigestA.symm.trans hDigestB
  have hParent : parentA = parentB :=
    same_parentCEB_of_encoded_digest_binding hBinding hHash
  calc
    parentResiduesA = project parentA := hProjectA.symm
    _ = project parentB := by rw [hParent]
    _ = parentResiduesB := hProjectB

end ParentEncoding

end DirectCcsFPrime
