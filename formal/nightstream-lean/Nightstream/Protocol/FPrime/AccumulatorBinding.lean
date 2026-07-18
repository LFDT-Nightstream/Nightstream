import Nightstream.Protocol.FPrime.XOut

/-!
Nested binding for the exact ordered Construction-2 accumulator.

Assurance tier: model-level security partition.

Owns: one domain-separated hash interface for claim and accumulator messages;
the exact ordered child-digest preimage; exhaustive encoding/hash failures; and
composition with the compact `state_x_out` authority theorem.

Does not own: a concrete CE serializer, Poseidon2 parameters or collision
bounds, Rust/R1CS refinement, rows, costs, or row removal.

Emits constraints: no.

Authority boundary: the accumulator digest is compression, never authority.
It binds a child list only when the claim encoding is injective and the single
domain-separated hash family does not collide. The checked Pi_RLC parent is
deliberately absent because Pi_DEC recomposition is not child-vector unique.

| Stage path | Mathematical obligation | Authority class | Lean owner |
|---|---|---|---|
| `fprime.accumulator.claim.encoding` | encode every authority-bearing field of one child | computed or encoding failure | `Scheme.encodeClaim`, `EncodingCollision` |
| `fprime.accumulator.claim.hash` | compress one encoded child with the claim domain | security boundary | `claimDigest`, `HashCollision` |
| `fprime.accumulator.children.order` | retain arity and child digests in exact index order | computed | `AccumulatorPreimage`, `preimage` |
| `fprime.accumulator.children.hash` | compress the exact ordered child message | security boundary | `digest` |
| `fprime.accumulator.binding` | equal handles imply equal child lists or a named failure | derived | `digest_eq_or_failure` |
| `fprime.x_out.accumulator` | equal compact outputs recover equal child lists or an inner/outer failure | derived | `claims_eq_or_chainFailure` |
-/

namespace Nightstream.Protocol.FPrime.AccumulatorBinding

universe uClaim uEncoding uDigest uParams uStructure uHeader uRunning uFresh
  uNebulaDigest

/-- Exact typed preimage of the ordered-child accumulator hash. `count` is
explicit because Rust absorbs it before the child digests. -/
structure AccumulatorPreimage (Digest : Type uDigest) where
  count : Nat
  childDigests : List Digest
deriving Repr, DecidableEq

/-- One hash family with constructor-level domain separation. Production must
instantiate both constructors with Poseidon2 and distinct fixed domain tags. -/
inductive Message (Encoding : Type uEncoding) (Digest : Type uDigest) where
  | claim (preimage : Encoding)
  | accumulator (preimage : AccumulatorPreimage Digest)
deriving Repr, DecidableEq

/-- Minimal interface for the nested accumulator commitment. -/
structure Scheme
    (Claim : Type uClaim)
    (Encoding : Type uEncoding)
    (Digest : Type uDigest) where
  encodeClaim : Claim -> Encoding
  hash : Message Encoding Digest -> Digest

/-- Compression of one exact claim encoding. -/
def claimDigest
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest)
    (claim : Claim) : Digest :=
  scheme.hash (.claim (scheme.encodeClaim claim))

/-- Exact accumulator preimage: explicit arity followed by child digests in
list order. -/
def preimage
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest)
    (claims : List Claim) : AccumulatorPreimage Digest where
  count := claims.length
  childDigests := claims.map (claimDigest scheme)

/-- Compact ordered-child accumulator handle. -/
def digest
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest)
    (claims : List Claim) : Digest :=
  scheme.hash (.accumulator (preimage scheme claims))

/-- Two distinct claims have the same pre-hash encoding. This is a concrete
serialization bug, not a cryptographic event. -/
def EncodingCollision
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest) : Prop :=
  ∃ left right : Claim,
    left ≠ right ∧ scheme.encodeClaim left = scheme.encodeClaim right

/-- Collision in either domain of the sole nested hash family. -/
def HashCollision
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest) : Prop :=
  ∃ left right : Message Encoding Digest,
    left ≠ right ∧ scheme.hash left = scheme.hash right

/-- Exhaustive failures at the nested accumulator binding boundary. -/
inductive BindingFailure
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest) : Prop where
  | encoding (collision : EncodingCollision scheme)
  | hash (collision : HashCollision scheme)

/-- Equality of one child digest recovers the exact child or exhibits the
precise encoding/hash failure that prevented recovery. -/
theorem claim_eq_or_failure
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest)
    (left right : Claim)
    (sameDigest : claimDigest scheme left = claimDigest scheme right) :
    left = right ∨ BindingFailure scheme := by
  classical
  by_cases sameClaim : left = right
  · exact Or.inl sameClaim
  by_cases sameEncoding : scheme.encodeClaim left = scheme.encodeClaim right
  · exact Or.inr (.encoding ⟨left, right, sameClaim, sameEncoding⟩)
  · apply Or.inr
    apply BindingFailure.hash
    exact ⟨Message.claim (scheme.encodeClaim left),
      Message.claim (scheme.encodeClaim right),
      (by simpa using sameEncoding), sameDigest⟩

/-- Pointwise recovery lifts to the complete ordered child list. -/
private theorem claims_eq_or_failure
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest)
    (left right : List Claim)
    (sameDigests :
      left.map (claimDigest scheme) = right.map (claimDigest scheme)) :
    left = right ∨ BindingFailure scheme := by
  induction left generalizing right with
  | nil =>
      cases right with
      | nil => exact Or.inl rfl
      | cons head tail => simp at sameDigests
  | cons leftHead leftTail inductionHypothesis =>
      cases right with
      | nil => simp at sameDigests
      | cons rightHead rightTail =>
          simp only [List.map_cons, List.cons.injEq] at sameDigests
          rcases claim_eq_or_failure scheme leftHead rightHead
              sameDigests.1 with headEq | failure
          · rcases inductionHypothesis rightTail sameDigests.2 with
              tailEq | failure
            · exact Or.inl (by simp [headEq, tailEq])
            · exact Or.inr failure
          · exact Or.inr failure

/-- Equal nested handles bind the complete ordered claim list, modulo exactly
one concrete encoding failure or one collision in the sole nested hash family. -/
theorem digest_eq_or_failure
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest)
    (left right : List Claim)
    (sameDigest : digest scheme left = digest scheme right) :
    left = right ∨ BindingFailure scheme := by
  classical
  by_cases samePreimage : preimage scheme left = preimage scheme right
  · apply claims_eq_or_failure scheme left right
    exact congrArg AccumulatorPreimage.childDigests samePreimage
  · apply Or.inr
    apply BindingFailure.hash
    exact ⟨Message.accumulator (preimage scheme left),
      Message.accumulator (preimage scheme right),
      (by simpa using samePreimage), sameDigest⟩

/-- A state coordinate is bound to the exact ordered claims only through this
recomputation equation. A carried digest without this equation has no
authority. -/
def StateBinds
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    (scheme : Scheme Claim Encoding Digest)
    (stateDigest : Digest)
    (claims : List Claim) : Prop :=
  stateDigest = digest scheme claims

/-- Every failure that can explain equal compact `x_out` values with distinct
ordered children. -/
inductive ChainFailure
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (xOutSemantics :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (scheme : Scheme Claim Encoding Digest) : Prop where
  | xOut (failure : XOut.BindingFailure xOutSemantics)
  | accumulator (failure : BindingFailure scheme)

/-- Composition theorem for Rust's compact state shape: equal `state_x_out`
values recover the exact ordered accumulator claims when both states recompute
their accumulator handles, or expose one outer/inner binding failure. -/
theorem claims_eq_or_chainFailure
    {Claim : Type uClaim}
    {Encoding : Type uEncoding}
    {Digest : Type uDigest}
    {Params : Type uParams}
    {StructureDigest : Type uStructure}
    {Header : Type uHeader}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Nebula : Type}
    {NebulaDigest : Type uNebulaDigest}
    (xOutSemantics :
      XOut.Semantics Params StructureDigest Header Digest Nebula NebulaDigest)
    (scheme : Scheme Claim Encoding Digest)
    (leftMode rightMode : XOut.Mode)
    (leftContext rightContext :
      XOut.Context Params StructureDigest Header Digest)
    (leftState rightState :
      Nightstream.HyperNova.Construction2.State Digest Running Fresh Nebula)
    (leftClaims rightClaims : List Claim)
    (leftPinned :
      XOut.StatePinned xOutSemantics leftMode leftContext leftState)
    (rightPinned :
      XOut.StatePinned xOutSemantics rightMode rightContext rightState)
    (leftBinds :
      StateBinds scheme leftState.accumulatorDigest leftClaims)
    (rightBinds :
      StateBinds scheme rightState.accumulatorDigest rightClaims)
    (sameOutput :
      XOut.compute xOutSemantics leftMode leftContext leftState =
        XOut.compute xOutSemantics rightMode rightContext rightState) :
    leftClaims = rightClaims ∨ ChainFailure xOutSemantics scheme := by
  rcases XOut.xOut_binding_or_collision xOutSemantics leftMode rightMode
      leftContext rightContext leftState rightState leftPinned rightPinned
      sameOutput with authorityEq | failure
  · have stateDigestEq :
        leftState.accumulatorDigest = rightState.accumulatorDigest :=
      congrArg
        (fun authority => authority.construction2Accumulator)
        authorityEq
    have nestedDigestEq :
        digest scheme leftClaims = digest scheme rightClaims :=
      leftBinds.symm.trans (stateDigestEq.trans rightBinds)
    rcases digest_eq_or_failure scheme leftClaims rightClaims nestedDigestEq with
      claimsEq | failure
    · exact Or.inl claimsEq
    · exact Or.inr (.accumulator failure)
  · exact Or.inr (.xOut failure)

end Nightstream.Protocol.FPrime.AccumulatorBinding
