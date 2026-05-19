import DirectCcsFPrime.ParentEncoding

/-!
Parent CE(B) hash-binding boundary.

This module owns the exact cryptographic assumption needed by the reduced
parent-handle path. It does not implement or prove the concrete hash
permutation. It packages the assumption that the protocol hash used for parent
`CE(B)` handles binds the canonical `ParentEncoding.encodeSomeParentCEB`
source.
-/

namespace DirectCcsFPrime

namespace ParentCEBHashBinding

/-- Hash context for canonical parent `CE(B)` handles.

`hashEncoded` is the field-list sponge/encoding function used in the protocol
binding path. `encodedBinding` is the exact collision-resistance/binding
assumption over canonical parent `CE(B)` encodings.
-/
structure ParentCEBHash (Digest : Type) where
  hashEncoded : List Nat → Digest
  encodedBinding :
    ParentEncoding.EncodedParentCEBDigestBinding hashEncoded

/-- Digest of a canonical parent `CE(B)` handle. -/
def digest {Digest : Type} (ctx : ParentCEBHash Digest)
    (parent : ParentEncoding.SomeParentCEB) : Digest :=
  ParentEncoding.hashEncodedSomeParentCEB ctx.hashEncoded parent

/-- Digest source induced by a canonical parent `CE(B)` handle. -/
def source {Digest : Type} (ctx : ParentCEBHash Digest)
    (parent : ParentEncoding.SomeParentCEB) :
    DigestParentBinding.Source Digest :=
  { digest := digest ctx parent }

/-- The packaged context supplies the encoded-parent binding premise. -/
theorem encodedParentCEBDigestBinding
    {Digest : Type}
    (ctx : ParentCEBHash Digest) :
    ParentEncoding.EncodedParentCEBDigestBinding ctx.hashEncoded :=
  ctx.encodedBinding

/--
Equal parent-handle digests recover the exact parent `CE(B)` handle,
including its encoded shape.
-/
theorem same_parentCEB_of_digest_eq
    {Digest : Type}
    (ctx : ParentCEBHash Digest)
    {parentA parentB : ParentEncoding.SomeParentCEB}
    (hDigest : digest ctx parentA = digest ctx parentB) :
    parentA = parentB :=
  ParentEncoding.same_parentCEB_of_encoded_digest_binding
    ctx.encodedBinding
    hDigest

/--
The parent-handle digest functionally binds every deterministic DEC
parent-residue projection from the opened parent `CE(B)` handle.
-/
theorem projected_residue_source_functional
    {n : Nat}
    {Digest : Type}
    (ctx : ParentCEBHash Digest)
    (project : ParentEncoding.SomeParentCEB → (Fin n → Nat)) :
    GoldilocksChildTableAuthorization.SourceBindsParentFunctionally
      (ParentEncoding.BindsProjectedParentCEBResidues
        (n := n)
        ctx.hashEncoded
        project) :=
  ParentEncoding.bindsProjectedParentCEBResidues_functionally_of_encodedDigestBinding
    ctx.encodedBinding

end ParentCEBHashBinding

end DirectCcsFPrime
