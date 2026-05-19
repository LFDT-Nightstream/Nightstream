# Poseidon2 Parent CE(B) Hash

`Poseidon2ParentCEBHash` specifies the implementation-facing hash boundary for
the parent-only direct CCS `F'` path.

The module does not define the Poseidon2 permutation. It treats the verifier's
parent-handle hash as a field-list function:

```text
hashEncoded : List Nat -> Digest
```

and assumes only the binding property needed by the terminal theorem:

```text
hashEncoded(encodeSomeParentCEB(parentA))
=
hashEncoded(encodeSomeParentCEB(parentB))
=>
encodeSomeParentCEB(parentA) = encodeSomeParentCEB(parentB)
```

The `Hash` object packages that exact function with that exact assumption and
constructs the canonical:

```text
ParentCEBHashBinding.ParentCEBHash
```

The local consequences are:

```text
equal parent digests recover the same parent CE(B) handle
the digest source functionally binds every deterministic parent-residue projection
```

The terminal theorem consumes this Poseidon2 parent hash object directly. It
does not accept a loose digest-binding premise and it does not treat the digest
as folded `F'` authority. Prior authority still comes only from either an
arbitrary sound authority predicate or a `CompressedFPrimeAuthority.SoundVerifier`.
