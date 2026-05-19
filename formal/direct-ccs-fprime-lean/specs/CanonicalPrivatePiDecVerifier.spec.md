# Canonical Private Pi_DEC Verifier

`CanonicalPrivatePiDecVerifier` specifies the terminal private `Pi_DEC`
relation for the reduced `CE(B)^1` handle strategy.

The verifier relation is over the actual SuperNeo child bundle:

```text
Verify(source, parent_residues, child_bundle, proof)
```

It accepts exactly when the child bundle's checked digit table satisfies:

```text
binary digits
fixed column length k_dec = 14
Goldilocks modular recomposition to parent_residues
```

The child bundle supplies the protocol wiring outside this relation:

```text
CE.Holds for every child CE
Ajtai opensTo for every child commitment
child CE relation equals the fixed stage CE relation
child Ajtai parameters equal the fixed stage Ajtai parameters
digit table equals the table extracted from child CE witnesses
next Pi_CCS inputs equal the checked digit table
```

The theorem target is:

```text
canonical private Pi_DEC verifier
+ source binds parent residues functionally
+ accepted SuperNeo child bundles
=>
same next Pi_CCS child inputs
```

The implementation-facing authorization relation is existential:

```text
AuthorizedNextPiCCSInputs(source, next_inputs)
```

It holds when there exists an opened parent residue vector, an accepted
SuperNeo child bundle, and a proof accepted by the canonical private `Pi_DEC`
relation such that the bundle uses the fixed CE relation and fixed Ajtai
parameters, and `next_inputs` is the bundle's next `Pi_CCS` input table.
For a fixed source, this relation is functional under the same parent-binding
and child-bundle obligations:

```text
AuthorizedNextPiCCSInputs(source, next_a)
+ AuthorizedNextPiCCSInputs(source, next_b)
=>
next_a = next_b
```

For the digest-bound parent path, the source-binding condition is supplied by
canonical parent `CE(B)` encoding, digest binding, deterministic statement
commitment encoding, and Ajtai-backed CE-opening residue binding.

This component treats the terminal proof as directly enforcing this relation.
It does not introduce a separate sumcheck protocol or rely on a standalone
child table disconnected from the SuperNeo child bundle.
