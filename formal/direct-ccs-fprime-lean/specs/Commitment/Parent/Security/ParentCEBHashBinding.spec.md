# Parent CE(B) Hash Binding

`ParentCEBHashBinding` specifies the cryptographic binding boundary for the
direct CCS reduced parent-handle path.

The protocol source is:

```text
parent_CE_B
  -> encodeSomeParentCEB(parent_CE_B)
  -> hash digest
```

The encoded parent handle contains:

```text
domain tag
profile tag
relation tag
parameter tag
shape lengths
commitment fields
public input fields
evaluation point fields
evaluation fields
auxiliary fields
```

The required binding assumption is exactly:

```text
hash(encodeSomeParentCEB(parentA))
=
hash(encodeSomeParentCEB(parentB))
=>
encodeSomeParentCEB(parentA) = encodeSomeParentCEB(parentB)
```

Together with parent-encoding injectivity, equal digests recover the exact
parent `CE(B)` handle and its shape:

```text
digest(parentA) = digest(parentB)
=>
parentA = parentB
```

For any deterministic projection from parent `CE(B)` handles to the DEC parent
residue vector, the same digest source authorizes at most one projected parent
residue vector:

```text
source.digest = digest(parentA)
project(parentA) = residuesA

source.digest = digest(parentB)
project(parentB) = residuesB

=>
residuesA = residuesB
```

This module does not implement or prove the concrete hash permutation. It states
the exact hash-binding premise needed by the reduced-handle theorem and proves
the local consequences used by private `Pi_DEC` authorization. The Lean theorem
consumes only the binding assumption over canonical parent encodings.
