# Parent Encoding

`ParentEncoding` defines canonical field-list encodings for parent data used by
the direct CCS reduced-handle strategy.

The core target is:

```text
parent residue vector
  -> canonical length-tagged field list
  -> hash digest source
```

The encoding must be injective for a fixed parent shape:

```text
encode(parentA) = encode(parentB) => parentA = parentB
```

For shape-indexed `CE(B)` parent handles, the encoding must also recover the
shape:

```text
encode((shapeA, parentA)) = encode((shapeB, parentB))
=>
shapeA = shapeB and parentA = parentB
```

This module does not prove concrete hash security. It separates the proof
obligation into two parts:

```text
1. prove the parent encoding is canonical/injective;
2. assume the digest binds the canonical encoded parent list.
```

The module also defines a flattened `CE(B)` parent-handle encoding shape with
explicit domain/profile/relation/parameter tags and per-field length tags. This
is the encoding surface a terminal implementation should use when replacing a
full `CE(b)^k` transcript source with one parent `CE(B)` handle plus a checked
`Pi_DEC` authorization.

For the reduced-handle theorem, the full parent handle must authorize DEC
parent residues only through a deterministic projection:

```text
digest = Hash(encode(parent_CE_B))
project(parent_CE_B) = parent_residues
```

Under the encoded-parent digest-binding assumption, this makes the authorized
residue vector functional for a fixed digest source. The projection may select
the concrete parent coefficient/residue vector needed by the `Pi_DEC`
child-table check, but it must be deterministic and part of the proved
statement shape.

Digest values are not authority. A digest source is usable only when it is
computed from this canonical parent encoding and paired with the protocol's
explicit parent-digest binding assumption.
