# SuperNeoBridge

`SuperNeoBridge` defines the formal boundary between the direct CCS reduced
CE-handle argument and the existing SuperNeo Lean protocol surfaces.

The bridge must not redefine CE, Ajtai commitments, `Pi_CCS`, `Pi_RLC`, or
`Pi_DEC`. It imports those objects from `formal/superneo-lean` and states the
direct CCS reduced-handle theorem over those imported objects.

The bridge target is:

```text
existing SuperNeo CE/Ajtai child bundle
+ functionally parent-bound reduced source
+ Goldilocks child-table proof soundness
+ wire identity into next Pi_CCS
=>
the same reduced source cannot authorize different next Pi_CCS child inputs
```

The CE/Ajtai child bundle has two dimensions:

```text
k = 14 low-norm CE children
n = flattened witness/commitment coefficient columns
```

The bridge must explicitly connect the lower-level child table to the real CE
witnesses. A proof over a standalone digit table is not enough; the table
accepted by the reduced-handle verifier must be the table extracted from the
CE witnesses, and the next `Pi_CCS` input must be wire-identical to that same
table. Extraction is coefficient-wise: each child row is read from the
canonical residue of the corresponding flattened `CE.Witness.assignment`
coefficient imported from the SuperNeo Lean model.

The bridge also exposes the reusable stage authority:

```text
ceRelation ctx
=>
Pi_CCS strong statement for ctx
Pi_RLC weak statement for ctx
Pi_DEC knowledge statement for ctx
```

Digests are not authority in this spec. If a reduced source is a digest, the
source-to-parent binding assumption must state the binding property explicitly,
and the bridge theorem may use only that binding property plus proof-checked
child authorization. A digest-only source is acceptable only under an explicit
parent-digest binding hypothesis; the bridge must not infer parent binding
from the existence of a digest value alone.

The preferred digest-source bridge uses the canonical parent encoding:

```text
hashEncoded(encodeParentResidues(parent))
```

Binding over this encoded source is sufficient to recover the parent-binding
condition used by the SuperNeo child-input uniqueness theorem.

For the implementation target that hashes one full parent `CE(B)` instead of
all `CE(b)^k` children, the preferred bridge is:

```text
hashEncoded(encodeParentCEB(parent_CE_B))
project(parent_CE_B) = parent_residues
private Pi_DEC proof verifies child table against parent_residues
next Pi_CCS inputs are wire-identical to that child table
```

The bridge must prove that, under encoded-parent digest binding, the same
digest source cannot authorize different projected parent residues and
therefore cannot authorize different next `Pi_CCS` child inputs.

The strongest bridge shape requires the projected residues to come from an
accepted opening of the same encoded parent:

```text
hashEncoded(encodeParentCEB(parent_CE_B))
statement encodes parent_CE_B
CE.Holds(statement, parent_witness)
parent_residues = coefficients(parent_witness)
private Pi_DEC proof verifies child table against parent_residues
next Pi_CCS inputs are wire-identical to that child table
```

This is the preferred theorem for implementation work. A bridge based on an
arbitrary projection from public parent fields is only a structural stepping
stone; it is not enough unless that projection is separately proved
authoritative for the parent opening.

For the concrete Ajtai-backed implementation shape, the bridge consumes:

```text
NoAjtaiBindingCollision(params)
CEOpeningAdapter(params, ce)
StatementEncodesByCommitment(commitmentOfParent)
```

and derives fixed-CE opening residue binding internally. The deterministic
statement encoder satisfies statement-commitment consistency by definition: a
statement encodes a parent exactly when its commitment equals
`commitmentOfParent(parent_CE_B)`. The theorem must not ask callers to assume
arbitrary residue functional binding or arbitrary statement-encoding
consistency when the intended source of those facts is concrete Ajtai opening
binding over accepted `CE.Holds` witnesses plus deterministic commitment
encoding.
