# Parent Opening Authorization

`ParentOpeningAuthorization` defines the proof boundary that makes a compact
parent `CE(B)` digest usable as the source for private `Pi_DEC` child
authorization.

The target statement is:

```text
digest = Hash(encode(parent_CE_B))
statement encodes parent_CE_B
CE.Holds(parent_statement, parent_witness)
parent_residues = coefficients(parent_witness)
```

For a fixed digest source, the authorized `parent_residues` must be functional.
This requires two independent obligations:

```text
1. the digest binds the canonical encoded parent CE(B) handle;
2. the encoded parent CE(B) handle has functionally bound opening residues.
```

The second obligation is the formal place where a concrete implementation must
use Ajtai commitment binding and CE membership. A public parent handle
`(c, x, r, y)` alone is not automatically an authoritative source of witness
coefficients; the proof must connect those coefficients to an accepted opening
of that same parent handle.

The opening-residue obligation decomposes into:

```text
statement encodes parent_CE_B
=>
statement.commitment equals the commitment field encoded in parent_CE_B

same fixed CE relation
+ same commitment
+ CE.Holds(statementA, witnessA)
+ CE.Holds(statementB, witnessB)
=>
coefficients(witnessA) = coefficients(witnessB)
```

The second implication is where the concrete Ajtai commitment binding theorem
must be applied. It is not a consequence of digest binding alone.

A deterministic commitment-facing statement encoder has the form:

```text
StatementEncodes(stmt, parent_CE_B)
  := stmt.commitment = commitmentOfParent(parent_CE_B)
```

Such an encoder satisfies the statement-commitment consistency obligation by
construction.

The fixed-CE relation requirement is essential. A theorem that allows two
arbitrary CE relations with unrelated commitment maps is too strong and is not
the target protocol statement. The parent opening used to authorize `Pi_DEC`
must be checked against the one fixed CE relation for the current fold/program
shape.

The local theorem-facing commitment obligation is:

```text
ce.commitMap(assignmentA) = ce.commitMap(assignmentB)
=>
residues(assignmentA) = residues(assignmentB)
```

A stronger Ajtai theorem may prove full witness equality; residue equality is
the exact projection required by private `Pi_DEC`.

The module must not assume that arbitrary projections from a parent handle are
sound. Any projection used to feed `Pi_DEC` must be justified by an accepted
parent opening relation.
