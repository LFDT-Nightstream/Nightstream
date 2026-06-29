# Direct Terminal Soundness

`DirectTerminalSoundness` specifies the composed theorem boundary for the
terminal direct CCS `F'` proof using the reduced parent `CE(B)` handle.

The accumulator update used by the terminal transition is:

```text
ParentSourceStep:
  Pi_CCS output claims
  Pi_RLC parent CE(B) source

AuthorizedNextPiCCSInputs:
  digest-bound parent CE(B)
  opened parent residues
  canonical private Pi_DEC
  child CE(b)^k witness table
  wire identity into next Pi_CCS
```

The direct Construction-2 transition uses:

```text
BoundaryStep
AccumulatorStep = ReducedAccumulatorStep(ParentSourceStep, AuthorizedNextPiCCSInputs)
```

The terminal theorem requires proof-carrying folded prior `F'` authority. The
prior authority must imply reachability of the prior public image from the base
image. A digest label alone is not sufficient.

The theorem target is:

```text
accepted proof-carrying prior authority
accepted latest direct F' transition
=>
final public image is reachable
```

The strengthened uniqueness target is:

```text
Pi_CCS stage functionality
Pi_RLC stage functionality
encoded parent digest binding
deterministic parent-statement commitment encoding
Ajtai no-collision
CE-opening adapter
accepted terminal latest transition
any alternate latest transition from the same prior image
=>
same parent CE(B) source
same authorized next Pi_CCS child inputs
```

For deterministic implementations, the stage functionality premises are
discharged by representing the accepted stages as function-computed relations:

```text
Pi_CCS(i, prior, out) := out = computePiCCS(i, prior)
Pi_RLC(i, out, source) := source = computePiRLC(i, out)
```

The computed-stage terminal theorem has the same conclusion as the strengthened
uniqueness target, but it does not require separate `Pi_CCS` or `Pi_RLC`
functionality assumptions.

For deterministic parent-statement encoding:

```text
StatementEncodes(stmt, parent_CE_B)
  := stmt.commitment = commitmentOfParent(parent_CE_B)
```

the terminal theorem derives the required statement encoding from commitment
equality by construction.
The remaining premises are the binding/security boundaries: encoded parent
digest binding, Ajtai no-collision, and the CE-opening adapter.

For a fixed SuperNeo CE commitment map, the terminal theorem may take the
assignment-level Ajtai opening adapter instead:

```text
AssignmentOpeningAdapter(params, ce.commitMap)
```

Together with `CE.Holds`, this induces the local CE-opening adapter because
`CE.Holds` includes the commitment equality:

```text
statement.commitment = ce.commitMap(witness.assignment)
```

The strongest implementation-facing theorem uses the theorem-facing Ajtai
binding assumption from the SuperNeo proof-system layer:

```text
AjtaiBindingAssumption(params)
```

instead of a bespoke local no-collision premise.

It also has an MSIS-facing form that derives the Ajtai binding premise through
the SuperNeo lattice reduction surface:

```text
MSISToAjtaiReductions(params)
MSISHardnessAssumption(params)
```

The concrete-facing form replaces the free assignment-opening adapter with a
canonical Ajtai-backed commitment map for the fixed CE commitment function:

```text
AjtaiBackedCommitMap(params, ce.commitMap)
```

This map ties `ce.commitMap` to a fixed public Ajtai matrix, canonical
`M || Mz` commitments, bounded openings, and the residue projector used by
private `Pi_DEC`. The terminal theorem derives the assignment-opening adapter
from this canonical backing.

This is the theorem-level shape needed for the reduced-handle strategy. The
remaining obligations are concrete instantiations of the abstract relations,
not additional digest assumptions.
