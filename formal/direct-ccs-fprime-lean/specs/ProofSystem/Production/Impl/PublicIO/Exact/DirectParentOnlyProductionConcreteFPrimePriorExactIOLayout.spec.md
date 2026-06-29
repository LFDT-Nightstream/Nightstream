# DirectParentOnlyProductionConcreteFPrimePriorExactIOLayout

`DirectParentOnlyProductionConcreteFPrimePriorExactIOLayout` specifies the
split discipline for exact public-IO verification in the concrete prior `F'`
path.

The exact public-IO verifier output contains:

```text
terminal public fields
Construction-2 boundary public fields
raw public vector
raw = terminal ++ boundary
```

The production split requirement is that whenever the raw verifier vector equals
the canonical terminal and boundary concatenation for the same `(steps, image)`
statement, the exposed terminal slice has the canonical terminal public-vector
length.

An implementation can satisfy that requirement by exposing the terminal slice as
the prefix obtained from the raw vector at the canonical terminal public-vector
length:

```text
terminal = raw.take(canonical terminal length)
```

From that length requirement and raw vector equality, the module derives the
full exact layout binding consumed by the split raw public-IO soundness surface:

```text
terminal = canonical terminal public values
boundary = canonical Construction-2 boundary public values
```

This prevents a verifier output from satisfying raw public-vector equality while
hiding a different terminal/boundary split inside the structured public IO.
Poseidon2 binding and compressed proof soundness are separate assumptions at
their cryptographic verifier boundary.

The terminal-length surface also induces the concrete certified prior verifier
used by the production parent-only endpoint. Accepted verifier evidence opens
folded `F'` authority for the same `(steps, image)` pair, reaches the claimed
prior public image, exposes the prior public-image invariants, rejects
unreachable prior images, and feeds the parent-only terminal end-to-end theorem
with the non-aggregate private DEC and Section 7.1 stage-audit projections.
