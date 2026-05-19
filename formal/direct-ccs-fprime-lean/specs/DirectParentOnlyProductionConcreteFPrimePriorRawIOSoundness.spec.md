# DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness

`DirectParentOnlyProductionConcreteFPrimePriorRawIOSoundness` defines the
split soundness certificate for the production raw public-vector prior `F'`
verifier.

The verifier-visible check record contains:

```text
compact image replay
Construction-2 boundary replay
transcript replay
canonical statement public validity
proof boundary = canonical statement boundary
terminal committed verifier returns raw public vector
raw public vector = terminal_public_values ++ boundary_public_values
```

Replay must bind the opaque proof statement to the canonical `(steps, image)`
statement. The trusted compressed-verifier theorem then consumes the resulting
bound raw public-vector statement and opens folded `F'` authority for the same
public pair.

The module proves that the split certificate:

```text
induces the raw public-IO runtime verifier surface
extracts a bound statement witness from accepted raw verification
opens folded F' authority for accepted raw verification
cannot accept an unreachable prior image
fixes prior public-image invariants
feeds the certified terminal end-to-end theorem and production audit projections
```

Poseidon2 transcript binding and compressed terminal proof soundness remain
trusted at the `rawBoundStatementAuthoritySound` field. Raw public-vector
equality is verifier evidence, not folded `F'` authority by itself.
