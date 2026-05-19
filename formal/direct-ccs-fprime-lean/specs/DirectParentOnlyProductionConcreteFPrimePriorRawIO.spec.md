# DirectParentOnlyProductionConcreteFPrimePriorRawIO

`DirectParentOnlyProductionConcreteFPrimePriorRawIO` exposes audit-facing
consequences of the raw public-vector backend for the production concrete prior
`F'` verifier.

The raw verifier predicate accepts only verifier-visible checks:

```text
compact image replay
Construction-2 boundary replay
transcript replay
canonical statement public validity
proof boundary = canonical statement boundary
terminal committed verifier returns raw public vector
raw public vector = terminal_public_values ++ boundary_public_values
```

The module proves that accepted raw verification:

```text
opens a concrete folded F' authority object
cannot accept an unreachable prior image
fixes the prior image step, verifier-key digest, initial boundary, and well-formedness
is functional for one opaque prior proof
feeds the certified terminal end-to-end theorem and its non-aggregate/stage-audit projections
```

The raw verifier also induces the strict `SoundVerifier` consumed by terminal
production soundness. Strict verifier acceptance is equivalent to the raw
public-vector checks, opens folded `F'` authority for the same `(steps, image)`
pair, preserves prior public-image invariants, rejects unreachable prior
images, is same-proof functional, and passes latest-step acceptance through the
same strict verifier object.

These consequences depend on the raw backend soundness obligation from
`DirectParentOnlyProductionConcreteFPrimePriorBackend`. That obligation is the
cryptographic boundary for Poseidon2 transcript binding and compressed terminal
proof soundness; raw public-vector equality is not treated as folded `F'`
authority by itself.
