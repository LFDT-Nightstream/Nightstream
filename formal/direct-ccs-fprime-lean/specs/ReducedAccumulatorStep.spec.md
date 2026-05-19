# Reduced Accumulator Step

`ReducedAccumulatorStep` specifies the direct CCS `F'` accumulator update that
uses one compact parent `CE(B)` source plus private `Pi_DEC` authorization.

The accumulator handle contains:

```text
parentSource
nextPiCCSInputs
```

`parentSource` is the compact source for the parent `CE(B)` claim produced by
the latest SuperNeo fold. `nextPiCCSInputs` is the authorized `CE(b)^k` child
table consumed by the next `Pi_CCS`.

The reduced accumulator step is:

```text
ParentSourceStep(i, prior, next.parentSource)
Authorized(next.parentSource, next.nextPiCCSInputs)
```

`ParentSourceStep` owns the deterministic derivation of the parent source from
the prior accumulator and the latest fold. In the concrete protocol this is the
`Pi_CCS -> Pi_RLC` parent `CE(B)` source computation.

`Authorized` owns the private `Pi_DEC` proof that the hidden children are the
canonical low-norm children for the opened parent and are wire-identical to the
next `Pi_CCS` accumulator inputs.

The theorem target is functional accumulator update:

```text
ParentSourceStep is functional
Authorized is functional
Step(i, prior, next_a)
Step(i, prior, next_b)
=>
next_a.parentSource = next_b.parentSource
next_a.nextPiCCSInputs = next_b.nextPiCCSInputs
```

For the canonical private `Pi_DEC` verifier, `Authorized` is functional under
encoded parent digest binding, deterministic statement commitment encoding,
Ajtai no-collision, and the local CE-opening adapter.

Applied inside the direct Construction-2 transition, the same theorem says two
accepted latest `F'` transitions from the same prior image cannot silently
authorize different reduced-handle accumulator children.
