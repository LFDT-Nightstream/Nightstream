# Direct Stage SuperNeo Reuse

`DirectStageSuperNeoReuse` specifies the adapter from theorem-native SuperNeo
Section 7.1 contexts to the direct CCS `F'` contextual reused-stage package.

The reusable authority source is:

```text
SuperNeo.ceRelation(ctx)
```

From that single relation the imported SuperNeo interfaces provide:

```text
Pi_CCS strong statement for ctx
Pi_RLC weak statement for ctx
Pi_DEC knowledge statement for ctx
```

A direct contextual stage package consists of deterministic computations:

```text
computePiCCS(i, prior_accumulator) -> out
computePiRLC(i, out) -> parent_CE_B_source
piRLCContext(out, parent_CE_B_source) -> ctx
```

and theorem-native Section 7.1 owner objects for the contexts used by those
computations:

```text
piCCSSection71(i, prior_accumulator).target = out.ctx
piRLCSection71(i, out).target =
  piRLCContext(out, computePiRLC(i, out))
```

The adapter converts those Section 7.1 owner objects into
`DirectStageSemantics.ContextualReusedStageComputations`. This is the preferred
reuse path for the direct proof: each accepted direct stage context must be the
same context that the upstream SuperNeo theorem package proves, rather than a
caller-selected digest or aggregate summary.
