# Parent Only Private Children Flow

`ParentOnlyPrivateChildrenFlow` specifies the local soundness boundary behind
the parent-only terminal optimization.

The public accumulator handle carries the parent source. The post-DEC
`CE(b)^14` children are private proof values. They remain valid only when the
terminal proof exposes all of the following facts:

```text
parent handle binding
private parent opening
pointwise Pi_DEC recomposition
child CE membership and Ajtai opening evidence
next Pi_CCS wire identity
Pi_CCS -> Pi_RLC parent-source computation using those exact wires
```

The module proves that public child projection hashes are not required for the
local no-swap guarantee:

```text
PointwisePrivateDecRequirements(source, next_a)
PointwisePrivateDecRequirements(source, next_b)
=>
next_a = next_b
```

and for full parent-only latest steps:

```text
Step(prior, next_a)
Step(prior, next_b)
=>
exists one private child table reused by both stage computations
next_a.parentSource = next_b.parentSource
```

This is not an aggregate child check. The pointwise DEC certificate includes
binary digits, fixed length 14, coordinate-wise Goldilocks recomposition,
CE-witness-derived child-table identity, and exact next-`Pi_CCS` wire reuse.

Poseidon2 is not implemented here. The module consumes the existing
`ParentCEBHashBinding.ParentCEBHash` object, whose binding assumption is the
trusted hash boundary for the parent handle.
