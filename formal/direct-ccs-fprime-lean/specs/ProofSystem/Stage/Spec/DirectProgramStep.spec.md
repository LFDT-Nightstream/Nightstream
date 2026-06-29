# Direct Program Step

`DirectProgramStep` specifies the deterministic direct-computation boundary
used by the direct CCS terminal `F'` proof.

The direct computation boundary relation has the form:

```text
ComputedBoundaryStep(computeBoundary, i, priorBoundary, nextBoundary)
  := nextBoundary = computeBoundary(i, priorBoundary)
```

For a fixed prior public image, two accepted latest direct `F'` transitions
using this boundary relation must agree on:

```text
currentBoundary
```

Each accepted transition also exposes the exact computed boundary:

```text
next.currentBoundary = computeBoundary(i, prior.currentBoundary)
```

If the reduced accumulator fields are also functional for the same prior
image, then the whole latest public image is functional:

```text
nextImageA = nextImageB
```

The concrete terminal theorem combines:

```text
deterministic direct boundary update
canonical Ajtai-backed CE commitment map
encoded parent CE(B) digest binding
MSIS-to-Ajtai reductions
MSIS hardness
proof-carrying folded F' prior authority
accepted latest direct F' step
alternate latest direct F' step from the same prior image
```

and proves:

```text
the final image is reachable
and the accepted latest public image is unique
```

This spec does not define a particular application transition. A frontend
supplies `computeBoundary`; the theorem states the exact proof obligation once
that frontend boundary function is fixed.
