# Construction 2 Direct F Prime

`Construction2DirectFPrime` specifies the compact public-image shape and latest
F' transition for the single-relation direct CCS path.

The public image contains:

```text
vkDigest
step
initialBoundary
currentBoundary
accumulator
pc
```

The direct CCS path is single-relation, so well-formed images use:

```text
pc = 1
```

One latest F' transition from `prior` to `next` at step `i` requires:

```text
prior.step = i
next.step = i + 1
prior.vkDigest = next.vkDigest
prior.initialBoundary = next.initialBoundary
prior.pc = 1
next.pc = 1
BoundaryStep(i, prior.currentBoundary, next.currentBoundary)
AccumulatorStep(i, prior.accumulator, next.accumulator)
```

`BoundaryStep` owns the direct computation boundary update. `AccumulatorStep`
owns the verifier-side NIFS.V accumulator update for the latest step.

The canonical latest-step verifier accepts exactly this transition. Therefore
the verifier is sound for the transition by definition.

The terminal theorem combines this latest-step verifier with a separate
`FPrimeInduction.PriorAuthoritySound` premise. The prior authority premise is
essential: a digest or handle for the prior accumulator is not an induction
proof unless acceptance of that authority implies reachability of the prior
public image.

Reachability from a well-formed zero-step base image preserves the public
Construction-2 invariants that the terminal verifier exposes:

```text
final.step = number_of_accepted_F'_steps
final.vkDigest = initial.vkDigest
final.initialBoundary = initial.initialBoundary
final.pc = 1
```

These invariants are consequences of the accepted transition relation. They are
not independent digest checks.
