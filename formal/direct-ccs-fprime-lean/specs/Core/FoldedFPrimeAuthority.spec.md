# Folded F′ Authority

`FoldedFPrimeAuthority` specifies the prior-authority object consumed by
terminal direct CCS `F'` compression.

Terminal compression may check only the latest `F'` step. That is sound only
when the prior object already proves reachability of the prior public image
from the base image under the `F'` transition relation.

The authority contains:

```text
steps
image
reachable : Reachable(Transition, initial, steps, image)
```

Acceptance for a requested `(steps, image)` is:

```text
authority.steps = steps
authority.image = image
```

The theorem target is:

```text
Accepts(steps, authority, image)
=>
Reachable(Transition, initial, steps, image)
```

This discharges the `PriorAuthoritySound` premise used by terminal
Construction-2 compression.

The base authority is the zero-step initial image. Extending authority by one
accepted transition produces authority for the next step.

The base constructor authorizes exactly the zero-step initial image. The
extension constructor authorizes exactly the successor image produced by the
accepted transition used to extend the authority.

Digest-only data is not authority. A digest may commit to or name a folded
authority in an implementation, but terminal proof soundness requires an
accepted authority object whose acceptance implies reachability. Any predicate
that accepts an unreachable image is not a sound prior-authority predicate.
