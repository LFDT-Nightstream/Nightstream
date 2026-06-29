# F Prime Trace Authority

This component specifies explicit trace-carrying authority for folded `F'`
prior proofs.

## Mathematical Object

A trace authority for `(steps, image)` consists of:

```text
imageAt(0) = initial
imageAt(steps) = image
for every i < steps:
  Transition(i, imageAt(i), imageAt(i + 1))
```

## Guarantees

The trace authority implies:

```text
Reachable(Transition, initial, steps, image)
```

It can also be converted into the existing proof-carrying folded authority
object consumed by compressed verifier soundness surfaces.

## Verifier Soundness Shape

A verifier is trace-sound when every accepted prior proof opens to such a trace
for the same `(steps, image)` pair:

```text
VerifyPrior(steps, proof, image)
=>
exists trace:
  TraceAuthority(Transition, initial, steps, image)
```

This trace-sound verifier satisfies the terminal prior-authority contract and
can be viewed as a compressed `F'` sound verifier.

## Boundary Assumptions

This component does not implement Poseidon2, Fiat-Shamir, or a SNARK backend.
It only states the exact trace-opening obligation that those mechanisms must
provide at the Lean boundary.
