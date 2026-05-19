# F Prime Trace Authority Red Team

This component records negative checks for shortcuts that must not authorize a
folded `F'` prior image.

## Digest-Only Attack

A verifier that accepts a proof only because a digest-shaped predicate accepts
is not trace-sound. The red-team model has no valid transitions and still lets
the digest-only predicate accept a forged image; Lean proves that such a
verifier cannot satisfy trace soundness.

## Aggregate Child Attacks

Aggregate child summaries do not determine the private DEC child table:

```text
same aggregate digit sum
same aggregate norm total
```

are both compatible with different child positions. These summaries cannot
replace pointwise DEC recomposition, child CE membership, and exact wire reuse.

## DEC Shortcut Attacks

Signed low-norm base-2 recomposition is not unique, and modular recomposition
without a range proof is not unique. A valid private DEC proof must therefore
include canonical digit shape, fixed length, and no-wrap/range evidence.
