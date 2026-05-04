# Binary Child Table Authorization

This spec states a concrete arithmetic target for authorizing hidden
post-`Pi_DEC` children without hashing the full `CE(b)^k` accumulator.

## Objects

- `parent`: a vector of parent coefficient representatives.
- `children`: a column-oriented base-2 digit table. For column `j`,
  `children j` is the least-significant-first digit list that recomposes to
  `parent j`.
- `source`: the reduced public transcript source.
- `proof`: a proof object checked by `F'`.

## Proof Obligation

The proof verifier is acceptable only if it implies all of the following over
the same child table wires that feed the next `Pi_CCS` accumulator:

1. every digit is binary;
2. every column has the fixed digit length `k`;
3. recomposing each column gives the parent coefficient.

The fixed-length condition is mandatory. Binary recomposition alone is not
unique because leading zeroes can be added.

If recomposition is checked through field arithmetic modulo `q`, the verifier
must also establish a no-wrap condition, such as `2^k < q`, so that modular
equality implies integer equality for each recomposed column.

## Theorem Target

If two accepted authorizations use the same source and parent, and their proof
verifier satisfies the obligation above, then both accepted authorizations feed
the same next child table to `Pi_CCS`.

This theorem is the arithmetic core behind a possible sumcheck/table proof for
the reduced `CE(B)^1` handle strategy.
