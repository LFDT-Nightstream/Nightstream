# Goldilocks No-Wrap Specification

## Mathematical Target

This module instantiates the fixed-length binary DEC no-wrap condition for the
SuperNeo Goldilocks parameter profile.

For `k_dec = 14`, every binary digit list of exact length `14` recomposes to an
integer below `2^14`. The Goldilocks modulus `q` is larger than `2^14`.
Therefore a field equality modulo `q` between two such recompositions is also
an integer equality.

## Required Theorems

- `2^14 < q` for the SuperNeo Goldilocks modulus.
- A binary digit list of length `14` recomposes below `q`.
- A column table whose columns are binary digit lists of length `14`
  recomposes below `q` column-wise.
- Two binary column tables of length `14` that have equal Goldilocks modular
  recompositions are equal.

## Soundness Role

This module closes the wraparound side condition required by the reduced
direct-CCS/F' Fiat-Shamir source strategy. If the implementation checks DEC
recomposition through Goldilocks field arithmetic, the verifier must also know
that the binary recomposition values are below the field modulus. Otherwise
distinct child tables can collide modulo the field.
