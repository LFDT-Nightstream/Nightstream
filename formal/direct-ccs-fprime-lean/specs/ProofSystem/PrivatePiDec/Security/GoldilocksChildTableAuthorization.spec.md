# Goldilocks Child-Table Authorization Specification

## Mathematical Target

This module models the DEC authorization boundary when recomposition is checked
as Goldilocks field equality rather than integer equality.

For a fixed reduced source and fixed parent residue vector, if a verifier
accepts a hidden child table, acceptance must imply:

- every child digit is binary,
- every coefficient column has exact length `k_dec = 14`,
- every column recomposes to the corresponding parent residue modulo the
  Goldilocks modulus,
- the next `Pi_CCS` accumulator input is the same child table.

Under those conditions, the accepted next accumulator input is unique.

## Required Theorems

- Two accepted Goldilocks child-table authorizations for the same source and
  parent residues have equal next inputs.
- A deterministic challenge derived from that reduced source cannot be made to
  authorize different next inputs.
- If the source-to-parent binding is functional, then two accepted
  authorizations for the same source have equal next inputs even when the
  parent residue vectors are not assumed equal up front.

## Soundness Role

The direct CCS/F' implementation may want to hash a compact parent handle
instead of hashing the full post-DEC `CE(b)^14` child accumulator. This is sound
only if the hidden children are uniquely determined by the public parent source
and the proof checked inside `F'`.

This module proves the Goldilocks modular version of that condition. It depends
on exact length and binary digit checks. Without those checks, field
recomposition equality is not enough to authorize hidden child state.

The source binding must also be functional: a single reduced source can
authorize at most one parent-residue vector. This converts the theorem from
"same source and same parent" to the implementation-facing statement
"same source".
