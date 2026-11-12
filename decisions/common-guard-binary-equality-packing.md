# Common-guard binary equality packing

## Status

Proposed.

## Problem

The native application contains separate guarded equalities between binary
words. These rows use bits that already have Boolean certificates.

## SuperNeo

SuperNeo v1.2 fixes the folding relation and its commitment and transcript
rules. It does not select the compiler representation of these application
equalities.

## Decision

Use the source-index compiler pass in [SPEC.md section 7.13](../SPEC.md#713-common-guard-binary-equality-packing).
Run it after field lowering and guard fusion. One group owns the source
indices, retained certificate indices, normalized guard, and ordered bit
pairs. Derive the replacement rows from that group.

Accept source words of at most 32 bits. Pack at most 63 equalities per row.
Keep every Boolean certificate. A width-one slot alone cannot certify a bit.
Preserve source columns, logical assignments, source rows, and four matrices.
The generic equivalence is
`Export.Generic.WordPacking.ordered_iff`. The existing compiler trust boundary
covers the concrete matcher and encoding checks.
