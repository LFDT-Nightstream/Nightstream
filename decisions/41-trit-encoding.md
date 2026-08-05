# Canonical 41-Trit Field Opening

## Status

Accepted.

## Problem

Auxiliary protocol-binding SIS maps consume general Goldilocks field values,
but Ajtai binding requires low-norm message coefficients. A direct field
residue can exceed the selected bound `b = 2`. Each authoritative field slot
also needs one unambiguous low-norm opening.

## SuperNeo

SuperNeo requires `‖z‖∞ < b` and selects `b = 2` and ring degree
`d = 54` for its Goldilocks profile. Its coefficient embedding does not define
an encoding for an arbitrary field value. See
[Definition 12](../docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md)
and [Appendix B.2](../docs/superneo-paper/11-b-concrete-parameters.md).

## Decision

Each authoritative Goldilocks field slot that enters an auxiliary
protocol-binding SIS map must use exactly 41 balanced trits
`d_i ∈ {-1,0,1}`. The constraints must bind those trits to the field slot and
must enforce the canonical residue interval. All consumers of the same owned
slot reuse that opening. Equal values in different authoritative slots do not
share ownership.

Forty-one is minimal because `3^40 < p < 3^41`. Lean proves the coverage and
minimality in
[Ajtai/Parameters.lean](../formal/ajtai-lean/Ajtai/Parameters.lean). Physical
gate, row, and column counts belong to lowering conformance evidence, not this
decision.
