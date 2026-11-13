# Canonical 41-Trit Field Opening

## Status

Accepted.

Changed on 2026-11-13: [`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md) replaces the Goldilocks field and `d = 54` of this record with `q = 2^64 − 59` and `d = 64`. The 41-trit rule does not change, because `3^41` is larger than `2^64`.

## Problem

Auxiliary protocol-binding SIS maps consume general Goldilocks field values,
but Ajtai binding requires low-norm message coefficients. A direct field
residue can exceed the selected bound `b = 2`. Each authoritative field slot
also needs one unambiguous low-norm opening.

## SuperNeo

SuperNeo requires `‖z‖∞ < b` and selects `b = 2` and ring degree
`d = 54` for its Goldilocks profile. Its coefficient embedding does not define
an encoding for an arbitrary field value. See
[Definition 20](../docs/superneo-paper-v1_2/07_superneo_folding_scheme_for_ccs.md)
and [Section 8.2](../docs/superneo-paper-v1_2/08_concrete_parameters.md) of SuperNeo v1.2.

## Decision

Each authoritative Goldilocks field slot that enters an auxiliary
protocol-binding SIS map must use exactly 41 balanced trits
`d_i ∈ {-1,0,1}`. The constraints must bind those trits to the field slot and
must enforce the canonical residue interval. All consumers of the same owned
slot reuse that opening. Equal values in different authoritative slots do not
share ownership.

Forty-one is minimal because `3^40 < p < 3^41`. Lean proved the coverage and
minimality in `formal/ajtai-lean/Ajtai/Parameters.lean` at commit `f7e093d8f`.
That project was later deleted. Physical
gate, row, and column counts belong to lowering conformance evidence, not this
decision.
