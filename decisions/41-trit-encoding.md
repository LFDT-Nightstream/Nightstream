# 41-Trit Field Slot

## Status

Accepted.

## Problem

The Ajtai commitment binds only a witness whose coefficients have a norm
below `b = 2`. A step circuit has columns that hold a general value of `F_q`.
Such a value can be larger than the bound, so each such column needs a
low-norm encoding.

## SuperNeo

SuperNeo v1.2 requires `‖z‖∞ < b` for a fresh witness (Definition 20). Its
coefficient embedding does not define an encoding for an arbitrary field
value.

## Decision

A field slot of a step package holds exactly 41 balanced trits
`d_i ∈ {−1, 0, 1}` (`Layout.BalancedTernary.width`,
`Layout.LowNormSlot.Kind`). The value of the slot is `Σ d_i · 3^i` in `F_q`
(`Layout.BalancedTernary.recompose`). Each row that reads the column reads
this linear form of the slot (`Export.Generic.SlotLayout.layout`,
[SPEC.md](../SPEC.md) section 4.4). Equal values in two columns have two
slots.

The honest digits are the balanced-ternary digits of the centered value.
Each digit has a norm below 2 (`Layout.BalancedTernary.digits_norm`), and the
digits give the value again (`Layout.BalancedTernary.recompose_digits`).

41 is the least count. 41 trits reach each magnitude up to `(3^41 − 1)/2`,
which is not below `(q − 1)/2` (`Layout.BalancedTernary.halfModulus_le_radius`).
40 trits reach only `(3^40 − 1)/2`, which is below `(q − 1)/2` (computed).

A column that a row forces to 0 or 1 has a slot of one coordinate instead
([`bounded-depth-slots-of-read-columns.md`](./bounded-depth-slots-of-read-columns.md)).
