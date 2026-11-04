# Option-B slots only for read columns

## Status

Accepted (owner, 2026-11-03).

## Problem

An option-B step package commits one low-norm slot for each source column that
has one. A slot for a column that no row of the relation reads costs
coordinates and commitment work but checks nothing. A bit slot holds one
coordinate, but the completeness proof must show that the honest value is 0 or
1, and the completeness of the gadgets does not state the values of the
private `bit` hints.

## SuperNeo

SuperNeo v1.2 commits the witness of a CCS instance and requires the norm bound
of each coordinate. It does not fix how a circuit's source columns map to
witness coordinates.

## Decision

A package gives a column a slot only if a row of its exported relation reads
the column: a term of a row, a lane, output or quotient of a ring piece, a lane
or output local of a block, or an S-box column. The slot is a bit only if the
column is a bit candidate (a `bit` hint, or a bit word of a native
application) and a row forces it to 0 or 1; every other slot is a field
element. Lean defines the rule (`Export.OptionB.StepPackage.runsOf`) and Rust
computes the same slots (`StepPackage::rule_slots`). The relation is sound for
every slot layout, and the rule makes each bit slot valid wherever the rows
hold (SPEC.md section 4.4).
