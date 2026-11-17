# Marked Poseidon2 blocks

## Status

Accepted (owner, 2026-11-16).

## Problem

A bounded-depth step package packs each Poseidon2 permutation into a block
whose 1,080 inner locals have no column. This is sound only when no other row
reads an inner local, and complete only when the honest rows pack. The exporter
found the blocks by matching the permutation template against the circuit's
rows, so a proof that the honest rows pack had to show that no other run of
rows matches the template, for every gadget.

## SuperNeo

SuperNeo v1.2 does not fix how a circuit's rows are cut into CCS rows or how
a hash permutation is arithmetized.

## Decision

Each gadget that places a permutation marks it with an operation that has no
rows, no locals and no witness (`Circuit.Op.mark`). A mark gives the
permutation's first local and its 16 input lanes, and a child's marks reach its
parent (`Circuit.Subcircuit.marks`). The exporter makes a block only at the
first row of the next mark, when the mark's rows follow and its 16 lanes are
linear (`Export.Generic.Pieces.nextPiece`). A ring-product block must end before
the next mark's first row. A mark's rows are the canonical template with its
lanes and locals, which are the compiler's rows of the permutation of its lanes
(`Export.Generic.PermutationWindow.window_eq_markWindow`). The marks do not
change any emitted package. The step relation packs when the rows are free rows
that read no inner local and the marks' rows, in order, and the marks ascend
after the public columns with affine lanes that read only earlier live columns
(`Export.Generic.MarkedBlocks.packable_of_segmented`). Each gadget states these
facts for its own rows (`Gadgets.Poseidon2.LiveRows`, SPEC.md section 4.4).
