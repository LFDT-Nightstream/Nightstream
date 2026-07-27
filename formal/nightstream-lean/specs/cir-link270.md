# CIR-LINK270 — canonical 270-coordinate terminal link

```text
property_id: CIR-LINK270
claim:
  The fixed-active 270-coordinate terminal link has a canonical Lean-owned
  encoding: one affine equality per coordinate, with every coefficient and
  column identity computed in Lean and the row count derived from the
  construction rather than declared. Satisfaction of that encoding is
  equivalent to the paper link relation for canonical assignments, every
  emitted row is owned by exactly one coordinate, and removing any
  coordinate's row admits a violation at exactly that coordinate.
assumptions:
  - Canonical assignments: constant-one wire at column 0, all values canonical
    residues, the two blocks reading the two carriers (`CanonicalAssignment`).
  - The Phi81 profile fixes ringDegree = 54 and publicRingColumns = 5.
non_goals:
  - NOT global minimality. Necessity is proved only within the declared
    coordinatewise-affine normal form: no packing, no cross-coordinate
    compression. A different valid normal form gives a different exact count.
  - NOT a production claim. No captured coefficient, generated row, measured
    dimension, or artifact value appears anywhere in the canonical module.
  - NOT column closure against the rest of the program. The canonical encoding
    owns its own allocation; whether production's range is column-closed
    against neighbouring families is a Phase-1b obligation.
paper_sources:
  - docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md:15-19 (Definition
    13 CE and L_in), :111-133 (Pi_RLC public-input combination)
rust_surfaces:
  - none in Phase 1, by construction.
  - Phase 1b compares against the selected range [19,274,957, 19,275,227).
circuit_or_encoding_artifacts:
  - none. `carrierWidth = ringDegree * publicRingColumns` is computed; `270` is
    a theorem (`carrierWidth_eq`), not an input.
failure_class:
  An encoding that pins tail coordinates 257..269 to zero rather than copying
  them rejects an honest post-fold running instance, since a sampler-valid
  Pi_RLC challenge maps external coordinate 256 into coordinate 257
  (REL-LIN-RUNNING-CLOSURE).
counterexample_or_witness:
  `dropCoordinate_admits_violation` constructs, for every coordinate, an
  assignment satisfying all remaining rows while violating the link at exactly
  that coordinate. `copies_not_pinsZero` proves copying and zero-pinning are
  mutually exclusive, so the Phase-1b measurement is decisive.
lean_theorems:
  - Link270.canonicalRows_holds_iff
  - Link270.canonicalRows_length_eq
  - Link270.affine_zero_iff
  - Link270.dropCoordinate_admits_violation
  - Link270.nonzeroTail_linked
  - Link270.canonicalRows_owned
  - Link270.coordinateRow_injective
  - Link270.columns_disjoint
  - Link270Production.copies_not_pinsZero
  - Link270Production.tailPinsZero_not_agrees
  - Link270Production.tail_count
axiom_report:
  [propext] or [propext, Classical.choice, Quot.sound] throughout; tail_count
  depends on no axioms. No theorem depends on Lean.trustCompiler, and neither
  module uses native_decide. Guarded in tests/Axioms/CanonicalLink270.lean.
conformance_status:
  Phase 1 model-proved. Phase 1b specified but not measured: the comparison
  surface is fixed over an arbitrary `Capture` so the canonical side cannot be
  tuned to match, and no production capture has been supplied.
retest_commands:
  - cd formal/nightstream-lean && lake build
      Nightstream.Implementation.R1CS.Canonical.Link270Production
      tests.CanonicalLink270 tests.Axioms.CanonicalLink270
```

## Derived cost tuple

`carrierWidth = ringDegree * publicRingColumns = 54 * 5`, and
`canonicalRows.length = carrierWidth` follows from the construction. The
complete cost is `canonicalCost_eq : canonicalCost = ⟨270, 540, 1⟩` —
**270 recurring rows, 540 allocated columns** (two preallocated blocks of 270),
and **1 shared read column** (the constant wire, read but never allocated).
`rowColumns_accounted` proves no row touches any column outside that
allocation, and `constantWire_not_allocated` keeps the shared wire out of the
owned set. So **N_canonical(link270) = 270 rows / 540 columns** is a theorem,
not a measurement. This is the
first constraint count in the project derived from the protocol rather than
from an artifact.

That it coincides with production's 270 rows is only meaningful *because* the
derivation never consulted them.

## Phase 1b — what remains

The comparison surface is in place and decides one question:

```text
TailCopies capture   -- rows 257..269 enforce destination i = source i
TailPinsZero capture -- rows 257..269 enforce destination i = 0
```

`copies_not_pinsZero` proves these are mutually exclusive, and
`classify_exhaustive` proves the three-way classification
`exactCopies | tailPinsZero | other` is exhaustive — exclusivity alone would
not make the measurement decisive, since a capture may satisfy neither.

**Scope of the comparison.** Full equality is over all 270 rows
(`capture_eq_canonical`), not the thirteen tail rows; `tail_count` bounds only
the focused security regression. And row-shape agreement is insufficient on its
own: `AgreesAt` compares against the capture's *own claimed* column identities,
so `ColumnsAligned` is a separate obligation tying those to the authoritative
coordinates. `agreesAt_of_aligned` is what upgrades shape agreement to equality
with the canonical row.

What is missing is the capture itself:
sparse coefficients and column identities for `[19,274,957, 19,275,227)`,
Rust-originated, supplied as a `Capture`.

Outcomes:

- `CaptureAgrees` → production copies the tail; the link is correct.
- `TailPinsZero` → live defect, with the correct rows already proved here.
- Neither → sort the delta into cosmetic / encoding-choice / extra / missing /
  wrong-coefficient before proposing any Rust change.
