# Constraint encoding

## Purpose

`ConstraintEncoding` owns protocol-neutral algebraic encodings that are shared
by more than one recursive-verifier phase. Its modules describe deterministic
constraint schedules and prove their exact mathematical meaning. Protocol
modules remain responsible for selecting coordinates, while generated
artifacts remain responsible for connecting those schedules to concrete rows,
matrices, selectors, and column maps.

## Boolean pair rows

For an ordered finite sequence of Boolean coordinates, the schedule consumes
coordinates from left to right:

| Input shape | Emitted row | Obligation | Degree with linear selector |
|---|---|---|---:|
| two coordinates `a, b` | pair row | `r(a)^2 - 7*r(b)^2 = 0` | 5 |
| one trailing coordinate `a` | tail row | `r(a) = 0` | 3 |

Here `r(x) = x(x - 1)`. Seven is a quadratic nonresidue in the Goldilocks
field, so the pair equation holds exactly when both residuals vanish. The
schedule therefore accepts exactly when every coordinate is Boolean.

For a sequence of length `n`, it has:

- `n / 2` pair rows;
- `n % 2` tail rows;
- `(n + 1) / 2` total rows.

Flattening the coordinates named by the scheduled rows must recover the input
sequence exactly. Consequently the schedule preserves order, covers every
coordinate once per input occurrence, and preserves `Nodup` exactly.

## Consumer boundaries

| Consumer family | Coordinate sequence | Required concrete bridge |
|---|---|---|
| common Boolean membership | retained Boolean slots in stage order | generated global row/matrix schedule and fixed-selector binding |
| Pi_RLC acceptance tree | fourteen product-tree outputs per chunk | generated acceptance artifact and output-column projection |
| Pi_RLC Mod-5 quotient | thirteen low bits followed by the derived high bit | generated Mod-5 artifact and exact decoder projection |

No consumer may remove or replace a production row solely from the generic
equivalence. A valid integration proves that its authoritative coordinate
order, emitted matrices, row interval, linear selector, and inactive behavior
instantiate the generic schedule without aliasing or omitted uses.

## Acceptance criteria

1. Pairing is sound and complete for every finite coordinate list.
2. Row counts, coordinate order, coverage, and duplicate preservation are
   proved for the deterministic schedule.
3. Selector-gated degree is at most five when the selector is one linear,
   verifier-fixed factor.
4. Separate witnesses show that deleting a pair row or an odd-tail row admits
   a non-Boolean coordinate assignment.
5. Theorems use the proved Goldilocks nonresiduosity of seven and introduce no
   additional algebraic assumption.
