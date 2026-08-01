# Ajtai Lean

Formal model for the F′ Ajtai opening gadget and its rank-two width bound.

## Proven results

```text
balanced trits per Goldilocks field     41
coefficients per Phi81 ring column      54
canonical-opening rows                  21
canonical-opening coordinates           61
maximum rank-two ring columns        50,371
maximum packed Goldilocks fields     66,342
```

The 21-row gadget composes pairs of radix-3 borrow transitions. Lean proves
equivalence to the scalar canonicality chain. Split-NC supplies the strict
bound on each retained digit and borrow coordinate.

## Estimator policy

The rank-two width search fixes:

```text
q                         2^64 - 2^32 + 1
lattice rank dimension   2 * 54
coefficient columns      ring_columns * 54
collision length bound   2 * sqrt(coefficient_columns)
cost model               ADPS16 quantum, 0.265 * beta
post-union target         128 bits
rank-two attack targets   7, rounded to 8
raw target                131 bits
minimum accepted beta     495
```

`EstimatorModel.lean` bounds logarithms with rational intervals, checks all
integer lattice dimensions against the Chen root-Hermite threshold, and
searches the width. It proves:

- `50,371` ring columns are accepted.
- `50,372` has a block-size-494 attack certificate at dimension `1,182`.
- Every larger width is rejected by monotonicity.
- `50,371` ring columns pack at most `66,342` source fields.

## Trust boundary

Lean proves the interval arithmetic, search result, and packing limit. The
local security interface composes an explicit collision-to-MSIS extractor with
the matching MSIS boundary. It does not construct either input. It also does
not prove the Core-SVP cost model, the Euclidean lattice-attack model,
pseudorandomness of the structured matrix, or Module-SIS hardness.

The concrete Nightstream commitment relation must instantiate both events and
the extractor before this boundary can support a protocol security claim.

The estimator's collision bound and SuperNeo's generic relaxed-binding norm
parameter are different statements. The estimator result does not instantiate
the generic hardness premise.

## External reproduction

Use
[`scripts/estimate_nebula_sis.sage`](../../scripts/estimate_nebula_sis.sage)
with
[`malb/lattice-estimator`](https://github.com/malb/lattice-estimator/tree/3e48ef421ec256afddb3e7d2249a77eab6e9ba12)
at commit `3e48ef421ec256afddb3e7d2249a77eab6e9ba12`. Keep that repository outside
this one.

The relevant upstream path is `SISLattice.cost_euclidean` followed by
`reduction.beta` and `ADPS16(mode="quantum")`. It uses

```text
log(delta) = (log(length_bound) - (n / d) * log(q)) / (d - 1).
```

Upstream selects an approximate optimal `d`; Lean checks every integer
`d >= 2`. The pinned run agrees with the formal boundary:

```text
ring columns   beta   d      quantum cost
50,371         495    1,182  131.2 bits
50,372         494    1,182  130.9 bits
```

Validate with:

```bash
lake build
lake exe check
```
