# Auxiliary Rank-Two SIS Security

## Status

Accepted.

## Problem

The symbol `κ` is a module rank, not a security level. Long auxiliary SIS
maps need an explicit width limit and a union bound over all collision targets.
They must not be confused with the main SuperNeo witness commitment.

## SuperNeo

The selected main commitment uses `κ = 18`. The SuperNeo paper does not
define a rank-two profile for Nightstream's auxiliary protocol-binding maps.
See [Appendix B.2](../docs/superneo-paper/11-b-concrete-parameters.md) and the
[Nightstream commitment profile](../protocol-contract/src/normative/50-nightstream-profile.md).

## Decision

The main witness commitment remains `κ = 18`. An auxiliary
protocol-binding SIS map may use `κ = 2` only if:

- its source fields use the [canonical 41-trit opening](41-trit-encoding.md);
- its ring-column width is at most `50,371`;
- the active system has at most seven rank-two collision targets, all included
  in the union bound; and
- its claim states the external Module-SIS and estimator assumptions.

For `W` source fields, `r = ceil(41W/54)`, so the certified limit is also
`W <= 66,342`. The selected model targets 128 post-union quantum Core-SVP bits
and rounds seven targets to eight. Lean proves the arithmetic boundary in
[Ajtai/EstimatorModel.lean](../formal/ajtai-lean/Ajtai/EstimatorModel.lean) and
states the external hardness boundary in
[Ajtai/SecurityBoundary.lean](../formal/ajtai-lean/Ajtai/SecurityBoundary.lean).
