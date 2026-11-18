# Auxiliary Rank-Two SIS Security

## Status

Accepted.
[`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md) gives the rank of
the main commitment and states the limit of this record for the ring
`X^64 + 1`.

## Problem

The symbol `κ` is a module rank, not a security level. A long auxiliary SIS
map needs an explicit width limit and a union bound over all collision
targets. Such a map must not be confused with the main witness commitment,
which has the rank `κ = 18`.

## SuperNeo

SuperNeo v1.2 does not define a rank-two profile for an auxiliary
protocol-binding map.

## Decision

An auxiliary protocol-binding SIS map can use `κ = 2` only if:

- its source fields use the [41-trit field slot](41-trit-encoding.md);
- its width is at most 2,720,034 coefficients, which is 42,500 ring columns
  of degree 64;
- the active system has at most seven rank-two collision targets, all
  included in the union bound; and
- its claim states the external Module-SIS and estimator assumptions.

The limit comes from an estimate for the prior profile, the Goldilocks prime
and the ring degree 54, at 50,371 ring columns
(`scripts/estimate_nebula_sis.sage`). The model targets 128 bits of quantum
Core-SVP cost after the union bound and counts seven targets as eight. No
estimate was run for the profile Nightstream P64, k_rho = 14. The field
record keeps the limit as a count of coefficients: the rank in coefficients
grows from 108 to 128 at the same width and norm, so the instance is not
easier.

No map of the current Lean, `SPEC.md` or Rust uses the rank two. No Lean
result in the current tree states this limit.
