# F′ Stage 1 Domain `2^26`

## Status

Accepted by the owner on 2026-08-27.

## Decision

The Nightstream F′ Stage 1 production relation uses a 26-variable row cube.
Its maximum joint row and carrier domain is `2^26`, and PiCCS therefore uses
exactly 26 SumCheck rounds.

Every dependent layout theorem, package field, fixture, Rust loader check,
relation identity, and final fixed-point proof must use this one value. Old
`2^25` artifacts and identities are not evidence for the new profile.

This decision does not change `b = 2`, `k_rho = 16`, `B = 2^16`, the 14
separate CCS matrices, the Poseidon2-only binding rule, or the Stage 2 and
proof-backend boundaries.

## Footprint note

The larger domain does not make the current all-direct low-norm Poseidon2
plan fit. Lean proves that its retained S-box outputs alone require
108,019,010 coordinates, which is greater than `2^26 = 67,108,864`.
The final compiler still needs an actual constraint and column reduction.
