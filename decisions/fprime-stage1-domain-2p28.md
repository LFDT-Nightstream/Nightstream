# F′ Stage 1 Domain `2^28`

## Status

Accepted by the owner on 2026-08-28. This decision supersedes
`fprime-stage1-domain-2p26.md`.

## Decision

The Nightstream F′ Stage 1 production relation uses a 28-variable row cube.
Its maximum joint row and carrier domain is `2^28`, and PiCCS therefore uses
exactly 28 SumCheck rounds.

Every dependent layout theorem, package field, fixture, Rust loader check,
relation identity, and final fixed-point proof must use this one value. Old
`2^25` and `2^26` artifacts and identities are not evidence for this profile.

This decision does not change `b = 2`, `k_rho = 16`, `B = 2^16`, the 14
separate CCS matrices, the Poseidon2-only binding rule, or the Stage 2 and
proof-backend boundaries.

## Footprint note

Lean proves that the current direct low-norm Poseidon2 plan retains
108,160,050 S-box coordinates. This fits below
`2^28 = 268,435,456`, with 160,275,406 coordinates left before final outputs,
non-Poseidon source values, and the remaining Stage 1 phases are added.

This decision does not prove that the complete relation fits. The final
compiler must still construct the complete low-norm assignment and 14-matrix
plan and prove their joint domain at or below `2^28`.
