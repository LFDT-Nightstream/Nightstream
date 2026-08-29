# F′ Stage 1 Domain `2^26` (superseded)

## Status

Superseded by `fprime-stage1-domain-2p28.md` on 2026-08-28.

## Decision

This file records the previous 26-variable decision. It is not the active
Stage 1 profile.

Every dependent layout theorem, package field, fixture, Rust loader check,
relation identity, and final fixed-point proof must use this one value. Old
`2^25` artifacts and identities are not evidence for the new profile.

This decision does not change `b = 2`, `k_rho = 16`, `B = 2^16`, the 14
separate CCS matrices, the Poseidon2-only binding rule, or the Stage 2 and
proof-backend boundaries.

## Footprint note

Lean later established the corrected direct S-box count as 108,068,374
coordinates, which is greater than `2^26 = 67,108,864`.
