# F′ Stage 1 Domain `2^28`

## Status

Accepted by the owner on 2026-08-28.

## Problem

Π_CCS runs one sum-check over a row cube that covers the rows of the relation
and the committed assignment
([`padded-row-identity-piccs.md`](./padded-row-identity-piccs.md)). The
number of variables of this cube fixes the sum-check rounds, the largest key
width and the error numerators. Each of these values must use one number.

## SuperNeo

SuperNeo v1.2 Section 7.3 runs one sum-check on the `log m` row cube. It does
not fix `m` for an implementation.

## Decision

The row cube has 28 variables (`Lifecycle.cubeVariables`). So Π_CCS has
exactly 28 sum-check rounds in the light fold and in the refresh fold
([SPEC.md](../SPEC.md) section 3). The rows of each relation and the logical
width of each key are at most `2^28`. The two circuit keys have the logical
width `2^26` (`Export.BoundedDepth.StepRelation.keyWidth`). The Module-SIS
estimate of [`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md) uses
the width `2^28`.

This decision does not change `b = 2`, `k_rho` or `B = 2^k_rho`.
