# Public-Seed MSIS Assumption for the ChaCha20 Key

## Status

Superseded by
[`fprime-ajtai-shake128-setup.md`](./fprime-ajtai-shake128-setup.md) on
2026-09-29. The owner approved this assumption on 2026-09-08 for the matrix
of `nightstream-ajtai-chacha20-wide256-v1` only. That expander is removed,
and the approval does not cover another expansion.

## Problem

The key comes from one fixed public seed. So the commitment matrix is one
fixed matrix, not a sample of a uniform matrix, and anyone can compute it.

## SuperNeo

SuperNeo `Setup` samples the commitment matrix uniformly at random
(Definition 8), and binding follows from MSIS for a uniform matrix
(Theorem 6). The paper does not discuss seeds.

## Decision

The superseded premise: MSIS is hard for the one matrix that ChaCha20
expands from the fixed seed, with the seed and the full procedure public.
This premise is specific to Nightstream. It is stronger than a claim about
the average over random seeds, and no proof supports it.

The current premises are SHAKE128 as a random oracle and MSIS for a uniform
matrix (`[KECCAK]` and `[MSIS]` in
[the security model](../formal/nightstream-fprime/SECURITY_MODEL.md)).
[The security argument](fprime-ajtai-shake128-security-argument.md) gives the
reduction.

These points of the record still apply:

- The seed is fixed and public ([SPEC.md](../SPEC.md) Algorithm 1). It gives
  no average over seeds, and an adversary that is precomputed for this seed
  is outside the current premises (security model, trust item 1).
- A setup that samples a new seed, or that changes the expansion, needs its
  own record and its own premise.
- The matrix of this record had 22 rows and 4,708,530 ring columns of degree
  54, which is 254,260,620 coefficients. Rust keeps this count of
  coefficients as the limit of the key columns (`MAX_MESSAGE_COLUMNS` in
  `crates/neo-ajtai/src/nightstream_fprime_setup.rs`).
