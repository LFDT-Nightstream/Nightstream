# F′ Stage 1 Main Ajtai Setup

## Status

Superseded. The owner accepted this record on 2026-08-31. Three records
replace its rules:

- [`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md) replaces the
  Goldilocks field, the rank `κ = 22` and the security calculation with
  `q = 2^64 − 59`, the ring `X^64 + 1` and `κ = 18`;
- [`fprime-ajtai-shake128-setup.md`](./fprime-ajtai-shake128-setup.md)
  replaces the ChaCha20 expander with SHAKE128; and
- [`fprime-ajtai-rejection-sampling.md`](./fprime-ajtai-rejection-sampling.md)
  replaces the 256-bit wide reduction with 8-byte rejection sampling and a
  fallback.

Only the seed rule below stays in force.

## Problem

The main witness commitment needs a public key that each verifier can
compute again. An earlier setup used 64-bit rejection sampling with no
fallback. It needed a proof that sampling succeeded at each key index, and
that proof was not practical.

## SuperNeo

SuperNeo `Setup` samples the commitment matrix uniformly at random
(Definition 8). The paper does not specify how to derive the matrix from a
seed.

## Decision

The key comes from one verifier-owned 32-byte seed. An operating-system
CSPRNG gave the seed once. The seed is public:

```text
fc404984d44c1b878d68a6a80092d7d7ab44d81ac17b45a8e7bd4c1f1e371702
```

Lean and Rust hold this value
(`Export.Stage1.Poseidon2HashChainV1SetupAuthority.productionSeed`,
[SPEC.md](../SPEC.md) Algorithm 1). The prover cannot select the seed.

The replaced rules were the rank `κ = 22`, the setup ID
`nightstream-ajtai-chacha20-wide256-v1`, one ChaCha20 block for each
coefficient, and the reduction of 256 output bits modulo the Goldilocks
prime. The Ajtai key of the current tree does not use them.
