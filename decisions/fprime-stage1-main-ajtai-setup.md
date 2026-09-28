# F′ Stage 1 Main Ajtai Setup

## Status

Accepted by the owner on 2026-08-31.

The owner first approved `κ = 25`, then corrected the selected rank to
`κ = 22` on 2026-08-31. The final approval is
`nightstream-ajtai-chacha20-wide256-v1`,
256-bit wide reduction, and generation of one 32-byte operating-system CSPRNG
seed. The generated public verifier-owned seed is:

```text
fc404984d44c1b878d68a6a80092d7d7ab44d81ac17b45a8e7bd4c1f1e371702
```

## Problem

The prior main commitment rank was `κ = 18`. At the concrete
`Poseidon2HashChainV1` footprint, the selected ADPS16 quantum Core-SVP model
gives BKZ block size 320, or 84.8 raw bits. This is below the project target.

The prior `AjtaiSetupV1.Setup` also required a proof that 64-bit rejection
sampling succeeded at every key index. A `κ = 22` key can contain as many as
5,905,580,076 Goldilocks coefficients under the `2^28` domain bound. Checking
all rejection paths is not a practical Lean proof or development task.

## Decision

Use this exact main witness-commitment setup:

- module rank `κ = 22`;
- one verifier-owned 32-byte seed from an operating-system CSPRNG;
- ChaCha20 as specified by [RFC 8439](https://www.rfc-editor.org/rfc/rfc8439.html);
- setup ID `nightstream-ajtai-chacha20-wide256-v1`;
- one direct random-access ChaCha20 block per `(row, block, lane)`;
- nonce `row_u32_le || block_u64_le`;
- block counter `lane_u32`;
- the first 32 output bytes interpreted as one little-endian 256-bit integer;
- coefficient equal to that integer modulo the Goldilocks prime; and
- no rejection, fallback, retry, or materialized full key.

Lean must define this function, prove every coefficient canonical, and emit
the setup authority. Rust must implement the same indexed function and pass
complete selected-index parity and mutation checks. The seed, rank, setup ID,
and message-column count must be raw verifier authority in the final package.

## Security calculation

The largest possible ring-column count under the approved domain is

```text
ceil(2^28 / 54) = 4,971,027.
```

At `κ = 22` and this worst-case width, the pinned estimator model gives:

```text
BKZ block size:                 429
ADPS16 quantum cost:            113.685 bits
eight-target allowance:           3.000 bits
post-allowance cost:             110.685 bits
```

The owner explicitly selected this approximately 110-bit post-allowance
level. It does not meet the former 128-bit post-allowance target.

The wide reduction follows the method in [RFC 9380, Section
5](https://www.rfc-editor.org/rfc/rfc9380.html#section-5). For a 64-bit prime,
32 input bytes give at most `2^-192` reduction bias per coefficient. The
complete worst-case key has fewer than `2^33` coefficients, so the union
bound is below `2^-159`.

The raw expanded key would contain 47,244,640,608 bytes. Production must keep
the key indexed and lazy. This number is a size calculation, not an artifact
to generate.

The final security theorem must name these external assumptions:

- Module-SIS hardness under the pinned estimator model;
- ChaCha20 pseudorandomness for the indexed expansion;
- the wide-reduction statistical bound; and
- the existing low-norm invertibility boundary.

## Scope and supersession

This decision supersedes only the statement in
[`rank-two-sis-security.md`](./rank-two-sis-security.md) that the main witness
commitment remains `κ = 18`. It does not change the accepted auxiliary
rank-two limits in that decision.

It does not change `b = 2`, `k_rho = 16`, `B = 2^16`, the `2^28` domain,
Poseidon2 protocol binding, the selected application, or any Stage 2 rule.

## Required closure after approval

After the rank and seed are selected:

1. recompute and prove the complete concrete footprint and `2^28` bound;
2. emit the final package and all binding values;
3. pin the final package identity in Rust;
4. rerun exact A/B/C equality and independent assignment evaluation;
5. rerun all Lean, `paper_exact`, optimized, handoff, and mutation gates; and
6. obtain exact-cut external review before any Conformance-closed claim.

Proof-backend work remains separately owner-controlled.
