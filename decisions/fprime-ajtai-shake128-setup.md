# F′ Ajtai SHAKE128 Key Expander

## Status

Accepted by the owner on 2026-09-29. It supersedes only the expander rules of
[`fprime-stage1-main-ajtai-setup.md`](./fprime-stage1-main-ajtai-setup.md).

## Problem

The seed is public, so anyone can compute the key. Pseudorandomness of the
expander under a secret key therefore gives no security argument for the
commitment. The previous premise was MSIS hardness for one fixed
ChaCha20-generated matrix
([`PUBLIC_SEED_MSIS_ASSUMPTION.md`](../docs/reviews/nightstream-fprime-requirements/PUBLIC_SEED_MSIS_ASSUMPTION.md)).
We found no published argument for ChaCha20 as a public-matrix expander.

## SuperNeo

SuperNeo `Setup` samples the commitment matrix uniformly at random (Definition
8), and binding follows from MSIS for a uniform matrix (Theorem 6). The paper
does not specify how to derive the matrix from a seed.

## Decision

Keep the rank, dimensions, seed, 256-bit wide reduction and authority layout.
Replace the expander:

- setup ID `nightstream-ajtai-shake128-wide256-v1` (37 bytes);
- key element `(row, block)` is SHAKE128 (FIPS 202) of the 81-byte input
  `setup_id || seed || row_u32_le || block_u64_le`;
- coefficient `lane` is output bytes `32 * lane` to `32 * lane + 31`, read as
  one little-endian integer and reduced modulo the Goldilocks prime; and
- no rejection, fallback, retry, or materialized full key.

Kyber and FrodoKEM expand public matrices with SHAKE128 over an indexed input
and model SHAKE128 as a random oracle. This construction uses the same model.
It uses no rejection sampling; the wide reduction bounds the bias.

The intended security theorem assumes:

- SHAKE128 as a random oracle; and
- Module-SIS hardness for a uniform matrix under the pinned estimator model.

It also uses the wide-reduction bound and low-norm invertibility, which Lean
proves.

The reduction from these premises to binding of the expanded key is
[`SECURITY_ARGUMENT.md`](../docs/reviews/ajtai-key-expander/SECURITY_ARGUMENT.md).
It is not yet proved in Lean or externally reviewed. Until it is, the
fixed-matrix premise stays open for the new matrix. The seed-and-domain policy
questions in that document stay open.
