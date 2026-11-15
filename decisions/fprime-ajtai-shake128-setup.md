# F′ Ajtai SHAKE128 Key Expander

## Status

Accepted by the owner on 2026-09-29. It supersedes the ChaCha20 expander of
[`fprime-stage1-main-ajtai-setup.md`](./fprime-stage1-main-ajtai-setup.md).

[`fprime-ajtai-rejection-sampling.md`](./fprime-ajtai-rejection-sampling.md)
supersedes the first setup ID of this record,
`nightstream-ajtai-shake128-wide256-v1`, and its 256-bit wide reduction. The
decision below states the current rule.
[`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md) gives the modulus
and the rank.

## Problem

The seed is public, so anyone can compute the key. Pseudorandomness of the
expander under a secret key therefore gives no security argument for the
commitment. The earlier premise was MSIS hardness for one fixed
ChaCha20-generated matrix
([public-seed MSIS assumption](fprime-ajtai-public-seed-msis-assumption.md)).
We found no published argument for ChaCha20 as a public-matrix expander.

## SuperNeo

SuperNeo `Setup` samples the commitment matrix uniformly at random (Definition
8), and binding follows from MSIS for a uniform matrix (Theorem 6). The paper
does not specify how to derive the matrix from a seed.

## Decision

Expand the key with SHAKE128 (FIPS 202):

- key element `(row, block)` is a function of the SHAKE128 output of the
  input `setup_id || seed || row_u32_le || block_u64_le`
  (`Spec.AjtaiSetupV1.elementInput`);
- the seed is the 32-byte verifier-owned seed of
  [`fprime-stage1-main-ajtai-setup.md`](./fprime-stage1-main-ajtai-setup.md);
- the setup ID and the map from the output to the 64 coefficients are those
  of
  [`fprime-ajtai-rejection-sampling.md`](./fprime-ajtai-rejection-sampling.md);
  and
- there is no retry: one XOF call gives one key element.

[SPEC.md](../SPEC.md) Algorithm 1 is the rule.

Kyber and FrodoKEM expand public matrices with SHAKE128 over an indexed input
and model SHAKE128 as a random oracle. This construction uses the same model.

The security argument assumes:

- SHAKE128 as a random oracle (`[KECCAK]`); and
- Module-SIS hardness for a uniform matrix (`[MSIS]`), in the estimator model
  of [`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md).

[The security model](../formal/nightstream-fprime/SECURITY_MODEL.md) lists
both assumptions. Low-norm invertibility is not a premise, because Lean proves
it (`Spec.RingStrongSet.lowNormInvertibility`).

The reduction from these premises to binding of the expanded key is
[the security argument](fprime-ajtai-shake128-security-argument.md).
Lean proves its ideal-model steps
(`Spec.AjtaiSetupV1.Programming.binding_le_solver`,
`Export.BoundedDepth.RandomOracleSetup.programmingError_lt`). The premises
stay external, and no external review has checked the argument. The
seed-and-domain policy question of that record stays open.
