# Poseidon2 Instance over `2^64 − 59`

## Status

Proposed. The owner must review the choices below. A cryptographer should
review the open points.

## Problem

Poseidon2 makes each Fiat–Shamir challenge of the folding scheme and each
in-circuit hash. The prior instances were the Plonky3 instances for
Goldilocks. No standard Poseidon2 instance exists for the field of
[`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md).

## SuperNeo

SuperNeo v1.2 does not fix a hash. It models the Fiat–Shamir hash as a random
oracle.

## Decision

`scripts/poseidon2/derive.py` derives the instances for widths 16 and 12.
`scripts/poseidon2/check.py` checks them again and exits with an error on a
failed check. `scripts/poseidon2/params.json` holds the parameters and
`scripts/poseidon2/vectors.json` holds known answers. Lean and Rust take
their constants from `params.json` and test against `vectors.json`.

| Parameter | Value |
|---|---|
| S-box | `x^7`; `gcd(7, q − 1) = 1` (`Spec.Poseidon2.coprime_seven`), so the S-box is a permutation (`Spec.Poseidon2.sbox_bijective`) |
| Rounds | 8 full, 22 partial, for both widths |
| External matrix | `circ(2·M4, M4, …)` with the Plonky3 `MDSMat4` |
| Internal matrix | `J + diag(D)` |
| `D`, width 16 | `[1, 2, −2, 3, −3, 4, −4, −5, 6, −6, 7, −7, 8, −8, 9, −9]` |
| `D`, width 12 | `[1, 2, −2, 3, −3, 4, −4, 5, 6, −6, −7, 8]` |
| Round constants | SHAKE128 of a label that names `q`, the width, the degree, and the round counts; 8-byte little-endian words; a word that is not below `q` is rejected |

The layer conventions and the order of the constants are those of Plonky3
`p3-poseidon2` 0.5.3. So the shape of the step-circuit template does not
change. Only the constants change.

The rules that select the values:

- The round counts are the cheapest pair that meets the 128-bit bounds of the
  Poseidon and Poseidon2 papers with the usual margin. The same search gives
  the Plonky3 table for 64-bit primes.
- `D` is the first candidate, in a fixed order of small integers, for which
  `M_I` has nonzero entries, is invertible, and has an irreducible
  characteristic polynomial for each power `M_I^i`, `1 ≤ i ≤ 2t`
  (ePrint 2023/323, Section 5.3).
- `M4` is MDS over `F_q`: each square minor is nonzero.

### Open points

- The round counts come from the formulas of the papers. They are not a
  cryptanalysis of this instance. Attacks published after ePrint 2023/537
  were not reviewed.
- The degree `x^3` needs 42 partial rounds and about 43% fewer
  multiplications in R1CS for each permutation. It changes the shape of the
  template. The degree `x^7` was kept because it keeps that shape.
- The condition on `M_I` stops at `i = 2t`. The paper calls it sufficient.
- SHAKE128 replaces the Grain LFSR of the reference generator. It runs only
  in the derivation, not in a proof path.
