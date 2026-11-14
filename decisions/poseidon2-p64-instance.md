# Poseidon2 Instance over `2^64 − 59`

## Status

Accepted. The owner approved this instance on 2026-11-14. Nightstream and
ProtogaLattice use the same settings. Each project keeps its own code.

## Problem

Poseidon2 makes each Fiat–Shamir challenge of the folding scheme and each
in-circuit hash. No standard Poseidon2 instance exists for the field of
[`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md). The two projects
use the same field, so one instance lets each project check the hashes of the
other.

## SuperNeo

SuperNeo v1.2 does not fix a hash. It models the Fiat–Shamir hash as a random
oracle.

## Decision

`scripts/poseidon2/derive.py` derives the instances for widths 16 (rate 12)
and 12. `scripts/poseidon2/check.py` checks them again with other algorithms
and exits with an error on a failed check. `scripts/poseidon2/params.json`
holds the parameters and `scripts/poseidon2/vectors.json` holds known
answers. Lean and Rust take their constants from `params.json` and test
against `vectors.json`.

| Parameter | Value |
|---|---|
| S-box | `x^3`; `gcd(3, q − 1) = 1` (`Spec.Poseidon2.coprime_three`), so the S-box is a permutation (`Spec.Poseidon2.sbox_bijective`) |
| Rounds | 8 full, 43 partial, for both widths |
| External matrix | `M_E = circ(2·M4, M4, …)` with `M4 = circ(2, 3, 1, 1)`, the Plonky3 `MDSMat4` |
| Internal matrix | `J + diag(D)` |
| `D`, width 16 | `[1, 2, −2, 3, −3, 4, −4, −5, 6, −6, 7, −7, 8, −8, 9, −9]` |
| `D`, width 12 | `[1, 2, −2, 3, −3, 4, −4, 5, 6, −6, −7, 8]` |
| Round constants | the Grain LFSR of the Poseidon2 reference generator (HorizenLabs `poseidon2_rust_params.sage`) for `q`, the width, `R_F = 8` and `R_P = 43`, in round order |

The sponge hash `Spec.Poseidon2.hash` uses `10*` padding: after the full
blocks of 12 words, the last block holds the `r < 12` other words in lanes 0
to `r − 1` and 1 in lane `r`, so the padding is injective
(`Spec.Poseidon2.blocks_injective`).

The layer conventions are those of Plonky3 `p3-poseidon2` 0.5.3.

The rules that select the values:

- The S-box degree is the least `d` with `gcd(d, q − 1) = 1`. It is the
  degree of the step-circuit constraints, so `x^3` makes the CCS degree 3.
- The Poseidon2 reference rule with its margin gives `R_F = 8` and
  `R_P = 42` for both widths. `R_P` is then the least value at or above 42
  for which the resultant bound `d^(2.5(R_F − 1) + 1.5 R_P)` of ePrint
  2026/1905, Section 6, is at least `2^128`. That bound is `2^127.6` at 42
  and `2^130.0` at 43. It does not depend on the width.
- The constants come from the public reference generator, so each project
  can regenerate them.
- `D` is the first candidate, in a fixed order of small integers, for which
  `M_I` has nonzero entries, is invertible, and has an irreducible
  characteristic polynomial for each power `M_I^i`, `1 ≤ i ≤ 2t`
  (ePrint 2023/323, Section 5.3). Small entries keep the native internal
  layer cheap.
- `M4` is MDS over `F_q`: each square minor is nonzero.

### Open points

- The round counts come from the formulas of the papers and from the
  resultant bound of ePrint 2026/1905. They are not a cryptanalysis of this
  instance. That paper's bound rests on heuristics that it does not prove.
- The condition on `M_I` stops at `i = 2t`. The paper calls it sufficient.
