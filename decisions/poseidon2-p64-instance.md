# Poseidon2 Instance over `2^64 − 59`

## Status

Accepted. The owner approved this instance on 2026-11-14. Nightstream and
ProtogaLattice use one shared rule and one shared generator, with the S-box
degree α chosen per project. Nightstream uses `x^7`. ProtogaLattice uses
`x^3`. Each project keeps its own code.

## Problem

Poseidon2 makes each Fiat–Shamir challenge of the folding scheme and each
in-circuit hash. No standard Poseidon2 instance exists for the field of
[`field-2p64-59-ring-x64.md`](./field-2p64-59-ring-x64.md). The two projects
use the same field, so one rule and one generator let each project make the
instance of the other again and check its hashes.

## SuperNeo

SuperNeo v1.2 does not fix a hash. It models the Fiat–Shamir hash as a random
oracle.

## Decision

`scripts/poseidon2/derive.py` derives the instances for widths 16 (rate 12)
and 12 from the S-box degree `SBOX_DEGREE`. `scripts/poseidon2/check.py`
checks them again with other algorithms and exits with an error on a failed
check. `scripts/poseidon2/params.json` holds the parameters and
`scripts/poseidon2/vectors.json` holds known answers. Lean and Rust take
their constants from `params.json` and test against `vectors.json`. For a
check between the two projects, Lean emits
`formal/nightstream-fprime/artifacts/nightstream-fprime-poseidon2-vectors-v1.json`
(`Export/Poseidon2VectorsMain.lean`): the width-16 shape, six permutations of
`Spec.Poseidon2.permute`, the first of the words 0 to 15, and 52 sponge
hashes of `Spec.Poseidon2.hash`, for the lengths 0 to 25.

Each instance has a label in `params.json`, for example
`Nightstream/Poseidon2/q=2^64-59/t=16/d=7/rf=8/rp=22/v2`. The owner approved
the version `v2` on 2026-11-14: the round constants changed from a SHAKE128
stream to the Grain generator, and two constant sets must not share one
label. The label is an input of the three `shake` test states of each width
in `vectors.json` only. It is not an input of a constant.

| Parameter | Nightstream | ProtogaLattice |
|---|---|---|
| S-box | `x^7`; `gcd(7, q − 1) = 1` (`Spec.Poseidon2.coprime_seven`), so the S-box is a permutation (`Spec.Poseidon2.sbox_bijective`) | `x^3`; `gcd(3, q − 1) = 1` (`Spec.Poseidon2.coprime_three`) |
| Rounds, for both widths | 8 full, 22 partial | 8 full, 43 partial |

The two projects have the same values for the other parameters:

| Parameter | Value |
|---|---|
| External matrix | `M_E = circ(2·M4, M4, …)` with `M4 = circ(2, 3, 1, 1)`, the Plonky3 `MDSMat4` |
| Internal matrix | `J + diag(D)` |
| `D`, width 16 | `[1, 2, −2, 3, −3, 4, −4, −5, 6, −6, 7, −7, 8, −8, 9, −9]` |
| `D`, width 12 | `[1, 2, −2, 3, −3, 4, −4, 5, 6, −6, −7, 8]` |
| Round constants | the Grain LFSR of the Poseidon2 reference generator (HorizenLabs `poseidon2_rust_params.sage`) for `q`, the width, `R_F` and `R_P`, in round order |

The sponge hash `Spec.Poseidon2.hash` uses `10*` padding: after the full
blocks of 12 words, the last block holds the `r < 12` other words in lanes 0
to `r − 1` and 1 in lane `r`, so the padding is injective
(`Spec.Poseidon2.blocks_injective`).

The layer conventions are those of Plonky3 `p3-poseidon2` 0.5.3.

The shared rule:

- The S-box degree α has `gcd(α, q − 1) = 1`. Each project selects it. The
  power map is a permutation for each such α
  (`Spec.Poseidon2.pow_bijective_of_coprime`).
- The Poseidon2 reference rule with its margin gives `R_F` and a least
  `R_P`. `R_P` is then the least value at or above it for which the
  resultant bound `α^(2.5(R_F − 1) + 1.5 R_P)` of ePrint 2026/1905,
  Section 6, is at least `2^128`. That bound does not depend on the width.
  For `x^7` the reference rule gives 8 + 22 and the bound is `2^141.8`. For
  `x^3` the reference rule gives 8 + 42, where the bound is `2^127.6`, so
  `R_P` is 43, where the bound is `2^130.0`.
- The constants come from the public reference generator, so each project
  can make them again. The round counts are an input of the generator, so
  the two projects have different constants.
- `D` is the first candidate, in a fixed order of small integers, for which
  `M_I` has nonzero entries, is invertible, and has an irreducible
  characteristic polynomial for each power `M_I^i`, `1 ≤ i ≤ 2t`
  (ePrint 2023/323, Section 5.3). Small entries keep the native internal
  layer cheap.
- `M4` is MDS over `F_q`: each square minor is nonzero.

Why Nightstream uses `x^7`: the F′ gate `A·B − C + S^7` has degree 7, so
one row computes one S-box. With `x^3` a permutation has 171 S-box rows, 21
more than the 150 rows with `x^7`, and each hash has 43 partial rounds. A
measurement gave a step prove time that was 5.7% longer with `x^3` at
8 + 43, and prover speed comes first.

### Open points

- The round counts come from the formulas of the papers and from the
  resultant bound of ePrint 2026/1905. They are not a cryptanalysis of this
  instance. That paper's bound rests on heuristics that it does not prove.
- The condition on `M_I` stops at `i = 2t`. The paper calls it sufficient.
