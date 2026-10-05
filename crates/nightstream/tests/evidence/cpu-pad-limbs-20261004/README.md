# CPU opening and key-limb optimizations

The same four-step Optimized CPU lifecycle takes **57.809367 seconds median**.
The 38.816910-second target remains **unmet**. The original median was
77.633821 seconds. The previous candidate had a 60.307949-second median;
its saved executable took 64.210162 seconds in this session.
Use that control to distinguish the new changes from variation between sessions.

| Measurement | Lifecycle seconds |
|---|---:|
| Original baseline median | 77.633821 |
| Previous candidate, control rerun | 64.210162 |
| Final candidate 1 | 59.562066 |
| Final candidate 2 | 57.809367 |
| Final candidate 3 | 57.365073 |
| Final candidate median | 57.809367 |

The benchmark still includes package loading, a base step, three active folds,
and terminal verification on the same saved package and inputs. Compilation
and one-time package preparation remain outside the timed interval. The existing
114-bit minimum-security policy is unchanged; this report makes no new security
claim. Final validation uses the maintained 300-second deadline and 16-GiB RSS guard.
The three final runs peak at 9,347,104,768 bytes RSS. The native checks peak at
9,430,728,704 bytes. Build and test commands run sequentially with sleep prevention.

## Changes and ownership

`neo-reductions::superneo_eval` factors the Pad equality weights before ring
multiplication. A degree-54 block crosses at most two 64-lane tensor intervals
and has 32 possible alignment phases. It accumulates weighted witnesses by
phase, then multiplies those aggregates by the corresponding ring forms.
It uses exact distributivity and no inversion, including at zero/one points.
The same module skips empty matrix scans and limits a pure identity scan to
the current worker's actual coordinate range. All 14 matrix families remain.

`neo-math::SplitRing` can represent the original 256-bit key coefficient with
two bounded integer limbs before field reduction. `neo-ajtai` uses those limbs
directly from the unchanged SHAKE128 output. Integer bounds still fit i64 for
the maximum approved key prefix. Seed, framing, key addresses, hash functions,
profile, transcript and proof format stay unchanged.

These changes preserve the existing public lifecycle calls and ownership.
There is one production execution path, no persistent key cache, and no new
Nightstream feature or environment variable. The upstream `keccak/parallel`
feature was explicitly approved for the previous change. The serial design
comparison and algebra are in [DESIGN.md](DESIGN.md). Repository instructions
prohibit subagents; this is not an independent multi-agent review.

## Validation

The source fingerprints, commands, receipts and individual results are in
`results.json` and `logs/`. Fingerprints identify source; they do not establish
correctness. The checks compare actual coefficients and complete file bytes.

- Three signed-sum tests, two Pad formula tests, 15 matrix tests, the independent
  SHAKE pair test and 13 fixed-key/commitment tests pass. One existing commitment
  test remains ignored. The pair test also passes with the portable Keccak backend.
- All 14 fresh native phases pass: a base, three proofs and successors, terminal
  acceptance, a recommitted relation mutation and rejection, and separate balanced
  `Eval_K` and `Eval_A` preparation/rejection checks.
- The three complete proof byte comparisons pass. A direct comparison also
  matches all 78 selected files (1,462,666,359 bytes)
  against the original baseline: proof, PiCCS/result, caller, physical assignment,
  child witnesses, and state/fresh-witness files.

No Lean source or saved protocol artifact changes. No new Lean validation or
GPU comparison is claimed. Large fresh checkpoints remain in the scratch
directory recorded in `results.json`; they are not committed fixtures.

## What the 38-second target needs

The original profile put more than half of the candidate runtime in commitments.
The bounded changes above help, but do not remove enough of that work.
A separate exact finite-field transform prototype was tested on a real
65,536-column witness tile. Its code was removed from the production tree.

| Tile commitment, milliseconds | Eight witnesses | One witness |
|---|---:|---:|
| Current shift sums, including key expansion | 410.338 | 210.364 |
| Transform method, including key expansion | 358.819 | 249.487 |
| Cached ordinary key | 268.686 | 68.428 |
| Cached transformed key | 145.717 | 30.923 |

Cached timings exclude construction: about 158–159 ms for this ordinary key
tile and another 66–73 ms for its transform. Those costs must be inside any
cold lifecycle measurement. The single-witness regression rejects uncached
transforms as a general replacement. Tile results do not establish whole-run
speed, memory bandwidth, or a 38-second result.

The full benchmark key alone would take 15,259,774,464 bytes in ordinary form
or 36,171,317,248 bytes in transformed form, before the working data. Keeping
the transformed key violates the current 16-GiB contract. No such run occurred.
The user was asked whether to keep that limit or approve one larger-memory
experiment. That decision is pending. If approved, an immutable key derived by
the compiled circuit could be shared by prover and verifier; both must still
recompute commitments from the current witness. Cache construction, all existing
checks, complete proof bytes and the exact memory peak must count in the result.

With the current memory limit, reaching 38 seconds still requires a measured
bounded-memory commitment improvement or further reduction of opening/SumCheck
work. No tested design yet proves that target. More memory is a candidate
experiment, not an established solution or a change to the production requirement.

## Reproduce

Build the release CPU benchmark, then use the same saved package for both
source versions. Package creation is a separate operation.

```sh
cargo build -p nightstream --release --locked --bin nightstream-poseidon2-bench
timeout --signal=KILL 300 target/release/nightstream-poseidon2-bench run --package PACKAGE_PATH --engine optimized --steps 4 --minimum-security-bits 114
```

These measurements additionally use `run_recursive_phase.py`'s memory guard.
The local `validate.py` and `measure.py` in the recorded scratch directory
contain the exact sequential run and fresh-proof commands.
