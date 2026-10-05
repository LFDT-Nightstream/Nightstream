# CPU commitment batching

This records the first optimization snapshot. The subsequent Pad and integer-limb
changes have [separate measurements](../cpu-pad-limbs-20261004/README.md).

The four-step Optimized CPU lifecycle falls from a 77.633821-second median
to 60.307949 seconds: **1.2873× speed, or 22.32% less time**. The requested
2× target, 38.816910 seconds, is **not achieved**.

## Measurement

The baseline is `5d79b19d643cbef4f3199d04764bc26587d2769f`. The candidate is
the local change whose source-file hashes are in `results.json`. Measurements
use an Apple M5 Max, 18 CPUs, 128 GiB RAM, macOS 27.0.1, and Rust 1.94.1.
The same prepared Poseidon2 package performs a base step, three active folds,
and terminal verification. The benchmark's existing minimum-security policy
is 114 bits. This is not a new security claim. Compilation and package
preparation are outside the timed interval; package loading remains inside it.

| Run | Lifecycle seconds |
|---|---:|
| Original baseline 1 | 77.633821 |
| Original baseline 2 | 78.342499 |
| Original baseline 3 | 74.449618 |
| Original executable, control rerun | 77.537552 |
| Candidate 1 | 61.638413 |
| Candidate 2 | 60.307949 |
| Candidate 3 | 60.076043 |

The three final candidate runs peak at 8,693,792,768 bytes RSS. Each command
uses the maintained 300-second deadline and 16-GiB RSS guard. Commands run
sequentially; sleep prevention is active during final validation. No GPU,
reduced step count, changed protocol parameter, or reduced verification is
used. This measures the Rust lifecycle, not independent Lean generation.

## Grounding and design

The caller continues to load a `Circuit`, create its `Prover` and `Verifier`,
and call `prove`, `extend`, and `verify`. The circuit already shares its
retained matrix window between these owners.

An active extension constructs PiCCS messages and openings, combines the
witnesses with transcript-derived challenges, splits the parent into digit
witnesses, computes their commitments and openings, and executes the next
physical and packed logical witness. Terminal verification independently
recommits the witnesses and checks all Pad and matrix openings and the fresh
relation. Those responsibilities and checks remain in place.

The baseline sample puts its largest CPU costs in indexed key expansion and
signed convolution sums. The sample counts are thread totals, not phase wall
times. Source history `e8cb58727` explains the existing exact integer
accumulation: split coefficients into 32-bit halves, accumulate signed shifts
in i64, and reduce after a task. This change preserves that bound and reduction.

Three designs were compared serially. Repository instructions prohibit
subagents, so this was not an independent multi-agent design review.

| Design | Assessment |
|---|---|
| Retain an expanded immutable key across folds | Eliminates repeated hashing, but the final witnesses alone touch 1,463,919 key columns. Their canonical coefficients require 13,913,086,176 bytes. Keeping that data resident through the existing lifecycle requires a different memory design to fit the 16-GiB guard. Rejected for this change. |
| Keep the current owners and improve exact commitment kernels | Selected. It preserves public interfaces, requires no persistent key cache, and has independent scalar parity checks. |
| Commit children and the next fresh witness in one pass | The fresh witness consumes the child commitments. It cannot be constructed before that pass finishes. Rejected because of the actual dependency order. |

The selected design keeps the complexity below the existing commitment API:

```rust,ignore
let commitment = commit_production_signed_unit_prefix_matrix(&witness)?;
let commitments = commit_production_signed_unit_prefix_matrices(&witnesses)?;
```

`nightstream_fprime_setup.rs` still validates dimensions and every signed-unit
coordinate before expansion. It builds the union of nonzero column addresses
once, then shares it across the 22 key rows. Each worker processes consecutive
pairs from that ordered list. Witness order and key addresses remain fixed.

The private `nightstream_fprime_setup/key.rs` owns:

```rust,ignore
fn coefficient_pair(seed: &[u8; 32], row: u32, columns: [u64; 2])
    -> [[u64; 54]; 2];
```

Its `KeyPair` adapter holds those inputs and the output buffer required by
RustCrypto's backend API. The library selects the CPU permutation. ARM SHA3
uses two lanes; the portable backend processes the same inputs sequentially.
Odd tails use one result. Framing, SHAKE128 padding, all 24 rounds, output
length, and Goldilocks reduction are unchanged. The user explicitly approved
the existing upstream `keccak/parallel` dependency feature. There is no new
Nightstream feature or environment variable, and no project unsafe code.

`neo-math/src/signed_sums.rs` borrows each complete shifted slice before its
loop. This lets LLVM see the fixed loop length. It performs the same additions,
subtractions, and final reductions. No new public arithmetic interface is
required. The design adds no caller-managed cache, stage API, wire format,
or forwarding layer.

The accepted tradeoff is two small dependencies and a private SHAKE framing
adapter in exchange for batched hardware use. The unchanged scalar SHAKE
implementation remains the independent coefficient oracle. Full key storage
and a new protocol hash are not part of this design.

## Validation

The final receipts and verification status are in `results.json` and `logs/`.
All 14 native phases passed. A direct byte comparison also matched 78 files
(1,462,666,359 bytes), including every folding proof, child witness, caller
input, physical assignment, and saved state/fresh-witness file selected for
comparison. The three complete benchmark runs verified successfully.
Checks include:

- All paired coefficients against the independent SHAKE128 implementation,
  with varied seeds, key rows, boundary columns, and repeated addresses.
- The same test with the portable backend forced at build time.
- Existing signed-sum, fixed-key fixture, scalar commitment, batch commitment,
  dimension, norm, zero, and column-range tests.
- A fresh base and all three folds, with complete proof bytes compared to the
  original executable's saved results.
- Fresh successor construction, terminal acceptance, wrong-state rejection,
  recommitted relation-mutation rejection, and separate balanced `Eval_K`
  and `Eval_A` rejection checks.

CI includes the focused arithmetic and commitment tests. The Lean source,
package, transcript, key setup, and proof format do not change. No new Lean
build or independent Lean generation is claimed here.

Formatting and `git diff --check` pass. Strict Clippy is blocked by two
pre-existing warnings: `manual_contains` in `neo-ccs/src/sparse.rs:73`, and
`manual_is_multiple_of` in `neo-ajtai/src/s_module.rs:79` after restricting
the check to the changed crates. Both lines are unchanged from the baseline;
the failure logs are retained. No unrelated lint repair is included.

## What the 2× target still requires

A separate diagnostic run of the candidate took 63.712482 seconds: 33.973425
seconds in commitment calls and 29.739057 seconds elsewhere. Temporary timers
were removed before the final measurements. If all other work stayed at that
diagnostic level, commitments would need to fall to 9.077853 seconds, another
3.74× improvement, to reach the original target. Kernel-level speedups must
not be presented as whole-lifecycle speedups.

The next design must reduce more of the actual arithmetic, or reduce work in
both commitments and openings/SumCheck. Its owners should remain `neo-ajtai`
for indexed key expansion and commitment scheduling, `neo-math` for exact
convolution, and `neo-reductions::superneo_eval` for opening forms. A different
exact convolution algorithm needs a measured complete-commitment benefit and
the same scalar/proof-byte/rejection checks before adoption. This report does
not establish such an algorithm or promise that it will reach 2×.

## Reproduce

Build the CPU benchmark in release mode. Compile the reference application
once to a fresh package path, outside the timed run. Then use the same package
for both source versions:

```sh
cargo build -p nightstream --release --locked --bin nightstream-poseidon2-bench
timeout --signal=KILL 300 target/release/nightstream-poseidon2-bench compile --output PACKAGE_PATH
timeout --signal=KILL 300 target/release/nightstream-poseidon2-bench run --package PACKAGE_PATH --engine optimized --steps 4 --minimum-security-bits 114
```

The local measurements also use `run_recursive_phase.py`'s RSS/deadline guard.
Large native checkpoints, executable copies, and the sample remain in
`/tmp/nightstream-rust-2x-20261004/`; they are not committed fixtures. The
original baseline evidence is identified in `results.json`.
