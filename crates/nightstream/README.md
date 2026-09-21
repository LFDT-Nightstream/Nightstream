# Nightstream

Rust application circuits, a generic assembler for the Lean-exported recursive
verifier, and the native `prepare`, `prove`, `extend`, and `verify` lifecycle.
Lean is used only by maintainers to produce and check the shared exports.

The package includes the selected verifier blueprint at
`artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json`.
Copy this file into the application's assets and pass its bytes to
`Circuit::prepare`. The shared manifest and formula library are included by
the Rust build. No Lean installation or artifact generation is needed.

```rust
use nightstream::{application::poseidon2_hash_chain_v1, Circuit};

let reference = std::fs::read("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")?;
let circuit = Circuit::prepare(&reference, poseidon2_hash_chain_v1()?)?;
```

Use the packaged blueprint as local verifier configuration. `prepare` checks
the selected reference identities before it inserts the Rust application.
Keep the prepared `Circuit` for later proving and verification.
Call `prove` for the first step, `extend` for each later step, and `verify`
with the state expected by the application.

`prepare` uses the optimized CPU engine. Select an engine explicitly with:

```rust
use nightstream::{application::poseidon2_hash_chain_v1, Circuit, Engine};

let circuit = Circuit::prepare_with_engine(
    &reference,
    poseidon2_hash_chain_v1()?,
    Engine::Optimized,
)?;
```

Engine selection applies to active PiCCS, PiRLC, and PiDEC proving. Preparation,
the exported relation, the fixed commitment key, and terminal verification keep
the same meaning. The engine is not part of the circuit identity. There is no
automatic fallback when a selected engine is unavailable.

| Engine | Current status |
| --- | --- |
| `Optimized` | Existing optimized CPU path. |
| `PaperExact` | Direct paper formulas with original exported matrix rows; checked against optimized CPU. |
| `Crosscheck` | Runs PaperExact and Optimized in parallel. Returns the optimized result only if all proof fields, accumulator values, witness values, and transcript state and cursor match. |
| `Metal` | Device fixed-key commitments, PiCCS evaluation, PiDEC openings, and terminal row checks. Small comparisons and two complete production C/R/D proofs match CPU results. Requires the `metal` feature and an Apple Metal device. |
| `Cuda` | The `cuda` feature connects the availability boundary. Selection fails explicitly because the canonical CUDA kernel is missing. |

Metal retains its device session and matrix plan across folds. PiRLC, witness
generation, and verifier control flow use the shared host code. Fixed-key
commitments, terminal row arithmetic, and nonzero running openings use Metal.
The GPU dependencies do not add `neo-fold-clean` to the production graph.

Parity checks follow `PaperExact ↔ Optimized`, then
`Optimized ↔ Metal ↔ Cuda`. They compare complete C/R/D proof bytes, transcript
state, returned claims, and witness matrices. The small fixtures include
nonzero carried data and completion tails. The selected production polynomial
has a check with two distinct rows across PaperExact, Optimized, and Metal.
Its nonzero input ports use two witness columns. These are
small-input conformance checks. They pass with compact norm storage and the
carried projection. The first production PiCCS proof also matches every CPU
proof byte, including output openings. Both complete production C/R/D folds
match all 945,983 CPU proof bytes. The second fold also matches all sixteen
returned child matrices and feeds a successful Metal successor and terminal
check. These separate phases do not establish the 5× full lifecycle target.
See [the production comparison](VALIDATION.md#nonzero-carried-production-fold).
The CUDA comparisons are explicitly ignored until its kernel exists.

Metal rejects a buffer request that would put its tracked device allocation
above 16 GB. This does not establish the total CPU-plus-GPU memory bound for
every circuit; that requires the complete lifecycle measurements.

Cache construction keeps coefficient runs compact and takes about 5 s on the
saved production input, down from about 55 s. Device commitments generate the
fixed key in threadgroup tiles and match CPU results: 3.08 s versus 97.42 s for
the full fresh witness. See [the commitment checks](VALIDATION.md#device-fixed-key-commitments).

The current three-step Metal benchmark passes in 179.51 s with 16.78 GB peak
RSS. This includes preparation, the base step, two active folds, and terminal
verification. Both full fold proofs and returned child matrices match CPU
references. The full CPU benchmark is still needed to establish the 5× ratio.
See [the current lifecycle result](VALIDATION.md#cpu-opening-buffer-reuse).

The current CPU storage path also passes the second production PiCCS proof
comparison: 182.43 s and 17.10 GB RSS (15.93 GiB), below the working 16 GiB guard.
Openings reuse the completed SumCheck buffer. Full CPU lifecycle timing is
still open. See [the CPU storage record](VALIDATION.md#cpu-opening-buffer-reuse).

Row evaluation reads signed masks directly, application storage shrinks with
each round, and zero openings avoid matrix scratch. Decomposition releases the
parent witness after producing its signed digits. Witness masks are written
directly into shared Metal storage, with no full host mask vector. Opening forms
and CPU storage still need bounded evaluation. The 5× lifecycle target and
general 16 GB bound remain open.
See [the storage measurements](VALIDATION.md#application-and-parent-storage).

Select `Engine::Crosscheck` in `Circuit::prepare_with_engine` to check each
active fold. The engines receive separate copies of the same inputs and
transcript. PaperExact reads the original exported matrix rows. A difference
or prover error fails the call. The caller's transcript advances only after
the comparison succeeds. The base step has no
active fold, so `prove` alone does not run this cross-check.

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --features metal,cuda --lib engine::parity
```

The five-minute test cap comes from the repository's `AGENTS.md`. Build the
test executable separately with `--no-run` when compilation needs its own
invocation. The `cuda` availability boundary needs no CUDA SDK. The existing
driver-backed CUDA crate still requires its pinned cuda-oxide build workflow
when its own `cuda` feature is enabled.

PaperExact is a reference evaluator with exponential work in the joint-domain
dimension. Both `PaperExact` and `Crosscheck` are for small correctness checks;
neither is practical for the selected production circuit.
The engine name does not change the Nightstream Goldilocks profile or make that
profile an exact copy of SuperNeo Appendix B.2.

## Stored Lean Poseidon2 checks

The application tests use saved Lean exports and execution results. They compare
all application rows and witness values, then run the stored base and recursive
inputs through the Rust circuit. Computed outputs must match the stored Lean
outputs, and the witnesses must satisfy all 7,700 exported application rows.
A changed output must fail those constraints. These tests run by default and
need no Lean installation or artifact generation.

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --test application_poseidon2
```

## Poseidon2 benchmark

`nightstream-poseidon2-bench` measures one chain through the public `Circuit`
API. It uses the Rust Poseidon2 application and the selected `b = 2`,
`k_rho = 16` profile. Each run reports preparation, base proving, every active
`extend`, and terminal verification as JSON lines. Preparation includes loading
the packaged reference and building the application. The prepared circuit is
reused for all steps. Native Poseidon2 supplies the expected state, and a run
finishes successfully only after terminal verification accepts it.

The binary uses normal dependencies and has no dependency on the old crate.
Run engines separately with the same step count, then compare their logs
manually. The owner's Metal target is at least 5× faster over the full lifecycle,
including preparation, proving, and terminal verification. The current memory
target is at most 16 GB for either engine on every supported circuit; 8 GB is
the future mobile target. The owner has accepted approximately 16 GB for this
pass, using RSS, and deferred further memory tuning. The general bound and full
lifecycle speed target have not passed. Inputs are fixed and
included in the first record. The step count
includes the base step: `--steps 1` performs no active fold, while `--steps 3`
covers the base and two active folds. PaperExact remains a small-input parity
reference; CUDA selection fails until its kernel is available.

Build before timing:

```sh
cargo build -p nightstream --release --bin nightstream-poseidon2-bench --features metal
```

On macOS, run the compiled binary under GNU `timeout` and the OS process timer:

```sh
/usr/bin/time -l timeout --foreground --signal=KILL 300 target/release/nightstream-poseidon2-bench --engine optimized --steps 3 > cpu.jsonl 2> cpu.time
/usr/bin/time -l timeout --foreground --signal=KILL 300 target/release/nightstream-poseidon2-bench --engine metal --steps 3 > metal.jsonl 2> metal.time
```

`maximum resident set size` in the `.time` file is peak process memory in bytes
on macOS. It is separate from GPU allocation counters. On Linux, build the CPU
binary without `--features metal` and use `/usr/bin/time -v`; its peak RSS is
reported in KiB. Compilation is excluded from these measurements. Each command
measures one chain; there is no warm-up or automatic speedup threshold.

The 300-second cap comes from `AGENTS.md`. Phase records are flushed as they
finish. A timeout leaves partial measurements and no `benchmark_finished`
record; it is not a successful lifecycle result. The full production chain
can exceed this cap. A longer invocation requires explicit approval for that
specific run. `--foreground` keeps the timeout process alive to reap the killed
benchmark, so the process timer can retain its resource usage.

Applications use four Goldilocks state words, private inputs, affine operations,
multiplication, and equality constraints. The assembler keeps every required
verifier component and binds the resulting application and circuit identity.

The current `Circuit` uses a prefix of the selected production commitment key.
Private input words plus generated local words must be at most **7,701**. This
comes from the existing layout
`252,695,531 + 41 × (private_words + local_words)` and the selected key's
253,011,276-coefficient capacity. The exported row and domain checks also apply.
A larger application needs a separately supported key; preparation rejects it.

The selected profile remains `b = 2`, `k_rho = 16`, `B = 2^16`, with one fresh
claim and sixteen carried claims. Terminal verification checks all remaining
openings. A production compression backend is outside this crate goal.

The independent Rust Poseidon2 application supplies the first comparison with
the existing Lean package. Exact comparisons are implementation evidence, not
a universal Lean proof of Rust application semantics or assembly. The project
contract is `NIGHTSTREAM_CRATE_GOAL.md` in the repository root. See the
[artifact instructions](artifacts/README.md) for consumer files, saved tests,
and the separate maintainer workflow.

The Cargo package includes the saved test inputs. Tests use package-local data
and do not run Lean. `neo-fold-clean` is a development dependency for comparison
with the unchanged implementation; it is not a production dependency.

See [VALIDATION.md](VALIDATION.md) for the completed fresh two-fold replay,
full reference comparisons, terminal checks, measured costs, and scope limits.
The original migration checks passed. Engine speed and memory requirements
remain open. A single-process active `extend` run and a universal Rust
refinement proof are not claimed.
