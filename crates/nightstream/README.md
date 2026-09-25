# Nightstream

Rust application circuits, an assembler for the exported recursive verifier,
and separate compilation, loading, proving and verification. Production builds
and execution do not require Lean.

Compile the application once using the packaged reference, then save it:

```rust
use nightstream::{application::poseidon2_hash_chain_v1, Circuit};

let reference = std::fs::read("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")?;
let circuit = Circuit::compile(&reference, poseidon2_hash_chain_v1()?)?;
circuit.write("app.nsc")?;
```

Compilation validates the selected reference and application, and computes the
unchanged Poseidon2 identities. Writing publishes one complete file and refuses
to replace an existing destination. The compiled circuit can also be reused
directly in the same process.

Load the saved data and select execution engines:

```rust
use nightstream::{Circuit, Engine, Verifier};

let package = Circuit::load("app.nsc")?;
// Set this from the integrating application's security policy.
let minimum_security_bits = policy.minimum_statistical_security_bits;
let prover = package.prover(Engine::Metal, minimum_security_bits)?;
let verifier = Verifier::from_package(&package, Engine::Metal, minimum_security_bits)?;
let proof = prover.prove(initial_state, &private_inputs)?;
verifier.verify(&expected_state, &proof)?;
```

Loading checks the format, layout, dimensions and recipe structure. It does not
repeat whole-circuit identity hashing. `Circuit::identity()` returns the saved
claim; comparing it with a known value does not authenticate the loaded data.
`Prover::load(path, engine, minimum_security_bits)` is available
when only proving data is needed. `Verifier::compile` derives expected
configuration from a local application. The caller selects the verifier's
expected configuration and is responsible for its provenance; the crate does
not impose an authentication policy. Proof data cannot replace that configured
relation.

Prover and verifier creation require an explicit, positive statistical-security
minimum. Preparation rejects a profile below that minimum. The current stored
Poseidon2 profile's estimator reports 114 bits; this is a statistical estimate,
not a claim about the complete system's security. The application selects its
own acceptance policy. Compilation and loading do not change that policy.

`prover.extend(&proof, inputs)` returns a new proof. The supplied proof remains
usable if extension fails. Packed witness buffers are shared and immutable;
the prior proof remains live until the caller replaces or drops it.

Engine selection applies to proof arithmetic and terminal row checks. It does
not change the circuit identity, formulas or fixed commitment key. An unavailable
engine returns an error without a CPU fallback.

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
The GPU dependencies do not add `neo-fold-legacy` to the production graph.

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

Before the matrix-row window change, matching Time Profiler runs measured the
full three-step lifecycle at **1,256.02 s
on Optimized and 164.74 s on Metal: 7.62×**. Both use the same saved executable,
inputs, profile, and circuit identity, and verify the same final state. This
includes preparation, the base step, two active folds, and terminal verification.
The ratio is measured under profiling; a full CPU run without Instruments is
still pending. The separate Metal run without Instruments takes 164.66 s with
16.81 GB peak RSS. A fresh second-fold check matches every CPU proof byte and
all returned child matrices.
See [the matched lifecycle profiles](VALIDATION.md#matched-full-lifecycle-profiles).
GPU traces identify commitments and opening products/forms as the main device
costs. See [the profile](tests/evidence/metal-gpu-profile-20260921).

The earlier CPU storage path also passed the second production PiCCS proof
comparison: 182.43 s and 17.10 GB RSS (15.93 GiB), below the working 16 GiB guard.
Openings reuse the completed SumCheck buffer.
See [the CPU storage record](VALIDATION.md#cpu-opening-buffer-reuse).

A separate Optimized CPU check accepts the stored Metal terminal state and
rejects a wrong final state. An earlier full CPU profile stopped during terminal
verification at the 30-minute cap; the saved build above completed within that cap.
The source review identified an accepted row shape that required a 27.51 GB
Metal application table. The new application owner uses bounded replay when
a resident prefix does not fit. CPU and Metal now also generate bounded matrix
windows directly from original rows. Application rows, recipe syntax and their
indexes now stay in sealed files; identity hashing replays their original order.
This removes the eager application-row storage gap. Total process RSS across
every supported circuit still needs separate evidence.
See [the CPU profile and memory gap](VALIDATION.md#cpu-lifecycle-and-terminal-profile).

Forced replay matches complete CPU proof bytes, transcript state, and openings.
A production second fold on that earlier build matched all saved CPU proof bytes
and all sixteen child matrices, with 16.38 GB peak RSS.
See [bounded application tables](VALIDATION.md#bounded-metal-application-tables).

The bounded matrix-window build measured **4.5597×** under matching Time
Profiler runs: CPU 1,330.70 s, Metal 291.84 s. This is below the 5× target.
The newer sealed-record and reusable-row-buffer build matches all 945,983 saved
CPU proof bytes and all sixteen complete child matrices. Its production second
fold takes 83.36 s with 13.19 GB peak RSS. Its complete Metal lifecycle takes
**264.52 s with 12.75 GB peak RSS** without Instruments. Matching Time Profiler
runs on that saved build give **4.7249×**: CPU 1,308.78 s and Metal 277.00 s.
Both verify; observed RSS peaks are 15.79 GB and 12.63 GB. That saved build
remained below the 5× target. Final post-exit RSS peaks are unavailable for the profiled runs.
See [matrix row windows](VALIDATION.md#bounded-matrix-row-windows) and
[sealed application records](VALIDATION.md#sealed-application-records-and-reusable-row-buffers).

Incremental row-storage counting and single-input template substitution reduce
the full Metal lifecycle to **250.28 s with 12.68 GB final peak RSS** without
Instruments. The production second fold takes 77.44 s and matches all saved CPU
proof bytes and complete outputs. No matched ratio was measured for that image.
See [the incremental row-count evidence](tests/evidence/incremental-row-count-20260922).

Recipe-stack reuse further reduces raw Metal to **244.83 s with 12.72 GB peak
RSS**. Matched Time Profiler runs on the same saved image measure **5.2782×**:
CPU 1,300.84 s and Metal 246.45 s. Both verify and remain below the accepted RSS
guard. These results include compilation. Before the fixed-metadata node guard,
matched prepared-package runs measured **5.9191×**, including loading, proving
and terminal verification but excluding compilation. The final image with the
guard measures **5.9332×**: CPU 1,262.10 s and Metal 212.72 s, with observed RSS
peaks of 14.45 GB and 11.47 GB. Both verify. Raw Metal takes **212.47 s with
11.37 GB final peak RSS**. See
[recipe-stack reuse](VALIDATION.md#recipe-stack-batch-reuse) and
[the final prepared-package comparison](VALIDATION.md#final-prepared-package-cpumetal-comparison).

The CPU key loader now avoids 128-bit division for streamed coefficients.
The production fresh commitment falls from 95.14 s to 58.38 s, with exact
saved-commitment equality. All supported engine comparisons pass. This is one
operation; the full lifecycle comparison above uses this change.
See [the key reduction result](VALIDATION.md#cpu-indexed-key-reduction).

Row evaluation reads signed masks directly, application storage shrinks with
each round, and zero openings avoid matrix scratch. Decomposition releases the
parent witness after producing its signed digits. Witness masks are written
directly into shared Metal storage, with no full host mask vector. Opening forms
and CPU evaluation now use bounded matrix windows. This removes the earlier
full matrix-run allocation. The general 16 GB RSS bound remains open; the
working-storage budget does not include all allocator and driver residency.
See [the storage measurements](VALIDATION.md#application-and-parent-storage).

Select `Engine::Crosscheck` in `circuit.prover` to check each
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

`nightstream-poseidon2-bench` has separate `compile` and `run` commands.
Compilation builds the Rust Poseidon2 application and reports `compile` and
`save` times. Each fresh `run` loads the saved package, creates the selected
prover and verifier, proves the base and active steps, and verifies the terminal
state. It emits schema2 JSON with scope `prepared_package_load_prove_verify`.
The total begins before package loading. Native Poseidon2 supplies the expected
state, and a run succeeds only after terminal verification accepts it.

The binary uses normal dependencies and has no dependency on the old crate.
Run engines separately with the same step count, then compare their logs
manually. The owner's Metal target is at least 5× faster over the full lifecycle,
including package loading, proving, and terminal verification. One-time circuit
compilation is excluded from that total and measured separately. The current
memory target is at most 16 GB for either engine on every supported circuit; 8 GB is
the future mobile target. The owner has accepted approximately 16 GB for this
pass, using RSS, and deferred further memory tuning. Storage is bounded across
supported native circuit shapes; a universal process-RSS guarantee remains
unproven because runtime and driver residency also contribute. The final guarded
image measures **5.9332×** under matched Time Profiler runs over loading, proving
and terminal verification. Both engines use the same caller-selected package,
inputs and final state. The raw CPU/Metal ratio remains unmeasured. Inputs are
fixed and included in the first record. The step count
includes the base step: `--steps 1` performs no active fold, while `--steps 3`
covers the base and two active folds. PaperExact remains a small-input parity
reference; CUDA selection fails until its kernel is available.

Build before timing:

```sh
cargo build -p nightstream --release --bin nightstream-poseidon2-bench --features metal
```

On macOS, run the compiled binary under GNU `timeout` and the OS process timer:

```sh
timeout --signal=KILL 300 target/release/nightstream-poseidon2-bench compile --output app.nsc > compile.jsonl
/usr/bin/time -l timeout --foreground --signal=KILL 300 target/release/nightstream-poseidon2-bench run --package app.nsc --engine optimized --steps 3 --minimum-security-bits 114 > cpu.jsonl 2> cpu.time
/usr/bin/time -l timeout --foreground --signal=KILL 300 target/release/nightstream-poseidon2-bench run --package app.nsc --engine metal --steps 3 --minimum-security-bits 114 > metal.jsonl 2> metal.time
```

These benchmark commands explicitly accept the retained fixture's 114-bit
estimate. They do not select a production security policy.

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
184,359,564-coefficient capacity. The exported row and domain checks also apply.
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
and do not run Lean. The crate has no production or development dependency on
`neo-fold-legacy`. The old timing baseline lives in the legacy crate.

See [VALIDATION.md](VALIDATION.md) for the completed fresh two-fold replay,
full reference comparisons, terminal checks, measured costs, and scope limits.
The original migration checks passed. The final prepared-package comparison
meets the 5× speed target and both measured RSS peaks stay below the accepted
16 GiB guard. A universal RSS guarantee and a universal Rust refinement proof
remain unproven.
