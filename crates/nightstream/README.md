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

`prover.encode_proof(&proof)` returns the proof's strict byte form.
`verifier.decode_proof(&bytes)` checks the exact length that its configured
circuit fixes before it allocates. It then rejects noncanonical field words,
malformed witness masks and trailing data. A decoded proof still needs
`verify`. The bytes carry exactly the values that `verify` reads, so each
accepted proof has one encoding; the frame digests and the private witness
copy are not encoded, because no check reads them. Each witness column travels as two 64-bit lane masks for `+1` and
`-1`; the fixed-key commitment already rejects other witness values.

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
The GPU dependencies use the maintained folding and commitment interfaces.

Parity checks follow `PaperExact ↔ Optimized`, then
`Optimized ↔ Metal ↔ Cuda`. They compare complete C/R/D proof bytes, transcript
state, returned claims, and witness matrices. The small fixtures include
nonzero carried data and completion tails. The selected production polynomial
has a check with two distinct rows across PaperExact, Optimized, and Metal.
Its nonzero input ports use two witness columns. These are
small-input conformance checks. They pass with compact norm storage and the
carried projection. The CUDA comparisons are explicitly ignored until its kernel
exists.

Metal rejects a buffer request that would put its tracked device allocation
above 24 GB. This does not establish the total CPU-plus-GPU memory bound for
every circuit; that requires the complete lifecycle measurements.

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
profile an exact copy of SuperNeo v1.2 Section 8.2.

## Stored Lean Poseidon2 checks

The application tests use saved Lean exports and execution results. They compare
all application rows and witness values, then run the stored base and recursive
inputs through the Rust circuit. Computed outputs must match the stored Lean
outputs, and the witnesses must satisfy all 5,484 exported application rows.
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
memory target on every supported circuit is at most 24 GB for Metal, so a prover can
run on a current iPhone, and at most 16 GB for the CPU engine; 8 GB is the future
mobile target. The owner has accepted approximately 16 GB for this
pass, using RSS, and deferred further memory tuning. Storage is bounded across
supported native circuit shapes; a universal process-RSS guarantee remains
unproven because runtime and driver residency also contribute. An earlier image
measured **5.9332×** under matched Time Profiler runs over loading, proving
and terminal verification. Both engines use the same caller-selected package,
inputs and final state. The current source does not meet the 5× target; see the
current measurements below. Inputs are
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

At `3cc867401`, with the width-16 Poseidon2, these commands measured the
complete lifecycle without Instruments on an Apple M5 Max. The CPU column is
one run. The Metal column is the median of five runs:

| Phase | Optimized | Metal |
| --- | ---: | ---: |
| package load | 1.09 s | 1.13 s |
| base step | 4.66 s | 1.38 s |
| fold, step 2 | 15.48 s | 6.23 s |
| fold, step 3 | 19.76 s | 7.63 s |
| terminal verification | 12.88 s | 3.27 s |
| **total** | **53.87 s** | **19.64 s** |
| peak RSS | 8.36 GB | 9.43 GB |

Metal is **2.74×** faster than the CPU engine, below the 5× target. In the same
session, the width-8 base `3cca62f59` took 85.97 s on the CPU engine (one run)
and 29.00 s on Metal (median of five runs, alternating with the width-16 runs).
The width-16 relation has 35% fewer coordinates, so the CPU engine is 37%
faster and Metal is 32% faster. Single Metal runs vary by up to 1 s. Both
engines are faster than the earlier image (CPU 1,262.10 s and Metal 212.72 s
under Time Profiler). Both RSS peaks are below 16 GB. See
[the run records](tests/evidence/poseidon2-benchmark-20260930).

Applications use four Goldilocks state words, private inputs, affine operations,
multiplication, and equality constraints. The assembler keeps every required
verifier component and binds the resulting application and circuit identity.

Each `Circuit` uses the production commitment-key prefix required by its
application. Private input words plus generated local words must be at most
**5,134,675**. The logical width is
`43,738,906 + 41 × (private_words + local_words)`, and the maximum key supports
254,260,620 scalar coordinates (4,708,530 ring columns). The reference application
uses 814,144 columns; this smaller prefix is not the application capacity limit.
The exported row and domain checks also apply. See the
[capacity derivation](tests/evidence/prepared-fixed-source-bound.md).

The selected profile remains `b = 2`, `k_rho = 16`, `B = 2^16`, with one fresh
claim and sixteen carried claims. Terminal verification checks all remaining
openings.

`Prover::finish_with_spartan` compresses an active proof on the Optimized or
Metal engine. It runs PiCCS and PiRLC once more and stops before PiDEC
(SuperNeo Lemma 1). The `neo-spartan`
crate then proves the folded CE(B) claim with sum-checks and WHIR (layer 1).
Last, a shrink layer proves that the final verifier accepts both, with
SuperSpartan and one WHIR commitment. The final verifier's PiCCS, PiRLC and
prior-state hash are the F′ rows themselves: `formal/compression-lean` takes
them from the F′ Stage 1 package without change and proves what they imply.
The crate embeds that artifact and pins its identifier. The rest of the final
verifier (statement, layer 1) is written once over `neo_spartan::Backend`.
The final proof carries no witness. Compression needs a one-time setup per
circuit: `Prover::compression_setup` writes the setup files (about 29 GB for
the production package) and returns the setup, whose `CompressionKey` is
1,988 bytes. `Verifier::verify_final` checks a final proof with that key and
the embedded terminal package: no files, no commitment key, no matrix rows.
It first refuses a key of another circuit or below the verifier's security
minimum. The first verification in a process loads the package (about 6 s).
Take the key from `Verifier::compression_key` (about 60 s, no files) or from
a copy you pinned; never from a prover. In the
production test, one fold gives a final proof of about 129-130 KB
(`FinalProof::to_bytes`; the shrink's proof-of-work changes the size a
little), against 221,738,816 bytes for the accumulator proof. Verification
takes about 305 ms. Finishing takes about 48 s on CPU and 30 s on Metal. The
peak memory footprint of the finish is about 9.3 GB on CPU. On Metal the last
fold still holds the 8 GB commitment key, so the peak is about 11.7 GB, and
layer 1 itself stays near 10 GB. These are medians of five runs, with
`MallocLargeCache=0`, on a loaded machine (2026-10-20). Before it returns,
the finish verifies its own final proof (about 0.4 s more), so a device fault
fails the call instead of returning a proof that does not verify. Layer 1
keeps its Merkle rows and digest layers and its GKR levels in an unlinked
scratch file,
so it also needs temporary disk space. On Metal the device hashes the WHIR
Merkle leaves, runs the WHIR DFTs and the GKR rounds, computes the key
weights one key row at a time and runs the last fold's PiCCS; the proof bytes
do not depend on the engine before the shrink's proof-of-work. The shrink commitment uses WHIR at rate 1/8: a smaller
proof (rate 1/64: 97.6 KB) costs about 30 s more to finish.

The independent Rust Poseidon2 application supplies the first comparison with
the existing Lean package. Exact comparisons are implementation evidence, not
a universal Lean proof of Rust application semantics or assembly. The project
contract is `NIGHTSTREAM_CRATE_GOAL.md` in the repository root. See the
[artifact instructions](artifacts/README.md) for consumer files, saved tests,
and the separate maintainer workflow.

The Cargo package includes the saved test inputs. Tests use package-local data
and do not run Lean.

[The benchmark](#poseidon2-benchmark) records the current CPU and Metal costs.
A universal RSS guarantee and a universal Rust refinement proof remain unproven.
