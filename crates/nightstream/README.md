# Nightstream

The `nightstream` crate proves the steps of an application recursively,
verifies the proofs, and compresses a proof into a final proof. Lean emits
three circuit packages in `formal/nightstream-fprime/artifacts`. They are the
light step circuit, the refresh step circuit and the finish circuit. The
crate embeds them as zstd copies (`artifacts/README.md`). Builds and proofs do
not run Lean.

## Compile an application

```rust
use nightstream::{application::poseidon2_hash_chain_v1, Circuit};

let circuit = Circuit::compile(poseidon2_hash_chain_v1()?)?;
circuit.write("app.nsc")?;
```

`Circuit::compile` puts the Rust application in place of the Lean application
in both step circuits of the three embedded packages. The reference hash chain
(`poseidon2_hash_chain_v1`) keeps Lean's packages, because it computes the step
of Lean's application region, so its circuit is the one that the Lean checks
read. The crate decodes the
packages once per process, and their identities are their pins
(`OPTION_B_PACKAGES_IDENTITY` and `OPTION_B_FINISH_IDENTITY` in
`nightstream-fprime`). A test checks that each embedded copy hashes to its
pin (`artifacts/README.md`). `Circuit::identity` returns the identity of
the packages with this application.

`Circuit::write` saves only the application. The destination must not exist.
A prover loads it with `Prover::load(path, engine, minimum_security_bits)`,
which compiles it again against the embedded packages. A verifier compiles its own circuit and never loads a saved
file. A memory relation has its verifier in `nebula::Circuit::verifier`.

## Prove and verify

```rust
use nightstream::{Engine, Prover, State};

let prover = Prover::load("app.nsc", Engine::Optimized, minimum_security_bits)?;
let verifier = circuit.verifier(Engine::Optimized, minimum_security_bits)?;
let first = prover.prove(initial_state, &private_inputs)?;
let second = prover.extend(&first, &private_inputs)?;
verifier.verify(&State::new(2, initial_state, expected_state), &second)?;
```

The caller sets `minimum_security_bits` from its own security policy. The
value must be positive. Prover and verifier creation fail when the
statistical security of a step relation is below this minimum. That
security is the zero-query error of a refresh fold in Lean's
random-oracle result, about `2^-109.4`, so the selected profile accepts
a minimum of at most 109 bits.

`Engine::Optimized` proves and verifies on the CPU. `Engine::Metal` needs the
`metal` feature and an Apple GPU. `Engine::Cuda` needs the `cuda` feature and
an NVIDIA GPU; the feature builds only through `cargo +nightly-2026-04-03
oxide`, the cuda-oxide codegen backend (`crates/neo-prover-cuda/README.md`).
A device engine runs the commitments, the Π_CCS round oracle, the Π_RLC mix,
the Π_DEC openings, the row checks and the compression loops on the device.
The host keeps the transcript, the proof assembly and every public check. Its
proofs and final witnesses are byte-identical to those of
`Engine::Optimized`. The shrink proof grinds with a parallel search, so its
bytes can differ between two runs on any engine. Without its feature, a
device engine returns `EngineError::Unavailable`. The crate does not use the
CPU in place of a device. A device failure, which includes a request over the
device memory ceiling, returns `Error::Engine`.

`prove` makes the proof of the first step from the initial state. `extend`
makes the proof of the next step. The proof that you give to `extend` does not
change. The Rust application computes the output state.

`State::new(iteration, z0, current)` is the expected statement. It has no
authority. `Verifier::verify` checks the proof and its witnesses against it.
At iteration 0, the current state must equal `z0`. At a later iteration, the
public input of the fresh instance must be the hash of the state with both
slots. The claim of each slot must hold under the relation of its circuit,
with a norm below `B`. The fresh instance must hold under the relation of the
circuit that ran last.

`Prover::encode_proof` returns the strict bytes of a proof with its
witnesses. The bytes start with the tag `NS-OPTIONB-PRF-1`.
`Verifier::decode_proof` checks the exact length before it allocates memory,
because the relations set every size. Each field word must be below the
modulus. Each column of the fresh witness is two lane masks, one for `+1` and
one for `-1`. The bytes do not include the frame digests, because no check
reads them. A decoded proof still needs `verify`, and you can extend it.

## Folding schedule

The state has two slots. Slot 0 holds the accumulator of the light circuit.
Slot 1 holds the accumulator of the refresh circuit. Each slot is one CE claim
and a depth.

The first step is the base step of the light circuit. Each later step folds
the fresh instance of the previous step into the slot of the circuit that made
that instance. There are two kinds of fold.

- A light fold runs when the depth of the slot is below 64. It runs Π_CCS
  over the fresh instance and the accumulator, then Π_RLC with the weights ρ
  and 1. It adds 1 to the depth.
- A refresh fold runs when the depth is 64. Π_DEC splits the accumulator into
  16 children. Then Π_CCS runs over the fresh instance and the 16 children,
  and Π_RLC runs with 17 sampled challenges. The depth becomes 0.

No fold ends with Π_DEC. With `cap = 64`, the first period of steps is
`(L^(cap+1) R)^(cap+1) R`, and each later period is `(L^(cap+1) R)^cap R`
(`src/lifecycle/option_b/mod.rs`, `src/folding/option_b.rs`).

## Final proof

`Prover::compression_setup(dir)` writes the two layer-1 setups into the
existing directory `dir`, and returns them with their key.
`Prover::compress(&proof, &setup)` compresses a proof. The proof does not
change. The last fold is a refresh fold of the last fresh instance into its
slot. Then one layer-1 proof (`neo-spartan`) for each slot proves the claim of
that slot under the relation of its circuit. Last, a shrink proof shows that
the final program accepts. The final program is the rows of the finish
package and the two layer-1 verifiers. The final proof has no witness, and you
cannot extend it.

`Verifier::compression_key` derives the key from the relations, without
files. Use this key, or a copy that you pinned, as the authority for
`Verifier::verify_final`. Do not take the key from a prover.
`Verifier::verify_final(expected, key, final_proof)` checks a final proof
against the expected state. `formal/compression-lean` proves a soundness bound
for this final verifier under the assumptions in its `TRUST.md`.

## Applications

`nightstream::application` builds applications. An application reads four
Goldilocks state words and a number of private input words, and writes four
state words. `ApplicationBuilder` gives affine operations, multiplication and
equality assertions. `poseidon2_hash_chain_v1` is the reference application.
`poseidon2_hash_chain_step` computes its step natively.

`nightstream::nebula` builds Nebula memory applications on the same packages.
Use `nebula::Verifier` for a memory circuit. `Circuit::verifier` refuses a
memory circuit.

## Tests

The tests read the Lean packages in `formal/nightstream-fprime/artifacts` and
the recorded Lean outputs in [`tests/fixtures`](tests/fixtures/README.md).
They do not run Lean.

- `tests/application_poseidon2.rs` compares the Rust Poseidon2 application
  with the recorded Lean application, row by row and value by value.
- `tests/option_b_api.rs` runs the public API from compilation to
  `verify_final`.
- `tests/option_b_lifecycle/` holds the lifecycle tests. They compile into
  the library, so `--lib lifecycle::` selects them.
- `tests/option_b_lifecycle/device_parity.rs` (feature `metal` or `cuda`)
  compares `Engine::Metal` or `Engine::Cuda` with `Engine::Optimized`. Every
  device session of a process shares one device budget, so run these tests
  one at a time. Two of them are ignored, because the layer-1 setups take
  about two minutes on the CPU.

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --lib application::poseidon2_tests
```

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --features metal --lib device_parity -- --test-threads=1
```

## Poseidon2 benchmark

`nightstream-poseidon2-bench` has two commands. `compile` builds the Rust
Poseidon2 application and saves it. `run` loads the saved application with
`Prover::load` and compiles the verifier's own circuit. Then it proves
`COUNT` steps and verifies the last proof. `COUNT` includes the first step.
The binary writes one JSON line when each phase starts and stops. Only
`--engine optimized` can prove. The binary refuses a debug build.

```sh
cargo build -p nightstream --release --bin nightstream-poseidon2-bench
timeout --signal=KILL 300 target/release/nightstream-poseidon2-bench compile --output app.nsc > compile.jsonl
/usr/bin/time -l timeout --foreground --signal=KILL 300 target/release/nightstream-poseidon2-bench run --package app.nsc --engine optimized --steps 3 --minimum-security-bits 109 > cpu.jsonl 2> cpu.time
```

The 300-second cap comes from the repository `AGENTS.md`. A run that reaches
the cap fails. It keeps the records that it wrote before the cap, and it has
no `benchmark_finished` record. On macOS, `maximum resident set size` in the
`.time` file is the peak process memory in bytes. On Linux, use
`/usr/bin/time -v`, which gives the peak in KiB.
