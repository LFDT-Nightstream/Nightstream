# Nightstream

The `nightstream` crate proves the steps of an application recursively and
verifies the proofs. Lean emits the light step circuit and the refresh step
circuit as packages in `formal/nightstream-fprime/artifacts`. The crate
embeds them as zstd copies (`artifacts/README.md`). Builds and proofs do
not run Lean.

## Compile an application

```rust
use nightstream::{application::poseidon2_hash_chain_v1, Circuit};

let circuit = Circuit::compile(poseidon2_hash_chain_v1()?)?;
circuit.write("app.nsc")?;
```

`Circuit::compile` puts the Rust application in place of the Lean application
in both embedded step circuits. The reference hash chain
(`poseidon2_hash_chain_v1`) keeps Lean's packages, because it computes the step
of Lean's application region, so its circuit is the one that the Lean checks
read. The crate decodes the
packages once per process, and their identity is its pin
(`BOUNDED_DEPTH_PACKAGES_IDENTITY` in `nightstream-fprime`). A test checks
that the embedded copies hash to the pin (`artifacts/README.md`). `Circuit::identity` returns the identity of
the packages with this application.

`Circuit::write` saves only the application. The destination must not exist.
A prover loads it with `Prover::load(path, engine, minimum_security_bits)`,
which compiles it again against the embedded packages. A verifier compiles its own circuit and never loads a saved
file. A memory relation has its verifier in `nebula::Circuit::verifier`.

## Prove and verify

```rust
use nightstream::{Engine, Prover, State};

// This gates one fold. Account for proof depth and all attempts separately.
let minimum_security_bits = policy.minimum_per_fold_security_bits;
let prover = Prover::load("app.nsc", Engine::Optimized, minimum_security_bits)?;
let verifier = circuit.verifier(Engine::Optimized, minimum_security_bits)?;
let first = prover.prove(initial_state, &private_inputs)?;
let second = prover.extend(&first, &private_inputs)?;
verifier.verify(&State::new(2, initial_state, expected_state), &second)?;
```

The caller sets `minimum_security_bits` from its own security policy. The
value must be positive. Prover and verifier creation fail when the
statistical security of a step relation is below this minimum. The gate
charges one error for both folds, which bounds the zero-query statistical
term of either fold in Lean's random-oracle result. Rust computes the error
and its bits in integers, and Lean emits the same bits: 109 for each step
relation (`SPEC.md` section 3). So the selected profile accepts a minimum of
at most 109 bits.

`minimum_security_bits` is not an end-to-end security claim. It gates the
selected one-fold error. A Nebula circuit also compares it with one memory
fingerprint challenge. The deployment must account separately for recursive
depth, adversary queries, failed retries, and concurrent attempts.

Steps prove and verify on the CPU path only (`Engine::Optimized`).
`Engine::Metal` and `Engine::Cuda` return `EngineError::Unavailable`. The
crate does not use the CPU in their place.

`prove` makes the proof of the first step from the initial state. `extend`
makes the proof of the next step. The proof that you give to `extend` does not
change. The Rust application computes the output state.

`State::new(iteration, z0, current)` is the expected statement. It has no
authority. `Verifier::verify` checks the proof and its witnesses against it.
At iteration 0, the current state must equal `z0`. At a later iteration, the
public input of the fresh instance must be the hash of the state with both
slots. The claim of each slot must hold under the relation of its circuit.
The depth of the slot must be at most 64, and the norm of its witness must be
at most `(17 + depth) · 128`, the bound of an honest slot at that depth. So
`extend` works on each proof that `verify` accepts. The fresh
instance must hold under the relation of the circuit that ran last.

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
(`src/lifecycle/bounded_depth/mod.rs`, `src/folding/bounded_depth.rs`).

## Applications

`nightstream::application` builds applications. An application reads four
field state words and a number of private input words, and writes four
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

- `tests/application/poseidon2.rs` compares the rows of the Rust Poseidon2
  application with the row digest of the Lean application.
- `tests/bounded_depth_api.rs` runs the public API from compilation to
  `verify`.
- `tests/bounded_depth_lifecycle/` holds the lifecycle tests. They compile into
  the library, so `--lib lifecycle::` selects them.

```sh
timeout --signal=KILL 300 cargo test -p nightstream --release --lib application::poseidon2_tests
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
