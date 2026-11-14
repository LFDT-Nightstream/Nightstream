# Nightstream

Nightstream is a research proving system that combines SuperNeo folding for
CCS with HyperNova-style recursion. The active field
is the prime field of `2^64 − 59` with a degree-two extension. Ajtai commitments bind witnesses,
and the transcripts of the folding scheme and of the step circuits use Poseidon2 only.
The Akita commitment scheme (ePrint 2026/1983) compresses the final proof. Akita uses
Keccak-family hashes for its own challenges.

Nightstream is not production-ready and has not had an independent audit.

Application integrations use [`nightstream`](crates/nightstream/README.md).
Lean emits two step circuits, light and refresh. The Rust application takes the place of the Lean application in both step
circuits. Rust CI checks the runtime. Local Lean validation checks the active
`formal/nightstream-fprime` package.

## Implemented paths

- SuperNeo NIFS: PiCCS, PiRLC, and PiDEC
- Optimized and PaperExact reduction engines
- HyperNova-style recursive R1CS F' induction with two accumulator slots, a
  light fold below the depth cap and a refresh fold at the cap
- Verification of the fresh instance and the claims of both slots
- Compression of a proof to a final proof without witnesses
  (`Prover::compress`, `Verifier::verify_final`). It is implemented and
  tested in Rust, and Lean bounds the soundness of its argument in a model
- A required CUDA backend target that fails explicitly until its canonical
  device kernel is implemented

Each step package contains the application and verifier rows. The wide
PiRLC sampler uses four transcript field elements per challenge and has no
rejection or shortfall path. Its statistical bound does not replace the
explicit Fiat–Shamir and MSIS assumptions in the security argument.

## Compressed proof

`Prover::compress` proves that witnesses exist for both slot claims and for
the fresh instance. `Verifier::verify_final` checks the final proof against
the expected state. [SPEC.md](SPEC.md) Algorithms 9, 10 and 11 give the
protocol, and [the decision record](decisions/akita-compression.md) gives
the design choices and their status.

The verifier does not read the Ajtai key. It reads a `CompressionKey` of
about 1 KB (`Verifier::compression_key`), which holds the Akita commitment
of the key rows that the circuit reads, as one polynomial. The key is a
function of the public seeds and of the circuit size, so the library pins
the key of each size, and a test computes each pinned key again from the
seeds. The prover makes its own `CompressionSetup` from its circuit
(`Prover::compression_setup`). No proof carries the key.
`Verifier::decode_compression_key` accepts only the bytes of the pinned key
of the circuit size, so a loaded key is checked, not trusted. One Akita
proof opens the key polynomial and the two polynomials of the prover.

These numbers are from runs of
`cargo test -p nightstream --release a_compressed_proof_verifies -- --ignored --nocapture`
on an M5 Ultra with 30 cores, with LTO off, for the Poseidon2 hash chain
with the cap 2. The times are the range of four runs. They are not a
benchmark.

| | Light circuit last | Refresh circuit last |
|---|---|---|
| Witness cells in `W` | 41.4 M | 68.4 M |
| Variables of `W` | 26 | 27 |
| `compress` | 2.5 to 2.7 s | 3.3 to 3.7 s |
| `verify_final`, 30 threads | 0.17 to 0.20 s | 0.22 to 0.25 s |
| `verify_final`, 1 thread | 0.9 to 1.1 s | 1.5 to 1.7 s |
| Final proof | 133.3 KB | 133.4 KB |

The compression setup takes 1.0 s and holds about 10 GB. To compute the
compression key again from the seeds, as the test of the pin does, takes
1.0 s on 30 threads and 19 s on one thread, with a peak of 6 GB. A verifier
does not make this computation.
The test process has a peak of 30 GB. Of the proof bytes, the Akita opening is 75.7 KB and
the claims are about 54 KB. Of the `verify_final` time, Akita is 9 ms on 30
threads and 31 ms on one thread, and the rest is one pass over the rows of
the matrices of each claim.

The verifier is not succinct: its time is linear in the matrices. Akita is
version 0.1.0 and has no audit. The error that Rust computes for the
argument, about `2^-119.02`, does not include the error of Akita, of the
PiCCS run or of Fiat–Shamir
([security model](formal/nightstream-fprime/SECURITY_MODEL.md), adversary 6).
The argument also needs the compression key to hold the commitment of the
Ajtai key.

## Main crates

| Crate | Ownership |
|---|---|
| `nightstream` | Application circuits, CPU proving and verification |
| `neo-reductions` | Optimized and PaperExact SuperNeo reductions |
| `ns-compress` | The argument of the compressed proof over Akita |
| `neo-ccs` | CCS and committed-evaluation relation types |
| `neo-ajtai` | Ajtai setup, commitments, and openings |
| `neo-math` | Field, extension-field, and ring arithmetic |
| `neo-transcript` | Poseidon2 Fiat-Shamir transcript |
| `neo-prover-metal` | Apple Metal prover kernels; `nightstream` does not use them now |
| `neo-prover-cuda` | Required CUDA backend target |

## Prover choices

| Choice | Status |
|---|---|
| `Engine::Optimized` | Implemented; the only engine that proves and verifies steps |
| `Engine::Metal` | Returns an error; no CPU fallback |
| `Engine::Cuda` | Required WIP target; returns an error; no CPU fallback |

`PaperExact` is a reference engine of `neo-reductions` only. Its cost is
exponential.

The proof format and verifier do not depend on the prover choice.

## Build and check

```sh
cargo build --release
timeout 300s cargo test -p neo-reductions --release
timeout --signal=KILL 300 cargo test -p nightstream --release
```

Run `cargo fmt --all` after Rust changes. All non-Lean test commands have a
five-minute cap. See [AGENTS.md](AGENTS.md) for the full project rules.

For Lean proof work, use `formal/nightstream-fprime/scripts/validate.sh` and
read [the project instructions](formal/nightstream-fprime/AGENTS.md).

## Papers and implementation notes

- [SuperNeo v1.2, September 4](docs/superneo-paper-v1_2/)
- [HyperNova paper](https://eprint.iacr.org/2023/573)
- [Wiki](wiki/index.md)
- [Lean package and step circuits](formal/nightstream-fprime/README.md#step-circuits)
- [Lean evidence workflow design](docs/trellis-nightstream-proposal.md)
