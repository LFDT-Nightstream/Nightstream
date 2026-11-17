# Nightstream

Nightstream is a research proving system that combines SuperNeo folding for
CCS with HyperNova-style recursion. The active field
is Goldilocks with a degree-two extension. Ajtai commitments bind witnesses,
and protocol transcripts use Poseidon2 only.

Nightstream is not production-ready and has not had an independent audit.

Application integrations use [`nightstream`](crates/nightstream/README.md).
Lean emits two step circuits, light and refresh, and a finish circuit. The
Rust application takes the place of the Lean application in both step
circuits. Rust CI checks the runtime. Local Lean validation checks the active
`formal/nightstream-fprime` package.

## Implemented paths

- SuperNeo NIFS: PiCCS, PiRLC, and PiDEC
- Optimized and PaperExact reduction engines
- HyperNova-style recursive R1CS F' induction with two accumulator slots, a
  light fold below the depth cap and a refresh fold at the cap
- Verification of the fresh instance and the claims of both slots
- Compression of the final proof with `neo-spartan`, with one layer-1 proof
  for each slot and a shrink proof
- A required CUDA backend target that fails explicitly until its canonical
  device kernel is implemented

Each step package contains the application and verifier rows. The wide
PiRLC sampler uses four transcript field elements per challenge and has no
rejection or shortfall path. Its statistical bound does not replace the
explicit Fiat–Shamir and MSIS assumptions in the security argument.

## Main crates

| Crate | Ownership |
|---|---|
| `nightstream` | Application circuits, proving, verification and compression |
| `neo-reductions` | Optimized and PaperExact SuperNeo reductions |
| `neo-ccs` | CCS and committed-evaluation relation types |
| `neo-ajtai` | Ajtai setup, commitments, and openings |
| `neo-math` | Goldilocks, extension-field, and ring arithmetic |
| `neo-transcript` | Poseidon2 Fiat-Shamir transcript |
| `neo-prover-metal` | Apple Metal prover kernels for `Engine::Metal` |
| `neo-prover-cuda` | NVIDIA CUDA prover kernels for `Engine::Cuda` (cuda-oxide) |

## Prover choices

| Choice | Status |
|---|---|
| `Engine::Optimized` | Implemented on the CPU |
| `Engine::Metal` | Implemented on Apple GPUs with the `metal` feature; without it, returns an error; no CPU fallback |
| `Engine::Cuda` | Implemented on NVIDIA GPUs with the `cuda` feature (cuda-oxide toolchain); without it, returns an error; no CPU fallback |

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
`formal/compression-lean/scripts/crosschecks.sh` runs the Rust tests that
compare Rust with Lean.

## Papers and implementation notes

- [SuperNeo v1.2, September 4](docs/superneo-paper-v1_2/)
- [HyperNova paper](https://eprint.iacr.org/2023/573)
- [Wiki](wiki/index.md)
- [Lean package and step circuits](formal/nightstream-fprime/README.md#step-circuits)
- [Lean evidence workflow design](docs/trellis-nightstream-proposal.md)
