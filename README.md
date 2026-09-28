# Nightstream

Nightstream is a research proving system that combines SuperNeo folding for
CCS with HyperNova-style recursion. The active field
is Goldilocks with a degree-two extension. Ajtai commitments bind witnesses,
and protocol transcripts use Poseidon2 only.

Nightstream is not production-ready and has not had an independent audit.

Application integrations use [`nightstream`](crates/nightstream/README.md).
There is one maintained assembly and witness path. Rust CI checks the runtime;
local Lean validation checks the active `formal/nightstream-fprime` package.

## Implemented paths

- SuperNeo NIFS: PiCCS, PiRLC, and PiDEC
- Optimized and PaperExact reduction engines
- HyperNova-style recursive R1CS F' induction
- Terminal verification of the fresh relation and running commitment openings
- Metal acceleration for the canonical one-joint prover on supported Apple builds
- A required CUDA backend target that fails explicitly until its canonical
  device kernel is implemented

The recursive package contains the application and verifier rows. The wide
PiRLC sampler uses four transcript field elements per challenge and has no
rejection or shortfall path. Its statistical bound does not replace the
explicit Fiat–Shamir and MSIS assumptions in the security argument.

## Main crates

| Crate | Ownership |
|---|---|
| `nightstream` | Application circuits, prepared packages, CPU/Metal proving, and terminal verification |
| `neo-reductions` | Optimized and PaperExact SuperNeo reductions |
| `neo-ccs` | CCS and committed-evaluation relation types |
| `neo-ajtai` | Ajtai setup, commitments, and openings |
| `neo-math` | Goldilocks, extension-field, and ring arithmetic |
| `neo-transcript` | Poseidon2 Fiat-Shamir transcript |
| `neo-prover-metal` | Apple Metal prover work |
| `neo-prover-cuda` | Required CUDA backend target |

## Prover choices

| Choice | Status |
|---|---|
| `Optimized CPU` | Implemented; default host prover |
| `PaperExact` | Implemented reference path; exponential cost |
| `Metal` | Implemented on Apple builds with the production shader library |
| `Cuda` | Required WIP target; no silent CPU fallback |

The proof format and verifier do not depend on the prover choice.

## Build and check

```sh
cargo build --release
timeout 300s cargo test -p neo-reductions --release
timeout --signal=KILL 300 cargo test -p nightstream --release
```

Run `cargo fmt --all` after Rust changes. All non-Lean test commands have a
five-minute cap. See [AGENTS.md](AGENTS.md) for the full project rules.

For Lean proof work, [lean-graph](scripts/lean_graph/README.md) records proof
obligations, runs and resumes validation checkpoints, and answers dependency
queries. Fresh runtime and Lean comparisons use the
[golden conformance workflow](scripts/GOLDEN_CONFORMANCE.md).

## Papers and implementation notes

- [SuperNeo v1.2, September 4](docs/superneo-paper-v1_2/)
- [HyperNova paper](https://eprint.iacr.org/2023/573)
- [Wiki](wiki/index.md)
- [Active Lean proof work](formal/nightstream-fprime/CONSTRAINT_TREE.md)
- [Lean evidence workflow design](docs/trellis-nightstream-proposal.md)
