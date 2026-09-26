# Nightstream

Nightstream is a research proving system that combines SuperNeo folding for
CCS with HyperNova Construction 2. The active field
is Goldilocks with a degree-two extension. Ajtai commitments bind witnesses,
and protocol transcripts use Poseidon2 only.

Nightstream is not production-ready and has not had an independent audit.

Applications use [`nightstream`](crates/nightstream/README.md). It compiles
application circuits into the Lean-exported F′ relation and provides the
public proving and verification lifecycle. Rust CI tests this crate.

## Implemented paths

- SuperNeo NIFS: PiCCS, PiRLC, and PiDEC
- Optimized and PaperExact reduction engines
- HyperNova-style recursive R1CS F' induction
- Terminal R1CS compilation and a WIP Spartan proof over WHIR
- Metal acceleration for the canonical one-joint prover on supported Apple builds
- A required CUDA backend target that fails explicitly until its canonical
  device kernel is implemented

The application compiler and terminal verifier use the same exported relation.
The selected Nightstream Goldilocks profile uses `b = 2`, `k_rho = 16`,
`B = 65536`, and the wide PiRLC sampler.

## Main crates

| Crate | Ownership |
|---|---|
| `nightstream` | Application circuits, prepared packages, CPU/Metal proving, and terminal verification |
| `nightstream-fprime` | Exported package loading, witness execution, and matrix interpretation |
| `neo-reductions` | Optimized and PaperExact SuperNeo reductions |
| `neo-ccs` | CCS and committed-evaluation relation types |
| `neo-ajtai` | Ajtai setup, commitments, and openings |
| `neo-math` | Goldilocks, extension-field, and ring arithmetic |
| `neo-transcript` | Poseidon2 Fiat-Shamir transcript |
| `wip-spartan` | Direct sparse-R1CS Spartan proof with WHIR |
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
timeout 300s cargo test -p wip-spartan --release
```

Run `cargo fmt --all` after Rust changes. All non-Lean test commands have a
five-minute cap. See [AGENTS.md](AGENTS.md) for the full project rules.

For Lean proof work, [lean-graph](scripts/lean_graph/README.md) records proof
obligations, runs and resumes validation checkpoints, and answers dependency
queries. Its current configuration covers the Nightstream F′ pilot/PiCCS chain.

## Papers and implementation notes

- [SuperNeo v1.2 paper](docs/superneo-paper-v1_2/)
- [HyperNova paper](docs/hypernova-paper/)
- [Nebula paper](docs/nebula-paper/)
- [Wiki](wiki/index.md)
- [Active Lean proof work](formal/nightstream-fprime/CONSTRAINT_TREE.md)
- [Lean evidence workflow design](docs/trellis-nightstream-proposal.md)
