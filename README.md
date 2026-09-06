# Nightstream

Nightstream is a research proving system that combines SuperNeo folding for
CCS with HyperNova Construction 2 and Nebula memory checking. The active field
is Goldilocks with a degree-two extension. Ajtai commitments bind witnesses,
and protocol transcripts use Poseidon2 only.

Nightstream is not production-ready and has not had an independent audit.

## Implemented paths

- SuperNeo NIFS: PiCCS, PiRLC, and PiDEC
- Optimized and PaperExact reduction engines
- HyperNova-style recursive R1CS F' induction
- Nebula offline memory checking
- Direct CCS folding with full-history audit verification
- Terminal R1CS compilation and a WIP Spartan proof over WHIR
- Metal acceleration for the canonical one-joint prover on supported Apple builds
- A required CUDA backend target that fails explicitly until its canonical
  device kernel is implemented

The recursive R1CS and Nebula frontends compile the authoritative F' relation.
Their terminal verifier checks the Construction 2 induction. The direct CCS
frontend proves the supplied CCS instances and NIFS continuity; multi-chunk
direct CCS proofs still require the audit replay path.

## Main crates

| Crate | Ownership |
|---|---|
| `neo-fold-clean` | Lifecycle, F', NIFS composition, frontends, and terminal statement |
| `neo-reductions` | Optimized and PaperExact SuperNeo reductions |
| `neo-ccs` | CCS and committed-evaluation relation types |
| `neo-ajtai` | Ajtai setup, commitments, and openings |
| `neo-math` | Goldilocks, extension-field, and ring arithmetic |
| `neo-transcript` | Poseidon2 Fiat-Shamir transcript |
| `wip-spartan` | Direct sparse-R1CS Spartan proof with WHIR |
| `neo-prover-metal` | Apple Metal prover work |
| `neo-prover-cuda` | Required CUDA backend target |
| `neo-wasm` | WASM relation and Nebula integration |

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
timeout 300s cargo test -p neo-fold-clean --release --test nifs_round_trip
timeout 300s cargo test -p wip-spartan --release
```

Run `cargo fmt --all` after Rust changes. All non-Lean test commands have a
five-minute cap. See [AGENTS.md](AGENTS.md) for the full project rules.

For Lean proof work, [lean-graph](scripts/lean_graph/README.md) records proof
obligations, runs and resumes validation checkpoints, and answers dependency
queries. Its current configuration covers the Nightstream F′ pilot/PiCCS chain.

## Papers and implementation notes

- [SuperNeo paper](docs/superneo-paper/)
- [HyperNova paper](docs/hypernova-paper/)
- [Nebula paper](docs/nebula-paper/)
- [Wiki](wiki/index.md)
- [Active Lean proof work](formal/nightstream-fprime/CONSTRAINT_TREE.md)
- [Lean evidence workflow design](docs/trellis-nightstream-proposal.md)
