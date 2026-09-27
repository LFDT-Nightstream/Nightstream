# Crates

Workspace members (root `Cargo.toml`), lowest layer first:

| Crate | Role | Page |
|---|---|---|
| `neo-params` | Validated parameter bundles + the canonical Poseidon2 config | [neo-params](neo-params.md) |
| `neo-math` | Goldilocks `F`, extension `K`, ring `R_q`, bar transform, norms, S-action | [neo-math](neo-math.md) |
| `neo-ccs` | CCS/CE relations, matrices, polynomial, R1CS→CCS | [neo-ccs](neo-ccs.md) |
| `neo-transcript` | Poseidon2 Fiat-Shamir transcript | [neo-transcript](neo-transcript.md) |
| `neo-ajtai` | Ajtai (module-SIS) commitments, decomposition, S-module | [neo-ajtai](neo-ajtai.md) |
| `neo-reductions` | Π_CCS / Π_RLC / Π_DEC engines (optimized + paper-exact) | [neo-reductions](neo-reductions.md) |
| `nightstream` | Current circuit compilation, package loading, proving, and verification | [nightstream](../../crates/nightstream/README.md) |
| `nightstream-fprime` | Shared package format and exported verifier execution | [source](../../crates/nightstream-fprime/src/lib.rs) |

See [Architecture](../architecture/index.md) for the current ownership boundaries.

## Protocol authority

Crates do not own copied protocol specifications. Protocol-critical rules cite
the pinned paper, the selected decision record, or the active Lean model.
Executable behavior checks live in each crate's normal `tests/` directory.
