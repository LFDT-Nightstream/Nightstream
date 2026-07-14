# Crates

Workspace members (root `Cargo.toml`), lowest layer first:

| Crate | Role | Page |
|---|---|---|
| `neo-params` | Validated parameter bundles + the canonical Poseidon2 config | [neo-params](neo-params.md) |
| `neo-math` | Goldilocks `F`, extension `K`, ring `R_q`, bar transform, norms, S-action | [neo-math](neo-math.md) |
| `toy-spartan` | Standalone Spartan backend with Goldilocks/Poseidon2 WHIR | [toy-spartan](toy-spartan.md) |
| `neo-ccs` | CCS/CE relations, matrices, polynomial, R1CS→CCS | [neo-ccs](neo-ccs.md) |
| `neo-transcript` | Poseidon2 Fiat-Shamir transcript | [neo-transcript](neo-transcript.md) |
| `neo-ajtai` | Ajtai (module-SIS) commitments, decomposition, S-module | [neo-ajtai](neo-ajtai.md) |
| `neo-reductions` | Π_CCS / Π_RLC / Π_DEC engines (optimized + paper-exact) | [neo-reductions](neo-reductions.md) |
| `neo-fold-clean` | Main proving crate: lifecycle, Construction 2, F′, decider | [neo-fold-clean](neo-fold-clean.md) |

Dependency graph: see [Architecture](../architecture/index.md). The rule of thumb:
authority flows downward (the paper layer of `neo-fold-clean` trusts the engine
crates), data flows upward.

## Spec files

The lower crates carry RFC-style spec files under `specs/` documenting MUST/SHOULD
behavior; they are the per-crate normative reference:

- `neo-math/specs/` — Goldilocks, Ring, BarTransform, Norms, SAction
- `neo-ajtai/specs/` — AjtaiCommit, Decomposition, PublicParameters, SModule
- `neo-ccs/specs/` — Relations, Matrix, Polynomial, Gadgets, Crypto
- `neo-reductions/specs/` — PiCCS, PiRLC, PiDEC, SumCheck, Engines, SuperNeoEval
- `neo-transcript/specs/` — Transcript
