# neo-fold-clean

The main proving crate: public lifecycle API, HyperNova Construction 2, the F′
recursive-step shell, the SuperNeo reduction sequencing, and the decider. Everything an
integrator or auditor touches starts here.

This page is a directory; the architecture section covers each part in depth.

| Module | Role | Deep dive |
|---|---|---|
| `lifecycle/` | The only public surface: `preprocess`, `prove`, `extend`, `finish_uncompressed*`, `compress`, `verify*`, `FoldSchedule` | [Lifecycle API](../architecture/lifecycle.md) |
| `paper/` | Paper-faithful protocol layer — relations, params, sampling, Π seams + verifier circuits, NIFS, Construction 2, F′ relation, digests, decider contract, terminal CE | [Protocol](../protocol/index.md), [Decider](../architecture/decider.md) |
| `frontends/` | direct_ccs, the F′ image shell, r1cs_f_prime, bellpepper | [Frontends](../architecture/frontends.md) |
| `engine/` | Optimized-engine seam, R1CS gadget builder, CCS-native Poseidon2, decider R1CS synthesis | [Decider](../architecture/decider.md) |
| `config.rs` | The audited parameter profile constants | [Parameters](../protocol/parameters.md) |

## Orientation rules

- Start with the crate docs in `src/lib.rs` (canonical lifecycle example) and the
  paper-symbol glossary in `src/paper/mod.rs`.
- Authority layering: `lifecycle` (public) → `paper` (auditable claims) → `engine`
  (no protocol authority) → lower crates.
- Public API uses lifecycle names only; internal state-machine constructors stay
  private behind the lifecycle entry points.

## Tests

The crate carries the workspace's heavyweight test suites — system/e2e, red-team,
gadget isolation, reduction circuits, and perf snapshots — organized under
`tests/{system, direct_ccs, f_prime, gadgets, nifs, reductions, perf, support}/`.
See [Testing](../development/testing.md).
