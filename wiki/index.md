# Nightstream Wiki

Nightstream is a **post-quantum proving system**: a lattice-based folding scheme for
**CCS** (SuperNeo, building on Neo, ePrint 2025/294) composed with a
**HyperNova-style recursive IVC layer** (Construction 2). The active path proves CCS
over the **Goldilocks** field with a degree-2 extension for sum-check soundness,
**Ajtai (module-SIS) commitments**, and a **Poseidon2-only** Fiat-Shamir transcript.
The in-tree **Toy Spartan** + WHIR backend is standalone and is not connected to
terminal compression.

> **Status**: research software under active development. `neo-fold-clean` is the main
> proving crate. The earlier `neo-fold-prototype` sandbox (RV32IM/CHIP-8 pipelines) has
> been removed from the tree. The compressed Spartan decider ("PR5") is not wired yet;
> `lifecycle::compress` / `lifecycle::verify` return an explicit unsupported error.
> Not production-ready, not independently audited.

## The one-paragraph mental model

Each IVC step folds a batch of fresh CCS instances into a running accumulator of `k`
low-norm CE (committed-evaluation) claims using SuperNeo's three-reduction chain
`Π_CCS → Π_RLC → Π_DEC` (= `NIFS` in HyperNova terms). HyperNova's Construction 2
turns that folding scheme into IVC: an augmented function `F′` re-runs `NIFS.V`
in-circuit and hash-chains the public state (`x_out`). At the end, a decider checks the
final accumulator — today via direct relation checks and chain replay, eventually via a
compact backend that has not yet been selected or connected.

## Sections

| Section | What it covers |
|---|---|
| [Getting started](getting-started.md) | Build, test, lifecycle quickstart, where to read code |
| [Glossary](glossary.md) | Paper symbols ↔ code identifiers |
| [Protocol](protocol/index.md) | How SuperNeo (folding) and HyperNova (IVC) compose |
| — [SuperNeo folding](protocol/superneo-folding.md) | Relations, Π_CCS / Π_RLC / Π_DEC, norm control |
| — [HyperNova IVC](protocol/hypernova-ivc.md) | Construction 2, F′, NIFS, the state chain |
| — [Parameters](protocol/parameters.md) | Appendix B.2 Goldilocks profile, soundness bounds |
| — [Transcript & digests](protocol/transcript-and-digests.md) | Fiat-Shamir binding, digest authority rules |
| [Architecture](architecture/index.md) | Workspace layering and the `neo-fold-clean` module map |
| — [Lifecycle API](architecture/lifecycle.md) | `prove` / `extend` / `finish` / `verify`, the two verifier paths |
| — [Frontends](architecture/frontends.md) | direct-CCS, the F′ shell, R1CS-F′, Bellpepper |
| — [Decider](architecture/decider.md) | Terminal compression: statement, audit R1CS, terminal CE |
| [Crates](crates/index.md) | Per-crate reference for all 8 workspace members |
| [Testing](development/testing.md) | Test layout, red-team suites, project test policies |
| [Profiling](development/profiling.md) | Perf snapshots and profiling scripts |
| [Formal (Lean)](formal/index.md) | The five Lean subprojects |
| [Security](security.md) | Security model, soundness boundaries, known gaps |
| [Roadmap](roadmap.md) | What works, what is in progress |
