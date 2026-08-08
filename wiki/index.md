# Nightstream Wiki

Nightstream combines SuperNeo folding for CCS, HyperNova Construction 2, and
Nebula memory checking. The active protocol uses Goldilocks, a degree-two
extension field, Ajtai commitments, and Poseidon2 transcripts.

The code is research software. It is not production-ready and has not had an
independent audit.

## Current model

Each step folds fresh CCS claims into a low-norm running accumulator through
PiCCS, PiRLC, and PiDEC. The recursive R1CS and Nebula frontends compile F' so
that a recursive step verifies the preceding NIFS fold. The terminal path
closes the accumulator relation and can prove its sparse R1CS with the WIP
Spartan and WHIR backend.

Optimized CPU and PaperExact paths are implemented. Metal performs device work
on supported Apple builds. CUDA is a required backend target, but its canonical
kernel is not implemented and selection fails explicitly.

## Sections

| Section | Content |
|---|---|
| [Getting started](getting-started.md) | Build and code orientation |
| [Glossary](glossary.md) | Paper symbols and code names |
| [Protocol](protocol/index.md) | SuperNeo, HyperNova, parameters, transcripts |
| [Architecture](architecture/index.md) | Crate and module ownership |
| [Frontends](architecture/frontends.md) | Direct CCS, recursive R1CS, and Nebula |
| [Decider](architecture/decider.md) | Terminal relation and WIP Spartan |
| [Crates](crates/index.md) | Per-crate reference |
| [Testing](development/testing.md) | Test rules and active checks |
| [Formal](formal/index.md) | Lean projects and evidence boundaries |
| [Security](security.md) | Assumptions and open work |
| [Roadmap](roadmap.md) | Required implementation work |
