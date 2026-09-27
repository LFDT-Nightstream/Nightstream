# Nightstream Wiki

The current prover API and benchmark are in [`nightstream`](../crates/nightstream/README.md).
Nightstream combines SuperNeo folding with HyperNova-style recursion over
Goldilocks, Ajtai commitments, and Poseidon2 transcripts. The active formal
authority is `formal/nightstream-fprime`.

There is one application assembly and witness execution path. The wide
PiRLC sampler is part of its canonical layout. The old lifecycle, WASM bridge,
and Spartan consumer are retired. The current terminal verifier checks the
fresh relation and running witness openings.

The code remains research software. Independent review and the concrete
Fiat–Shamir security argument are separate from component conformance.

## Sections

| Section | Content |
|---|---|
| [Getting started](getting-started.md) | Build and code orientation |
| [Glossary](glossary.md) | Paper symbols and code names |
| [Protocol](protocol/index.md) | SuperNeo, HyperNova, parameters, transcripts |
| [Architecture](architecture/index.md) | Crate and module ownership |
| [Frontends](architecture/frontends.md) | Application programs and package binding |
| [Decider](architecture/decider.md) | Fresh relation and running openings |
| [Crates](crates/index.md) | Per-crate reference |
| [Testing](development/testing.md) | Test rules and active checks |
| [Formal](formal/index.md) | Lean projects and evidence boundaries |
| [Security](security.md) | Assumptions and open work |
| [Roadmap](roadmap.md) | Required implementation work |
