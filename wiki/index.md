# Nightstream Wiki

The current prover API and benchmark are in [`nightstream`](../crates/nightstream/README.md).
Nightstream combines SuperNeo folding with HyperNova-style recursion over
the prime field of `2^64 − 59`, Ajtai commitments, and Poseidon2 transcripts. The active formal
authority is `formal/nightstream-fprime`.

Lean emits a light and a refresh step circuit. The Rust application takes
the place of the Lean application in both. The state has two slots, one for
the accumulator of each circuit. The verifier checks the fresh instance and
the claims of both slots. `Prover::compress` makes a final proof without
witnesses over the Akita commitment scheme, and `Verifier::verify_final`
checks it. This part is implemented and tested in Rust. Lean bounds the
soundness of its argument in a model with ideal Akita openings.

The code remains research software. Independent review and the concrete
Fiat–Shamir security argument are separate from component conformance.

## Sections

| Section | Content |
|---|---|
| [Getting started](getting-started.md) | Build and code orientation |
| [Glossary](glossary.md) | Paper symbols and code names |
| [Specification](../SPEC.md) | The protocol: parameters, encodings, transcript, algorithms |
| [Architecture](architecture/index.md) | Crate and module ownership |
| [Frontends](architecture/frontends.md) | Application programs and package binding |
| [Crates](crates/index.md) | Per-crate reference |
| [Testing](development/testing.md) | Test rules and active checks |
| [Formal](formal/index.md) | Lean projects and evidence boundaries |
| [Security](security.md) | Assumptions and open work |
| [Roadmap](roadmap.md) | Required implementation work |
