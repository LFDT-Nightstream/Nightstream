# Security

Nightstream is research software. Independent review and a complete numerical
security claim for concrete Poseidon2 execution remain open.

The formal argument keeps these boundaries explicit:

- Binding for the selected fixed-seed Ajtai key requires its same-key MSIS assumption.
- Concrete Poseidon2 requires the stated collision and Fiat–Shamir transfer assumptions.
- The wide sampler's bias bound assumes uniform field draws; it is not itself a random-oracle theorem for the sponge.
- The low-norm invertibility fact for the ring `X^64 + 1` is proved for the selected strong set.
- Completeness and soundness are separate obligations.

The verifier checks actual relation rows, commitments, norms, public values,
Pad openings, and all matrix openings. A carried digest must be recomputed
from authoritative data or replayed into the verifier-driven transcript.
Self-consistent rehashing is not evidence of a valid witness.

## The compressed proof

`Prover::compress` and `Verifier::verify_final` are implemented and tested
in Rust. They are not proved. Lean bounds the soundness of the argument
in a model with ideal Akita openings, but no Lean result joins it to the
terminal relation.

- The argument uses the Akita commitment scheme (ePrint 2026/1983) at one
  pinned git revision. Akita is version 0.1.0 and has no audit.
- Rust computes an error of about `2^-119.02` for one interactive run of
  the argument, for the reference application at 4 lookup challenges. This
  number does not include the knowledge error of the Akita opening, the
  error of the PiCCS run of the fresh instance, or the loss of Fiat–Shamir.
  Lean bounds the argument with `Q` oracle queries by `Q` times an error
  below `2^-119`, with ideal Akita openings.
- `verify_final` does not read the Ajtai key. It reads a compression key,
  which holds the Akita commitment of the key rows and Akita's commitments
  of a prefix of its own setup. The library pins the key of each circuit
  size, and a test computes each pinned key again from the public seeds.
  The verifier takes the pinned key of its circuit
  (`Verifier::compression_key`). No proof and no prover value carries it.
  The argument is sound only when the key holds the commitments of those
  public tables. So `Verifier::decode_compression_key` accepts loaded bytes
  only when they equal the pinned key of the circuit size.
- Akita states a target of a 128-bit Fiat–Shamir rate for each query in the
  classical random-oracle model, and a price of 169 bits of quantum
  Core-SVP cost for its Module-SIS instances. It states that these are not
  an end-to-end bound. It has no analysis in the quantum random-oracle
  model.
- Akita uses Keccak-family hashes for its own challenges. Poseidon2 makes
  each challenge of the folding scheme, of the step circuits and of the
  argument outside Akita.
- The final proof is not zero-knowledge.
- `verify_final` is not succinct. Its time is linear in the matrices.

See the [security model](../formal/nightstream-fprime/SECURITY_MODEL.md)
for the adversaries, the Lean results and the premises, and the
[compression decision record](../decisions/akita-compression.md) for the
design choices. No security claim for a removed frontend, or for the removed
Spartan and WHIR path, applies to the maintained implementation.
