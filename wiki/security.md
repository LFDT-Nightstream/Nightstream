# Security

Nightstream is research software. Independent review and a complete numerical
security claim for concrete Poseidon2 execution remain open.

The formal argument keeps these boundaries explicit:

- Binding for the selected fixed-seed Ajtai key requires its same-key MSIS assumption.
- Concrete Poseidon2 requires the stated collision and Fiat–Shamir transfer assumptions.
- The wide sampler's bias bound assumes uniform field draws; it is not itself a random-oracle theorem for the sponge.
- The low-norm Phi81 invertibility fact is proved for the selected strong set.
- Completeness and soundness are separate obligations.

The verifier checks actual relation rows, commitments, norms, public values,
Pad openings, and all matrix openings. A carried digest must be recomputed
from authoritative data or replayed into the verifier-driven transcript.
Self-consistent rehashing is not evidence of a valid witness.

See the [security model](../formal/nightstream-fprime/SECURITY_MODEL.md)
for the adversaries, the Lean results and the premises. No security claim
for a removed frontend or compression backend transfers to the maintained
implementation.
