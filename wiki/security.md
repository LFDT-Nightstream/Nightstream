# Security

Nightstream is research software. It has no independent audit. Do not deploy
it as a production verifier.

## Assumptions

- Ajtai commitment binding is based on the selected Module-SIS parameters.
- SuperNeo reductions use the paper's low-norm and decomposition conditions.
- Fiat-Shamir uses Poseidon2 and the classical random-oracle analysis.
- Sum-check challenges use the configured Goldilocks extension field.
- The terminal Spartan path relies on its sum-check and WHIR commitment
  assumptions.

The repository does not claim quantum-prover extraction or security in the
quantum random-oracle model.

## Enforced boundaries

- Verifier-owned preprocessing fixes the relation and Ajtai setup.
- Protocol-binding paths use Poseidon2 only.
- A digest compresses data but is not proof authority.
- Verifiers recompute carried digests or replay them into an authoritative
  transcript.
- Final accumulator claims are checked against witness openings, public-input
  projections, norm bounds, and committed-evaluation relations.
- Red-team tests mutate proof and transcript fields and require rejection.

## Open security work

- The direct CCS frontend does not prove the recursive F' induction. Its
  multi-chunk path requires full-history audit replay.
- `wip-spartan` is connected to the terminal R1CS path, but it still needs
  focused cryptographic review and performance analysis.
- The CUDA backend has no canonical device kernel. It fails explicitly and
  does not claim CPU work as CUDA work.
- Side-channel resistance has not been established.
- Parameter selection and the complete end-to-end security argument need
  independent review.

The contribution rules in [AGENTS.md](../AGENTS.md) are part of this security
boundary.
