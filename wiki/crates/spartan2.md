# spartan2 (vendored)

A vendored copy of Spartan2 — Microsoft's sum-check-based SNARK with a linear-time
prover — used as the terminal compression backend for the decider. The workspace
builds it with the `p3_backend` feature (Plonky3 Goldilocks field types), and it reads
its Poseidon2 configuration from `neo-params`; the unmodified upstream lives in
`external/Spartan2` (excluded from the workspace).

## Role in Nightstream

One job: prove the decider R1CS. `lifecycle::compress` builds a
`decider::Statement` and hands it to `paper::decider::prove`, which targets this crate.
The circuit side is produced by `neo-fold-clean`'s `engine/r1cs_circuit` builder and
`engine/decider.rs` synthesis; Bellpepper-style circuits are the interchange format.
The PR5 decider is not implemented yet, so this path currently terminates in an
explicit `Unsupported` error — see [Decider](../architecture/decider.md) and
[Roadmap](../roadmap.md).

## Caveats inherited from upstream

Per the crate README: proofs are **not zero-knowledge** in the current implementation,
and the Spark sparse-polynomial-commitment layer is not implemented, so verifier work
is proportional to the number of non-zero R1CS entries. Both matter for the eventual
compressed-proof design and should be re-evaluated when PR5 lands.

Keep in mind this is a fork-in-tree: changes here should stay minimal and documented,
since divergence from upstream is an audit liability.
