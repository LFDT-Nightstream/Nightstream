# toy-spartan

An in-tree Spartan fork used for standalone backend experiments. The workspace builds
it with the `p3_backend` feature. Its `GoldilocksWhirEngine` uses Plonky3 WHIR with
SuperNeo's Goldilocks degree-2 extension, 125-bit target, 18-bit grinding budget, and
canonical Poseidon2 parameters from `neo-params`. The upstream reference remains in
`external/Spartan2` (excluded from the workspace).

## Role in Nightstream

The crate is deliberately not connected to `neo-fold-clean`, the SuperNeo lifecycle,
or terminal compression. It exposes a complete prescribed-point WHIR PCS and a Toy
Spartan engine that can prove Bellpepper R1CS circuits on its own. Connecting a final
decider remains separate work — see [Decider](../architecture/decider.md) and
[Roadmap](../roadmap.md).

## Caveats inherited from upstream

The surrounding Spartan proofs are **not zero-knowledge**. WHIR is used in non-hiding
mode, and the Spark sparse-polynomial-commitment layer is not implemented, so verifier
work remains proportional to the number of non-zero R1CS entries. WHIR commitments
are not homomorphic; this adapter accepts exactly one witness commitment and rejects
split shared/precommitted witness layouts.

The prescribed opening point, claimed evaluation, commitment, and Toy Spartan
transcript anchor are all bound into WHIR's Poseidon2 challenger. No Keccak, SHA, or
Blake-family hash participates in that protocol-binding path.
