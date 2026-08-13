# wip-spartan

This crate is Nightstream's work-in-progress terminal Spartan backend. It accepts a direct sparse R1CS shape, proves it over Goldilocks, uses a Poseidon2 transcript, and opens the witness with WHIR.

Nightstream calls this crate from the terminal R1CS path in `neo-fold-clean`. The crate does not synthesize circuits and does not implement the recursive folding protocol.

## Implemented scope

- Direct `SplitR1CSShape` setup, proving, and verification
- Goldilocks scalar field
- Poseidon2 Fiat-Shamir transcript
- WHIR polynomial commitment scheme
- Lockstep repetition for the Appendix B.2 statistical target

The prover checks the supplied sparse R1CS assignment before it creates a proof. The verifier reconstructs the public statement from verifier-owned inputs in `neo-fold-clean`.

## Status

This backend is not production-ready. The name is intentional. Its cryptographic review, performance work, and terminal integration are still in progress.

Removed inherited paths include Bellpepper synthesis, curve engines, Hash-MLE commitments, Keccak transcripts, and NeutronNova. Add a path only when Nightstream has an active use for it.

## Checks

```sh
cargo test -p wip-spartan --release
cargo test -p neo-fold-clean --release --test system_lean_native_ccs_manifest
```
