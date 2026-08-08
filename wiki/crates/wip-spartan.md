# wip-spartan

`wip-spartan` is Nightstream's work-in-progress terminal proof backend. It
accepts a direct sparse R1CS shape over Goldilocks, uses a Poseidon2
Fiat-Shamir transcript, and opens the witness with WHIR.

The terminal R1CS path in `neo-fold-clean` calls this crate. The crate does
not synthesize application circuits and does not implement recursive folding.

## Implemented scope

- direct `SplitR1CSShape` setup, proving, and verification;
- Goldilocks scalar arithmetic;
- Poseidon2 transcript binding;
- WHIR polynomial commitments; and
- lockstep repetition derived from the selected terminal relation and protocol
  security target.

The prover checks the sparse R1CS assignment before it creates a proof. The
caller supplies the verifier-owned public statement.

## Excluded code

The active path does not need Bellpepper synthesis, curve engines, Hash-MLE
commitments, Keccak transcripts, NeutronNova, or alternate PCS providers.
Those inherited paths were removed.

## Status

This backend is connected but is not production-ready. It still needs
cryptographic review, performance work, and deployment integration.

```sh
timeout 300s cargo test -p wip-spartan --release
timeout 300s cargo test -p neo-fold-clean --release --test system_r1cs_ivc_terminal
```
