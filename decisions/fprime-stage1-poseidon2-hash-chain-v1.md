# F′ Stage 1 Poseidon2 Hash-Chain Application

**Status:** Accepted

## Decision

`Poseidon2HashChainV1` is the first verifier-owned Stage 1 application.

- The input state is exactly four Goldilocks field words.
- The witness message is exactly four Goldilocks field words.
- The output state is exactly four Goldilocks field words.
- The domain tag is the 40 ASCII bytes of
  `Nightstream/Stage1/Poseidon2HashChain/v1`, with each byte mapped by
  `Poseidon2.ofNat` to one field word.
- The transition is

  ```text
  Poseidon2.hash(domain_tag ++ prior_state ++ message)
  ```

Lean owns the exact framing, semantics, circuit, layout, package, identity,
fixed-point proof, and conformance evidence. The verifier selects and binds
the resulting package. The prover cannot select the application or package.

## Exclusions

This decision does not authorize Stage 2, a memory relation, WASM state, or a
proof backend. Backend acceptance is not evidence for Lean semantics, matrix
equality, or assignment conformance.
