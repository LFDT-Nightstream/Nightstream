# F′ Stage 1 Poseidon2 Hash-Chain Application

**Status:** Accepted

## Problem

The step circuits need one application for the reference packages, the tests
and the executables. The verifier must own it, and the prover must not
select it.

## SuperNeo

SuperNeo v1.2 and HyperNova do not fix an application step `F`.

## Decision

`Poseidon2HashChainV1` is the reference application
(`Lifecycle.Stage1.Poseidon2HashChainV1.program`).

- The input state is exactly four words of `F_q`.
- The witness message is exactly four words of `F_q`.
- The output state is exactly four words of `F_q`.
- The domain tag is the 40 ASCII bytes of
  `Nightstream/Stage1/Poseidon2HashChain/v1`, with each byte mapped by
  `Spec.Poseidon2.ofNat` to one field word
  (`Lifecycle.Stage1.Poseidon2HashChainV1.domainTag`).
- The transition is

  ```text
  Poseidon2.hash(domain_tag ++ prior_state ++ message)
  ```

  (`Lifecycle.Stage1.Poseidon2HashChainV1.step`), with the hash of
  [SPEC.md](../SPEC.md) section 5.1.

Lean owns the framing, the semantics, the circuit and the packages. The
verifier pins the identity of the packages ([SPEC.md](../SPEC.md)
Algorithm 2). The prover cannot select the application or the packages.
