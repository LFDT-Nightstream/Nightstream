# PiCCS Prior-State Digest Transcript

**Status:** Accepted

## Problem

The non-interactive PiCCS challenges must bind the complete running statement
before the verifier derives `alpha`, `gamma`, or a SumCheck challenge. The
HyperNova pilot already recomputes a Poseidon2 state digest from that running
statement. Nightstream must decide whether PiCCS can absorb this constrained
digest instead of absorbing the complete running statement again.

This choice must not create weak Fiat--Shamir, make an ambiguous encoding
authoritative, or let a prover select the digest or relation. The detailed
implementation obligation is recorded in the
[PiCCS schedule issue](../formal/nightstream-fprime/OPEN_ISSUES_LEAN_REFACTOR.md#1-remove-duplicate-piccs-statement-absorption).

## SuperNeo

SuperNeo v1.1 Section 7.3 defines an interactive public-coin PiCCS protocol.
The complete input is fixed before the verifier samples uniform `alpha` and
`gamma`. Each SumCheck message is also fixed before its challenge. The paper
does not define a concrete Fiat--Shamir hash, field-word encoding, transcript
framing rule, or statement-compression rule. See the
[SuperNeo v1.1 PiCCS protocol](../docs/superneo-paper-v1_1/07_superneo_folding_scheme_for_ccs.md).

The general HyperNova construction uses a hash of the complete recursive state
as the public link to its next step. Its non-interactive folding security is in
the random-oracle model and its concrete hash is a heuristic instantiation.
This supports the state-binding pattern. It does not prove that a state digest
can replace the complete PiCCS Fiat--Shamir statement. That replacement needs
the committed-statement reduction below. See
[HyperNova](https://eprint.iacr.org/2023/573.pdf).

## Decision

PiCCS uses the following schedule:

```text
authoritative running state
  -> canonical, prefix-free encoding
  -> pilot-recomputed Poseidon2 state digest
  -> equality with the verifier-visible prior digest
  -> fresh zero-state PiCCS transcript
  -> schedule-specific domain tag
  -> framed prior digest
  -> framed fresh commitment
  -> framed fresh public input
  -> alpha, gamma
  -> absorb each SumCheck message before its challenge
  -> absorb the complete PiCCS output before PiRLC sampling
```

The prior digest is data absorbed by a new transcript. It is not an initial
sponge state and it is not an authority supplied by the prover.

The digest preimage must contain:

- one domain-separated verifier-context digest;
- the iteration, initial state, current state, and program counter;
- the shared running point;
- all 16 running commitments and public inputs;
- every separate `Eval_K` value; and
- every separate value in all 14 `Eval_A` families.

The verifier-context digest must commit to one canonical, non-self-referential
description of all static authority used by the fold. This description
includes the profile and transcript schedule, the logical relation and
application identity, and all static NIFS and commitment-key material. The
production verifier owns this context. Lean must prove that the context digest
covers the exact static values consumed by the selected relation.

The sealed package identity and the verifier-context digest are different
objects unless a theorem proves that one canonical preimage covers the other.
Do not set them equal only because both have four field words. The package
loader must pin the sealed package identity, and the package rows must enforce
the selected verifier-context digest. This separates package integrity from
protocol context and avoids a package-identity self-reference.

The fresh commitment and fresh public input are not in the prior-state digest.
PiCCS must absorb them directly before it derives `alpha` or `gamma`.

The encoding must have a Lean theorem that distinct well-formed preimages have
distinct encodings. It must also prove that no valid encoding is a
trailing-zero extension of another valid encoding. This condition is required
because the selected state hash does not absorb a raw input-list length. The
existing [state serializer](../formal/nightstream-fprime/NightstreamFPrime/Lifecycle/XOut.lean)
uses length-prefixed blocks, but the security result must prove the property
instead of relying on a comment.

### Binding argument

Let `S` be the complete running statement, let `enc` be the canonical
encoding, and let `d = stateHash(enc(S))`.

For two distinct valid statements `S` and `S'`:

1. If `enc(S) = enc(S')`, the canonical-encoding theorem has failed.
2. If the two encodings differ but their digests are equal, there is a named
   Poseidon2 state-hash collision.
3. If their digests differ, the PiCCS transcript inputs differ. Equal verifier
   challenges then require a named transcript collision or Fiat--Shamir
   failure.

Thus, digest absorption binds the complete running statement unless one of
the named security events occurs. Full-statement absorption gives an extra
check after a state-hash collision, but the recursive public-state link has
already failed in that event. Digest-only therefore removes defense in depth,
but it does not change the stated end-to-end theorem, which already names
state-hash collision resistance.

The deterministic Lean theorem proves only that satisfying package rows
enforce this data flow. The separate security-composition theorem must name
Poseidon2 collision resistance, commitment binding, Fiat--Shamir and sampling
security, and the SuperNeo knowledge reduction, as required by the
[F-prime proof boundary](../FPRIME_LEAN_ARCHITECTURE_SPEC.md#10-proof-boundary).
It must also prove the committed-statement step: accepted digest-bound PiCCS
implies PiCCS for the exact decoded running statement, or a named context,
encoding, state-hash, transcript, or Fiat--Shamir failure.

### Published attack checks

- Weak Fiat--Shamir omits public statement data. This schedule includes the
  complete prior statement through a constrained digest and includes the fresh
  statement directly. See [How Not to Prove Yourself](https://eprint.iacr.org/2016/771.pdf)
  and [Weak Fiat-Shamir Attacks on Modern Proof Systems](https://eprint.iacr.org/2023/691.pdf).
- Multi-round Fiat--Shamir must bind all earlier prover messages. This
  schedule absorbs each message before its challenge and keeps the complete
  transcript state. Nightstream still needs a reduction for its exact
  special-soundness and chaining conditions. See
  [Fiat--Shamir Transformation of Multi-Round Interactive Proofs](https://link.springer.com/article/10.1007/s00145-023-09478-y)
  and [The Last Challenge Attack](https://eprint.iacr.org/2024/398.pdf).
- Ambiguous serialization can break transcript binding without a hash
  collision. The required unique-decoding theorem and framed blocks address
  this class. See [Fiat-Shamir in the Wild](https://eprint.iacr.org/2024/1565.pdf).
- Strong Fiat--Shamir does not by itself prevent concrete-hash
  self-reference attacks. The production verifier must prevent circuit
  selection and use one verifier-owned, identity-pinned Lean package. It must
  reject a functionally equivalent replacement and any prover-supplied
  universal circuit description. The selected application and outer proof
  backend require separate checks for these conditions. See
  [How to Prove False Statements](https://eprint.iacr.org/2025/118.pdf).

The selected Poseidon2 sponge has four capacity field elements and emits a
four-field digest. Under the random-permutation model, this gives a generic
collision ceiling near 128 bits. This is an inference from the
[Poseidon2 analysis](https://eprint.iacr.org/2023/323.pdf), not a deterministic
proof about the concrete permutation. End-to-end security is the minimum of
this boundary and the exact SuperNeo statistical and commitment bounds.

This accepted choice changes transcript order and proof bytes. The F-prime
Stage 1 Lean relation must implement the rule and select a new verifier-owned
relation identity before PiCCS conformance closes. The separate legacy
`protocol-contract` profile fixes incompatible dimensions and is not authority
for this Stage 1 package.
