# PiCCS Prior-State Digest Transcript

**Status:** Accepted

## Problem

The non-interactive PiCCS challenges must bind the complete running statement
before the verifier derives `alpha`, `gamma`, or a SumCheck challenge. The
step circuit already recomputes a Poseidon2 state hash from that running
statement. Nightstream must decide whether PiCCS can absorb this constrained
digest in place of the complete running statement.

This choice must not create weak Fiat--Shamir, make an ambiguous encoding
authoritative, or let a prover select the digest or the relation.

## SuperNeo

SuperNeo v1.2 Section 7.3 defines an interactive public-coin PiCCS protocol.
The complete input is fixed before the verifier samples uniform `alpha` and
`gamma`. Each SumCheck message is also fixed before its challenge. The paper
does not define a concrete Fiat--Shamir hash, field-word encoding, transcript
framing rule, or statement-compression rule.

HyperNova uses a hash of the complete recursive state as the public link to
its next step ([HyperNova](https://eprint.iacr.org/2023/573.pdf)). This
supports the state-binding pattern. It does not prove that a state digest can
replace the complete PiCCS Fiat--Shamir statement.

## Decision

PiCCS absorbs the prior state digest, not the running statement. The rule is
in [SPEC.md](../SPEC.md) sections 4.2 and 5.2 and Algorithm 3.

1. The state hash binds the key digests of both circuits, the step counter,
   the initial and current states, the program counter and the digest of
   each slot. A slot digest binds the commitment, the evaluations, the
   point, the public input and the depth of the accumulator
   (`Lifecycle.BoundedDepth.stateHash`, `Lifecycle.BoundedDepth.slotDigest`).
2. The step circuit recomputes this hash from the prior state and requires
   that it is the prior digest in the public input of the fresh claim.
3. Each fold starts a new sponge. It absorbs one constant 12-word domain
   chunk, `Nightstream/SuperNeo/fold/light/v1` or
   `Nightstream/SuperNeo/fold/refresh/v1`
   (`Lifecycle.BoundedDepth.lightFoldDomainChunk`,
   `Lifecycle.BoundedDepth.refreshFoldDomainChunk`). It then absorbs the
   prior digest, the fresh commitment and the fresh public input as one
   stream, and for a refresh fold the Π_DEC children.
4. The sponge gives `alpha` and `gamma`. It absorbs each SumCheck message
   before its challenge, and the complete PiCCS output before the PiRLC
   challenges.

The prior digest is data that a new transcript absorbs. It is not an initial
sponge state, and the prover does not supply it as an authority. The fresh
commitment and the fresh public input are not in the state hash, so PiCCS
absorbs them directly before it derives `alpha` or `gamma`.

The stream has no length prefix and no framed block. The profile fixes each
absorbed length, so the stream has one parse
(`Lifecycle.RandomOracle.Schedule.pieces_alike`,
`Lifecycle.RandomOracle.Sponge.calls_injective`).

### Binding argument

Let `S` and `S'` be two different well-formed states.

1. If their state hashes are equal, there is a Poseidon2 collision on their
   state preimages or on the preimages of one slot
   (`Lifecycle.BoundedDepth.stateHash_collision`).
2. If their state hashes differ, the PiCCS transcript inputs differ. Equal
   verifier challenges then need a transcript collision or a Fiat--Shamir
   failure.

So two folds with one fresh instance that link to well-formed prior states
start from one running claim, or the two prior states give a state-hash
collision (`Lifecycle.BoundedDepth.priorLink_binds`). Absorption of the full
statement would give one more check after a state-hash collision, but the
recursive public-state link has already failed in that event. The digest
removes defense in depth. It does not add a premise, because the results
already name Poseidon2 collisions
([security model](../formal/nightstream-fprime/SECURITY_MODEL.md), trust
items 2 and 3).

The Poseidon2 sponge has four capacity words and gives a digest of four
words. In the random-permutation model, two distinct inputs give one digest
with a probability of at most `N²/q⁴` plus the sponge error, for `N`
permutation calls (security model, `[SPONGE]`). This is a bound in a model,
not a proof about the concrete permutation.

### Published attack classes

- Weak Fiat--Shamir omits public statement data. This schedule includes the
  prior statement through a constrained digest and the fresh statement
  directly ([ePrint 2016/771](https://eprint.iacr.org/2016/771.pdf),
  [ePrint 2023/691](https://eprint.iacr.org/2023/691.pdf)).
- Multi-round Fiat--Shamir must bind all earlier prover messages. This
  schedule absorbs each message before its challenge
  ([ePrint 2024/398](https://eprint.iacr.org/2024/398.pdf)).
- Ambiguous serialization can break transcript binding without a hash
  collision. The fixed lengths give one parse
  ([ePrint 2024/1565](https://eprint.iacr.org/2024/1565.pdf)).
- Strong Fiat--Shamir does not prevent attacks that use the hash inside the
  circuit. The verifier pins the identity of its packages, and such attacks
  are outside each random-oracle model
  ([ePrint 2025/118](https://eprint.iacr.org/2025/118.pdf); security model,
  Out of scope).
