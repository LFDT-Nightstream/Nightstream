# Lifecycle

`crates/nightstream/src/lifecycle/bounded_depth/` owns proving, extension and
verification. A caller compiles or loads a `Circuit`,
selects a prover and verifier with an explicit security minimum, calls
`prove` and `extend`, then checks the expected final state with
`Verifier::verify`.

The state has two slots. Slot 0 holds the accumulator of the light circuit,
and slot 1 holds the accumulator of the refresh circuit. Each step folds the
fresh instance of the previous step into the slot of the circuit that made it.
A light fold runs PiCCS and PiRLC while the depth of the slot is below 64. A
refresh fold runs at depth 64. It splits the accumulator with PiDEC into 16
children, then runs PiCCS over 17 sources and PiRLC. No fold ends with PiDEC.
See the [API guide](../../crates/nightstream/README.md).

`Verifier::verify` recomputes the hash of the state with both slots and
compares it with the public input of the fresh instance. It checks the claim
of each slot under the relation of its circuit, with a norm below `B`, and the
fresh instance under the relation of the circuit that ran last. Matching
digests do not replace these checks.

## Compress a proof

`Prover::compression_setup` makes a `CompressionSetup` once from the
prover's circuit. It expands the rows of the Ajtai key that the circuit
reads, for the blocks of the larger relation up to a power of two, and
commits them as one polynomial with the Akita commitment scheme.
`Prover::compress` makes a `FinalProof` from a proof and the setup. The
final proof has no witness, and `extend` cannot use it.
`FinalProof::to_bytes` and `FinalProof::from_bytes` are its byte form.

`Verifier::compression_key` gives the `CompressionKey` of the verifier's
circuit. `Verifier::verify_final` checks a final proof against the expected
state under that key. The key holds the
commitment of the key polynomial and Akita's commitments of a prefix of its
own setup, in about 1 KB. No proof carries it, and the prover's setup gives
it to no one. It is a function of the seeds and of the size of the circuit,
so a prover and a verifier of one circuit agree, and each circuit of one
size has the same key. The library pins the key of each size, and a test
computes each pinned key again from the seeds.

`CompressionKey::to_bytes` and `Verifier::decode_compression_key` are the
byte form of the key. Under a key that does not hold the commitment of the
Ajtai key, `verify_final` can accept false statements. So the decoder
accepts only the bytes of the pinned key of the circuit size.
`nebula::Prover` and `nebula::Verifier` have the same functions for a
memory proof.

The final proof has both slot claims, the CE claim that one PiCCS run makes
of the fresh instance, and one argument from `crates/ns-compress`. The
argument commits two polynomials with Akita: `W` holds the three witnesses
and the counts of the lookup tables, and `V` holds the lookup inverses. One
sum-check proves the Ajtai openings, the evaluations, the public inputs and
the norm bound of each slot. A second sum-check reduces the part of the
public weights that the Ajtai key gives to one value of the key polynomial.
One Akita proof opens the key polynomial, `W` and `V`. See the module
comment of `crates/ns-compress/src/lib.rs` for the transcript and the
error. [SPEC.md](../../SPEC.md) Algorithms 9, 10 and 11 and the
[decision record](../../decisions/akita-compression.md) describe the proof
and the key.

`compress` does not verify its input or its result. `verify_final` computes
the hash of the state from the slot claims of the proof, as `verify` does.
It expands no element of the Ajtai key. It is not succinct: it makes one
pass over the rows of each relation.

These numbers are from runs of
`cargo test -p nightstream --release a_compressed_proof_verifies -- --ignored --nocapture`
on an M5 Ultra with 30 cores, with LTO off, for the Poseidon2 hash chain
with the cap 2. The times are the range of four runs. They are not a
benchmark.

| | Light circuit last | Refresh circuit last |
|---|---|---|
| Witness cells in `W` | 41.4 M | 68.4 M |
| Variables of `W` | 26 | 27 |
| `compress` | 2.5 to 2.7 s | 3.3 to 3.7 s |
| `verify_final`, 30 threads | 0.17 to 0.20 s | 0.22 to 0.25 s |
| `verify_final`, 1 thread | 0.9 to 1.1 s | 1.5 to 1.7 s |
| Final proof | 133.3 KB | 133.4 KB |

Of the proof bytes, the Akita opening is 75.7 KB and the claims are about
54 KB. Of the `verify_final` time, Akita is 9 ms on 30 threads and 31 ms on
one thread. The other time is the pass over the rows of the matrices.

The compression setup takes 1.0 s and holds about 10 GB: the key rows and
the copy that Akita opens. To compute the compression key again from the
seeds, as the test of the pin does, takes 1.0 s on 30 threads and 19 s on
one thread, with a peak of 6 GB of resident memory. A verifier does not make
this computation. The
process of the test that compresses both circuits has a peak of 30 GB of
resident memory (one run), so the tests that compress a step of the hash
chain are ignored. CI runs the public test alone, and the other tests run
by hand ([testing](../development/testing.md)).
