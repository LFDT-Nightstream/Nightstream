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

`Prover::compress` makes a `FinalProof` from a proof. The final proof has no
witness, and `extend` cannot use it. `Verifier::verify_final` checks it
against the expected state. `FinalProof::to_bytes` and
`FinalProof::from_bytes` are its byte form. `nebula::Prover::compress` and
`nebula::Verifier::verify_final` do the same for a memory proof.

The final proof has both slot claims, the CE claim that one PiCCS run makes
of the fresh instance, and one argument from `crates/neo-compress`. The
argument commits two polynomials with the Akita commitment scheme: `W` holds
the three witnesses and the counts of the lookup tables, and `V` holds the
lookup inverses. One sum-check proves the Ajtai openings, the evaluations,
the public inputs and the norm bound of each slot. It ends in one Akita
opening of each polynomial. See [SPEC.md](../../SPEC.md) Algorithms 9 and 10
and the [decision record](../../decisions/akita-compression.md).

`compress` does not verify its input or its result. `verify_final` computes
the hash of the state from the slot claims of the proof, as `verify` does.
It is not succinct: it makes one pass over the Ajtai key and one pass over
the rows of each relation.

These numbers are from one run of
`cargo test -p nightstream --release a_compressed_proof_verifies -- --nocapture`
at commit `8d924bf32`, on an M5 Ultra with 30 cores, with LTO off, for the
Poseidon2 hash chain with the cap 2. They are not a benchmark, and a later
change to the key expansion makes the times lower.

| | Light circuit last | Refresh circuit last |
|---|---|---|
| Witness cells in `W` | 41.4 M | 68.4 M |
| Variables of `W` | 26 | 28 |
| `compress` | 2.3 s | 3.2 s |
| `verify_final` | 0.83 to 0.86 s | 0.83 to 0.86 s |
| Final proof | 201 to 202 KB | 201 to 202 KB |

Of the proof bytes, the two Akita openings are about 145 KB and the claims
are about 50 KB. Of the `verify_final` time, Akita is 14 to 17 ms. The other
time is the pass over the public data: the verifier expands the Ajtai key
from SHAKE128, about 18.3 GB of output for each pass.

The memory of `compress` was not measured. An Akita commitment and opening
of one polynomial of 28 variables used about 7 GB of resident memory in a
prototype run on the same machine (one run).
