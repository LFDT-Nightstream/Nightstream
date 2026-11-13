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
