# Lifecycle

`crates/nightstream/src/lifecycle/` owns proving, extension, and terminal
verification. A caller compiles or loads a `Circuit`, selects a prover and
verifier with an explicit security minimum, calls `prove` and `extend`, then
checks the expected final state with `Verifier::verify`.

Each recursive step runs PiCCS, PiRLC, and PiDEC and constructs the next F′
witness. The selected application and recursive verifier belong to the same
sealed package. See the [API guide](../../crates/nightstream/README.md).

Terminal verification checks the fresh relation and the running commitment
openings, public values, norms, Pad evaluations, and all matrix evaluations.
It recomputes the bound public state. Matching digests do not replace these
checks. The maintained API has no separate legacy audit or Spartan route.
