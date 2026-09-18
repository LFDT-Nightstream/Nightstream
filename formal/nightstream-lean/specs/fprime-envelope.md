# F' state-envelope integrity

## Reverse-round card

- Surface: `construction2/state.rs`, `construction2/transition.rs`, and the
  base/recursive branch in `f_prime/native.rs`.
- Failure class: a forged initial/active tag, zero or stale counter, out-of-range
  fixed program counter, mutated initial boundary, or inconsistent public trace.
- Mathematics: executable decidability of the exact envelope equations followed
  by `check_sound`.
- Artifact: `tests/FPrimeEnvelope.lean` contains accepted and rejected concrete
  states, including forged initial and active cases.

## Scope

`check_sound` establishes only:

- `pc = 1` on the input state;
- initial iff both counters are zero and `z0 = zi`;
- active implies both counters are nonzero;
- chunk and step counter advancement;
- immutability of `z0`, `pc`, and initial semantic state;
- `publicTrace = zi` on the output;
- the output fold state is active;
- the output fresh batch is nonempty (`Error::EmptyStep` parity) and the
  declared fresh count equals the installed batch's cardinality.

## Non-goals

The theorem does not establish NIFS correctness, accumulator authority,
application semantics, Poseidon2 security, Nebula validity, circuit soundness,
or full Rust conformance. Its current assurance tier is model-level only.

## Conformance decision

- Theorem: `Nightstream.Implementation.FPrime.Envelope.check_sound`
- Route: model-first
- Statement parity: pass for the documented envelope only
- State parity: pass with deliberate omission of fields unused by the theorem;
  the fresh batch is a genuine list so cardinality is expressible
- Transition parity: **fail found 2026-07-09, repaired same day.** Review
  Finding 6: the model accepted a `freshCount = 0` successor that Rust rejects
  with `Error::EmptyStep` (`native.rs`). `AdvanceCoherent` now requires a
  nonempty batch whose length equals the declared count, and
  `tests/FPrimeEnvelope.lean` pins the forgery permanently. Parity holds
  against the named Rust helpers for the documented envelope; full transition
  parity remains open until the tracer-bullet artifact gate exists
- Concurrency parity: not applicable
- Cancellation/drain parity: not applicable
- Runtime evidence: pending; Lean examples exist, but no Rust conformance test is
  linked yet
- Drift gate: `lake exe check` verifies symbol anchors in the mapped Rust
  sources (`Err(Error::EmptyStep)`, `fn state_base_case_check`,
  `fn advance_state`, `pub struct State`) and exits nonzero when any is
  missing; content hashes over generated artifacts land with the tracer bullet
- Result: model-level theorem; no Rust-alignment claim
