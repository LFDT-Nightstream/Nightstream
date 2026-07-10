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
- the output fold state is active.

## Non-goals

The theorem does not establish NIFS correctness, accumulator authority,
application semantics, Poseidon2 security, Nebula validity, circuit soundness,
or full Rust conformance. Its current assurance tier is model-level only.

## Conformance decision

- Theorem: `Nightstream.Implementation.FPrime.Envelope.check_sound`
- Route: model-first
- Statement parity: pass for the documented envelope only
- State parity: pass with deliberate omission of fields unused by the theorem
- Transition parity: pass against the named Rust helpers
- Concurrency parity: not applicable
- Cancellation/drain parity: not applicable
- Runtime evidence: pending; Lean examples exist, but no Rust conformance test is
  linked yet
- Drift gate: pending until the new theorem has a committed baseline
- Result: model-level theorem; no Rust-alignment claim
