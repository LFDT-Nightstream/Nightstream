# Actual NIFS-to-successor assignment

2026-09-13 UTC. Checked code:
`68c5d94a5839cec355e20f39a5eaff2204a9956b`, on
`nico/f-prime-constraints-cuda-formal`.

`Poseidon2HashChainV1Package.step_inputs` now verifies the actual native NIFS
proof and builds the complete next caller packet. The package owns the
context, parameters and `pc = 1`. The bridge recomputes the prior state hash,
checks the fresh public input and all incoming child/parent frames, and
rejects zero or wrapping recursive counters. It derives child data from the
verifier output and hashes the next application state and running claims.

Only after verification does the returned carrier's frame metadata change
to the new state hash. The original proof stays unchanged. This metadata is
excluded from the formal preimage and supplies no independent authority.

## Executed connection

The actual-base C/R/D proof and complete results in
[NATIVE_NIFS_EVIDENCE.md](NATIVE_NIFS_EVIDENCE.md) are reused. The existing
`emitRecursiveStepFixture` consumes their exact C inputs and children to
produce the independent iteration-one to iteration-two caller packet.

`actual_nifs_builds_the_checked_successor_assignment` checks all 945,983
retained native proof bytes, compares every caller input word with Lean,
invokes `execute_step_witness`, and checks the full low-norm logical carrier
and next public projection. It checks exact returned claims and parent,
allows only the frame update, and rejects detached states, incoming frames,
fresh public data, child commitments and points, plus invalid counters.

The existing independent evaluator accepts the same packet: all 29,225,729
physical rows, 6,377,559 active logical rows, complete logical coordinate
equality and 45 alignment zeros pass. Separate child commitment, Eval_K and
Eval_A mutations are rejected. Only the old caller-input adapter changed to
read the complete C/R/D result; the evaluators and mutation checks are unchanged.

## Validation

| Check | Result | Seconds |
| --- | --- | ---: |
| Existing Lean successor emitter | Pass | 16.50 |
| Native handoff and rejection test, including compilation | Pass | 98.21 |
| Independent complete assignment | Pass | 35.88 |
| Independent child assignment mutations | Pass | 27.02 |
| Static | Pass | 8.88 |
| Library | Pass, 3,920 jobs | 6.43 |
| Axioms | Pass, 4,008 jobs | 6.73 |

The test itself took 43.97 seconds; compilation took 54.01 seconds. The
conformance executable build took 27.46 seconds. One build queue and the
existing 300/1,500-second caps were used. Formatting and independent source
review pass. Package bytes, identities and protocol semantics are unchanged;
prior package identity evidence is reused.

The requirements update records scoped Rust evidence on eight direct handoff
records. It passes 2,146 source-location checks, 19 Python tests and 12
JavaScript tests. This local build does not publish the live site.

Evidence: [source, exact inputs, logs and review](NATIVE_SUCCESSOR_EVIDENCE.zip).
Archive: 2,106,593 bytes, SHA-256
`c62823dad86c82f4ddfc257e48e3df9de73957efaaa772e68ea34842ef417315`.
The manifest records retained hashes and the prior native archive. These are
local records, not protected-checker approval or universal Rust proofs.

## Remaining parent scope

This closes the actual child-to-complete-assignment handoff. The packet's
returned carrier contains claims only. Retaining the matching child witnesses,
committing the new fresh assignment in the returned envelope, and selected
terminal verification remain required steps in the parent goal. Approved
FS/MSIS assumptions remain explicit. Performance and the proof backend remain
outside this milestone.
