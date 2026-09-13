# Selected native terminal verification

Checked code: `5b544af3ed00c1a7b5610e8c314d269150ad98e8` on
`nico/f-prime-constraints-cuda-formal`, 2026-09-13 UTC.

`Poseidon2HashChainV1Package.verify` checks the selected
`Lifecycle.Stage1.Terminal.HoldsFor` boundary against an external state.
The package fixes the application, complete relation, verifier context,
Ajtai key and `pc = 1`. The initial case requires iteration zero and equal
endpoints. An active envelope requires a positive canonical field counter,
16 running claims and openings, and one fresh claim and opening.

The verifier checks all witness dimensions, public projections, strict unit
norms and fixed-key commitments. It recomputes the state hash and complete
fresh public input. Its cache comes from the package's actual rows. It
computes every running Pad and matrix evaluation from the complete retained
matrices, then checks every fresh CCS row. Only the fresh completion tail
must be zero. Running tails remain part of their complete openings.
Parent caches, carried frame digests and redundant scalar `w` storage are
non-authoritative. Terminal verification performs no extra NIFS fold.

## Required execution evidence

The explicit `check-owned-terminal` action loads the actual successor
envelope and original child matrices retained by
[the envelope milestone](NATIVE_ENVELOPE_EVIDENCE.md). The independent Lean
successor fixture supplies the external endpoint. Each case has the existing
300-second cap and uses the single build queue.

- `accepted`: initial and actual successor acceptance; external-state,
  zero-active-counter and noncanonical-counter rejection; the final canonical
  counter reaches its state-hash check. Valid acceptance is independent of
  the non-authoritative caches.
- `ce-evaluation`: change one Pad evaluation, recompute the terminal state
  hash, fresh public input and fresh commitment, and require the Pad rejection.
- `ce-matrix-evaluation`: do the same for one matrix evaluation and require
  the separate matrix-evaluation rejection.
- `fresh-private`: change a private signed-unit coordinate, recompute its
  fresh commitment and require a failed CCS row.

The first acceptance run exceeded the cap. The outer timeout returned 137;
the child later printed success at 311.00 seconds. This is a failed run,
not acceptance evidence. Later commands use the guard's process-group cleanup.
The identified avoidable work was the fresh witness's expansion to
253,011,276 extension-field values followed by conversion back to blocks.
The repair uses the existing compact block representation in the same exact
row checker, with width and real-input guards retained.

The repaired acceptance run passes in 267.16 seconds including the command
wrapper (266.57 seconds inside the fixture). The same input and checks took
311.00 seconds inside the failed variant. The focused raw/dense/compact
relation tests pass, including signed-unit row failure, padded-width and
non-real rejection. Independent source and final-scope reviews pass.

| Check | Result | Seconds, including wrapper |
| --- | --- | ---: |
| Initial and actual successor terminal acceptance | Pass | 267.16 |
| Rehashed and recommitted Pad mutation | Required Pad rejection | 280.71 |
| Rehashed and recommitted matrix mutation | Required matrix rejection | 281.43 |
| Recommitted private witness mutation | CCS row 0 fails | 282.54 |
| Raw/dense/compact relation tests | Pass | 15.39 |
| Fixture executable build | Pass | 55.42 |
| Static | All boundary checks pass | 8.93 |
| Library | Build completed, 3,920 jobs | 0.92 |
| Axioms | Build completed, 4,008 jobs | 1.02 |

Formatting passes. The ordered Lean gates ran after the terminal tests.
Package bytes, identities and the selected profile are unchanged.

The [archive](NATIVE_TERMINAL_EVIDENCE.zip) contains the exact code, result
files, command logs, reviews, input hashes and prior-evidence references.
It has 35 verified entries and is 177,825 bytes. SHA-256:
`2485915eb86f50cddfa3ad8406e512075d4a0c3fd56c6f7a0745768d1daa021a`.
The existing envelope archive retains the fresh witness; unchanged child
witnesses retain their prior manifest. They are not duplicated here.

## Scope

This closes the selected executed trace:
initial iteration 0, complete base assignment at iteration 1, actual C/R/D,
the exact 1-to-2 caller packet and assignment, retained successor envelope,
and terminal verification. The base branch retains canonical zero running
openings; it requires no actual recursive NIFS prover call.

The final security target remains `LeanGraph.Targets.HyperNovaLinearSecurity`,
proved through `HyperNovaVisitedSecurity.history_probability_linear_bound`
at `1ad23f55`. No Lean library or audit source changed in the native milestones.
The existing lean-graph query confirms passing, current gates and a passing
review on that frozen proof snapshot. The current-tree security validation
is also current; its broader boundary/decomposition snapshot is stale after
the Rust edits. The current native checkpoint has the ordered gates and
independent reviews above. These local records do not grant the graph's
separate protected-checker approval.

The local requirements map records scoped native evidence for the eight
terminal records and reconciles the existing base-default evidence. Its
build checks 2,172 source locations; 19 Python and 12 JavaScript tests pass.
Code citations are pinned to the checked commit above. Earlier execution
reports keep their historical scope; the current record notes connect them
to this completed terminal consumer.

The Lean completeness and history-security theorems retain their stated
sampler, counter and approved cryptographic premises. This evidence supplies
no universal Rust-semantics proof, later-running-input coverage, new proof
backend, numerical security budget, performance result or production approval.
It does not publish the live requirements site.
