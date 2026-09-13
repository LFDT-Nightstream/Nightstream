# Actual selected successor envelope

Checked code: `50127ac92e4ceb9baac3a6fb07c2494429d0bcf3` on
`nico/f-prime-constraints-cuda-formal`, 2026-09-13 UTC.

`Poseidon2HashChainV1Package.complete_step` checks each supplied child
witness's shape, public projection, strict signed-unit norm and fixed-key
commitment. It retains the ordered matrices in the existing `RunningInstance`.
It then runs the emitted witness IR and logical transport, checks the next
public input, packs the complete fresh carrier with a zero completion tail,
and commits and retains that same matrix. `Stage1Envelope` uses the existing
`ProofState` and one `LatestInstance`; it stores no proof history.

`Stage1Envelope.initial` creates the exact empty case with iteration zero
and equal initial/current states. The complete fresh opening uses `Z`;
its redundant `w` vector is empty. Running children retain their full tails.

## Executed evidence

The explicit `complete-owned-envelope` fixture action uses the saved actual
NIFS proof and all 16 original child matrices. Their hashes match the prior
native manifest. No C/R/D producer is rerun. The test rejects missing/extra
children, wrong dimensions, a changed public projection, a nonunit private
value, and a changed private witness with an unchanged commitment.

The returned child matrices and claims match the supplied and verified
values. Every fresh carrier coordinate matches the logical assignment built
from the independent Lean caller packet. The complete reference carrier's
commitment, computed with `commit_production_signed_units`, matches the
retained fresh claim. All fresh public words and the zero tail match. The
initial empty state matches the Lean base fixture. Full independent rows and
child mutations are reused from [the preceding handoff](NATIVE_SUCCESSOR_EVIDENCE.md).

| Check | Result | Seconds |
| --- | --- | ---: |
| Actual envelope, rejections and complete reference comparison | Pass | 156.28 |
| Fixture executable build | Pass | 19.60 |
| Existing integration test compilation | Pass | 45.30 |
| Static | Pass | 8.93 |
| Library | Pass, 3,920 jobs | 0.92 |
| Axioms | Pass, 4,008 jobs | 1.02 |

Formatting and independent source review pass. The first fixture build
failed on a missing test-call label; the corrected build passes. All commands
used the existing caps and one build queue. Package bytes and pins are unchanged.

The map records scoped Rust evidence on four envelope records. Its local
build passes 2,154 source-location checks, 19 Python tests and 12 JavaScript
tests. This does not publish the live site.

Evidence: [source, output envelope, fresh witness, logs and review](NATIVE_ENVELOPE_EVIDENCE.zip).
Archive size: 39,165,413 bytes. SHA-256:
`b9ab74dc98c7d5875dcc5925d4f7be4ca42729f841f3708c24043f2096f679fb`.
The archive includes the new fresh matrix and the manifest for the unchanged
child matrices. These are local validation records, not protected-checker approval.

The result is prover data. Full child Eval_K/Eval_A checks, fresh CCS row
membership and the public-state link must still meet the selected terminal
predicate. Terminal acceptance is the next required consumer. This result
supplies no new backend, cryptographic assumption or machine-time theorem.
