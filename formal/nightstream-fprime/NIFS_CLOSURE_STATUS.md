# NIFS closure status

Checked source: `01a8fd8ca68280c7f642451ab33bc714cee5bc98`. All four gates pass.
The selected extraction consumer now constructs the implementation checks.
Requirement-link changes below are proposals, as requested in the closure
brief; this task has not changed `requirements.json` statuses.

## Proved implementation connection

`Export.Stage1.NifsClosure.finishValue_probability_and_expected_work`
consumes the concrete `NifsExtractionProvider.provider`. It proves the
selected stored source-return lower bound and the declared expected-work
bound. The provider checks the actual PiDEC attempt, all 16 exact child
openings, binary recomposition and the exact parent CE claim. `batchAt_eq`
uses the existing relation, statement and checked receipt.

`Lifecycle.Nifs.ClaimCheck.check_eq_true_iff` checks every CE field with the
existing commitment and streaming evaluation. The final consumer has no free
provider, suffix correctness, parent-check correctness, call-value equality
or prepared-context equality premise. Its prefix call is the existing
checked run; preparation returns the context supplied by the experiment.

The owner-approved parametric FS boundary, external low-norm invertibility,
raw adversary calls and tape laws, declared clock bounds and moment bounds
remain explicit. This result does not construct an efficient adversary
translation or PMF sampler. Its real success event requires accepted NIFS
output with valid witnesses for all 16 exact children; public acceptance
alone is not the paper's reduction-of-knowledge success event.

`NifsInvalidSource.real_success_bound_of_invalid_source` additionally bounds
`g Q p_real` by `deltaFS Q + weakLoss + sqrt(17 * epsilonMSIS + testError)`
when every positive-mass input context has no joint source witness. It does
not condition a mixed adaptive input law on semantic invalidity.

## Gates and retained evidence

The coordinator ran the gates in this order, with one build process and
1,500-second caps. The complete logs and source hashes are retained in
[NIFS_CONCRETE_CONSUMER_EVIDENCE_01a8fd8c.zip](../../docs/reviews/nightstream-fprime-requirements/NIFS_CONCRETE_CONSUMER_EVIDENCE_01a8fd8c.zip),
SHA-256 `9808e2878384349c3d6d8af284216525475b9df0d0a557b6f0043c9631c2a8c5`.

| Gate | Last output |
|---|---|
| `scripts/validate.sh static` | `[boundary] all checks passed` |
| `scripts/validate.sh build` | `Build completed successfully (3836 jobs).` / `[bounded] exit=0 elapsed=57s` |
| `scripts/validate.sh axioms` | `Build completed successfully (3926 jobs).` / `[bounded] exit=0 elapsed=46s` |
| `scripts/validate.sh identity` | `[bounded] exit=0 elapsed=130s` / `[identity] canonical binding, structural identity, package identity and verifier-key pins match` |

All new public theorems are in `tests/AxiomsNifsClosure.lean`. Only the allowed
`propext`, `Classical.choice` and `Quot.sound` axioms occur. The focused final
consumer elaborated in 1.9 seconds. `ClaimCheck.evaluation_eq` passed on
attempt four after the owner authorized up to ten attempts for that
particular declaration. No resource override or diagnostic command was added.

Independent reviewers checked the provider/event correspondence and reuse
of the retained Lean/optimized execution. See
[NIFS_CONFORMANCE_CLOSURE.md](../../docs/reviews/nightstream-fprime-requirements/NIFS_CONFORMANCE_CLOSURE.md).
The exact source and archive check ran no new native test. The native
sources and selected emitted identities did not change.

## Requirement-link proposals

| Record | Axis | Before → proposed after | Closing declaration and evidence | Checked commit |
|---|---|---|---|---|
| `N.security.binding` | Link | partial → connected | `NifsBinding.bindingEvent_to_shortKernel`; approved fixed-seed MSIS premise; selected `NifsClosure` consumer; all gates above | `01a8fd8c` |
| `N.security.fiat_shamir` | Link | open → connected | `NifsClosure.finishValue_probability_and_expected_work`; owner-approved `FIAT_SHAMIR_MODEL.md`; all gates above | `01a8fd8c` |
| `N.conformance.owners` | Link | partial → connected | `NifsExtractionProvider.suffixProgram_correct`, `parentChecker_spec`, `batchAt_eq`; selected `NifsClosure` consumer; retained same-input Lean/optimized replay | `01a8fd8c` |

Binding and FS retain Proof axis `assumption`. The owner record retains
Proof axis `not_required`. These changes do not claim new cryptographic
hardness proofs, machine-runtime bounds, arbitrary-input Rust verification,
a backend or full-history extraction. The older `NifsFiatShamir` theorem
remains the general provider/preparation interface; the selected final
consumer is now `NifsClosure`.

## Blocked on

- Apply the three reviewed link changes above? The supplied closure brief
  says: “Do not edit requirements.json statuses; propose the change in the
  report.” Concurrent site edits are preserved and are not in these commits.

The broader `N.security.error_budget` record remains partial/open. It asks
for deployed verifier false-acceptance over uses, depth and shared queries.
The approved FS boundary selects no numerical `g_d`, `delta_d`, query budget,
depth or MSIS advantage. The existing union bound covers named test and
sampler-abort events under their per-call laws. It excludes FS/hash/MSIS and
history terms. The invalid-source corollary does not close that record.
The blanket full-profile mutation diagnostic and production backend remain
separate from the scoped NIFS execution evidence.
