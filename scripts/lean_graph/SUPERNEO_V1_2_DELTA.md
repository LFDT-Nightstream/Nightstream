# SuperNeo v1.2 proof changes and implementation obligations

Reviewed source cut: `aa04ac75263121d28ad7d99fedd41b48b9d579e8`.
The owner made the supplied September 4 v1.2 paper the review reference for
the active HyperNova/NIFS closure goal. This note precedes proof changes.
It grants no v1.2 security or production-conformance status.

The new source is `docs/superneo-paper-v1_2`, split without text changes from
`superneo_v1_2.pdf.md`, SHA-256
`e7ac49cd4e2b45d96443c69bd654498a002c057446c95ea9e83c31d2f2f2cf82`.
The comparison source is the local P1-corrected v1.1 corpus in
`docs/superneo-paper-v1_1`. Its corrections are local review changes, not
author-issued wording. Both papers credit Wilson Nguyen and Srinath Setty.

## Changes that affect the checked proof

| v1.2 source | Difference from the checked v1.1 copy | Required implementation connection |
| --- | --- | --- |
| Section 6, Definition 16 | Adds an expected-polynomial-time uniqueness adversary that can make adaptive oracle calls and select two actual responses from different calls. Fresh randomness and all oracle work are included. | Keep actual receipts, weak endpoints and clocks through selection; prove support and work for the selected pair. The existing independent-pair law is insufficient by itself. |
| Section 6, Definitions 17 and 18 | Weak and strong uniqueness now quantify over those adversaries. Their old numbers were 16 and 17. | Derive the required uniqueness failure bound from the computed fixed-key MSIS reduction. Do not add adaptive uniqueness as a new hardness assumption. |
| Appendix B.1 | The strong/weak composition simulates each adaptive oracle query and keeps its returned values. | Connect the selected continuation to the same fresh-call law under the original context. Preserve aborts, prior inputs and transcript state. |
| Appendix B.2, proof of Lemma 7 | Enters a retry loop after extraction error, waits for another relaxed success, and obtains error at most `epsilon_test + epsilon_uniq`. The old argument used a square root of a two-call disagreement term. | Prove the stopped law, almost-sure termination and unconditional work bound. Carry the new event through the actual source-return and history consumers. |
| Appendix B.3, proof of Lemma 8(ii) | Stores the internal extraction result for every oracle call and retrieves the two selected results before computing a relaxed-binding collision. | Reuse `BindingReduction.run` on those retained endpoints. Charge selection/custody and all calls, including discarded calls. Bound its actual emitted-vector success. |
| Section 4, Theorem 7 | The supplied text writes the loss with the vector challenge-set denominator. Our local correction used the one-coordinate denominator. | Retain the checked coordinate loss. `CoordinateRetry` and the coordinate-fork laws establish the one-coordinate bound; no stronger bound follows from the supplied notation. |

Section 5 contains native/packed notation corrections, including flattened
padding and the native matrix product in the constant-term identity. The
selected code already separates those carriers. Section 7 retains the
separate Pad and genuine-matrix families, the same C/R/D order, strict norm
checks and recombination equations. R and D appear as HTML tables in the
supplied Markdown; they were read as protocol text, not executed as code.
No changed verifier check or wire field was identified in this comparison.
Exact implementation conformance remains supported by its named Lean and Rust
evidence, not by this textual comparison alone.

The source still has extraction/formatting defects: the B.2 adversary's fresh
witness slice is printed inconsistently with the extractor above it; B.3
Equation (37) runs four norm expressions together. The intended slice and
four separate strict bounds are established by the surrounding proof and
the existing typed constructors. Preserve the supplied paper and record such
issues here rather than editing it.

Section 8 still gives reference parameter rows. The selected implementation
remains Goldilocks, degree 54, `b = 2`, `k_rho = 16`, `B = 65536`, one fresh
source, 16 running sources, 14 matrices and 28 rounds. Its fixed public-seed
MSIS premise and same-seed prefix reduction remain the authority for its
commitment setup. The paper's uniform-setup estimates and reference radix
depths do not select Nightstream parameters or a numerical security budget.

The supplied paper omits the local v1.1 classical-scope warning. This does
not prove quantum extraction or QROM security. The approved classical
additive-Poseidon2 FS boundary remains explicit.

## Chosen retry experiment

At one fixed original context let `p` be the actual relaxed-success
probability, `a` the actual source-error probability, `d` the disagreement
probability of two independent successful calls, and `e` the existing test
error. The checked private lemma `StrongProbability.pair_bound` gives
`a * p <= d + a * e`, and the checked ranges give `0 <= a <= p <= 1`.

Use this Definition 16 adversary: make one actual call; abort if its relaxed
check rejects; otherwise keep making fresh calls until the relaxed check
accepts again, then return the two retained responses. This is a valid
adaptive uniqueness experiment. It enters after relaxed success rather than
only after source error, so it does not need a new source-membership checker.
The existing checked ambient verifier supplies its acceptance test.

For `p > 0`, its selected disagreement mass must be proved equal to `d / p`.
The pair inequality then gives `a <= e + d / p`. If `p = 0`, the loop is not
entered and `a = 0`. This division is a proposed observable until its exact
first-hit law is proved; a normalized expression alone is not a sampler.
Each original context keeps its own `p` and `d`. The final bound averages
`d / p` over the unchanged context law, including aborted experiments; it
does not divide the global disagreement mean by the global success mean.

If one complete checked call has mean work `t`, the unconditional oracle work
is `t + p * (t / p) <= 2 * t`, with the zero-success case handled separately.
Each rejected response contributes its actual work. Call work may depend on
the response. Record retention, loop control and final reduction work must
also be charged under the existing declared clock. The same accounting is
needed for query counts. This gives no fixed worst-case runtime budget.

## Closing declarations and consumers

Names marked proposed identify the next obligations, not checked results.

| Obligation | Closing declaration | Existing consumer or authority |
| --- | --- | --- |
| Exact finite retry output and charged rejected prefix | Proposed `AcceptedRetry.search_firstHit` | Existing `PaperForkExtractionWork.Result`; actual checked observation values |
| First-hit distribution, zero-success case, termination and unconditional call/work sums | Proposed `AcceptedRetry.firstHit_hasSum`, `exhaustion_tendsTo_zero`, `entered_work_hasSum` | The repeated-call experiment above; existing coordinate-retry proofs are reference lemmas |
| Pointwise reduction for supported selected endpoints | Proposed `BindingProbability.supported_binding_le_success` | Existing `BindingReduction.bindingEvent_implies_success` and `runPair` |
| Actual stopped selection to fixed-key MSIS success and work | Proposed `AdaptiveBinding.selected_binding_le_success` and its charged driver | `BindingProbability`, `BindingWork`, the selected primitive/accessor correctness proofs |
| Linear source-error bound on unchanged context laws | Proposed `StrongProbability.source_error_le_retry` | Existing `pair_bound`, source-success partition and causal test bound |
| Final NIFS and history conclusions | Linear-bound consumers in `NifsClosure` and `HyperNovaVisitedSecurity` | `InteractiveComposition`, `InteractiveAgreement`, `SupportedExtraction`, `NifsInvalidSource`, `HyperNovaSourceWork` |

The concrete primitive and witness checks, complete selected assignment,
accepted-successor and deterministic history theorems remain reusable.
Their statements do not acquire a stronger result merely by changing the
paper citation. The source extractor still makes its existing call; the new
retry adversary belongs to the security reduction.

The approved fixed-seed assumption selects no numerical adversary budget.
The new reduction must expose its full work/query requirements before a
concrete hardness advantage can bound it. If a fixed worst-case budget is
needed, a proved truncation loss is required; an expected-work bound alone
does not supply that budget. No stronger hardness premise is approved here.

## Validation and remaining scope

Check declarations in isolation, then run static, build and axioms in order
at each proof checkpoint. Reuse package/conformance evidence only while its
relevant source and identities remain unchanged. Record all failed attempts
against the existing owner-approved limit; renaming does not reset it.

The FS applicability study, later actual-witness Rust lifecycle handoffs,
and final event/resource composition remain part of the active goal.
No general FS formalization, machine-runtime proof, backend choice or
architecture change is introduced by this paper upgrade.
