# NIFS conformance closure evidence

Source reviewed: `b2cfe16559ae1684dfc711335690f183da8644ab`.
This note checks reuse of the recorded nonzero Lean/optimized NIFS execution.
The concrete provider and final consumer were then checked at `01a8fd8ca68280c7f642451ab33bc714cee5bc98`.
Their static, full-library, axiom and unchanged-identity gates pass, as
recorded in `formal/nightstream-fprime/NIFS_CLOSURE_STATUS.md`. This does not
declare complete deployed-security or backend closure.

The selected execution uses the Nightstream Goldilocks profile: `b = 2`,
`k_rho = 16`, `B = 65536`, one fresh source, 16 running sources, 16 output
children, 14 matrices, 28 rounds and Poseidon2 transcript binding. Pad and
matrix evaluations remain separate. This is not the paper's reference
`k_rho = 14` profile.

## Retained evidence

The complete replay ran at
`9dcb4e7b9d7b16ac15e7f19c2ab4a4db007708f5`. The independent semantic and
consumer review starts at
`773f3d0f29209b33e2325538d5f258f541569c25`; its later sections name their
separate source cuts. Archive hashes identify files, not protocol authority.

| File | SHA-256 |
|---|---|
| [NIFS_COMPLETE_REPLAY_EVIDENCE.zip](NIFS_COMPLETE_REPLAY_EVIDENCE.zip) | `5a74f1f2527f5de2d7a88ddc6ff40d1c7709352fc26f7e2318346f5dc0a28987` |
| [NIFS_DEC_AND_FINAL_OUTPUT_EVIDENCE.zip](NIFS_DEC_AND_FINAL_OUTPUT_EVIDENCE.zip) | `11aa7369a8399d256cbbf2ae12eb24329ee219852a9a73f354224aa71da8d5b7` |
| [NIFS_RUNNING_AND_NATIVE_RLC_EVIDENCE.zip](NIFS_RUNNING_AND_NATIVE_RLC_EVIDENCE.zip) | `b69f229bbbda9cbdc734d103c3536fcea307e5725c136c3f88b2e4d7b18264e4` |
| [NIFS_PUBLIC_AND_BATCH_EVIDENCE.zip](NIFS_PUBLIC_AND_BATCH_EVIDENCE.zip) | `950783ee98dbb14d1414713f0f4387f61bde178fb27cf6a376db9cf9bd2ad961` |
| [NIFS_CONFORMANCE_REVIEW_773f3d0f.md](NIFS_CONFORMANCE_REVIEW_773f3d0f.md) | `7f64a406232a319d70347462f9128d175cf35850dff0db9310e258bfcf9c06d3` |

The current read-only archive check verified all 35 source hashes, 15 fixture
hashes and 22 log hashes listed in the complete replay manifest. It compared
the full common C input, R parent fields, ordered D/NIFS children, eight-word
outgoing state and selected structural identity. All comparisons agree.
Independent encoding from the retained raw Lean fields reproduces all
945,983 native proof bytes, with SHA-256
`3cc9d9d9f58fb4a11d30fff99c632ab657f3f2747e8a94ea685ac5deb0387f3e`.
The comparison covers every length, commitment, public input, point, Pad and
matrix evaluation, padding word and carried frame digest in that encoding.

## Source changes and execution scope

Every Rust source file listed by the complete replay manifest is unchanged
at the reviewed cut. Three listed Lean files changed: `PiCCSInputCheck`,
`PiRLCInputCheck` and `PiRLCParent`. Their changes add the exact probe view and
correctness links; they do not change the existing executable C/R checks or
parent calculation. The selected protocol, profile, transcript and NIFS
verifier owners inspected since `773f3d0f` are unchanged.

The separate native circuit entry changed after that review. It now rejects
unsupported compressed PiCCS synthesis before it changes the builder or
transcript. The retained public-and-batch archive records the regression
failing on the former body and passing on the guarded body, plus release
checks for `neo-fold-clean` and `neo-wasm`. This guard does not provide a
package-backed recursive implementation.

All results below are reused execution evidence. This review ran no Lean
build, Rust build, native test, fixture generator, PaperExact path or backend.

| Recorded check | Result and limit |
|---|---|
| Optimized fixed NIFS, round trip and cache substitution | 3/3, 9/9 and 1/1 passed. The final fixed test also rejects a same-shape cache for another relation. |
| Complete selected C/R/D and public NIFS replay | C/R took 88.68 seconds, D 12.03 seconds and the whole verifier/wire/NIFS mutation check 0.116 seconds. Exact final children, checked parent and full transcript state agree. |
| Rejection cases | The retained log has 43 NIFS and 55 PiDEC rejections. Earlier independent Lean mutation checks retain their stated scope in the D archive. |
| Affected caller targets | Six targets compiled; that record does not say their tests ran. The earlier empty-running `nifs_r1cs_isolated` failure remains recorded. |

The selected runtime uses the caller-selected relation and parameters. Its
prior-parent check replays PiDEC over the carried children before C; it does
not accept a matching digest as proof of those claims. Fixed and lifecycle
callers retain preprocessing validation. Lean proves its stated semantic
implications. Equality on the recorded inputs and rejection cases does not
prove Rust behavior for every input, the Rust compiler, or a proof backend.

## Record decisions

`N.conformance.chain` and `N.conformance.executed` already have connected
links for the recorded Lean/optimized execution. No new execution is needed
to preserve that unchanged claim. The earlier chain requirement named
Lean/PaperExact/optimized three-way equality, while its evidence and remaining
text covered Lean/optimized equality. This update aligns the requirement
with the recorded and authorized NIFS scope:

> Compare the complete phase results and caller handoffs for the archived
> nonzero base and actual-child recursive fixtures between executable Lean
> and Rust optimized, with the same selected key, profile, serialized inputs
> and proof fields. Record broader PaperExact comparisons separately.

`N.conformance.owners` link is connected after the recorded model approval
and concrete consumer gates. `NifsExtractionProvider`
constructs `suffixCorrect` and `parentChecker_spec`; `batchAt_eq` connects
the selected relation, statement and receipt. The validated
`NifsClosure.finishValue_probability_and_expected_work` consumes that
provider and the exact checked prefix. The retained unchanged Lean/optimized
replay supplies the scoped native connection. The independent final review
found no remaining checker-correctness or hidden source-witness premise.
The owner approved the separate parametric Fiat–Shamir boundary; the
conformance review supplies no cryptographic model instance.

Same-input PaperExact R/D comparison, full-profile evaluator approval,
production backend execution and full history extraction retain separate
obligations. The historical blanket logical-mutation diagnostic still fails
for three duplicate allocations unused by the decoded NIFS path. The
independent review identifies the actual constrained consumers; it does not
find an authoritative NIFS field that those mutations can change. This note
does not turn that failed diagnostic into a passing result.
