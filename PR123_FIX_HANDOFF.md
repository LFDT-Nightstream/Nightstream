# PR #123: completed fixes and validation

September 26, 2026. Branch: `nico/pirlc-wide-sampler-integration`.
This completion started from `1e90095b805c30f1a86a9c0665dcbc9197e9006f`.

**The implementation fixes and local validation pass. Independent review and
signed acceptance remain pending.** The current review requests identify the
source cut in [the review directory](docs/reviews/nightstream-fprime-requirements/wide-target-reviews/README.md).
Passing tests are conformance evidence, not a universal proof of Rust correctness
or a production security certification.

The main body of [PR123_DEEP_REVIEW.md](PR123_DEEP_REVIEW.md) reviews the earlier
`bb7fb8db6` checkpoint. The previous partial-repair handoff is retained in
[Git history](https://github.com/LFDT-Nightstream/Nightstream/blob/1e90095b805c30f1a86a9c0665dcbc9197e9006f/PR123_FIX_HANDOFF.md).

## Result

There is one selected emitter, one manifest schema, one general application
assembly route, and one physical witness executor. Canonical application
compilation and proved placement supply the exact application rows. Required
mathematical reference facts remain in the Lean dependency graph; they do not
select another runtime protocol.

The profile remains Goldilocks, `b = 2`, `k_rho = 16`, `B = 65536`, 14 matrix
slots and 28 rounds. Poseidon2 binding, the key seed, approved maximum of
4,708,530 key columns, and zero-extension policy are unchanged. This completion
adds no Rust feature, environment variable, protocol assumption, or proof premise.

The selected package, manifest and verifier pins from the preceding repair were
retained. Fresh canonical binding generation confirms that the structural,
package and verification-key identities match. The current constants are in
[identity.rs](crates/nightstream-fprime/src/identity.rs).

## Repairs completed in this continuation

- Fixed the key-prefix regression to derive the selected width from the
  version-3 manifest's sole reference and affine geometry.
- Replaced the saved native proof/result, Lean NIFS/caller fixtures and the
  19-file golden archive with freshly checked outputs. No private witness files
  were added to the interface archive.
- Removed the unreachable Metal selector, its unused pipeline and shader code.
  The generic coefficient evaluator remains the sole joint-round path.
- Removed the two unused legacy-producer checkers from `protocol-contract`.
  Cleared their deleted source anchors, which already had evidence level `none`,
  and marked the missing circuit and decider artifacts unresolved. Regenerated
  the derived views without changing normative rules or claiming replacement
  refinement evidence. Repository validation and the sealed migration audit pass.
- Corrected the graph regression to require the complete NIFS result instead
  of the retired folded-metadata input.
- Added change-selection and conformance-registration tests to CI and removed
  the obsolete commented WASM job. Lean remains a local check.

## Fresh execution and conformance

All 22 native stages passed. Both folds include sources, PiCCS, wide PiRLC,
PiDEC decomposition and openings, NIFS verification, and successor construction.
Terminal acceptance and the recommitted-witness rejection passed. Maximum native
stage duration was 126.13 seconds; maximum recorded RSS was 9.75 GiB.

The balanced opening tests preserve weighted PiDEC recomposition, rebuild the
complete fresh witness and commitment, and pass all earlier terminal checks.
The verifier then rejects child 0 specifically at `Eval_K` or `Eval_A`.
The preparation/check durations were 77.70/122.62 seconds for K and
76.70/123.86 seconds for A. The earlier combined-test timeout is resolved by the
separate preparation and verification stages.

Both folds passed fresh Lean verification, exact canonical proof-byte equality,
every C/R/D result comparison, complete caller-word equality, all 55 native
PiDEC mutation cases, and the Lean rejection cases.

The maintained PiCCS checker also passed 562 proof, 282 statement and 843 output
mutations on each fold, plus 56 nonzero-point mutations on the second fold:
3,430 PiCCS mutations in total. Both child handoff checks recomputed commitments
and public values from all 17 sources.

Independent recursive assignment checks passed on both folds. Each checked all
27,724,114 physical rows, all logical coordinates and six alignment zeros, and
all 3,256,394 logical rows. The complete 10-field NIFS result is required at this
boundary. Assignment and mutation commands took 62.58/17.76 seconds for the
first fold and 67.59/22.10 seconds for the second.

These checks passed before saved fold fixtures were promoted. The preceding
repair's exact matrix and candidate checks remain recorded in the historical
handoff; this continuation made no matrix or circuit change.

## Final checks

Durations below cover the command, including any compilation.

| Check | Result |
|---|---|
| Workspace release all-target check with `nightstream/metal` | Passed; 8.48 s |
| Full Nightstream release suite | 46 passed, 13 ignored; 111.42 s |
| Full F-prime release suite | 113 passed, 40 ignored; 296.76 s |
| Packed witness / matrix-row / wide sampler tests | 5 / 13 / 3 passed |
| Parameter tests | 20 passed |
| Saved Lean foundation comparisons | Field, extension, bar and signed-binary checks passed |
| Metal unit tests | 23 passed |
| Two-link application mutation | Passed; 242.53 s |
| Public two-link Metal fold | Passed; 49.91 s |
| CPU-proof acceptance and matching rejection on Metal | Passed; 52.62 s |
| Independent minimizer | 26 passed, 1 ignored |
| Golden coordinator and change selection | 17 passed |
| Native coordinator / Lean driver / archive regressions | 8 / 6 / 3 passed |
| Lean graph tools | 95 passed, 1 skipped |
| Protocol-contract tests | 50 passed; repository and sealed-import checks also passed |
| Lean static boundaries | Passed |
| `NightstreamFPrime` and `NightstreamFPrimeTests` | Passed together; 132.14 s |
| Fresh canonical binding and identity pins | Passed; 82.68 s |

Ignored or skipped tests are not counted as passes. The named full-profile checks
above were explicitly executed. All native commands retained the 300-second cap;
Lean commands used the pinned 4.32.2 toolchain and `validate.sh`'s 1,500-second cap.
No longer invocation was needed. Rust formatting and whitespace checks pass.

## Reproduction and review

The maintained fresh workflow is:

```sh
elan run leanprover/lean4:v4.32.2 python3.12 -B scripts/golden_conformance_ci.py --directory NEW_DIRECTORY
```

[The conformance instructions](scripts/GOLDEN_CONFORMANCE.md) describe the native
stages and the separate row checks. The current checker accepts `compare`,
`encode`, `ccs` and `child-handoff` JSON requests on stdin. PiCCS requests use the
six-field prefix of the fresh ten-field NIFS result and the structural identifier;
package identity and transcript context are distinct values. Current graph
registrations specify the request fields and mutation groups.

Local receipts and logs are under `/tmp/nightstream-pr123-final/`; they are not
required to build the repository. The checked interface data is committed in
`golden-wide-v1.zip` and the refreshed native and Lean fixtures. Reproduction on
another computer must regenerate private witnesses for terminal-opening checks.

The review directory records the source cut for the four security review requests.
Their acceptance is still pending. An independent reviewer and the controlled
review process must provide those acceptances; old responses and this author's
test results do not approve the new source.
