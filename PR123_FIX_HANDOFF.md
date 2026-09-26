# PR #123: partial repair handoff

Date: September 26, 2026. Branch: `nico/pirlc-wide-sampler-integration`.
Exact starting commit: `bb7fb8db6dc3d525ce5435eb844804f34cd81feb`.
This file is part of the new handoff commit. The user asked to stop implementation,
commit and push the work, and let another AI continue on a different computer.

**This change is unfinished and is not ready to merge.** Do not use the passed
component checks as proof that the final source and all saved fixtures agree.
The original review, paper analysis, commit ledger, and review-comment assessment
are in [PR123_DEEP_REVIEW.md](PR123_DEEP_REVIEW.md). Its opening update describes
this repair; its main review describes the starting commit.
The earlier [PR123_HANDOFF.md](PR123_HANDOFF.md) is a historical record.

## Contract and fixed choices

The requested result is one maintained production path, with no supported old
application layout or legacy lifecycle dependency. Close R1–R4 and the earlier
handoff gaps. Prove actual application-row custody; use matching artifacts and
verifier pins; preserve exact matrix, assignment, nonzero-fold, and rejection
checks. Keep Lean local, as the owner requested.

The profile remains Goldilocks, `b = 2`, `k_rho = 16`, `B = 65536`, with Poseidon2
protocol binding. The existing rank, 28 rounds, domain, key seed, maximum
4,708,530 key columns, and zero-extension policy are unchanged. This work adds no
Rust feature, environment variable, protocol hash family, cryptographic assumption,
or application source-custody premise.

Read the September 4 SuperNeo v1.2 text in `docs/superneo-paper-v1_2` and the
HyperNova sections cited in the deep review. Preserve separate Pad/`Eval_K` and
matrix/`Eval_A` checks, all 14 matrix slots, and the verifier-driven transcript.
Some repository HyperNova notes contain local edits; the review distinguishes
those notes from the original paper.

## Completed source and artifact work

| Area | Changes and evidence |
|---|---|
| R2: exact application rows | Fixed guarded instruction and assertion mapping. Added structural compiled-row mapping and index facts, canonical application row pairs and bounds, supported-column inverse recovery, and the actual application case in physical source custody. The focused proofs and their audits pass. |
| R2: dimensions | Updated wide and reused reference counts. The wide carrier has 137,646,804 logical coordinates, 137,646,810 padded coordinates, 3,256,394 logical rows, and 2,549,015 key columns. The general application has 4 witness values, 7,696 locals, and 7,700 rows. |
| R2: audits | Added `tests/AxiomsWidePhysicalArchive.lean`, removed stale audits, and registered the new target. Full production and axiom targets passed together. No new axiom, `sorry`, recursion limit, or heartbeat limit was added. |
| R1: one schema | Rust requires manifest version 3 and rejects version 2, the old paired reference, and a missing application-local index. The emitted manifest has one reference and 14 phase children. Lean shared-verifier checks passed for both the selected and identity applications. All seven Rust assembly tests passed after promotion. |
| R1: selected artifacts | Emitted and checked the package, physical expansion, binding, setup, base fixture, component parity files, and manifest, then promoted them with matching Rust pins. Removed the unused duplicate candidate artifact. The crate package remains a symlink to the formal artifact. |
| R3: current tools | Removed the obsolete replay and bridge coordinators. The golden workflow uses the maintained library checker. Added maintained PiCCS and child-handoff checks; these compile, but their new operations still need execution on fresh output. Updated graph registrations and removed orphaned gates and inputs. |
| R3: process cleanup | Fixed timeout cleanup to stop nested process groups. The real nested-timeout regression and Python suites passed. |
| R3: deprecated support | Removed six unused Metal kernels and their old shader files. Retained the production ChaCha20 key expansion and moved its unchanged helpers into its owning shader. Refreshed the independent minimizer lockfile. Final Metal and minimizer checks are pending. |
| R4 | Removed the old eight-field Phi81 challenge encoding, obsolete sampler census API/constants, and unused old-key successor wrappers. Kept the generic membership lemmas needed by the selected path. Matrix tests passed 23 cases, with one ignored; parameter tests passed 10 cases. |

The application is compiled once at canonical coordinates. Its exact row syntax
is then relocated. The column inverse is proved on the application's supported
reads, not on all natural-number columns. This retains the general application
path without an identity-based application mode.

The application parity emitter previously built a large row plan only to count
it. It now uses the derived count, with a private theorem proving equality to
the plan's row count. The repaired emitter built and ran in 47 seconds.

The last full Lean production and axiom build passed in 8 seconds, incrementally.
After that pass, four unused old identity/context definitions were removed from
`VerifierContextCandidate.lean`, and reference-fixture headers were clarified.
A final full Lean build and static check after these edits are still required.

## Current selected identities

| Value | Four Goldilocks words |
|---|---|
| Structural identity | `[1079497172432010844, 16672848630866421234, 8750472192629984461, 1306370760931750244]` |
| Package identity | `[15031848204567304573, 4564442748971577180, 11142206294796127265, 239069948372924710]` |
| Transcript context | `[12467448792212941571, 15091756229906015131, 169265933689485490, 16666007258455189345]` |
| Verification key digest | `[6983138835289037493, 10537484322204503039, 12099023242372410068, 11555272426941468087]` |

The transcript context and verification key digest are different values. An early
scratch base fixture used the latter in the context field. It was regenerated
with the correct context before the conformance checks and promotion.

Physical geometry: 27,724,114 rows; 27,867,239 columns; 27,866,960 private/constant
columns; 278 public columns. The matrix program has 40 blocks. The general
application ordinary block is 37, the next block is 38, and the final pin is 39.
The manifest reference is `[4, 7696, 7700]`; `application_local_index` is 21.

## Evidence before pin changes

All candidate checks below ran with explicit candidate identities while the old
production pins were still installed. Each non-Lean command had the required
300-second cap. The comparison uses values and rows, not digest agreement alone.

| Check | Result |
|---|---|
| Physical expansion | Passed, 30.19 s. Compared every A/B/C entry with the separate Lean expansion. Nonzero counts: `[91935932, 37291416, 27552988]`. |
| All logical matrices | Passed, 64.36 s. Exact independent comparison of all 14 slots; 2,609,213,916 total nonzeros. |
| Matrix mutations | Passed, 99.59 s. Rejected block order, column, and coefficient mutations. |
| Base assignment | Passed, about 93 s. Checked all physical rows, every logical coordinate, six alignment zeros, and all logical rows. |
| Detached application | Passed, 92.91 s. Application mismatch rejected at logical row 3,256,381. |
| Nonzero assignment | Passed, 103.80 s. Included 76 assignment-block mutations, 13 matrix-slot mutations, zero-slot rejection, 256 digest-bit and four digest-word mutations. |
| Phi81 recipe mutation | Passed, 144.97 s. Rejected at row 3,069,186. |
| Challenge-bit mutation | Passed, 146.45 s. Rejected at row 3,058,284. |
| Output-digest mutation | Passed, 145.90 s. Rejected at row 3,054,641. |
| Sparse commitment | Passed after repair of a stale last-block probe. Compared all 1,188 coefficients and three support coordinates, including the actual final carrier coordinate. |
| Wide sampler parity | Three tests passed. |

The exact logical nonzero counts are
`[16904205, 3151742, 265314580, 25940975, 801029256, 1496768506, 0, 104652, 0, 0, 0, 0, 0, 0]`.

## Fresh native run: partial success and current failure

The first fresh native run failed because two producer phases still compared
new output with old saved fixtures. Removed those producer dependencies from
`staged_fold.rs` and `staged_terminal.rs`. Fresh NIFS verification remains, and
the separate golden checker still compares complete results, proof bytes, and
caller words.

After that change, the resumed CPU run passed these stages on the new artifacts:

| Stage | Step 1 | Step 2 |
|---|---:|---:|
| Sources | 77.13 s | 140.97 s |
| PiCCS | 98.32 s | 144.71 s |
| PiRLC | 31.44 s | 34.44 s |
| Split | 99.98 s | 106.80 s |
| Openings | 125.55 s | 131.82 s |
| NIFS | 109.55 s | 116.39 s |
| Successor | 143.09 s | 152.80 s |

Base construction passed in 78.51 seconds. Terminal acceptance at state 3 passed
in 239.14 seconds. Preparation of a changed witness and a new commitment passed
in 83.47 seconds. Its terminal relation rejection passed in 239.57 seconds.

**The combined balanced `Eval_K` test reached 300 seconds and was killed. This
is a failed check.** The weighted PiDEC sum was preserved, and the fresh witness
and commitment were rebuilt, but the terminal check did not finish under the
cap. `Eval_A` and both fresh Lean comparisons were not reached.

The last code change splits each balanced-opening test into preparation and
verification. Preparation saves the complete changed state and witness.
Verification reloads it, checks the exact balanced changes, and requires the
production verifier to reject at `Running { index: 0, reason: Eval_K/Eval_A }`.
It must not pass merely because some earlier check rejects the input.

The new phases are `opening-k-prepare`, `opening-k`, `opening-a-prepare`, and
`opening-a`. The driver runs each preparation before its check. The eight
coordinator tests pass. `cargo fmt --all` ran, and the final
`timeout --signal=KILL 300 cargo test -p nightstream --release --lib --no-run`
passed in 45 seconds. **The split tests have not run.**

There is no successful `cpu-result.json` and no fresh `lean-step-1` or
`lean-step-2` result. Do not turn the failure record into a pass or promote the
old fixtures on the basis of these partial results.

## Work to continue

1. Execute the split balanced-opening preparation and rejection checks under the
   existing caps. The exact commands are in `scripts/GOLDEN_CONFORMANCE.md`.
2. Complete fresh Lean verification and full byte/value comparisons for both
   native folds. Use the pinned toolchain and the maintained
   `crates/nightstream/tests/check_lean_fold.py` driver. The normal fresh workflow
   is `python3.12 -B scripts/golden_conformance_ci.py --directory NEW_DIRECTORY`.
   On the original computer, successful native stages can be reused with their
   real receipts. On another computer, regenerate the required native output.
3. Execute the new maintained PiCCS checker operations on both folds. The checks
   cover acceptance, 562 proof mutations, 282 statement mutations, and 843 output
   mutations. Run the 56 nonzero-point mutations on the second fold. Execute
   child handoff checks for both folds; they recompute commitments and public
   values from all 17 current sources. These new operations currently have
   compile evidence only.
4. Regenerate the saved fold fixtures only after fresh comparisons pass. See the
   table below. The current saved native proof, Lean NIFS/caller fixtures, and
   `golden-wide-v1.zip` are still for the old package.
5. Rebuild `check_package_conformance` after its last interface edit. Its
   recursive checks now require the complete 10-field NIFS result, not the old
   13-field folded-metadata form. Run actual recursive raw-assignment and
   mutation checks with the new caller and complete result. The earlier base and
   exact matrix passes do not prove this recursive handoff.
6. Run the final Lean production targets, axiom audits, static checks, and
   identity checks. Run the workspace release all-target check with
   `nightstream/metal`, the affected Rust release tests, and minimizer tests.
   Final Metal compilation and runtime checks have not run after shader cleanup.
7. After fixture refresh, run the public two-link application mutation check,
   the Metal two-link flow, and CPU/Metal terminal compatibility. Relevant test
   names are `two_link_hash_chain_base_step_verifies_and_rejects_changes`,
   `two_link_hash_chain_folds_on_metal`, and
   `metal_terminal_accepts_cpu_proof_and_matches_cpu_rejection`.
8. Update current documentation, this report's implementation status, and the
   source-bound review requests after the final source and artifacts agree.
   Independent review acceptance remains pending. Do not sign the author's own
   acceptance records or treat old accepted records as approval of this change.

The maintained golden test is
`lifecycle::tests::staged::fold::lean::golden::native_checker`. It reads JSON on
stdin. Its operations include `compare`, `encode`, `ccs`, and `child-handoff`.
The child-handoff request takes `package`, `identity`, `input`, `lean`,
`children`, and `output`. Use the current PiCCS input and complete PiCCS result;
the first six fields of a successful 10-field NIFS result are that result.
Graph registrations in `scripts/lean_graph/obligations.json` specify the
current checker requests.

### Saved fixture update after successful fresh comparisons

| Fresh checked source | Saved target |
|---|---|
| `lean-step-1/step-1-nifs.json` | `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-base-nifs-result-v1.json` |
| `lean-step-1/step-1-caller.json` | `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-actual-recursive-step-fixture-v1.json` |
| `cpu/fold-1/actual_result.json` and `proof.native` | Matching names in `crates/nightstream/tests/fixtures/stage1_actual_nifs/` |
| Both folds, their checked caller/NIFS results, and state envelopes | `crates/nightstream/tests/fixtures/golden-wide-v1.zip` |

The vector archive has 19 interface files. For each step 1 and 2, retain
`native/fold-N/{pi_ccs_input.json,children.json,actual_result.json,proof.native,caller-inputs.json}`,
`native/step-N/{envelope.json,fresh-claim.json}`, and
`expected/step-N-{nifs,caller}.json`. Also retain `native/step-3/envelope.json`.
Do not add large private witness files to this interface archive.

## Local evidence and execution rules

The original computer has logs and candidate files in
`/tmp/nightstream-pr123-fix/`. The partial native run is
`golden-current/cpu`. The two old-fixture failures are retained under
`cpu/attempts/`; the latest timeout is in `cpu/logs/opening-k.log` and
`golden-resume.log`. `cpu/conformance.json` records failure. The last build log
is `opening-stages-build.log`.

These temporary files are not in Git and may not exist on the next computer.
They are not required to understand the open work. A scratch
`promote_fold_fixtures.py` has not run; it requires successful CPU and both Lean
result records. A scratch resume script predates the new preparation stages and
must not be used unchanged. Unsigned review proposals exist only in scratch;
repository review requests and independent acceptance records were not changed.

Use the pinned Lean toolchain on the original computer; its local override is
old. From `formal/nightstream-fprime`, use:

```sh
elan run leanprover/lean4:v4.32.2 bash scripts/validate.sh COMMAND
```

Use the applicable `AGENTS.md` instructions. Run one Lean or Rust build/test at
a time. The project requires a cap of 300 seconds for each non-Lean test and
1,500 seconds for each Lean command. Lean commands must use `validate.sh`.
Use release Rust tests and `FoldingMode::Optimized`. Do not raise limits to make
a failed run pass. Do not add proof premises or unsafe evaluation shortcuts.

No protected owner file or frozen Lean project was changed in this repair.
Keep the existing owner files and independent review records intact. The last
external-review check, at 17:20 UTC on September 26, found no change. Those
reviews refer to older source and are not acceptance of this handoff.
