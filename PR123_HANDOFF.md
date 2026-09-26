**PR #123 — implementation handoff**

The user asked to stop this implementation slice, commit and push it to `nico/pirlc-wide-sampler-integration`, and let another AI continue. This is an unfinished implementation based on PR #123 head `99b3b095366f7c8328de44d828e14c0dce6a571d`. The original review and all six findings are in [PR123_REVIEW.md](PR123_REVIEW.md).

Exact base: `99b3b095366f7c8328de44d828e14c0dce6a571d`.

**Current state**

Rust compiles across the workspace, including all targets with the existing Metal feature. The complete application witness and the extracted Poseidon2 witness lemmas compile in Lean. The full Lean package is not closed. The package artifacts and verifier pins have not been regenerated. In particular, Rust now expects shared-verifier manifest version 3, while the saved manifest is still version 2. End-to-end assembly tests therefore cannot be reported green on this handoff.

No new Rust feature, environment variable, protocol hash family, production profile, or cryptographic assumption was added. The profile remains Goldilocks, `b = 2`, `k_rho = 16`, `B = 65536`, with Poseidon2 binding. Protected owner files were not edited. No subagents were used.

The user explicitly chose **removal**, rather than porting, for the obsolete WASM frontend and constraint-minimizer bridge that depend on `neo-fold-legacy`.

| Finding | Implementation in this handoff | Remaining work |
|---|---|---|
| F1: old emitter | The normal emitter dispatches to the wide builder. Old emitter commands, streaming implementations, prefix loader, First54 expander, and old prefix artifacts are removed. | Validate the complete emitter after F3 closes; regenerate the selected package and exact conformance evidence. |
| F2: deprecated dependency | Golden comparison and encoding moved into maintained `nightstream` tests. The legacy crate, obsolete WASM frontend, minimizer bridge, and GPU adapters are removed. Workspace members and the root lockfile are updated. | Retire remaining old replay-script consumers, update associated tests/docs and the minimizer lockfile. |
| F3: two application layouts | Runtime recognition and reverse conversion are removed. Lean uses the general application layout. The manifest has one reference. A full physical application witness is proved. | Finish exact physical-row custody, rebuild all Lean targets and audits, regenerate artifacts, update pins only after conformance checks, and rerun assembly/public-flow checks. |
| F4: two witness executors | The row-skipping executor and identity-based routing are removed. Cached application values feed the same full physical executor and logical transport. | Repeat public-flow validation after the F3 artifact change. |
| F5: wrong mutation coordinate | The test derives its target from the exported retained application-local port and checks its interval. The recomputed-commitment rejection remains. | Repeat after artifact regeneration. |
| F6: missing CI checks | CI runs sampler parity, matrix-program tests, and assignment-transport tests. | Preserve the owner's local-only Lean decision. |

**Start with the open Lean obligation**

The latest focused build of `NightstreamFPrime.Export.Stage1.Wide.ApplicationRelocation` fails at lines 66 and 72: `rfl` does not close `instruction_map` or `assertion_map`. After the row-bound guard was added, the simplifier leaves `PhysicalRelabel.Map.combination` calls for the unfolded mapping; `combination_map` is reported unused. Resolve these correspondence lemmas first. The log is `/tmp/nightstream-pr123-review/handoff-relocation-lean.log`; it exited with code 1 after 2 seconds. An earlier build before the guard change passed, but that pass does not apply to this handoff.

`Export/Stage1/Wide/PhysicalSourceCustody.lean`, theorem `custody`, still contains this obsolete application-row case:

```lean
  · intro absent
    cases absent
```

It was valid only because the selected compact application's ordinary-row premise was impossible. That premise is now removed. Replace this case with a proof of the actual application rows. Do not add a source-custody assumption or weaken the theorem.

The implementation direction is to compile the application once at its canonical reference coordinates, then relocate its exact row and witness syntax into the wide physical archive. This avoids having to prove that an arbitrary `Application.Program` independently compiled at two offsets has identical syntax; the current `Program` contract does not promise that property.

The new `Export/Stage1/Wide/ApplicationRelocation.lean` defines the column/row map, mapped application plan, row-map injectivity, and instruction/assertion correspondence. `Wide/ApplicationPackage.plan` now uses it. `Wide/ApplicationPackageCounts.lean` forwards its preserved counts. These changes still need to be connected to archive custody:

1. Add application-row pairs to `Wide/PhysicalSourceCustody.BuilderRows`. Use `ArchiveRows.decoded_pairs`, `ApplicationRelocation.instruction_map` and `assertion_map`, and the canonical application row bounds. The correspondence lemmas require that the source row lies after the reference application start.
2. Replace `ArchiveRows.application_bounds`. It still refers to the removed `ApplicationPackage.columns` and the old independent lowering. Derive bounds from the mapped canonical row list.
3. Prove application row recovery through `PhysicalMatrixSource.inverse`. Shared input/output columns are in the unchanged prefix. Application-private columns move from reference start `28784740` to wide start `27859260`. The inverse's final range already includes the application width.
4. Add an application-source lemma beside `base_source` and `next_source`, then discharge `custody.applicationRows`. `PerApplicationPackageSourceRows.applicationPlan_decodedRows_perm`, `applicationRows_rowIndices`, and `ArchiveRows.mapped_single` are useful existing facts.
5. Register any new exported theorem in the axiom audit. Run the focused targets before the complete build.

`ApplicationRelocation.mapping.row` rejects source rows before the application interval, which makes it injective. Its pure plan is built only from canonical compiled rows. Keep the proof structural; do not evaluate thousands of concrete rows in the kernel or raise recursion/heartbeat limits.

**Completed Lean changes to preserve**

- `Lifecycle/Stage1/Application.Program` no longer carries `compactHashChain` or its certificate type.
- The duplicate `ApplicationRetainedGeometry` and `ApplicationSelectedBlocks` are removed. Their users now use `ApplicationOrdinaryGeometry` and `ApplicationRetainedBlocks`.
- `ApplicationDirectPlan`, `ApplicationMatrixProgram`, and their soundness/interpretation proofs have one application case.
- Obsolete `ApplicationPoseidon*`, compact hash-chain layout/witness, and constant-prefix specialization modules are removed.
- `Gadgets/Poseidon2/Formal` exports a computable `witness`, `witness_agreesOutside`, and `witness_complete`. The old existential completeness theorem delegates to them, with the same premises.
- `Export/Stage1/ApplicationWitness.lean` constructs every physical application local, proves source agreement, and transports all rows through `ApplicationOrdinaryPlan.rowsZero_iff_rowsHold`. This target passed.
- `ApplicationWitnessCompleteness` and `Wide/ApplicationCompletedAssignment` now pass the full target environment to `privateSuffix`.
- `SharedVerifier.lean` emits one child program per phase, manifest version 3, and a top-level `application_local_index`. There is no `selected_reference` or paired relocation state.

Expected one-link geometry from the general layout is 7,696 application locals, 7,700 application rows, 137,646,804 logical coordinates, 137,646,810 ring-padded coordinates, and 3,256,394 logical rows. The whole-circuit coordinate increase is 304,958, about 0.222%. Treat these as intended values until the complete source and emitted package pass their gates.

**Remaining deprecated tooling**

The new golden runner is the ignored test:

```text
lifecycle::tests::staged::fold::lean::golden::native_checker
```

It accepts JSON on stdin: either `operation: compare` with `package`, `directory`, and `lean` paths, or `operation: encode` with `lean` and `output` paths. The comparison reads `proof.native`, checks all saved C/R/D fields, independently encodes every native proof byte from Lean numeric fields, replays PiDEC with the maintained verifier, and retains all 55 rejection cases.

`scripts/golden_conformance_ci.py` now builds the maintained library test binary once. `crates/nightstream/tests/check_lean_fold.py` uses that binary for both operations. No new production API or Cargo feature was added for this test tool.

The following cleanup remains necessary before F2 is closed:

- `formal/nightstream-fprime/scripts/replay_recursive_loop.py` still invokes the removed `generate_pi_ccs_fixture` legacy binary. Its old independent-production mode and callers cannot remain advertised as working. Preserve the current fresh Lean comparisons and independent raw-row checks when retiring this obsolete consumer.
- Inspect `scripts/check_selected_replay.py`, its tests, `scripts/lean_graph/tests/test_conformance_registrations.py`, `scripts/tests/test_golden_conformance_changes.py`, and current golden reproduction instructions. Some still name legacy paths or expect the old replay workflow.
- Refresh `tools/recursive-constraint-minimizer/Cargo.lock` after removing its bridge workspace member. Keep the independent minimizer itself and its current wide-sampler experiments.
- Inspect newly unused GPU adapter-specific shader or selector surface. The maintained Metal code compiles; the old adapter API and guarded methods are gone.
- Finish current documentation updates. Historical evidence may describe old paths, but current commands must not depend on removed code.

**Artifacts, tests, and review receipts**

Do not copy a new artifact identity into the verifier pins as the first step. The owner contract requires exact expanded-matrix comparison, independent raw-assignment evaluation, and applicable nonzero parity checks before repinning. Then regenerate the canonical package, shared-verifier manifest, bindings, application references, and affected lifecycle fixtures together. Check both copies of selected artifacts under `formal/nightstream-fprime/artifacts` and `crates/nightstream/artifacts`.

The Rust application-plan materializer now lives only in `tests/assembly_internal/materialize.rs`. Preserve its exact row/recipe comparison and the complete assembled-value comparison. Generic applications and the selected one must use the same production assembly code.

Independent review acceptances were already pending at the reviewed PR head. Implementation work does not authorize the author to sign them. Updated source and artifact cuts need updated requests and independent acceptance records. Do not treat old accepted receipts as approval of this handoff.

**Evidence obtained**

These checks passed at the stated phase. Earlier component passes are not a claim that the final unfinished source/artifact combination passes all gates.

| Check | Result |
|---|---|
| Final `cargo check --workspace --all-targets --release --features nightstream/metal` | Passed after legacy removal and Rust formatting |
| Final golden coordinator Python tests | 10 passed |
| Final Lean-fold driver Python tests | 6 passed |
| Final `validate.sh static` | Passed; all boundary checks |
| Current package malformed-row and retired-envelope tests, before F3 | 4 passed; 15.21 s |
| Full package library unit suite, before F3 | 81 passed, 7 ignored; 53.44 s |
| Wide sampler parity | 3 passed |
| Matrix-program tests | 23 passed, 1 ignored |
| Assignment-transport tests | 8 passed |
| Compact product execution/rejection tests | 3 passed |
| Corrected application mutation test, before F3 | Passed on CPU; 221.02 s |
| Public two-link recursive flow after executor consolidation, before F3 | Passed on Metal; 57.70 s |
| Maintained golden checker regression, before F3 artifact regeneration | Passed; all 55 PiDEC mutations, all 17 result fields, complete wire comparison; 14.90 s |
| Simplified Lean writer `NightstreamFPrime.Export.Main` | Passed; 250 s |
| Extracted Poseidon2 witness lemmas | Passed; 2 s incremental |
| Complete `NightstreamFPrime.Export.Stage1.ApplicationWitness` | Passed; 4 s incremental |
| Latest `NightstreamFPrime.Export.Stage1.Wide.ApplicationRelocation` | Failed: `rfl` at lines 66 and 72 in the guarded mapping correspondence lemmas |
| Full Lean build, regenerated artifacts, and final end-to-end conformance | Incomplete; do not treat the component passes as acceptance |

Detailed local logs are under `/tmp/nightstream-pr123-review/`. This handoff contains the durable conclusions; those temporary files are not required to understand the open obligations.

**Validation discipline**

Use `cargo fmt --all` after Rust edits and release tests with a hard cap of 300 seconds. Use `FoldingMode::Optimized`. Run one Lean or Rust build at a time. Lean commands go through `formal/nightstream-fprime/scripts/validate.sh`, whose per-command cap is at most 1,500 seconds. It now uses `timeout --signal=KILL` and rejects zero/no-timeout mode. `scripts/check_lean.sh` no longer adds a redundant outer timeout that can orphan an inner process group. Other Python coordinators with an outer timeout around `validate.sh` still need their process-group cleanup reviewed when that path is exercised.

Read the September 4 SuperNeo v1.2 text under `docs/superneo-paper-v1_2` and the HyperNova sections cited in the review. Preserve the separate Pad and matrix evaluation families, verifier-driven transcript, fixed parameter profile, and the existing key policy. No unresolved implementation correspondence may be replaced with a cryptographic assumption.
