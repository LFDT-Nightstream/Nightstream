# neo-fold-next proof-completeness review

Date: 2026-04-25

Scope: current dirty working tree under `crates/neo-fold-next`, with paper context from:

- `docs/superneo-paper/05_5_Embedding_products_with_evaluation_homomorphism.md`
- `docs/superneo-paper/07_7_Neo_s_folding_scheme_for_CCS.md`
- `docs/superneo-paper/12_C_Additional_Background.md`
- `docs/superneo-paper/13_D_Deferred_theorems_and_proofs.md`
- `docs/hypernova-paper/13_6_2_NIVC_Compatible_multi_folding_schemes.md`
- `docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md`

This is a theorem-boundary review, not a style review. The question is whether the current `neo-fold-next` RV32IM/SuperNeo stack can honestly claim to be paper-faithful, proof-complete, and sound.

## Revision note

This file supersedes my first pass from the same date. The second pass reconciles the supplied external review with direct paper/code inspection. The main corrections are:

- I no longer classify the base-step `u_perp` issue as a P1 soundness break. HyperNova Construction 2 initializes the running folded list to the default pair in the base case; it does not require the base current committed instance `u_i` to be verifier-checked as canonical before the first step. The Rust native path locally enforces a stronger base-state contract than the recursive-step circuit. That is a contract/auditability mismatch, not a paper violation by itself.
- I no longer classify the folded-accumulator debug-only recomputation as a proven release-mode proof break. The normal F' evaluator computes the digest immediately from `next_state.carry`, and `Rv32imIvcState::validated_surface` also recomputes it. The remaining issue is a defensive invariant check being release-disabled at one bridge point.
- I agree that the external recursion VK parameter is not, by itself, a soundness break: the SNARK verification still checks the proof against a public image that includes `vk_fs_digest`. The issue is verifier lifecycle/deployment fragility because `rv32im_verifier_context_digest` does not pin the supplied VK/shape identity.
- I agree that the SuperNeo Lean formalization is paper-faithful and proof-complete for the tracked §4-§7 milestone route per the maintained README. The stricter gap is Rust-to-Lean / Nightstream RV32IM artifact closure, not the SuperNeo math formalization.
- I added the reduction-level PaperExact cross-check evidence. The remaining coverage gap is narrower: no active `neo-fold-next` published-path Optimized-vs-PaperExact guard.

## Executive verdict

| Dimension | Status | Verdict |
|---|---|---|
| SuperNeo math formalization | Strong | `formal/superneo-lean` reports every tracked milestone as `Done (Proof-Complete)`, with only theorem-level assumptions left explicit. |
| Generic Rust SuperNeo spine | Strong but not self-documenting enough | `Pi_CCS -> Pi_RLC -> Pi_DEC` is structurally aligned and verifier-recomputed. Some paper inequalities are enforced centrally or implicitly and should be made easier to audit at the boundary. |
| RV32IM compressed IVC path | Plausible and much stronger than stale reports | Final CE, terminal F', and Construction-2 opening are now present. The remaining gap is an explicit bridge contract tying those checks to HyperNova Construction 2 verifier semantics for the single-lane specialization. |
| Published Nightstream verifier | Soundly anchored with one deployment seam | The verifier recomputes authoritative digests and verifies the main SNARK against a recomputed public image. It should still pin VK/shape digests in the statement context. |
| Rust-to-Lean proof completeness | Not closed | The SuperNeo math is closed, but the active RV32IM Rust artifact is still too summary-shaped for Lean to reconstruct every theorem-bearing boundary. The archived Rust-refinement subtree is not maintained in the default build. |
| Test/redteam coverage | Good on cheap structural tamper, weak on compressed theorem seams | There are many active redteam and F' conformance tests, plus reduction-level PaperExact checks. Expensive Spartan/compressed-boundary and side-soundness tests remain ignored. |
| Auditability | Needs work | Several protocol-critical files exceed the 1,500-line project rule. |

Short answer: I do not see a confirmed structural soundness break in the active published RV32IM/Nightstream verifier path from this review. I also would not claim strict "proof-complete Rust implementation" yet. The honest statement is: SuperNeo's formal math route is proof-complete; Rust is structurally faithful and soundly anchored by inspection/tests; the active RV32IM publication path still needs bridge specs/tests and a maintained Rust-to-Lean artifact story before the whole implementation can be called proof-complete.

## Paper obligations

### SuperNeo CCS folding

From SuperNeo §7 and Appendix D, the implementation must preserve:

- CCS instances: `c = L(z)`, bounded `z`, and zero-sum CCS products over `M_j z`.
- CE instances: `c = L(z)`, `x = L_in(z)`, bounded `z`, and evaluation claims `y_j = M_j z(r)`.
- `Pi_CCS`: transcript-derived challenges reduce CCS claims to CE claims. Sumcheck soundness in Appendix D uses a per-round degree bound of the form `max(u, 2b + 1, 2)` in the text.
- `Pi_RLC`: ring challenges are sampled from a strong set. Definition 14 requires `(K + k)T(b - 1) < B` and `(2B, C)` relaxed binding.
- `Pi_DEC`: decomposition/recomposition keeps children inside the short-norm bound.
- Digest chains are never authority. The verifier must recompute public transcript challenges and verify the proof obligations summarized by any digest.

### Evaluation-homomorphism embedding

From SuperNeo §5, the Rust algebra must preserve:

- coefficient embedding into the selected ring/module;
- `ct(Mbar z) = Mz`;
- homomorphic recombination of commitments, public inputs, and evaluation claims.

This is why mixed hashing or unchecked digest compression in theorem-facing paths would be dangerous. The reviewed `neo-fold-next` proof/transcript/public digest paths are Poseidon2-oriented.

### HyperNova Construction 2

From HyperNova §6.2/§6.3:

- The multi-folding scheme must have default `R1` instances for the base case.
- F' takes `(vk_fs, U_i, u_i, pc_i, (i, z_0, z_i), omega_i, pi)`.
- For `i = 0`, F' checks `z_0 = z_i` and initializes `U_{i+1}` to defaults.
- For `i > 0`, F' checks the current instance hash, `pc_i`, and runs the NIFS verifier update.
- The final verifier checks the public image hash, `1 <= pc_i <= ell`, all `R1` folded running instances, and the active `R2` committed relation.

The current Rust implementation is a single-lane/single-PC specialization of that compiler pattern. Claims should say "HyperNova Construction-2-style single-lane RV32IM IVC over the SuperNeo folding spine", not general multi-function NIVC.

## What is strong

### Generic SuperNeo chunk verification

`crates/neo-fold-next/src/verifier.rs` follows the paper spine:

- chunk artifacts and transcript state are validated before proof checks;
- `Pi_CCS` verification runs through optimized or reference verification;
- replayed header digest and output fold digests are checked;
- `Pi_RLC` challenges are transcript-derived and public recomposition is checked;
- `Pi_DEC` recomposition is checked.

`crates/neo-fold-next/src/finalize.rs` recomputes final statement/proof digests and verifies chunks against public final claims. This is the right digest-as-compression model.

### Reduction-level paper/optimized cross-checking exists

The external review was right to push on PaperExact coverage, but the precise state is nuanced:

- `crates/neo-reductions/tests/optimized_oracle_me_outputs_match_paper_exact.rs` checks optimized ME outputs against the paper-exact builder.
- `crates/neo-reductions/tests/optimized_oracle_row_stream_smoke.rs`, `claimed_initial_sum_compat.rs`, `k_mcs_parity.rs`, and `dec_reduction_y_zcol.rs` include additional optimized-vs-paper-exact checks.
- `crates/neo-reductions/tests/rlc_dec_k_gt1.rs` has explicit `FoldingMode::Optimized` vs `FoldingMode::PaperExact` parity tests for RLC and DEC.

The remaining gap is not "no PaperExact checks anywhere." It is "no active `neo-fold-next` published-path cross-engine guard that exercises the SuperNeo folding path as consumed by the RV32IM/Nightstream proof."

### Compressed RV32IM verifier is no longer terminal replay

The current `ivc_snark` boundary includes:

- final main claims;
- final CE proof;
- Construction-2 opening proof;
- terminal recursive-step Spartan proof;
- verification against a recomputed `Rv32imIvcPublicImage`.

`crates/neo-fold-next/src/rv32im/main_proof.rs:569-592` rebuilds the public image from the published statement, including `vk_fs_digest`, `chunk_count`, terminal handle, current `x_i`, Construction-2 `u_i`, folded accumulator digest, and terminal statement digests. `crates/neo-fold-next/src/nightstream/rv32im/verify_perf.rs:81-86` verifies the main SNARK against that recomputed image.

This resolves the stale critique that the compressed path is merely a terminal native replay fallback.

### Nightstream digest boundary is soundly shaped

The top-level verifier path:

- recomputes `verifier_context_digest`;
- recomputes `public_statement.digest`;
- compares statement public IO to the carried published main statement;
- verifies the carried boundary;
- verifies side proofs through statement/runtime surfaces;
- verifies the main IVC SNARK against a recomputed public image.

This is not a self-consistent digest chain. It is a recompute/verify boundary. The remaining issue is that the verifier context is too small: it binds root params, but not VK/shape identity.

### SuperNeo Lean status is strong

`formal/superneo-lean/README.md` states:

- every tracked milestone row is `Done (Proof-Complete)`;
- the active native Goldilocks / `paperCarrier` difference route is proof-complete through `S7.6`;
- archived Rust-vector / Rust-refinement machinery is outside the maintained default build path.

I did not rerun `lake build` in this correction pass. The maintained documentation is nevertheless clear: the SuperNeo mathematical formalization is not the weak point. The weak point is connecting the active Rust/Nightstream RV32IM artifact boundary to theorem-bearing Lean inputs.

## Findings

### Proof-completeness blocker: active RV32IM Rust artifacts are not Lean-catch capable

This is the biggest strict "proof-complete" gap, but it is not a direct Rust verifier soundness bug.

`formal/nightstream-lean/README.md` says RV32IM theorem modules prove consequences above `ExactKernelBoundaries`, but do not yet prove that the current exported Rust RV32IM public-proof artifact carries enough data to construct those boundaries. The same README says the accepted artifact is still summary-shaped and lists missing theorem-bearing exports such as row-local root encodings, row-local CCS acceptance objects, selected-row payload/provenance chains, and stage obligation payloads.

Implication:

- SuperNeo §4-§7 math can be proof-complete.
- Rust can be structurally faithful by inspection.
- The combined Rust RV32IM publication path is not yet proof-complete in the strict "Lean would catch a missing binding edge in the accepted artifact" sense.

Closure requirement:

- Define a stable Rust artifact/witness-export ABI for the theorem-bearing RV32IM side/kernel boundaries.
- Make `nightstream-lean` construct `ExactKernelBoundaries` from accepted Rust artifacts, or fail constructively when an artifact omits a required binding.

### P1: verifier context does not pin VK/shape identity

`crates/neo-fold-next/src/nightstream/rv32im.rs:824-831` computes `rv32im_verifier_context_digest` from only `root_params_id`.

`crates/neo-fold-next/src/nightstream/rv32im/verify_perf.rs:33-41` receives the recursion, side-opening, and side-binding verifier keys out-of-band. The main proof is then verified at `verify_perf.rs:81-86` against a public image rebuilt from the published statement.

This is not the critical soundness failure I first implied. A wrong recursion VK should fail to verify the SNARK proof against the recomputed public image. The fragility is lifecycle and deployment:

- an on-chain/public verifier needs a pinned notion of the accepted VKs/shapes;
- the statement context currently cannot reject a valid proof under an unintended trusted setup before trying proof verification;
- side VK identities are also external to the context digest.

Recommended fix:

- Bind `(root_params_id, recursion_vk_digest, side_opening_vk_digest, side_binding_vk_digest, shape/profile digest, version)` into `rv32im_verifier_context_digest`.
- Reject supplied VKs whose digests do not match the statement context.
- Keep witness-derived setup helpers as prover convenience only, not as the public verifier lifecycle.

### P1: CE projection equality is too implicit for a carried-state boundary

`crates/neo-fold-next/src/rv32im/chunk_step_ivc.rs:478-489` compares CE claims with a custom projection:

- compares commitment data, `m_in`, `r`, and `y_ring`;
- compares only dimensions of `X`;
- samples `X[(col % rows, col)]` for `col in 0..m_in`;
- ignores fields such as `s_col`, `ct`, `aux_openings`, `y_zcol`, `fold_digest`, `c_step_coords`, `u_offset`, and `u_len`.

That projection feeds `rv32im_chunk_step_ivc_states_match` at `chunk_step_ivc.rs:491-496`. The broader accumulator code also carries explicit projection digests, for example `Rv32imChunkFoldCarry::main_projection_digests` in `chunk_fold_step.rs:23-67`, and final accumulator digests are built from projection digests in `final_relation.rs:133-167`.

This may be intentional: some full `CeClaim` fields may be transport or proof-local cargo rather than authoritative recursive state. The problem is that the type at the comparison site is still a full `CeClaim`, while the equality relation is a narrow projection.

Recommended fix:

- Introduce a named projection type, such as `Rv32imCarriedCeProjection`, with exactly the authoritative fields.
- Use that type in carried-state equality and digest code.
- Add tests that mutate each ignored field and each non-sampled `X` cell. If mutation should be accepted, the test name should say the field is non-authoritative.

### P1: compressed theorem-boundary tests remain ignored

There is useful cheap coverage, including `rv32im_compressed_public_boundary`. But the strongest tests that actually exercise Spartan/compressed proof seams are ignored:

- `crates/neo-fold-next/tests/rv32im_spartan2_decider.rs` ignores debug/setup/round-trip/public-image tamper/coherent terminal metadata tamper/direct SNARK boundary tamper/unrelated-chain proof-swap tests.
- `crates/neo-fold-next/tests/spartan2_backend_binding_shell.rs` ignores VK/binding-shell style checks behind the stale "native NIFS and F' replacement" reason.
- `crates/neo-fold-next/tests/side_soundness/positive.rs` ignores many side-soundness tests with the same stale reason.

The always-on tests catch structural boundary tamper. They do not replace at least one live full compressed proof round-trip or VK-swap rejection lane.

Recommended fix:

- Add one smallest-viable always-on compressed verifier smoke test if it can stay under the 60-second rule.
- If it cannot, document a named manual/CI theorem-boundary lane and treat it as required evidence for release/audit closure.
- Re-enable or replace the stale parked tests whose reason no longer matches the current architecture.

### P1 coverage: no active neo-fold-next PaperExact-vs-Optimized published-path guard

Reduction-level paper-exact tests exist, but `neo-fold-next` tests overwhelmingly use `FoldingMode::Optimized`, consistent with the project testing rule. That is good for performance, but it leaves no active end-to-end guard that the optimized folding path consumed by `neo-fold-next` still matches the paper-exact construction on a tiny theorem-shaped instance.

Recommended fix:

- Add a tiny feature-gated cross-engine test outside normal fast CI, or under an explicit `paper-exact`/manual lane, that compares Optimized and PaperExact for the `neo-fold-next` chunk relation surface.
- Keep normal tests on `FoldingMode::Optimized`; do not violate the project rule casually.

### P2: base `u_perp` is a native-vs-circuit contract mismatch, not a confirmed paper break

Native code enforces a canonical base Construction-2 default:

- `crates/neo-fold-next/src/rv32im/construction2.rs:1162-1187` checks base `current_input_fresh_instance == expected_default`.
- `crates/neo-fold-next/src/rv32im/ivc.rs:641-660` checks the native base state carries the canonical Construction-2 default pair.

The recursive-step circuit is weaker at the base current-input side:

- `crates/neo-fold-next/src/rv32im/main_relation_spartan/recursive_step.rs:987-993` gates initial transcript/base handling.
- `recursive_step.rs:1163-1169` checks current input `x_i` only for non-base steps.
- `recursive_step.rs:1225-1238` binds the output Construction-2 public boundary to `x_out`.

After re-reading HyperNova §6.3, this should be downgraded. The paper base case initializes the running `U` list from defaults; it does not require a current committed `u_i` base input to be verifier-constrained as canonical in the same way as non-base steps.

Still, local code should not have two different contracts.

Recommended fix:

- Decide whether the local Rust theorem wants the stronger native base-current-input invariant.
- If yes, add a base-case circuit gate for the canonical current `u_perp`.
- If no, reword/remove the native test expectation so native F' and circuit F' state the same base relation.

### P2: folded-accumulator digest recomputation is debug-only at one bridge point

`crates/neo-fold-next/src/rv32im/f_prime.rs:330-365` applies a verified step image. The check that `output.folded_accumulator_digest` equals `rv32im_chunk_fold_carry_recursive_accumulator_digest(&output.next_state.carry)` is behind `#[cfg(debug_assertions)]` at `f_prime.rs:340-349`.

This is less severe than my first pass said:

- the honest F' evaluator computes the digest directly from `next_state.carry` at `f_prime.rs:1652-1665`;
- `Rv32imIvcState::validated_surface` recomputes the carried digest at `ivc.rs:669-676`.

The remaining issue is audit discipline: theorem-facing bridge invariants should not depend on debug-only checks, especially in a project that runs proof tests in `--release`.

Recommended fix:

- Make this check unconditional unless profiling shows a real cost.
- Add a release-mode unit test around the bridge API if a tampered step-image constructor is reachable in tests.

### P2: verifier-profile constants are not in `vk_fs.expected_digest`

`crates/neo-fold-next/src/rv32im/f_prime.rs:57-62` defines theorem-profile constants: trivial PC, accumulator slot count, side witness active, phi side active, and derived side lane active.

`Rv32imVerifierKeyFs::expected_digest` at `f_prime.rs:212-228` hashes the domain tag digest, main-lane shape digest, and step cap, but not those profile constants.

Some constants are enforced elsewhere, so this is not an immediate exploit. But if a future rebuild changes a theorem-profile constant while leaving shape/step-cap digest unchanged, the public `vk_fs_digest` would not distinguish two different local theorem profiles.

Recommended fix:

- Hash a canonical profile digest into `expected_digest`.
- Add a regression test that changing any profile constant changes `vk_fs.expected_digest`.

### P2: digest-to-field packing assumes canonical limbs at every ingress

`crates/neo-fold-next/src/finalize.rs:60-67` maps each 8-byte digest limb with `F::from_u64`. In Goldilocks, that conversion is non-injective for `u64 >= p`.

Some important paths already check canonical digest limbs:

- `crates/neo-fold-next/src/rv32im/encoded_public_input.rs:6-15`;
- `crates/neo-fold-next/src/rv32im/ivc.rs:133-156`;
- `crates/neo-fold-next/src/rv32im/ivc_snark/construction2_opening.rs:802-803` and `892-893`.

The remaining requirement is global auditability: every external/public ingress that uses `digest32_as_fields` must either prove the bytes came from canonical Poseidon2 field limbs or reject non-canonical limbs before conversion.

Recommended fix:

- Add `digest32_as_fields_checked` for public ingress paths.
- Add a regression test using two byte strings that differ by the Goldilocks modulus in one limb and must not be accepted as the same digest at a public boundary.

### P2: side-opening standalone validator is structural only despite taking a statement

`crates/neo-fold-next/src/nightstream/rv32im/authoritative_side.rs:614-626` accepts a `NightstreamStatement` and discards it.

The top-level path is stronger:

- `crates/neo-fold-next/src/nightstream/rv32im.rs:906-923` calls structural validation, runtime-surface binding, and SNARK verification.
- `crates/neo-fold-next/src/nightstream/rv32im/side_runtime_binding.rs:14-47` recomputes `nightstream_statement.core_digest()`.
- `authoritative_side.rs:445-461` builds side binding statements with `nightstream_statement.core_digest()`.

So this is not a top-level verifier break. It is an API sharp edge: the standalone function looks statement-aware but is only structural.

Recommended fix:

- Rename it to make the structural-only contract explicit, or make it actually bind to statement-derived surfaces.
- Add statement-substitution tests at the top-level side verifier boundary.

### P2: SuperNeo numeric guards are present but hidden from the consumer layer

The external review called out missing visible guards for paper inequalities. The code is better than that, but the audit trail is still thin.

Evidence:

- `crates/neo-params/src/lib.rs:124-128` enforces `(k_rho + 1)T(b - 1) < B`, matching the current single-MCS/specialized parameter route.
- `crates/neo-reductions/src/engines/utils.rs:97-101` computes `d_sc = max(s.max_degree() + 1, degree_bound_nc(params))` and runs the extension-field policy check.
- `crates/neo-reductions/src/sumcheck.rs:356-365` rejects round polynomials whose length exceeds the degree bound.

Qualification:

- SuperNeo Definition 14 states `(K + k)T(b - 1) < B`; the Rust param guard is specialized as `(k_rho + 1)T(b - 1) < B`.
- Appendix D's degree text uses `max(u, 2b + 1, 2)`, while `degree_bound_nc` documents/uses `max(2, 2b)` under the split-NC variant. That may be correct for the implemented polynomial layout, but the paper-to-code mapping deserves an explicit comment or assertion.

Recommended fix:

- Add a small theorem-facing parameter check/comment near the `neo-fold-next` chunk-relation entry point saying which specialization sets `K = 1` and why the centralized `NeoParams` guard suffices.
- Add a comment or assertion documenting the split-NC degree mapping from the paper's bound to `d_sc`.

### P2: stale ignored Spartan-path tests cover theorem seams

Several tests remain parked with old architecture reasons:

- `rv32im_main_relation_ce_consistency.rs`;
- `rv32im_main_relation_pi_ccs_claims.rs`;
- `rv32im_main_relation_ce_spartan.rs`;
- `rv32im_main_relation_terminal_identity.rs`;
- `rv32im_main_relation_initial_sum.rs`;
- many `rv32im_main_recursion_step_spartan.rs` tests.

These are not all required in fast CI, but stale ignored tests at theorem seams reduce confidence. Each should either be re-enabled, replaced by a live equivalent, or explicitly marked as a manual expensive lane with a current reason.

### P2: theorem statement is single-lane, not general NIVC

The implementation explicitly fixes:

- `RV32IM_MAIN_RECURSION_TRIVIAL_PC = 1`;
- `RV32IM_MAIN_RECURSION_ACCUMULATOR_SLOTS = 1`;
- verifier-side checks that the public image PC equals the trivial lane.

That is fine if documented. It should not be described as full HyperNova multi-function NIVC.

### P2 hygiene: file size exceeds the project rule

Files over the 1,500-line rule in `crates/neo-fold-next` include:

- `src/rv32im/main_relation_spartan/recursive_step/diagnostics.rs`: 4,013 lines
- `src/bin/rv32im_main_recursion_shape_probe.rs`: 2,434 lines
- `src/rv32im/main_relation_circuit/claim.rs`: 2,381 lines
- `src/rv32im/construction2.rs`: 2,326 lines
- `src/rv32im/main_relation_spartan/chunk_diagnostics.rs`: 1,822 lines
- `src/rv32im/f_prime.rs`: 1,729 lines
- `src/rv32im/main_relation_spartan/chunk_step_recursive.rs`: 1,717 lines
- `src/rv32im/kernel/simple.rs`: 1,681 lines
- `src/decider/spartan2.rs`: 1,618 lines
- `src/rv32im/main_relation_circuit/pi_rlc.rs`: 1,576 lines
- `tests/validate_issues/kernel_progress.rs`: 1,516 lines
- `src/rv32im/kernel/openings/claim_reduction.rs`: 1,510 lines

This is not a soundness bug, but it is an auditability problem. Protocol-critical files should be split by ownership boundary when touched, not with churn-only helper extraction.

## Consolidated recommendation order

1. Pin VK/shape/profile digests into `rv32im_verifier_context_digest` and reject mismatched supplied VKs.
2. Add a theorem-facing compressed Construction-2 boundary spec mapping paper symbols to `Rv32imIvcPublicImage`, final CE, terminal F', and Construction-2 opening fields.
3. Replace the implicit CE projection matcher with a named projection type and mutation tests.
4. Add one live compressed proof-boundary smoke test or document a mandatory manual lane if it cannot meet the 60-second rule.
5. Add a tiny `neo-fold-next` published-path Optimized-vs-PaperExact guard under an explicit feature/manual lane.
6. Resolve the base `u_perp` contract mismatch by either enforcing the stronger local invariant in-circuit or narrowing the native contract to paper semantics.
7. Make the folded-accumulator digest check unconditional.
8. Hash verifier-profile constants into `vk_fs.expected_digest`.
9. Audit `digest32_as_fields` public ingress paths and add checked conversion where needed.
10. Update/re-enable stale ignored Spartan/side theorem-seam tests.
11. Start splitting >1,500-line protocol-critical files only when a nearby change gives a natural ownership boundary.
12. Plan the RV32IM Rust artifact -> Lean `ExactKernelBoundaries` witness-export ABI.

## Bottom line

The external review was right that the codebase is stronger than my first severity framing implied. I would not describe `neo-fold-next` as structurally broken. The generic SuperNeo spine is paper-aligned, the SuperNeo Lean route is documented as proof-complete, and the Nightstream boundary follows the digest-as-compression discipline.

The accurate remaining critique is stricter and more useful: the active Rust RV32IM/Nightstream artifact is not yet proof-complete in the same sense as the SuperNeo Lean math, and a few public-boundary disciplines need tightening before an external audit can stop asking the same questions. The main concrete engineering fixes are VK/context pinning, named CE projections, live compressed-boundary coverage, one `neo-fold-next` cross-engine guard, and a formal/prose bridge contract for final CE + terminal F' + Construction-2 opening.

## Verification

Earlier in this review cycle I ran:

```bash
timeout 60s cargo test -p neo-fold-next --release --test rv32im_compressed_public_boundary -- --nocapture
```

Result: both tests passed:

- `rv32im_compressed_public_boundary_rejects_construction2_u_i_tamper`
- `rv32im_compressed_public_boundary_rejects_terminal_metadata_tamper`

For this correction pass, I did not rerun Cargo or Lean builds. I used direct paper/code/test/README inspection to correct the report. No Rust source code was changed.
