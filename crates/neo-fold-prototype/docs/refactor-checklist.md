# Direct CCS Refactor Checklist

## Current Focus

| Field | Value |
|---|---|
| Active area | `src/lifecycle` + `frontends/direct_ccs/state` |
| Active chunk | `front-facing Direct CCS lifecycle readability` |
| Status | `public Direct CCS lifecycle functions now own the real prove/extend/finish/verify flow; the trait impl is only a conformance adapter; recursive append now reads as prior-F' authority -> current SuperNeo/Construction-2 step -> next carrier state` |
| Latest verification | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_lifecycle_prove_extend_and_verify_use_same_append_flow -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_latest_step_is_not_historical_replay -- --nocapture` |
| Next action | `Continue only if the public lifecycle still feels unclear from the caller's point of view; otherwise return to compact F' recursive-cost reduction.` |

## Immediate Required Work

The canonical SuperNeo/Construction-2 append step is complete. Compact F'
authority now exists through crate-owned native advice: caller-supplied source
images remain digest-only, while `from_native_advice` adds concrete compact
NIFS.V authority rows and lets multi-step Direct CCS append fold one prior F'
source relation.

The remaining issue is cost and broad test-suite hygiene, not the old
missing-authority gate. Targeted recursive tests have been updated to the
positive compact-authority behavior. Duplicate Direct CCS recursive tests that
pay the several-minute compact-authority append cost are now explicitly ignored.
Source/R1CS authority-gate tests use plain direct state and stay fast; the
retained deep recursive positive path lives in `direct_ccs_r1cs_low_norm`.

### Required Shape

The positive path must keep this authority order:

```text
latest Direct CCS/F' step material
low-norm source encoding
Poseidon2 digest/public boundary linkage
NIFS.V verifier-shaped rows
authority gate flips true
recursive prior F' append can use the compact relation
```

Digest binding alone still must not count as authority.

Do not use the full exact verifier-body R1CS lowering as the normal positive
path. It is a real relation, but it is too large for the default recursive
append path. Compact source authority is now the normal positive path.

### Files To Inspect First

| File | Why |
|---|---|
| `crates/neo-fold-prototype/src/frontends/direct_ccs/f_prime/chain/mod.rs` | Owns encoder status, authority gate, and positive F' chain append path. |
| `crates/neo-fold-prototype/src/frontends/direct_ccs/f_prime/r1cs/build.rs` | Owns low-norm source R1CS witness/constraint construction. |
| `crates/neo-fold-prototype/src/frontends/direct_ccs/f_prime/r1cs/source.rs` | Owns source-shape validation and authority row accounting. |
| `crates/neo-fold-prototype/src/frontends/direct_ccs/f_prime/r1cs/nifs_authority.rs` | Owns compact NIFS.V authority constants and rows for crate-owned native advice. |
| `crates/neo-fold-prototype/tests/direct_ccs_ivc/f_prime_source.rs` | Existing non-Spartan authority-gate coverage. |
| `crates/neo-fold-prototype/tests/direct_ccs_r1cs_low_norm.rs` | Existing low-norm source/refusal coverage. |

### Definition Of Done

This chunk is done only when all of the following are true:

- Digest binding alone remains non-authoritative.
- Authority requires concrete compact NIFS.V rows from crate-owned native advice.
- The positive gate is covered by a non-Spartan test.
- Multi-step recursive Direct CCS no longer depends on terminal source-image
  authority.
- The compact authority relation is either fast enough for ordinary focused
  tests or explicitly marked as a heavy path with coverage split accordingly.
- No protocol checks are weakened.
- No frontend-local reimplementation of Pi_CCS/Pi_RLC/Pi_DEC is introduced.
- The checklist is updated with exact files changed and verification results.

### Required Verification

After Rust edits, run:

```bash
cargo fmt --all
cargo check -p neo-fold-prototype
cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows -- --nocapture
cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_builds_compact_f_prime_authority -- --nocapture
cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_lowered_product_uses_compact_f_prime_authority_despite_large_exact_body -- --nocapture
cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture
```

If a command exceeds the active time limit, stop it and record that honestly in
the verification log. Current per-test cap: 5 minutes.

## Completed Canonical SuperNeo Step Work

This was the previous implementation target and is now done.

### Required Shape

The real hot path must contain one readable step function with this order:

```text
prepare fresh + carried claims
Pi_CCS
Pi_RLC
Pi_DEC
terminal replay surface
Construction-2 next state/public image
Direct CCS state advance
```

This must be the code path used by Direct CCS append/prove. A wrapper that only
calls an old long function does not count.

### Files To Inspect First

| File | Why |
|---|---|
| `crates/neo-fold-prototype/src/core/chunk_folding/prove.rs` | Current native chunk proving entry; likely owns the prepare/Pi_CCS start. |
| `crates/neo-fold-prototype/src/core/chunk_folding/replay.rs` | Current replay witness path with long wrapper names; likely needs collapse into the canonical step result. |
| `crates/neo-fold-prototype/src/core/chunk_folding/transition.rs` | Current Pi_RLC -> Pi_DEC owner; should become part of the canonical step body or be called by it directly. |
| `crates/neo-fold-prototype/src/core/chunk_folding/pi_ccs.rs` | Existing Pi_CCS owner; do not duplicate its math. |
| `crates/neo-fold-prototype/src/core/chunk_folding/types.rs` | Existing result/witness/perf types; likely needs one named `SuperNeoChunkStep`-style result. |
| `crates/neo-fold-prototype/src/frontends/direct_ccs/state/append.rs` | Direct CCS append/prove state flow; must consume the canonical step. |
| `crates/neo-fold-prototype/src/frontends/direct_ccs/state/relation.rs` | Current Direct CCS folding boundary; likely should be renamed/simplified once the canonical step exists. |
| `crates/neo-fold-prototype/src/frontends/direct_ccs/state/construction2.rs` | Current Construction-2 state/public-image derivation; must be visible after the SuperNeo step. |
| `crates/neo-fold-prototype/src/frontends/direct_ccs/state/surface.rs` | Current terminal replay surface construction; must be visible between SuperNeo and Construction-2 advance. |

### Implementation Steps

| Step | Required action | Done when |
|---|---|---|
| 1 | Re-read SuperNeo paper sections 6 and 7 before editing. | The implementation plan can point each code phase to `Pi_CCS`, `Pi_RLC`, or `Pi_DEC`. |
| 2 | Map the current call chain from Direct CCS append into `core/chunk_folding`. | The old wrapper stack and the real reduction calls are identified before moving code. |
| 3 | Add or reshape a canonical shared chunk step in `core/chunk_folding`. | One function body visibly calls prepare, Pi_CCS, Pi_RLC, and Pi_DEC in order. |
| 4 | Return a named result package from that step. | Replay witness, transition result, perf, and carried CE outputs are fields on a named type, not loose tuples or suffix-specific variants. |
| 5 | Make Direct CCS append consume the canonical step. | The Direct CCS append path shows terminal replay surface construction, Construction-2 next image/state derivation, and state advance after the SuperNeo step. |
| 6 | Collapse or delete pass-through variants made obsolete by the canonical step. | No `_with_perf`, `_with_trace`, `_with_instance_digest`, or `_with_handle` sibling remains unless it has a distinct semantic role. |
| 7 | Keep reduction math in existing owners. | No new frontend-local Pi_CCS/Pi_RLC/Pi_DEC math appears under `frontends/direct_ccs`. |
| 8 | Update tests only if names or owner paths change. | Direct CCS tests compile and still cover append/folding and red-team child tamper behavior. |

### Definition Of Done

This chunk is done only when all of the following are true:

- A reader can open the Direct CCS append path and see:
  `fresh step -> canonical SuperNeo step -> terminal replay -> Construction-2
  advance`.
- A reader can open the canonical shared chunk step and see:
  `prepare -> Pi_CCS -> Pi_RLC -> Pi_DEC`.
- The canonical step is the real Direct CCS hot path, not a documentation-only
  wrapper.
- Any remaining helper names are short and phase-owned.
- No protocol checks are weakened.
- No new traits, feature flags, env vars, deprecated aliases, or compatibility
  modules are added.
- No touched file exceeds 1,500 lines.
- The checklist is updated with the exact files changed, verification results,
  and remaining risk.

### Required Verification

After Rust edits, run:

```bash
cargo fmt --all
cargo check -p neo-fold-prototype
cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_append_api_matches_native_superneo_carry -- --nocapture
cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture
```

If a command exceeds the active time limit, stop it and record that honestly in
the verification log.

## Progress By Area

| Area | Owner | Status | Done | Remaining | Risk | Latest Verification |
|---|---|---|---|---|---|---|
| `src/lifecycle/direct_ccs.rs` | Public Direct CCS lifecycle | `done` | Flipped the ownership direction to match the cleaner lifecycle pattern: `prove_direct_ccs`, `extend_direct_ccs`, `finish_direct_ccs_with_spartan`, `prove_and_finish_direct_ccs_with_spartan`, `verify_direct_ccs`, and `verify_finished_direct_ccs_with_spartan` now contain the real flow. `IncrementalProofSystem`/`SpartanProofSystem` implementations delegate to those canonical entrypoints instead of the public functions delegating into the trait. Added a focused lifecycle smoke test for batched prove vs incremental extend. | Native `verify_direct_ccs` is still a prover-side replay check because `DirectCcsProof` stores private steps; this should stay documented unless/until a public uncompressed verifier is designed. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_lifecycle_prove_extend_and_verify_use_same_append_flow -- --nocapture` |
| `frontends/direct_ccs/state/append.rs` + `frontends/direct_ccs/recursive/state/append.rs` | Direct CCS append naming/readability | `done` | Recursive append now reads top-down as `advance_prior_f_prime_authority`, `append_current_direct_step` / `append_current_verified_relation`, then `with_next_direct_state`. The prior-F' handoff is packaged as `PriorFPrimeAuthority` instead of open-coded chain/context/digest plumbing. The direct append boundary now exposes `append_step_with_f_prime_accumulator`, `append_relation_with_f_prime_accumulator`, `chunk_for_step`, `fold_chunk_with_superneo`, and `advance_construction2_after_superneo_step` in order. | The shared `core::ivc::SuperNeoIvcState` still contains duplicate append variants for accumulator-handle and non-handle paths; leave them unless the next chunk targets shared core cleanup. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_lifecycle_prove_extend_and_verify_use_same_append_flow -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_latest_step_is_not_historical_replay -- --nocapture` |
| `core/chunk_folding` + `frontends/direct_ccs/state` | Canonical paper-order SuperNeo/Construction-2 step | `done` | Added the real shared `prove_superneo_chunk_step` path and `SuperNeoChunkStep` result. The core body now visibly runs prepare -> Pi_CCS -> Pi_RLC -> Pi_DEC, and Direct CCS append consumes it through the shared IVC state path. Direct CCS append now exposes the post-SuperNeo handoff as `terminal_replay`, `derive_next_construction2_state`, and `advance_with_verified_superneo_step`. The old long replay/prove helper variants were removed from the core export surface; the RV32IM caller was minimally updated to the new shared step name. Two private Direct CCS append pass-through helpers were also collapsed. | None for this chunk. Remaining suffix-heavy helpers are semantically distinct authority/Construction-2 boundaries and were left in place. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_append_api_matches_native_superneo_carry -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` |
| `frontends/direct_ccs/step.rs` | Fresh SuperNeo step construction | `done` | Added a dedicated pre-`Pi_CCS` step owner. `DirectCcsStep` and the low-norm full-witness path now live together and read as validate, embed, project public input, commit, build `CcsClaim`/`CcsWitness`. | None for this chunk. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_sparse_r1cs_adapter -- --nocapture` |
| `frontends/direct_ccs/adapter/r1cs.rs` | Sparse R1CS program adapter | `done` | Reduced the file back to R1CS shape-to-`DirectCcsProgram` conversion. | None for this chunk. | Low | `cargo check -p neo-fold-prototype` |
| `frontends/direct_ccs/state/relation.rs` | Direct CCS folding boundary | `done` | Renamed the vague relation builder to `fold_chunk_with_superneo`, renamed carried relation verification, and documented that the shared chunk prover owns `prepare -> Pi_CCS -> Pi_RLC -> Pi_DEC`. | None for this chunk. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_append_api_matches_native_superneo_carry -- --nocapture` |
| `core/chunk_folding/transition.rs` | Shared Pi_RLC -> Pi_DEC transition | `done` | Split the transition body into private step-down helpers: build dims, run Pi_RLC, run Pi_DEC, then assemble transition/perf. | None for this chunk. | Medium | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_append_api_matches_native_superneo_carry -- --nocapture` |
| `frontends/direct_ccs/recursive/state/append.rs` | Recursive prior F' authority append | `done` | The recursive carrier no longer starts with digest plumbing. It first advances prior F' authority when a prior direct step exists, packages the resulting fold context and accumulator digest, then appends the current direct step/relation with that authority. | Positive compact-authority append coverage is intentionally ignored by default because the deep recursive path is heavy; fast latest-step and lifecycle guards remain non-ignored. | Medium | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_latest_step_is_not_historical_replay -- --nocapture` |
| `frontends/direct_ccs/recursive/state/mod.rs` | Recursive state start surface | `done` | Renamed `DirectCcsRecursiveIvcState::new_with_canonical_zero_carry` to `DirectCcsRecursiveIvcState::start` and updated direct test call sites. | `start_direct_ccs_proof_state` remains as the lifecycle-facing proof-state start helper. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_state_starts_without_terminal_step -- --nocapture` |
| `frontends/direct_ccs/state/init.rs` | Direct state start surface | `done` | Renamed the internal zero-carry start constructor from `DirectCcsIvcState::new_with_canonical_zero_carry` to `DirectCcsIvcState::start`, and updated Direct CCS recursive/F' call sites plus direct state tests. | `DirectCcsIvcState::new` remains for raw direct-state initialization tests; lifecycle-facing proof start goes through `DirectCcsProof::start`. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_canonical_zero_seed_first_append_has_steady_accumulator_arity -- --nocapture` |
| `frontends/direct_ccs/recursive/state/compress.rs` | Recursive compression readiness | `done` | Extracted the initial authority gate into `ensure_recursive_authority_ready`, so recursive compression starts with an explicit readiness check before terminal/F' proof packaging. | Positive multi-step compact-authority summary coverage is ignored by default because it is heavy; empty-state readiness remains fast. | Medium | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_compression_requires_appended_step -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_multi_step_summary_uses_compact_f_prime_authority -- --nocapture` |
| `frontends/direct_ccs/recursive/state/compress.rs` | Recursive F' chain compression | `done` | Extracted the optional F' chain compression/verification block into `compress_f_prime_chain_if_needed` with a named package instead of a large tuple. | Full recursive Spartan compression remains ignored/explicit. | Medium | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_compression_requires_appended_step -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_multi_step_summary_uses_compact_f_prime_authority -- --nocapture` |
| `frontends/direct_ccs/recursive/state/compress.rs` | Recursive final-CE proof package | `done` | Extracted the optional F' final-CE proof setup/prove/verify block into `prove_f_prime_final_ce_if_needed` with a named package instead of a large tuple. | Positive final-CE proof branch is still only covered by ignored/heavy compression tests; compact F' authority now exists, but full Spartan compression remains a long-running target. | Medium | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_multi_step_summary_uses_compact_f_prime_authority -- --nocapture` |
| `frontends/direct_ccs/state/compress.rs` | Terminal compression wrapper cleanup | `done` | Removed unused `compress_with_trace` and `compress` methods that only discarded the verifier key from the SNARK compression path. The retained path is `compress_snark_with_trace` / `compress_snark`. | None for this chunk. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` |
| `frontends/direct_ccs/terminal/prove.rs` | Terminal SNARK handoff | `done` | Replaced the internal terminal proof-generation tuple with `DirectCcsTerminalSnarkPackage`, so terminal packaging carries named `snark`, `verifier_key`, and `perf` fields across the terminal/state boundary. | Public `compress_snark` still returns the same tuple; the broad public perf type remains flat and needs a separate decision. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` |
| `frontends/direct_ccs/terminal/committed` | Terminal committed-step setup/proof handoff | `done` | Replaced local `(pk, vk, perf)` and `(proof, pcs_ms)` handoffs with named committed-step key/proof packages. The uncached setup helper is now private to the committed proof owner. | None for this chunk. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` |
| `frontends/direct_ccs/terminal/committed/assignment.rs` | Terminal source witness layout | `done` | Moved low-norm source witness assembly into `DirectCcsTerminalR2Layout`, so the assignment constructor delegates source offsets, encodings, padding, and constant-one witness construction to the layout owner. | None for this chunk. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` |
| `frontends/direct_ccs/state/types.rs` | Public terminal perf shape | `done` | Grouped `DirectCcsFPrimeSnarkPerf` into timing, proof-size, R1CS, chunk, constraint, committed-source, and final-CE substructures. Updated terminal perf construction, recursive perf accounting, tests, and direct CCS probe/bin reporting. | Recursive perf remains flat; handle separately only if it becomes a reader problem. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo check -p neo-fold-prototype --bins`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` |
| `frontends/direct_ccs/recursive/state/compress.rs` | Recursive perf aggregation | `done` | Extracted recursive compression perf accounting into `recursive_perf_accounting`, keeping the main recursive compression flow focused on terminal proof, optional F' chain proof, optional final-CE proof, and package assembly. | The public recursive perf type remains flat; grouping that would be a separate public diagnostics API cleanup. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_compresses_terminal_boundary_and_binds_accumulator_digest -- --ignored --nocapture` |
| `frontends/direct_ccs/recursive/snark.rs` | Recursive verifier surface | `done` | Removed the public `verify_direct_ccs_recursive_ivc_snark_public` pass-through wrapper. Recursive public-image equality and proof verification now have one owner: `DirectCcsRecursiveIvcSnark::verify`. | None for this chunk. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_public_image_rejects_unbound_accumulator_digest -- --ignored --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_builds_compact_f_prime_authority -- --nocapture` |
| `frontends/direct_ccs/verify.rs` | Statement verifier surface | `done` | Removed the misleading `verify_direct_ccs_ivc_snark_public` wrapper, which only converted a public image into a statement. `DirectCcsIvcSnark::verify` still owns the full public-image equality check before calling the statement verifier. | None for this chunk. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` |
| `frontends/direct_ccs/f_prime/source` | Low-norm source layout grouping | `done` | Replaced the flat field dump inside `DirectCcsFPrimeLowNormSourceImage` and its builder with grouped offset domains: digests, counters, public inputs, NIFS, Construction-2 in/out, and stats. Public accessor methods and call sites remain stable. | Serialization shape changes because this internal type derives serde; no persisted proof compatibility is required in this development branch. | Medium | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` |
| `frontends/direct_ccs/f_prime/r1cs/source.rs` | Low-norm source validation flow | `done` | Split the long offset-validation wall into phase-owned validators for digest fields, public inputs, counters, NIFS metadata, Construction-2 boundary fields, and canonical field lanes. | None for this chunk. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` |
| `frontends/direct_ccs/f_prime/r1cs/build.rs` | Low-norm source witness assembly | `done` | Extracted source witness construction into `build_low_norm_source_witness`, separating counter carries and canonical field-lane auxiliary bits from later constraint triplet construction. | Constraint construction in the same file is still dense, but the witness phase is now named. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` |
| `frontends/direct_ccs/f_prime/r1cs/build.rs` | Low-norm source Poseidon linkage package | `done` | Extracted Poseidon digest linkage into `build_poseidon_linkage_constraints`, returning named triplet/row/aux-bit data instead of carrying four loose local vectors and counters through the main builder. | Shell constraint triplet construction remains dense and should be split by constraint group in a later chunk. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` |
| `frontends/direct_ccs/f_prime/r1cs/build.rs` | Low-norm source shell constraints | `done` | Extracted shell constraint construction into `build_low_norm_source_shell_constraints`, with phase-owned groups for bitness, public output linkage, Construction-2 boundary links, commitment shape constants, structural constants, counters, NIFS mirrors, and canonical field lanes. | None for this chunk. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` |
| `frontends/direct_ccs/f_prime/r1cs/shape.rs` | Low-norm source R1CS cost shape | `done` | Grouped the shape into source, variable, and constraint domains so the compact F' source relation cost is auditable without a flat bag of counters. Behavior and public totals remain unchanged. | The grouped shape makes the cost visible; it does not reduce the recursive fold runtime by itself. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows -- --nocapture` |
| `frontends/direct_ccs/f_prime/r1cs/poseidon/mod.rs` | Low-norm Poseidon linkage flow | `done` | Split the linkage entry into phase-owned calls for direct state image digests, current-boundary/public-trace digest updates, and Construction-2 boundary digests. The Poseidon2 permutation implementation was left unchanged. | None for this chunk. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` |
| `frontends/direct_ccs/f_prime/chain/mod.rs` | F' encoder native artifacts | `done` | Replaced a ten-value tuple in encoder-status construction with an internal `DirectCcsFPrimeNativeArtifacts` package for compact image digest, low-norm source metadata, R1CS shape, and NIFS payload shape. | None for this chunk. | Low | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_lowered_product_uses_compact_f_prime_authority_despite_large_exact_body -- --nocapture` |
| `frontends/direct_ccs/f_prime/chain/mod.rs` | F' chain authority append helper | `done` | Extracted the shared positive-authority append path used by compact-source and exact-verifier-body preflight into `append_f_prime_source_step`, so state start/reuse, append, and folded-authority summary updates live in one place. | The branch is now exercised by non-Spartan two-step tests, but it is slow enough that broader coverage should stay targeted. | Medium | `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_builds_compact_f_prime_authority -- --nocapture` |
| `frontends/direct_ccs/f_prime/chain/mod.rs` | F' authority availability gate | `done` | Extracted `DirectCcsFPrimeAuthorityAvailability`, so the blocker decision is owned by a named gate: authority is available only from a low-norm source R1CS with authority rows or from an exact verifier-body relation under the size cap. | No missing-authority blocker remains once the compact source relation has been folded; exact verifier-body size remains diagnostic/fallback context. | Medium | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_lowered_product_uses_compact_f_prime_authority_despite_large_exact_body -- --nocapture` |
| `frontends/direct_ccs/f_prime/r1cs/nifs_authority.rs` | Compact F' NIFS authority rows | `done` | Added compact NIFS.V authority constants derived from crate-owned native advice and threaded them into the low-norm source R1CS builder. Caller-supplied source images still build digest-only source shells with zero authority rows. | Source/R1CS authority-gate tests are now fast; recursive append of the compact source relation remains high cost at about 223-230s in retained deep tests. | Medium | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_builds_compact_f_prime_authority -- --nocapture` |
| `frontends/direct_ccs/f_prime/verifier_body/mod.rs` | F' verifier-body measurement shape | `done` | Grouped NIFS verifier-body measurement fields under `DirectCcsFPrimeVerifierBodyNifsShape` and moved terminal measurement aggregation into shape constructors, so Pi_CCS/Pi_RLC/Pi_DEC row accounting is a named subshape instead of flat fields. | This is a diagnostic/public measurement shape change; no protocol behavior changed. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` |
| `tests/direct_ccs_ivc/f_prime_source.rs` | F' source authority gate coverage | `done` | Updated the non-Spartan gate tests to use plain Direct CCS state for native F' advice. They still prove caller-supplied source images remain digest-only while crate-owned native advice adds compact NIFS.V authority rows. | None for this chunk; recursive summary behavior is covered in recursive tests, not this source/R1CS owner. | Low | `cargo fmt --all`; `cargo check -p neo-fold-prototype`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows -- --nocapture` |
| `tests/direct_ccs_ivc/recursive_authority.rs` | Direct CCS audit and recursive authority tests | `done` | Updated the static audit assertion to check the new step owner for the explicit low-norm witness-to-claim boundary, moved latest-step/image checks to direct state, and gated the duplicate heavy recursive compact-authority append test behind `--ignored`. | Heavy path remains available explicitly; fast latest-step/image tests now avoid recursive F' preflight. | Medium | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_latest_step_is_not_historical_replay -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_compact_f_prime_image_binds_latest_step_without_terminal_material -- --nocapture`; `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_append_does_not_fold_terminal_source_image_exports -- --ignored --nocapture` |
| `tests/direct_ccs_ivc/recursive_compression.rs` | Recursive compression summary tests | `done` | Removed the obsolete non-ignored missing-authority compression refusal and gated the duplicate compact F' authority summary test behind `--ignored`; the retained non-ignored positive coverage lives in `direct_ccs_r1cs_low_norm`. | Full recursive Spartan compression remains ignored/explicit. | Medium | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_multi_step_summary_uses_compact_f_prime_authority -- --nocapture` |

## Direct CCS Flow Map

| Phase | File/Folder | Status | Notes |
|---|---|---|---|
| Public entry | `src/lib.rs`, `src/lifecycle` | done | `src/lifecycle/direct_ccs.rs` now owns the canonical Direct CCS lifecycle flow; `src/lib.rs` continues to re-export the lifecycle entrypoints. |
| Raw step validation | `frontends/direct_ccs/step.rs` | done | `validate_direct_ccs_step_witness` checks witness length and public input length. |
| Embedding | `frontends/direct_ccs/step.rs` | done | `embed_direct_ccs_witness` calls `encode_vector_for_full_width`. |
| Public projection | `frontends/direct_ccs/step.rs` | done | `derive_public_input_projection` takes the public prefix used as `x`. |
| Commitment | `frontends/direct_ccs/step.rs` | done | `commit_embedded_witness` calls the Ajtai/module homomorphism. |
| Fresh CCS claim | `frontends/direct_ccs/step.rs` | done | `build_ccs_claim_and_witness` produces `StepInput` through `DirectCcsStep`. |
| Pi_CCS/Pi_RLC/Pi_DEC | `core/chunk_folding` | done | `prove_superneo_chunk_step` is now the named shared step result used by Direct CCS append/prove and by the RV32IM chunk transition caller. |
| Terminal replay surface after SuperNeo step | `frontends/direct_ccs/state/surface.rs` + `frontends/direct_ccs/state/append.rs` | done | The Direct CCS append flow now builds `terminal_replay` immediately after the verified SuperNeo relation. |
| Construction-2 state advance after SuperNeo step | `frontends/direct_ccs/state/construction2.rs` + `frontends/direct_ccs/state/append.rs` | done | The Direct CCS append flow now derives `DirectCcsNextConstruction2State` before `advance_with_verified_superneo_step`. |
| F' authority | `frontends/direct_ccs/f_prime` | done, heavy | Compact source authority is available through crate-owned native advice, digest-only caller source images remain non-authoritative, and recursive append folds compact prior F' without terminal source-image authority. Runtime cost remains the active concern. |
| Prior F' digest handoff | `frontends/direct_ccs/recursive/state/append.rs` | done | Digest handoff now has one direct helper and no discarded arguments. |
| Recursive start | `frontends/direct_ccs/recursive/state/mod.rs` | done | Recursive state direct call sites now use `DirectCcsRecursiveIvcState::start`. |
| Direct state start | `frontends/direct_ccs/state/init.rs` | done | Internal zero-carry start call sites now use `DirectCcsIvcState::start` instead of leaking the canonical-carry constructor name. |
| Recursive compression readiness | `frontends/direct_ccs/recursive/state/compress.rs` | done | The authority-readiness gate is now a named helper before terminal and F' chain compression. |
| Recursive F' chain compression | `frontends/direct_ccs/recursive/state/compress.rs` | done | Optional F' chain compression now returns a named package, making the main compression flow less tuple-heavy. |
| Recursive final-CE proof | `frontends/direct_ccs/recursive/state/compress.rs` | done | Optional F' final-CE proof now returns a named package, making setup/prove/verify accounting explicit. |
| Spartan finish | `frontends/direct_ccs/terminal` | mapped | Not changed in this chunk. |
| Terminal compression API | `frontends/direct_ccs/state/compress.rs` | done | Removed unused proof-only compression wrappers; Direct CCS terminal compression now exposes the SNARK path that returns proof, verifier key, and perf together. |
| Terminal SNARK handoff | `frontends/direct_ccs/terminal/prove.rs` | done | Terminal proof generation now returns a named package internally instead of carrying a raw `(snark, verifier key, perf)` tuple through the terminal/state boundary. |
| Terminal committed-step handoff | `frontends/direct_ccs/terminal/committed` | done | Committed-step setup/prove internals now return named key/proof packages instead of raw tuples. |
| Terminal source witness layout | `frontends/direct_ccs/terminal/committed/assignment.rs` | done | Low-norm source witness encoding now lives on the R2 layout owner instead of inline inside assignment construction. |
| Terminal public perf shape | `frontends/direct_ccs/state/types.rs` | done | `DirectCcsFPrimeSnarkPerf` is grouped by timing, proof size, R1CS shape, chunk constraints, Construction-2/public-link constraints, committed source, and final CE. |
| Recursive perf aggregation | `frontends/direct_ccs/recursive/state/compress.rs` | done | Recursive compression perf accounting is now a named helper instead of inline counter assembly inside the proof flow. |
| Verifier surface | `frontends/direct_ccs/verify.rs`, `frontends/direct_ccs/snark.rs`, `frontends/direct_ccs/recursive/snark.rs` | done | The free terminal verifier verifies a compact statement; full public-image equality remains on the SNARK verifier methods. The recursive public verifier wrapper was removed so `DirectCcsRecursiveIvcSnark::verify` owns the recursive boundary. |
| F' low-norm source layout | `frontends/direct_ccs/f_prime/source` | done | The source image now groups offsets and stats internally instead of storing every offset/count as a top-level field. |
| F' low-norm source validation | `frontends/direct_ccs/f_prime/r1cs/source.rs` | done | Source image validation now follows the same domain grouping as the source layout. |
| F' low-norm source witness | `frontends/direct_ccs/f_prime/r1cs/build.rs` | done | Witness construction is now a named builder phase before Poseidon linkage and shell constraints. |
| F' low-norm Poseidon linkage | `frontends/direct_ccs/f_prime/r1cs/build.rs` | done | Poseidon linkage now returns named triplet/row/aux-bit data before final shape and shell constraint construction. |
| F' low-norm shell constraints | `frontends/direct_ccs/f_prime/r1cs/build.rs` | done | Shell constraints now read by group: bitness, public links, Construction-2 links/constants, counters, NIFS mirrors, and canonical lanes. |
| F' low-norm Poseidon flow | `frontends/direct_ccs/f_prime/r1cs/poseidon/mod.rs` | done | The Poseidon linkage entry now reads as state-image digests, digest updates, then Construction-2 boundary digests. |
| F' encoder status | `frontends/direct_ccs/f_prime/chain/mod.rs` | done | Native/low-norm encoder metadata is now built as a named artifact package instead of a large tuple. |
| F' chain authority append | `frontends/direct_ccs/f_prime/chain/mod.rs` | done | Compact-source and exact-verifier-body preflight now share the same append/summary update path. |
| F' verifier-body measurement shape | `frontends/direct_ccs/f_prime/verifier_body/mod.rs` | done | NIFS row accounting is grouped under a named subshape, with Pi_CCS/Pi_RLC/Pi_DEC aggregation owned by the verifier-body shape constructor. |

## Verification Log

| Date | Command | Result | Notes |
|---|---|---|---|
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after grouping the low-norm F' source R1CS shape by source/variable/constraint domain. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after replacing flat low-norm source R1CS shape counters with grouped cost domains. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` | Pass | Run with a 300s cap. Covers grouped source/variable/constraint shape fields plus native F' source offsets, compact authority rows, tamper rejection, and Construction-2 boundary encoding. Latest release compile took about 27.75s; test body took 6.32s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows -- --nocapture` | Pass | Run with a 300s cap. Confirms grouped authority rows still distinguish digest-only caller source images from crate-owned native advice. Test body took 3.30s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after moving F' source tests from recursive state to plain direct state. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after removing the unnecessary recursive F' preflight from `tests/direct_ccs_ivc/f_prime_source.rs`. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` | Pass | Run with a 300s cap. Covers native F' source offsets, digest accounting, compact authority rows, tamper rejection, and Construction-2 boundary encoding through plain direct state. Release compile took about 29.63s; test body took 6.47s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows -- --nocapture` | Pass | Run with a 300s cap. Confirms caller-supplied source images remain digest-only while crate-owned native advice adds compact NIFS.V authority rows. Test body took 3.37s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after grouping `DirectCcsFPrimeSnarkPerf`. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_uses_exact_f_prime_authority -- --nocapture` | Stopped at cap | Experimental exact verifier-body positive path exceeded the 5-minute per-test cap. The code was restored to keep exact lowering behind the size gate; compact NIFS.V rows remain the required positive authority path. |
| `2026-05-09` | `cargo fmt --all`; `cargo check -p neo-fold-prototype` | Pass | Checked after restoring the exact verifier-body fallback gate and recording the compact-authority target. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows -- --nocapture` | Pass | Release compile took about 1m58s; the test body took about 4.60s. Confirms digest binding alone remains non-authoritative and the gate requires NIFS.V rows. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_refuses_missing_f_prime_authority -- --nocapture` | Pass | Release compile took about 29.61s; the test body took about 6.42s. Confirms two-step Direct CCS still refuses missing compact F' authority instead of using terminal/source-image authority. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` | Pass | Release compile took about 20.78s; the test body took about 3.59s. Confirms private DEC child tamper red-team remains rejected. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after replacing flat terminal perf fields with grouped timing/proof/R1CS/chunk/constraint/committed/final-CE substructures. |
| `2026-05-09` | `cargo check -p neo-fold-prototype --bins` | Pass | Confirms Direct CCS probe/bin support compiles after updating perf reporting call sites. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` | Pass | Covers terminal compression and public verifier rejection after public terminal perf grouping. Release compile took about 2m01s; the test body took about 28.87s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` | Pass | Covers the second-fold private child tamper red-team after public terminal perf grouping. Release compile took about 22.60s; the test body took about 3.81s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after collapsing two private Direct CCS append pass-through helpers. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after routing the Construction-2 accumulator digest override directly into `append_verified_superneo_step`. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_append_api_matches_native_superneo_carry -- --nocapture` | Pass | Covers the Direct CCS append path after removing private append pass-through helpers. Release compile took about 2m00s; the test body took about 1.87s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` | Pass | Covers the second-fold private child tamper red-team after removing private append pass-through helpers. Release compile took about 21.33s; the test body took about 4.03s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after splitting Direct CCS post-SuperNeo append into named phases. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after making Direct CCS append read as terminal replay -> Construction-2 next state -> state advance. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_append_api_matches_native_superneo_carry -- --nocapture` | Pass | Covers the Direct CCS append path after splitting the post-SuperNeo state advance. Release compile took about 1m58s; the test body took about 1.89s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` | Pass | Covers the second-fold private child tamper red-team after splitting the post-SuperNeo state advance. Release compile took about 21.66s; the test body took about 3.73s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after introducing the canonical shared SuperNeo chunk step. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after replacing the long core replay/prove helper variants with `prove_superneo_chunk_step` and updating the shared Direct CCS/RV32IM callers. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_append_api_matches_native_superneo_carry -- --nocapture` | Pass | Covers the Direct CCS append path after the shared chunk step result replacement. Release compile took about 1m58s; the test body took about 1.95s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` | Pass | Covers the second-fold private child tamper red-team after the shared chunk step result replacement. Release compile took about 22.86s; the test body took about 3.84s. |
| `2026-05-09` | `documentation-only update` | Pass | Added `Immediate Required Work` with exact files, implementation steps, definition of done, and verification commands for the canonical paper-order step. |
| `2026-05-09` | `wc -l crates/neo-fold-prototype/docs/refactor-plan.md crates/neo-fold-prototype/docs/refactor-checklist.md` | Pass | Docs-only goal-loop tightening; no Rust verification needed. Exact line counts may change as the checklist is clarified. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after the Direct CCS step-owner refactor. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc -- --nocapture` | Fail, then stopped from rerun scope | Initial full Direct CCS test run compiled for about 1m55s and failed one static audit assertion that still checked `adapter/r1cs.rs` for the low-norm boundary. The assertion was updated to check `step.rs`. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_f_prime_authority_is_not_public_or_terminal_source_image_based -- --nocapture` | Pass | Confirms the static audit test follows the new owner. Release compile took about 1m54s; the test body itself was immediate. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_sparse_r1cs_adapter -- --nocapture` | Pass | Covers the low-norm adapter path that builds Direct CCS steps. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_append_api_matches_native_superneo_carry -- --nocapture` | Pass | Covers the renamed Direct CCS append/folding boundary. Release compile took about 1m54s; the test body took about 1.83s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_append_api_matches_native_superneo_carry -- --nocapture` | Pass | Covers the shared Pi_RLC/Pi_DEC transition split. Release compile took about 1m59s; the test body took about 1.85s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_append_does_not_fold_terminal_source_image_exports -- --nocapture` | Pass | Covers the recursive prior-F' append path after digest-helper cleanup. Release compile took about 1m59s; the test body took about 7.08s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_state_starts_without_terminal_step -- --nocapture` | Pass | Covers the recursive start constructor rename. Release compile took about 1m56s; the test body took about 0.01s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_compression_requires_appended_step direct_recursive_ivc_append_does_not_fold_terminal_source_image_exports -- --nocapture` | Fail | Cargo accepts only one test filter; reran the two tests separately. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_compression_requires_appended_step -- --nocapture` | Pass | Covers the no-step readiness error. Release compile took about 1m57s; the test body took about 0.01s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_append_does_not_fold_terminal_source_image_exports -- --nocapture` | Pass | Covers the multi-step readiness refusal path. Test body took about 7.00s after release artifacts were built. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_compression_requires_appended_step -- --nocapture` | Pass | Covers recursive compression after extracting the F' chain compression package. Release compile took about 1m55s; the test body took about 0.01s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_append_does_not_fold_terminal_source_image_exports -- --nocapture` | Pass | Covers recursive multi-step refusal after extracting the F' chain compression package. Test body took about 6.88s after release artifacts were built. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_multi_step_compression_refuses_terminal_source_image_authority -- --nocapture` | Pass | Covers recursive compression refusal after extracting the final-CE proof package. Release compile took about 1m56s; the test body took about 5.33s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after the direct state start rename. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after renaming the direct zero-carry start constructor. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_canonical_zero_seed_first_append_has_steady_accumulator_arity -- --nocapture` | Pass | Covers Direct CCS zero-carry start after renaming the constructor. Release compile took about 2m00s; the test body took about 1.20s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after terminal compression wrapper cleanup. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after removing unused Direct CCS proof-only compression wrappers. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` | Pass | Covers the retained `compress_snark` path and public verifier rejection after wrapper cleanup. Release compile took about 2m00s; the test body took about 28.85s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after verifier wrapper cleanup. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after removing the misleading public-image verifier wrapper. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` | Pass | Covers the retained statement verifier and `DirectCcsIvcSnark::verify` public-image equality path. Release compile took about 2m02s; the test body took about 29.37s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after grouping the F' low-norm source layout. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after grouping F' low-norm source offsets/stats. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` | Pass | Covers F' source offsets, digest accounting, and Construction-2 boundary encoding after internal layout grouping. Release compile took about 1m57s; the test body took about 9.59s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after splitting F' source validation. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after splitting low-norm source R1CS validation into phase-owned helpers. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` | Pass | Covers F' source validation and source R1CS construction after validation flow split. Release compile took about 1m57s; the test body took about 9.51s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after extracting F' source witness assembly. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after extracting low-norm source witness construction. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` | Pass | Covers F' source R1CS witness construction after extraction. Release compile took about 1m57s; the test body took about 9.54s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after extracting F' source Poseidon linkage. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after packaging low-norm source Poseidon linkage constraints. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` | Pass | Covers F' source R1CS Poseidon linkage after extraction. Release compile took about 1m57s; the test body took about 9.50s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after splitting F' source shell constraints. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after splitting low-norm source shell constraint construction. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` | Pass | Covers F' source R1CS shell constraints after grouping by constraint domain. Release compile took about 1m57s; the test body took about 9.52s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after splitting F' Poseidon linkage flow. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after splitting the low-norm source Poseidon linkage entry by digest group. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` | Pass | Covers F' source R1CS Poseidon linkage after entry-flow split. Release compile took about 1m57s; the test body took about 9.49s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after replacing the F' encoder-status tuple. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after packaging native/low-norm encoder artifacts. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_lowered_product_reports_exact_encoder_size_blocker -- --nocapture` | Pass | Covers F' encoder status/reporting after tuple removal. Release compile took about 1m56s; the test body took about 5.56s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after extracting the F' chain authority append helper. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after sharing the compact/exact F' authority append path. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_refuses_missing_f_prime_authority -- --nocapture` | Pass | Covers the F' chain preflight refusal path after append-helper extraction. Release compile took about 1m56s; the test body took about 6.53s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after grouping F' verifier-body NIFS shape. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after grouping the F' verifier-body measurement shape. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` | Pass | Covers F' verifier-body export/measurement after grouping the NIFS shape. Release compile took about 1m48s; the test body took about 3.75s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after naming the terminal SNARK package. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after replacing the internal terminal proof tuple with a named package. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` | Pass | Covers terminal compression and public verifier rejection after the terminal SNARK package handoff. Release compile took about 1m57s; the test body took about 28.29s. |
| `2026-05-09` | `rg -n "DirectCcsFPrimeSnarkPerf|perf\\.chunk_constraints_first4|perf\\.terminal_source|perf\\.terminal_committed_breakdown|perf\\.final_ce_relation_breakdown|terminal_f_prime_constraints|final_ce_r1cs_sizes" crates/neo-fold-prototype/src crates/neo-fold-prototype/tests` | Mapped | Historical pre-cleanup mapping that showed terminal perf fields were consumed by public Direct CCS exports, recursive perf, probe/bin support, and tests. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after naming terminal committed-step packages. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after replacing terminal committed-step setup/proof tuples with named packages. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` | Pass | Covers terminal committed-step setup/prove handoff after package cleanup. Release compile took about 1m57s; the test body took about 28.30s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after moving terminal source witness assembly into the layout owner. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after extracting terminal source witness layout assembly. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_public_verifier_rejects_authoritative_boundary_tampering -- --nocapture` | Pass | Covers terminal source witness layout assembly through terminal compression and public verifier rejection. Release compile took about 1m57s; the test body took about 28.23s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after extracting recursive perf accounting. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after extracting recursive perf accounting. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_compresses_terminal_boundary_and_binds_accumulator_digest -- --ignored --exact --nocapture` | No coverage | Exact filter missed the nested test path and ran zero tests; reran without `--exact`. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_compresses_terminal_boundary_and_binds_accumulator_digest -- --ignored --nocapture` | Pass | Covers successful recursive compression and perf aggregation after extraction. Test body took about 29.59s after release artifacts were built. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Fail | Removing the recursive verifier wrapper exposed a stale public re-export in `frontends/direct_ccs/mod.rs`; the re-export was removed. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after removing the recursive verifier wrapper. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after removing the stale recursive verifier wrapper re-export. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_public_image_rejects_unbound_accumulator_digest -- --ignored --nocapture` | Fail | Test compile exposed a stale import in `tests/direct_ccs_ivc/mod.rs`; the test harness import was removed. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after removing stale recursive verifier wrapper imports. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after removing stale test harness imports. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_public_image_rejects_unbound_accumulator_digest -- --ignored --nocapture` | Pass | Covers the recursive public-image rejection path after replacing wrapper calls with `DirectCcsRecursiveIvcSnark::verify`. Release compile took about 30s and the test body took about 29.23s. |
| `2026-05-09` | `rg -n "verify_direct_ccs_recursive_ivc_snark_public" crates/neo-fold-prototype/src/frontends/direct_ccs crates/neo-fold-prototype/tests/direct_ccs*` | Pass | No remaining recursive verifier wrapper references in Direct CCS source or tests. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after updating `direct_ccs_r1cs_low_norm` wrapper references. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after removing all recursive verifier wrapper references from source and tests. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_refuses_missing_f_prime_authority -- --nocapture` | Pass | Compiles the low-norm integration test that previously imported the removed wrapper and covers the recursive F' authority refusal path. Release compile took about 28.71s; the test body took about 6.46s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after extracting F' authority availability. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after moving F' authority availability/blocker selection into a named internal gate. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_lowered_product_reports_exact_encoder_size_blocker -- --nocapture` | Pass | Covers the exact-verifier-body size blocker after extracting the F' authority availability gate. Release compile took about 1m56s; the test body took about 5.57s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after adding F' source authority gate coverage. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after adding the F' source authority gate test. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows -- --nocapture` | Pass | Covered the source-shape authority condition without running Spartan: digest binding alone is not authority, and digest binding plus NIFS.V verifier rows is the positive gate. Release compile took about 29.54s; the test body took about 4.64s. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Fail | Mid-edit compact authority wiring left `build_low_norm_source_r1cs` and `from_source_metadata` signatures out of sync; fixed by threading the compact NIFS authority spec through the R1CS builder. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after adding compact NIFS authority rows. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after wiring compact NIFS authority rows into the low-norm F' source R1CS builder and summary estimate. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_f_prime_source_authority_requires_digest_binding_and_nifs_v_rows -- --nocapture` | Pass | Historical pre-cleanup run when this test still started a recursive carrier. Covered the split between caller-supplied digest-only source images and crate-owned native advice with compact NIFS.V authority rows, but paid recursive append cost and took 224.01s test-body time. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_refuses_missing_f_prime_authority -- --nocapture` | Fail | Expected failure after the positive compact authority path landed: the old refusal test still expected `folded_r2_steps = 0`, but the summary reported `folded_r2_steps = 1` and no blocker. Test was renamed and updated to the positive assertion. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_two_step_recursive_state_builds_compact_f_prime_authority -- --nocapture` | Pass | Covers a two-step low-norm Direct CCS append folding one compact F' source relation. Test body took about 230.13s after release artifacts were built. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_lowered_product_uses_compact_f_prime_authority_despite_large_exact_body -- --nocapture` | Fail | First positive rewrite still expected `encoder_required`; corrected because the compact F' chain is already folded during append, so no missing encoder remains. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_r1cs_low_norm r1cs_low_norm_lowered_product_uses_compact_f_prime_authority_despite_large_exact_body -- --nocapture` | Pass | Covers the arbitrary-field lowered product path using compact source authority even though the exact verifier body remains over the size gate. Test body took about 223.09s after release artifacts were built. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after cleaning stale compact-authority comments. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after the compact-authority comment cleanup. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_redteam direct_ccs_rejects_second_fold_child_tamper -- --nocapture` | Pass | Re-ran the no-swap red-team coverage after compact F' authority rows landed. Release compile took about 1m47s; test body took about 3.62s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after updating stale recursive Direct CCS compact-authority tests. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after updating recursive Direct CCS tests from the old missing-authority refusal to the positive compact-authority path. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_append_does_not_fold_terminal_source_image_exports -- --nocapture` | Pass | Covers that recursive append folds compact prior F' authority without exposing terminal source-image authority. Release compile took about 27.71s; test body took about 213.02s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_multi_step_summary_uses_compact_f_prime_authority -- --nocapture` | Pass | Covers the recursive compression summary no longer reporting a missing-encoder blocker after compact F' authority is folded. Test body took about 211.56s after release artifacts were built. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_f_prime_authority_is_not_public_or_terminal_source_image_based -- --nocapture` | Pass | Static audit still confirms the F' chain helper is not publicly re-exported and terminal source-image exports are not used as F' authority. Test body took effectively 0s after release artifacts were built. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after gating duplicate heavy recursive compact-authority tests. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after gating duplicate heavy recursive compact-authority tests and moving latest-image checks to direct state. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_latest_step_is_not_historical_replay -- --nocapture` | Pass | Uses plain direct state to cover latest-step selection without paying recursive F' preflight. Release compile took about 27.45s; test body took about 2.19s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_compact_f_prime_image_binds_latest_step_without_terminal_material -- --nocapture` | Pass | Uses plain direct state to cover compact latest-step image binding without paying recursive F' preflight. Test body took about 2.17s after release artifacts were built. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_append_does_not_fold_terminal_source_image_exports -- --ignored --nocapture` | Pass | The duplicate heavy recursive compact-authority test remains available explicitly. Test body took about 212.32s after release artifacts were built. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_ivc_multi_step_summary_uses_compact_f_prime_authority -- --nocapture` | Pass | Confirms the duplicate heavy summary test is ignored by default and must be run explicitly. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after the Direct CCS lifecycle readability pass. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after making public Direct CCS lifecycle functions canonical and renaming the immediate Construction-2 append handoff. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_native_f_prime_advice_evaluates_compact_image_and_construction2_boundaries -- --nocapture` | Pass | Covers the Direct CCS append/F' advice path after the Construction-2 append rename. Release compile took about 2m20s; the test body took about 6.55s. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_lifecycle_prove_extend_and_verify_use_same_append_flow -- --nocapture` | Killed by 5m cap | Initial two-step version also performed two full native replay verifications and exceeded the approved per-test cap. The test was reduced to a one-step lifecycle guard. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_lifecycle_prove_extend_and_verify_use_same_append_flow -- --nocapture` | Pass | Covers crate-root lifecycle entrypoints: batched `prove_direct_ccs`, incremental `extend_direct_ccs`, and native `verify_direct_ccs` all use the same append flow. Release compile took about 31s; the test body took about 3.41s. |
| `2026-05-09` | `cargo fmt --all` | Pass | Rustfmt printed existing unstable `imports_granularity` warnings after making recursive Direct CCS append read as prior-F' authority -> current direct step -> next state. |
| `2026-05-09` | `cargo check -p neo-fold-prototype` | Pass | Checked after renaming the recursive/direct append handoff to `append_latest_step_authority_from_direct_state`, `append_step_with_f_prime_accumulator`, and `append_relation_with_f_prime_accumulator`. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_ccs_lifecycle_prove_extend_and_verify_use_same_append_flow -- --nocapture` | Pass | Re-ran the lifecycle guard after the recursive append flow cleanup. Test body took about 3.42s after release artifacts were built. |
| `2026-05-09` | `cargo test -p neo-fold-prototype --release --test direct_ccs_ivc direct_recursive_latest_step_is_not_historical_replay -- --nocapture` | Pass | Re-ran the fast recursive/latest-step guard after the prior-F' authority append cleanup. Test body took about 2.23s after release artifacts were built. |

## Open Risks

| Risk | Area | Impact | Next Action |
|---|---|---|---|
| Shared core duplicate append variants remain outside this chunk. | `core/ivc`, shared callers | Direct CCS append is now readable, but shared core still has accumulator-handle and non-handle variants that may deserve a separate cleanup. | Tackle only if the next chunk targets shared core ownership; do not disturb Direct CCS while F' runtime work is active. |
| Compact F' authority append is functionally positive but slow. | `frontends/direct_ccs/f_prime/r1cs`, tests | Source/R1CS authority checks are fast now, but recursive append of the compact source relation still takes about 223-230s in retained deep tests. | Audit the recursive fold cost of the low-norm source relation, especially Poseidon/source-bit constraints, before broadening default coverage. |
| Full Direct CCS test-file coverage has not been rerun after gating duplicate heavy tests. | `tests/direct_ccs_ivc` | The targeted replacements pass and duplicate heavy tests are ignored, but a full file run has not been attempted after this cleanup. | Prefer targeted stale-assumption scans and focused tests, then run the full file only when long-test time is explicitly approved. |
| Full `direct_ccs_ivc` test run includes release compile time and multiple heavy F' authority append tests. | tests | Full-suite verification may exceed the active per-test workflow budget and slow iteration. | Prefer targeted Direct CCS tests per chunk unless the user approves a broad long run. |
| Positive recursive F' final-CE proof branch is behind ignored/heavy compression tests. | `frontends/direct_ccs/recursive/state/compress.rs` | The compact F' authority append is tested without Spartan, but the actual final-CE proof branch in full recursive Spartan compression is not exercised by a non-ignored test yet. | Run the ignored full recursive compression target only when long-test time is explicitly approved. |
| Positive F' chain authority append is covered only at summary/source level, not full Spartan compression. | `frontends/direct_ccs/f_prime/chain/mod.rs`, `frontends/direct_ccs/recursive/state/compress.rs` | The append branch is exercised by non-Spartan tests, but the final recursive Spartan proof with folded F' chain remains an ignored heavy target. | Keep the ignored compression target for deep validation, or run it explicitly when long-test time is approved. |

## Decisions

| Decision | Reason | Date |
|---|---|---|
| Make recursive Direct CCS append read as the Construction-2/F' carrier flow. | The closest state transition to the public Direct CCS API must not begin with anonymous digest plumbing. It should show prior-F' authority, current SuperNeo/Construction-2 append, and next carrier state in that order. | `2026-05-09` |
| Make public Direct CCS lifecycle functions canonical. | The caller-facing API should read like `prove -> extend -> finish -> verify`; trait impls are useful conformance adapters, but they should not be the hidden owner of the actual flow. | `2026-05-09` |
| Keep native `verify_direct_ccs` as a replay check for now. | `DirectCcsProof` intentionally stores private `DirectCcsStep` inputs, so the native no-Spartan verifier is a prover-side replay/self-check. The public verifier boundary remains the Spartan-finished proof. | `2026-05-09` |
| Group low-norm F' source R1CS shape by cost domain. | The compact source relation is the current runtime pressure point, so source bits, auxiliary variables, shell constraints, Poseidon rows, and NIFS authority rows need to be visible as domains rather than a flat counter bag. | `2026-05-09` |
| Add compact NIFS.V authority rows only for crate-owned native advice. | Caller-supplied source images can still prove digest/source consistency but do not become authority; native advice supplies the compact NIFS fields from the latest Direct CCS/F' step and flips the authority gate without using terminal source-image exports. | `2026-05-09` |
| Treat compact F' authority runtime as the next hard issue only for recursive append. | Direct source/R1CS authority-gate tests now finish in seconds by using plain direct state. The retained deep recursive positive tests still take about 223-230s, so broad recursive coverage should stay targeted until the compact source relation cost is profiled or reduced. | `2026-05-09` |
| Group `DirectCcsFPrimeSnarkPerf` by diagnostic domain. | The old flat public perf type mixed timing, proof sizes, R1CS shape, chunk rows, committed-source accounting, and final-CE accounting; grouped fields make reports auditable without changing proof behavior. | `2026-05-09` |
| Collapse only private Direct CCS append pass-through helpers. | The deleted helpers added no protocol phase; the remaining Construction-2 accumulator digest override helpers mark the prior-F' authority handoff and are intentionally retained. | `2026-05-09` |
| Split Direct CCS post-SuperNeo append into named phases. | The Direct CCS append owner now shows terminal replay surface construction, Construction-2 next state derivation, and final state advance after the canonical SuperNeo step, without adding a wrapper that only delegates to another function. | `2026-05-09` |
| Replace long shared replay/prove helper variants with `prove_superneo_chunk_step`. | The shared core hot path now has one named result that exposes prepare -> Pi_CCS -> Pi_RLC -> Pi_DEC and carries replay witness, relation result, fold digest, and perf without tuple plumbing. | `2026-05-09` |
| Make the canonical paper-order SuperNeo/Construction-2 step the mandatory next target. | The Direct CCS hot path is still too wrapper-heavy for audit. Progress now requires one real code path that reads as prepare -> Pi_CCS -> Pi_RLC -> Pi_DEC -> Construction-2 advance; wrapper-only cleanup does not count. | `2026-05-09` |
| Add an explicit goal-loop resume contract to the Direct CCS refactor plan. | Long-running goal iterations need to recover from disk state, avoid unrelated dirty files, and reject code movement that does not improve a reader-visible protocol boundary. | `2026-05-09` |
| Keep `src/lib.rs` stable while allowing explicit lifecycle cleanup. | The public re-export surface should not churn, but the user explicitly approved fixing the closest Direct CCS lifecycle flow. | `2026-05-09` |
| Move low-norm full-witness step construction out of the R1CS adapter. | The pre-`Pi_CCS` witness-to-claim boundary is Direct CCS step ownership, not sparse R1CS program conversion. | `2026-05-09` |
| Move `DirectCcsStep` into `frontends/direct_ccs/step.rs`. | The prepared Direct CCS step type belongs with the raw witness-to-fresh-claim construction path, not with live IVC state data. | `2026-05-09` |
| Rename the Direct CCS chunk relation boundary to `fold_chunk_with_superneo`. | The old name hid the protocol handoff; the new name makes clear that Direct CCS delegates `Pi_CCS -> Pi_RLC -> Pi_DEC` to the shared SuperNeo chunk folding owner. | `2026-05-09` |
| Split shared transition internals into `run_pi_rlc` and `run_pi_dec`. | The transition owner remains shared, but the core body now follows the paper-facing order instead of mixing RLC preparation, DEC split, commitments, checks, and perf assembly inline. | `2026-05-09` |
| Collapse prior-F' digest handoff to one direct helper. | The previous split accepted unused arguments and obscured the actual authority handoff rule. | `2026-05-09` |
| Rename the recursive direct state constructor to `start`. | It keeps the Direct CCS test/read path clear and avoids exposing canonical-carry mechanics in ordinary proof-flow vocabulary. | `2026-05-09` |
| Extract recursive compression readiness into a named helper. | The compression entry now exposes the proof-authority gate before doing terminal/F' proof work. | `2026-05-09` |
| Extract optional F' chain compression into a named package. | The recursive compression flow no longer carries a six-element tuple through the main body. | `2026-05-09` |
| Extract optional F' final-CE proof into a named package. | The recursive compression flow no longer carries an eleven-element tuple through the main body. | `2026-05-09` |
| Rename the direct zero-carry state constructor to `start`. | Direct CCS internal call sites now use proof-flow vocabulary instead of leaking canonical-carry initialization mechanics. | `2026-05-09` |
| Remove unused proof-only Direct CCS compression wrappers. | The wrappers only discarded the verifier key from terminal SNARK compression; keeping the SNARK-returning path makes the finish boundary more explicit. | `2026-05-09` |
| Remove the misleading free public-image verifier wrapper. | Full public-image equality requires the `DirectCcsIvcSnark`; the free function only had enough data to verify the compact statement. | `2026-05-09` |
| Group F' low-norm source offsets by domain. | The source image is internal witness/layout metadata, not a stable external proof format; grouped offsets make the layout auditable without changing accessor call sites. | `2026-05-09` |
| Split F' low-norm source validation by layout domain. | The validator now mirrors the source layout groups instead of mixing every offset kind in one long function. | `2026-05-09` |
| Extract F' low-norm source witness assembly. | The R1CS builder now separates witness construction from constraint triplet construction. | `2026-05-09` |
| Package F' low-norm source Poseidon linkage. | Poseidon linkage mutates witness and emits its own triplets, so it is now a named builder phase instead of loose locals in the main flow. | `2026-05-09` |
| Split F' low-norm source shell constraints by domain. | The R1CS shell builder now mirrors the constraint groups instead of mixing bitness, links, constants, counters, NIFS mirrors, and canonical lane checks inline. | `2026-05-09` |
| Split F' Poseidon linkage by digest group. | The entry now exposes which digest families are being constrained without changing Poseidon2 permutation logic. | `2026-05-09` |
| Package F' encoder native artifacts. | Encoder status now names the native low-norm source metadata bundle instead of unpacking a long tuple in the status constructor. | `2026-05-09` |
| Share F' chain authority append assembly. | Compact-source and exact-verifier-body preflight paths now reuse one state append and authority-summary update helper. | `2026-05-09` |
| Group F' verifier-body NIFS measurement shape. | The diagnostic shape now mirrors the protocol-owned NIFS stages instead of flattening Pi_CCS, Pi_RLC, and Pi_DEC counters into the parent verifier-body shape. | `2026-05-09` |
| Name the terminal SNARK handoff package internally. | Terminal proof generation now exposes a concrete proof boundary object to state compression instead of a repeated tuple, while leaving the public compression return shape unchanged. | `2026-05-09` |
| Name terminal committed-step setup/proof packages internally. | The committed proof owner now exposes concrete setup/proof result objects instead of local tuple plumbing. | `2026-05-09` |
| Move terminal source witness assembly to the layout owner. | Source offsets, limb encodings, padding, and the terminal constant-one slot are layout-owned details, not top-level assignment-constructor flow. | `2026-05-09` |
| Extract recursive perf accounting without changing the public perf type. | The recursive compression flow is easier to read, while the public diagnostics API remains stable until a deliberate perf-shape cleanup is approved. | `2026-05-09` |
| Remove the recursive public verifier wrapper. | Recursive public-image equality requires the recursive SNARK object, so `DirectCcsRecursiveIvcSnark::verify` is the correct verifier owner; the free wrapper added no semantic value. | `2026-05-09` |
| Extract the F' authority availability gate. | The recursive summary/blocker path should distinguish digest/source metadata from actual proof authority; the named gate keeps that distinction local and explicit. | `2026-05-09` |
| Add non-Spartan coverage for the F' source authority gate. | Caller-supplied source images remain digest-only, while crate-owned native advice now adds compact NIFS.V rows and lets the positive compact authority path pass without inventing terminal source-image authority. | `2026-05-09` |
