# IVC Refactor Progress

## Current State
- Goal 2 fixed-shape closure is still green on the single-step tree; native multi-step family work is now live and in progress.
- `src/rv64im/ivc.rs` remains the live native owner and now carries an explicit frozen `step_cap`.
- Native recursion family identity is no longer implicit single-step:
  - `RecursionShape` includes `step_cap`
  - `Rv64imVerifierKeyFs` includes `step_cap`
  - canonical recursion-shape / `vk_fs` digests now change when `step_cap` changes
- `Rv64imIvcState` init is explicit: callers must choose `step_cap` up front.
- Native append/verify now enforce:
  - non-terminal relation width `== step_cap`
  - terminal relation width `1..=step_cap`
- Construction-2 fresh-instance commitment now pads the full `z` image to the canonical family width before committing; underfull terminal chunks no longer shrink the native family.
- `tests/perf_rv64im_native.rs` derives native `step_cap` from the fold schedule:
  - `RowsPerChunk(1)` -> `step_cap = 1`
  - `WholeTrace` -> `step_cap = semantic_step_count`
- Live `k_rho` remains `48`.
- Compression-terminal relation rebuilding now derives native `step_cap` from the final statement’s fold schedule + semantic step count instead of assuming the legacy single-step family.
- Nightstream `proof_binding_root` now binds the full carried `Rv64imProofStatement` digest; self-consistent public-statement shell swaps no longer verify against a reused Nightstream proof.

## Open Chunks
- Add explicit fixed-shape canaries for `M > 1`, especially “full chunk at cap M” vs “short terminal chunk padded internally to the same M”.
- Sweep remaining published/audit/backend helpers that still default to the single-step family where a schedule-derived `step_cap` should be explicit.
- Add/refresh `Rv64imIvcState` runtime coverage for `M > 1`:
  - serialize -> deserialize -> append -> verify on a multi-step family
  - one-shot `WholeTrace` family creation and verify
- Continue the Nightstream trust-boundary sweep beyond the public-statement shell fix:
  - root-execution surfaces are still compact/digest-heavy
  - decide which theorem-bearing side objects must move across the published boundary vs stay under an existing proof

## Recent Evidence
- Chunk 74 — align the parked side-binding Spartan path with the new three-field Nightstream side statement (2026-04-20)
- **Idea** — fix the side-binding circuit/test fallout after the Nightstream boundary reduction by hashing the same `(nightstream_statement_core_digest, public_statement_digest, public_instance_digest)` statement shape everywhere and by moving the forged-public soundness check to the circuit-unsat boundary we actually own.
- **Files touched** — `src/nightstream/rv64im.rs`, `src/nightstream/rv64im/side_relation_spartan.rs`, `tests/rv64im_side_spartan_roundtrip.rs`, `tests/side_soundness/positive.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — none; this chunk repaired stale parked test/circuit ownership rather than deleting a live path.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --test rv64im_side_spartan_roundtrip --test side_soundness_suite`).
- **Tests run** — green ignored exact `cargo test -p neo-fold-next --release --test rv64im_side_spartan_roundtrip rv64im_side_binding_roundtrip_with_same_and_rebuilt_vk -- --ignored --exact`; green ignored exact `cargo test -p neo-fold-next --release --test side_soundness_suite side_soundness::positive::rv64im_side_soundness_positive_binding_rejects_forged_public_witness -- --ignored --exact`.
- **Tests not run** — no broader `side_soundness_suite` or `rv64im_side_spartan_roundtrip` rerun yet; this chunk only revalidated the two parked exact regressions touched by the statement-shape change.
- **Goal 2 delta** — none; this was Nightstream/side-binding cleanup.
- **`spartan2` hits in touched files** — only in the existing side-binding backend `src/nightstream/rv64im/side_relation_spartan.rs`; no new native/compression ownership was introduced.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — continue the Nightstream trust-boundary review on any remaining compact-surfaces claim shells, then return to the five-step terminal-padding Goal 2 blocker.
- **Paper section realized** — HyperNova §6.3 verifier-boundary discipline: the side-binding backend now matches the same published-statement-bound statement digest that the native builder/verifier use, instead of silently carrying an older two-field shell.
- **Poseidon2-only still preserved?** — yes.
- **`vk_fs` still shape-keyed only?** — yes; this chunk did not touch recursion-family ownership.

- Chunk 73 — demote Nightstream linkage metadata and bind side authority to the shared published statement (2026-04-20)
- **Idea** — remove linkage/chunk-summary digest shells from the authoritative Nightstream boundary and prove main-lane ↔ side-lane linkage only through the published statement plus the carried side-opening theorem seam.
- **Files touched** — `src/nightstream/mod.rs`, `src/nightstream/chip8.rs`, `src/nightstream/rv64im.rs`, `src/nightstream/rv64im/authoritative_side.rs`, `src/nightstream/rv64im/build_perf.rs`, `src/nightstream/rv64im/side_runtime_binding.rs`, `src/rv64im/main_proof.rs`, `tests/nightstream.rs`, `tests/chip8_nightstream.rs`, `tests/side_soundness/common.rs`, `tests/side_soundness/positive.rs`, `tests/rv64im_side_opening_native.rs`, `tests/rv64im_side_spartan_roundtrip.rs`, `tests/rv64im_main_proof_surface.rs`, `tests/rv64im_main_proof_side_lane.rs`, `tests/rv64im_spartan2_decider.rs`, `tests/rv64im_final_relation.rs`, `tests/support/rv64im_n2.rs`, `tests/support/perf_rv64im_snapshot.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — authoritative `NightstreamStatement` no longer carries `chunk_summaries`/`linkage_root`; `NightstreamProofBindingInputs` no longer carries `linkage_binding_digest`; RV64IM Nightstream no longer carries `Rv64imLinkageClaims`; `Rv64imCompressedMainProof` no longer carries `linkage_anchor_digest`.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --test nightstream --test rv64im_nightstream_components --test chip8_nightstream --test side_soundness_suite --test rv64im_main_proof_surface --test rv64im_side_opening_native --test rv64im_side_spartan_roundtrip --test rv64im_main_proof_side_lane --test rv64im_spartan2_decider --test rv64im_final_relation`).
- **Tests run** — green focused suite `cargo test -p neo-fold-next --release --test rv64im_nightstream_components rv64im_side_opening_relation_ -- --nocapture`; green ignored exact `cargo test -p neo-fold-next --release --test nightstream rv64im_nightstream_rejects_self_consistent_tampered_public_statement_root_params_id -- --ignored --exact`; green ignored exact `cargo test -p neo-fold-next --release --test side_soundness_suite side_soundness::positive::rv64im_side_soundness_positive_statement_digest_is_recomputed_not_trusted -- --ignored --exact`; green ignored exact `cargo test -p neo-fold-next --release --test chip8_nightstream chip8_nightstream_round_trips_against_current_recursive_seam -- --ignored --exact`.
- **Tests not run** — no broader `nightstream`/`chip8_nightstream`/`side_soundness_suite` rerun yet; exploratory ignored `rv64im_side_binding_roundtrip_with_same_and_rebuilt_vk` still goes red deeper in the parked side-binding debug path after schedule normalization and is not used as passing evidence for this chunk.
- **Goal 2 delta** — none; this was a Nightstream trust-boundary chunk.
- **`spartan2` hits in touched files** — unchanged; this chunk only reduced public-binding ownership and side-binding inputs.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — keep reducing the published Nightstream boundary to theorem-backed facts only, then return to the five-step terminal-padding Goal 2 canary and the remaining parked side-binding/debug test fallout separately.
- **Paper section realized** — HyperNova §6.3 verifier-boundary discipline plus SuperNeo §7 reduction-of-knowledge ownership: linkage is now carried through `(main proof -> published statement)` and `(side-opening theorem seam -> public statement digest)`, not through a self-consistent digest tree.
- **Poseidon2-only still preserved?** — yes.
- **`vk_fs` still shape-keyed only?** — yes; this chunk did not widen the recursion family or change `vk_fs` ownership.

- Chunk 72 — delete the remaining unprovable root-execution/export shell from the carried side-opening statement (2026-04-20)
- **Idea** — finish the Nightstream boundary simplification by deleting the stale root-execution/export validation tail and leaving the carried side-opening statement bound only to the theorem-bearing stage/kernel-opening + main-lane packaged surfaces it actually verifies.
- **Files touched** — `src/nightstream/rv64im.rs`, `src/nightstream/rv64im/compact_surfaces.rs`, `src/nightstream/rv64im/side_opening_relation.rs`, `tests/rv64im_nightstream_components.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — the carried side-opening statement no longer pretends to authorize `root_execution` / `kernel_export_source` via compact digests; the dead root-execution/export digest helpers are removed, and the obsolete kernel-export-source tamper regression is replaced by a main-lane-bridge regression.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --test rv64im_nightstream_components --test nightstream`).
- **Tests run** — green focused suite `cargo test -p neo-fold-next --release --test rv64im_nightstream_components rv64im_side_opening_relation_ -- --nocapture`; green ignored Nightstream regression `cargo test -p neo-fold-next --release --test nightstream rv64im_nightstream_rejects_self_consistent_tampered_public_statement_root_params_id -- --ignored --exact`.
- **Tests not run** — no broader `nightstream` suite rerun yet; the remaining trust-boundary work is deciding what, if anything, beyond the compact carried side-opening theorem seam must still cross the published Nightstream boundary.
- **Goal 2 delta** — none; this was a Nightstream trust-boundary chunk.
- **`spartan2` hits in touched files** — unchanged; no native/compression ownership moved.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — inspect the remaining published Nightstream surfaces for any leftover digest-only trust shell, especially linkage or side-proof binding surfaces that still rely on self-consistent summaries instead of a carried authoritative theorem seam.
- **Paper section realized** — HyperNova §6.3 Construction 2 verifier discipline and SuperNeo §7 reduction-of-knowledge ownership: the carried Nightstream side-opening statement now ends at the compact surfaces it can actually replay/validate, instead of carrying extra digest shells as fake authority.
- **Poseidon2-only still preserved?** — yes.
- **`vk_fs` still shape-keyed only?** — yes; this chunk only narrowed the Nightstream carried boundary.

- Chunk 71 — carry and verify the missing compact export/root-execution side-boundary surfaces (2026-04-20)
- **Idea** — move the missing compact export/root-execution surfaces into the carried Nightstream side-opening statement so verification consumes them directly instead of leaving `kernel_export_source_digest` and `root_execution` authority stranded in the uncarried side bundle.
- **Files touched** — `src/nightstream/rv64im.rs`, `src/nightstream/rv64im/compact_surfaces.rs`, `src/nightstream/rv64im/side_bridges.rs`, `src/nightstream/rv64im/side_opening_relation.rs`, `src/nightstream/rv64im/side_runtime_binding.rs`, `tests/rv64im_nightstream_components.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — the side-opening relation no longer treats `kernel_export_source_digest` and the root-execution digest shell as opaque carried bytes; those compact surfaces are now checked from carried bridge inputs.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --test rv64im_nightstream_components`; `cargo check -p neo-fold-next --release --test nightstream`).
- **Tests run** — green focused suite `cargo test -p neo-fold-next --release --test rv64im_nightstream_components rv64im_side_opening_relation_ -- --nocapture`; green exact roundtrip `cargo test -p neo-fold-next --release --test rv64im_nightstream_components rv64im_side_opening_relation_roundtrips_from_accepted_artifact -- --exact --nocapture`; green ignored Nightstream regression `cargo test -p neo-fold-next --release --test nightstream rv64im_nightstream_rejects_self_consistent_tampered_public_statement_root_params_id -- --ignored --exact`.
- **Tests not run** — no broader `nightstream` suite rerun yet; the remaining open trust-boundary question is still about which theorem-bearing root-execution side objects must cross the published boundary, not this compact-surface check.
- **Goal 2 delta** — none; this was a Nightstream trust-boundary chunk.
- **`spartan2` hits in touched files** — unchanged; no native/compression ownership moved.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — keep drilling on the broader P1 review finding: the verifier now checks the carried export/root-execution compact surfaces, but root execution is still represented by compact summaries and needs an explicit decision about which theorem-bearing side objects must cross the published boundary versus remain under an existing proof.
- **Paper section realized** — HyperNova §6.3 Construction 2 and SuperNeo §7 verifier-boundary discipline: the carried side-opening relation now recomputes theorem-facing compact surfaces instead of trusting self-consistent digest shells.
- **Poseidon2-only still preserved?** — yes.
- **`vk_fs` still shape-keyed only?** — yes; Nightstream boundary tightening did not touch the native recursion family.

- Chunk 70 — bound the full RV64IM public-statement shell into Nightstream proof binding (2026-04-20)
- **Idea** — close the concrete trust-boundary hole where Nightstream verification accepted a self-consistent swapped `Rv64imProofStatement` because only the side-opening runtime used it and `proof_binding_root` did not bind its digest.
- **Files touched** — `src/nightstream/mod.rs`, `src/nightstream/rv64im.rs`, `src/nightstream/rv64im/build_perf.rs`, `src/nightstream/rv64im/verify_perf.rs`, `src/nightstream/chip8.rs`, `tests/nightstream.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — none; this chunk tightened the existing Nightstream binding root instead of widening the proof surface.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --test nightstream`).
- **Tests run** — green exact ignored regression `cargo test -p neo-fold-next --release --test nightstream rv64im_nightstream_rejects_self_consistent_tampered_public_statement_root_params_id -- --ignored --exact`.
- **Tests not run** — no broader Nightstream suite rerun yet; the remaining review finding around root-execution compact surfaces is still open.
- **Goal 2 delta** — none; this was a Nightstream trust-boundary chunk.
- **`spartan2` hits in touched files** — unchanged; no new native/compression ownership was introduced.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — keep drilling on the broader P1 review finding: the verifier now pins the full public-statement shell, but root-execution authority is still reconstructed from compact surfaces and needs an explicit decision about which theorem-bearing side objects must cross the published Nightstream boundary.

- Chunk 69 — narrowed the remaining five-step-cap Goal 2 drift to the live padded recursive-step body (2026-04-20)
- **Idea** — fixed the padded-stage profiler and then used the longer runtime budget to push the short-terminal five-step-cap relation through the live padded FE/NC path until the first real structural mismatches were visible.
- **Files touched** — `src/rv64im/main_relation_circuit/terminal_identity.rs`, `src/rv64im/main_relation_spartan/chunk_step_recursive.rs`, `src/rv64im/main_relation_spartan/nifs_v_stages.rs`, `src/rv64im/main_relation_spartan/chunk_diagnostics.rs`, `tests/f_prime_conformance/fixed_shape_invariance.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — none; this chunk was a diagnostics-and-normalization pass on the live padded family.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --test f_prime_shape_invariance --test perf_rv64im_native --test rv64im_ivc`).
- **Tests run** — red exact canary `cargo test -p neo-fold-next --release --test f_prime_shape_invariance fixed_shape_invariance::f_prime_five_step_cap_terminal_padding_preserves_fixed_shape -- --exact`; green ignored breakdowns `fixed_shape_invariance::f_prime_five_step_cap_terminal_padding_stage_aux_breakdown` and `fixed_shape_invariance::f_prime_five_step_cap_terminal_padding_chunk_replay_aux_breakdown`; red ignored padded profile `fixed_shape_invariance::f_prime_five_step_cap_terminal_padding_padded_stage_profile`.
- **Tests not run** — no broader `f_prime_shape_invariance` suite rerun yet; the remaining blocker is still localized to the five-step terminal-padding family.
- **Goal 2 delta** — the canary is still red, but the first live padded-path drift is now localized instead of opaque:
  - FE initial-sum now uses the real public-step count instead of the padded fresh width
  - canonical padded CCS outputs no longer slide ME outputs into inactive fresh slots
  - inner padded chunk-replay drift is now isolated to `after_pi_ccs` (`9260` aux), and the full recursive-step canary drift is fully inside `after_chunk_replay` (`38334735` vs `38304111`)
- **`spartan2` hits in touched files** — unchanged; this chunk only touched the existing recursive-step ownership.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — close the remaining Goal 2 drift by finishing the live padded `Π_CCS/Π_RLC` ownership: the short-terminal family still differs after `chunk_replay`, and the live padded profiler now reaches `rlc_public`, which points at remaining fresh-gap handling in the recursive-step wrapper.

- Chunk 67 — wire frozen per-proof native `step_cap` through the live native family and unbreak one-shot whole-trace native IVC (2026-04-20)
- **Idea** — replaced the implicit single-step native family with an explicit schedule-derived `step_cap`, widened canonical Construction-2/F' family identity to include it, and padded underfull terminal chunks at the native commitment boundary instead of letting them shrink the family.
- **Files touched** — `src/rv64im/recursion_shape.rs`, `src/rv64im/f_prime.rs`, `src/rv64im/construction2.rs`, `src/rv64im/construction2_default.rs`, `src/rv64im/ivc.rs`, `src/rv64im/ivc_snark.rs`, `src/rv64im/main_proof.rs`, `src/rv64im/main_relation_spartan/recursive_step.rs`, `src/rv64im/mod.rs`, `tests/perf_rv64im_native.rs`, `tests/rv64im_ivc.rs`, `tests/rv64im_recursion_shape.rs`, `tests/rv64im_accumulator_public_statement.rs`, `tests/support/perf_rv64im_ivc_product_surface.rs`, `tests/support/perf_rv64im_snapshot.rs`, `src/bin/rv64im_ivc_closure_probe.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — the live implicit single-step-only native init path and the native recursive-family assumption that the commitment context could shrink to the live witness width.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --lib --bin rv64im_ivc_closure_probe --test perf_rv64im_native --test rv64im_ivc --test rv64im_recursion_shape --test rv64im_accumulator_public_statement`).
- **Tests run** — hot exact `cargo test -p neo-fold-next --release --test rv64im_recursion_shape rv64im_recursion_shape_digest_tracks_step_cap -- --exact`; hot ignored perf `NS_DEBUG_N=5 cargo test -p neo-fold-next --release --test perf_rv64im_native -- --ignored --nocapture rv64im_mixed_opcode_native_ivc_perf_snapshot_whole_trace`; hot ignored perf `NS_DEBUG_N=5 cargo test -p neo-fold-next --release --test perf_rv64im_native -- --ignored --nocapture rv64im_mixed_opcode_native_ivc_perf_snapshot`; hot exact `cargo test -p neo-fold-next --release --test rv64im_ivc rv64im_ivc_base_state_round_trips_through_serde -- --exact`.
- **Tests not run** — no truthful under-cap cold exact coverage on the touched release test targets; first relinks for `perf_rv64im_native` and `rv64im_ivc` still breached the repo’s 10s cap before the hot reruns.
- **Goal 2 delta** — no canary disposition changed yet; dedicated `M > 1` fixed-shape invariance coverage is still open.
- **`spartan2` hits in touched files** — only inside the existing compression subtree; no new native `spartan2` ownership was introduced.
- **Any dead-code / hybrid / fallback introduced?** — no compat flags, no hybrid owner, no new fallback path; the only remaining default single-step wrappers are legacy builder surfaces that still need the schedule-derived sweep.
- **Next chunk idea** — add the real `M > 1` fixed-shape canaries, then sweep the remaining published/audit/backend helper constructors so they derive the same frozen `step_cap` family instead of silently defaulting to `1`.
- **Paper section realized** — HyperNova §6.3 Construction 2 fixed-family ownership: `M` is now a frozen native family parameter, while short terminal chunks are padded structurally instead of changing the recursive family mid-proof.
- **Poseidon2-only still preserved?** — yes.
- **`vk_fs` still shape-keyed only?** — yes; the new `step_cap` is part of the recursion shape / `vk_fs` family digest and nothing else in the public binding path changed hash families.

## Hard Stops
- None.

- Chunk 73 — align stale `Π_RLC` helper/debug paths with the fixed-width compiled family after the live five-step-cap closure (2026-04-20)
- **Idea** — after the live five-step-cap Goal 2 canary went green, sweep the remaining debug/range/helper `Π_RLC` owners so they no longer sampled an active-prefix rho stream and then padded it locally; every path that claims to model the compiled verifier should now sample one fixed-width rho family over the padded CE-output arity, matching HyperNova §6.3 and SuperNeo §7.
- **Files touched** — `src/rv64im/main_relation_spartan.rs`, `src/rv64im/main_relation_spartan/chunk_stage_ranges.rs`, `src/rv64im/main_relation_spartan/chunk_diagnostics.rs`, `src/rv64im/main_relation_spartan/nifs_v_stages.rs`, `src/rv64im/main_relation_spartan/recursive_step/diagnostics.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — the remaining helper/debug split-rho padding paths (`sample active rhos -> append zero/pad shape locally`) are removed from the touched helper surfaces; the live owner had already been moved in the prior chunk.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --test f_prime_shape_invariance --test rv64im_ivc --test perf_rv64im_native`).
- **Tests run** — green exact whole-trace family runtime test `cargo test -p neo-fold-next --release --test rv64im_ivc rv64im_ivc_whole_trace_family_round_trips_and_verifies -- --exact`; green exact multi-step family runtime test `cargo test -p neo-fold-next --release --test rv64im_ivc rv64im_ivc_multi_step_family_survives_serde_and_resume -- --exact`.
- **Tests not run** — I started a redundant cold exact rerun of `fixed_shape_invariance::f_prime_five_step_cap_terminal_padding_preserves_fixed_shape` and a cold whole-trace perf rerun, but stopped them after they spent the budget relinking. The exact five-step-cap canary had already passed before this helper-only sweep, and this chunk did not touch the live verifier body again.
- **Goal 2 delta** — unchanged from the prior live fix: the five-step terminal-padding family remains green, and the stale helper surfaces are now aligned with the same fixed-width `Π_RLC` rule instead of reintroducing a runtime-arity transcript model.
- **`spartan2` hits in touched files** — unchanged ownership; this chunk only normalized the existing recursive-step helper/debug paths.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — run the broader hot validation sweep again from the now-aligned tree (fixed-shape breakdowns + native whole-trace perf) and then do one last paper-faithfulness pass over any remaining trust-boundary surfaces that still look like summaries rather than theorem-backed facts.
- **Paper section realized** — SuperNeo §7.4 `Π_RLC` fixed `K+k` challenge family and HyperNova §6.3 compiled-family discipline: runtime-active slot count no longer forks the transcript model in helper/debug paths either.
- **Poseidon2-only still preserved?** — yes.
- **`vk_fs` still shape-keyed only?** — yes; this chunk only normalized helper/debug replay logic.

- Chunk 74 — remove duplicate native relation retraces from the hot append/verify path (2026-04-21)
- **Idea** — the live native IVC carrier was still rebuilding the same authoritative chunk relation surface multiple times per append: once to recover the verified-step statement, once again to rebuild the authoritative main-circuit trace, and once again to recover the Pi-fold payload. Cut that duplication by treating the authoritative chunk trace as the single source of truth for the native recursive bridge. This stays aligned with HyperNova §6.3 / SuperNeo §7 because the carried summary is still revalidated by the authoritative trace builder before it is used.
- **Files touched** — `src/rv64im/ivc.rs`, `src/rv64im/construction2.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — the append path no longer calls the relation-level Construction-2 retrace helpers just to rebuild artifacts it can derive from the authoritative chunk trace it already owns; the verify path no longer performs a second Construction-2 relation replay just to recover the carried chunk/public digests.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --test rv64im_ivc --test perf_rv64im_native`).
- **Tests run** — no fresh truthful runtime result yet after this optimization patch.
- **Tests not run** — I attempted the hot native perf rerun `NS_DEBUG_N=5 cargo test -p neo-fold-next --release --test perf_rv64im_native -- --ignored --nocapture rv64im_mixed_opcode_native_ivc_perf_snapshot`, but the release test binary never reached execution within the allowed budget because it got stuck in the same full relink path; I killed it instead of treating compile wall time as perf evidence.
- **Goal 2 delta** — none; this was a hot-path ownership/perf cleanup, not a shape-family change.
- **`spartan2` hits in touched files** — unchanged; this chunk stayed entirely in the native pre-compression carrier.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — get one honest hot native perf rerun from the now-slimmer append path and then decide whether the next ROI is inside `evaluate_f_prime` itself or in the release test-binary relink path that is dominating iteration time.
- **Paper section realized** — HyperNova §6.3 compiled verifier-family discipline and SuperNeo §7 reduction ownership: the native carrier now derives its recursive-step bridge artifacts from one authoritative replay trace instead of replaying the same relation surface multiple times under different helper wrappers.
- **Poseidon2-only still preserved?** — yes.
- **`vk_fs` still shape-keyed only?** — yes; this chunk only removed redundant replay/rebuild work in the native carrier.
