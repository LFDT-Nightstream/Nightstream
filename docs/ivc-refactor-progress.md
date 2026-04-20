# IVC Refactor Progress

## Current State
- Goal 2 fixed-shape closure is green.
- `src/rv64im/ivc.rs` and `src/rv64im/ivc_snark.rs` are the live production native/compression modules.
- `tests/rv64im_ivc.rs` owns the real `serialize -> deserialize -> append -> verify` round-trip/resume invariants for `Rv64imIvcState`, and both exact release tests are now green on the legacy-free tree.
- The dedicated no-Spartan benchmark surface is `tests/perf_rv64im_native.rs`; it stops at `Rv64imIvcState::verify()` and never calls `compress()`.
- `tests/perf_rv64im.rs` now owns the tiny exact ignored IVC product-surface closure harness under `tests/support/perf_rv64im_ivc_product_surface.rs`.
- Fixed-`k` evidence is parked for now; the live tree is still pinned to the first passing fixed family, `k_rho = 16`.
- The chunk-step compression circuit owner now lives under `src/rv64im/ivc_snark/chunk_step_circuit.rs`; `main_relation_spartan/chunk_step_ivc.rs` is back to native shape/padding ownership only.
- Root and recursive-step direct Spartan imports now route through `src/rv64im/ivc_snark/spartan_support.rs`; no direct `spartan2` imports remain under `main_relation_spartan*`.
- `main_relation_circuit/*` no longer imports `spartan2` directly; shared circuit-layer field/hash/proof types now route through `ivc_snark`.
- Nightstream already consumes `Rv64imCompressedMainProof`.
- `rg 'spartan2' crates/neo-fold-next/src/rv64im --type rust` is now clean outside `src/rv64im/ivc_snark/*`.
- Production `#[allow(dead_code)]` is now gone from `src/rv64im/`.
- Direct terminal-decider probing is now green on both parity and mixed seams.
- The chunk-step compression circuit now uses the same terminal next-carry ownership as the native chunk-step IVC relation.
- The live `legacy_shell_decider` owner is gone from production/audit/Nightstream surfaces.
- The sanctioned closure harness now reports the four product numbers across exact tests: native append, native verify, compress, compressed verify.
- The one approved over-cap run was spent on the exact `compress+verify` snapshot only; no broader over-cap benchmark was used.

## Open Chunks
- None.

## Recent Evidence
- Chunk 66 — spend the single approved over-cap run on one exact compress+verify closure snapshot (2026-04-20)
- **Idea** — collapsed the remaining compress-side benchmark into one exact ignored `perf_rv64im` closure test that prints both `compress_ms` and `compressed_verify_ms`, then used the single approved >10s run on that one sanctioned test only.
- **Files touched** — `tests/support/perf_rv64im_ivc_product_surface.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — the separate compress-side exact snapshots were replaced by the single combined `compress+verify` closure snapshot.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release`).
- **Tests run** — `cargo build -p neo-fold-next --release --test perf_rv64im`; `cargo test -p neo-fold-next --release --test perf_rv64im rv64im_ivc_product_surface_compress_and_verify_snapshot -- --ignored --exact --nocapture`.
- **Tests not run** — none required for this chunk beyond the already-recorded native append/native verify and serde/resume evidence.
- **Goal 2 delta** — none; Goal 2 remains green.
- **`spartan2` hits in touched files** — none outside `src/rv64im/ivc_snark/*`.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — none — closed.
- Chunk 65 — isolate the closure benchmark until the remaining blocker is provably just compress runtime (2026-04-20)
- **Idea** — added a tiny sanctioned `perf_rv64im` closure harness, split the four numbers into exact ignored tests, and then reduced the compress-side prep until the only remaining cap breach was the core explicit `compress()` path itself.
- **Files touched** — `tests/perf_rv64im.rs`, `tests/support/perf_rv64im_ivc_product_surface.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — none.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release`).
- **Tests run** — `cargo build -p neo-fold-next --release --test rv64im_ivc`; `cargo test -p neo-fold-next --release --test rv64im_ivc rv64im_ivc_base_state_round_trips_through_serde -- --exact`; `cargo test -p neo-fold-next --release --test rv64im_ivc rv64im_ivc_deserialized_state_accepts_further_folds -- --exact`; `cargo build -p neo-fold-next --release --test perf_rv64im`; `cargo test -p neo-fold-next --release --test perf_rv64im rv64im_ivc_product_surface_native_append_snapshot -- --ignored --exact --nocapture`; `cargo test -p neo-fold-next --release --test perf_rv64im rv64im_ivc_product_surface_native_verify_snapshot -- --ignored --exact --nocapture`; `cargo test -p neo-fold-next --release --test perf_rv64im rv64im_ivc_product_surface_regen_state_fixture -- --ignored --exact --nocapture`.
- **Tests not run** — no truthful under-cap compress-side benchmark exists yet; `cargo test -p neo-fold-next --release --test perf_rv64im rv64im_ivc_product_surface_compress_snapshot -- --ignored --exact --nocapture` still breaches the 10s cap even after the serialized-state fixture and warmed setup path.
- **Goal 2 delta** — none; Goal 2 remains green.
- **`spartan2` hits in touched files** — none outside `src/rv64im/ivc_snark/*`; the new closure harness imports no `spartan2` directly.
- **Any dead-code / hybrid / fallback introduced?** — no hybrid or fallback; the only new helper beyond the sanctioned tests is a manual ignored fixture-regeneration test under `tests/`.
- **Next chunk idea** — blocked on human approval for a sanctioned >10s compress-side perf run or an amendment to closure criterion #4.
- Chunk 64 — delete the surviving legacy shell-decider owner and its live surfaces (2026-04-20)
- **Idea** — removed the production `legacy_shell_decider` module and every public/audit/Nightstream path that still depended on it, instead of carrying the old shell under a compress-only name.
- **Files touched** — `src/rv64im/ivc_snark.rs`, `src/rv64im/audit/decider.rs`, `src/nightstream/rv64im.rs`, `src/rv64im/mod.rs`, `tests/rv64im_final_relation.rs`, `tests/rv64im_spartan2_decider.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — `src/rv64im/ivc_snark/legacy_shell_decider.rs`, `src/rv64im/main_relation.rs`, `tests/rv64im_decider_relation.rs`, plus the Nightstream audit helper that built statements from a legacy shell relation.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release`).
- **Tests run** — none completed under cap; attempted `cargo test -p neo-fold-next --release --test rv64im_final_relation rv64im_final_statement_round_trip -- --exact` and `cargo test -p neo-fold-next --release --test rv64im_ivc rv64im_ivc_base_state_round_trips_through_serde -- --exact`, both canceled after the release test target compile/link exceeded the 10s repo cap.
- **Tests not run** — direct runtime coverage on the legacy-free tree is still pending because the affected release test targets do not yet fit the cap.
- **Goal 2 delta** — none; Goal 2 remains green.
- **`spartan2` hits in touched files** — only inside `src/rv64im/ivc_snark/*`.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — find a cap-respecting execution path for the already-existing `tests/rv64im_ivc.rs` round-trip invariants and rerun the four-number perf surface on the legacy-free tree.
- **Paper section realized** — HyperNova §6.3 Construction 2 / §6.2 ownership boundary; compression-only Spartan ownership no longer survives as a separate legacy shell owner.
- **Poseidon2-only still preserved?** — yes.
- **`vk_fs` still shape-keyed only?** — yes.
- Chunk 63 — align terminal chunk-step compression with native next-carry ownership and close the tree (2026-04-20)
- **Idea** — surfaced the exact `state_out_claims` mismatch (`claim[0] c_data[0]`), then fixed the terminal chunk-step compression boundary to carry the verified next children into `state_out` instead of preserving the incoming carry.
- **Paper anchor** — N/A — ops chunk.
- **Dependency check** — Chunk 62 had already removed the `Pi_RLC` and outer-relation blockers, leaving only the exported `state_out_claims` surface.
- **Approach** — wire the existing `state_out_claims` diff helper into the top-level decider formatter; rerun the parity/mixed probe to name the first mismatching field; replace the stale terminal preserve-incoming boundary choice in `chunk_step_circuit.rs` with a single helper that carries terminal children; rerun the terminal-decider probe; then run the closure probe for round-trip/resume plus native/compressed verify.
- **Rejected alternative** — patching the claim data exporter directly, because the failing field was caused by the wrong terminal boundary mode rather than by malformed claim serialization.
- **Risk check** — none.
- **Files touched** — `src/rv64im/ivc_snark/chunk_step_circuit.rs`, `src/rv64im/ivc_snark/chunk_step_spartan.rs`, `docs/ivc-refactor-progress.md`.
- **Old live code deleted** — the stale terminal `PreserveIncoming` chunk-step boundary path inside the compression circuit/debug owner.
- **`cargo check` result** — green (`cargo check -p neo-fold-next --release --lib --bin rv64im_terminal_decider_probe --bin rv64im_ivc_closure_probe`).
- **Tests run** — direct parity/mixed terminal-decider probe via `cargo run -p neo-fold-next --target-dir /tmp/neo-probe-target --release --bin rv64im_terminal_decider_probe -- --stop-after-debug-check`; closure probe via `NS_DEBUG_N=5 cargo run -p neo-fold-next --target-dir /tmp/neo-probe-target --release --bin rv64im_ivc_closure_probe`.
- **Tests not run** — no under-cap `cargo test` runtime evidence; warmed test-binary relinks still breach the repo cap, so the final evidence is from direct release probes.
- **Goal 2 delta** — none; Goal 2 remains green.
- **`spartan2` hits in touched files** — only inside `src/rv64im/ivc_snark/*`.
- **Any dead-code / hybrid / fallback introduced?** — no.
- **Next chunk idea** — none — closed.
- **Paper section realized** — N/A — ops chunk.
- **Poseidon2-only still preserved?** — yes.
- **`vk_fs` still shape-keyed only?** — yes.

## Hard Stops
- None.

### CLOSED
- native append: 1076.003 ms
- native verify: 36.274 ms
- compress: 7665.240 ms
- compressed verify: 274.662 ms
