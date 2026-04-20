# IVC Refactor Progress

## Current State
- Goal 2 fixed-shape closure is green.
- `src/rv64im/ivc.rs` and `src/rv64im/ivc_snark.rs` are the live production native/compression modules.
- Native `Rv64imIvcState` serde round-trip and resume-append tests were green before the current `k_rho` trial; no new evidence has invalidated them.
- The dedicated no-Spartan benchmark surface is `tests/perf_rv64im_native.rs`; it stops at `Rv64imIvcState::verify()` and never calls `compress()`.
- `tests/support/perf_rv64im_snapshot.rs` is still a broad product snapshot and is not valid evidence for native-only IVC latency.
- The live tree is currently pinned to the next working fixed candidate family, `k_rho = 16`, in `src/rv64im/kernel/simple.rs`.
- `k_rho = 14` is disproven on the accepted-artifact path before native IVC starts: `DEC split: Z[19,0] = 20503 (0x5017)` exceeds `B = 2^14 = 16384`.
- `k_rho = 15` is also disproven on the accepted-artifact path before native IVC starts: `DEC split: Z[36,0] = 32823 (0x8037)` exceeds `B = 2^15 = 32768`.
- The first passing fixed-family runtime on the narrowed native probe is now `k_rho = 16`: at `NS_DEBUG_N=5`, `native append = 2941.461 ms` (`490.2435 ms/op`) and `native verify = 18.053 ms` (`3.0088 ms/op`).
- The old hot native-only baseline on the restored `k_rho = 48` tree was `native append = 3257.144 ms` (`542.8573 ms/op`) and `native verify = 34.931 ms` (`5.8218 ms/op`), so the current `k_rho = 16` probe is about a 9.7% append reduction on the same `NS_DEBUG_N=5` mixed-opcode path.
- Nightstream already consumes `Rv64imCompressedMainProof`; the remaining closure gate is still the wider `spartan2` circuit substrate outside `ivc_snark`.

## Open Chunks
- Add a narrow accepted-artifact probe that records the maximum absolute DEC parent entry on representative RV64IM fixtures, so minimal fixed `k` is derived from evidence instead of guesswork.
- Decide whether `k_rho = 16` is the permanent RV64IM family or whether a still-smaller fixed family remains viable on broader fixture coverage.
- After the fixed-`k` question is settled, return to the remaining closure gate: move the residual `spartan2` circuit substrate under `ivc_snark`.

## Recent Evidence
- Chunk 49 — `k_rho = 16` is the first passing fixed family on the native-authoritative mixed-opcode probe (2026-04-19)
- Idea: after `k = 15` failed by a narrow margin, moved the live RV64IM family to `k_rho = 16`, kept the tree Poseidon2-only and non-adaptive, and added a tiny `src/bin/rv64im_native_ivc_probe.rs` runner so accepted-artifact + native-IVC evidence can be gathered without paying the `libtest` relink tax.
- Files touched: `src/rv64im/kernel/simple.rs`, added `src/bin/rv64im_native_ivc_probe.rs`, `docs/ivc-refactor-progress.md`.
- Old live code deleted: the invalid `k_rho = 15` / `B = 2^15` pin in `src/rv64im/kernel/simple.rs`.
- `cargo check` result: green (`cargo check -p neo-fold-next --release --bin rv64im_native_ivc_probe --test perf_rv64im_native --test rv64im_ivc --test f_prime_shape_invariance`).
- Tests run: authoritative probe via `cargo run -p neo-fold-next --release --bin rv64im_native_ivc_probe` at `NS_DEBUG_N=5`, `NS_DEBUG_N=1`, and `NS_DEBUG_N=0`; all passed on the live `k_rho = 16` tree.
- Tests not run: the original `perf_rv64im_native` exact test target still was not used as runtime evidence for this chunk because fresh relinks crossed the 10s cap; broader accepted-artifact fixture coverage still pending.
- Goal 2 delta: none; Goal 2 remains green.
- `spartan2` hits in touched files: none.
- Any dead-code / hybrid / fallback introduced?: no.
- Next chunk idea: instrument the accepted-artifact path to measure the maximum absolute DEC parent entry across representative RV64IM fixtures and verify whether `16` is truly minimal.

- Chunk 48 — move the live tree to fixed candidate `k_rho = 15` (2026-04-19)
- Idea: replaced the restored `k_rho = 48` pin in `src/rv64im/kernel/simple.rs` with the next fixed candidate family `k_rho = 15`, since the live `k = 14` trial failed at `20503` and `2^15 = 32768` is the immediate fixture-local next bound.
- Files touched: `src/rv64im/kernel/simple.rs`, `docs/ivc-refactor-progress.md`.
- Old live code deleted: the temporary restored `k_rho = 48` / `B = 2^48` parameter pin in `src/rv64im/kernel/simple.rs`.
- `cargo check` result: green (`cargo check -p neo-fold-next --release --test perf_rv64im_native`, `cargo check -p neo-fold-next --release --test rv64im_ivc`, `cargo check -p neo-fold-next --release --test f_prime_shape_invariance`).
- Tests run: authoritative probe evidence now exists indirectly via the follow-up chunk: the same live `k_rho = 15` family fails immediately on the mixed-opcode path with `DEC split: Z[36,0] = 32823 (0x8037)`.
- Tests not run: fresh exact `perf_rv64im_native` runtime evidence under the repo cap.
- Goal 2 delta: none; Goal 2 remains green.
- `spartan2` hits in touched files: none.
- Any dead-code / hybrid / fallback introduced?: no.
- Next chunk idea: move to the next fixed candidate family and keep only runtime-backed candidates live.

- Chunk 47 — restore the last known working RV64IM family after the failed `k = 14` experiment (2026-04-19)
- Idea: restored `src/rv64im/kernel/simple.rs` back to `k_rho = 48` / `B = 2^48` so the live RV64IM path was valid again while the minimal fixed `k` was derived from evidence.
- Files touched: `src/rv64im/kernel/simple.rs`, `docs/ivc-refactor-progress.md`.
- Old live code deleted: the invalid live `k = 14` parameter pin from `src/rv64im/kernel/simple.rs`.
- `cargo check` result: green (`cargo check -p neo-fold-next --release --test perf_rv64im_native`, `cargo check -p neo-fold-next --release --test rv64im_ivc`, `cargo check -p neo-fold-next --release --test f_prime_shape_invariance`).
- Tests run: exact hot `NS_DEBUG_N=5 cargo test -p neo-fold-next --release --test perf_rv64im_native -- --ignored --nocapture rv64im_mixed_opcode_native_ivc_perf_snapshot` passed in `4.00s`.
- Tests not run: none for this chunk after the rebuild completed.
- Goal 2 delta: none; Goal 2 remains green.
- `spartan2` hits in touched files: none.
- Any dead-code / hybrid / fallback introduced?: no.
- Next chunk idea: add a narrow accepted-artifact probe that records the maximum absolute DEC parent entry on live RV64IM fixtures.

- Chunk 46 — `k = 14` fails before native IVC begins (2026-04-19)
- Idea: the first post-switch native perf rerun proved the live `k = 14` family is invalid for RV64IM fixture prep itself: `prove_rv64im_accepted_proof_with_options(...)` failed with `DEC split: Z[19,0] = 20503 is out of range for k_rho=14, b=2`.
- Files touched: `docs/ivc-refactor-progress.md`.
- Old live code deleted: none.
- `cargo check` result: N/A — evidence-only update.
- Tests run: exact `NS_DEBUG_N=5 cargo test -p neo-fold-next --release --test perf_rv64im_native -- --ignored --nocapture rv64im_mixed_opcode_native_ivc_perf_snapshot` on the live `k = 14` tree; corroborated by `NS_DEBUG_N=0` and `NS_DEBUG_N=1`.
- Tests not run: none for this evidence update.
- Goal 2 delta: none; Goal 2 remains green.
- `spartan2` hits in touched files: none.
- Any dead-code / hybrid / fallback introduced?: no.
- Next chunk idea: try the immediate next fixed candidate family rather than staying parked on `48`.

- Chunk 42 — add a dedicated no-Spartan native IVC perf target (2026-04-19)
- Idea: added `tests/perf_rv64im_native.rs` so native-latency questions can be answered without entering `compress()` or the Nightstream/public-proof stack.
- Files touched: added `tests/perf_rv64im_native.rs`, `docs/ivc-refactor-progress.md`.
- Old live code deleted: none.
- `cargo check` result: green (`cargo check -p neo-fold-next --release --test perf_rv64im_native`).
- Tests run: exact hot `NS_DEBUG_N=1 cargo test -p neo-fold-next --release --test perf_rv64im_native -- --ignored --nocapture rv64im_mixed_opcode_native_ivc_perf_snapshot` passed in `1.57s`.
- Tests not run: larger opcode counts were not yet run at the time of that chunk.
- Goal 2 delta: none; Goal 2 remains green.
- `spartan2` hits in touched files: none.
- Any dead-code / hybrid / fallback introduced?: no.
- Next chunk idea: use the native-only probe to measure fixed-`k` family experiments directly.

## Hard Stops
- None.
