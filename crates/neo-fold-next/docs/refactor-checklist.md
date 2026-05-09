# neo-fold-next Refactor Checklist

## Current Focus

| Field | Value |
|---|---|
| Active area | `frontends/rv32im/kernel` |
| Active chunk | Kernel proof/opening ownership audit |
| Status | `in progress` |
| Latest verification | `cargo test -p neo-fold-next --release --test rv32im_ivc_architecture` |
| Next action | Map kernel proof/opening modules, then split the highest-risk broad file by flow, data, circuit, and measurement ownership. |

## Progress By Area

| Area | Bucket | Status | Done | Remaining | Risk | Latest Verification |
|---|---|---|---|---|---|---|
| `frontends/direct_ccs` | `frontend_adapter` / `orchestrator` / `circuit` | `in progress` | Split API, state, F' source/R1CS, recursive state, terminal committed circuit, public image, low-norm adapter, and Construction-2 fold surfaces. | Audit remaining public surface and constructor placement. | Medium | `cargo test -p neo-fold-next --release --test direct_ccs_ivc` |
| `core/proof/perf` | `measurement/perf` | `done` | Split proof timing into prove and verify owners. | None known. | Low | `cargo test -p neo-fold-next --release --test finalized_proof` |
| `circuit/superneo/transcript` | `transcript` | `done` | Split absorb, squeeze, snapshot, and hash helpers. | None known. | Low | `cargo check -p neo-fold-next --tests` |
| `decider/spartan2` | `proof_boundary` / `circuit` | `done` | Split monolithic decider into relation, packing, public-target shell, backend-binding shell, and decider flow. | SNARK shell tests remain ignored by test policy; contract tests run explicitly. | Medium | `cargo test -p neo-fold-next --release --test spartan2_backend_contract -- --ignored` |
| `public_proof/rv32im` | `proof_boundary` | `done` | Grouped published proof build/verify timing under `flow/{build,verify,perf}.rs`; moved side-opening relation statement/witness types into `side_opening_relation/types.rs`; split side-opening Spartan debug and setup-witness helpers into owned files; moved side-bridge data contracts into `side_bridges/types.rs`; split side-binding Spartan debug/setup helpers; moved Phase 0 eval-claim data contracts into `side_eval_claim_relation/types.rs`. | None known for this pass. | Low | `cargo test -p neo-fold-next --release --test rv32im_ivc_architecture` |
| `frontends/rv32im/kernel` | `orchestrator` / `proof_boundary` | `mapped` | Large-file list identified. | Split broad kernel proof/opening modules by flow, witness/data, verification, and measurement ownership. | Medium | `cargo check -p neo-fold-next --tests` |

## Protocol Ownership Map

| File/Folder | Bucket | Current Responsibility | Problem | Destination/Owner | Risk | Red Flags |
|---|---|---|---|---|---|---|
| `frontends/direct_ccs` | `frontend_adapter` / `orchestrator` | Direct CCS frontend lowering, F' source, terminal and recursive proof composition. | Remaining surface still needs audit after structural split. | `frontends/direct_ccs/*` by ownership. | Medium | Possible public re-export drift. |
| `decider/spartan2` | `proof_boundary` / `circuit` | Spartan2 decider relation and shells. | Split completed; SNARK shell tests are intentionally ignored. | `decider/spartan2/*`. | Medium | Ignored proof-heavy tests. |
| `public_proof/rv32im` | `proof_boundary` | Published RV32IM proof boundary, side openings, verification helpers, audit surfaces. | Split completed for this pass; largest file is under 1,100 lines and ownership paths are explicit. | `public_proof/rv32im/*` grouped by proof boundary ownership. | Low | None known. |
| `frontends/rv32im/kernel` | `orchestrator` / `proof_boundary` | RV32IM kernel proving, accepted-artifact verification, stage opening packages, and public proof assembly. | Several broad files are near the 1,500-line cap and mix proof flow, witness assembly, verification, and perf data. | `frontends/rv32im/kernel/*` grouped by kernel proof/opening ownership. | Medium | Near-cap files, mixed proof/perf/witness surfaces. |

## Verification Log

| Date | Command | Result | Notes |
|---|---|---|---|
| 2026-05-08 | `cargo fmt --all` | pass | After `decider/spartan2` split; rustfmt warned that `imports_granularity = Crate` is nightly-only. |
| 2026-05-08 | `cargo check -p neo-fold-next` | pass | Library compile after `decider/spartan2` split. |
| 2026-05-08 | `cargo check -p neo-fold-next --tests` | pass | Test target compilation after `decider/spartan2` split. |
| 2026-05-08 | `cargo test -p neo-fold-next --release --test spartan2_public_target_shell` | pass | Target compiled; 3 tests are ignored by current Spartan-path policy. |
| 2026-05-08 | `cargo test -p neo-fold-next --release --test spartan2_backend_binding_shell` | pass | Target compiled; 14 tests are ignored by current Spartan-path policy. |
| 2026-05-08 | `cargo test -p neo-fold-next --release --test spartan2_backend_contract` | pass | Target compiled; 11 tests are ignored by default. |
| 2026-05-08 | `cargo test -p neo-fold-next --release --test spartan2_backend_contract -- --ignored` | pass | 11 passed; covers cheap backend public-IO/shape contract behavior. |
| 2026-05-08 | `cargo test -p neo-fold-next --release --test rv32im_ivc_architecture` | pass | 26 passed after moving RV32IM published verifier flow into `public_proof/rv32im/flow`. |
| 2026-05-08 | `cargo check -p neo-fold-next --tests` | pass | Test target compilation after `public_proof/rv32im/flow` split. |
| 2026-05-08 | `cargo check -p neo-fold-next` | pass | Library compile after moving `side_opening_relation` into `mod.rs` plus `types.rs`. |
| 2026-05-08 | `cargo check -p neo-fold-next --tests` | pass | Test target compilation after moving `side_opening_relation` into `mod.rs` plus `types.rs`. |
| 2026-05-08 | `cargo check -p neo-fold-next` | pass | Library compile after moving `side_opening_spartan` into `mod.rs`, `debug.rs`, and `witness_setup.rs`. |
| 2026-05-08 | `cargo check -p neo-fold-next --tests` | pass | Test target compilation after moving `side_opening_spartan` into `mod.rs`, `debug.rs`, and `witness_setup.rs`. |
| 2026-05-08 | `cargo check -p neo-fold-next` | pass | Library compile after moving `side_bridges` into `mod.rs` plus `types.rs`. |
| 2026-05-08 | `cargo check -p neo-fold-next --tests` | pass | Test target compilation after moving `side_bridges` into `mod.rs` plus `types.rs`. |
| 2026-05-08 | `cargo check -p neo-fold-next` | pass | Library compile after moving `side_relation_spartan` into `mod.rs`, `debug.rs`, and `setup_witness.rs`. |
| 2026-05-08 | `cargo check -p neo-fold-next --tests` | pass | Test target compilation after moving `side_relation_spartan` into `mod.rs`, `debug.rs`, and `setup_witness.rs`. |
| 2026-05-08 | `cargo test -p neo-fold-next --release --test rv32im_ivc_architecture` | pass | 26 passed after public-proof RV32IM side module splits. |
| 2026-05-08 | `cargo check -p neo-fold-next` | pass | Library compile after moving `side_eval_claim_relation` into `mod.rs` plus `types.rs`. |
| 2026-05-08 | `cargo check -p neo-fold-next --tests` | pass | Test target compilation after moving `side_eval_claim_relation` into `mod.rs` plus `types.rs`. |
| 2026-05-08 | `cargo test -p neo-fold-next --release --test rv32im_ivc_architecture` | pass | 26 passed after completing the `public_proof/rv32im` pass. |
