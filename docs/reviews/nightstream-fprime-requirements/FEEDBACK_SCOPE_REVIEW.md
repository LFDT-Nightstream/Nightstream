# Folding closure feedback review

Source reviewed: `708431f5f8e33fec89669f5ada9876c19ab0d5b6`. The supplied
review used the preceding staged snapshot. Its Lean and Rust implementation
files match the pushed commit; the commit also contains supporting documents.
This follow-up changes map wording, references and dependencies only.

## Decisions

1. **Qualify the large PiCCS run.** At `2^28`, the common optimized protocol
   driver, transcript code and both Rust verifiers ran with the fixture's
   bounded evaluator and prepared complete ring openings. The default
   `OptimizedPaperJointOracle` was compared on a `2^7` case. Its 34 full
   extension-field tables would use 136 GiB at `2^28`, before other storage.
   `C.connection.native` now states this scope. `P.runtime.piccs` remains open
   for the full-profile production evaluator and caller.

2. **The feature gate is not required by this conformance contract.**
   `prove_with_complete_oracle` is public and its current callers are the
   fixture binary and its integration test. It accepts no structure cache;
   the cached route still calls `cache.validate_structure`. Both routes use
   `pi_ccs_joint_protocol::bind_and_sample_with_trace`, whose
   `_expected_matrix_digest` argument is unused on both routes. The new
   entry point therefore did not remove a matrix transcript binding.
   The driver keeps its common shape, transcript, challenge, round, complete
   output and terminal checks. The selected loader and independent row and
   opening checks establish the fixture's source/matrix connection.
   Architecture section 11 permits different prover evaluators. The helper
   does not add another verifier relation. This review adds no feature and
   does not duplicate or move the common driver.

3. **Separate the local PiDEC consumer from history extraction.**
   `accepted_outputWitness_extracts_parent` consumes valid output witnesses
   for any accepted fold. `recursiveTerminal_supplies_childOpenings` supplies
   them at the terminal step. Obtaining witnesses for interior steps remains
   open in `H.security.reverse_recursive`. The dependency points from that
   history requirement to the local PiDEC consumer and terminal openings.
   It does not point from the proved PiDEC theorem back to history extraction.
   The D proof denominator changed from 25 to 26 when this conditional
   consumer became proved instead of being excluded as an assumption.

4. **Close only the matching H contract.** `H.recursive.fold_exact` is now
   proved and connected through the actual parent, attempt, check and output
   theorems. `N.local.actual_step` and `H.compat.decode_arbitrary` remain
   partial/open because their contracts include verifier-selected context.
   They now cite `ActualPiDECOutput.selectedRowsAndPublic_imply_step` and
   identify context selection as the remaining gap.

5. **Pinned-package tests still exist.**
   `crates/neo-fold-clean/tests/f_prime/package_production.rs:40` contains
   `production_package_pins_relation_and_key`.
   `crates/nightstream-fprime/tests/production_binding.rs:245` contains
   explicit comparisons with the verifier-owned pins. These are separate
   from the passing candidate checks. The pinned artifact is still a
   134-byte LFS pointer locally. `L.export.current` remains open and now
   explains this distinction.

6. **No unrelated cleanup or history rewrite.** The recorded sampler-column
   debt and file-size limits remain separate from these map corrections.
   The implementation commit was already pushed at the user's request;
   the supplied suggestion to split it does not authorize rewriting it.

## Source checks

- Native route: `crates/neo-reductions/src/engines/optimized_engine/paper_joint.rs:552`
  and `:839`; fixture caller at
  `crates/neo-fold-clean/src/bin/generate_pi_ccs_fixture/native_driver.rs:264`.
- Common binding: `crates/neo-reductions/src/engines/pi_ccs_joint_protocol.rs:171`.
- Exact step: `formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualPiDECOutput.lean:153`.
  Its check/output declarations and the parent/message declarations are
  registered in `formal/nightstream-fprime/tests/AxiomsStage1PiRLCExport.lean`.
- Child consumer: `formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/PiDEC/OutputWitnessConsumer.lean:136`
  and `formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/OutputWitnessConsumer.lean:109`.

Two parallel source reviews confirmed these scope and dependency decisions.
The earlier full Lean/axiom and 91 native conformance results retain their
stated scopes. This documentation change does not rerun or broaden them.
The website build and its seven export tests validate the changed map.
