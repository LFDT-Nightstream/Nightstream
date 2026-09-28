# Production path audit for the proposed goal splits

Reviewed commit: `4fc02857c3aa4207f3290739cc62d794ba86f5f9`. Read-only source review and focused tests. This file covers the implementation work that remains in Goal B of proposals 1, 2, 3, and 4.

## Result

Useful production components exist, but the selected Stage 1 package is not connected to the public lifecycle. The remaining work is more than replacing a file path or choosing a proof backend.

The current public lifecycle and its terminal checks must not be counted as integration of the selected package merely because both exist in this repository. They consume different relation interfaces.

## Traced boundaries

| Boundary | Current code | Exact remaining connection |
|---|---|---|
| Load selected package | `Poseidon2HashChainV1Package::load` calls the strict sealed-package loader and retains its structure and verifier binding. [Adapter](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/stage1/mod.rs:24), [strict loader](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/nightstream-fprime/src/package/sealed.rs:496). | The stored artifact and pins still precede the shared-value repair. The local artifact is a Git LFS pointer. A validated current artifact and matching pins are required before this can supply the selected production path. |
| Matrix authority | The adapter supplies a matrix-content-free `Structure` and `visit_matrix_rows`. [Header contract](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-ccs/src/relations.rs:106), [adapter methods](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/stage1/mod.rs:56). | The selected rows must feed the actual prover/verifier evaluator. Passing this header to ordinary preprocessing is insufficient: ordinary cache construction cannot derive matrix content from a header. |
| Optimized preprocessing | Ordinary `preprocess_shared` builds its optimized cache from the structure. A separate path can take a `VerifiedSuperneoCacheArtifact`. [Ordinary path](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/lifecycle/mod.rs:650), [artifact path](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/lifecycle/mod.rs:676). | Connect the Stage 1 matrix source, evaluator/cache authority, dimensions, setup, and verifier binding. Existing cache support can be reused; there is no Stage 1 caller that does this. [Header rejection in cache construction](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-reductions/src/superneo_eval/cache.rs:296), [verified-cache constructor](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-reductions/src/engines/optimized_engine/mod.rs:120). |
| PiCCS proof input | `PiCcsV1_1ProofInputs::from_proof` checks the fixed source, commitment, round, and separate evaluation-family dimensions. `into_package_inputs` adds lifecycle fields and the verifier context. [Bridge](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/stage1/inputs.rs:45). | The production lifecycle must use this conversion on the same actual fold and state that it verifies. Its existence does not prove the caller's state or context is authoritative. |
| State serialization | The Stage 1 serializer writes the context, counter, application state, full ordered running data, and counter field into the Lean-defined state preimage. [Serializer](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/stage1/inputs.rs:152). | Connect actual lifecycle inputs and outputs to this serializer and to the fresh public digest. The current lifecycle's compact state vocabulary is not evidence of this equality. |
| PiDEC proof input | `PiDecV1_1PackageInputs::new` validates 16 child commitment, public-input, Eval_K, and Eval_A arrays. [Input type](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/nightstream-fprime/src/package/v1_1.rs:242). | No production source call to this constructor was found in `neo-fold-clean/src`. The actual native PiDEC output must be converted and connected to the selected witness program. |
| Physical and logical assignment | The public Stage 1 adapter executes the package witness program. The lower-level loaded package separately provides `execute_logical_assignment`. [Physical execution](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/stage1/mod.rs:70), [logical transport](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/nightstream-fprime/src/package/sealed.rs:291). | Wire physical execution, logical transport, commitment, and exact selected CCS evaluation into one production fold. The public wrapper does not currently perform that sequence. |
| Recursive induction capability | General preprocessing starts with `f_prime_recursive_link = false` and `terminal_induction = false`. [Defaults](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/lifecycle/mod.rs:839). | Enable the selected package's induction only through the proved complete F′ relation and its validated implementation connection. A caller-selected flag is not a substitute. |
| Existing native-CCS frontend | `LeanNativeCcsPreprocessing::new` consumes a distinct `LeanNativeCcsManifest`, emits its relation, and enables terminal induction. [Frontend](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/frontends/r1cs_f_prime/native_ccs.rs:53), [manifest format](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/frontends/r1cs_f_prime/lean_native_ccs_manifest.rs:34). | This is not a caller of `Poseidon2HashChainV1Package`. Reuse of generic execution code is possible, but these relation formats must not become competing production authorities. |
| Terminal semantics | Lean already defines the outer terminal checks: all 16 running CE openings and the fresh CCS opening. [Exact selected relation](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PerApplicationTerminal.lean:35). | Connect the actual accepted terminal proof to those checks for the same selected package. There is no requirement to add another NIFS fold inside F′. |
| Terminal metadata | Lean installs metadata without adding rows. Rust validates its exact row range and 16/1 claim counts. [Lean metadata](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/TerminalPackage.lean:20), [Rust validation](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/nightstream-fprime/src/package.rs:948). | The Stage 1 adapter does not expose a terminal acceptance operation. Metadata validation is not a complete terminal proof verifier. |
| Existing lifecycle verification | `verify_uncompressed` checks preprocessing authority, state anchors, non-replay scope, and terminal relations. [Verifier entry](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/src/lifecycle/verify.rs:229). | Route it through the selected package or integrate the selected package into the required production verifier. Existing successful lifecycle tests on other relations do not close this connection. |
| Final proof backend | The owner goal requires a separately approved production `prove → verify` run. [Outcome](/Users/nicarq/starstream/develop/nightstream-clean-up/FPRIME_STAGE1_GOAL.md:101). | No backend is approved for this cut. Existing Spartan-related code is not approval and is not evidence of a run on the selected package. |

## Caller search

The source search for `Poseidon2HashChainV1Package` in `crates/neo-fold-clean/src` finds its declaration/implementation and the crate-root re-export. It finds no lifecycle caller. Broader searches for the strict package loader and `Stage1VerifierBinding` find the lower-level package implementation and the same adapter.

The test named `f_prime_package_production` checks package pins, one matrix-row visit, and package mutations. Its body never invokes the public lifecycle's `prove`, `extend`, or terminal verifier. The word `production` in that test name must not be used as evidence of an integrated proof run. [Test body](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/tests/f_prime/package_production.rs:41).

## Executed package test

Executed from the repository root:

```sh
RUSTC_WRAPPER="" PATH="/opt/homebrew/bin:$PATH" timeout -s KILL 300 \
  cargo test --locked --offline -p nightstream-fprime --release \
  --test package_loader -- --nocapture
```

Result: build completed in 22.02 s; tests finished in 0.01 s; 1 passed and 7 failed. Every failure reaches or depends on parsing the stored package, which is a 134-byte Git LFS pointer in this checkout. This does not establish seven loader defects. It establishes that the package-dependent tests cannot validate the stored package here.

The source of `sealed_package_builds_the_package_owned_logical_relation_header` still asserts 264,627,433 logical columns. The current Lean candidate proves 254,260,583 logical columns and 254,260,620 aligned carrier coordinates. The expected pins and package-dependent assertions must be checked when a replacement artifact is authorized. [Old test expectation](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/nightstream-fprime/tests/package_loader.rs:54), [current dimensions](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/CONSTRAINT_TREE.md:418), [stored pins](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/nightstream-fprime/src/identity.rs:38).

No artifact was regenerated or substituted in this review. No production code, pin, source policy, or goal file was changed.

## Effect on the proposals

- Proposal 1 can defer these circuit/package integration tasks to B, but A must define its native input-binding contract and close its own executable conformance.
- Proposal 2 can close phase conformance without this complete lifecycle integration. It must use the exact phase scope and cannot claim production Stage 1 completion.
- Proposal 3 closes the difficult selected SuperNeo circuit proof in A, while these lifecycle and final-acceptance connections remain in B.
- Proposal 4 puts the complete Lean proof in A. B still contains a real implementation integration project, not only a backend command.
