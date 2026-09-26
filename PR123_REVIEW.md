**PR #123 — implementation, protocol, and migration review**

Review date: September 26, 2026. [Pull request](https://github.com/LFDT-Nightstream/Nightstream/pull/123).

Reviewed head: **99b3b095366f7c8328de44d828e14c0dce6a571d**. Base: **7f51e1010ce382d15206d4d1fabcb27d88754cfe**, on the feature branch named nico/f-prime-constraints-cuda-formal. This is a review against that base, not against main.

Scope: **62 commits, 821 changed files, 71,839 added lines, and 10,954 removed lines**. The first 59 commits were reviewed at c5595d978. Three commits arrived during the review; I reviewed those changes too. They change rules, comments, and review records. The Rust implementation, Lean proof source, and test vectors used by my implementation checks are unchanged between these two heads.

**Decision**

I recommend **requesting changes before merge**, chiefly because the migration does not yet meet your requirement for one maintained path without deprecated support.

The wide sampler is connected to the selected runtime. This is not merely an unused candidate: the selected package, native sampler, context binding, and main security targets use it. My public API test completed a base proof, a recursive fold, terminal verification, and rejection of a changed endpoint on Metal.

However, the branch retains an executable old emitter route, a deprecated crate required by current validation, two application assembly routes, and a direct-witness route that the public prover does not use. There is also a concrete mutation-test indexing error, and CI omits several cheap tests for the code this PR changes.

I found **no demonstrated false-acceptance attack against the selected wide package**. This report is not a certification of all 821 files or a proof of concrete end-to-end cryptographic security. The findings below distinguish defects, architectural conflicts with your requested outcome, and limits of the evidence.

**How the review was done**

I read the commit subjects and bodies and examined the changes by implementation phase. I traced the final selected path through application assembly, package decoding, witness construction, PiCCS/PiRLC/PiDEC, terminal verification, and the corresponding Lean claims. For the many repetitive Lean edits, I checked the selected ownership boundaries, theorem premises, dependency changes, and final consumers; I did not manually rederive every proof term.

GitHub returned **zero PR discussion comments, zero inline review threads, zero submitted reviews, and zero commit comments on these commits**. Thus there is no hidden list of GitHub comments to reconcile. The relevant review feedback is in the checked-in review records and commit bodies. The commit ledger later in this report accounts for all 62 commits.

I did not modify the implementation, create a commit, submit a GitHub review, or dispatch other agents.

**Findings**

| ID | Finding | Effect |
|---|---|---|
| F1 | The old emitter remains executable beside the wide emitter | The migration leaves two package-generation routes |
| F2 | Current validation still requires the deprecated Rust crate | Deprecated support remains an active maintenance obligation |
| F3 | The selected application has a separate assembly specialization | There are two application lowering routes, with different layouts and proof coverage |
| F4 | Public prove/extend bypass the new direct CCS witness route | The claimed integration does not reach the normal public call path |
| F5 | A new application mutation test uses the wrong coordinate space | The test does not establish the application-binding property described by its comment |
| F6 | CI omits the new sampler and matrix/transport tests | The green CI result has narrower coverage than this PR's central changes |

F1–F3 are merge-scope findings under your stated preference for a single maintained path. F4 is an implementation and maintenance defect, and F5–F6 are verification gaps. None is presented as a proved cryptographic forgery.

**F1. Complete the emitter replacement; do not retain the old default**

The new wrapper intercepts only the two explicit Poseidon2 hash-chain flags. Every other argument still goes to the old Main.run. In particular, the supported validate.sh emit and emit-expanded commands still reach the earlier package constructors. Old per-application emitters also remain exposed and are called by reference executables.

Evidence: [new dispatch](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/formal/nightstream-fprime/NightstreamFPrime/Export/Entrypoint.lean#L8); [old per-application emitter](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/formal/nightstream-fprime/NightstreamFPrime/Export/Main.lean#L1013); [old default dispatch](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/formal/nightstream-fprime/NightstreamFPrime/Export/Main.lean#L1080); [retained commands](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/formal/nightstream-fprime/scripts/validate.sh#L213).

This is an actual reachable alternate route, rather than a name that happens to contain v1_1. The old Main module still serializes the earlier assignment transport and First54-related data. A maintainer can therefore successfully select an obsolete generator while following a command still exposed by the current validation script. The current sealed Rust loader rejects the retired transport schema, so the resulting problem is divergent tooling and incompatible artifacts; I did not establish that the selected verifier accepts an old package.

The Rust source also still contains the old First54 compact-plan expansion. Its continued presence should be resolved with the old generator/loader surface, rather than described as a completed deletion. See [RawFirst54Block](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream-fprime/src/package/plan.rs#L18) and [expand_first54](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream-fprime/src/package/plan.rs#L90).

Recommended change: make the maintained emitter own the normal command, remove obsolete command branches and their obsolete consumers, and retain only the shared serialization primitives that the wide emitter actually needs. Do not add a selector for old/new packages. Do not delete every module with v1_1 in its path: PiCCS and PiDEC reuse unchanged semantics, and some old layout lemmas currently support the wide remapping proof.

Closing evidence: every documented package-generation command reaches the same selected builder; old package schemas fail explicitly; the selected package identity and exact matrix/assignment checks still pass.

**F2. Remove the current validation dependency on neo-fold-legacy**

The new runtime itself is separated correctly. I checked the normal dependency tree of nightstream with both metal and cuda features: **neo-fold-legacy is absent**.

The repository as a whole is not separated. The workspace still includes neo-fold-legacy; neo-wasm depends on it, GPU crates expose legacy adapters, and the constraint-minimizer bridge depends on it. More directly relevant to this PR, the current golden coordinator builds generate_pi_ccs_fixture from neo-fold-legacy, and the retained-vector reproduction instructions require that executable.

Evidence: [current checker build](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/scripts/golden_conformance_ci.py#L35); [golden reproduction instructions](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/docs/reviews/nightstream-fprime-requirements/golden-conformance-wide/README.md#L75); [WASM dependency](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/neo-wasm/Cargo.toml#L20); [workspace membership](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/Cargo.toml#L13).

This deprecated crate predates the PR. The PR did not create the entire legacy track, but it continues to maintain it: nine legacy files change, including its sampler adapter, generator, fixtures, and tests. The test repair in 48c8e0b99 also spends effort on the legacy Metal adapter. The new golden replay is thus still tied to a deprecated implementation surface.

Recommended change: move the smallest current-package checker into maintained test tooling, using the maintained verifier/reduction code. Remove the dependency on the deprecated lifecycle. Remove or port the remaining supported consumers according to the product scope; do not repair deprecated APIs just to keep them buildable.

The selected runtime's independence is a success to preserve. Current conformance tooling should have the same property.

**F3. The PR introduces a selected-application route beside generic assembly**

Assembly first materializes an application plan and compares it with the stored selected Poseidon2 plan. Exact equality takes a compact specialization branch. Every other application restores an ordinary reference and then rewrites matrix, assignment, and source coordinates.

Evidence: [the two branches](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream/src/assembly/mod.rs#L75); [selected-plan recognition](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream/src/assembly/application.rs#L60); [ordinary reference restoration](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream/src/assembly/connect.rs#L155); [both manifest forms](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/formal/nightstream-fprime/NightstreamFPrime/Export/SharedVerifier.lean#L411).

This is a deliberate optimization, not old-sampler compatibility. The exact rows, recipes, ports, and counts are compared, and reference authorization runs independently. I found no evidence that shape-only matching can substitute a different application. Both assembly variants passed the current tests.

It nevertheless creates the kind of parallel implementation you are concerned about. The two branches have different application matrix blocks, retained coordinates, assignment plans, and relocation work. The selected Lean application has the strongest exact prepared-package correspondence. Generic Rust assembly remains an explicit trusted implementation link; the manifest itself states that it does not prove arbitrary Rust application semantics, relocation, or assembly.

The saved manifest quantifies the tradeoff for the one-link application:

| Quantity | Ordinary representation of the same app | Selected compact representation |
|---|---:|---:|
| Logical coordinates | 137,646,804 | 137,341,846 |
| Logical rows | 3,256,394 | 3,248,956 |

The specialization saves **304,958 logical coordinates, about 0.222% of the ordinary total**. It has a much larger effect within the small application suffix, but that suffix is a small part of F′. These are layout counts, not measured speedups.

Recommended change: use one application assembly mechanism. If the compact representation is required, make its transformation part of that mechanism with a common contract. Otherwise use the ordinary route and delete the special-case recognition, reverse conversion, and paired manifest state. For your stated preference for lean code, retaining a separate application route needs a concrete benefit stronger than the current whole-circuit coordinate saving.

**F4. The new direct CCS witness execution is not used by public prove/extend**

The public Prover::extend always executes the application and passes Some(witness.values()). Prover::prove delegates to that method.

The completion code chooses the new direct CCS function only for None. For Some, it calls execute_stage1_v1_1_witness_with_application_values, which invokes execute_assignment_source with direct_product_outputs set to false. Therefore the normal public API continues through the full physical execution path, even for the selected application.

Evidence: [public prove and extend](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream/src/circuit.rs#L129); [completion branch](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream/src/lifecycle/complete.rs#L147); [new direct entrypoint](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream-fprime/src/package/sealed.rs#L343); [cached-application path disables it](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream-fprime/src/package/sealed.rs#L437).

The direct/full equivalence checks are useful, but they do not demonstrate that the public API uses the new route. The current direct entrypoint also still constructs a physical source assignment before transport; its concrete benefit is skipping selected redundant product work, not eliminating the complete physical allocation.

Recommended change: route the public prover through the chosen witness implementation while preserving reuse of application values. If the direct route is not to be used, delete the unused production alternative and describe the optimization as deferred. Do not expose a new user-facing mode.

Closing evidence: a check through the actual public API must establish the chosen execution route and preserve complete witness/output equality and rejection behavior.

**F5. Correct the generic application's mutation-test coordinate**

The new test describes changing an application coordinate, but obtains the column from application().private_range().start. That range is in the physical source assignment. It then indexes the committed CCS witness as if the same number were a logical retained coordinate.

For the tested two-link application, the emitted manifest gives:

| Item | Coordinate |
|---|---:|
| Physical application-local start used by the test | 27,859,268 |
| Logical retained application-local start | 132,904,887 |
| Logical block containing the actually mutated coordinate | Block 0, interval [270, 43,546,370) |

Thus the test mutates a different part of the witness. Rejection of that mutation, even after recomputing the commitment, does not prove the claimed application-local binding.

Evidence: [mutation and witness indexing](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream/tests/circuit/hash_chain_vector.rs#L72); [source and retained port metadata](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream/artifacts/shared-verifier-v1.json#L1). The relevant application dimensions are eight private input words and 15,392 local words.

Recommended change: derive the logical coordinate from the retained assignment/port mapping. Check that the chosen coordinate lies in the application interval before mutating it. Keep the recomputed-commitment control. An application-suffix substitution with a detached state is a stronger check than changing an unrelated bit.

This defect is in the test introduced by 9e9cd23d1. It is not evidence that the application is unsound. The separately maintained selected-application detached-output test uses the retained ranges correctly and passed my rerun.

**F6. Add the cheap tests for this PR's central changes to CI**

The workflow runs nightstream tests, a matrix_rows integration test in neo-reductions, and two filtered nightstream-fprime unit suites. Cargo does not run a dependency's own unit and integration tests merely because nightstream uses that dependency.

Consequently the workflow does not run the new pi_rlc_wide_lean_parity integration test or the matrix_program and wide-transport unit suites I executed. Those checks took less than one second of test execution each after compilation. They test transcript advancement, direct challenge centering, removed-column reads, projection gaps and overlaps, family ordering, and out-of-range sampler digits.

Evidence: [CI test selection](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/.github/workflows/ci.yml#L66); [sampler parity suite](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/neo-reductions/tests/pi_rlc_wide_lean_parity.rs#L1); [matrix reader controls](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream-fprime/tests/unit/matrix_program.rs#L1).

Recommended change: add those focused Rust tests to the existing job. There is no need for another implementation, profile, or test framework.

The decision to run Lean locally instead of in CI is explicitly recorded by the owner in 00226950e. I am not treating that decision itself as a defect or recommending an unrequested reversal. It does mean that green GitHub checks cannot stand in for a source-matched local Lean build, axiom audit, and identity check.

**What was completed correctly**

The selected sampler migration has several substantive strengths:

- The native old decoder was removed from neo-reductions and replaced with the whole-vector decoder. The selected optimized prover uses one four-field digest per scalar.
- The decoder's integer arithmetic matches the stated map. Canonical Goldilocks lanes form a base-p integer below p⁴; repeated division by five gives the required 54 centered digits without rejection.
- The Lean gadget constrains canonical input words, quotient bits, digit range, and modular checks. Its soundness does not trust the prover's hint values.
- The checked remapping rejects missing source reads before normalization can erase zero or cancelling terms. That directly addresses the dangerous deleted-coordinate fallback removed in 80f6f12e6.
- PiDEC consumes the same wide PiRLC parent outputs and retains the commitment, public-input, Pad-evaluation, and matrix-evaluation families.
- The selected context serializer binds the wide schedule. The package, verification-key, and fixture identities were refreshed.
- Rust accepts assignment transport schema 4 and rejects retired schema 3; there is no compatibility fallback in that decoder.
- bab9b858f ports the principal NIFS/history security statements to Wide.Target and deletes three baseline-specific security owners. This is a real change of the theorem target, not merely a renamed theorem.
- The key-capacity change keeps each application's exact same-seed prefix bound into its configuration. It does not silently use the selected application's smaller dimensions for all applications.

See [native decoder](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/neo-reductions/src/common/pi_rlc_wide.rs#L1), [sole transport schema](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/crates/nightstream-fprime/src/package/assignment_transport.rs#L21), [wide target](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Wide/TerminalSecurity.lean#L29), and [same-seed MSIS reduction](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Wide/SetupBinding.lean#L53).

**Paper checks and security boundaries**

I read the September 4, 2026 revision of [Neo and SuperNeo, ePrint 2026/242](https://eprint.iacr.org/2026/242), specifically §§4–8 and Appendix B.1–B.4. I also read [HyperNova, ePrint 2023/573](https://eprint.iacr.org/2023/573), §§5–6.3 and Appendix H.2–H.3, using the February 20, 2026 revision. I inspected rendered protocol pages as well as extracted text. The older Neo ePrint 2025/294 is not the authority used for this review.

SuperNeo's relevant requirements are a common evaluation point, low-norm committed inputs, ring-linear combination of every carried family, and recomposition of all decomposition children. Its security proof uses uniform strong-set challenges and separates extraction and binding losses. HyperNova Construction 2 requires the F′ circuit to bind the prior state and verifier key, verify the appropriate fold, and hash the new state; terminal verification checks the remaining running and fresh relations. These requirements guided the implementation review.

The following conclusions are specific to this repository, rather than claims that the paper mandates this exact implementation:

| Obligation | Evidence inspected | Conclusion |
|---|---|---|
| Same challenge map in native execution and F′ | Wide sampler definition, gadget, transcript schedule, Rust decoder, parity tests | The selected map agrees on the tested boundary and transcript vectors |
| Sampler range and integer equality | CanonicalU64 children, Boolean/range rows, six modular checks, CRT argument | No range/wrap defect found in the inspected argument |
| Ring product and retained projection | Quotient layout, checked source support, matrix program, mutation controls | The change has explicit preservation arguments; it is not justified only by matching hashes |
| Full parent and child obligations | Wide TerminalSecurity, OutputWitnessConsumer, native terminal verifier | Current claims check actual openings and both evaluation families |
| HyperNova context/state induction | Wide opening/public/context binding, natural counter bounds, history consumers | The main soundness consumers refer to the wide relation |
| Concrete Fiat–Shamir security | WideFiatShamir and the recorded model | Remains an explicit assumption with uninstantiated loss functions |
| Rust-to-Lean correspondence | Package equality, primitive/challenge/output parity, executed examples | Strong evidence at the checked interfaces; no universal proof of Rust execution |

I independently recomputed the main sampler arithmetic with exact integers:

- p = 18,446,744,069,414,584,321; N = 5⁵⁴; M = p⁴.
- For r = M mod N, the total variation distance is r(N−r)/(NM), approximately 2^(-132.98) per scalar.
- The independent 17-scalar hybrid bound is approximately 2^(-128.89). This is a sampler bound, not total proof security.
- The separate coordinate-extraction loss 17/5⁵⁴ is approximately 2^(-121.30).
- The largest honest quotient needs 131 bits. The six check moduli are pairwise coprime; their product has 300 bits, while the largest represented result side needs at most 257 bits. Conservative bounds for both sides of each check equation remain below the Goldilocks modulus.
- The selected folding norm bound is consistent: 17 × 216 × (2−1) = 3,672 < 65,536.

The subtle point is how these sampler facts enter security. The production history theorem does **not** consume the adaptive bias theorem as a completed concrete Poseidon2-to-uniform reduction. The current source explicitly keeps the wide/uniform difference inside the assumed Fiat–Shamir term deltaFS. Its g and deltaFS remain symbolic. An arbitrary nonlinear g does not allow moving an additive bias bound through it.

That scope is stated in [Fiat–Shamir model](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Nifs/WideFiatShamir.lean#L110) and [selected-wide assurance notes](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/formal/nightstream-fprime/ASSURANCE_SURFACE.md#L92). The extension is recorded as owner approved. I therefore do not call it an unauthorized assumption. I also would not describe the sampler bias proof, a zero-sorry audit, or the 114-bit runtime estimator as a complete numerical security proof.

Likewise, the fixed public seed and expanded maximum key rely on the project's stated fixed-instance MSIS premise. Zero-extending a short-kernel witness supports the prefix reduction; it does not by itself derive hardness for that fixed public matrix from the paper's uniform-setup experiment. Classical extraction/Fiat–Shamir results do not establish QROM extraction merely because the commitment assumption is lattice based.

**Do not confuse these different kinds of “multiple tracks”**

| Surface | Current state | Treatment |
|---|---|---|
| Selected native old sampler vs wide sampler | Wide replaces the old native decoder | Keep the replacement |
| Old schema-3 vs schema-4 assignment transport | Old schema rejected | Keep the rejection; no compatibility mode |
| Default old Lean emitter vs explicit wide emitter | Both remain reachable | Consolidate and remove obsolete entrypoints |
| Selected Poseidon2 vs generic application assembly | Two active lowering/layout routes | Consolidate under one application contract |
| Full vs direct witness execution | Both exist; public API takes full execution | Choose and wire one public path |
| nightstream vs neo-fold-legacy | New runtime independent; validation and other consumers still depend on legacy | Remove the maintenance dependency |
| CPU vs Metal | Execution backends for the same package/proof contract | These are not competing protocol versions |
| PaperExact and cross-check code | Reference evaluators, with high cost | Keep only a clear test/reference purpose if still needed |
| Old layout definitions used in preservation proofs | Some are still in the selected import closure | Extract shared facts before deletion |
| Historical review records | Evidence about named earlier source snapshots | May remain as records; must not count as current approval |

A syntactic import-closure check found 918 local Lean modules reachable from Wide.Emitter, including old sampler/layout modules. That count shows build and ownership coupling, not that all of those modules execute or are semantically redundant. Removing names indiscriminately would risk breaking the preservation argument.

**Size reduction and cost**

| Measure | PR base / original checkpoint baseline | Selected wide head | Change |
|---|---:|---:|---:|
| Committed carrier coordinates | 253,011,276 | 137,341,872 | −45.717% |
| Logical constraint rows | 6,377,559 | 3,248,956 | Reduced |
| Normalized matrix nonzeros | 2,335,822,475 | 2,607,606,765 | **+11.635%** |

The final wide step also improves all three measures relative to the intermediate reduced package. It does not reduce every measure relative to the original base. Denser quotient-evaluation rows explain why fewer committed coordinates can coexist with more nonzeros.

These counts come from the emitted geometry and the recorded full-matrix ledger, not a new full-matrix recount by this review. My executed matrix tests cover the formulas and remapping controls; they are not a recount of 2.6 billion entries. The ledger is in [quotient baseline comparison](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/tools/recursive-constraint-minimizer/experiments/PLAN.md#L192) and [wide comparison](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/tools/recursive-constraint-minimizer/experiments/wide-sampler-integration.md#L453).

The later “additional 50%” target was relative to 184,359,564 coordinates, meaning at most 92,179,782. The selected 137,341,872 does not meet that target. The branch records this honestly. No whole-lifecycle speedup should be inferred from the width reduction alone.

**Review records and claimed closure**

The four current security review requests remain **pending independent review and controlled signed acceptance**. The older passing response files are historical diagnostic records, not approval of the current source.

The late 99b3b0953 commit corrects a material evidence-record issue: the earlier security requests included an uncommitted copy of the saved primitive archive, so no commit reproduced that capture. It regenerates the requests against 00226950e. The earlier c5595d978 assertion that all requests remained valid should therefore not be carried into a merge decision.

The current README is explicit about this correction and the pending status. See [current request record](https://github.com/LFDT-Nightstream/Nightstream/blob/99b3b095366f7c8328de44d828e14c0dce6a571d/docs/reviews/nightstream-fprime-requirements/wide-target-reviews/README.md#L1). This report does not create a signed acceptance record.

The main deterministic wide theorem has meaningful premises and conclusions. I did not find a premise that simply assumes the desired arbitrary-assignment conclusion. Still, package completeness is conditional on a checked compiled sampler, successful package preparation, and the stated valid step/NIFS conditions. The baseline HyperNovaCompleteness and HyperNovaAcceptedNext declarations are not an unconditional honest-prover existence theorem for every current wide application. The project records compiler-success and recursive-terminal existence as unproved claims. These are limits on completeness claims, not demonstrated false acceptance.

**Commit-by-commit ledger**

The entries below distinguish what each commit added from what is true at the reviewed head. “Present” means the named implementation or proof boundary exists in the final source; it does not imply that I independently rebuilt every historical checkpoint.


| # | Commit and subject | What was done; status at reviewed head |
|---:|---|---|
| 1 | [8b7c07d85](https://github.com/LFDT-Nightstream/Nightstream/commit/8b7c07d85ca0ff47c1efada56dbe165e2011bb16) — Checkpoint proved Phi81 quotient layout and 27.13% width reduction | Adds the Phi81 quotient layout and its preservation/witness links. The 27.13% number is a historical checkpoint; final width improves further, but nonzero count remains above the original baseline. No final speed claim follows. |
| 2 | [a650fa771](https://github.com/LFDT-Nightstream/Nightstream/commit/a650fa771505b821e918f5b91410c93dde5c8686) — Prove direct PiRLC phase and record remaining reduction limits | Adds the direct PiRLC phase proof and documents remaining reduction limits. The phase is useful, but the public-runtime execution issue in F4 remains. |
| 3 | [12158c929](https://github.com/LFDT-Nightstream/Nightstream/commit/12158c929dce30abbb06804004a106a070e7b6ed) — Prove canonical witness read support and stored execution invariants | Establishes which canonical witness values execution can read and the stored-execution invariants. These are relevant support for safe scratch removal, rather than a separate protocol. |
| 4 | [6bcdbb7c6](https://github.com/LFDT-Nightstream/Nightstream/commit/6bcdbb7c6a6d3b4aad83e67b3325bb81f2059bc0) — Prove canonical direct CCS witness production and inspect cost candid… | Constructs the direct CCS witness proof chain and records cost candidates. Current final source retains direct and full execution alternatives; existence/equivalence alone does not select the public call path. |
| 5 | [f42a6d53d](https://github.com/LFDT-Nightstream/Nightstream/commit/f42a6d53d36ceace31838b7b13a51f2e983f7f0f) — Integrate direct CCS witnesses with the proved quotient package | Connects the direct witness entrypoint to the quotient package. Final completion has a None/Some split; public prove/extend take Some and bypass the new route. See F4. |
| 6 | [961ba3d2e](https://github.com/LFDT-Nightstream/Nightstream/commit/961ba3d2e6dac05a8b194e80bad112d49898dae6) — Share checked preparation across child opening batches | Shares checked preparation for child opening batches. This changes execution reuse, not the relation or challenge distribution; no new protocol track identified here. |
| 7 | [c9aa75b04](https://github.com/LFDT-Nightstream/Nightstream/commit/c9aa75b04a8a55f1f357ed414c72825fda2bbb6b) — Reduce committed witness with a shared transition flag | Replaces transition scratch with a shared Boolean flag/inverse and removes redundant Poseidon output pins. Counts and identities are refreshed at that checkpoint. Later wide work builds on these reductions. |
| 8 | [6c8703a4e](https://github.com/LFDT-Nightstream/Nightstream/commit/6c8703a4ee0dcf8af6472360c97273ec221bb6f6) — Specify the whole-vector PiRLC sampler and prove its law | Defines the total four-field, base-p-to-base-five sampler and proves its mathematical law. No Poseidon2 randomness law is claimed. Independent integer checks and current Rust parity support the map. |
| 9 | [e1348a5a1](https://github.com/LFDT-Nightstream/Nightstream/commit/e1348a5a12a0e58c13b1834519a488691c6e7e6a) — Transfer the PiRLC extraction bound to sampler-drawn challenges | Transfers bounded-test expectations by a hybrid argument. The extractor on the comparison side still uses uniform challenges; this does not independently prove the concrete wide Fiat–Shamir reduction. |
| 10 | [a8283c366](https://github.com/LFDT-Nightstream/Nightstream/commit/a8283c3662599ec54c537615d828bdc11cb60a5c) — Add the whole-vector sampler gadget and prove its soundness | Adds the sampler's checked range/CRT gadget. Canonical words, quotient/digit bits, and all six checks have real roles in the inspected soundness argument. |
| 11 | [959e2fa78](https://github.com/LFDT-Nightstream/Nightstream/commit/959e2fa783804e3807288341a206d52fd0a2638b) — Prove the exact footprint of the whole-vector sampler gadget | States the 681-row footprint. The early 617-variable count is conditional on the supplied hint allocation; later executable/layout work is needed to turn it into package counts. |
| 12 | [919cd12b4](https://github.com/LFDT-Nightstream/Nightstream/commit/919cd12b481a7c3d9335be8df94ac08718857e70) — Prove completeness of the whole-vector sampler gadget | Adds gadget completeness and audits. Correctly states that this early result is existential and does not yet provide the executable witness program. |
| 13 | [929d53686](https://github.com/LFDT-Nightstream/Nightstream/commit/929d53686be93ea6cf03a1abac5c3a7f88a6834a) — Checkpoint compact PiCCS lowering proofs | Compacts PiCCS Horner evaluation and gamma powers and starts application compaction. Package/fixture selection was still pending here; later commits perform it. |
| 14 | [0592ea2c9](https://github.com/LFDT-Nightstream/Nightstream/commit/0592ea2c915a718c02e38569e86b337f38057470) — Prove the compact application specialization and placed relation | Proves the compact application specialization and placement. This becomes the selected-app branch discussed in F3; it does not by itself cover arbitrary Rust application assembly. |
| 15 | [b2c5a8cfd](https://github.com/LFDT-Nightstream/Nightstream/commit/b2c5a8cfda8c639212d71d07c69edbb08230958f) — Merge the reviewed wide PiRLC sampler proofs | Merges the wide sampler as an explicitly unselected component. The review scope at this point was clear. Later production selection supersedes the unselected status. |
| 16 | [8daa92d0f](https://github.com/LFDT-Nightstream/Nightstream/commit/8daa92d0fcb93940b243596e625e1a7498f9a469) — Prove compact application matrix program correspondence | Adds compact-application matrix program correspondence. This supports the selected specialization; generic assembly still needs its own implementation conformance. |
| 17 | [49c54577c](https://github.com/LFDT-Nightstream/Nightstream/commit/49c54577c595b5cff0867ee9b5de1a7aa2fa0029) — Select compact application rows and construct their witness directly | Selects the compact application rows and witness construction. The selected application now differs from ordinary application lowering; this separation remains in the final code. |
| 18 | [41aa5ef63](https://github.com/LFDT-Nightstream/Nightstream/commit/41aa5ef635fcb098697e3f255fc941e18c375d46) — Integrate reduced package and verify matrix and witness conformance | Publishes the reduced package and refreshed identities. NIFS/recursive fixtures were explicitly not yet refreshed; the next commit closes that checkpoint's gap. |
| 19 | [f77181b7e](https://github.com/LFDT-Nightstream/Nightstream/commit/f77181b7ea976d2871a392a01e0ff02beb542ecd) — Connect the reduced package to application assembly and recursive fix… | Updates assembly and recursive fixtures for the reduced package. The exact selected-plan branch plus ordinary connector is introduced here. F3 is therefore a real PR delta. |
| 20 | [be5b7fcdc](https://github.com/LFDT-Nightstream/Nightstream/commit/be5b7fcdc17cf599acb7d22b0f217cf5a2c2cdb2) — Repair ignored conformance tests for the reduced package | Repairs ignored conformance tests, pins, and invocation inputs. These are test repairs for the current package, not proof of a new reduction or performance improvement. |
| 21 | [07588c4b8](https://github.com/LFDT-Nightstream/Nightstream/commit/07588c4b8fb640f309abe0d44314fc218ca8dfb9) — Prove executable wide sampler and compact CCS candidate | Adds executable sampler hints, compact CCS range compilation, and oracle-model comparison results. The compiled plan remains a checked success value; unconditional compilation is a separate claim. |
| 22 | [6d1d4c16d](https://github.com/LFDT-Nightstream/Nightstream/commit/6d1d4c16d0d6c9c9ea68b46c2ae1cfa0a7c328ac) — Connect wide sampler to PiRLC phase and prove source layout | Connects wide sampling to PiRLC and adds wide phase/layout owners beside the older ones. This is the start of the substantial parallel proof/layout structure. |
| 23 | [ce03cab61](https://github.com/LFDT-Nightstream/Nightstream/commit/ce03cab618f95849ca6ff34d1b73d2ce39d315cc) — Checkpoint candidate PiRLC retained witness work for review | Publishes partial retained-witness work without claiming a rerun or complete integration. Treat as a checkpoint; the later closure commits are the relevant final evidence. |
| 24 | [21facf523](https://github.com/LFDT-Nightstream/Nightstream/commit/21facf5231803fb5ee2bd9f803bd130c8413723d) — Connect the wide sampler to the candidate Stage 1 retained layout | Builds the retained layout, row/width counts, wide response key, and parity checks. At this point it is still a candidate, not a completed runtime switch. |
| 25 | [14e83424c](https://github.com/LFDT-Nightstream/Nightstream/commit/14e83424c2b25b753992451970c73b885873f61a) — Prove PiDEC reads the wide sampler ring outputs | Proves the PiDEC parent forms read the wide PiRLC outputs, including extension-cell ordering. Important connection between otherwise locally correct phases. |
| 26 | [5d2d2a45b](https://github.com/LFDT-Nightstream/Nightstream/commit/5d2d2a45b17fcf050323e98e228d6360bc48d8ac) — Derive the wide sampler candidate recursive fixed point | Derives the candidate fixed point from its own matrices. A fixed-point theorem is necessary for recursion; this commit alone did not select the package. |
| 27 | [80f6f12e6](https://github.com/LFDT-Nightstream/Nightstream/commit/80f6f12e6a19d8c6f1c5a514236ca654458e117d) — Require proved read support for wide sampler coordinate remapping | Removes the deleted-coordinate fallback and requires read support. This addresses a real soundness-sensitive risk; final Rust tests also reject gaps, overlaps, and removed reads. |
| 28 | [dd194fade](https://github.com/LFDT-Nightstream/Nightstream/commit/dd194fadeb8624ddc45d72b8f9675fd5642d7669) — Prove retained witness projection and wide PiDEC source reads | Proves checked projection and PiDEC source mappings. Keeps missing old sampler/scratch coordinates from being silently read through the new layout. |
| 29 | [2349f5cfe](https://github.com/LFDT-Nightstream/Nightstream/commit/2349f5cfefa7d328066cd373e7922d53ba0d26d4) — Prove wide source assignment and PiDEC witness transport | Connects source assignments and PiDEC witness transport. This is a preservation bridge to reused rows, not evidence that the old sampler executes in the selected relation. |
| 30 | [066f357f4](https://github.com/LFDT-Nightstream/Nightstream/commit/066f357f43b809940a543859a40169c9157e18de) — Keep reference output and quotient order in the wide candidate | Simplifies output/quotient indexing to preserve the existing order and removes the obsolete permutation module. A useful reduction in remapping complexity. |
| 31 | [e840c4f7e](https://github.com/LFDT-Nightstream/Nightstream/commit/e840c4f7ea87e9af2cb3b3f91ee4fac63291aecc) — Prove compact PiDEC acceptance from the wide physical witness | Connects the physical wide witness to compact PiDEC acceptance. Establishes equality of the four parent families before applying the reused decomposition constraints. |
| 32 | [040528834](https://github.com/LFDT-Nightstream/Nightstream/commit/04052883411e0a97212654135450954f485ca463) — Prove wide assignment acceptance for the pilot and PiCCS prefix | Proves the wide assignment accepts pilot and PiCCS prefix rows. It closes part of whole-package completeness, not a separate endpoint claim. |
| 33 | [cf41056ad](https://github.com/LFDT-Nightstream/Nightstream/commit/cf41056ad99330034c47c3e3ba70ba395f9e1f86) — Prove all wide candidate rows accept one constructed source assignment | Connects all candidate row families to one constructed assignment. This is stronger evidence than proving each family with unrelated witnesses. |
| 34 | [6becee6d8](https://github.com/LFDT-Nightstream/Nightstream/commit/6becee6d806473525932dca4f6cbcec958872107) — Prove wide carrier norm and public digest transport | Adds carrier norm and public-digest transport. These obligations prevent a correct internal witness from being detached from the committed public statement. |
| 35 | [5fea59579](https://github.com/LFDT-Nightstream/Nightstream/commit/5fea5957915c7eabaff97247a625afb13dae62e7) — Prove complete wide-sampler CCS witness construction | Packages complete wide CCS witness construction. Its semantic-step, encoding, and acceptance premises still matter; this is not an unconditional public Rust prover theorem. |
| 36 | [12ea7bf81](https://github.com/LFDT-Nightstream/Nightstream/commit/12ea7bf810e28ba51ea21599419e9ffdd966dfb6) — Prove wide-sampler HyperNova step soundness | Adds arbitrary-row wide HyperNova step soundness and decoded state/output links. This is a key deterministic boundary inspected in the review. |
| 37 | [d369c409a](https://github.com/LFDT-Nightstream/Nightstream/commit/d369c409ae63e4b4bf1f0ccbd2b3e428404fde2f) — Prove wide sampler matrix decoding and public binding | Adds matrix decoding and public binding for the sampler/reused phases. The ring-product matrix bridge was still pending at this checkpoint and is supplied next. |
| 38 | [ac54e082b](https://github.com/LFDT-Nightstream/Nightstream/commit/ac54e082b4e9b044e08e239b1340097c3830f235) — Prove full wide-sampler matrix correspondence and count entries | Completes candidate matrix correspondence and counts. Final selected dimensions retain these 3,248,956 rows and 137,341,872 committed coordinates. |
| 39 | [d2c90b6d2](https://github.com/LFDT-Nightstream/Nightstream/commit/d2c90b6d2a4746ab64b30d3a2cf1e09f94d7c9de) — Connect wide sampler security and verifier context binding | Adds wide context/security consumers. The concrete block-oracle comparison and Fiat–Shamir transfer remain distinct; the production transfer is still an explicit model assumption. |
| 40 | [3db3c0902](https://github.com/LFDT-Nightstream/Nightstream/commit/3db3c0902a19be387262486f9e3dec96885b1cb6) — Read embedded wide sampler rows and direct matrix inputs | Adds Rust embedded-row and direct-input matrix readers. The new opcodes support the wide package and are covered by focused matrix tests, which CI currently omits. |
| 41 | [d8e83e510](https://github.com/LFDT-Nightstream/Nightstream/commit/d8e83e510ddbf1209548ef1935e766aa08dc6b59) — Complete checked matrix decoding and wide witness parity | Finishes checked matrix decoding and hint parity. Current matrix/transport unit checks pass, including malformed projection and challenge controls. |
| 42 | [965351823](https://github.com/LFDT-Nightstream/Nightstream/commit/965351823a751ce5e2abff560d8c7e8995421017) — Emit wide-sampler source package and validate remapped matrices | Adds the wide physical package emitter and remapped matrix input. Later corrections bind emitted parts to the proved pure constructor. |
| 43 | [daf5575c4](https://github.com/LFDT-Nightstream/Nightstream/commit/daf5575c4501b28cca7b4cac8037dc1d09271628) — Construct sealed wide-sampler witnesses with checked source transport | Adds schema-4 retained transport and sealed witness construction. The final Rust decoder keeps only schema 4, which is a successful part of removing compatibility. |
| 44 | [a0f4b5b41](https://github.com/LFDT-Nightstream/Nightstream/commit/a0f4b5b41f7ba0621747181d9ffbe581e1a690e8) — Prove complete wide package witness and archive correspondence | Adds archive, source-custody, matrix, transport, and package-completeness links. Claims remain conditional on successful preparation and the stated semantic conditions. |
| 45 | [c65a8605a](https://github.com/LFDT-Nightstream/Nightstream/commit/c65a8605a9b6cb02443449b16ef44fe743873a6f) — Checkpoint wide-sampler production selection and fresh fixtures | Saves the production selection checkpoint and explicitly records a failed detached-application regression plus unconfirmed final gates. It should not be described as green by itself. |
| 46 | [982538618](https://github.com/LFDT-Nightstream/Nightstream/commit/9825386184734bb0152b95fb0e209ff605fb7414) — Merge Lean 4.32.2 base into the wide-sampler integration | Merges the updated base and fixes the emitter constructor, opening binding, mapped application test, selected recognition, and family-order checks. My detached-application rerun passes; old emitter fallback and broader duplication remain. |
| 47 | [bab9b858f](https://github.com/LFDT-Nightstream/Nightstream/commit/bab9b858f046dabb531eefbea6a533b6750c838b) — Port the NIFS and HyperNova security chain to the wide key | Moves the main NIFS/history security chain to SecurityInstance and Wide.Target and deletes three old security modules. This port is substantively present. Other baseline completeness/replay owners still remain. |
| 48 | [ed5fa1b95](https://github.com/LFDT-Nightstream/Nightstream/commit/ed5fa1b950764fb30ba6f30d1ba1eeb2a9c9dfef) — Use the Mathlib cache in the Lean CI job | Enables use of the Mathlib cache in the then-current Lean CI job. This workflow arrangement is later superseded by the local-only Lean decision. |
| 49 | [9e9cd23d1](https://github.com/LFDT-Nightstream/Nightstream/commit/9e9cd23d140952c68889092b4d2bfea70b1b6880) — Support key prefixes up to the approved MSIS matrix | Raises supported key-prefix capacity to the approved matrix and adds the two-link generic application/vector. Capacity and assembly checks pass. The new application-witness mutation uses the wrong coordinate space; see F5. |
| 50 | [f0dbdef8b](https://github.com/LFDT-Nightstream/Nightstream/commit/f0dbdef8ba048d6f18b11adcb7c57fe8a0ca38a0) — Record the four wide-key target reviews | Records initial four-target review responses and fixes labels/references. Those responses cover an earlier source; they are not approval of the current head. |
| 51 | [5b276d7be](https://github.com/LFDT-Nightstream/Nightstream/commit/5b276d7be8171e15e34bae7b7a81ffdeffba1835) — Test the wider key capacity on the fold and the Metal commitment | Adds wider-prefix boundary checks and a public recursive Metal test. I reran the Metal proof/fold/terminal test successfully. The long CPU counterpart remains ignored. |
| 52 | [eb3f78599](https://github.com/LFDT-Nightstream/Nightstream/commit/eb3f78599f968f34ae363272d7306c89d7889019) — Exclude dangling Lean-artifact symlinks from Rust source capture | Excludes Lean-artifact symlinks from the Rust-only source capture. This repairs the capture boundary; it is not a proof that omitted artifacts are correct or approved. |
| 53 | [0522b50d7](https://github.com/LFDT-Nightstream/Nightstream/commit/0522b50d7e18d2787281243e4e5b87c5ec94b25d) — Record the final wide-key target reviews | Records the next set of passing diagnostic target reviews. Their README correctly reserves controlled acceptance. Later requests supersede these records. |
| 54 | [48c8e0b99](https://github.com/LFDT-Nightstream/Nightstream/commit/48c8e0b9919a65459b3d4ddb5f3e1220d09c7bac) — Fix the Metal NIFS adapter tests after bounded row streaming | Repairs Metal adapter expectations after streamed row windows. The change preserves proof comparisons but continues maintenance of the deprecated adapter; see F2. |
| 55 | [8d2f3ae8b](https://github.com/LFDT-Nightstream/Nightstream/commit/8d2f3ae8bfbf4ae58105be806b0c13db572a7c4a) — Record current-package golden conformance at the four parity interfaces | Records two-fold current-package golden conformance and removes the retired physical-witness comparison. The comparison boundary is defensible; the coordinator handoff still needed the following repair. |
| 56 | [ce0159538](https://github.com/LFDT-Nightstream/Nightstream/commit/ce0159538534579f3459adacecfca325b5b9692b) — Fix golden conformance handoff and retain the current vectors | Repairs that coordinator handoff, retains both fold vectors, and adds arithmetic parity. Current Python regression suites and saved primitive comparisons pass in this review. |
| 57 | [202b2fbf4](https://github.com/LFDT-Nightstream/Nightstream/commit/202b2fbf41b289a69858c66f1eff9c67f5796c4c) — Prepare current source-bound review requests for the golden fixes | Refreshes source-bound requests and corrects a false-acceptance dependency name. The later source-capture correction in 99b3b0953 supersedes these requests. |
| 58 | [31148bcce](https://github.com/LFDT-Nightstream/Nightstream/commit/31148bcce3195a1d4be48d886ee621e86102882f) — Run Lean locally and compare saved primitive results in CI | Moves Lean checks to a local command and compares saved primitive vectors in CI. This is now an explicit owner choice. Cheap wide-sampler and matrix/transport CI coverage remains missing. |
| 59 | [c5595d978](https://github.com/LFDT-Nightstream/Nightstream/commit/c5595d978947a1b79c0c7aecfc0c0ac7387d96b9) — Record review scope for the local Lean check workflow | Claims the earlier requests remain valid after the local-check workflow change. The later archive/source-capture correction supersedes that claim; do not rely on this checkpoint's request IDs. |
| 60 | [e3c86bda2](https://github.com/LFDT-Nightstream/Nightstream/commit/e3c86bda228005d89eb4f4dc1a74c0f0d961ad89) — Update the deprecated Lean project rule after its removal | Updates the rule to reflect removal of formal/nightstream-lean. Correctly directs reference access to Git history rather than recreating the project. It does not remove the separately retained Rust legacy crate or old owners inside the active Lean package. |
| 61 | [00226950e](https://github.com/LFDT-Nightstream/Nightstream/commit/00226950e51f389e09ecb4cfdadb845d14b36908) — Record the owner decision to check Lean locally, not in CI | Records the owner decision that Lean runs locally, not in CI. I respect that scope; F6 concerns missing focused Rust tests, not this decision. |
| 62 | [99b3b0953](https://github.com/LFDT-Nightstream/Nightstream/commit/99b3b095366f7c8328de44d828e14c0dce6a571d) — Regenerate the four target review requests at the current source | Regenerates requests against the committed 00226950e tree and acknowledges the earlier uncommitted archive capture. This repairs the record. All four reviews and controlled acceptance remain pending. |

**Validation performed for this review**

All Rust tests used release mode. Every test invocation had an outer timeout of at most 300 seconds. Times below are test execution times, excluding compilation.

| Check | Result | Time / scope |
|---|---|---|
| neo-reductions: pi_rlc_wide_lean_parity | 3 passed | Decoder boundaries, one-window state, full production sampler; <0.01 s |
| nightstream-fprime: unit filter wide | 5 passed, 3 ignored | Hint parity, family order, invalid challenge bits/digits; 0.03 s |
| nightstream-fprime: unit filter matrix_program | 23 passed, 1 ignored | Formula equivalence, quotient rows, centering, mapped reads, gaps/overlaps; 0.67 s |
| nightstream: assembly_circuits | 3 passed | Selected application, another application, wider two-link prefix; 30.64 s |
| Public two-link recursive test on Metal | 1 passed | Public prove → extend → verify and changed endpoint rejection; 56.86 s |
| Selected application detached-output regression | 1 passed | Arbitrary low-norm application-suffix substitution rejected in application rows; 147.11 s |
| Saved Lean primitive comparisons | 3 passed | Field/extension arithmetic, ring transform, signed decomposition |
| Golden coordinator/runner/replay Python suites | 26 passed | 10 + 7 + 6 + 3 tests |
| Lean static boundary gate | Passed | Matrix codec and layer checks |
| Fresh Lean library plus axiom-target build | Timed out; unverified | Outer timeout expired at 300 seconds, exit 137; remaining nested build processes were stopped explicitly |
| neo-reductions: k_mcs_end_to_end | 1 passed, 6 failed | Existing empty-running-claim failures; 0.01 s |
| Git diff whitespace check | Passed | Complete PR diff |
| Runtime dependency tree | Passed | No neo-fold-legacy in normal nightstream dependencies with metal,cuda |

The fresh Lean build compiled dependencies and part of the project. The outer timeout fired at 300 seconds, but the validation script's nested timeout had a separate process group and left the inner build running. I explicitly killed that owned build tree and checked that no review compiler process remained. This is an incomplete build, not a successful full library or axiom-target result. Future bounded runs need a timeout controller that stops the complete process tree. The passing static gate and the PR's recorded local proof checks must not be relabeled as fresh complete Lean validation by this review.

The detached application failed at an earlier PR checkpoint because the test did not locate a mapped application block. The final test handles the checked wrapper, constructs a valid replacement application suffix, detaches it from the original state, and rejects at logical row 3,248,943. This closes that specific historical failure in my rerun.

The six k_mcs failures are not attributed to this PR. Both the test file and the rejecting prior_digest_fields guard are unchanged from the base, and the failure happens before PiRLC. I checked that source provenance; I did not execute a second full base checkout. The PR already discloses these failures, and CI does not run them.

Two local setup errors were corrected before counting results: the managed checkout initially held Git LFS pointers instead of the selected JSON artifact, and system Python 3.9 lacked tomllib for the coordinator test. Fetching the selected LFS objects and using Python 3.12 for that test resolved them. They are not PR defects.

GitHub reported passing Rustfmt, DCO, and shared regression jobs at the final observed head. [Observed CI run](https://github.com/LFDT-Nightstream/Nightstream/actions/runs/36240634557). That is consistent with the configured test subset, not evidence that every workspace test or Lean gate passed in CI.

Commands for the main checks, from the reviewed checkout:

    timeout --signal=KILL 300 cargo test -p neo-reductions --release --test pi_rlc_wide_lean_parity
    timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib wide
    timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --lib matrix_program
    timeout --signal=KILL 300 cargo test -p nightstream --release --test assembly_circuits
    timeout --signal=KILL 300 cargo test -p nightstream --release --features metal --lib circuit::hash_chain_vector::two_link_hash_chain_folds_on_metal -- --exact --ignored
    timeout --signal=KILL 300 cargo test -p nightstream-fprime --release --test base_step_assignment base_step_rows_reject_a_detached_application_output -- --exact --ignored
    timeout --signal=KILL 300 bash scripts/check_fprime_foundation_parity.sh
    timeout --signal=KILL 300 formal/nightstream-fprime/scripts/validate.sh static
    timeout --signal=KILL 300 formal/nightstream-fprime/scripts/validate.sh build NightstreamFPrime NightstreamFPrimeTests

The three artifact-driven wide unit tests remain ignored without their emitter inputs. I did not rerun the complete fresh two-fold Lean/native golden workflow, all 2.6 billion matrix entries, the ignored CPU recursive benchmark, or every workspace test. The compact retained golden archive does not include private terminal openings; its replay alone cannot repeat terminal opening verification. The recorded historical checks remain evidence from their named runs, not fresh results of this review.

**Minimal consolidation plan**

The recommendation follows these assumptions, challenged against the task rather than inherited from the current layout:

| Assumption | Necessary or conventional? | Decision |
|---|---|---|
| Preserve accepted protocol semantics, state binding, and low-norm obligations | Necessary for correctness | Retain and check them during consolidation |
| Keep old package versions working | Not required; explicitly contrary to your development policy | Remove compatibility/emitter paths |
| Keep a second app route because it already exists | Convention, not a protocol requirement | Require one assembly contract |
| Retain independent conformance checks | Necessary at the Rust/Lean boundary | Keep the checks; relocate them out of deprecated code |
| Use the old lifecycle to parse current golden proofs | Incidental tool ownership | Remove that dependency |
| Treat two execution backends as two protocols | Incorrect assumption | Preserve a shared relation and conformance contract |
| Keep every historical proof module forever | Not established | Retain shared lemmas only while an actual selected proof consumes them |

A practical order is:

1. **Make the selected emitter the only maintained package entrypoint.** Remove old command dispatch, old exported application builders, obsolete schema consumers, and stale reproduction instructions. Preserve shared codec helpers.
2. **Move current golden checking out of neo-fold-legacy.** Then remove deprecated workspace support and adapters that no maintained consumer needs. Use Git history for old implementations.
3. **Choose one application assembly route.** Given your preference, the smallest immediate design is the generic route; retaining compact application lowering should be done through a common mechanism, not a package-identity branch.
4. **Choose and wire one witness execution route through public prove/extend.** Preserve cached application values, but do not let their presence choose a different implementation.
5. **Fix the logical-coordinate mutation and add the omitted focused Rust gates.** Reuse the existing tests and fixtures.
6. **Regenerate the selected package and source-bound evidence after consolidation.** Check the same package, same proof source, same transcript, and same terminal relation, then obtain the project's required current review acceptance.

**Acceptance criteria for a follow-up change**

- There is one documented package builder and one selected sampler schedule.
- A supported current validation command does not require the deprecated lifecycle.
- The public API uses the intended witness implementation; cached app values do not silently select an older path.
- A generic application's binding test mutates its actual retained application coordinates and rejects a detached/recommitted false application state.
- Cheap sampler, projection, matrix, and transport controls run in CI.
- The final source, selected artifacts, local Lean gate results, and current review requests identify the same revision and scope.
- Claims about speed, concrete security, and completeness stay within the measured or proved result.
