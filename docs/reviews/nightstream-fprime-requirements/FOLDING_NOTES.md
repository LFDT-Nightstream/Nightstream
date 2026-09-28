# Independent source review: folding requirements

This review maps SuperNeo v1_1 to the current PiCCS, PiRLC, PiDEC and NIFS owners. The result is [folding.json](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-requirements/folding.json). It contains 152 nodes and 123 leaves. These counts describe the result; they are not acceptance limits.

The source scope was the full local [Section 6](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/superneo-paper-v1_1/06_strong_and_weak_interactive_reductions.md:1), full [Section 7](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/superneo-paper-v1_1/07_superneo_folding_scheme_for_ccs.md:1), [Appendix B.1–B.4](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/superneo-paper-v1_1/11_appendix_B_deferred_theorems_and_proofs.md:1), and the cited current Lean and Rust owners. No frozen project was inspected. No build, test, backend, upload, external call or production edit was made. Only these review files were written. The root agent owns validation.

Each indexed leaf requires the equation or check for **every** indicated source, matrix, coefficient, round or child. An indexed family is not a sample. Foundation operations are dependencies on the F branch. The folding branch states their protocol use rather than restating their primitive proofs.

`proved` means that a cited theorem proves the stated local fact under its premises. `definition` records a specified operation or relation; it is not a proof. `connection` refers to the cited local consumer or owner contract. An open selected-row or native link does not make every primitive open. `assumption` marks an explicit security or extraction premise. `recorded_only` includes existing parity tests and earlier reported diagnostics that this agent did not run.

## Paper coverage

Line numbers below refer to the linked source for that section. Each JSON leaf also carries its exact paper and code references.

| Paper section and conjunct | Requirement nodes |
|---|---|
| §6, Definition 16, lines 3–15: complete/public coin; efficient weak extraction; fixed phi and witness agreement | `N.security.weak`, `R.prover.complete`, `R.security.fork`, `R.security.probability`, `R.security.unique` |
| §6, Definition 17, lines 17–35: complete/public coin; same output phi with probability one; extraction under relaxed-output success and witness agreement | `N.security.strong`, `C.security.phi`, `C.security.extraction`, `C.security.probability` |
| §6, Theorem 12, lines 37–42: strong–weak composition with the same phi | `N.security.same_phi`, `N.security.composition` |
| §7.1, Definition 18, lines 5–7: selected matrices, polynomial and dimensions | `N.profile.relation`, `N.profile.shape`, `C.sumcheck.degree` |
| §7.1, Definition 19, lines 11–13: packed fresh assignment, public prefix, commitment, strict norm and CCS relation | `C.input.fresh`, `C.prover.ccs`, `C.prover.norm`; F packing, commitment and public projection dependencies |
| §7.1, Definition 20, lines 15–21: prior CE commitment/public opening, norm, Pad Eval_K and matrix Eval_A | `C.input.running`, `C.prover.pad`, `C.prover.matrix`, `C.prover.eval_k`, `C.prover.eval_a` |
| §7.2, Definition 21, lines 29–33: fields and dimensions, counts, strong set and expansion inequality, relaxed binding, public linear map, selected structure | `N.profile.*`, `R.prover.bound`, `R.sampler.member`; F field, ring, strong-set and commitment dependencies |
| §7.3, lines 45–57: fresh/prior inputs, output shape, shared setup and encoder | `C.input.*`, `C.output.*`, `N.profile.setup` |
| §7.3, step 1, line 61: alpha and gamma | `C.transcript.input`, `C.transcript.alpha`, `C.transcript.gamma` |
| §7.3, step 2, lines 62–72: Pad/matrix tables and separate gamma index families | `C.prover.pad`, `C.prover.matrix`, `C.prover.eval_k_index`, `C.prover.eval_a_index` |
| §7.3, step 2, lines 74–84: fresh CCS, all-source norm, Eval_K, Eval_A and joint Q | `C.prover.ccs`, `C.prover.norm`, `C.prover.eval_k`, `C.prover.eval_a`, `C.prover.joint` |
| §7.3, step 2, lines 88–94: initial claim, sumcheck, Boolean total, endpoint | `C.sumcheck.*`, `C.transcript.round` |
| §7.3, steps 3–4, lines 96–100: full output messages and constant-term interpretation | `C.output.eval_k`, `C.output.eval_a`, `C.prover.constant`, `C.transcript.output` |
| §7.3, step 4, lines 102–106: four terminal terms and final identity | `C.terminal.*`, `C.sumcheck.endpoint` |
| §7.3, step 5, line 108: preserve source fields and witnesses, use new common point | `C.output.preserve` |
| §7.3, Lemma 7, line 110: strong reduction | `C.security.*`, `N.security.strong` |
| §7.4, lines 122–140: CE batch, common point, output stage, common setup/encoder, packing and transforms | `R.input.batch`, `R.combine.output`, `N.profile.setup`, `R.prover.assignment`; F packing and transform dependencies |
| §7.4, step 1, lines 142–146: sample/send rho and compute commitment, public, Eval_K and Eval_A sums | `R.sampler.*`, `R.combine.*` |
| §7.4, steps 2–3, lines 150–152: witness sum and combined output | `R.prover.assignment`, `R.prover.opening`, `R.combine.output` |
| §7.4, Lemma 8, line 154: weak reduction | `R.security.*`, `N.security.weak` |
| §7.5, lines 166–176: combined parent, child shape, common setup and encoder | `D.verify.shape`, `D.verify.output`, `N.profile.setup` |
| §7.5, step 1, lines 182–188: signed witness split, child commitments, Pad and matrix evaluations | `D.prover.*` |
| §7.5, step 2, lines 190–192: reject unbounded public parent; verifier-owned coordinate split | `D.public.*` |
| §7.5, step 2, lines 196–198: commitment, Eval_K and Eval_A recomposition; reject any failure | `D.verify.commitment`, `D.verify.eval_k`, `D.verify.eval_a`, `D.verify.decision` |
| §7.5, step 3, line 200: exact public child tuple | `D.verify.output` |
| §7.5, Theorem 13, line 202: reduction of knowledge and composition | `D.security.*`, `N.local.*`, `N.security.composition` |
| Appendix B.1, lines 3–41: split adversary, preserve phi and use weak extraction | `N.security.same_phi`, `N.security.weak`, `N.security.aligned_fork` |
| Appendix B.1, lines 50–70: reconstruct public-coin suffix, strong extraction, completeness and knowledge claim | `N.security.strong`, `N.security.composition`, `N.local.complete` |
| Appendix B.2, Lemma 9, lines 76–126: Q/T identity iff four separate obligations; coefficient blocks, norm zero test and coefficient-map bijection | `C.security.four_obligations`, `C.prover.*`; F zero-test and coefficient-evaluation dependencies |
| Appendix B.2, Lemma 10, lines 134–168: honest sumcheck, full output evaluation, terminal equality, valid outputs, public coin | `C.sumcheck.honest_sum`, `C.sumcheck.degree`, `C.transcript.round`, `C.output.*`, `C.terminal.*`, `N.local.complete` |
| Appendix B.2, lines 174–195: same phi, relaxed-output/witness-agreement premise, returned source witnesses and expected time | `C.security.phi`, `C.security.extraction`, `C.security.probability` |
| Appendix B.2, lines 217–307: invalid-source cases, nonzero residual, mixing/sumcheck bounds and conditional probability argument | `C.security.four_obligations`, `C.security.mixing`, `C.security.sumcheck`, `C.security.probability` |
| Appendix B.3, Lemma 11, lines 313–327: linear opening equations, norm growth and public coin | `R.combine.*`, `R.prover.*` |
| Appendix B.3, lines 353–417: coordinate forks, inverse difference, expected calls and sampling loss | `R.security.fork`, `R.security.coordinate`, `R.security.probability`; F strong-set dependencies |
| Appendix B.3, lines 425–465: extracted commitment/public, Eval_K, Eval_A and ambient norm | `R.security.commitment`, `R.security.eval_k`, `R.security.eval_a`, `R.security.ambient` |
| Appendix B.3, lines 477–504: same-phi uniqueness, norm<2B differences and relaxed-binding collision | `R.security.unique`, `N.security.binding` |
| Appendix B.4, lines 508–536: honest public/witness split, commitment/Pad/matrix recomposition, child bounds and no random challenge | `D.public.*`, `D.verify.*`, `D.security.complete`, `D.security.public_coin` |
| Appendix B.4, lines 542–564: simulate accepted outputs, obtain valid child witnesses, recompose parent witness/public/evaluations and bound | `D.security.extract`, `D.security.children`, `D.security.bound`, `D.security.identity` |

## Connection and security limits

The actual PiCCS phase is proved for arbitrary selected rows and their actual public input. Actual PiRLC sum equations and an actual PiDEC local phase are also proved. These are useful local results. They do not yet prove every decoded value equals the exact NIFS verifier value.

The remaining connection sequence is explicit in `R.connection.challenges`, `D.connection.parent`, `D.connection.messages`, `D.connection.output`, and `N.binding.context`. The first missing folding equality is actual rho equals the verifier-derived rho. Its direct sampler owners already prove ordinary-row custody, but the full bridge retains Encodes and source/semantic premises. After that equality, use the existing public sum and attempt/output theorems. [ActualStep.selectedRowsAndPublic_step_iff_baseOrPiDec](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualStep.lean:350) exposes the remaining PiDEC check and exact output equality. It uses the decoded context, which still needs identification with the verifier-selected context. These facts identify unfinished proofs; they do not establish that a broad refactor is necessary.

The selected PiCCS transcript uses the prior-state digest instead of directly absorbing all running point/evaluation data. Lean obtains that digest from the fresh public input; native code checks the carried running digest and common point. Keep the authenticated prior-state boundary. Native arithmetic can run without FPrime fixed-point construction or export, but an independent soundness claim still needs authenticated inputs or the retained proof boundary. No second transcript, relation identity, state digest or input schema is needed by this hierarchy.

The native PiRLC wrapper also performs an extra projection transcript update after rho and the public combination. The data is absent from the NIFS output; PiDEC draws no further challenge and the next PiCCS resets. This is a source-level schedule difference. The review did not establish changed protocol outputs or an exploit.

[PaperSecurityComposition.accepted_implies_securityOutcome](/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/Nifs/PaperSecurityComposition.lean:533) proves a deterministic case split. It retains mixing, sumcheck, parent-binding, forking and child-opening failures. The module explicitly excludes probabilistic forking, Fiat–Shamir security, commitment binding and child extraction from its ownership. Do not report that theorem as a complete probability or expected-time security proof.

The foundation branch separately records extension-field irreducibility as partial and low-norm invertibility as an assumption. The actual norm expansion theorem bounds coefficients by 216 for valid five-symbol challenges acting on norm<2 assignments. It does not prove a universal norm ratio. The ambient bound uses `floor(q/2)+1` with a strict Nat comparison, which represents the paper strict rational bound for odd q. This does not change the selected production policy: `b=2`, `k_rho=16`, `B=65536`.

## Test evidence

The root agent reported these same-source optimized runs under the 300-second cap, with `--release --locked --offline` and an empty `RUSTC_WRAPPER`: `nifs_fixed` passed 3/3; `nifs_round_trip` passed 9/9; `nifs_r1cs_isolated` failed because its running batch was empty. The successful fixtures use [toy_instance](/Users/nicarq/starstream/develop/nightstream-clean-up/crates/neo-fold-clean/tests/support/mod.rs:30), which ignores its seed and returns zero assignments. They do not establish production nonzero conformance. Root holds `/tmp/nightstream-goal-splits-native.log` and `/tmp/nightstream-goal-splits-round-trip.log`.

Separate Lean/Rust phase parity tests exist and earlier honest PiCCS diagnostics were recorded in the same-source review. This agent did not run them. Some fixtures construct synthetic opening messages or use zero relation/key values. `N.conformance.chain` therefore stays open. The unrelated package-loader run reported by root failed on stored LFS pointer files; it is not evidence against folding equations. No PaperExact invocation was made for this review.
