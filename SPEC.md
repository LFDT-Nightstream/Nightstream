# Nightstream specification

## 1. Status and authority

This document describes the protocol that the `nightstream` crate runs. Each step is HyperNova recursion over SuperNeo v1.2 folding. There are two step circuits, a light circuit and a refresh circuit (section 2.5, Algorithm 3). Section 7 adds Nebula memory checking, and Algorithms 7 and 8 compress the final proof. The Lean packages in `formal/nightstream-fprime/` and `formal/compression-lean/` are normative. This document restates them for readers outside the project, such as an auditor or a second implementer. It is not a second authority. Where the two differ, Lean is correct and this document has a defect.

Rust in `crates/` implements the protocol and is not proved. Lean emits the circuit packages. Rust reads them from the caller and compares their identity with a pinned value (Algorithm 2). A statement that only Rust fixes is marked "(not checked against Lean)".

Lean proves these results. `formal/nightstream-fprime/SECURITY_MODEL.md` gives the adversary of each result and what you must trust.

- If the rows of the light circuit hold and its public input encodes a digest, the light case of the step relation holds on the step that the assignment decodes to (`Export.OptionB.StepRelation.lightEnv`), with that digest as its output hash. The refresh circuit has the same result (`Export.OptionB.StepRelation.light_stepHolds` (`b7e4cd56feac24c6`), `Export.OptionB.StepRelation.refresh_stepHolds` (`f9a9a0f752cfbb3e`)). No hardness premise is used. Both results take a premise about the circuit that the application selects (Algorithm 3).
- If the terminal relation accepts a proof, and the key of each circuit carries the relation that Lean exports for that circuit, then the circuit that ran last holds its step relation under these keys on the step that the fresh witness decodes to, and that step ends in the terminal's statement, or the decoded next state and the terminal's state collide (`Export.OptionB.TerminalBridge.bridge` (`22d55afec87191ff`), `Export.OptionB.TerminalBridge.Collision`). The decode is a function of the proof (`Export.OptionB.TerminalBridge.Exports.decode`), so the collision names two states that a reduction computes from the proof. The step's circuit selector returns the circuit that ran on a valid step (`Export.OptionB.TerminalBridge.chosen`), so the prover picks the circuit, as Rust does from the slot depth, and the decoded step's witness is valid for the application. The terminal does not read the selector (`Export.OptionB.TerminalBridge.holdsFor_recursive_control` (`5186c151886d3946`)). Its premises hold for the reference packages, whose two circuits run one application (`SECURITY_MODEL.md`, trust item 4).
- From two accepted outputs of one light proof that differ only in the challenge `ρ`, extraction gives a fresh witness with norm below `b` and a witness of the input accumulator at the output bound plus `T(b − 1)`, or a Π_CCS bad event occurred (`Spec.Folding.Light.accepted_implies_securityOutcome` (`221e504198809b6f`)). For a refresh fold, a fork of its challenges gives the fresh witness and a witness of the input accumulator below `B`, or a Π_CCS bad event occurred (`Spec.Folding.Refresh.accepted_implies_securityOutcome` (`c4569719a78d68b7`)). These results are deterministic. Section 3 gives the error numerators of the bad events.
- Along an accepted history of one slot, the bound at which extraction recovers each accumulator is at most `B + cap·T·(b − 1)` (`Spec.HyperNova.Construction2.BoundedDepth.accepted_bounds` (`55d6d9921f728747`), computed on the recovered bounds; `Spec.HyperNova.Construction2.BoundedDepthKnowledge.execution_extracts` (`4288991b81eaa3d9`) gives the extraction). A relaxed binding collision at twice such a bound gives a kernel vector of the Ajtai key with norm below the chain norm 137,134,080 (`Spec.Phi81Relation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel`, `Spec.Folding.Schedule.production_msisChainBound` (`9dd6b989fb98393c`)).
- Under HyperNova errata Assumption 1 for the light and refresh folds, the closure premise for the reverse extractor's stages, and an admitted IVC adversary (`SECURITY_MODEL.md`, trust item 3), the probability that the terminal relation accepts a statement and a proof but no `iteration` valid steps of the application, each with a witness of the application's word count, lead from `z0` to `zi` is at most the sum over the reverse extractor's stages of the stage's error and the probability of a Poseidon2 collision at the stage's visit (`Export.OptionB.IvcSecurity.probability_bound` (`b55ed0319a8c26bf`)). The reverse extractor walks back from the terminal one step per stage: the fresh witness decodes to the last step, whose fold of the selected slot is the stage's attempt, and the extractor's answer gives the prior visit (`Export.OptionB.IvcWalk.covered` (`cb5e37ca653b6dc8`)). Each slot keeps its own witness bound, which grows by `T(b − 1)` per light fold and returns to `B` at a refresh fold, so every attempt stays within `B + cap·T·(b − 1)`. The collision at a visit is on two states that the stage computes from its tape (`Export.OptionB.TerminalBridge.Collision`). Each attempt is linked to the prior state of its step (`Lifecycle.OptionB.PriorLink`): two folds with one fresh instance that link to well-formed prior states start from one running claim, or the two prior states give a state-hash collision (`Lifecycle.OptionB.priorLink_binds` (`61b37f32b4818d2c`)). In a random-oracle model of the two folds, Assumption 1 holds with an explicit error: the folds' knowledge errors and a truncation loss for an extractor with a fixed number of reruns (`Lifecycle.OptionB.RandomOracle.Assumption1.romAssumption1` (`db98573ce8c40a75`)). The knowledge error of each fold is a statistical term, at most `(Q + 58) · 531/2^128` for light and `(Q + 74) · 50/(87 · 2^115)` for refresh with `Q` oracle queries, plus the probabilities of a short kernel vector of the Ajtai key and of a collision on two named prior states (`Lifecycle.OptionB.RandomOracle.LightKnowledge.knowledge_error_le` (`773c8d8550dbf4c8`), `Lifecycle.OptionB.RandomOracle.RefreshKnowledge.knowledge_error_le` (`7b4812c504978421`)). Averaged over the SHAKE128 key setup, the kernel term is the advantage of an explicit MSIS solver on a uniform matrix at the chain norm and the collision terms are the advantage of an explicit Poseidon2 collision finder, plus a programming error below `2^−192` (`Export.OptionB.RandomOracleSetup.light_production_knowledge_error_lt_solvers` (`b30a2f31ebb7dd36`), `Export.OptionB.RandomOracleSetup.refresh_production_knowledge_error_lt_solvers` (`1c59f61f358c930d`)). At an oracle that answers like the Poseidon2 sponge, the model's events are the walk's (`Export.OptionB.RomWalk.succeeds_iff_walkSucceeds` (`6d63b1cc5c29007b`)). The step circuits recompute the challenges with Poseidon2, so the IVC bound keeps Assumption 1 for the concrete hash as a premise (`SECURITY_MODEL.md`, trust item 3). The bound holds for every depth, but each stage's adversary reruns the earlier stages, so it gives no concrete security for a long history. HyperNova proves knowledge soundness only for a constant number of steps (`SECURITY_MODEL.md`, trust item 3). For one slot, the probability that an accepted history is not returned is at most the sum of the stages' errors (`Spec.HyperNova.Construction2.BoundedDepthSecurity.history_failure_le` (`a3fc56cd24a9a455`)), and an accepted execution of both slots makes each slot's history accepted (`Spec.HyperNova.Construction2.BoundedDepthSecurity.slot_accepted` (`7732ad00c2d7b3f4`)).
- An honest light fold and an honest refresh fold are accepted (`Spec.Folding.Light.complete` (`299333a0c5be08f2`), `Spec.Folding.Refresh.complete` (`fef35ffcbe47f8c9`)). At the production keys, the honest fold of a slot claim is accepted by the option-B folds, and its output is a slot claim at the input bound plus `T(b − 1)` after a light fold and below `B` after a refresh fold (`Lifecycle.OptionB.FoldCompleteness.light_folds` (`a36c07bbd2a7bf67`), `Lifecycle.OptionB.FoldCompleteness.refresh_folds` (`20a4a024edcc71d0`)). The light challenge always exists, because the Poseidon2 sponge is a deployed oracle. Along a schedule that the light circuit accepts, the bound `accumulatorBound` of each depth stays below `B` (`Spec.Folding.Schedule.honest_bounds` (`82d0559dff9c6fc8`)). Along an honest run, each slot's witness stays within the bound of its depth, so it stays below `B` (`Export.OptionB.StepExtend.Honest`). Given an accepted fold, an honest light step and an honest refresh step each have an assignment in which every row of the step circuit's source layout holds (`Layout.OptionB.LightAssemblerCompleteness.complete` (`d1f37ffaecfb8430`), `Layout.OptionB.RefreshAssemblerCompleteness.complete` (`03059f5579db6f8a`)). From such an assignment, the light and refresh keys accept a fresh instance whose public input is `encHash` of the step's output hash: the commitment to an honest assignment of the exported relation, read through the package's slot layout widened to the key width (`Export.OptionB.StepCompleteness.light_complete` (`b57559fdfcb47c93`), `Export.OptionB.StepCompleteness.refresh_complete` (`48e12535dfdfdf9f`)). The honest assignment sets to 0 each bit slot that no constraint `x * (x - 1)` forces, and no constraint reads those slots. This result takes four computed checks of the package as premises: the blocks pack, every piece reads only columns below the first fresh column, no piece reads such a zero slot, and no row of the circuit reads a public column. The emitter evaluates them and writes no step package when one fails (`Export/OptionB/EmitMain.lean`). For every valid run of the application from `z0`, the terminal accepts the honest prover's proof (`Export.OptionB.RunCompleteness.run_complete` (`36dec510f922702a`)). The first step folds the default fresh instance into the default slot with HyperNova's zero proof (`Lifecycle.OptionB.BaseFold.zero_folds` (`55dd84ac8c07b5b9`)). Each later step folds the fresh instance into the selected slot, with a light fold below the cap and a refresh fold at it, as Rust does (`Export.OptionB.StepExtend.light_extend` (`fb24ce30a5a5a389`), `Export.OptionB.StepExtend.refresh_extend` (`8fec9647ad38a299`)). The premises are the four package checks, keys that carry the exported relations at the packages' widened slot layouts (`Export.OptionB.StepExtend.Complete`), key digests of four words, and witnesses of the application's witness length. For the finish step, an honest finish fold gives values on which every row of the finish circuit holds at the refresh step's source layout (`Layout.OptionB.FinishAssemblerCompleteness.complete` (`942dca7528c1dfd7`)). From these values, the final program's checks on the finish package accept the proof's statement: the package's rows and Poseidon2 blocks hold, the key words are the setup's key digests, the decoded step passes the finish checks, and both next slot claims hold with witnesses within the accumulator bound of the light cap (`Export.OptionB.FinishCompleteness.package_complete` (`457582368f0bec07`), `Export.OptionB.FinishCompleteness.finish_complete` (`41632b5abbcba31f`)). This holds for every valid run of at least one step (`Export.OptionB.FinishCompleteness.run_finish_complete` (`1a336065fc38a317`)). It takes three computed checks of the finish package as premises: every piece reads only columns below the first fresh column, no row of the finish circuit reads a public column, and the shared words lie below the first fresh column. The emitter evaluates them and writes no finish package when one fails (`Export/OptionB/EmitMain.lean`). Along an honest run, each slot witness is zero on every ring block from its circuit's package width, and both next slots of the finish step keep this (`Lifecycle.OptionB.BlockSupport.Supported`, `Export.OptionB.StepExtend.Honest`). For each slot, the converse of the layer-1 parent result puts the honest slot claim in its layer-1 relation, given the norm bound and the block support (`CompressionLean.Final.parent_complete` (`e9746d783374ba4e`), `CompressionLean.Final.OptionB.slot_complete` (`d1dfca9e6d50d15f`)). A check of Rust's key files gives the two premises that join the parts: the depth-64 accumulator bound is at most the layer-1 norm bound, and the package's logical width fits in the blocks that layer 1 commits (`CompressionLean.Claims.option_b_covers_of_check` (`1137f679617ebd49`)). So for every valid run of at least one step, the claim of the final verifier holds for the honest statement (`CompressionLean.Final.OptionB.holds_of_honest` (`1bd53a4c86cad485`), `CompressionLean.Claims.option_b_complete` (`5a660cc941cecedf`)). Its premises are those of `run_complete`, the three finish package checks, the setup and key premises of the soundness claims, and the cover check for each slot. This is completeness at the claim level. No Lean result says that the honest layer-1 and shrink provers produce proofs that the verifiers accept. The Rust tests of lifecycle rule 2 are the evidence for that part (`SECURITY_MODEL.md`, Out of scope).
- If the final verifier accepts, a last fold exists whose finish checks hold and whose two slot claims hold under their circuits' relations (SuperNeo v1.2 Definition 21), except with probability `calls / 2^t` in the state-restoration experiment at Rust's layer bits `t` (`CompressionLean.Claims.option_b_parent_bound` (`20d4a178d46ed614`)). For the reference application, each premise of this result is met by a Lean proof, a Lean check of Rust's files, Rust, or a Lean value that compiled code evaluates and the kernel does not check (`CompressionLean.Claims.option_b_poseidon2_bound` (`a74f67c4cb3e5d99`)). For another application, the premises about its keys and setups are open (`formal/compression-lean/STATUS.md`). Idealized WHIR openings, Poseidon2 as a random oracle, Fiat–Shamir inside the final program and a key root that is the honest root of the claims' setup are assumptions (`formal/compression-lean/TRUST.md`, sections 3 and 5).
- The last fold of an accepted final proof starts the reverse walk of the IVC bound. If an answer extracts the finish fold, the terminal relation accepts the finish step's input with the terminal proof that the answer gives (`Export.OptionB.FinishWalk.holdsFor_of_extracted` (`9fbe38653171e865`)). So, if no `iteration` valid steps lead to that statement, the answer does not extract the finish fold, or the finish output is a false acceptance of the IVC bound (`Export.OptionB.FinishWalk.finish_covered` (`429fb2520aea2063`)). Except with probability `calls / 2^t`, each statement that the final verifier accepts has a last fold whose finish checks hold and whose next slots hold under their circuits' relations (`CompressionLean.Claims.option_b_last_fold_bound` (`db62b2bbc63e4f7b`)). One experiment joins the two bounds. It draws an outcome of the final verifier's experiment, then the coins of the finish stage's extractor, then the coins of the reverse extractor's stages. In this experiment, the probability that the final verifier accepts a statement whose iteration is at most `depth` and that no valid steps reach is at most `calls / 2^t`, plus the Assumption 1 error of the finish stage, plus the terms of the IVC bound (`CompressionLean.Claims.option_b_ivc_bound` (`e63fe95ae879be36`)). Its premises are those of the two bounds and two more: the finish adversary and the chain adversary are admitted. These two adversaries select the last fold with `Classical.epsilon`. The compression claims prove that the last fold exists, not that an algorithm computes it from the final proof, so Lean does not show that the two adversaries are efficient (`SECURITY_MODEL.md`, adversary 6).
- If a chain of valid memory steps ends in a state that passes the terminal checks of the memory verifier, the chain gives a run that the memory model accepts, or a state-digest collision occurred (`Lifecycle.Nebula.accepts_or_collision` (`31d12d5e456bee76`)). Each honest execution has a history whose statement passes the statement checks of the memory verifier, Algorithm 10 steps 3 and 4 (`Lifecycle.Nebula.MemoryApp.history_of_execution` (`a1ef468b8bc6e701`)). For every such execution, the option-B terminal accepts the honest prover's recursive proof of that statement, which is the Algorithm 6 branch of step 5 (`Export.OptionB.NebulaCompleteness.memory_complete` (`20b62362ffe2ee1c`)). Its premises are those of `run_complete`, with the memory program as the exported program. Lean has no result for the final-proof branch of step 5. Under Assumption 1, the closure premise and the memory-round premise, the probability that the memory verifier accepts but the history that the reverse extractor decodes does not attest the claimed memory statement is at most the terms of the IVC bound, the state-link and chain collision probabilities of the extracted runs, and the memory-round error (`Export.OptionB.NebulaSecurity.bad_le` (`8714ad3d96df7396`), `SECURITY_MODEL.md`, adversary 5). In a random-oracle model of `η`, for a secure geometry and every `η` sampler with `Q_η` queries, the event of the memory round has probability at most `(Q_η + 1) · 2^−109.91` plus a retry term in which each counted pair is a Poseidon2 transcript collision (`Export.OptionB.NebulaSecurity.romMemoryRound` (`592310c00928ad64`)). This does not bound the deployed check, whose `η` is Poseidon2.

Every audited theorem uses only the axioms `propext`, `Classical.choice` and `Quot.sound`. The repository claims no single security level for the system (`SECURITY_MODEL.md`, Out of scope).

These commands check the development and this document. Run the Lean commands on a machine with the Lean toolchain of `formal/nightstream-fprime/lean-toolchain`.

```bash
cd formal/nightstream-fprime && bash scripts/validate.sh axioms && bash scripts/validate.sh compression
python3 scripts/check_spec.py
```

`validate.sh axioms` builds the tests and runs `formal/nightstream-fprime/tests/EndpointCensus.lean`. `validate.sh compression` builds `formal/compression-lean/` and runs its census. That census fails when this document cites a name in the namespaces `Export`, `Layout`, `Lifecycle` or `Spec` that does not exist, a theorem without a pin, or a pin with another fingerprint. `formal/compression-lean/CompressionLeanTests/SpecCensus.lean` makes the same check for names in the namespace `CompressionLean`. A pin is a theorem name followed by its fingerprint, such as `` `Name` (`0123456789abcdef`) ``. The fingerprint hashes the theorem's statement and every package definition that the statement reaches, so a change to one of them changes it. A change to a proof does not. When a pin fails, reread the section that cites it, then write the new fingerprint that the census prints. `scripts/check_spec.py` fails when a pin is malformed, when a backticked path does not exist, or when section 3's table is stale. CI runs `check_spec.py` and no Lean.

Names of the F′ package omit its namespace `NightstreamFPrime`. A Lean path in this document is under `formal/nightstream-fprime/NightstreamFPrime/` or `formal/compression-lean/CompressionLean/`.

| Section | Lean |
|---|---|
| 2, notation | `Spec/Algebra.lean`, `Spec/Phi81StrongSet.lean`, `Lifecycle/OptionB/State.lean`, `Spec/Folding/Light/Verifier.lean`, `Spec/Folding/Refresh/Verifier.lean` |
| 3, parameters | `Spec/Profile.lean`, `Spec/Folding/Schedule.lean`, `Spec/Folding/Light/ErrorBudget.lean`, `Export/OptionB/StepPackage.lean` |
| 4, encodings | `Lifecycle/XOut.lean`, `Lifecycle/OptionB/State.lean`, `Export/OptionB/StepPackage.lean` |
| 5, transcript | `Spec/Poseidon2.lean`, `Lifecycle/Transcript.lean`, `Lifecycle/OptionB/Keys.lean` |
| 6, algorithms | `Spec/AjtaiSetupV1.lean`, `Spec/Folding/Light/`, `Spec/Folding/Refresh/`, `Spec/HyperNova/Construction2/`, `Lifecycle/OptionB/`, `Export/OptionB/`, `Final/` |
| 7, Nebula memory | `Spec/Nebula/`, `Lifecycle/Nebula/`, `Export/Stage1/NebulaMemoryPackage.lean`, `Export/OptionB/NebulaSecurity.lean`, `Export/OptionB/NebulaCompleteness.lean` |

## 2. Notation

### 2.1 The field and the extension

- `q = 2^64 − 2^32 + 1`, the Goldilocks prime. `F_q` is `Spec.F`, the integers modulo `q`, each written as its value in `[0, q)`.
- `|x|` for `x ∈ F_q` is `min(x, q − x)` (`Spec.centeredMagnitude`). A vector `z` has norm below `β` when every entry has `|z_i| < β` (`Spec.normBounded`).
- `K = F_q[u]/(u² − 7)` (`Spec.K`). An element is `c_0 + c_1·u`, and `(a_0 + a_1 u)(b_0 + b_1 u) = (a_0 b_0 + 7 a_1 b_1) + (a_0 b_1 + a_1 b_0) u` (`Spec.K.mul`). 7 is not a square modulo `q`, so `K` is a field (`Spec.GoldilocksExtension.sevenProjectiveNonresidue` (`0bb4cd489773c8b0`)). `F_q` embeds in `K` as `c_0` (`Spec.K.embed`).

### 2.2 The ring and the challenge set

- `R_q = F_q[X]/(Φ_81)` with `Φ_81 = X^54 + X^27 + 1`, so the ring degree is `d = 54`. An element is its 54 coefficients (`Spec.RingF`). Multiplication reduces the product with `X^54 = −X^27 − 1` (`Spec.ringFMul`).
- `C`, the strong set, is the ring elements whose coefficients are all in `{−2, −1, 0, 1, 2}` (`Spec.Phi81StrongSet.ProductionMember`), so `|C| = 5^54`. Every nonzero ring element of norm at most `β` with `3β² < q` is invertible (`Spec.Phi81StrongSet.lowNormInvertibility` (`33d62ee657026b54`), SuperNeo v1.2 Theorem 4). The difference of two elements of `C` has norm at most 4, so it is a unit.

### 2.3 Commitments

- A witness `z ∈ F_q^m` packs into `⌈m/d⌉` ring elements: coefficient `h` of element `j` is `z_{d·j + h}`, or 0 past the end (`Spec.packAssignment`).
- The Ajtai key `A` has `κ` rows of `⌈m/d⌉` ring elements. The commitment of `z` is `A · pack(z) ∈ R_q^κ` (`Spec.ajtaiCommit`). Algorithm 1 derives `A`.

### 2.4 The relations

- Each step circuit has its own relation, a CCS over `F_q` with `t = 4` matrices `M_0, …, M_3` (`Lifecycle.ProductionKey.LogicalRelation`). Each matrix has its rows on the cube `{0,1}^ℓ` and `m` columns, and each row past the relation's `n` rows is zero. An assignment `z` satisfies the CCS when `f((M_0 z)_r, …, (M_3 z)_r) = 0` for every row `r`, where `f = M_0·M_1 − M_2 + M_3^7` (`Spec.ProductionRelation.polynomial`).
- Both circuit keys have the logical width `m = 2^26` (`Export.OptionB.StepRelation.keyWidth`). A circuit with fewer columns commits a witness that is zero past them.
- The public input of `z` is its first 270 entries, five ring elements (`Spec.projectPublicInput`, `Lifecycle.PaperAlgebra.publicRingColumns`).
- `eq(x, y) = ∏_i (x_i y_i + (1 − x_i)(1 − y_i))` for `x, y ∈ K^ℓ`. `ỹ(r)` is the multilinear extension of a table `y` on `{0,1}^ℓ`, evaluated at `r ∈ K^ℓ`.

### 2.5 Claims, slots and fold proofs

- A fresh claim is a CCS claim: a commitment `c ∈ R_q^κ` and a public input `x ∈ F_q^270`. Its witness satisfies the CCS of the circuit that made it, opens `c`, projects to `x` and has norm below `b`.
- An accumulator is a CE claim (SuperNeo v1.2 Definition 21). It has a commitment, a public input, a point `r ∈ K^ℓ` and one evaluation family (`Lifecycle.OptionB.SlotClaim`). The family has `Eval_K`, 54 elements of `K` (the Pad evaluations), and `Eval_A`, `4 × 54` elements of `K` (the matrix evaluations). Its witness opens the commitment, projects to the public input, has norm below `B`, and gives these evaluations at `r`.
- A slot is an accumulator and its depth, the number of light folds since its last refresh (`Lifecycle.OptionB.SlotData`). The state has two slots. Slot 0 accumulates the fresh claims of the light circuit, and slot 1 those of the refresh circuit.
- The default slot has the point 0, a zero commitment, public input and evaluations, and depth 0 (`Lifecycle.OptionB.defaultSlot`, HyperNova Definition 12).
- A light proof has `ℓ` sum-check round polynomials of degree at most 8, each its 9 coefficients in `K` from degree 0 up, then the Π_CCS output, one evaluation family for each of the 2 sources, fresh first (`Spec.Folding.Light.Proof`).
- A refresh proof has the 16 Π_DEC children of the input accumulator, each a commitment and an evaluation family, then `ℓ` round polynomials and the Π_CCS output for the 17 sources, fresh first, then children 0 to 15 (`Spec.Folding.Refresh.Proof`). The refresh verifier does not read the output children of the NIFS proof type that it reuses (`Spec.Folding.Refresh.verify`).

## 3. Parameters

The table below is generated from Lean. Its values come from the integer equations of seven Lean theorems, two Lean definitions, the headers of the Lean-emitted packages of the reference application `Poseidon2HashChainV1` (`decisions/fprime-stage1-poseidon2-hash-chain-v1.md`), and the row counts of the Lean-emitted parity file. The names of the header fields come from their Lean format `Export.OptionB.StepPackage.Package.format`. Another application has other packages, with the same profile but other sizes.

<!-- params -->
Generated by `scripts/check_spec.py`. Do not edit by hand: run `python3 scripts/check_spec.py --write`.

| Source | Lean | Symbol | Value | Meaning |
|---|---|---|---|---|
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.q` | `q` | 18446744069414584321 | The field modulus, the Goldilocks prime. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.b` | `b` | 2 | The decomposition base and the fresh norm bound. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.k` | `k_rho` | 16 | The decomposition digits, and the Π_DEC children. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.maxFresh` | — | 287 | The largest fresh count with `(maxFresh + k_rho)·T·(b − 1) < B` (SuperNeo v1.2 Definition 22). |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.expansionT` | `T` | 216 | The expansion factor of the challenge set. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.bigB` | `B` | 65536 | The combined norm bound, `b^k_rho`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.eta` | `η` | 81 | The cyclotomic index: `R_q = F_q[X]/Φ_η`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.ringDegree` | `d` | 54 | The ring degree, `φ(η)`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.commitmentWidth` | `κ` | 22 | The rows of the Ajtai key. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.extensionDegree` | — | 2 | The degree of `K` over `F_q`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.challengeSetBitsFloor` | — | 125 | A floor of `log₂ |C|`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.piRlcInputs` | — | 17 | The Π_RLC inputs of a refresh fold, the fresh claim and the 16 Π_DEC children. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.piDecChildren` | — | 16 | The Π_DEC children of the accumulator that a refresh fold splits. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.ccsMatrices` | `t` | 4 | The CCS matrices. |
| `Spec/Profile.lean`, `production_norm_stages` | `NormStage.bound productionGlobalParams .fresh` | — | 2 | The norm bound of a fresh witness, `b`. |
| `Spec/Profile.lean`, `production_norm_stages` | `NormStage.bound productionGlobalParams .combined` | — | 65536 | The norm bound of a Π_RLC parent, `B`. |
| `Spec/Profile.lean`, `production_norm_stages` | `NormStage.bound productionGlobalParams .ambient` | — | 9223372034707292161 | The ambient bound, `q/2 + 1`. |
| `Spec/Profile.lean`, `production_msis_norm_bound` | `productionGlobalParams.msisNormBound` | — | 113246208 | The MSIS norm `8TB` of SuperNeo v1.2 Section 8, the chain norm at cap 0. |
| `Lifecycle/Nifs/SelectedTestNumerator.lean`, `numerator_eq` | `numerator` | — | 4589 | The numerator of the refresh fold's Π_CCS test error, 17 sources. |
| `Spec/Folding/Light/ErrorBudget.lean`, `production_testNumerator` | `testNumerator (shape 28) 8` | — | 524 | The numerator of the light fold's Π_CCS test error, 2 sources. |
| `Spec/Folding/Light/ErrorBudget.lean`, `production_forkCoordinates` | `forkCoordinates` | — | 1 | The `ρ` coordinates of a light fork. A refresh fork has 17. |
| `Spec/Folding/Schedule.lean`, `production_msisChainBound` | `msisChainBound productionGlobalParams productionLightFoldCap` | — | 137134080 | The chain MSIS norm `8T(B + cap·T·(b − 1))`. |
| `Spec/Folding/Schedule.lean`, `productionLightFoldCap` | `productionLightFoldCap` | `cap` | 64 | The depth at which the light circuit rejects a fold. |
| `Export/OptionB/StepRelation.lean`, `keyWidth` | `keyWidth` | — | 67108864 | The logical width of both circuit keys, `2^26`. |
| computed | `ε_test` | `ε_test` | 4589/q² ≈ 2^−115.83 | The refresh fold's Π_CCS test error, `numerator / q²`. |
| computed | `ε_test light` | — | 524/q² ≈ 2^−118.96 | The light fold's Π_CCS test error, `testNumerator / q²`. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.circuit` | — | 0 | The circuit index: 0 light, 1 refresh, 2 finish. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.publicColumns` | — | 270 | The public input columns, five ring elements. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.width` | — | 16777216 | The source width. Every source column is below it. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.firstFresh` | — | 1105621 | The first fresh lowering column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.sboxStart` | — | 1110194 | The first S-box output column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.pieces` | — | 15601 | The pieces, which are lowered rows, ring products and Poseidon2 blocks. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.witness` | — | 21 | Witness batches. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.slots` | — | 1040 | Slot runs of the low-norm layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.application[0]` | — | 4 | The witness word count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.application[1]` | — | 17746 | The first local of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.application[2]` | — | 5480 | The local count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.application[3]` | — | 9 | The pieces of the application region, from the first piece. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-light-v1.json` | `light.application[4]` | — | 0 | The fresh columns of the application region, from `firstFresh`. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.circuit` | — | 1 | The circuit index: 0 light, 1 refresh, 2 finish. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.publicColumns` | — | 270 | The public input columns, five ring elements. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.width` | — | 16777216 | The source width. Every source column is below it. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.firstFresh` | — | 4537892 | The first fresh lowering column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.sboxStart` | — | 4577406 | The first S-box output column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.pieces` | — | 64122 | The pieces, which are lowered rows, ring products and Poseidon2 blocks. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.witness` | — | 180 | Witness batches. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.slots` | — | 4738 | Slot runs of the low-norm layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.application[0]` | — | 4 | The witness word count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.application[1]` | — | 57814 | The first local of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.application[2]` | — | 5480 | The local count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.application[3]` | — | 9 | The pieces of the application region, from the first piece. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-refresh-v1.json` | `refresh.application[4]` | — | 0 | The fresh columns of the application region, from `firstFresh`. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.circuit` | — | 2 | The circuit index: 0 light, 1 refresh, 2 finish. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.publicColumns` | — | 270 | The public input columns, five ring elements. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.width` | — | 16777216 | The source width. Every source column is below it. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.firstFresh` | — | 4537892 | The first fresh lowering column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.sboxStart` | — | 4577406 | The first S-box output column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.pieces` | — | 63748 | The pieces, which are lowered rows, ring products and Poseidon2 blocks. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.witness` | — | 180 | Witness batches. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.slots` | — | 4380 | Slot runs of the low-norm layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.application[0]` | — | 4 | The witness word count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.application[1]` | — | 57814 | The first local of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.application[2]` | — | 0 | The local count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.application[3]` | — | 0 | The pieces of the application region, from the first piece. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-finish-v1.json` | `finish.application[4]` | — | 0 | The fresh columns of the application region, from `firstFresh`. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-parity-v1.json` | `light.relationRows` | — | 36930 | R1CS rows of the light relation, from the pieces, the ring products and the bindings. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-parity-v1.json` | `light.sboxRows` | — | 147900 | S-box rows of the light relation, 150 for each Poseidon2 block. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-parity-v1.json` | `refresh.relationRows` | — | 197013 | R1CS rows of the refresh relation, from the pieces, the ring products and the bindings. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-parity-v1.json` | `refresh.sboxRows` | — | 599550 | S-box rows of the refresh relation, 150 for each Poseidon2 block. |

A probability is shown as `2^−x`, where `x` is `−log₂` of the exact fraction rounded down to two decimals, so the shown bound is never below the true value.
<!-- /params -->

Both folds use one Π_CCS shape. It has `ℓ = 28` rounds (`Lifecycle.cubeVariables`), 4 matrices, `d = 54` and round polynomials of degree at most 8. A light fold has 1 fresh source and 1 running source (`Spec.HyperNova.Construction2.OptionBFolds.lightShape`). A refresh fold has 1 fresh source and 16 children. The light relation has `36,930 + 147,900 = 184,830` rows and the refresh relation has `197,013 + 599,550 = 796,563` rows (computed from the table). Both are below `2^ℓ`.

The light circuit rejects a fold of a slot at depth `cap = 64`. An honest accumulator at depth `j` has norm at most `accumulatorBound j = (1 + j + k_rho)·T·(b − 1)` (`Spec.GlobalParams.accumulatorBound`, Cyclo, ePrint 2026/359, Theorem 3). Lean proves the step for one light fold (`Spec.Folding.Light.complete` (`299333a0c5be08f2`)) but not the bound over a schedule. This bound is below `B` for each `j ≤ 286` (`Spec.GlobalParams.production_accumulatorBound_lt_bigB` (`55ac7df5f5ad4385`)), and `1 + cap ≤ maxFresh` (`Spec.Folding.Schedule.production_cap_fits` (`7659dbeec8ddc746`)). Extraction runs backward over a history. A light fold recovers its input at its output bound plus `T(b − 1)`, and a refresh fold recovers its input below `B`. So each recovered accumulator is at most `B + cap·T·(b − 1) = 79,360` (computed; `Spec.Folding.Schedule.recoveredBounds_le` (`793c72eedb9a44e6`)). The binding reduction then needs MSIS at the chain norm `msisChainBound = 8·T·(B + cap·T·(b − 1)) = 137,134,080`, about `2^27.03` (computed), in place of `8TB` (`Spec.Folding.Schedule.msisChainBound`, SuperNeo v1.2 Section 8).

The error terms are these. The refresh fold's Π_CCS test error is `ε_test = (ℓ · 8 + 4365)/q² = 4589/q²` (`Lifecycle.Nifs.VerifierErrorBudget.test_error_eq` (`4826645c11e6724e`)), where 4365 is `4338 − 1 + ℓ` and 4338 is the number of joint coefficients (`Spec.Folding.PiCCS.PaperJoint.Shape.jointCoefficientCount`). The light fold has 273 joint coefficients, so its numerator is `ℓ · 8 + 273 − 1 + ℓ = 524` (`Spec.Folding.Light.ErrorBudget.production_testNumerator` (`01c2e0b59702531e`)). A light fork has one `ρ` coordinate, and a refresh fork has 17 (`Spec.Folding.Light.ErrorBudget.production_forkCoordinates` (`2a6b63ad122cfcd0`)). The law of each Π_RLC challenge is within `2^−132` of the uniform law on `C` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt` (`ed2b1cc8c235a81e`)). Lean does not add these terms into one bound. Rust `Params::for_ccs_shape` refuses a caller minimum above the zero-query error of the refresh fold in the random-oracle result (`Lifecycle.OptionB.RandomOracle.RefreshKnowledge.statisticalError`), `74 · ε_test + 17² · (1/|C| + 2^−132)` with Rust's census numerator in `ε_test`, about `2^−109.4`. So the profile accepts a minimum of at most 109 bits, and the tests use 109 bits (`the_gate_refuses_more_bits_than_the_fold_error_gives` in `crates/nightstream/tests/option_b_api.rs`; not checked against Lean).

## 4. Encodings

### 4.1 Words

Lean hashes and absorbs field elements, called words. A natural number `n` becomes the word `n mod q` (`Spec.Poseidon2.ofNat`, `Lifecycle.natWord`).

- Bytes become one word per 8 bytes, little-endian, reduced modulo `q` (`Lifecycle.packBytes`).
- An element `c_0 + c_1 u` of `K` is the two words `c_0, c_1` (`Lifecycle.serializeK`).
- A ring element is its 54 coefficients in order (`Lifecycle.serializeRingF`). A commitment is its `κ` ring elements in row order, `κd = 1188` words (`Lifecycle.serializeCommitment`).
- A public input is its 270 words (`Lifecycle.serializePublicInput`). A point of `K^ℓ` is its coordinates in order, `2ℓ` words (`Lifecycle.serializePoint`).
- An evaluation family is `Eval_K`, 54 elements of `K`, then `Eval_A` matrix by matrix, each 54 elements of `K`: `2 · 5 · d = 540` words (`Lifecycle.serializeEvaluations`).
- A package value is its nodes in tree order. Each node is four words: the tag (0 for a number, 1 for an array), then the number or the array length as its low 32 bits, its next 32 bits and the rest (`Export.Package.valuePreimage`).

### 4.2 The state hash

The state hash of HyperNova Construction 2 binds the key digests of the two circuits, the step counter, the initial and current states, the program counter and the two slots (`Lifecycle.OptionB.statePreimage`, `Lifecycle.OptionB.stateHash`). A state is four words. Each slot enters as its digest, the Poseidon2 hash of its own domain and its words (`Lifecycle.OptionB.slotDigest`). The slot words are these (`Lifecycle.OptionB.serializeSlot`).

| Part | Words | Content |
|---|---|---|
| domain | 12 | the ASCII bytes of `HyperNova/NIVC/slot/v3`, packed 8 bytes to a word (`Lifecycle.OptionB.slotDomainChunk`) |
| commitment | `κd` | the accumulator's commitment |
| evaluations | `10d` | `Eval_K`, then `Eval_A` |
| point | `2ℓ` | the accumulator's point `r` |
| public input | 270 | the accumulator's public input |
| depth | 1 | the slot's depth |

Without the domain, a slot is 2,055 words (`Lifecycle.OptionB.serializeSlot_length_eq` (`622a6ce74e228411`)). The state preimage is these words.

| Part | Words | Content |
|---|---|---|
| domain | 12 | the ASCII bytes of `HyperNova/NIVC/state/v3`, packed 8 bytes to a word (`Lifecycle.OptionB.stateDomainChunk`) |
| `vk_0`, `vk_1` | 8 | the key digests of the light and the refresh circuit (Algorithm 2) |
| `i` | 1 | the step counter |
| `z0` | 4 | the initial state |
| `zi` | 4 | the current state |
| `pc` | 1 | 0 before the first step, else 1 plus the index of the circuit that ran last |
| slot 0, slot 1 | 8 | the two slot digests |

A well-formed state preimage is 38 words (`Lifecycle.OptionB.statePreimage_length` (`9d01b2899758481d`)). Two different well-formed states with one state hash give a Poseidon2 collision on an input pair of one length that the two states fix: their state preimages or the slot preimages of one slot. The sponge pads with zeros and does not encode the length, so a pair of two lengths would not be a collision of the permutation (`Lifecycle.OptionB.stateHash_collision` (`f78f6a7418eaaf0d`)). The slot domain, the state domain and the fold domains of section 5.2 are different blocks (`Lifecycle.OptionB.domainChunks_distinct` (`68598947c81efd34`), `Lifecycle.OptionB.foldDomainChunks_distinct` (`6ca6d5d5e4fe2ba8`)).

### 4.3 The public input of a fresh claim

A fresh claim's public input encodes a four-word digest in 270 cells (`Lifecycle.encHash`, HyperNova Definition 12). Cell 0 is 1 (`Lifecycle.encHash_marker` (`d04bb6b10eebcdff`)). Cell `1 + 64w + k` is bit `k` of digest word `w`, for `w < 4` and `k < 64`. Cells 257 to 269 are 0. The verifier decodes the digest from bits 1 to 256 (`Lifecycle.decodeHash`).

### 4.4 Packages and circuit keys

A step package is the Lean value `Export.OptionB.StepPackage.Package`. It holds, in this order, the circuit index, the public columns, the source width, the first fresh column, the first S-box column, the pieces, the witness batches, the slot runs and the application region (`Export.OptionB.StepPackage.Package.format`). A piece is a list of R1CS rows, a ring product with its quotient columns, or a packed Poseidon2 block (`Export.OptionB.StepPackage.PackagePiece`). The relation's rows are the rows of the pieces, then one binding row for each block output lane (`Export.OptionB.StepPackage.Package.relationRows`), then 150 S-box rows for each block (`Export/OptionB/Parity.lean`). The slot runs give each source column its low-norm encoding, a bit or 41 balanced-ternary digits (`Export/Generic/SlotLayout.lean`). Every format decodes what it encodes, so the encoded package fixes the relation (`Export/OptionB/StepPackage.lean`).

The identity of the light and refresh packages is the Poseidon2 hash of the 31 bytes of `Nightstream/OptionB/packages/v1`, one word each, then the words of the pair (light package, refresh package) (`Export.OptionB.StepPackage.identity`). Lean also computes this hash as a stream over the encoding (`Export.OptionB.StepPackage.identityFast_eq` (`eb4250a072919dd7`)). The Rust test `identity_code_matches_lean_on_the_probe_packages` compares Rust's identity code with Lean's on two small packages (`crates/nightstream-fprime/tests/option_b_packages.rs`).

These digests are Rust's (not checked against Lean).

1. The identity of the finish package is the Poseidon2 hash of the bytes of `Nightstream/OptionB/finish/v1`, one word each, then the words of the finish package (`crates/nightstream-fprime/src/option_b.rs`).
2. With a Rust application in place of Lean's, the identity is the Poseidon2 hash of the bytes of `Nightstream/OptionB/application/v1`, one word each, then the Lean identity, the witness word count, the application's rows and its recipe rows (`crates/nightstream-fprime/src/option_b/application.rs`).
3. The key digest of circuit `c` is `vk_c = Poseidon2.hash(domain ++ identity ++ [c])`, where `domain` is the ASCII bytes of `Nightstream/OptionB/circuit-key/v1` packed 8 bytes to a word in 12 words, `c = 0` for the light circuit and `c = 1` for the refresh circuit (`crates/nightstream/src/lifecycle/option_b/mod.rs`). Lean takes the two key digests as part of the setup (`Lifecycle.OptionB.CircuitKey`).

### 4.5 Proof bytes

These layouts are Rust's (not checked against Lean). Every word is 8 bytes, little-endian, and a field word must be below `q`. A proof (`crates/nightstream/src/lifecycle/option_b/encoding.rs`) is these parts.

| Part | Size | Content |
|---|---|---|
| tag | 16 | `NS-OPTIONB-PRF-1` |
| header | 80 | `pc`, `i`, `z0` and `zi` as ten words |
| slot 0, then slot 1 | `8 + 8(κd + 270 + 2ℓ + 10d) + 8d·w_s` each | the depth, then the accumulator's commitment, public projection, point, `Eval_K` and `Eval_A`, then its witness as `d · w_s` field words |
| fresh claim | `8(κd + 270) + 16 w_c` | the commitment, the public input, then for each witness column a 64-bit mask of the `+1` coefficients and one of the `−1` coefficients |

Here `w_s` is the number of ring columns of the relation of circuit `s`, and `c = pc − 1`. A proof with `pc = 0` is the initial proof, and it has only the tag and the header. The decoder checks the exact length that `pc` and the two relations fix before it reads. It rejects a word not below `q`, a malformed witness mask, and an initial proof with `i ≠ 0` or `zi ≠ z0`. An accumulator witness is not in `{−1, 0, 1}`, so its entries are field words.

The public API has no byte form of a final proof or of a compression key (`crates/nightstream/src/circuit.rs`). `Circuit::write` saves the Rust application, and its file starts with the 8 bytes `NSOPTB`, 0, 1, or `NSOPTM`, 0, 1 for a memory circuit.

## 5. Transcript

### 5.1 The permutation and the hash

The hash is Poseidon2 over `F_q` with width 16 and the S-box `x^7` (`Spec.Poseidon2.permute`). It has 4 full rounds, 22 partial rounds and 4 full rounds, with Plonky3's round constants and internal diagonal (`Spec.Poseidon2.initialConstants`, `Spec.Poseidon2.internalDiagonal`). The state is 16 words, and the rate is 12.

- To absorb a block, add its words to lanes 0 to 11, then permute (`Spec.Poseidon2.absorbBlock`).
- `Poseidon2.hash(words)` starts from the zero state, absorbs the words in blocks of 12, the last one short, then adds 1 to lane 0, permutes and outputs lanes 0 to 3 (`Spec.Poseidon2.hash`). It does not bind the length.

### 5.2 The fold transcripts

Each fold verifier and each step circuit runs one Poseidon2 sponge for the fold (`Lifecycle.Transcript`). To absorb a list, add it in blocks of 12, each followed by one permutation (`Lifecycle.Transcript.absorb`). The profile fixes every absorbed length, so the stream has one parse without framing (`decisions/piccs-prior-state-digest.md`).

A challenge in `K` at position `p` reads lanes `2(p mod 6)` and `2(p mod 6) + 1` as `c_0, c_1`. After the read at a position with `p mod 6 = 5`, the sponge absorbs a zero block (`Lifecycle.Transcript.squeezeAt`). Coordinate `j` of `α` has position `j`, `γ` has position `ℓ`, and every round challenge has position 0 (`Lifecycle.OptionB.lightCoinPosition`, `Lifecycle.Transcript.coinPosition`). `S` is the number of sources, 2 for a light fold and 17 for a refresh fold.

| # | Operation | Words or reads | Value |
|---|---|---|---|
| 1 | absorb | 12 | the ASCII bytes of `Nightstream/SuperNeo/fold/light/v1` or `Nightstream/SuperNeo/fold/refresh/v1`, packed 8 bytes to a word (`Lifecycle.OptionB.lightFoldDomainChunk`, `Lifecycle.OptionB.refreshFoldDomainChunk`) |
| 2 | absorb | `4 + κd + 270`, plus `16(κd + 10d)` for a refresh fold | the prior digest decoded from the fresh public input, the fresh commitment and the fresh public input (`Lifecycle.ProductionKey.publicInputBlocks`). A refresh fold adds each Π_DEC child's commitment and evaluation family, child 0 first, in the same list (`Lifecycle.OptionB.refreshWords`). |
| 3 | read | `ℓ` | `α_0, …, α_{ℓ−1}` at positions 0 to `ℓ − 1` |
| 4 | read | 1 | `γ` at position `ℓ` |
| 5 | for each round `j < ℓ`: absorb | 18 | the 9 coefficients of round polynomial `j` (`Lifecycle.Transcript.serializeMessage`) |
| 6 | then read | 1 | the round challenge `r'_j` at position 0 |
| 7 | absorb | `S · 10d` | the Π_CCS output, source by source (`Lifecycle.OptionB.lightOutputWords`, `Lifecycle.ProductionKey.absorbFullOutput`) |
| 8 | Π_RLC | 4 lanes for each challenge | `ρ` for a light fold, `ρ_0, …, ρ_16` for a refresh fold (section 5.3) |

The accumulator does not enter the sponge directly. It enters through the prior digest in the fresh public input, which is the state hash (Algorithm 3). The refresh children are prover messages, so the sponge absorbs them. Their point and public inputs come from the accumulator. Equal absorbed words give equal children's commitments and evaluation families (`Lifecycle.OptionB.refreshWords_children` (`c2be7b9fe5371617`)). Rust replays these transcripts in `derive_light_transcript` and `derive_refresh_transcript` (`crates/nightstream-fprime/src/package/pi_ccs_v1_2_transcript.rs`).

### 5.3 The Π_RLC sampler

Let `S_0` be the sponge state after row 7. For challenge `s`, absorb the block `[4, s]` into `S_s` to get `E_s` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter`). Read lanes 0 to 3 of `E_s` as `x_0, …, x_3`, and set `S_{s+1} = permute(E_s)` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt`).

1. Set `N = Σ_{k<4} x_k · q^k` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.drawIndex`) and `R = N mod 5^54` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.reduce`).
2. Coefficient `h` of `ρ_s` is `⌊R / 5^h⌋ mod 5`, minus 2 (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample`, `Spec.Phi81StrongSet.embedScalar`).

A refresh fold takes `ρ_0, …, ρ_16` (`Lifecycle.ProductionKey.piRlcResponse`). A light fold takes only `ρ_0` as its `ρ` (`Lifecycle.OptionB.lightChallenge`). The sampler cannot fail (`Lifecycle.ProductionKey.piRlcResponse_ne_none` (`d42d0844252c6562`)), and each challenge is in `C` (`Lifecycle.Transcript.PiRlcSampler.piRlcChallenges_member` (`3c3147745741757a`), `Lifecycle.OptionB.lightChallenge_valid` (`c9b6128ed41936b5`)).

## 6. Algorithms

### Algorithm 1. Commitment-key setup

**Input.** A 32-byte seed, `κ` and the number of key columns `w`.

**Output.** The Ajtai key `A ∈ R_q^{κ × w}`.

**Steps.**

1. For each row `i < κ` and column `j < w`, let `in` be the 37 ASCII bytes of `nightstream-ajtai-shake128-wide256-v1`, then the seed, then `i` as 4 bytes and `j` as 8 bytes, little-endian (`Spec.AjtaiSetupV1.elementInput`).
2. Read the first `4d` 64-bit lanes of SHAKE128 of `in` (`Spec.AjtaiSetupV1.elementLanes`).
3. Set coefficient `h` of `A_{i,j}` to the 256-bit little-endian value of lanes `4h` to `4h + 3`, reduced modulo `q` (`Spec.AjtaiSetupV1.wideCoefficientNat`).

**Rejects.** None.

**Source.** `Spec.AjtaiSetupV1.Setup.verifierKey`. Each coefficient is one SHAKE128 chunk reduced modulo `q` (`Spec.AjtaiSetupV1.Programming.verifierKey_eq` (`59e988e0131c808d`)). The production seed is `fc404984d44c1b878d68a6a80092d7d7ab44d81ac17b45a8e7bd4c1f1e371702` (`Export.Stage1.Poseidon2HashChainV1SetupAuthority.productionSeed`). Both circuits commit with the first `w_c` columns of the production key, where `w_c` is the number of ring columns of circuit `c` (`commit_production_signed_unit_prefix_matrix` in `crates/neo-ajtai/src/nightstream_fprime_setup.rs`; not checked against Lean). Records `decisions/fprime-stage1-main-ajtai-setup.md` and `decisions/fprime-ajtai-shake128-setup.md`.

### Algorithm 2. Packages and circuit keys

**Input.** The light, refresh and finish packages, their pinned identities, and optionally a Rust application.

**Output.** The identity and the key digests `vk_0` and `vk_1`, or reject.

**Steps.**

1. Decode each package with `Package.format`. Require the circuit index 0 for the light package, 1 for the refresh package and 2 for the finish package.
2. Compute the identity of the light and refresh packages and the identity of the finish package (section 4.4). Require that each equals its pin. For the copies that the crate embeds, the test `embedded_packages_have_their_pinned_identities` does this step once, and at run time each identity is its pin, because the bytes are part of the binary (`crates/nightstream/src/artifacts.rs`). The embedded packages are decoded once per process.
3. With a Rust application, put its rows and recipes in place of the application region of both step packages, move the finish package to its witness word count, and compute the new identity (section 4.4). The reference hash chain keeps both step packages and their pinned identity, because it computes the step of Lean's application region (`the_reference_hash_chain_keeps_lean_packages` in `crates/nightstream/tests/option_b_api.rs`).
4. Compute `vk_0` and `vk_1` from the identity (section 4.4).

**Rejects.** A package that does not decode, a wrong circuit index, an identity that is not its pin, and an application piece that reads past its region.

**Source.** `load_option_b_packages`, `load_option_b_finish`, `load_pinned_option_b_packages`, `load_pinned_option_b_finish` and `OptionBPackages::with_application` (`crates/nightstream-fprime/src/option_b.rs`, `crates/nightstream-fprime/src/option_b/application.rs`), `Circuit::compile` and `Circuit::load_packages` (`crates/nightstream/src/circuit.rs`), and `OptionB::new` (`crates/nightstream/src/lifecycle/option_b/mod.rs`); not checked against Lean. Lean defines the identity of the light and refresh packages (`Export.OptionB.StepPackage.identity`). Every state hash carries `vk_0` and `vk_1` (section 4.2).

### Algorithm 3. The step relation

Each step proves one augmented step of HyperNova Construction 2 for the circuit that runs, with two slots and the depth cap (`Lifecycle.OptionB.StepHolds`, `Spec.HyperNova.Construction2.BoundedDepth.verifier`).

**Input.** The circuit `c`, 0 for light and 1 for refresh, the step counter `i`, the states `z0` and `zi`, the program counter `pc`, the two slots, a fresh claim `u`, a fold proof of the kind of `c`, and the application witness `w`.

**Output.** The next state `z_{i+1}`, the next slots, the next program counter `c + 1`, and the digest `x` that the next fresh claim's public input encodes.

**Steps.**

1. Require that the circuit selector gives `c` at `(zi, w)`. Option B takes the selector that returns the circuit that runs on a valid step and the other circuit otherwise (`Export.OptionB.TerminalBridge.chosen`), so the prover picks `c` and the step's witness is valid. Set `z_{i+1} = F(zi, w)`, where `F` is the application's step (`Lifecycle.Stage1.Application.Program`). Both circuits run the same `F`, so the choice of `c` does not change `z_{i+1}`.
2. If `i = 0`: require `z0 = zi`, and set both next slots to the default slot.
3. If `i > 0`: require `pc ∈ {1, 2}`, and let `s = pc − 1` be the selected slot. Require that the public input of `u` is `encHash(stateHash(vk, i, z0, zi, pc, slots))`. If `c` is the light circuit, require that the depth of slot `s` is below `cap`. Require that the fold of the kind of `c` (Algorithm 4) accepts the accumulator of slot `s`, `u` and the proof, with the relation of circuit `s`. Set slot `s` to the fold's output, at depth `d + 1` after a light fold and at depth 0 after a refresh fold. The other slot does not change.
4. Set `x = stateHash(vk, i + 1, z0, z_{i+1}, c + 1, next slots)`.

**Rejects.** The relation does not hold.

**Source.** `Lifecycle.OptionB.stepHolds_iff` (`4b2804f6a80e539d`) states these steps. The light circuit places the application, the depth guard, the prior state hash with its two slot hashes, the next state hash with its two slot hashes, the light Π_CCS and Π_RLC on the selected slot, and the slot select (`Lifecycle.OptionB.LightStepLayout`). The depth guard decomposes the depth word into 64 bits and requires bits 6 to 63 to be zero, so the depth is below 64 (`Lifecycle.OptionB.DepthGuard`). The refresh circuit has no depth guard. It places Π_DEC on the selected slot before the refresh Π_CCS and Π_RLC, and the new slot's depth word is 0 (`Lifecycle.OptionB.RefreshStepLayout`). The rows of either circuit give its case of the step relation (`Layout.OptionB.StepCircuits.soundness` (`3435e35e19903188`)). The circuits do not compute the selector. Their rows give the application's validity predicate, and each soundness theorem takes as a premise that every valid step of the circuit's application has the selector value `c`. The selector `chosen c` meets this premise for every application. For the reference application, `F(z, w) = Poseidon2.hash(tag ++ z ++ w)`, where `tag` is the 40 bytes of `Nightstream/Stage1/Poseidon2HashChain/v1`, one word each (`decisions/fprime-stage1-poseidon2-hash-chain-v1.md`). HyperNova Construction 2, steps 4 and 5.

### Algorithm 4. Fold verifiers

A fold combines the previous step's fresh claim with one accumulator. The light fold runs Π_CCS on two sources and Π_RLC with the weights `ρ` and 1 (Cyclo, ePrint 2026/359, contribution 2, on SuperNeo v1.2 Sections 7.3 and 7.4). The refresh fold runs Π_DEC on the accumulator first, then Π_CCS on 17 sources and Π_RLC (SuperNeo v1.2 Sections 7.3 to 7.5). Neither fold ends with Π_DEC.

**Input.** The kind, light or refresh, the accumulator `U = (r, c_U, x_U, y_U)`, the fresh claim `(c, x)`, and a proof of that kind.

**Output.** The next accumulator, or reject.

**Steps.**

1. For a refresh fold, read the 16 children's commitments `c_k` and families `y_k` from the proof. Split `x_U` into 16 digit vectors `x_0, …, x_15` with `x_U = Σ_k 2^k x_k`, and reject if it does not split. Require `c_U = Σ_k 2^k c_k` and `y_U = Σ_k 2^k y_k` (`Spec.Folding.Refresh.inputCheck`, SuperNeo v1.2 Section 7.5). The running sources are the children `(r, c_k, x_k, y_k)`. For a light fold, the running source is `U`.
2. Order the sources fresh first, then the running sources. Let `R = S − 1` be the number of running sources. Run rows 1 to 6 of section 5.2 to get `α`, `γ` and `r' = (r'_0, …, r'_{ℓ−1})`.
3. Compute the initial claim `c_0 = Σ γ^{s + R h} · Eval_K(y_s)_h + Σ γ^{Rd + s + R j + 4R h} · Eval_A(y_s)_{j,h}`, over the running sources `s < R`, `j < 4` and `h < 54` (`Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial.VerifierInput.initial`, `Spec.Folding.PiCCS.PaperJoint.PadCoordinate.gammaExponent`, `Spec.Folding.PiCCS.PaperJoint.MatrixCoordinate.gammaExponent`).
4. For each round `j`, require `g_j(0) + g_j(1) = c_j`, and set `c_{j+1} = g_j(r'_j)` (`Spec.SumCheck.Finite.FixedPhase.checkChain`).
5. Require `c_ℓ = V`, where `V` is computed from the Π_CCS output at `r'` (`Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial.terminalFromMessage`):

   ```text
   V = eq(r', r) · Σ_coord γ^e · Pad_coord
     + γ^(Rd) · eq(r', r) · Σ_coord γ^e · Mat_coord
     + γ^(5Rd) · eq(r', α) · ( f(M_0 z_0, …, M_3 z_0) + γ · Σ_{s<S} γ^s · N(z_s) )
   N(y) = (y + 1) · y · (y − 1)
   ```

   Here `Pad_coord`, `Mat_coord`, `M_j z_0` (the fresh source) and `z_s` are the values that the output claims at `r'`, and `e` is the local exponent of each coordinate (`Spec.Folding.PiCCS.PaperJoint.StrongReduction.Statement.projectOutput`). `Rd` is 864 and `5Rd` is 4320 for a refresh fold (`Spec.Folding.PiCCS.PaperJoint.Shape.constraintOffset_eq` (`395ce359b5e699ad`)), and they are 54 and 270 for a light fold (`Lifecycle.OptionB.LightPiCCS.light_exponents` (`a48bd558eb090fcb`)). The light Π_CCS puts the running source into the norm batch as zero, so only the fresh source has a norm check (`Spec.Folding.Light.PiCCS.normAssignments`).
6. Absorb the Π_CCS output (row 7), and derive the challenges (section 5.3).
7. For a light fold, the next accumulator is `(r', c_U + ρ·c, x_U + ρ·x, y'_U + ρ·y'_0)`, where `y'_0` and `y'_U` are the families of the fresh source and of the accumulator in the Π_CCS output (`Spec.Folding.Light.Rlc.output_commitment` (`27c897818b0c19c0`), `Spec.Folding.Light.Rlc.output_publicInput` (`5872386765650c7b`), `Spec.Folding.Light.Rlc.output_evaluation` (`649ea10845fa2950`)). For a refresh fold, it is `(r', Σ_s ρ_s c_s, Σ_s ρ_s x_s, Σ_s ρ_s y'_s)` over the 17 sources, where `c_s` and `x_s` are the source's commitment and public input and `y'_s` is its family in the Π_CCS output (`Spec.Folding.PiRLC.combinedOutput`, SuperNeo v1.2 Section 7.4).

**Rejects.** The verifier rejects when step 1, 4 or 5 fails (`Spec.Folding.Light.verify`, `Spec.Folding.Refresh.verify_eq_some_iff` (`eb8c4eb08204ce2f`)).

**Source.** `Spec.Folding.Light.verify` and `Spec.Folding.Refresh.verify` at the production keys `Lifecycle.OptionB.keys`. The two keys share one relation, one commitment key and one Π_RLC algebra, and they differ in the transcript (`Lifecycle.OptionB.keys_coherent` (`4476947493afa656`), `Lifecycle.OptionB.refreshKey_relation` (`dfad4def6d49a9a7`)). The light verifier rejects a refresh proof, and the refresh verifier rejects a light proof (`Spec.HyperNova.Construction2.OptionBFolds.light_rejects_refresh_proof` (`1cf40c816ebb7110`), `Spec.HyperNova.Construction2.OptionBFolds.refresh_rejects_light_proof` (`841e78381123c5a4`)). Rust folds in `crates/nightstream/src/folding/option_b.rs` (not checked against Lean). Record `decisions/padded-row-identity-piccs.md`.

### Algorithm 5. Prove and extend

**Input.** For `prove`, the initial state `z0` and the application witness. For `extend`, a proof at step `i` and the next application witness.

**Output.** A proof at step `i + 1`.

**Steps.**

1. `prove` starts from the initial proof. It has `i = 0`, `pc = 0`, `zi = z0`, both slots at the default slot and no fresh claim.
2. At `i = 0`, run the base step of the light circuit (Algorithm 3, step 2).
3. At `i > 0`, let `s = pc − 1`. If the depth of slot `s` is below 64, fold the fresh claim into slot `s` with the light fold and run the light circuit. Else fold it with the refresh fold and run the refresh circuit. Set slot `s` to the fold's output and its next depth.
4. Fill the words of the circuit that runs, compute its logical assignment with the package's witness program, and commit it. Its commitment and public input are the next fresh claim.
5. The proof is the state `(i + 1, z0, z_{i+1})`, `pc = c + 1` for the circuit `c` that ran, both slots with their witnesses, and the fresh claim with its witness.

**Rejects.** None. A prover that fails returns an error, and the supplied proof does not change.

**Source.** `OptionB::initial` and `OptionB::extend` (`crates/nightstream/src/lifecycle/option_b/mod.rs`) and `crates/nightstream/src/folding/option_b.rs`; not checked against Lean. `Prover::extend` runs the Rust application to get `z_{i+1}`, and `Prover::extend_with_output` takes it from the caller (`crates/nightstream/src/circuit.rs`). Lean does not model this scheduler. The light verifier rejects a light fold at the cap, so the folds of each slot form an admissible schedule (`Spec.HyperNova.Construction2.BoundedDepth.admissible_of_accepted` (`95df2aa1a548b830`)). A run that the cap admits has at most `cap · r + 2 · cap` light steps, where `r` is its number of refresh steps (`Spec.HyperNova.Construction2.BoundedDepth.light_count_le` (`6b3396f5f261095d`)).

### Algorithm 6. Terminal verifier

**Input.** The expected state `(i, z0, zi)` and a proof.

**Output.** Accept or reject.

**Steps.**

1. Require that the proof's state is the expected state and that `i < q`.
2. If the proof is the initial proof, accept exactly when `i = 0` and `zi = z0`.
3. Require `i > 0` and `pc ∈ {1, 2}`.
4. For each slot, require that its witness opens its accumulator. The witness gives the commitment and the public projection, has norm below `B`, and gives the `Eval_K` and `Eval_A` values at the accumulator's point.
5. Require that the public input of the fresh claim is `encHash(stateHash(vk, i, z0, zi, pc, slots))`.
6. Require that the fresh witness satisfies the relation of circuit `pc − 1`, opens the fresh commitment and projects to the fresh public input.

**Rejects.** Each step's failure. Rust checks step 4 with Π_DEC on each slot's witness. Its digits reject a norm at or above `B`, and its recomposition must give the claim's commitment, evaluations and public input. Rust checks step 6 in this order: the claim shape, the witness shape, a zero tail past the relation's width, the public projection, the commitment, then every row (`crates/nightstream/src/lifecycle/option_b/mod.rs`, `crates/nightstream/src/lifecycle/option_b/relation.rs`; not checked against Lean).

**Source.** `Lifecycle.OptionB.Terminal.HoldsFor`, the terminal check of HyperNova Construction 2 with two slots (`Spec.HyperNova.Construction2.Paper.OuterTerminalTransition`, `Lifecycle.OptionB.Terminal.holdsFor_recursive_iff` (`d75b170dd2e62705`)). The initial proof is accepted exactly at `i = 0` and `zi = z0` (`Lifecycle.OptionB.Terminal.holdsFor_bottom_iff` (`06cc6310e8f07c1f`)). A counter that is larger by `q` is rejected (`Lifecycle.OptionB.Terminal.rejects_counter_shift` (`ad9cc3d1b653e2b4`)). With the exported relations as the circuit keys, acceptance gives the decoded last step or a collision of the decoded next state with the terminal's state (`Export.OptionB.TerminalBridge.bridge` (`22d55afec87191ff`)).

### Algorithm 7. Finish

**Input.** A proof with `i ≥ 1` and the compression setup of its circuits.

**Output.** A final proof.

**Steps.**

1. Fold the proof's fresh claim into slot `pc − 1` with the refresh fold. A refresh fold is admissible at any depth. Both slots then hold accumulators below their bounds, and no claim is fresh.
2. For each slot, prove its accumulator under its circuit's relation with layer 1, a sum-check and WHIR argument (`crates/neo-spartan/`). The norm bound of both slots is `accumulatorBound cap = (1 + 64 + 16)·216 = 17,496` (`Spec.GlobalParams.accumulatorBound`). The transcript seed absorbs the domain `Nightstream/OptionB/compression/v1`, then the state hash before the last fold and the slot index.
3. Prove with the shrink layer, SuperSpartan and one WHIR commitment, that the final program accepts the state: the finish package's rows on the last fold's words, and each slot's layer-1 verifier on the slot claim that those words hold.
4. The final proof is the state and the shrink proof. It carries no witness and cannot be extended.

**Rejects.** An initial proof, which has no fold to finish. Also a WHIR query sample of `p − 1`, which fails Algorithm 8, so `compress` returns an error and no proof. This has a probability of about `2^−64` for each query draw. The layer-1 verifier also rejects at a pole: a challenge `β` in the histogram table, a zero GKR root denominator of the lookup, scatter or block-scatter check, or a lane-weight pole of `ζ`. The Rust prover returns an error at the last one (`crates/neo-spartan/src/prover.rs`). Each pole has a probability below about `2^−150` over the cubic field. For example, the histogram pole has the probability `(2·bound + 1)/|Ext|`, about `2^−177` at bound 17,496. These numbers are estimates. Lean does not bound these events, and its layer-1 game does not model them. The proof is deterministic, so a new call at the same state fails again, but the prover can extend one step and then compress.

**Source.** `crates/nightstream/src/lifecycle/option_b/compression.rs` and `crates/nightstream/src/lifecycle/option_b/final_program.rs` (not checked against Lean). F′'s emitter writes the finish package from the finish relation of the application (`Export/OptionB/FinishRelation.lean`, `Export/OptionB/EmitMain.lean`). The finish circuit is the refresh step circuit without the application and the next hashes (`Lifecycle.OptionB.FinishStep`). The final program's checks on the finish package's columns give the finish relation of the last fold (`Export.OptionB.FinishRelation.package_finish` (`7e4add55cebac3e2`)). For every valid run of at least one step, the final program's checks accept the honest finish (`Export.OptionB.FinishCompleteness.run_finish_complete` (`1a336065fc38a317`)), and the claim of the final verifier holds for the honest statement (`CompressionLean.Claims.option_b_complete` (`5a660cc941cecedf`)).

### Algorithm 8. Final verifier

**Input.** The expected state `(i, z0, zi)`, a compression key and a final proof. The verifier derives the key from its circuits' relations, or uses a pinned copy, never a key from a prover.

**Output.** Accept or reject.

**Steps.**

1. Require that the proof's state is the expected state.
2. Build the final program of the key and the state, and require that its shrink layer meets the verifier's security minimum.
3. Verify the shrink proof for that final program.

**Rejects.** Each step's failure. In step 3, a WHIR query sample of `p − 1` is a failure, where Plonky3 draws again. This check is stricter than Plonky3 and affects only completeness, not soundness (`crates/neo-spartan/src/whir/mod.rs`; the Lean final program makes the same check, `Sim/Whir.lean`).

**Source.** `OptionB::verify_final` and `Verifier::verify_final` (`crates/nightstream/src/lifecycle/option_b/compression.rs`, `crates/nightstream/src/circuit.rs`; not checked against Lean). Layer 1 and the shrink layer run in one state-restoration experiment, so one bad-set argument covers both layer-1 games and the shrink layer. At Rust's layer bits `t`, it bounds an accepted false claim by `calls / 2^t` (`CompressionLean.Final.OptionB.restore_budget` (`035908924e9adc24`)). That theorem has the premise that the shrink CCS encodes the final program, which is the finish package and the two layer-1 verifiers. Lean proves this premise for the final program (`Sim/OptionBEncodes.lean`), and the claim joins the two (`CompressionLean.Claims.option_b_bound` (`7fc867ace3da79f6`)). The finish checks and both slots' layer-1 relations give the last fold's claims (`CompressionLean.Claims.option_b_parent_bound` (`20d4a178d46ed614`)). The premises about the keys and the setups are met for the reference application only (`CompressionLean.Claims.option_b_poseidon2_bound` (`a74f67c4cb3e5d99`), `formal/compression-lean/STATUS.md`). The last fold starts the reverse walk of the IVC bound, and one experiment bounds both (`CompressionLean.Claims.option_b_ivc_bound` (`e63fe95ae879be36`)), with the premises of section 1. The honest statement of a valid run is never a false acceptance of this experiment (`CompressionLean.Claims.option_b_honest_unbroken` (`d42f50736e923a52`)). Three things are assumed, not proved: the bound includes no WHIR error; it covers a valid program witness only where its in-program challenges are the experiment's oracle answers (the read condition); and the key's root is the honest root of the claims' setup (`formal/compression-lean/TRUST.md`, sections 3 and 5).

### Lifecycle rules

These four rules join Algorithms 5 to 8. Rust keeps them, and no theorem covers them, except rule 2 at the claim level (not checked against Lean). The tests check them on fixed inputs, and CI runs each test that this section names.

1. **Each returned proof verifies.** Algorithm 6 accepts each proof that `prove` or `extend` returns from a proof that Algorithm 6 accepts, at the proof's own state. `extend` does not verify the proof that it receives. Before it returns, it requires that the new fresh instance holds under its circuit's relation (`Relation::holds`), whose rows verify the fold, so a prover fault gives an error and not a proof (not checked against Lean). Tests: `hash_chain_proves_and_verifies` (`crates/nightstream/tests/option_b_api.rs`), and `light_folds_and_a_refresh_verify`, `refresh_folds_under_both_relations_verify`, `native_hash_chain_verifies` and `native_addition_verifies` (`crates/nightstream/tests/option_b_lifecycle/mod.rs`). The two lifecycle tests use the caps 2 and 0, so that a short run makes light folds into both slots and refresh folds under both relations. `light_step_zero_holds_every_row` and `refresh_step_zero_holds_every_row` check that the base step of each circuit satisfies every row (`crates/nightstream/tests/option_b_base_steps.rs`).
2. **Each returned final proof verifies.** Algorithm 8 accepts each final proof that `compress` returns, with the key that the verifier derives, for a verifier of the same circuits and security minimum. Before `compress` returns, it verifies its own shrink proof with the verifier of Algorithm 8 (not checked against Lean). Tests: `hash_chain_final_proof_verifies` (`crates/nightstream/tests/option_b_api.rs`) and `final_proof_verifies` (`crates/nightstream/tests/option_b_lifecycle/mod.rs`). Lean proves this rule only at the claim level: the claim of the final verifier holds for the honest statement (`CompressionLean.Claims.option_b_complete` (`5a660cc941cecedf`)). It does not prove that the shrink proof is accepted.
3. **An edited proof fails.** Algorithm 6 rejects a proof with a wrong state, a changed slot depth, a changed slot witness, a slot claim of the wrong shape, a recommitted fresh witness that fails a row or has a nonzero tail, and a changed fresh commitment (`light_folds_and_a_refresh_verify`). `decode_proof` rejects truncated bytes (`hash_chain_proves_and_verifies`) and an iteration or slot depth word not below `q` (`a_word_not_below_q_does_not_decode`), and Algorithm 6 rejects a decoded proof with a changed byte (`hash_chain_proves_and_verifies`). The final program rejects a changed finish column, current word or key word, and one slot's layer-1 proof in the place of the other's. Algorithm 8 rejects a final proof for another state (`final_proof_verifies`, `hash_chain_final_proof_verifies`).
4. **The selected circuit.** Rust selects the circuit of each step from the depth of the selected slot (Algorithm 5, step 3): a light fold below `cap`, a refresh fold at `cap`. The Lean step relation lets the prover pick the circuit (Algorithm 3, step 1), and the light depth guard rejects a light fold at `cap` for every pick. So the Lean results cover the Rust selection. No Lean result states that the Rust schedule is Lean's `Spec.HyperNova.Construction2.BoundedDepth.refreshSchedule` (`SECURITY_MODEL.md`, trust item 4).

## 7. Nebula memory

Nebula memory checking (Nebula, Sections 3 and 4) gives an application public ROM and RAM. A memory application is an application program that both step circuits run (`Lifecycle.Stage1.Application.Program`). Its rows hold the memory rows, three record chains, the memory carry and the application's own rows, which meet at fixed memory ports (`Lifecycle.Nebula.MemoryApp.program`). It adds no commitment, no claim, no evaluation and no field to the state of section 4.2, so Algorithms 1 to 8 do not change. The memory carry is in the application state: the four state words of section 4.2 are a digest of the application words and the carry (section 7.5).

The steps come in segments of `N` steps. At the start of a segment, the prover proposes two chain roots, and the relation derives the memory challenge `η` from them. Each step checks its memory accesses against fingerprint products and adds its own records to the chains. The step that completes the segment requires that the chains equal the proposed roots and that the products balance. The verifier then runs the terminal verifier or the final verifier and requires a closed segment (Algorithm 10). The aim of these checks is that an accepted proof attests one execution of the application from the statement's initial words and the program's memory to the statement's final words, in which each memory access goes through a port and reads the last value written (`SECURITY_MODEL.md`, adversary 5).

### 7.1 Geometry and program

A geometry is seven integers (`Spec.Nebula.Geometry`). A program is two images: the ROM image of `R` words and the RAM image of `M` words, each word below `2^32` (`Spec.Nebula.Plan`).

| Field | Meaning |
|---|---|
| `r`, `μ` | the ROM has `R = 2^r` cells and the RAM has `M = 2^μ` cells |
| `W_ts` | the timestamp width in bits |
| `B_ops` | the operation slots of a step, one for each memory port |
| `B_scan` | the scan slots of a step |
| `N` | the steps of a segment |
| `S_max` | the largest segment count of a proof |

A package is the relation of one application at one geometry. The relation reads no image, so one package and one setup serve every program of the geometry (`Lifecycle.Nebula.MemoryApp.program_eq_of_geometry` (`dbacc72f0440578a`)). The program enters a proof only through `D_init` (section 7.4), which the verifier computes from the images.

Setup rejects a geometry that breaks one of these rules (`Spec.Nebula.Geometry.Valid`, `Spec.Nebula.Geometry.Secure`):

1. **Exact cover:** `N · B_scan = R + M`.
2. **Timestamp range:** `S_max · N · B_ops < 2^W_ts`.
3. **Field encoding:** `W_ts ≤ 62`, `B_ops · (3 + μ + 64 + W_ts) < q` and `B_scan · (32 + W_ts) < q`. Then each timestamp and value is an integer below `q`, row O4 cannot wrap modulo `q`, and the length of each lane is one word.
4. **Address width:** `r ≤ μ` and `R + M < q`.
5. **Positive sizes:** `B_ops`, `B_scan`, `N` and `S_max` are at least 1.
6. **Security:** `R + M + N · B_ops ≤ 139,509`. This holds exactly when the fingerprint error `2(R + M + N · B_ops)/q²` is at most `2^−109.91` (`Spec.Nebula.Geometry.secure_iff` (`fca657b92bff91b6`)), the floor of `decisions/nebula-memory-checking.md`.

A Lean memory package is a light and a refresh package whose application is `Export.Stage1.NebulaMemoryPackage.application` at a geometry, with the proofs of rules 1 to 6. The emitter writes them as `nightstream-fprime-option-b-nebula-v1-{light,refresh}-v1.json` and `nightstream-fprime-option-b-nebula-n1-{light,refresh}-v1.json` (`--program`, `Export/OptionB/EmitMain.lean`). The verifier rejects a program whose ROM image does not have `R` words or whose RAM image does not have `M` words before it computes `D_init` (`Spec.Nebula.Plan.Valid`). The segment counter uses `W_seg = ⌈log₂(S_max + 1)⌉` bits.

The plan digest is `Digest("Nightstream/Nebula/v3/plan", [r, μ, W_ts, B_ops, B_scan, N, S_max])` (section 7.4, `Lifecycle.Nebula.planDigest`). The package rows hold it as a constant.

### 7.2 Memory model and records

The global index of ROM address `a < R` is `a`, and of RAM address `a < M` is `R + a`. A cell holds a value and the timestamp of its last access. `ts` is the global timestamp: the number of active operations so far. It never resets. An active operation on a cell with value `v` and timestamp `rt` adds the read tuple `RS = (rt, g, v)` and the write tuple `WS = (ts + 1, g, v')`, and sets the cell to `(v', ts + 1)`. `v'` is `v` for a read and the new value for a write. A write to ROM is invalid. At the start of a segment, the initial snapshot `IS` has one tuple `(t, g, v)` for each cell, and at the end, the final snapshot `FS` has one tuple for each cell. An honest segment has `IS ∪ WS = RS ∪ FS` as multisets (`Spec.Nebula.MemOp`, `Spec.Nebula.Tuple`).

An operation slot has these fields, in this order (`Spec.Nebula.OpSlot`):

| Field | Bits | Meaning |
|---|---|---|
| `pad` | 1 | 1 for an inactive slot |
| `is_write` | 1 | 0 for a read, 1 for a write |
| `is_ram` | 1 | 0 for ROM, 1 for RAM |
| `addr` | `μ` | the address |
| `v_r` | 32 | the value read, or for a write the old value |
| `v_w` | 32 | the value written, or for a read `v_r` |
| `rt` | `W_ts` | the timestamp of the last access to the cell |

An inactive slot has every other field zero, at any position. For slot `j` of a step with incoming timestamp `ts`: `cnt_j = Σ_{i ≤ j} (1 − pad_i)`, `wt_j = ts + cnt_j`, `g_j = addr_j + is_ram_j · R`, `RS_j = (rt_j, g_j, v_r_j)` and `WS_j = (wt_j, g_j, v_w_j)`.

A scan slot is a value of 32 bits and a timestamp of `W_ts` bits (`Spec.Nebula.ScanSlot`). Each step has `B_scan` IS slots and `B_scan` FS slots, and slot `j` of step `idx` holds cell `idx · B_scan + j`. The relation computes this index, so a scan slot has no pad.

A step has three lanes: the ops lane of its `B_ops` operation slots, and the IS and FS lanes of its scan slots (`Spec.Nebula.opsLane`, `Spec.Nebula.scanLane`). A lane is its slots in order, each slot its fields in order, each field its bits from the least significant, one bit to a coordinate. So the ops lane has `B_ops · (3 + μ + 64 + W_ts)` coordinates, and each scan lane has `B_scan · (32 + W_ts)`. `pack` splits a lane into chunks of 63 bits, the last one short, and turns each chunk into the word `Σ_k 2^k · b_k`, which is below `2^63 < q` (`Spec.Nebula.pack`). It is injective on the lanes of one length (`Spec.Nebula.pack_injective` (`958e936decf1f7b7`)).

**Hash residency.** A fingerprint tuple and its pad gate read only the lane coordinates of their own step, the carry fields `ts` and `idx`, the counters `cnt_j` and constants. `η` and the products are only coefficients and accumulators of the fingerprints. The memory rows keep this rule. The application's rows read only its view (section 7.6), which has no word for `η`, a product or a carry word. Rust `nebula::Circuit::compile` rejects an application row that reads another variable (`crates/nightstream/src/nebula/circuit.rs`; not checked against Lean).

### 7.3 Memory rows

Each step evaluates these rows on its own assignment. They read `η`, the products `h`, `ts` and `idx` of the carry after the step's open action, if any (section 7.5).

| # | Row, for operation slot `j` |
|---|---|
| O1 | every lane coordinate and auxiliary bit is 0 or 1 |
| O2 | `cnt_j = cnt_{j−1} + (1 − pad_j)`, with `cnt_{−1} = 0` |
| O3 | `(1 − is_write_j) · (v_w_j − v_r_j) = 0` |
| O4 | `(1 − pad_j) · (wt_j − rt_j − 1 − diff_j) = 0`, with `diff_j` a `W_ts`-bit auxiliary word |
| O5 | `is_write_j · (1 − is_ram_j) = 0` |
| O6 | `(1 − is_ram_j) · addr_bit_{j,k} = 0` for each `k` from `r` to `μ − 1` |
| O7 | `pad_j · w = 0` for each field `w` of the slot after `pad` |
| O8 | `h_rs_j = h_rs_{j−1} · (pad_j + (1 − pad_j) · f(RS_j))` |
| O9 | `h_ws_j = h_ws_{j−1} · (pad_j + (1 − pad_j) · f(WS_j))` |

| # | Row, for scan slot `j` with cell `g = idx · B_scan + j` |
|---|---|
| S1 | every IS and FS coordinate and auxiliary bit is 0 or 1 |
| S2 | `h_is_j = h_is_{j−1} · f(t^IS_j, g, v^IS_j)` |
| S3 | `h_fs_j = h_fs_{j−1} · f(t^FS_j, g, v^FS_j)` |

The fingerprint is `f(t, g, v) = g + η1 · v + η1² · t − η2` in `K` (`Spec.Nebula.fingerprint`), Nebula's `Hash` with the global index for the address. Each product starts from its carry value. The step's outputs are `ts + cnt_{B_ops−1}`, which must be below `2^W_ts`, the last value of each product, and `idx + 1`. Rule 3 keeps every term of O4 below `2^62`, so O4 is an integer equation and gives `rt_j < wt_j` (`Spec.Nebula.StepRows`, `Spec.Nebula.OpSlot.Rows`).

### 7.4 Record chains and the memory challenge

These hashes use a framing that section 5 does not use (`Lifecycle.Nebula.absorbBlock`, `Lifecycle.Nebula.squeezeDigest`). To absorb a block, absorb its length as one word, then its words, in chunks of 12 words, each chunk added to lanes 0 to 11 and followed by one permutation. `Digest(tag, blocks)` starts from the zero state, absorbs the tag as a block of one word for each ASCII byte, then each block, and outputs lanes 0 to 3.

```text
chain_ops(j, D, P) = Digest("Nightstream/Nebula/v3/chain-ops", [j, D], P)
chain_mem(j, D, P) = Digest("Nightstream/Nebula/v3/chain-mem", [j, D], P)
header_ops         = Digest("Nightstream/Nebula/v3/header-ops", [plan_digest])
header_mem         = Digest("Nightstream/Nebula/v3/header-mem", [plan_digest])
```

The chain of lane `l` over the steps `0` to `N − 1` of a segment is `D[0] = header_l`, `D[j + 1] = chain_l(j, D[j], pack(lane l of step j))` and root `D[N]` (`Spec.Nebula.chainRoot`). The IS and FS lanes both use `chain_mem`, so the FS root of one segment equals the IS root of the next when the snapshots are equal. Equal roots give equal packed lanes or a Poseidon2 collision among the chain inputs (`Spec.Nebula.chainRoot_eq_or_collision` (`fcca3437c4c86e42`)). `D_init` is the IS root of the initial snapshot: the image words with timestamp 0 (`Spec.Nebula.Context.initRoot`).

At the start of a segment, the prover proposes the ops root `D_pre.ops` and the FS root `D_pre.fs`. The IS root must be `D_mem` of the carry: the FS root of the last segment, or `D_init`. The challenge is

```text
absorb the block of the ASCII bytes of "Nightstream/Nebula/v3/eta"
absorb [plan_digest], then [ts], then [D_pre.ops, D_mem, D_pre.fs]
η1 ← read lane 0, permute, read lane 0, permute
η2 ← read lane 0, permute, read lane 0
```

(`Lifecycle.Nebula.etaChallenges`). Each step evaluates it, and only a step that opens a segment keeps it. Canonical inputs that differ absorb different chunks (`Lifecycle.Nebula.etaInputBlocks_injective` (`fa605102ccb54dad`)). `D_mem` must be absorbed: without it, a prover could learn `η` first and then choose the last segment's final snapshot.

### 7.5 Memory carry

The carry is these fields, in order, as 39 words: each integer is one word, each element of `K` two words and each digest four words (`Lifecycle.Nebula.carryWords`).

| Field | Content |
|---|---|
| `seg_idx` | the segments closed so far, at most `S_max` |
| `idx` | the steps run in the current segment, at most `N` |
| `ts` | the global timestamp |
| `η1`, `η2` | the challenges of the current segment |
| `h` | the four products: `rs`, `ws`, `is`, `fs` |
| `D_pre` | the proposed ops and FS roots |
| `D_seen` | the three chains of the current segment: ops, IS, FS |
| `D_mem` | the FS root of the last closed segment, or `D_init` |

The four state words are `Digest("Nightstream/Nebula/v3/state", app_words, carry_words)`, where `app_words` are the application's own words (`Lifecycle.Nebula.stateWords`). Between two steps, `1 ≤ idx ≤ N`, and `idx = N` means that the segment is closed. The start carry has `seg_idx = 0`, `idx = N`, `ts = 0`, `η = 0`, each product 1, and `D_init` in each digest field (`Spec.Nebula.startCarry`). It is closed, so the first step opens a segment.

The actions on a carry `c` are (`Spec.Nebula.openSegment`, `Spec.Nebula.stepSegment`, `Spec.Nebula.closeSegment`):

- **open**, with the prover's proposals: require `c.seg_idx < S_max`, set `D_pre` to the proposals, `η` by section 7.4, each product to 1, `D_seen` to `(header_ops, header_mem, header_mem)` and `idx` to 0.
- **step**: add the step's packed lanes to the three chains of `D_seen` at index `idx`, set `ts` and `h` to the outputs of section 7.3, and add 1 to `idx`.
- **close**, exactly when step sets `idx = N`: require `D_seen.ops = D_pre.ops`, `D_seen.is = D_mem`, `D_seen.fs = D_pre.fs` and `h_is · h_ws = h_rs · h_fs`. Then set `D_mem` to `D_pre.fs` and add 1 to `seg_idx`.

### 7.6 Application ports

An application gives its state words, its private words and its rows (`Lifecycle.Nebula.PortApp`). Its rows read only its view (`Lifecycle.Nebula.AppView`): the input and output state words, the private words, and the words of each of its `B_ops` operation slots. It proves two facts. Soundness: when its rows hold and each slot obeys O1, O3 and O7, the slots, read as ports, are one application step between the two states. Completeness: each application step has private words that satisfy its rows. An idle step, with every port inactive and the state unchanged, must be valid, because a segment has exactly `N` steps. The test application is a two-word machine with two ports (`Lifecycle.Nebula.Machine.portApp`). The free application has no row, so the memory rows are its only checks (`Lifecycle.Nebula.freeApp`).

Rust `nebula::Circuit::compile` builds a memory relation for any `nebula::Application`. It rejects a geometry that breaks section 7.1, an application whose port count is not `B_ops`, and a row that reads outside the view. A read past the end of the view is zero, as in Lean (`crates/nightstream/src/nebula/`; not checked against Lean). For the machine at the geometries of the two Lean memory packages, its rows equal the Lean rows (`crates/nightstream/tests/nebula/parity.rs`). For another application or geometry, no Lean package fixes the memory rows, and the application's rows need their own soundness and completeness.

**Catalog applications.** A catalog lists memories, each with an id, a kind (RAM or ROM) and ports, in the form of the Rocq proof of the WASM relation (`MemoryTrace.v` in `formal/wasm-rocq` on the `main` branch, copied in `Spec/Nebula/MemoryTrace.lean`). Each active port of a row gives one event. A binding (`Spec.Nebula.MemoryTrace.Binding`) gives each bound memory a region: a base address and one bit width for each address part, with the address `base + x₀ + 2^w₀ · (x₁ + 2^w₁ · …)`. A RAM memory is in the RAM space, and each other memory in the ROM space. Two memories of one space have disjoint regions (`Spec.Nebula.MemoryTrace.Binding.Disjoint`). A read event is a read access of its value. A write event is a write access of its value, which reads its prior value when it has one. The active slots, in order, must bind a reorder of the events that keeps each pair of one memory with a write in trace order (`Spec.Nebula.MemoryTrace.Reorders`). When the regions are disjoint, every event is in range and the initial state holds the program images (`Spec.Nebula.MemoryTrace.Binding.Initial`), an execution whose active ports bind such a reorder makes the trace memory consistent (`Spec.Nebula.MemoryTrace.memoryConsistent_of_executes` (`6c4718da6b0020b0`)).

Rust `nebula::CatalogBinding` makes these rows for the checked memories, from a catalog, a region for each checked memory and a slot plan (`crates/nightstream/src/nebula/catalog.rs`; not checked against Lean). For a port `p`, `a_p` is its activation: 1, the column `c`, or `1 − c`.

1. **Slot.** `1 − pad_j = Σ a_p`, `is_write_j = Σ a_p` over the write ports and `is_ram_j = Σ a_p` over the RAM ports, for the ports `p` of slot `j`, and `c · (c − 1) = 0` for each activation column `c`.
2. **Order.** For each port `p` with a dependent port `p'` of a higher rank in an earlier slot, `a_p · Σ a_p' = 0`. Two ports are dependent when they are ports of one memory and one of them writes.
3. **Address.** The slot address is the region's base plus the packed parts. The base of a region is a multiple of its size `2^Σw`, so the rows read each part from its run of the slot's address bits and the bits above the region from the base, and a part needs no other range check.
4. **Values.** A read port has `v_r = value`. A write port has `v_w = value`, and `v_r = prior` when it names a prior column.
5. **Host checks.** A memory without a region has no slot and no row. At build, the regions of one space are disjoint and inside the space, the plan has `B_ops` slots, and it names each checked port once.

When the plan names each checked port once, each slot shows at most one active port, and of two dependent active ports in different slots the port of the earlier slot has the lower rank, the events of the active ports in slot order are a reorder of the row's events that keeps each dependent pair (`Spec.Nebula.MemoryTrace.reorders_of_slotOrder` (`2b1d9f7b165426bd`)). The slot plan changes only completeness and cost. `CatalogBinding::program` puts each preloaded cell of a checked memory at `base + packParts(a)` of its image and rejects a cell outside its region or of a memory that the catalog does not have. No Lean result shows that the five rules above enforce these conditions, or covers the address and value rows (`SECURITY_MODEL.md`, Out of scope).

**Several rows in one step.** One step can hold `k` rows of a catalog application, so that the rows share the cost of one fold. The step uses the batched catalog (`Spec.Nebula.MemoryTrace.batchCatalog`). It has the memories of the catalog in order. Each memory has the ports of row 0, then of row 1, up to row `k − 1`, and a map `shift i` moves the columns of row `i` to columns of the step witness. At `shift i c`, the step witness holds the value of row `i` at column `c`, for each column `c` that a port reads (`Spec.Nebula.MemoryTrace.HoldsRows`). So a step lists its events memory by memory, and its rows list them row by row. When the memory ids are distinct, each memory's events come in the same order in the two lists. So batched steps are memory consistent exactly when their rows, in order, are memory consistent (`Spec.Nebula.MemoryTrace.memoryConsistent_batchCatalog_iff` (`8884c58e0831ed34`)). When the regions are disjoint, every event of the rows is in range and the initial state holds the program images, an execution whose active ports, step after step, bind a reorder of the batched events of each step makes the trace of the rows memory consistent (`Spec.Nebula.MemoryTrace.memoryConsistent_of_batched_executes` (`d36926c9499e93e5`)). A slot plan that obeys the conditions above in each step gives this reorder (`Spec.Nebula.MemoryTrace.memoryConsistent_of_slotOrder_executes` (`763d05fb5fd4b167`)). Rust `neo_application::MemoryCatalog::batched` builds the batched catalog, and the catalog rejects two memories with one id. The application of an R1CS relation of one row, `nebula::R1csApplication`, keeps the constant-one column for all rows and moves each other column of row `i` by `i` times the row width. Its relation also makes each row's `previous_step_column`s equal to the next row's `next_step_column`s, and its slot plan is the one-row plan for each row in turn. It refuses a size that gives a memory of the catalog no region, so that no memory of it is only host-checked. `RowRelation::of` refuses a relation with a verifier-supplied column other than the constant-one column, because no memory verifier supplies such a column (`crates/nightstream/src/nebula/r1cs.rs`; not checked against Lean). The WASM application `neo_wasm::nebula::WasmApplication` is this application of the WASM relation, with column 0 as the constant-one column. `neo_wasm::nebula::WasmSize` fixes three sizes of it, each a geometry, the rows of one step and the regions of `WasmLimits::regions`. A WASM verifier takes the size from its statement, never from the run (`decisions/wasm-fixed-sizes.md`).

### Algorithm 9. Memory step

**Input.** The input state, the carry `c` and the application words that open it, the output application words, the private words, the step's lanes, and the proposals.

**Output.** The output state.

**Steps.**

1. Require that the input state is the state digest of the input application words and `c`.
2. If `c.idx = N`, run open with the proposals.
3. Run the rows of section 7.3, then step.
4. If `idx = N`, run close.
5. Require the application's rows.
6. Set the output state to the state digest of the output application words and the new carry.

**Rejects.** The relation does not hold.

**Source.** `Lifecycle.Nebula.MemoryApp.program`, the step `Spec.Nebula.invoke`. Rows that hold imply the program's step and validity predicates (`Lifecycle.Nebula.MemoryApp.soundness` (`f1f3e13012f71ac2`)). On a reachable input carry, they give one step of `invoke` (`Lifecycle.Nebula.StepWitness.RowsHold.invoke` (`fb47b498ac87650e`)), and an honest step satisfies the rows (`Lifecycle.Nebula.MemoryApp.completeness` (`292c1ed52f1fd1da`)). Each execution of `S · N` steps with `1 ≤ S ≤ S_max` has a step history whose statement passes Algorithm 10 steps 3 and 4 (`Lifecycle.Nebula.MemoryApp.history_of_execution` (`a1ef468b8bc6e701`)). The first step is the base step of the light circuit (Algorithm 3, step 2).

### Algorithm 10. Memory verifier

**Input.** The package of the application at its geometry, a program, the initial and final application words, and a memory proof: the final carry with a step proof or a final proof. The proof bytes start with the 39 carry words, and each word must be below `q`. The statement is the step count `T`, the two application words, the segment count `S`, the final timestamp and the final memory root. The verifier takes `S = seg_idx`, `T = S · N`, the final timestamp `ts` and the final memory root `D_mem` from the final carry, so the caller never gives a carry or a statement integer.

**Output.** Accept or reject.

**Steps.**

1. Require that the program has the package's geometry and valid images, and compute `D_init`.
2. Require that each state of the statement has the application's word count, and that each application word and the final timestamp is below `q`. Each other field and carry counter must equal a value below `q`.
3. Compute the initial state as the state digest of the initial words and the start carry, and the final state as the state digest of the final words and the final carry.
4. Require `idx = N` in the final carry, `1 ≤ S ≤ S_max`, `T = S · N` and `seg_idx = S`, and require that the final timestamp and the final memory root are `ts` and `D_mem` of the final carry. So `T = 0` is rejected.
5. Run Algorithm 6 on `(T, initial state, final state)` and the proof, or Algorithm 8 on the final proof.

**Rejects.** Each step's failure.

**Source.** `nebula::Verifier::verify` and `nebula::Verifier::verify_final` (`crates/nightstream/src/nebula/proving.rs`), with steps 2 to 4 in `terminal_states` (`crates/nightstream/src/nebula/mod.rs`; not checked against Lean). Steps 3 and 4 are `Lifecycle.Nebula.TerminalChecks`. A chain of valid steps with these checks gives a run that the memory model accepts, or a state-digest collision in the run (`Lifecycle.Nebula.accepts_or_collision` (`31d12d5e456bee76`)). For an honest execution, steps 3 and 4 pass and Algorithm 6 accepts the honest proof (`Export.OptionB.NebulaCompleteness.memory_complete` (`20b62362ffe2ee1c`)). Algorithm 6 alone does not make steps 2 to 4, so `nightstream::Circuit::verifier` refuses a memory relation: a pinned Lean memory package, a circuit from `nebula::Circuit::compile`, or a saved memory circuit, whose file starts with `NSOPTM`, 0, 1 (`crates/nightstream/src/circuit.rs`; not checked against Lean). A verifier compiles its own circuit and never loads a saved file.

## 8. Differences from the papers
## 8. Differences from the papers

Each row is a place where Nightstream differs from SuperNeo v1.2 (ePrint 2026/242), HyperNova or Nebula.

| Paper | Nightstream | Record |
|---|---|---|
| SuperNeo v1.2 Section 8.2 gives a Goldilocks example with `κ = 18`, `k = 14`, `B = 2^14` and `m = 2^30`. | The profile binds `κ = 22`, `k_rho = 16` and `B = 2^16`, and the row cube is `2^28`. It is called "Nightstream Goldilocks, k_rho = 16". | `AGENTS.md` (Protocol), `decisions/fprime-stage1-main-ajtai-setup.md`, `decisions/fprime-stage1-domain-2p28.md` |
| SuperNeo v1.2 Section 7 runs Π_CCS, Π_RLC and Π_DEC in each fold. | Two step circuits. A light fold has no Π_DEC and combines the accumulator with the weight 1, so its norm grows by `T(b − 1)` for each fold (Cyclo, ePrint 2026/359, Theorem 3). A refresh fold runs Π_DEC on the accumulator before Π_CCS and Π_RLC. The light circuit rejects a fold at depth 64 (Algorithms 3 and 4). | `FPRIME_LEAN_ARCHITECTURE_SPEC.md` (amendment of 2026-10-31) |
| SuperNeo v1.2 Section 8 sets the MSIS norm to `8TB`. | Extraction recovers accumulators up to `B + 64·T·(b − 1)`, so binding needs MSIS at the chain norm 137,134,080 (section 3). | `FPRIME_LEAN_ARCHITECTURE_SPEC.md`, `SECURITY_MODEL.md` trust item 2 |
| SuperNeo v1.2 Definition 8 samples the commitment matrix uniformly at random. | Algorithm 1 expands the matrix from one fixed public seed with SHAKE128. The premises are SHAKE128 as a random oracle and MSIS for a uniform matrix. | `decisions/fprime-ajtai-shake128-setup.md`, `decisions/fprime-ajtai-shake128-security-argument.md` |
| HyperNova Construction 2 step 5 hashes the running instances in the state hash. | The state hash hashes the digest of each slot, and each slot digest hashes the accumulator and its depth under its own domain (section 4.2). | `Lifecycle/OptionB/State.lean` |
| HyperNova Construction 3 absorbs the structure, the public parameters and the running instance before the first challenge. | The fold transcripts absorb a domain block, the fresh statement, the Π_DEC children of a refresh fold and the prover messages. The accumulator enters through the state hash in the fresh public input. | `decisions/piccs-prior-state-digest.md`, `SECURITY_MODEL.md` trust item 3 |
| HyperNova Definitions 7 and 11 let the adversary or the IVC prover choose the structure. | The verifier's packages fix one application, and their identity is pinned (Algorithm 2). | `SECURITY_MODEL.md` (Applications and setups) |
| HyperNova's history result uses Definition 7 knowledge soundness of the NIFS. | Lean states the history result for each slot under errata Assumption 1 for the light and refresh folds. | `SECURITY_MODEL.md` trust item 3 |
| HyperNova NIVC selects the circuit of a step with `φ(z_i, w_i)`. | The prover picks the circuit: Lean takes the selector that returns the circuit that ran, and Rust selects it from the depth of the selected slot (lifecycle rule 4). Both circuits run the same step, so the pick does not change the computed state. | `SECURITY_MODEL.md` trust item 4 |
| SuperNeo samples each Π_RLC challenge uniformly from the strong set. | Each challenge is four sponge lanes reduced modulo `5^54` (section 5.3), within `2^−132` of uniform. | none |
| Nebula Section 4.3 proves `F`, hashes the carried commitments into the challenges, proves `F_ops` and `F_scan` in a second layer, and folds the three proofs with `F_final`. | One F′ chain. At the start of a segment, the prover proposes the ops and FS roots, and `η` comes from them, `D_mem`, `plan_digest` and `ts` (section 7.4). The close of the segment makes the `F_final` checks: the chains equal the proposals, the memory continues, and the products balance. | `decisions/nebula-memory-checking.md`, `SECURITY_MODEL.md` trust item 6 |
| Nebula Sections 4.2 and 4.3 take one challenge pair in the base field. | `η1` and `η2` are in `K`. Over `F_q`, the fingerprint error of a geometry at the rule 6 limit is about `2^−45.9`. | `decisions/nebula-memory-checking.md` |
| Nebula Constructions 1 and 2 chain commitments to a split-committed witness. | The record chains are the hash chain of Nebula Section 3.1 over the packed lanes, computed inside F′, as in the generic construction of Section 3.2. | `decisions/nebula-memory-checking.md` |
| In Nebula Section 4.3, a segment has as many steps as the proof, and the scan has that many chunks. | A segment has `N` steps and the scan has `B_scan = (R + M)/N` slots in each step. Idle steps fill the last segment. | `decisions/nebula-memory-checking.md` |
| Nebula Section 4.2 has one memory. | ROM and RAM share one index space. O5 forbids a ROM write, and O6 keeps a ROM address below `R`. | `decisions/nebula-memory-checking.md` |
| Nebula Construction 2 hashes the carried commitment as its own field of the state hash. | The memory carry is in the application-state digest, so the F′ state hash and layout do not change. | `decisions/nebula-memory-checking.md` |

The `F_ops` checks of Nebula are O4 and the write timestamp `ts + cnt_j`, and the `F_scan` address check is the computed cell of a scan slot. The published `F_scan` has no timestamp range check, and Nightstream adds none: one segment is sequentially consistent without it (`Spec.Nebula.segment_sequential` (`b186065f1fe7272e`)).

The scope choices that the papers leave open are records too: the reference application (`decisions/fprime-stage1-poseidon2-hash-chain-v1.md`), the 41-trit opening of field values in binding maps (`decisions/41-trit-encoding.md`) and the Poseidon2-only rule for protocol binding (`AGENTS.md`, Protocol).

## 9. Conformance

An implementation conforms when it reproduces these vectors and passes these checks. `SHA256SUMS` in each artifact folder pins the bytes of the Lean exports. CI runs each check that this table marks "(CI)".

| Vectors | Check | Covers |
|---|---|---|
| `formal/nightstream-fprime/artifacts/` | `python3 -B formal/nightstream-fprime/scripts/check_artifacts.py` (CI) | the digests of the Lean exports and of their copies in `crates/nightstream-fprime/artifacts/` |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-option-b-parity-v1.json` | `cargo test -p nightstream-fprime --release` (CI) | Rust's expansion of the light and refresh packages against Lean's row count and checksum, Rust's identity code against Lean's probe packages, the witness program and the logical assignment against Lean (`crates/nightstream-fprime/tests/option_b_packages.rs`) |
| `crates/nightstream/tests/fixtures/poseidon2-application-reference.json`, `crates/nightstream/tests/fixtures/poseidon2-application-execution.json` | `cargo test -p nightstream --release --lib application::` (CI) | the reference application's rows and witness against a recorded Lean export (`crates/nightstream/tests/fixtures/README.md`) |
| `crates/nightstream/artifacts/` | `cargo test -p nightstream --release --lib artifacts::` (CI) | each embedded zstd copy against its Lean export (`crates/nightstream/artifacts/README.md`) |
| the light, refresh and finish packages | `cargo test -p nightstream --release` (CI) | the base step of each circuit, light and refresh folds into both slots with the edits of lifecycle rule 3, and final proofs (`crates/nightstream/tests/option_b_base_steps.rs`, `crates/nightstream/tests/option_b_api.rs`, `crates/nightstream/tests/option_b_lifecycle/mod.rs`) |
| the finish package and the layer-1 keys | `formal/compression-lean/scripts/crosschecks.sh` | the final program, the layer-1 runs and setups against Lean (`crates/nightstream/tests/option_b_lifecycle/final_ir.rs`, `crates/nightstream/tests/option_b_lifecycle/setup_runs.rs`, `formal/compression-lean/TRUST.md`) |
| `formal/compression-lean/ports.json` | `cargo test -p nightstream --release --test lean_ports` (CI) | each Lean port names a Rust item and the hash of its tokens (`crates/nightstream/tests/lean_ports.rs`) |
| the four option-B memory packages | `cargo test -p nightstream --release --lib nebula::` (CI) | the Rust-built memory relation of the machine against the row digests of both Lean memory applications (`crates/nightstream/tests/fixtures/nebula-memory-v1-digest.json`, `crates/nightstream/tests/fixtures/nebula-memory-n1-digest.json`); the catalog rows; reads past the view. Ignored tests make memory proofs and check the rejection cases that need a bad witness (`crates/nightstream/tests/nebula/parity.rs`, `crates/nightstream/tests/nebula/conformance.rs`) |
| `crates/nightstream/tests/fixtures/nebula-memory-v1-terminal.json`, `crates/nightstream/tests/fixtures/nebula-memory-n1-terminal.json` | `cargo test -p nightstream --release --lib nebula::memory` (CI) | Algorithm 10 against the Lean terminal checks, `D_init` against Lean, and statements that are not canonical or have another word count |
| the WASM application | `cargo test -p neo-wasm --release` (CI) | the catalog rows of a WASM program at 1, 2 and 4 rows in each step, a load that reads a store of an earlier row of its step, and a step of rows that are not neighbors; ignored tests prove it at 1 and 2 rows in each step |

These checks cover their recorded inputs. No theorem covers arbitrary Rust execution (`SECURITY_MODEL.md`, trust item 5).
