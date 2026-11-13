# Nightstream specification

## 1. Status and authority

This document describes the protocol that the `nightstream` crate runs. Each step is HyperNova recursion over SuperNeo v1.2 folding. There are two step circuits, a light circuit and a refresh circuit (section 2.5, Algorithm 3). Section 7 adds Nebula memory checking. The Lean package in `formal/nightstream-fprime/` is normative. This document restates it for readers outside the project, such as an auditor or a second implementer. It is not a second authority. Where the two differ, Lean is correct and this document has a defect.

Rust in `crates/` implements the protocol and is not proved. Lean emits the circuit packages. Rust reads them from the caller and compares their identity with a pinned value (Algorithm 2). A statement that only Rust fixes is marked "(not checked against Lean)".

Lean proves these results. `formal/nightstream-fprime/SECURITY_MODEL.md` gives the adversary of each result and what you must trust.

- If the rows of the light circuit hold and its public input encodes a digest, the light case of the step relation holds on the step that the assignment decodes to (`Export.BoundedDepth.StepRelation.lightEnv`), with that digest as its output hash. The refresh circuit has the same result (`Export.BoundedDepth.StepRelation.light_stepHolds` (`d4fb535cf406e269`), `Export.BoundedDepth.StepRelation.refresh_stepHolds` (`e3a678fd18b88c69`)). No hardness premise is used. Both results take a premise about the circuit that the application selects (Algorithm 3).
- If the terminal relation accepts a proof, and the key of each circuit carries the relation that Lean exports for that circuit, then the circuit that ran last holds its step relation under these keys on the step that the fresh witness decodes to, and that step ends in the terminal's statement, or the decoded next state and the terminal's state collide, or the step is light and its next state's words and the terminal's state preimage collide (`Export.BoundedDepth.TerminalBridge.bridge` (`91114bf46368f23e`), `Export.BoundedDepth.TerminalBridge.Collision`). The decode is a function of the proof (`Export.BoundedDepth.TerminalBridge.Exports.decode`), so the collision names two inputs that a reduction computes from the proof. The light circuit reads only the active slot's words, so a light step's other slot is the terminal's slot (`Export.BoundedDepth.StepRelation.lightInput`). Its step relation needs the other slot's prior digest word to be that slot's digest, and the rows copy that word into the next state's words, which the output hash hashes (`Export.BoundedDepth.StepRelation.light_freshWords` (`2bc0d81f9185dfaa`)). The step's circuit selector returns the circuit that ran on a valid step (`Export.BoundedDepth.TerminalBridge.chosen`), so the prover picks the circuit, as Rust does from the slot depth, and the decoded step's witness is valid for the application. The terminal does not read the selector (`Export.BoundedDepth.TerminalBridge.holdsFor_recursive_control` (`2b6e69454378e7e8`)). Its premises hold for the reference packages, whose two circuits run one application (`SECURITY_MODEL.md`, trust item 4).
- From two accepted outputs of one light proof that differ only in the challenge `ρ`, extraction gives a fresh witness with norm below `b` and a witness of the input accumulator at the output bound plus `T(b − 1)`, or a Π_CCS bad event occurred (`Spec.Folding.Light.accepted_implies_securityOutcome` (`60b78f12dd0858af`)). For a refresh fold, a fork of its challenges gives the fresh witness and a witness of the input accumulator below `B`, or a Π_CCS bad event occurred (`Spec.Folding.Refresh.accepted_implies_securityOutcome` (`ba69a891a59c9241`)). These results are deterministic. Section 3 gives the error numerators of the bad events.
- Along an accepted history of one slot, the bound at which extraction recovers each accumulator is at most `B + cap·T·(b − 1)` (`Spec.HyperNova.Construction2.BoundedDepth.accepted_bounds` (`55d6d9921f728747`), computed on the recovered bounds; `Spec.HyperNova.Construction2.BoundedDepthKnowledge.execution_extracts` (`4288991b81eaa3d9`) gives the extraction). A relaxed binding collision at twice such a bound gives a kernel vector of the Ajtai key with norm below the chain norm 75,497,472 (`Spec.RingRelation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel`, `Spec.Folding.Schedule.production_msisChainBound` (`2454276b3c2234dd`)).
- Under HyperNova errata Assumption 1 for the light and refresh folds, the closure premise for the reverse extractor's stages, and an admitted IVC adversary (`SECURITY_MODEL.md`, trust item 3), the probability that the terminal relation accepts a statement and a proof but no `iteration` valid steps of the application, each with a witness of the application's word count, lead from `z0` to `zi` is at most the sum over the reverse extractor's stages of the stage's error and the probability of a Poseidon2 collision at the stage's visit (`Export.BoundedDepth.IvcSecurity.probability_bound` (`6160032bde6ff3c3`)). The same sum bounds the probability that the terminal relation accepts and the witnesses that the reverse extractor returns are not such steps (`Export.BoundedDepth.IvcSecurity.extraction_bound` (`dd0c7f22441b3a9c`)). This is the knowledge form of the result, and the first bound follows from it. The reverse extractor walks back from the terminal one step per stage: the fresh witness decodes to the last step, whose fold of the selected slot is the stage's attempt, and the extractor's answer gives the prior visit (`Export.BoundedDepth.IvcWalk.covered` (`8f667eace2076700`)). Each slot keeps its own witness bound, which grows by `T(b − 1)` per light fold and returns to `B` at a refresh fold, so every attempt stays within `B + cap·T·(b − 1)`. The collision at a visit is on two states that the stage computes from its tape (`Export.BoundedDepth.TerminalBridge.Collision`). Each attempt is linked to the prior state of its step (`Lifecycle.BoundedDepth.PriorLink`): two folds with one fresh instance that link to well-formed prior states start from one running claim, or the two prior states give a state-hash collision (`Lifecycle.BoundedDepth.priorLink_binds` (`176d6c368f0f870c`)). In a random-oracle model of the two folds, Assumption 1 holds with an explicit error: the folds' knowledge errors and a truncation loss for an extractor with a fixed number of reruns (`Lifecycle.BoundedDepth.RandomOracle.Assumption1.romAssumption1` (`e63c8ee3620af200`)). The knowledge error of each fold is a statistical term, at most `(Q + 58) · 575/2^128` for light and `(Q + 74) · 5390/2^128` for refresh with `Q` oracle queries, plus the probabilities of a short kernel vector of the Ajtai key and of a collision on two named prior states (`Lifecycle.BoundedDepth.RandomOracle.LightKnowledge.knowledge_error_le` (`b62cb37f49fb6198`), `Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge.knowledge_error_le` (`ed3805e7bec2fe93`)). Averaged over the SHAKE128 key setup, the kernel term is the advantage of an explicit MSIS solver on a uniform matrix at the chain norm and the collision terms are the advantage of an explicit Poseidon2 collision finder, plus a programming error below `2^−192` (`Export.BoundedDepth.RandomOracleSetup.light_production_knowledge_error_lt_solvers` (`adbddf8d8302a4d7`), `Export.BoundedDepth.RandomOracleSetup.refresh_production_knowledge_error_lt_solvers` (`cc9443b209366a62`)). At an oracle that answers like the Poseidon2 sponge, the model's events are the walk's (`Export.BoundedDepth.RomWalk.succeeds_iff_walkSucceeds` (`2121c56f0d2ebe4a`)). The step circuits recompute the challenges with Poseidon2, so the IVC bound keeps Assumption 1 for the concrete hash as a premise (`SECURITY_MODEL.md`, trust item 3). The bound holds for every depth, but each stage's adversary reruns the earlier stages, so it gives no concrete security for a long history. HyperNova proves knowledge soundness only for a constant number of steps (`SECURITY_MODEL.md`, trust item 3). For one slot, the probability that an accepted history is not returned is at most the sum of the stages' errors (`Spec.HyperNova.Construction2.BoundedDepthSecurity.history_failure_le` (`a3fc56cd24a9a455`)), and an accepted execution of both slots makes each slot's history accepted (`Spec.HyperNova.Construction2.BoundedDepthSecurity.slot_accepted` (`7732ad00c2d7b3f4`)).
- An honest light fold and an honest refresh fold are accepted (`Spec.Folding.Light.complete` (`00c5dff8506a54a2`), `Spec.Folding.Refresh.complete` (`65e5a003b7dd32c9`)). At the production keys, the honest fold of a slot claim is accepted by the bounded-depth folds, and its output is a slot claim at the input bound plus `T(b − 1)` after a light fold and below `B` after a refresh fold (`Lifecycle.BoundedDepth.FoldCompleteness.light_folds` (`3477fcd1f17aebfe`), `Lifecycle.BoundedDepth.FoldCompleteness.refresh_folds` (`9fd416e61549d32c`)). The light challenge always exists, because the Poseidon2 sponge is a deployed oracle. Along a schedule that the light circuit accepts, the bound `accumulatorBound` of each depth stays below `B` (`Spec.Folding.Schedule.honest_bounds` (`82d0559dff9c6fc8`)). Along an honest run, each slot's witness stays within the bound of its depth, so it stays below `B` (`Export.BoundedDepth.StepExtend.Honest`). Given an accepted fold, an honest light step and an honest refresh step each have an assignment in which every row of the step circuit's source layout holds (`Layout.BoundedDepth.LightAssemblerCompleteness.complete` (`1e199e676a723909`), `Layout.BoundedDepth.RefreshAssemblerCompleteness.complete` (`c98336ebd81d5bad`)). From such an assignment, the light and refresh keys accept a fresh instance whose public input is `encHash` of the step's output hash: the commitment to an honest assignment of the exported relation, read through the package's slot layout widened to the key width (`Export.BoundedDepth.StepCompleteness.light_complete` (`f9c60b6ea288caf5`), `Export.BoundedDepth.StepCompleteness.refresh_complete` (`51aade49086740ac`)). A package gives a slot only to a column that a row reads, and a bit slot only to a column that a row forces to 0 or 1 (section 4.4), so each bit slot of the honest assignment holds 0 or 1. This result takes three computed checks of the package as premises: the blocks pack, every piece reads only columns below the first fresh column, and no row of the circuit reads a public column. The emitter evaluates them and writes no step package when one fails (`Export/BoundedDepth/EmitMain.lean`). For every valid run of the application from `z0`, the terminal accepts the honest prover's proof (`Export.BoundedDepth.RunCompleteness.run_complete` (`1c17bf5906cf1f3f`)). One valid step extends each proof that the terminal verifier accepts (Algorithm 6) to a proof that it accepts at the next iteration, which is HyperNova Definition 11 (i) (`Export.BoundedDepth.RunCompleteness.verifies_step` (`2619c8b4213bc6d3`)). The first step folds the default fresh instance into the default slot with HyperNova's zero proof (`Lifecycle.BoundedDepth.BaseFold.zero_folds` (`b82d57e7f4f34804`)). Each later step folds the fresh instance into the selected slot, with a light fold below the cap and a refresh fold at it, as Rust does (`Export.BoundedDepth.StepExtend.light_extend` (`c2a7a09520dd7601`), `Export.BoundedDepth.StepExtend.refresh_extend` (`abd80d0c72fa9bb3`)). The premises are the three package checks, keys that carry the exported relations at the packages' widened slot layouts (`Export.BoundedDepth.StepExtend.Complete`), key digests of four words, and witnesses of the application's witness length.
- If a chain of valid memory steps ends in a state that passes the terminal checks of the memory verifier, the chain gives a run that the memory model accepts, or a state-digest collision occurred (`Lifecycle.Nebula.accepts_or_collision` (`7237d81815f8356a`)). Each honest execution has a history whose statement passes the statement checks of the memory verifier, Algorithm 8 steps 3 and 4 (`Lifecycle.Nebula.MemoryApp.history_of_execution` (`f12147627b97b445`)). For every such execution, the bounded-depth terminal accepts the honest prover's recursive proof of that statement, which is step 5 (`Export.BoundedDepth.NebulaCompleteness.memory_complete` (`6f8f45de5be0373a`)). Its premises are those of `run_complete`, with the memory program as the exported program. Under Assumption 1, the closure premise and the memory-round premise, the probability that the memory verifier accepts but the history that the reverse extractor decodes does not attest the claimed memory statement is at most the terms of the IVC bound, the state-link and chain collision probabilities of the extracted runs, and the memory-round error (`Export.BoundedDepth.NebulaSecurity.bad_le` (`4cbf10384e683406`), `SECURITY_MODEL.md`, adversary 5). In a random-oracle model of `η`, for a secure geometry and every `η` sampler with `Q_η` queries, the event of the memory round has probability at most `(Q_η + 1) · 2^−109.91` plus a retry term in which each counted pair is a Poseidon2 transcript collision (`Export.BoundedDepth.NebulaSecurity.romMemoryRound` (`50e26df5f83a58c1`)). This does not bound the deployed check, whose `η` is Poseidon2.

Lean has no statement for the compressed proof (Algorithms 9, 10 and 11). Rust implements it in `crates/neo-compress/` and `crates/nightstream/src/lifecycle/bounded_depth/compression.rs`, and Rust tests check it on fixed inputs (lifecycle rules 2 and 3). Its aim is that, when the final verifier accepts a statement, a proof with witnesses exists that the verifier of Algorithm 6 accepts for that statement (`Export.BoundedDepth.RunCompleteness.Verifies`), except with an error. No theorem states this aim, no theorem bounds the error, and no theorem joins the compressed proof to the results above. Section 3 gives the error terms that Rust computes and the terms that it does not compute. `SECURITY_MODEL.md`, adversary 6 and trust item 7, gives the adversary and the premises. `decisions/akita-compression.md` gives the design choices and which of them the owner accepted.

Every audited theorem uses only the axioms `propext`, `Classical.choice` and `Quot.sound`. The repository claims no single security level for the system (`SECURITY_MODEL.md`, Out of scope).

These commands check the development and this document. Run the Lean commands on a machine with the Lean toolchain of `formal/nightstream-fprime/lean-toolchain`.

```bash
cd formal/nightstream-fprime && bash scripts/validate.sh axioms
python3 scripts/check_spec.py
```

`validate.sh axioms` builds the tests and runs `formal/nightstream-fprime/tests/EndpointCensus.lean`. That census fails when this document cites a name in the namespaces `Export`, `Layout`, `Lifecycle` or `Spec` that does not exist, a theorem without a pin, or a pin with another fingerprint. The census also fails when a record in `decisions/` cites a name that does not exist. A pin is a theorem name followed by its fingerprint, such as `` `Name` (`0123456789abcdef`) ``. The fingerprint hashes the theorem's statement and every package definition that the statement reaches, so a change to one of them changes it. A change to a proof does not. When a pin fails, reread the section that cites it, then write the new fingerprint that the census prints. `scripts/check_spec.py` fails when a pin is malformed, when a backticked path does not exist, or when section 3's table is stale. CI runs `check_spec.py` and no Lean.

Names of the F′ package omit its namespace `NightstreamFPrime`. A Lean path in this document is under `formal/nightstream-fprime/NightstreamFPrime/`.

| Section | Lean |
|---|---|
| 2, notation | `Spec/Algebra.lean`, `Spec/RingStrongSet.lean`, `Lifecycle/BoundedDepth/State.lean`, `Spec/Folding/Light/Verifier.lean`, `Spec/Folding/Refresh/Verifier.lean` |
| 3, parameters | `Spec/Profile.lean`, `Spec/Folding/Schedule.lean`, `Spec/Folding/Light/ErrorBudget.lean`, `Export/BoundedDepth/StepPackage.lean` |
| 4, encodings | `Lifecycle/XOut.lean`, `Lifecycle/BoundedDepth/State.lean`, `Export/BoundedDepth/StepPackage.lean` |
| 5, transcript | `Spec/Poseidon2.lean`, `Lifecycle/Transcript.lean`, `Lifecycle/BoundedDepth/Keys.lean` |
| 6, algorithms | `Spec/AjtaiSetupV1.lean`, `Spec/Folding/Light/`, `Spec/Folding/Refresh/`, `Spec/HyperNova/Construction2/`, `Lifecycle/BoundedDepth/`, `Export/BoundedDepth/` |
| 7, Nebula memory | `Spec/Nebula/`, `Lifecycle/Nebula/`, `Export/Stage1/NebulaMemoryPackage.lean`, `Export/BoundedDepth/NebulaSecurity.lean`, `Export/BoundedDepth/NebulaCompleteness.lean` |

## 2. Notation

### 2.1 The field and the extension

- `q = 2^64 − 59 = 18446744073709551557`, a prime (`decisions/field-2p64-59-ring-x64.md`). `F_q` is `Spec.F`, the integers modulo `q`, each written as its value in `[0, q)`.
- `|x|` for `x ∈ F_q` is `min(x, q − x)` (`Spec.centeredMagnitude`). A vector `z` has norm below `β` when every entry has `|z_i| < β` (`Spec.normBounded`).
- `K = F_q[u]/(u² − 7)` (`Spec.K`). An element is `c_0 + c_1·u`, and `(a_0 + a_1 u)(b_0 + b_1 u) = (a_0 b_0 + 7 a_1 b_1) + (a_0 b_1 + a_1 b_0) u` (`Spec.K.mul`). 7 is not a square modulo `q`, so `K` is a field (`Spec.FieldExtension.sevenProjectiveNonresidue` (`dc898168aa41c8d7`)). `F_q` embeds in `K` as `c_0` (`Spec.K.embed`).

### 2.2 The ring and the challenge set

- `R_q = F_q[X]/(X^64 + 1)`. `X^64 + 1` is the cyclotomic polynomial `Φ_128`, so the ring degree is `d = 64`. An element is its 64 coefficients (`Spec.RingF`). Multiplication reduces the product with `X^64 = −1` (`Spec.ringFMul`).
- `C`, the strong set, is the ring elements whose coefficients are all in `{−2, −1, 0, 1, 2}` (`Spec.RingStrongSet.ProductionMember`), so `|C| = 5^64`. Every nonzero ring element of norm at most `β` with `2β² < q` is invertible (`Spec.RingStrongSet.lowNormInvertibility` (`d371a1825bcda9be`), SuperNeo v1.2 Theorem 4). The difference of two elements of `C` has norm at most 4, so it is a unit.

### 2.3 Commitments

- A witness `z ∈ F_q^m` packs into `⌈m/d⌉` ring elements: coefficient `h` of element `j` is `z_{d·j + h}`, or 0 past the end (`Spec.RingRelation.EvaluationHomomorphism.CarrierAction.assignmentBlock`).
- The Ajtai key `A` has `κ` rows of `⌈m/d⌉` ring elements. The commitment of `z` is `A · pack(z) ∈ R_q^κ` (`Spec.RingRelation.PiRLCAlgebra.Commitment.commit`). Algorithm 1 derives `A`.

### 2.4 The relations

- Each step circuit has its own relation, a CCS over `F_q` with `t = 4` matrices `M_0, …, M_3` (`Lifecycle.ProductionKey.LogicalRelation`). Each matrix has its rows on the cube `{0,1}^ℓ` and `m` columns, and each row past the relation's `n` rows is zero. An assignment `z` satisfies the CCS when `f((M_0 z)_r, …, (M_3 z)_r) = 0` for every row `r`, where `f = M_0·M_1 − M_2 + M_3^7` (`Spec.ProductionRelation.polynomial`).
- Both circuit keys have the logical width `m = 2^26` (`Export.BoundedDepth.StepRelation.keyWidth`). A circuit with fewer columns commits a witness that is zero past them.
- The public input of `z` is its first 320 entries, five ring elements (`Spec.RingRelation.projectPublicInput`, `Lifecycle.PaperAlgebra.publicRingColumns`).
- `eq(x, y) = ∏_i (x_i y_i + (1 − x_i)(1 − y_i))` for `x, y ∈ K^ℓ`. `ỹ(r)` is the multilinear extension of a table `y` on `{0,1}^ℓ`, evaluated at `r ∈ K^ℓ`.

### 2.5 Claims, slots and fold proofs

- A fresh claim is a CCS claim: a commitment `c ∈ R_q^κ` and a public input `x ∈ F_q^320`. Its witness satisfies the CCS of the circuit that made it, opens `c`, projects to `x` and has norm below `b`.
- An accumulator is a CE claim (SuperNeo v1.2 Definition 21). It has a commitment, a public input, a point `r ∈ K^ℓ` and one evaluation family (`Lifecycle.BoundedDepth.SlotClaim`). The family has `Eval_K`, 64 elements of `K` (the Pad evaluations), and `Eval_A`, `4 × 64` elements of `K` (the matrix evaluations). Its witness opens the commitment, projects to the public input, has norm below `B`, and gives these evaluations at `r`.
- A slot is an accumulator and its depth, the number of light folds since its last refresh (`Lifecycle.BoundedDepth.SlotData`). The state has two slots. Slot 0 accumulates the fresh claims of the light circuit, and slot 1 those of the refresh circuit.
- The default slot has the point 0, a zero commitment, public input and evaluations, and depth 0 (`Lifecycle.BoundedDepth.defaultSlot`, HyperNova Definition 12).
- A light proof has `ℓ` sum-check round polynomials of degree at most 8, each its 9 coefficients in `K` from degree 0 up, then the Π_CCS output, one evaluation family for each of the 2 sources, fresh first (`Spec.Folding.Light.Proof`).
- A refresh proof has the 16 Π_DEC children of the input accumulator, each a commitment and an evaluation family, then `ℓ` round polynomials and the Π_CCS output for the 17 sources, fresh first, then children 0 to 15 (`Spec.Folding.Refresh.Proof`). The refresh verifier does not read the output children of the NIFS proof type that it reuses (`Spec.Folding.Refresh.verify`).
- A final proof has the statement and, for one or more steps, the program counter, each slot's accumulator and depth, the CE claim that one Π_CCS run makes of the fresh instance, the `ℓ` round polynomials of that run, and the argument of Algorithm 9. It has no witness, and Algorithm 5 cannot extend it (`crates/nightstream/src/lifecycle/bounded_depth/compression.rs`; not checked against Lean).
- A compression key holds the Akita commitment of the Ajtai key as one polynomial, and Akita's commitments of prefixes of its own setup (Algorithm 11). The verifier takes the pinned key of its circuit's size, or loads bytes that must equal that key. No proof carries it (`crates/neo-compress/src/key.rs`; not checked against Lean).

## 3. Parameters

The table below is generated from Lean. Its values come from the integer equations of seven Lean theorems, two Lean definitions, the headers of the Lean-emitted packages of the reference application `Poseidon2HashChainV1` (`decisions/fprime-stage1-poseidon2-hash-chain-v1.md`), and the row counts of the Lean-emitted parity file. The names of the header fields come from their Lean format `Export.BoundedDepth.StepPackage.Package.format`. Another application has other packages, with the same profile but other sizes.

<!-- params -->
Generated by `scripts/check_spec.py`. Do not edit by hand: run `python3 scripts/check_spec.py --write`.

| Source | Lean | Symbol | Value | Meaning |
|---|---|---|---|---|
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.q` | `q` | 18446744073709551557 | The field modulus, `2^64 − 59`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.b` | `b` | 2 | The decomposition base and the fresh norm bound. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.k` | `k_rho` | 16 | The decomposition digits, and the Π_DEC children. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.maxFresh` | — | 495 | The largest fresh count with `(maxFresh + k_rho)·T·(b − 1) < B` (SuperNeo v1.2 Definition 22). |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.expansionT` | `T` | 128 | The expansion factor of the challenge set. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.global.bigB` | `B` | 65536 | The combined norm bound, `b^k_rho`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.eta` | `η` | 128 | The cyclotomic index: `R_q = F_q[X]/Φ_η`, and `Φ_128 = X^64 + 1`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.ringDegree` | `d` | 64 | The ring degree, `φ(η)`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.commitmentWidth` | `κ` | 18 | The rows of the Ajtai key. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.extensionDegree` | — | 2 | The degree of `K` over `F_q`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.challengeSetBitsFloor` | — | 148 | A floor of `log₂ |C|`. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.piRlcInputs` | — | 17 | The Π_RLC inputs of a refresh fold, the fresh claim and the 16 Π_DEC children. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.piDecChildren` | — | 16 | The Π_DEC children of the accumulator that a refresh fold splits. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.ccsMatrices` | `t` | 4 | The CCS matrices. |
| `Spec/Profile.lean`, `production_norm_stages` | `NormStage.bound productionGlobalParams .fresh` | — | 2 | The norm bound of a fresh witness, `b`. |
| `Spec/Profile.lean`, `production_norm_stages` | `NormStage.bound productionGlobalParams .combined` | — | 65536 | The norm bound of a Π_RLC parent, `B`. |
| `Spec/Profile.lean`, `production_norm_stages` | `NormStage.bound productionGlobalParams .ambient` | — | 9223372036854775779 | The ambient bound, `q/2 + 1`. |
| `Spec/Profile.lean`, `production_msis_norm_bound` | `productionGlobalParams.msisNormBound` | — | 67108864 | The MSIS norm `8TB` of SuperNeo v1.2 Section 8, the chain norm at cap 0. |
| `Lifecycle/Nifs/SelectedTestNumerator.lean`, `numerator_eq` | `numerator` | — | 5389 | The numerator of the refresh fold's Π_CCS test error, 17 sources. |
| `Spec/Folding/Light/ErrorBudget.lean`, `production_testNumerator` | `testNumerator (shape 28) 8` | — | 574 | The numerator of the light fold's Π_CCS test error, 2 sources. |
| `Spec/Folding/Light/ErrorBudget.lean`, `production_forkCoordinates` | `forkCoordinates` | — | 1 | The `ρ` coordinates of a light fork. A refresh fork has 17. |
| `Spec/Folding/Schedule.lean`, `production_msisChainBound` | `msisChainBound productionGlobalParams productionLightFoldCap` | — | 75497472 | The chain MSIS norm `8T(B + cap·T·(b − 1))`. |
| `Spec/Folding/Schedule.lean`, `productionLightFoldCap` | `productionLightFoldCap` | `cap` | 64 | The depth at which the light circuit rejects a fold. |
| `Export/BoundedDepth/StepRelation.lean`, `keyWidth` | `keyWidth` | — | 67108864 | The logical width of both circuit keys, `2^26`. |
| computed | `ε_test` | `ε_test` | 5389/q² ≈ 2^−115.60 | The refresh fold's Π_CCS test error, `numerator / q²`. |
| computed | `ε_test light` | — | 574/q² ≈ 2^−118.83 | The light fold's Π_CCS test error, `testNumerator / q²`. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.circuit` | — | 0 | The circuit index: 0 light, 1 refresh. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.publicColumns` | — | 320 | The public input columns, five ring elements. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.width` | — | 16777216 | The source width. Every source column is below it. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.firstFresh` | — | 765802 | The first fresh lowering column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.sboxStart` | — | 770165 | The first S-box output column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.pieces` | — | 5585 | The pieces, which are lowered rows, ring products and Poseidon2 blocks. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.witness` | — | 23 | Witness batches. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.slots` | — | 27 | Slot runs of the low-norm layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.application[0]` | — | 4 | The witness word count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.application[1]` | — | 18824 | The first local of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.application[2]` | — | 5480 | The local count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.application[3]` | — | 9 | The pieces of the application region, from the first piece. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | `light.application[4]` | — | 0 | The fresh columns of the application region, from `firstFresh`. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.circuit` | — | 1 | The circuit index: 0 light, 1 refresh. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.publicColumns` | — | 320 | The public input columns, five ring elements. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.width` | — | 16777216 | The source width. Every source column is below it. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.firstFresh` | — | 4649736 | The first fresh lowering column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.sboxStart` | — | 4688596 | The first S-box output column. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.pieces` | — | 65298 | The pieces, which are lowered rows, ring products and Poseidon2 blocks. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.witness` | — | 214 | Witness batches. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.slots` | — | 217 | Slot runs of the low-norm layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.application[0]` | — | 4 | The witness word count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.application[1]` | — | 62212 | The first local of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.application[2]` | — | 5480 | The local count of the application region. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.application[3]` | — | 9 | The pieces of the application region, from the first piece. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | `refresh.application[4]` | — | 0 | The fresh columns of the application region, from `firstFresh`. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-parity-v1.json` | `light.relationRows` | — | 11358 | R1CS rows of the light relation, from the pieces and the ring products. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-parity-v1.json` | `light.sboxRows` | — | 101100 | S-box rows of the light relation, 150 for each Poseidon2 block. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-parity-v1.json` | `refresh.relationRows` | — | 135507 | R1CS rows of the refresh relation, from the pieces and the ring products. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-parity-v1.json` | `refresh.sboxRows` | — | 611850 | S-box rows of the refresh relation, 150 for each Poseidon2 block. |

A probability is shown as `2^−x`, where `x` is `−log₂` of the exact fraction rounded down to two decimals, so the shown bound is never below the true value.
<!-- /params -->

Both folds use one Π_CCS shape. It has `ℓ = 28` rounds (`Lifecycle.cubeVariables`), 4 matrices, `d = 64` and round polynomials of degree at most 8. A light fold has 1 fresh source and 1 running source (`Spec.HyperNova.Construction2.BoundedDepthFolds.lightShape`). A refresh fold has 1 fresh source and 16 children. The light relation has `11,358 + 101,100 = 112,458` rows and the refresh relation has `135,507 + 611,850 = 747,357` rows (computed from the table). Both are below `2^ℓ`.

The light circuit rejects a fold of a slot at depth `cap = 64`. An honest accumulator at depth `j` has norm at most `accumulatorBound j = (1 + j + k_rho)·T·(b − 1)` (`Spec.GlobalParams.accumulatorBound`, Cyclo, ePrint 2026/359, Theorem 3). Lean proves the step for one light fold (`Spec.Folding.Light.complete` (`00c5dff8506a54a2`)). Lean also proves that, along an honest run, each slot's witness stays within the bound of its depth (`Export.BoundedDepth.StepExtend.Honest`, section 1). This bound is below `B` for each `j ≤ 494` (`Spec.GlobalParams.production_accumulatorBound_lt_bigB` (`f81f20b1576e69bc`)), and `1 + cap ≤ maxFresh` (`Spec.Folding.Schedule.production_cap_fits` (`bada6b7c2264f328`)). Extraction runs backward over a history. A light fold recovers its input at its output bound plus `T(b − 1)`, and a refresh fold recovers its input below `B`. So each recovered accumulator is at most `B + cap·T·(b − 1) = 73,728` (computed; `Spec.Folding.Schedule.recoveredBounds_le` (`793c72eedb9a44e6`)). The binding reduction then needs MSIS at the chain norm `msisChainBound = 8·T·(B + cap·T·(b − 1)) = 75,497,472`, about `2^26.17` (computed), in place of `8TB = 67,108,864` (`Spec.Folding.Schedule.msisChainBound`, SuperNeo v1.2 Section 8).

The error terms are these. The refresh fold's Π_CCS test error is `ε_test = (ℓ · 8 + 5165)/q² = 5389/q²` (`Lifecycle.Nifs.VerifierErrorBudget.test_error_eq` (`7bd6f7001837f16c`)), where 5165 is `5138 − 1 + ℓ` and 5138 is the number of joint coefficients (`Spec.Folding.PiCCS.PaperJoint.Shape.jointCoefficientCount`). The light fold has 323 joint coefficients, so its numerator is `ℓ · 8 + 323 − 1 + ℓ = 574` (`Spec.Folding.Light.ErrorBudget.production_testNumerator` (`8745110e6d180c33`)). A light fork has one `ρ` coordinate, and a refresh fork has 17 (`Spec.Folding.Light.ErrorBudget.production_forkCoordinates` (`b181aeed4a511f69`)). The law of each Π_RLC challenge is within `2^−173` of the uniform law on `C` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt` (`ae01cc1d1ba143a5`)). So one challenge takes a given value with probability below `1/5^64 + 2^−173` (computed). Lean does not add these terms into one bound. Rust `Params::for_ccs_shape` refuses a caller minimum above the zero-query error of the refresh fold in the random-oracle result (`Lifecycle.BoundedDepth.RandomOracle.RefreshKnowledge.statisticalError`), `74 · ε_test + 17² · (1/|C| + 2^−173)`. Rust puts its census numerator in `ε_test`. The census counts the 495 fresh sources that the Π_RLC guard `(K + k_rho) · T · (b − 1) < B` admits, so its numerator is `ℓ · 8 + (495 + 511 + 5120 − 1 + ℓ) = 6377`, which is not below Lean's 5389. The error is then about `2^−109.15`, and about `2^−109.39` with Lean's numerator (computed). So the profile accepts a minimum of at most 109 bits, and the tests use 109 bits (`the_gate_refuses_more_bits_than_the_fold_error_gives` in `crates/nightstream/tests/bounded_depth_api.rs`; not checked against Lean).

The compressed proof (Algorithms 9, 10 and 11) uses these values. Rust fixes them (not checked against Lean; `decisions/akita-compression.md`).

| Name | Value | Meaning |
|---|---|---|
| `L` | 4 | The lookup challenges of `F_q` (`LOOKUPS` in `crates/neo-compress/src/layout.rs`). |
| columns of `V` | 4 | One column for each lookup challenge. |
| `n_A` | 19 or 9 | The block variables of the key polynomial `Â` (Algorithm 11). 19 for the reference application, and 9 for the small claims of the tests. |
| variables of `Â` | `6 + n_A + 5`, so 30 or 20 | The lane, the block and the row of a key coefficient. `Â` has 32 rows, and each row from `κ = 18` is zero. |
| schedule rows | `(30, 26, 28)`, `(30, 27, 28)` and `(20, 16, 16)` | The variables of `Â`, `W` and `V` in each row of the Akita schedule files `crates/neo-compress/artifacts/key30.aks` and `crates/neo-compress/artifacts/key20.aks`. One row opens the three polynomials in one Akita proof. |
| `n` | 26 or 27 | The variables of `W` for the reference application: 26 when the light circuit ran last, and 27 when the refresh circuit ran last (computed from the layout of Algorithm 9). |
| variables of `V` | 28 | The value for the reference application (computed). |
| `T_s` | `2 · accumulatorBound(depth_s) + 1`, at most 20,737 | The values of the lookup table of slot `s`. |
| `c` | 6880439088830499533 | The constant of the field map of Algorithm 9. `2c² = 7` modulo `q`. |

The argument of Algorithm 9 has these error terms, with `|K| = q²`, `m = 320` public cells, `t = 4` matrices and `s = 3` claims (`error_bits` in `crates/neo-compress/src/lib.rs`; not checked against Lean).

| Term | Bound |
|---|---|
| The batch of the equations at `(ω, δ, γ, μ)` | `(κ + 1 + t + max(63, m − 1) + s − 1)/q² = 344/q²` |
| The lookup of slot `s`, whose witness slice has `N_s` cells | `((N_s + T_s)/q)^L` |
| The batch of the inverse checks and the sums at `(ψ, ξ, λ, β)` | `(2 + n)/q²` |
| The first sum-check, of degree 3 | `3n/q²` |
| The point for the columns of `V` | `2/q²` |
| The key sum-check, of degree 2 over `6 + n_A` variables | `2 · (6 + n_A)/q²` |

For the reference application, `N_0 = 2^23`, `N_1 = 2^25` and `n_A = 19`. At `L = 4` and `n = 27`, the lookups give about `2^−155.99`, the other terms give `(398 + 4n)/q² = 506/q²`, about `2^−119.02`, and the sum is about `2^−119.02` (computed). Rust refuses a layout whose sum is above the caller's security minimum. The gate of the folds already holds that minimum to 109 bits. This sum is not the error of a final proof. It does not include the knowledge error of the Akita opening, the error of the Π_CCS run of Algorithm 9, or the loss of Fiat–Shamir. Rust computes none of these three terms (`SECURITY_MODEL.md`, adversary 6). The sum holds only under the compression key of Algorithm 11.

## 4. Encodings

### 4.1 Words

Lean hashes and absorbs field elements, called words. A natural number `n` becomes the word `n mod q` (`Spec.Poseidon2.ofNat`, `Lifecycle.natWord`).

- Bytes become one word per 8 bytes, little-endian, reduced modulo `q` (`Lifecycle.packBytes`).
- An element `c_0 + c_1 u` of `K` is the two words `c_0, c_1` (`Lifecycle.serializeK`).
- A ring element is its 64 coefficients in order (`Lifecycle.serializeRingF`). A commitment is its `κ` ring elements in row order, `κd = 1152` words (`Lifecycle.serializeCommitment`).
- A public input is its 320 words (`Lifecycle.serializePublicInput`). A point of `K^ℓ` is its coordinates in order, `2ℓ` words (`Lifecycle.serializePoint`).
- An evaluation family is `Eval_K`, 64 elements of `K`, then `Eval_A` matrix by matrix, each 64 elements of `K`: `2 · 5 · d = 640` words (`Lifecycle.serializeEvaluations`).
- A package value is its nodes in tree order. Each node is four words: the tag (0 for a number, 1 for an array), then the number or the array length as its low 32 bits, its next 32 bits and the rest (`Export.Package.valuePreimage`).

### 4.2 The state hash

The state hash of HyperNova Construction 2 binds the key digests of the two circuits, the step counter, the initial and current states, the program counter and the two slots (`Lifecycle.BoundedDepth.statePreimage`, `Lifecycle.BoundedDepth.stateHash`). A state is four words. Each slot enters as its digest, the Poseidon2 hash of its own domain and its words (`Lifecycle.BoundedDepth.slotDigest`). The slot words are these (`Lifecycle.BoundedDepth.serializeSlot`).

| Part | Words | Content |
|---|---|---|
| domain | 12 | the ASCII bytes of `HyperNova/NIVC/slot/v3`, packed 8 bytes to a word (`Lifecycle.BoundedDepth.slotDomainChunk`) |
| commitment | `κd` | the accumulator's commitment |
| evaluations | `10d` | `Eval_K`, then `Eval_A` |
| point | `2ℓ` | the accumulator's point `r` |
| public input | 320 | the accumulator's public input |
| depth | 1 | the slot's depth |

Without the domain, a slot is 2,169 words (`Lifecycle.BoundedDepth.serializeSlot_length_eq` (`c375f39b01fb69f2`)). With the domain, a slot digest absorbs 2,181 words in 182 blocks (computed). The state preimage is these words.

| Part | Words | Content |
|---|---|---|
| domain | 12 | the ASCII bytes of `HyperNova/NIVC/state/v3`, packed 8 bytes to a word (`Lifecycle.BoundedDepth.stateDomainChunk`) |
| `vk_0`, `vk_1` | 8 | the key digests of the light and the refresh circuit (Algorithm 2) |
| `i` | 1 | the step counter |
| `z0` | 4 | the initial state |
| `zi` | 4 | the current state |
| `pc` | 1 | 0 before the first step, else 1 plus the index of the circuit that ran last |
| slot 0, slot 1 | 8 | the two slot digests |

A well-formed state preimage is 38 words (`Lifecycle.BoundedDepth.statePreimage_length` (`e58324f6be2f3c19`)). Two different well-formed states with one state hash give a Poseidon2 collision on an input pair of one length that the two states fix: their state preimages or the slot preimages of one slot. The sponge pads with zeros and does not encode the length, so a pair of two lengths would not be a collision of the permutation (`Lifecycle.BoundedDepth.stateHash_collision` (`377f6a1064979b8a`)). The slot domain, the state domain and the fold domains of section 5.2 are different blocks (`Lifecycle.BoundedDepth.domainChunks_distinct` (`592150d1f7c4d89d`), `Lifecycle.BoundedDepth.foldDomainChunks_distinct` (`aa05e7488ca3e756`)).

### 4.3 The public input of a fresh claim

A fresh claim's public input encodes a four-word digest in 320 cells (`Lifecycle.encHash`, HyperNova Definition 12). Cell 0 is 1 (`Lifecycle.encHash_marker` (`9b5f3aeed762324f`)). Cell `1 + 64w + k` is bit `k` of digest word `w`, for `w < 4` and `k < 64`. Cells 257 to 319 are 0. The verifier decodes the digest from bits 1 to 256 (`Lifecycle.decodeHash`).

### 4.4 Packages and circuit keys

A step package is the Lean value `Export.BoundedDepth.StepPackage.Package`. It holds, in this order, the circuit index, the public columns, the source width, the first fresh column, the first S-box column, the pieces, the witness batches, the slot runs and the application region (`Export.BoundedDepth.StepPackage.Package.format`). A piece is a list of R1CS rows, a ring product with its quotient columns, or a packed Poseidon2 block (`Export.BoundedDepth.StepPackage.PackagePiece`). The relation's rows are the rows of the pieces, in which each output local of a block is replaced by its linear form over the last 16 S-box columns of the block (`Export.Generic.CircuitRelation.outputForms`, `Export.Generic.Substitute.substitute`). The 16 output locals of a block are thus no columns of the relation. A set of substituted rows holds exactly when the rows hold in the environment that has each output local at the value of its form (`Export.Generic.Substitute.rowsHold_substitute` (`901877a6f62bfcfe`)). After these rows (`Export.BoundedDepth.StepPackage.Package.relationRows`), the relation has 150 S-box rows for each block (`Export/BoundedDepth/Parity.lean`). An input lane of a block reads the forms in the same way. The slot runs give each source column its low-norm encoding, a bit or 41 balanced-ternary digits (`Export/Generic/SlotLayout.lean`). Every format decodes what it encodes, so the encoded package fixes the relation (`Export/BoundedDepth/StepPackage.lean`). The emitter builds the pieces of the light and refresh packages with one definition, and they are the pieces of the step results of section 1 for every commitment key (`Export.BoundedDepth.StepCompleteness.piecesOf_lightOps` (`368629e91e1f03bc`), `Export.BoundedDepth.StepCompleteness.piecesOf_refreshOps` (`08fe21c5980fbaee`)).

The relation is complete. Let the constraints of every piece hold in the honest values and read only columns below the first fresh column, and let the blocks pack. The prover sets the dead locals of the blocks to 0, puts the products of the lowering and the quotient of each ring product by `X^64 + 1` in the fresh columns, and puts the template values of each block in its S-box columns (`Export.Generic.RelationCompleteness.completeEnv`, `Export.Generic.PieceRowsCompleteness.executePieces_spec` (`ac027f44095c9855`)). Each assignment whose layout reads these values on each column that a row reads then makes the rows zero (`Export.Generic.RelationCompleteness.rowsZero_of_holds` (`82920e476fd3b96b`)). The slot runs give a slot only to a column that a row of the relation reads: the columns of the terms of the rows, the lanes, outputs and quotients of each ring piece, the lanes of each block, and each S-box column, but no output local of a block (`Export.BoundedDepth.StepPackage.runsOf`). A column that no row reads, such as a dead local of a block, has no slot and no coordinate. The rows of a piece read only the columns that the piece lists, and these columns are below the first S-box column (`Export.BoundedDepth.SlotRuns.relationRows_reads` (`87b8a843e1e35ef6`), `Export.BoundedDepth.SlotRuns.reads_below` (`3fb3c88e757765be`)). Each such column from the public columns to the last S-box column has a slot (`Export.BoundedDepth.SlotRuns.runsOf_covers` (`020222ab529a2223`)), so the low-norm encoding of the honest values reads them on each column that a row reads (`Export.BoundedDepth.SlotCompleteness.represents_of_runs` (`17820208f6cdb53e`)). Each slot is a field slot, or a bit slot on a bit candidate that a row of a piece forces to 0 or 1 (`Export.BoundedDepth.SlotRuns.runsOf_kind` (`0d1bee7d8f6e68a6`)). The bit candidates are the columns of the `bit` hints, and every live column of a native application (section 7.7). A row forces a column `c` to 0 or 1 when its terms read only `c` and the constant column 0, and with `x` at `c` the row is `(a₀ + a₁x)(b₀ + b₁x) = c₀ + c₁x` with `a₁b₁ ≠ 0` and holds at `x = 0` and at `x = 1` (`Export.Generic.SlotLayout.forcedColumn?`). Such a column holds 0 or 1 wherever the row holds (`Export.Generic.NativeBlock.forced_valid` (`1e261defcef1db33`)). Thus the encoding is in the norm bound when each public column is (`Export.BoundedDepth.SlotCompleteness.norm_of_runs` (`8b6acb1739ee6f75`)). An honest step satisfies the relation of its package at the slot runs, and each coordinate is in the norm bound. A key can have more logical columns than the slots. Then the prover adds zero coordinates, which the widened layout does not read (`Export.BoundedDepth.SlotCompleteness.rowsZero_of_holds` (`43659752fba2a171`), `Export.Generic.WideLayout.sourceEnv_widen` (`db50ef5faa1d29a2`)).

The identity of the light and refresh packages is the Poseidon2 hash of the 31 bytes of `Nightstream/OptionB/packages/v1`, one word each, then the words of the pair (light package, refresh package) (`Export.BoundedDepth.StepPackage.identity`). Lean also computes this hash as a stream over the encoding (`Export.BoundedDepth.StepPackage.identityFast_eq` (`2d9e5280035c3f27`)). The Rust test `identity_code_matches_lean_on_the_probe_packages` compares Rust's identity code with Lean's on two small packages (`crates/nightstream-fprime/tests/bounded_depth_packages.rs`).

These digests are Rust's (not checked against Lean).

1. With a Rust application in place of Lean's, the identity is the Poseidon2 hash of the bytes of `Nightstream/OptionB/application/v1`, one word each, then the Lean identity, the witness word count, the application's rows, its recipe rows, and its shape: the block count, then the first local and the lanes of each block (section 7.7, step 5) (`crates/nightstream-fprime/src/bounded_depth/application.rs`).
2. The key digest of circuit `c` is `vk_c = Poseidon2.hash(domain ++ identity ++ [c])`, where `domain` is the ASCII bytes of `Nightstream/OptionB/circuit-key/v1` packed 8 bytes to a word in 12 words, `c = 0` for the light circuit and `c = 1` for the refresh circuit (`crates/nightstream/src/lifecycle/bounded_depth/mod.rs`). Lean takes the two key digests as part of the setup (`Lifecycle.BoundedDepth.CircuitKey`).

### 4.5 Proof bytes

These layouts are Rust's (not checked against Lean). Every word is 8 bytes, little-endian, and a field word must be below `q`. A proof (`crates/nightstream/src/lifecycle/bounded_depth/encoding.rs`) is these parts.

| Part | Size | Content |
|---|---|---|
| tag | 16 | `NS-OPTIONB-PRF-1` |
| header | 80 | `pc`, `i`, `z0` and `zi` as ten words |
| slot 0, then slot 1 | `8 + 8(κd + 320 + 2ℓ + 10d) + 8d·w_s` each | the depth, then the accumulator's commitment, public projection, point, `Eval_K` and `Eval_A`, then its witness as `d · w_s` field words |
| fresh claim | `8(κd + 320) + 16 w_c` | the commitment, the public input, then for each witness column a 64-bit mask of the `+1` coefficients and one of the `−1` coefficients |

Here `w_s` is the number of ring columns of the relation of circuit `s`, and `c = pc − 1`. A proof with `pc = 0` is the initial proof, and it has only the tag and the header. The decoder checks the exact length that `pc` and the two relations fix before it reads. It rejects a word not below `q`, a malformed witness mask, and an initial proof with `i ≠ 0` or `zi ≠ z0`. An accumulator witness is not in `{−1, 0, 1}`, so its entries are field words.

A final proof (`crates/nightstream/src/lifecycle/bounded_depth/compression.rs`; not checked against Lean) is these parts.

| Part | Size | Content |
|---|---|---|
| tag | 16 | `NS-OPTIONB-FIN-2` |
| header | 80 | `pc`, `i`, `z0` and `zi` as ten words |
| slot 0, then slot 1 | `8 + 8(κd + 320 + 2ℓ + 10d)` each | the depth, then the accumulator's commitment, public projection, point, `Eval_K` and `Eval_A` |
| fresh claim | `8(κd + 2ℓ + 10d) + 32` | the commitment, the point, `Eval_K` and `Eval_A` of the CE claim of the fresh instance, then the digest of the Π_CCS transcript as four words. The bytes do not hold the public input. |
| Π_CCS rounds | `8 · 18ℓ` | the 9 coefficients of each of the `ℓ` round polynomials |
| commitments | `8 + a` and `8 + a'` | the Akita commitment of `W`, then that of `V`, each after its byte count as one word |
| first sum-check | `8 + 64n` | the round count `n`, at most 30, then the 4 coefficients in `K` of each round polynomial |
| evaluations | `16 + 64 + 16` | `W` at the point `ρ` of the first sum-check, then the four columns `V_k` of `V`, then the key's part `K_w` of the public weights at `ρ` (Algorithm 9, step 9) |
| key sum-check | `8 + 48(6 + n_A)` | the round count, at most 30, then the 3 coefficients in `K` of each round polynomial |
| key value | 16 | `Â(ζ, s)`, the key polynomial at the point of the key sum-check and the row point (Algorithm 9, step 10) |
| opening | `8 + o` | the one Akita proof that opens `Â`, `W` and `V`, after its byte count as one word |

A final proof with `pc = 0` is the final proof of zero steps, and it has only the tag and the header. The decoder rejects another tag, `pc > 2`, a field word not below `q`, `pc = 0` with `i ≠ 0` or `zi ≠ z0`, a round count above 30 in either sum-check, short bytes, and a byte after the proof. Akita fixes the bytes of its commitments and opening, and Algorithm 10 rejects a commitment whose bytes are not the encoding of the commitment that they decode to. The parts before the Akita commitments have `96 + 2 · 17,352 + 14,816 + 4,032 = 53,648` bytes. For the reference application at `n = 27` and `n_A = 19`, the parts after the commitments, without the opening, have `1,736 + 96 + 1,208 + 16 + 8 = 3,064` bytes (both computed). `FinalProof::to_bytes` and `FinalProof::from_bytes` are the public byte form (`crates/nightstream/src/circuit.rs`). A compressed memory proof (section 7) has no byte form in the public API.

A compression key (`crates/neo-compress/src/key.rs`; not checked against Lean) is these parts.

| Part | Size | Content |
|---|---|---|
| tag | 16 | `NS-COMPRESS-VK-1` |
| blocks | 8 | the block count `2^(n_A)` of the key polynomial |
| key commitment | `8 + a_A` | the Akita commitment of `Â`, after its byte count as one word |
| setup prefixes | `8 + p` | Akita's commitments of the prefixes of its setup that the schedule rows offload, after their byte count as one word |

For `n_A = 19` the key has `40 + 471 + 548 = 1,059` bytes, as the ignored test `a_compressed_proof_verifies_and_a_changed_one_does_not` prints. `CompressionKey::to_bytes` and `Verifier::decode_compression_key` are the public byte form (`crates/nightstream/src/circuit.rs`). The decoder accepts exactly the bytes of the pinned key of the verifier's block count (Algorithm 11), and it rejects every other byte string.

`Circuit::write` saves the Rust application, and its file starts with the 8 bytes `NSOPTB`, 0, 1, or `NSOPTM`, 0, 1 for a memory circuit.

## 5. Transcript

### 5.1 The permutation and the hash

The hash is Poseidon2 over `F_q` with width 16 and the S-box `x^7` (`Spec.Poseidon2.permute`). It has 4 full rounds, 22 partial rounds and 4 full rounds. The round constants and the internal diagonal are those of the instance that `scripts/poseidon2/derive.py` derives for `q` (`Spec.Poseidon2.initialConstants`, `Spec.Poseidon2.internalDiagonal`, `decisions/poseidon2-p64-instance.md`). The external matrix and the order of the layers are those of Plonky3. The state is 16 words, and the rate is 12.

- To absorb a block, add its words to lanes 0 to 11, then permute (`Spec.Poseidon2.absorbBlock`).
- `Poseidon2.hash(words)` starts from the zero state, absorbs the words in blocks of 12, the last one short, then adds 1 to lane 0, permutes and outputs lanes 0 to 3 (`Spec.Poseidon2.hash`). It does not bind the length.

### 5.2 The fold transcripts

Each fold verifier and each step circuit runs one Poseidon2 sponge for the fold (`Lifecycle.Transcript`). To absorb a list, add it in blocks of 12, each followed by one permutation (`Lifecycle.Transcript.absorb`). The profile fixes every absorbed length, so the stream has one parse without framing (`decisions/piccs-prior-state-digest.md`).

A challenge in `K` at position `p` reads lanes `2(p mod 6)` and `2(p mod 6) + 1` as `c_0, c_1`. After the read at a position with `p mod 6 = 5`, the sponge absorbs a zero block (`Lifecycle.Transcript.squeezeAt`). Coordinate `j` of `α` has position `j`, `γ` has position `ℓ`, and every round challenge has position 0 (`Lifecycle.BoundedDepth.lightCoinPosition`, `Lifecycle.Transcript.coinPosition`). `S` is the number of sources, 2 for a light fold and 17 for a refresh fold.

| # | Operation | Words or reads | Value |
|---|---|---|---|
| 1 | absorb | 12 | the ASCII bytes of `Nightstream/SuperNeo/fold/light/v1` or `Nightstream/SuperNeo/fold/refresh/v1`, packed 8 bytes to a word (`Lifecycle.BoundedDepth.lightFoldDomainChunk`, `Lifecycle.BoundedDepth.refreshFoldDomainChunk`) |
| 2 | absorb | `4 + κd + 320`, plus `16(κd + 10d)` for a refresh fold | the prior digest decoded from the fresh public input, the fresh commitment and the fresh public input (`Lifecycle.ProductionKey.publicInputBlocks`). A refresh fold adds each Π_DEC child's commitment and evaluation family, child 0 first, in the same list (`Lifecycle.BoundedDepth.refreshWords`). |
| 3 | read | `ℓ` | `α_0, …, α_{ℓ−1}` at positions 0 to `ℓ − 1` |
| 4 | read | 1 | `γ` at position `ℓ` |
| 5 | for each round `j < ℓ`: absorb | 18 | the 9 coefficients of round polynomial `j` (`Lifecycle.Transcript.serializeMessage`) |
| 6 | then read | 1 | the round challenge `r'_j` at position 0 |
| 7 | absorb | `S · 10d` | the Π_CCS output, source by source (`Lifecycle.BoundedDepth.lightOutputWords`, `Lifecycle.ProductionKey.absorbFullOutput`) |
| 8 | Π_RLC | 5 lanes for each challenge | `ρ` for a light fold, `ρ_0, …, ρ_16` for a refresh fold (section 5.3) |

The accumulator does not enter the sponge directly. It enters through the prior digest in the fresh public input, which is the state hash (Algorithm 3). The refresh children are prover messages, so the sponge absorbs them. Their point and public inputs come from the accumulator. Equal absorbed words give equal children's commitments and evaluation families (`Lifecycle.BoundedDepth.refreshWords_children` (`f3be9e10b5609a92`)). The Rust prover derives these transcripts in `bind_and_sample_with_trace`, `prove_phase` and `absorb_outputs` (`crates/neo-reductions/src/engines/pi_ccs_joint_protocol.rs`). `derive_light_transcript` replays the light transcript for the base step, and only tests call `derive_refresh_transcript` (`crates/nightstream-fprime/src/package/pi_ccs_v1_2_transcript.rs`).

### 5.3 The Π_RLC sampler

Let `S_0` be the sponge state after row 7. For challenge `s`, absorb the block `[4, s]` into `S_s` to get `E_s` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter`). Read lanes 0 to 4 of `E_s` as `x_0, …, x_4`, and set `S_{s+1} = permute(E_s)` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt`).

1. Set `N = Σ_{k<5} x_k · q^k`, with lane 0 the least significant (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.drawIndex`), and `R = N mod 5^64` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.reduce`).
2. Coefficient `h` of `ρ_s` is `⌊R / 5^h⌋ mod 5`, minus 2 (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample`, `Spec.RingStrongSet.embedScalar`).

A refresh fold takes `ρ_0, …, ρ_16` (`Lifecycle.ProductionKey.piRlcResponse`). A light fold takes only `ρ_0` as its `ρ` (`Lifecycle.BoundedDepth.lightChallenge`). The sampler cannot fail (`Lifecycle.ProductionKey.piRlcResponse_ne_none` (`dde77b467bc3c6f1`)), and each challenge is in `C` (`Lifecycle.Transcript.PiRlcSampler.piRlcChallenges_member` (`f5cf1d7bdc81b6a0`), `Lifecycle.BoundedDepth.lightChallenge_valid` (`3fae52d3cdb1e617`)).

### 5.4 The compression transcript

Algorithms 9 and 10 run one Poseidon2 sponge of section 5.2 for the Π_CCS run of the fresh instance and then for the argument (not checked against Lean). The reads since the last absorb take the positions 0, 1, and so on, and the sponge absorbs a zero block after each read at position 5. Bytes become words in this way: the byte count, then one word for each 7 bytes, little-endian, the last one short (`byte_words` in `crates/neo-compress/src/transcript.rs`). Each such word is below `2^56`.

| # | Operation | Words or reads | Value |
|---|---|---|---|
| 1 | absorb | 12 | the ASCII bytes of `Nightstream/OptionB/compress/ccs/v1`, packed 8 bytes to a word |
| 2 | absorb | `4 + κd + 320`, then the words of the bytes | the prior digest decoded from the fresh public input, the fresh commitment, the fresh public input, then the Akita commitment of `W` as words, in one list |
| 3 | rows 3 to 7 of section 5.2 with `S = 1` | | `α`, `γ`, the `ℓ` rounds with `r'`, then the Π_CCS output of the fresh source, `10d` words. Lanes 0 to 3 of the state after this absorb are the digest of the Π_CCS transcript. |
| 4 | absorb | 12 | the ASCII bytes of `Nightstream/compress/akita/v2`, packed 8 bytes to a word |
| 5 | absorb | the words of the bytes | the bytes of the compression key (section 4.5) |
| 6 | for each claim: absorb | `5 + κd + 320 + 2ℓ + 10d` | the ring columns `w`, the table bound plus 1 or 0 for no table, 320, `ℓ` and 4. Then the commitment, the public input with cell `64j + h` at position `5h + j`, the point, `Eval_K` and `Eval_A`. |
| 7 | absorb | the words of the bytes | the Akita commitment of `W` |
| 8 | read | `⌈L/2⌉ = 2` | `τ_0, …, τ_3` are the coordinates of these reads, `c_0` before `c_1` |
| 9 | absorb | the words of the bytes | the Akita commitment of `V` |
| 10 | read | `4 + L + 2 + 2 + v` | `ω`, `δ`, `γ`, `μ`, then `ψ_0, …, ψ_{L−1}`, then `ξ_0, ξ_1`, then `λ_0, λ_1`, then `β_0, …, β_{v−1}` |
| 11 | for each round `j < n`: absorb | 8 | the 4 coefficients of round polynomial `j`, from degree 0 up |
| 12 | then read | 1 | the round challenge `ρ_j` |
| 13 | absorb | 2, then 8, then 2 | `W(ρ)`, then the four values `V_k`, then `K_w` |
| 14 | read | 2 | the column point `(σ_0, σ_1)` |
| 15 | for each round `j < 6 + n_A`: absorb | 6 | the 3 coefficients of round polynomial `j` of the key sum-check, from degree 0 up |
| 16 | then read | 1 | the round challenge `ζ_j` |
| 17 | absorb | 2 | the key value `Â(ζ, s)` |

Here `v` is the number of variables of the slot slices (Algorithm 9, step 3). The symbols of rows 6 to 17 are those of the argument: `γ` of row 10 is not `γ` of the Π_CCS run, and `ρ_j` is not a Π_RLC challenge. The row point `s` is a function of `ω` and takes no read (Algorithm 9, step 10). The session bytes of the Akita opening are the ASCII bytes of `Nightstream/compress/akita/v2/opening`, then lanes 0 to 3 of the sponge state after row 17, each as 8 bytes, little-endian. Akita makes its own challenges from these session bytes, the three commitments and the three claims, with its own Keccak sponge.

## 6. Algorithms

### Algorithm 1. Commitment-key setup

**Input.** A 32-byte seed, `κ` and the number of key columns `w`.

**Output.** The Ajtai key `A ∈ R_q^{κ × w}`.

**Steps.**

1. For each row `i < κ` and column `j < w`, let `in` be the 37 ASCII bytes of `nightstream-ajtai-shake128-wide256-v1`, then the seed, then `i` as 4 bytes and `j` as 8 bytes, little-endian (`Spec.AjtaiSetupV1.elementInput`).
2. Read the first `4d` 64-bit lanes of SHAKE128 of `in` (`Spec.AjtaiSetupV1.elementLanes`).
3. Set coefficient `h` of `A_{i,j}` to the 256-bit little-endian value of lanes `4h` to `4h + 3`, reduced modulo `q` (`Spec.AjtaiSetupV1.wideCoefficientNat`).

**Rejects.** None.

**Source.** `Spec.AjtaiSetupV1.Setup.verifierKey`. Each coefficient is one SHAKE128 chunk reduced modulo `q` (`Spec.AjtaiSetupV1.Programming.verifierKey_eq` (`d59bd57434c712fc`)). The production seed is `fc404984d44c1b878d68a6a80092d7d7ab44d81ac17b45a8e7bd4c1f1e371702` (`Export.Stage1.Poseidon2HashChainV1SetupAuthority.productionSeed`). Both circuits commit with the first `w_c` columns of the production key, where `w_c` is the number of ring columns of circuit `c` (`commit_production_signed_unit_prefix_matrix` in `crates/neo-ajtai/src/nightstream_fprime_setup.rs`; not checked against Lean). Records `decisions/fprime-stage1-main-ajtai-setup.md` and `decisions/fprime-ajtai-shake128-setup.md`.

### Algorithm 2. Packages and circuit keys

**Input.** The light and refresh packages, their pinned identity, and optionally a Rust application.

**Output.** The identity and the key digests `vk_0` and `vk_1`, or reject.

**Steps.**

1. Decode each package with `Package.format`. Require the circuit index 0 for the light package and 1 for the refresh package.
2. Compute the identity of the light and refresh packages (section 4.4). Require that it equals its pin. For the copies that the crate embeds, the test `embedded_packages_have_their_pinned_identities` does this step once, and at run time the identity is its pin, because the bytes are part of the binary (`crates/nightstream/src/artifacts.rs`). The embedded packages are decoded once per process.
3. With a Rust application, put its rows and recipes in place of the application region of both step packages and compute the new identity (section 4.4). The reference hash chain keeps both step packages and their pinned identity, because it computes the step of Lean's application region (`the_reference_hash_chain_keeps_lean_packages` in `crates/nightstream/tests/bounded_depth_api.rs`).
4. Compute `vk_0` and `vk_1` from the identity (section 4.4).

**Rejects.** A package that does not decode, a wrong circuit index, an identity that is not its pin, and an application piece that reads past its region.

**Source.** `load_bounded_depth_packages`, `load_pinned_bounded_depth_packages` and `BoundedDepthPackages::with_application` (`crates/nightstream-fprime/src/bounded_depth.rs`, `crates/nightstream-fprime/src/bounded_depth/application.rs`), `Circuit::compile` and `Circuit::load_packages` (`crates/nightstream/src/circuit.rs`), and `BoundedDepth::new` (`crates/nightstream/src/lifecycle/bounded_depth/mod.rs`); not checked against Lean. Lean defines the identity of the light and refresh packages (`Export.BoundedDepth.StepPackage.identity`). Every state hash carries `vk_0` and `vk_1` (section 4.2).

### Algorithm 3. The step relation

Each step proves one augmented step of HyperNova Construction 2 for the circuit that runs, with two slots and the depth cap (`Lifecycle.BoundedDepth.StepHolds`, `Spec.HyperNova.Construction2.BoundedDepth.verifier`).

**Input.** The circuit `c`, 0 for light and 1 for refresh, the step counter `i`, the states `z0` and `zi`, the program counter `pc`, the two slots, a fresh claim `u`, a fold proof of the kind of `c`, and the application witness `w`.

**Output.** The next state `z_{i+1}`, the next slots, the next program counter `c + 1`, and the digest `x` that the next fresh claim's public input encodes.

**Steps.**

1. Require that the circuit selector gives `c` at `(zi, w)`. The bounded-depth scheme takes the selector that returns the circuit that runs on a valid step and the other circuit otherwise (`Export.BoundedDepth.TerminalBridge.chosen`), so the prover picks `c` and the step's witness is valid. Set `z_{i+1} = F(zi, w)`, where `F` is the application's step (`Lifecycle.Stage1.Application.Program`). Both circuits run the same `F`, so the choice of `c` does not change `z_{i+1}`.
2. If `i = 0`: require `z0 = zi`, and set both next slots to the default slot. The relation requires no fold at `i = 0`, but the rows of a step circuit do: at each step, also at `i = 0`, they check the prior state hash in the public input of `u` and a fold into the selected slot, and the digest select then puts the default slot in place of the fold's output (`Lifecycle.BoundedDepth.LightFold.spec_implies_fold` (`88e3e1e086908dcf`), `Lifecycle.BoundedDepth.DigestSelect`). So the base step runs the light circuit on the base fold. Its prior state is `(vk, 0, z0, z0, pc = 0, default slots)`. Its fresh claim `u` has a zero commitment and the public input `encHash` of that state's hash. Its proof is the zero proof, in which each round polynomial and each Π_CCS output is zero. The light fold accepts this proof into the default slot (`Lifecycle.BoundedDepth.BaseFold.baseFresh`, `Lifecycle.BoundedDepth.BaseFold.zeroProof`, `Lifecycle.BoundedDepth.BaseFold.zero_folds` (`b82d57e7f4f34804`)).
3. If `i > 0`: require `pc ∈ {1, 2}`, and let `s = pc − 1` be the selected slot. Require that the public input of `u` is `encHash(stateHash(vk, i, z0, zi, pc, slots))`. If `c` is the light circuit, require that the depth of slot `s` is below `cap`. Require that the fold of the kind of `c` (Algorithm 4) accepts the accumulator of slot `s`, `u` and the proof, with the relation of circuit `s`. Set slot `s` to the fold's output, at depth `d + 1` after a light fold and at depth 0 after a refresh fold. The other slot does not change.
4. Set `x = stateHash(vk, i + 1, z0, z_{i+1}, c + 1, next slots)`.

**Rejects.** The relation does not hold.

**Source.** `Lifecycle.BoundedDepth.stepHolds_iff` (`8187124fa8b66a81`) states these steps. The light circuit places the application, the depth guard, the active slot's hash, the prior state hash over the two prior digest words, the next state hash, the light Π_CCS and Π_RLC on the selected slot, the hash of the new slot, and the digest select (`Lifecycle.BoundedDepth.LightStepLayout`). It does not read the other slot's words: the selected slot's prior digest is the active slot's digest, and the other slot enters as its prior digest word. The digest select picks each next slot's digest from the default slot's digest, the new slot's digest and the prior digest of that slot (`Lifecycle.BoundedDepth.LightSelect`, HyperNova §6.3). So a light step hashes two slots, the active slot and the new slot. The refresh circuit hashes both prior slots and the new slot (`Lifecycle.BoundedDepth.DigestSelect`). The depth guard decomposes the depth word into 64 bits and requires bits 6 to 63 to be zero, so the depth is below 64 (`Lifecycle.BoundedDepth.DepthGuard`). The refresh circuit has no depth guard. It places Π_DEC on the selected slot before the refresh Π_CCS and Π_RLC, and the new slot's depth word is 0 (`Lifecycle.BoundedDepth.RefreshStepLayout`). The rows of either circuit give its case of the step relation (`Layout.BoundedDepth.StepCircuits.soundness` (`11098d832080542b`)). The circuits do not compute the selector. Their rows give the application's validity predicate, and each soundness theorem takes as a premise that every valid step of the circuit's application has the selector value `c`. The selector `chosen c` meets this premise for every application. For the reference application, `F(z, w) = Poseidon2.hash(tag ++ z ++ w)`, where `tag` is the 40 bytes of `Nightstream/Stage1/Poseidon2HashChain/v1`, one word each (`decisions/fprime-stage1-poseidon2-hash-chain-v1.md`). HyperNova Construction 2, steps 4 and 5.

### Algorithm 4. Fold verifiers

A fold combines the previous step's fresh claim with one accumulator. The light fold runs Π_CCS on two sources and Π_RLC with the weights `ρ` and 1 (Cyclo, ePrint 2026/359, contribution 2, on SuperNeo v1.2 Sections 7.3 and 7.4). The refresh fold runs Π_DEC on the accumulator first, then Π_CCS on 17 sources and Π_RLC (SuperNeo v1.2 Sections 7.3 to 7.5). Neither fold ends with Π_DEC.

**Input.** The kind, light or refresh, the accumulator `U = (r, c_U, x_U, y_U)`, the fresh claim `(c, x)`, and a proof of that kind.

**Output.** The next accumulator, or reject.

**Steps.**

1. For a refresh fold, read the 16 children's commitments `c_k` and families `y_k` from the proof. Split `x_U` into 16 digit vectors `x_0, …, x_15` with `x_U = Σ_k 2^k x_k`, and reject if it does not split. Require `c_U = Σ_k 2^k c_k` and `y_U = Σ_k 2^k y_k` (`Spec.Folding.Refresh.inputCheck`, SuperNeo v1.2 Section 7.5). The running sources are the children `(r, c_k, x_k, y_k)`. For a light fold, the running source is `U`.
2. Order the sources fresh first, then the running sources. Let `R = S − 1` be the number of running sources. Run rows 1 to 6 of section 5.2 to get `α`, `γ` and `r' = (r'_0, …, r'_{ℓ−1})`.
3. Compute the initial claim `c_0 = Σ γ^{s + R h} · Eval_K(y_s)_h + Σ γ^{Rd + s + R j + 4R h} · Eval_A(y_s)_{j,h}`, over the running sources `s < R`, `j < 4` and `h < 64` (`Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial.VerifierInput.initial`, `Spec.Folding.PiCCS.PaperJoint.PadCoordinate.gammaExponent`, `Spec.Folding.PiCCS.PaperJoint.MatrixCoordinate.gammaExponent`).
4. For each round `j`, require `g_j(0) + g_j(1) = c_j`, and set `c_{j+1} = g_j(r'_j)` (`Spec.SumCheck.Finite.FixedPhase.checkChain`).
5. Require `c_ℓ = V`, where `V` is computed from the Π_CCS output at `r'` (`Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial.terminalFromMessage`):

   ```text
   V = eq(r', r) · Σ_coord γ^e · Pad_coord
     + γ^(Rd) · eq(r', r) · Σ_coord γ^e · Mat_coord
     + γ^(5Rd) · eq(r', α) · ( f(M_0 z_0, …, M_3 z_0) + γ · Σ_{s<S} γ^s · N(z_s) )
   N(y) = (y + 1) · y · (y − 1)
   ```

   Here `Pad_coord`, `Mat_coord`, `M_j z_0` (the fresh source) and `z_s` are the values that the output claims at `r'`, and `e` is the local exponent of each coordinate (`Spec.Folding.PiCCS.PaperJoint.StrongReduction.Statement.projectOutput`). `Rd` is 1024 and `5Rd` is 5120 for a refresh fold (`Spec.Folding.PiCCS.PaperJoint.Shape.constraintOffset_eq` (`395ce359b5e699ad`)), and they are 64 and 320 for a light fold (`Lifecycle.BoundedDepth.LightPiCCS.light_exponents` (`edeefee288db608f`)). The light Π_CCS puts the running source into the norm batch as zero, so only the fresh source has a norm check (`Spec.Folding.Light.PiCCS.normAssignments`).
6. Absorb the Π_CCS output (row 7), and derive the challenges (section 5.3).
7. For a light fold, the next accumulator is `(r', c_U + ρ·c, x_U + ρ·x, y'_U + ρ·y'_0)`, where `y'_0` and `y'_U` are the families of the fresh source and of the accumulator in the Π_CCS output (`Spec.Folding.Light.Rlc.output_commitment` (`27c897818b0c19c0`), `Spec.Folding.Light.Rlc.output_publicInput` (`5872386765650c7b`), `Spec.Folding.Light.Rlc.output_evaluation` (`649ea10845fa2950`)). For a refresh fold, it is `(r', Σ_s ρ_s c_s, Σ_s ρ_s x_s, Σ_s ρ_s y'_s)` over the 17 sources, where `c_s` and `x_s` are the source's commitment and public input and `y'_s` is its family in the Π_CCS output (`Spec.Folding.PiRLC.combinedOutput`, SuperNeo v1.2 Section 7.4).

**Rejects.** The verifier rejects when step 1, 4 or 5 fails (`Spec.Folding.Light.verify`, `Spec.Folding.Refresh.verify_eq_some_iff` (`a750a1a74be4db7f`)).

**Source.** `Spec.Folding.Light.verify` and `Spec.Folding.Refresh.verify` at the production keys `Lifecycle.BoundedDepth.keys`. The two keys share one relation, one commitment key and one Π_RLC algebra, and they differ in the transcript (`Lifecycle.BoundedDepth.keys_coherent` (`6ffc49b4807094c5`), `Lifecycle.BoundedDepth.refreshKey_relation` (`946385746d5f3a4a`)). The light verifier rejects a refresh proof, and the refresh verifier rejects a light proof (`Spec.HyperNova.Construction2.BoundedDepthFolds.light_rejects_refresh_proof` (`8b622ee42b82757c`), `Spec.HyperNova.Construction2.BoundedDepthFolds.refresh_rejects_light_proof` (`837d0902b6a5ce8a`)). Rust folds in `crates/nightstream/src/folding/bounded_depth.rs` (not checked against Lean). It has the prover of a fold and no verifier of a fold proof. The rows of the next step circuit check each fold (Algorithm 3), and a fold proof is not in the proof bytes (section 4.5). Record `decisions/padded-row-identity-piccs.md`.

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

**Source.** `BoundedDepth::initial` and `BoundedDepth::extend` (`crates/nightstream/src/lifecycle/bounded_depth/mod.rs`) and `crates/nightstream/src/folding/bounded_depth.rs`; not checked against Lean. `Prover::extend` runs the Rust application to get `z_{i+1}`, and `Prover::extend_with_output` takes it from the caller (`crates/nightstream/src/circuit.rs`). Lean does not model this scheduler. The light verifier rejects a light fold at the cap, so the folds of each slot form an admissible schedule (`Spec.HyperNova.Construction2.BoundedDepth.admissible_of_accepted` (`95df2aa1a548b830`)). A run that the cap admits has at most `cap · r + 2 · cap` light steps, where `r` is its number of refresh steps (`Spec.HyperNova.Construction2.BoundedDepth.light_count_le` (`6b3396f5f261095d`)).

### Algorithm 6. Terminal verifier

**Input.** The expected state `(i, z0, zi)` and a proof.

**Output.** Accept or reject.

**Steps.**

1. Require that the proof's state is the expected state and that `i < q`.
2. If the proof is the initial proof, accept exactly when `i = 0` and `zi = z0`.
3. Require `i > 0` and `pc ∈ {1, 2}`.
4. For each slot, require that its depth `j` is at most `cap` and that its witness opens its accumulator. The witness gives the commitment and the public projection, has norm at most `accumulatorBound j` (section 3), is zero on every ring block from the width of its circuit's package, and gives the `Eval_K` and `Eval_A` values at the accumulator's point.
5. Require that the public input of the fresh claim is `encHash(stateHash(vk, i, z0, zi, pc, slots))`.
6. Require that the fresh witness satisfies the relation of circuit `pc − 1`, opens the fresh commitment, projects to the fresh public input and is zero on every ring block from the width of that circuit's package.

**Rejects.** Each step's failure. In step 4, Rust compares the depth with `cap` and each coefficient of the witness with the bound of the depth, and then runs Π_DEC on the witness. Its recomposition must give the claim's commitment, evaluations and public input. The proof bytes hold only the blocks of the package, so a decoded slot witness is zero on the other blocks. Rust checks step 6 in this order: the claim shape, the witness shape, a zero tail past the relation's width, the public projection, the commitment, then every row (`crates/nightstream/src/lifecycle/bounded_depth/mod.rs`, `crates/nightstream/src/lifecycle/bounded_depth/relation.rs`; not checked against Lean).

**Source.** `Export.BoundedDepth.RunCompleteness.Verifies`: the terminal relation `Lifecycle.BoundedDepth.Terminal.HoldsFor`, with the slot checks of `Export.BoundedDepth.StepExtend.Honest` for a recursive proof. The terminal relation alone has each slot witness below `B`. It is the relation that extraction reaches, because a refresh fold recovers its input below `B` (section 3), and each security result is stated on it. The verifier accepts only what the terminal relation accepts (`Export.BoundedDepth.RunCompleteness.holds_of_verifies` (`16655a19ab01f5ae`)). The bound of the depth is the accumulator relation of Cyclo, ePrint 2026/359, Theorem 3 item 1. The terminal relation is the terminal check of HyperNova Construction 2 with two slots (`Spec.HyperNova.Construction2.Paper.OuterTerminalTransition`, `Lifecycle.BoundedDepth.Terminal.holdsFor_recursive_iff` (`41398ed6e45d8ba3`)). The initial proof is accepted exactly at `i = 0` and `zi = z0` (`Lifecycle.BoundedDepth.Terminal.holdsFor_bottom_iff` (`274d87719d362182`)). A counter that is larger by `q` is rejected (`Lifecycle.BoundedDepth.Terminal.rejects_counter_shift` (`6a39fa280214304b`)). With the exported relations as the circuit keys, acceptance gives the decoded last step or a collision of the decoded next state with the terminal's state (`Export.BoundedDepth.TerminalBridge.bridge` (`91114bf46368f23e`)).

### Algorithm 9. Compress

A final proof states that some proof exists that Algorithm 6 accepts for its statement. It has one Π_CCS run and one argument over the Akita commitment scheme (ePrint 2026/1983) in place of the three witnesses. The verifier reads no element of the Ajtai key. It reads the commitment of the Ajtai key in the compression key (Algorithm 11). Lean has no statement for Algorithms 9, 10 and 11 (`decisions/akita-compression.md`).

**Input.** A proof of Algorithm 5 at the state `(i, z0, zi)`, the prover's circuits and security minimum, and the prover's compression setup of Algorithm 11.

**Output.** A final proof, or an error.

**Steps.**

1. For the initial proof, the final proof is the statement alone. Else require `pc ∈ {1, 2}`.
2. The claims are, in this order, the accumulator of slot 0, the accumulator of slot 1, and the fresh instance of circuit `pc − 1`. Slot `s` has the lookup table `[−bound_s, bound_s]` with `bound_s = accumulatorBound(depth_s)` (section 3). The fresh instance has no table.
3. Lay out the polynomial `W`. A claim with `w` ring columns gets a witness slice of `2^(⌈log₂ w⌉ + 6)` cells, and coefficient `h` of ring column `j` is cell `64j + h` of the slice. Place the slices of the two slots first, the larger first and slot 0 first when they are equal, then the slice of the fresh instance. Then place for slot 0, and then for slot 1, a count slice of `2^⌈log₂(2bound_s + 1)⌉` cells. Each slice starts at the next multiple of its size. Let `2^v` be the least power of two that holds the two slot slices. Take the first row, in increasing order, of the schedule file of the compression key (section 3) whose `W` count `n` has `2^n` cells that hold every slice and whose `V` count is at least `v + 2`. `W` has `n` variables and `V` has the `V` count of that row. Require that each claim has at most `2^(n_A)` ring columns and that the error sum of section 3 is within the security minimum.
4. Fill `W`. Each witness is in its slice. Cell `t` of the count slice of slot `s` is the number of cells of the witness slice of slot `s` that hold `t − bound_s`. Each other cell is 0. Commit `W` with Akita. Bit `i` of a cell index goes with variable `i`.
5. Run Π_CCS on the fresh claim alone: steps 2 to 6 of Algorithm 4 with `S = 1` and no running source, under rows 1 to 3 of section 5.4. The initial claim is 0, and only the last term of `V` of Algorithm 4 is left, with `R = 0`. The output is the CE claim of the fresh instance: its commitment, its public input, the point `r'`, the family of the fresh source, and the digest of the Π_CCS transcript.
6. Run rows 4 to 8 of section 5.4. If a `τ_k` is a value of a table, return an error. Fill the polynomial `V`: for each cell `x` of a slot's witness slice and each `k < L`, cell `k · 2^v + x` is `1/(τ_k − W(x))`. Each other cell is 0. Commit `V` with Akita.
7. Run rows 9 and 10 of section 5.4. For claim `c` with the commitment rows `c_i`, the point `r = (r_lo, r_hi)` with 6 coordinates in `r_lo`, and the ring columns `z_j` of its witness, a CE claim holds these equations in `K[X]/(X^64 + 1)`:

   ```text
   Σ_j A_ij · z_j = c_i                                  for each i < κ
   conj(e) · Σ_j eq(r_hi, j) · z_j = Eval_K              e = Σ_h eq(r_lo, h) · X^h
   Σ_j conj(w_mj) · z_j = Eval_A_m                       for each m < 4
   ```

   `conj` maps `X` to `X^(−1)`, and `w_mj` is ring column `j` of `M_mᵀ · eq(r, ·)`. Batch them with `ω`: `P_cj = Σ_i ω^i A_ij + ω^κ eq(r_hi, j) conj(e) + Σ_m ω^(κ+1+m) conj(w_mj)` and `Y_c = Σ_i ω^i c_i + ω^κ Eval_K + Σ_m ω^(κ+1+m) Eval_A_m`. The value of a ring element at `δ` is the value of its polynomial of degree below 64. The public weights `G` are these. On coefficient `h` of ring column `j` of claim `c`, `G = μ^c · (X^h · P_cj)(δ)`, plus `μ^c · ω^(κ+5) · γ^p` when `p = 64j + h < 320`. On cell `t` of the count slice of slot `s`, `G = −λ_s · Σ_k ψ_k/(τ_k − (t − bound_s))`. `G` is 0 on each other cell.
8. Run the first sum-check, rows 11 and 12 of section 5.4, over the `2^n` cells `x`, with round polynomials of degree 3, for

   ```text
   Σ_x W(x) · (G(x) − E(x) · A(x)) + E(x) · B(x) + S(x) · A(x) = Y
   Y = Σ_c μ^c · (Y_c(δ) + ω^(κ+5) · Σ_{p<320} γ^p · x_cp) + (Σ_k ψ_k) · (ξ_0 + ξ_1)
   A(x) = Σ_k ψ_k · I_k(x)        B(x) = Σ_k ψ_k · τ_k · I_k(x)
   ```

   `I_k(x)` is cell `k · 2^v + x` of `V` for `x < 2^v`, and 0 for each other `x`. On cell `y` of the witness slice of slot `s`, which has `v_s` variables, `E = ξ_s · eq((β_0, …, β_{v_s − 1}), y)` and `S = λ_s`. `E` and `S` are 0 on each other cell. `x_cp` is cell `p` of the public input of claim `c`.
9. The key's part of `G(ρ)` is the part that the `κ` rows of `A` give. Let `E = Σ_h eq((ρ_0, …, ρ_5), h) · X^h`, and let `u ∈ K^64` be the lane weights with `Σ_h u_h · P_h = (E · P)(δ)` for each ring element `P`. Let `F_j = Σ_{i<κ} ω^i · A_ij` for each block `j < 2^(n_A)`. For claim `c` with `w_c` ring columns and a witness slice of `v_c` variables, let `sel_c` be the multilinear polynomial at `ρ` that is 1 on its slice and 0 on each other slice of its size, and `π_c = (ρ_6, …, ρ_{v_c − 1})`. Then

   ```text
   K_w = Σ_c μ^c · sel_c · Σ_{j < w_c} eq(π_c, j) · Σ_h u_h · F_{j,h}
   ```

   Run rows 13 and 14 of section 5.4 with `W(ρ)`, the four values `V_k = Σ_{x < 2^v} eq((ρ_0, …, ρ_{v−1}), x) · V(k · 2^v + x)` and `K_w`.
10. Run the key sum-check, rows 15 and 16 of section 5.4, over the `2^(6 + n_A)` cells `(h, j)`, lane first, with round polynomials of degree 2, for

    ```text
    Σ_{h, j} F_{j,h} · H(h, j) = K_w        H(h, j) = u_h · Σ_c μ^c · sel_c · eq(π_c, j) · [j < w_c]
    ```

    It ends at the point `ζ` with `6 + n_A` coordinates, where it needs the value `F̃(ζ)`. The row point of `ω` is `s ∈ K^5` with `s_k = ω^(2^k)/(1 + ω^(2^k))`, and `scale = Π_{k<5} (1 + ω^(2^k))`. Then `ω^i = scale · eq(s, i)` for each `i < 32`, and the rows of `Â` from `κ` are zero, so `F̃(ζ) = scale · Â(ζ, s)`. If a factor of `scale` is 0, return an error. Run row 17 of section 5.4 with `Â(ζ, s)`.
11. Make one Akita proof, under the session bytes of section 5.4, with the schedule row of step 3. It opens `Â` at `(ζ, s)` to `Â(ζ, s)` against the commitment of the compression key, `W` at `ρ` to `W(ρ)`, and `V` at `(ρ_0, …, ρ_{v−1}, σ_0, σ_1, 0, …, 0)` to `Σ_{k<4} eq((σ_0, σ_1), k) · V_k`. An element `a + b·u` of `K` goes to Akita as `a + (b·c)·u'`, where `u'² = 2` in Akita's extension (section 3).
12. The final proof is the state, `pc`, each slot's accumulator and depth, the CE claim of step 5 without its public input, the Π_CCS rounds, both Akita commitments, the rounds of step 8, the values of step 9, the rounds of step 10, `Â(ζ, s)` and the Akita proof (section 4.5).

**Rejects.** `compress` returns an error for a `pc` outside `{1, 2}` in a proof with steps, for a table of more than 65,535 values, for a claim with more ring columns than the key polynomial holds, for cells that no schedule row holds, for an error sum above the security minimum, when a lookup challenge is a table value, and when `ω` has no row point. The probability of a table value is at most `L · (T_0 + T_1)/q`, about `2^−46.7` (computed). `ω` has no row point when `ω^(2^k) = −1` for a `k < 5`. The 2-power roots of unity of `K` form a group of order 8, because `q² − 1` is divisible by 8 and not by 16, so 7 values of `ω` have no row point, and the probability is `7/q²` (computed). The proof is deterministic, so a new call at the same state fails again. The prover can extend one step and then compress. `compress` does not run Algorithm 6 on its input, and it does not run Algorithm 10 on its output.

**Source.** `Prover::compress` (`crates/nightstream/src/circuit.rs`), `BoundedDepth::compress` (`crates/nightstream/src/lifecycle/bounded_depth/compression.rs`) and `crates/neo-compress/src/`: `layout.rs` for step 3, `prover.rs` for steps 4, 6, 8, 9, 10 and 11, `weights.rs` and `ring.rs` for steps 7 and 9, `key.rs` for step 10, `sumcheck.rs` for both sum-checks, `lookup.rs` for the tables, and `akita.rs` for the Akita calls. This algorithm is not checked against Lean. No Lean statement exists for the layout, the argument, the key sum-check, or a Π_CCS run with one source. Lean defines the relations of the three claims. A slot claim is `Lifecycle.BoundedDepth.SlotClaim` under `Spec.CE.Holds` with the norm `Spec.normBounded` at `accumulatorBound(depth) + 1` (`Spec.GlobalParams.accumulatorBound`, `Export.BoundedDepth.StepExtend.Honest`), and the fresh instance is under `Spec.CCS.Holds`. The argument commits only the ring columns of each circuit's package. It does not prove that the fresh witness is zero from the relation's width to the end of its last ring column, which Algorithm 6 checks in Rust and `Lifecycle.BoundedDepth.BlockSupport.Supported` does not require. Akita is a git dependency at revision `d50f700fb1a292395a717cef0e6f437c7ce2cdb1` (`crates/neo-compress/Cargo.toml`). `scripts/akita_schedules.sh` makes the schedule files with Akita's planner at that revision, family `fp64_dense_recursive`, and `each_schedule_file_has_its_pinned_digest_and_its_rows` checks their SHA-256 (`crates/neo-compress/tests/unit/akita.rs`). The final proof is not zero-knowledge.

### Algorithm 10. Final verifier

**Input.** The expected state `(i, z0, zi)`, a final proof, the verifier's circuits and security minimum, and the verifier's compression key of Algorithm 11. The verifier derives each key from its circuits, never from the proof.

**Output.** Accept or reject.

**Steps.**

1. Require that the proof's state is the expected state and that `i < q`.
2. If the proof has no steps, accept exactly when `i = 0` and `zi = z0`.
3. Require `i > 0` and `pc ∈ {1, 2}`.
4. For each slot, require that its claim has the shape of an accumulator of its circuit and that its depth is at most `cap`.
5. Require that the block count of the compression key is the count of Algorithm 11 for the verifier's relations. Compute the claims and the layout of Algorithm 9, steps 2 and 3, from the verifier's relations, the key, `pc` and the two depths. Require that the error sum of section 3 is within the security minimum.
6. Compute `x = encHash(stateHash(vk, i, z0, zi, pc, slots))` from the proof's slot claims and depths. The fresh claim is the proof's fresh commitment and `x`. The verifier does not read a public input of the fresh claim from the proof.
7. Verify the Π_CCS run of Algorithm 9, step 5, under rows 1 to 3 of section 5.4 with the proof's commitment of `W`: steps 2 to 5 of Algorithm 4 for the fresh claim alone, with the proof's CE claim as the output. Require that the point of the CE claim is `r'` and that its digest is the digest of the Π_CCS transcript.
8. Require that the first sum-check has `n` rounds and the key sum-check has `6 + n_A` rounds. Run rows 4 to 17 of section 5.4 with the compression key, the three claims and the proof. For each round `j` of the first sum-check, require `g_j(0) + g_j(1) = c_j`, with `c_0 = Y` of Algorithm 9, and set `c_{j+1} = g_j(ρ_j)`. For each round `j` of the key sum-check, require `g'_j(0) + g'_j(1) = c'_j`, with `c'_0` the proof's `K_w`, and set `c'_{j+1} = g'_j(ζ_j)`.
9. Compute the row point `s` and `scale` of `ω` (Algorithm 9, step 10). Require

   ```text
   c'_{6+n_A} = scale · Â(ζ, s) · H̃(ζ)
   H̃(ζ) = (Σ_h eq(ζ_lo, h) · u_h) · Σ_c μ^c · sel_c · P_c(ζ_hi)
   ```

   Here `ζ_lo` is the first 6 coordinates of `ζ` and `ζ_hi` the other `n_A`. `P_c` is the multilinear extension of `j ↦ eq(π_c, j) · [j < w_c]` on `n_A` bits, where `eq(π_c, j)` is 0 when `j` has a bit at or above the length of `π_c`. The verifier computes `P_c(ζ_hi)` in a number of steps linear in `n_A`: an index is below `w_c` exactly when, at the highest bit where the two differ, the index has 0 and `w_c` has 1.
10. Compute `G(ρ)` with the proof's `K_w` as the key's part, and the other parts from the rows of each claim's relation, the claims and the challenges. Require

    ```text
    c_n = W(ρ) · (G(ρ) − E(ρ) · A(ρ)) + E(ρ) · B(ρ) + S(ρ) · A(ρ)
    A(ρ) = low · Σ_k ψ_k · V_k        B(ρ) = low · Σ_k ψ_k · τ_k · V_k        low = Π_{i ≥ v} (1 − ρ_i)
    ```

    `E(ρ)` and `S(ρ)` are the multilinear extensions of `E` and `S` of Algorithm 9 at `ρ`.
11. Verify the one Akita proof of Algorithm 9, step 11: `Â` at `(ζ, s)` with the proof's `Â(ζ, s)` against the commitment of the compression key, `W` at `ρ` with the proof's `W(ρ)`, and `V` at the point and the value of that step. Take the schedule row from the key's file and the variables of `W` and `V` that the verifier's layout gives. Require that each commitment's bytes are the encoding of the commitment that they decode to, and that the profile of each commitment is the profile of that row for its polynomial. The verifier derives the short prefix of Akita's setup from Akita's fixed public seed and that row, and takes the commitments of the longer prefixes from the compression key.

**Rejects.** Each step's failure, a lookup challenge that is a table value, and an `ω` with no row point. The verifier reads no element of the Ajtai key. Its time is linear in the rows of the matrices: step 10 makes one pass over the rows of each claim's relation. It is not a succinct verifier.

**Source.** `Verifier::verify_final` (`crates/nightstream/src/circuit.rs`), `BoundedDepth::verify_final` (`crates/nightstream/src/lifecycle/bounded_depth/compression.rs`), `crates/neo-compress/src/verifier.rs`, `crates/neo-compress/src/weights.rs`, `crates/neo-compress/src/key.rs` and `crates/neo-compress/src/akita/verifier.rs`. Not checked against Lean, and no Lean statement exists. The aim is that the final verifier accepts only when some proof with witnesses satisfies `Export.BoundedDepth.RunCompleteness.Verifies` for the statement, except with an error. If this holds, each security result of section 1 for the terminal relation applies, because that verifier accepts only what the terminal relation accepts (`Export.BoundedDepth.RunCompleteness.holds_of_verifies` (`16655a19ab01f5ae`)). No theorem states the aim. Section 3 gives the error terms that Rust computes. The premises are those of `SECURITY_MODEL.md`, adversary 6: the compression key of Algorithm 11, the binding and knowledge soundness of Akita for `Â`, `W`, `V` and Akita's setup prefixes, Keccak and Poseidon2 as random oracles, and Module-SIS for the Akita commitments. In step 11, the verifier fixes the schedule row from its key and its own layout, so the proof selects no field of an Akita profile (`decisions/akita-compression.md`). The test `the_verifier_expands_no_key_element` counts the SHAKE128 output of the key expansion of `neo-ajtai` on small claims and finds none while the verifier runs (`crates/neo-compress/tests/verifier_reads_no_key.rs`). The ignored test `hash_chain_compresses_and_verifies_without_the_ajtai_key` makes the same count for the reference application through the public API (`crates/nightstream/tests/compression.rs`).

### Algorithm 11. Compression key

The compression key replaces the Ajtai key in Algorithm 10. It is a function of the public seeds and of the block count of the key polynomial only, so each circuit with one block count has the same key.

**Input.** The light and refresh relations of a circuit (Algorithm 2).

**Output.** The compression key (section 4.5). For the prover, also the compression setup, which holds the key rows and Akita's prover data of `Â`.

**Steps.**

1. Let `w` be the larger number of ring columns of the two relations. Take the first schedule file of section 3, in increasing size, whose key polynomial holds `w` blocks. A key polynomial of `6 + n_A + 5` variables holds `2^(n_A)` blocks. Fail when no file holds `w` blocks.
2. Expand rows 0 to `κ − 1` and blocks 0 to `2^(n_A) − 1` of the production Ajtai key with Algorithm 1.
3. The key polynomial `Â` has, lowest variable first, 6 lane variables, `n_A` block variables and 5 row variables. Cell `h + 64j + 64 · 2^(n_A) · i` is coefficient `h` of `A_ij` for `i < κ`, and 0 for `i ≥ κ`.
4. Commit `Â` with Akita under the profile of the first row of the file. Each row of one file has the same profile for `Â`, so one commitment serves each row.
5. Compute Akita's commitments of the prefixes of its setup that the rows of the file offload (ePrint 2026/1983 section 9), from Akita's public seed.
6. The compression key is `2^(n_A)`, the commitment of step 4 and the commitments of step 5. The compression setup also keeps the rows of step 2 and Akita's prover data of `Â`.

**Rejects.** `compression_setup` and `compression_key` fail for a circuit whose larger relation has more blocks than each schedule file holds. The code pins the bytes of the key of each block count that has a schedule file, and a test runs steps 2 to 5 again for each pinned key. `Verifier::compression_key` gives the pinned key of the verifier's block count, and `Verifier::decode_compression_key` accepts loaded bytes only when they equal that key. So a verifier runs no step of this algorithm, and a loaded key is checked against the pin, not trusted.

**Source.** `Prover::compression_setup`, `Verifier::compression_key` and `Verifier::decode_compression_key` (`crates/nightstream/src/circuit.rs`), `BoundedDepth::compression_setup` and `BoundedDepth::compression_key` (`crates/nightstream/src/lifecycle/bounded_depth/compression.rs`), `crates/neo-compress/src/key.rs` for steps 2 to 6 and the pinned keys, and `crates/neo-compress/src/akita.rs`. Not checked against Lean, and no Lean statement exists. Step 2 is Algorithm 1, whose source is Lean. The tests `the_derived_verification_key_is_the_key_of_the_prover` and `another_verification_key_does_not_verify` (`crates/neo-compress/tests/argument.rs`) and `the_reduction_of_the_light_relation_gives_the_shake_value`, `the_powers_of_omega_are_a_scaled_equality_table` and `the_prefix_weight_is_the_extension_of_the_equality_table_of_a_prefix` (`crates/neo-compress/tests/unit/key.rs`) check the key and its reduction on fixed inputs. Record `decisions/akita-compression.md`.

### Lifecycle rules

These four rules join Algorithms 5, 6, 9, 10 and 11. Rust keeps them, and no theorem covers them (not checked against Lean). The tests check them on fixed inputs. CI runs each test that this section names, except the tests marked "ignored". Each ignored test holds the compression setup of the reference application, about 10 GB, and runs by hand (`wiki/development/testing.md`).

1. **Each returned proof verifies.** Algorithm 6 accepts each proof that `prove` or `extend` returns from a proof that Algorithm 6 accepts, at the proof's own state. `extend` does not verify the proof that it receives. Before it returns, it requires that every row of its circuit's relation, whose rows verify the fold, holds on the new assignment (`Relation::rows_hold`), so a prover fault gives an error and not a proof (not checked against Lean). `Relation::instance` stores the fresh witness as two masks, one of the coordinates that hold 1 and one of the coordinates that hold -1, and commits to it. Each assignment below the fresh norm bound `b = 2` is the assignment of its masks (`Lifecycle.BoundedDepth.FreshInstance.Masks.assignment_ofAssignment` (`3330bb04b539eb41`)). The commitment, the public input and the norm bound of this instance hold by construction, so the instance holds under its circuit's relation exactly when the rows hold (`Lifecycle.BoundedDepth.FreshInstance.built_holds_iff` (`1686beca193d8cd5`)). Thus the row check accepts the same assignments as `Relation::holds`, which the verifier of Algorithm 6 runs with the commitment. Tests: `hash_chain_proves_and_verifies` (`crates/nightstream/tests/bounded_depth_api.rs`), and `light_folds_and_a_refresh_verify`, `refresh_folds_under_both_relations_verify`, `native_hash_chain_verifies` and `native_addition_verifies` (`crates/nightstream/tests/bounded_depth_lifecycle/mod.rs`). The two lifecycle tests use the caps 2 and 0, so that a short run makes light folds into both slots and refresh folds under both relations. `light_step_zero_holds_every_row` and `refresh_step_zero_holds_every_row` check that the base step of each circuit satisfies every row (`crates/nightstream/tests/bounded_depth_base_steps.rs`).
2. **Each returned final proof verifies.** Algorithm 10 accepts each final proof that `compress` returns from a proof that Algorithm 6 accepts, for a verifier of the same circuits and security minimum, under the compression key that the verifier takes or loads (Algorithm 11). `compress` does not verify the proof that it receives or the final proof that it returns. Tests: `hash_chain_compresses_and_verifies_without_the_ajtai_key` (`crates/nightstream/tests/compression.rs`, ignored), and `a_compressed_proof_verifies_and_a_changed_one_does_not` (ignored) and `the_proof_of_zero_steps_compresses_to_its_statement` (`crates/nightstream/tests/bounded_depth_lifecycle/compression.rs`). The first test uses the public API. The verifier takes its key and loads it again from its bytes, the prover makes its own setup, and the test compresses a hash chain after 1 step and after 3 steps. The second test uses the cap 2 and compresses a proof whose last circuit is the refresh circuit and one whose last circuit is the light circuit. `the_derived_verification_key_is_the_key_of_the_prover` checks on small claims that the key of the verifier is the key of the prover's setup (`crates/neo-compress/tests/argument.rs`). No test compresses a proof with a slot depth near the cap 64, so no test uses a lookup table of the largest size.
3. **An edited proof fails.** Algorithm 6 rejects a proof with a wrong state, a changed slot depth, a changed slot witness, a slot claim of the wrong shape, a recommitted fresh witness that fails a row or has a nonzero tail, and a changed fresh commitment (`light_folds_and_a_refresh_verify`). `decode_proof` rejects truncated bytes (`hash_chain_proves_and_verifies`) and an iteration or slot depth word not below `q` (`a_word_not_below_q_does_not_decode`), and Algorithm 6 rejects a decoded proof with a changed byte (`hash_chain_proves_and_verifies`). Algorithm 10 rejects a final proof for another state or iteration, and a final proof with a changed program counter, slot depth, slot claim word, fresh claim word, Π_CCS digest, Π_CCS round, sum-check round, key sum-check round, evaluation, key part `K_w`, key value, Akita commitment byte or Akita opening byte. `FinalProof::from_bytes` rejects short bytes, a byte after the proof and a word not below `q` (`a_compressed_proof_verifies_and_a_changed_one_does_not`, ignored; `a_final_proof_has_one_encoding_and_a_circuit_and_depths_in_range`). Algorithm 10 and the decoder of the key reject a compression key of another block count (`a_final_proof_has_one_encoding_and_a_circuit_and_depths_in_range`). Algorithm 10 rejects a proof under a key with a changed byte (`a_proof_does_not_verify_under_another_compression_key` and `hash_chain_compresses_and_verifies_without_the_ajtai_key`, both ignored). From a proof that Algorithm 6 rejects for a slot witness above its bound or a changed slot witness, `compress` returns a final proof that Algorithm 10 rejects. For a fresh witness with an entry 2, it returns an error or such a final proof (`a_bad_witness_gives_no_accepted_proof`, ignored). The tests of `crates/neo-compress/tests/argument.rs` check the argument alone on small claims: a proof for another transcript, a changed proof, a false key part with a first round that agrees with it, a proof under another verification key, a false claim, a witness outside its table and another witness do not verify.
4. **The selected circuit.** Rust selects the circuit of each step from the depth of the selected slot (Algorithm 5, step 3): a light fold below `cap`, a refresh fold at `cap`. The Lean step relation lets the prover pick the circuit (Algorithm 3, step 1), and the light depth guard rejects a light fold at `cap` for every pick. So the Lean results cover the Rust selection. No Lean result states that the Rust schedule is Lean's `Spec.HyperNova.Construction2.BoundedDepth.refreshSchedule` (`SECURITY_MODEL.md`, trust item 4).

## 7. Nebula memory

Nebula memory checking (Nebula, Sections 3 and 4) gives an application public ROM and RAM. A memory application is an application program that both step circuits run (`Lifecycle.Stage1.Application.Program`). Its rows hold the memory rows, three record chains, the memory carry and the application's own rows, which meet at fixed memory ports (`Lifecycle.Nebula.MemoryApp.program`). It adds no commitment, no claim, no evaluation and no field to the state of section 4.2, so Algorithms 1 to 6, 9 and 10 do not change. The memory carry is in the application state: the four state words of section 4.2 are a digest of the application words and the carry (section 7.5).

The steps come in segments of `N` steps. At the start of a segment, the prover proposes two chain roots, and the relation derives the memory challenge `η` from them. Each step checks its memory accesses against fingerprint products and adds its own records to the chains. The step that completes the segment requires that the chains equal the proposed roots and that the products balance. The verifier then requires a closed segment and runs the terminal verifier (Algorithm 6) on a proof with witnesses, or the final verifier (Algorithm 10) on a compressed proof (Algorithm 8). The aim of these checks is that an accepted proof attests one execution of the application from the statement's initial words and the program's memory to the statement's final words, in which each memory access goes through a port and reads the last value written (`SECURITY_MODEL.md`, adversary 5).

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

A package is the relation of one application at one geometry. The relation reads no image, so one package and one setup serve every program of the geometry (`Lifecycle.Nebula.MemoryApp.program_eq_of_geometry` (`af7d3cb89059f0f8`)). The program enters a proof only through `D_init` (section 7.4), which the verifier computes from the images.

Setup rejects a geometry that breaks one of these rules (`Spec.Nebula.Geometry.Valid`, `Spec.Nebula.Geometry.Secure`):

1. **Exact cover:** `N · B_scan = R + M`.
2. **Timestamp range:** `S_max · N · B_ops < 2^W_ts`.
3. **Field encoding:** `W_ts ≤ 31`, `B_ops · (3 + A + 64 + W_ts) < q` and `B_scan · (32 + W_ts) < q`. A value has 32 bits and a timestamp has at most 31 bits, so `v + 2^32 · t` is below `2^63` and is one field word. The fingerprint of section 7.3 uses this word. Row O4 cannot wrap modulo `q`, and the length of each lane is one word.
4. **Address width:** `R + M < q`. The address field of an operation slot has `A = max(r, μ)` bits (`Spec.Nebula.Geometry.addrWidth`), so the RAM can be smaller than the ROM.
5. **Positive sizes:** `B_ops`, `B_scan`, `N` and `S_max` are at least 1.
6. **Security:** `R + M + N · B_ops ≤ 139,509`. This holds exactly when the fingerprint error `2(R + M + N · B_ops)/q²` is at most `2^−109.91` (`Spec.Nebula.Geometry.secure_iff` (`a0dfdd1bf945a81d`)), the floor of `decisions/nebula-memory-checking.md`.

A Lean memory package is a light and a refresh package whose application is `Export.Stage1.NebulaMemoryPackage.application` at a geometry, with the proofs of rules 1 to 6. The emitter writes them as `nightstream-fprime-bounded-depth-nebula-v1-{light,refresh}-v1.json` and `nightstream-fprime-bounded-depth-nebula-n1-{light,refresh}-v1.json` (`--program`, `Export/BoundedDepth/EmitMain.lean`). The verifier rejects a program whose ROM image does not have `R` words or whose RAM image does not have `M` words before it computes `D_init` (`Spec.Nebula.Plan.Valid`). The segment counter uses `W_seg = ⌈log₂(S_max + 1)⌉` bits.

The plan digest is `Digest("Nightstream/Nebula/v3/plan", [r, μ, W_ts, B_ops, B_scan, N, S_max])` (section 7.4, `Lifecycle.Nebula.planDigest`). The package rows hold it as a constant.

### 7.2 Memory model and records

The global index of ROM address `a < R` is `a`, and of RAM address `a < M` is `R + a`. A cell holds a value and the timestamp of its last access. `ts` is the global timestamp: the number of active operations so far. It never resets. An active operation on a cell with value `v` and timestamp `rt` adds the read tuple `RS = (rt, g, v)` and the write tuple `WS = (ts + 1, g, v')`, and sets the cell to `(v', ts + 1)`. `v'` is `v` for a read and the new value for a write. A write to ROM is invalid. At the start of a segment, the initial snapshot `IS` has one tuple `(t, g, v)` for each cell, and at the end, the final snapshot `FS` has one tuple for each cell. An honest segment has `IS ∪ WS = RS ∪ FS` as multisets (`Spec.Nebula.MemOp`, `Spec.Nebula.Tuple`).

An operation slot has these fields, in this order (`Spec.Nebula.OpSlot`):

| Field | Bits | Meaning |
|---|---|---|
| `pad` | 1 | 1 for an inactive slot |
| `is_write` | 1 | 0 for a read, 1 for a write |
| `is_ram` | 1 | 0 for ROM, 1 for RAM |
| `addr` | `A` | the address |
| `v_r` | 32 | the value read, or for a write the old value |
| `v_w` | 32 | the value written, or for a read `v_r` |
| `rt` | `W_ts` | the timestamp of the last access to the cell |

An inactive slot has every other field zero, at any position. For slot `j` of a step with incoming timestamp `ts`: `cnt_j = Σ_{i ≤ j} (1 − pad_i)`, `wt_j = ts + cnt_j`, `g_j = addr_j + is_ram_j · R`, `RS_j = (rt_j, g_j, v_r_j)` and `WS_j = (wt_j, g_j, v_w_j)`.

A scan slot is a value of 32 bits and a timestamp of `W_ts` bits (`Spec.Nebula.ScanSlot`). Each step has `B_scan` IS slots and `B_scan` FS slots, and slot `j` of step `idx` holds cell `idx · B_scan + j`. The relation computes this index, so a scan slot has no pad.

A step has three lanes: the ops lane of its `B_ops` operation slots, and the IS and FS lanes of its scan slots (`Spec.Nebula.opsLane`, `Spec.Nebula.scanLane`). A lane is its slots in order, each slot its fields in order, each field its bits from the least significant, one bit to a coordinate. So the ops lane has `B_ops · (3 + A + 64 + W_ts)` coordinates, and each scan lane has `B_scan · (32 + W_ts)`. `pack` splits a lane into chunks of 63 bits, the last one short, and turns each chunk into the word `Σ_k 2^k · b_k`, which is below `2^63 < q` (`Spec.Nebula.pack`). It is injective on the lanes of one length (`Spec.Nebula.pack_injective` (`958e936decf1f7b7`)).

**Hash residency.** A fingerprint tuple and its pad gate read only the lane coordinates of their own step, the carry fields `ts` and `idx`, the counters `cnt_j`, the auxiliary words of section 7.3 and constants. `η` and the products are only coefficients and accumulators of the fingerprints. The memory rows keep this rule. The application's rows read only its view (section 7.6), which has no word for `η`, a product or a carry word. Rust `nebula::Circuit::compile` rejects an application row that reads another variable (`crates/nightstream/src/nebula/circuit.rs`; not checked against Lean).

### 7.3 Memory rows

Each step evaluates these rows on its own assignment. They read `η`, the products `h`, `ts` and `idx` of the carry after the step's open action, if any (section 7.5).

| # | Row, for operation slot `j` |
|---|---|
| O1 | every lane coordinate and auxiliary bit is 0 or 1 |
| O2 | `cnt_j = cnt_{j−1} + (1 − pad_j)`, with `cnt_{−1} = 0` |
| O3 | `(1 − is_write_j) · (v_w_j − v_r_j) = 0` |
| O4 | `(1 − pad_j) · (wt_j − rt_j − 1 − diff_j) = 0`, with `diff_j` a `W_ts`-bit auxiliary word |
| O5 | `is_write_j · (1 − is_ram_j) = 0` |
| O6 | `(1 − is_ram_j) · addr_bit_{j,k} = 0` for each `k` from `r` to `A − 1`, and `is_ram_j · addr_bit_{j,k} = 0` for each `k` from `μ` to `A − 1` |
| O7 | `pad_j · w = 0` for each field `w` of the slot after `pad` |
| O8 | `h_rs_j = h_rs_{j−1} · (pad_j + (1 − pad_j) · f(RS_j))` |
| O9 | `h_ws_j = h_ws_{j−1} · (pad_j + (1 − pad_j) · f(WS_j))` |

| # | Row, for scan slot `j` with cell `g = idx · B_scan + j` |
|---|---|
| S1 | every IS and FS coordinate and auxiliary bit is 0 or 1 |
| S2 | `h_is_j = h_is_{j−1} · f(t^IS_j, g, v^IS_j)` |
| S3 | `h_fs_j = h_fs_{j−1} · f(t^FS_j, g, v^FS_j)` |

The fingerprint is `f(t, g, v) = g + η1 · (v + 2^32 · t) − η2` in `K` (`Spec.Nebula.fingerprint`). It is Nebula's `Hash` with the global index for the address, and with the value and the timestamp in one field word. Rule 3 of section 7.1 keeps `v + 2^32 · t` below `q`, so different pairs `(t, v)` give different words. Each product starts from its carry value. The step's outputs are `ts + cnt_{B_ops−1}`, which must be below `2^W_ts`, the last value of each product, and `idx + 1`. Rule 3 keeps every term of O4 below `2^32`, so O4 is an integer equation and gives `rt_j < wt_j` (`Spec.Nebula.StepRows`, `Spec.Nebula.OpSlot.Rows`).

**Auxiliary words.** Each counter `cnt_j` is a witness word, and so is each of the four products at the start of the step. Each fingerprint step `h_j = h_{j−1} · x` has factor words: the two coordinates of `f`, for an operation slot the two of the gate `x = pad + (1 − pad) · f`, and the cross products `h_{j−1}.c1 · x.c1` and `h_{j−1}.c1 · x.c0` (`Lifecycle.Nebula.fingerprintWords`, `Lifecycle.Nebula.opsFactorWords`, `Lifecycle.Nebula.scanFactorWords`). An operation slot has 6 factor words for each fingerprint, and a scan slot has 4. The word `v + 2^32 · t` is a sum of bit chunks, so it is not a witness word. So each row of a fingerprint step is a word minus a recipe with one product. It lowers to one R1CS row and no fresh column, and it reads the bit chunks of `t`, `g` and `v` once (`Lifecycle.Nebula.Rows.gatedRows`, `Lifecycle.Nebula.Rows.factorRows`). The rows hold exactly when the equations above hold and the auxiliary words are their honest values (`Lifecycle.Nebula.Rows.polyRows_iff` (`e2bc9a93cd5ef638`)), and the validity predicate of the memory program requires these values (`Lifecycle.Nebula.AuxHonest`).

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

(`Lifecycle.Nebula.etaChallenges`). Each step evaluates it, and only a step that opens a segment keeps it. Canonical inputs that differ absorb different chunks (`Lifecycle.Nebula.etaInputBlocks_injective` (`f5ce93c53513eb1e`)). `D_mem` must be absorbed: without it, a prover could learn `η` first and then choose the last segment's final snapshot.

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

**Catalog applications.** A catalog lists memories, each with an id, a kind (RAM or ROM) and ports, in the form of the Rocq proof of the WASM relation (`MemoryTrace.v` in `formal/wasm-rocq` on the `main` branch, copied in `Spec/Nebula/MemoryTrace.lean`). Each active port of a row gives one event. A binding (`Spec.Nebula.MemoryTrace.Binding`) gives each bound memory a region: a base address and one bit width for each address part, with the address `base + x₀ + 2^w₀ · (x₁ + 2^w₁ · …)`. A RAM memory is in the RAM space, and each other memory in the ROM space. Two memories of one space have disjoint regions (`Spec.Nebula.MemoryTrace.Binding.Disjoint`). A read event is a read access of its value. A write event is a write access of its value, which reads its prior value when it has one. The active slots, in order, must bind a reorder of the events that keeps each pair of one memory with a write in trace order (`Spec.Nebula.MemoryTrace.Reorders`). When the regions are disjoint, every event is in range and the initial state holds the program images (`Spec.Nebula.MemoryTrace.Binding.Initial`), an execution whose active ports bind such a reorder makes the trace memory consistent (`Spec.Nebula.MemoryTrace.memoryConsistent_of_executes` (`632f3e47b7a2481d`)).

Rust `nebula::CatalogBinding` makes these rows for the checked memories, from a catalog, a region for each checked memory and a slot plan (`crates/nightstream/src/nebula/catalog.rs`; not checked against Lean). For a port `p`, `a_p` is its activation: 1, the column `c`, or `1 − c`.

1. **Slot.** `1 − pad_j = Σ a_p`, `is_write_j = Σ a_p` over the write ports and `is_ram_j = Σ a_p` over the RAM ports, for the ports `p` of slot `j`, and `c · (c − 1) = 0` for each activation column `c`.
2. **Order.** For each port `p` with a dependent port `p'` of a higher rank in an earlier slot, `a_p · Σ a_p' = 0`. Two ports are dependent when they are ports of one memory and one of them writes.
3. **Address.** The slot address is the region's base plus the packed parts. The base of a region is a multiple of its size `2^Σw`, so the rows read each part from its run of the slot's address bits and the bits above the region from the base, and a part needs no other range check.
4. **Values.** A read port has `v_r = value`. A write port has `v_w = value`, and `v_r = prior` when it names a prior column.
5. **Host checks.** A memory without a region has no slot and no row. At build, the regions of one space are disjoint and inside the space, the plan has `B_ops` slots, and it names each checked port once.

When the plan names each checked port once, each slot shows at most one active port, and of two dependent active ports in different slots the port of the earlier slot has the lower rank, the events of the active ports in slot order are a reorder of the row's events that keeps each dependent pair (`Spec.Nebula.MemoryTrace.reorders_of_slotOrder` (`2a678a47159b7410`)). The slot plan changes only completeness and cost. `CatalogBinding::program` puts each preloaded cell of a checked memory at `base + packParts(a)` of its image and rejects a cell outside its region or of a memory that the catalog does not have. No Lean result shows that the five rules above enforce these conditions, or covers the address and value rows (`SECURITY_MODEL.md`, Out of scope).

**Several rows in one step.** One step can hold `k` rows of a catalog application, so that the rows share the cost of one fold. The step uses the batched catalog (`Spec.Nebula.MemoryTrace.batchCatalog`). It has the memories of the catalog in order. Each memory has the ports of row 0, then of row 1, up to row `k − 1`, and a map `shift i` moves the columns of row `i` to columns of the step witness. At `shift i c`, the step witness holds the value of row `i` at column `c`, for each column `c` that a port reads (`Spec.Nebula.MemoryTrace.HoldsRows`). So a step lists its events memory by memory, and its rows list them row by row. When the memory ids are distinct, each memory's events come in the same order in the two lists. So batched steps are memory consistent exactly when their rows, in order, are memory consistent (`Spec.Nebula.MemoryTrace.memoryConsistent_batchCatalog_iff` (`411b657361ef91f8`)). When the regions are disjoint, every event of the rows is in range and the initial state holds the program images, an execution whose active ports, step after step, bind a reorder of the batched events of each step makes the trace of the rows memory consistent (`Spec.Nebula.MemoryTrace.memoryConsistent_of_batched_executes` (`ced62611c59da5a9`)). A slot plan that obeys the conditions above in each step gives this reorder (`Spec.Nebula.MemoryTrace.memoryConsistent_of_slotOrder_executes` (`a78eb43ce17803e3`)). Rust `neo_application::MemoryCatalog::batched` builds the batched catalog, and the catalog rejects two memories with one id. The application of an R1CS relation of one row, `nebula::R1csApplication`, keeps the constant-one column for all rows and moves each other column of row `i` by `i` times the row width. Its relation also makes each row's `previous_step_column`s equal to the next row's `next_step_column`s, and its slot plan is the one-row plan for each row in turn. It refuses a size that gives a memory of the catalog no region, so that no memory of it is only host-checked. `RowRelation::of` refuses a relation with a verifier-supplied column other than the constant-one column, because no memory verifier supplies such a column (`crates/nightstream/src/nebula/r1cs.rs`; not checked against Lean). Each of its rows other than `one − 1 = 0` reads the constant 1 in place of the constant-one column. Where the rows have the row `one − 1 = 0`, the rows with the column read as 1 hold exactly when the rows hold (`Lifecycle.Nebula.holds_fix_iff` (`f87a5be4c9947467`)), so the application keeps its model, its port coverage and its completeness (`Lifecycle.Nebula.PortApp.fixColumn`). A row `x · (x − one)` then reads only `x` and constants, so the slot rule of section 4.4 gives `x` a bit slot. Lean does not check that the Rust rows are this rewrite. The WASM application `neo_wasm::nebula::WasmApplication` is this application of the WASM relation, with column 0 as the constant-one column. `neo_wasm::nebula::WasmSize` fixes three sizes of it, each a geometry, the rows of one step and the regions of `WasmLimits::regions`. A WASM verifier takes the size from its statement, never from the run (`decisions/wasm-fixed-sizes.md`).

### 7.7 Native pieces

A Rust-built application in a bounded-depth step package (`Export.BoundedDepth.StepPackage.Package`) places each Poseidon2 permutation as a block and each live word that a row forces to 0 or 1 in a bit slot, as the packages that Lean emits do.

**Input.** The application's rows and recipe rows, and its shape: for each permutation, its first local and its 16 input lanes.

**Output.** The application's pieces of the package. The first piece has every row except the 1,096 recipe rows of each block. Then each permutation is one block (`Export.Generic.Frame.Block`): 150 S-box rows on its S-box columns, which the application's blocks lead, and no slot for its 1,096 locals. Each row reads the form of an output local in place of the local (section 4.4). The live columns of the application are its witness words and its locals that are not dead locals. A live column that a row of the package forces to 0 or 1 has a bit slot of one coordinate (`Layout.LowNormSlot.Kind`, `Export.Generic.SlotLayout.forcedColumn?`). Every other word that a row reads has a field slot of 41 trits. A word that no row reads has no slot.

**Steps.**

1. Require that the blocks ascend and do not overlap, that each lane reads only live words before its block (`Export.Generic.Frame.Framed`), and that each kept row reads no dead local (`Export.Generic.NativeBlock.rowAvoids`).
2. Require that the 1,096 rows of each block are the rows of one permutation over its lanes, in the order of `formal/nightstream-fprime/NightstreamFPrime/Gadgets/Poseidon2/Permutation.lean`.
3. Drop the recipe rows of each block and add its block. Give each dead local no slot and the value 0.
4. Give the package the slots of Lean's rule on its pieces (`Export.BoundedDepth.StepPackage.runsOf`), with its `bit` hints and every live column of the application as the bit candidates.
5. Bind the shape into the package identity: each block's first local and lanes.

**Rejects.** A block outside the locals, a row in two blocks, a lane that reads its own block or a later word, a kept row or a lane that reads a dead local, and a block whose rows are not its permutation.

**Source.** Soundness: when the S-box rows are zero and the other rows hold with the output forms, the outputs of each block, in the environment with the output forms, are the Poseidon2 permutation of its lanes, and an environment that differs from it only on the dead locals satisfies the other rows and the 1,096 template constraints of every block (`Export.Generic.NativeBlock.blocks_sound` (`3f0b91065a0d59cc`)). Completeness: an honest template trace, with each S-box column at its template local, makes the S-box rows zero, and the environment with the output forms is then the honest one off the dead locals (`Export.Generic.NativeBlock.blocks_complete` (`9bd9a92bb49d2438`)). A package with a native application is complete too. Its pieces are one rows piece with the native rows, then the package pieces of Lean pieces, in which each native block is a permutation piece. So its rows are the native rows and the rows of the Lean pieces (`Export.BoundedDepth.NativeCompleteness.relationRows_eq` (`0c8035a8e13c3116`)). Let the constraints of the Lean pieces and the native rows hold in the honest values, let the Lean pieces meet the premises of section 4.4, and let the native rows read only columns below the first fresh column and no dead local, as step 1 requires. Then the honest assignment of the slots of step 4, read from the values of section 4.4, makes the rows and the S-box rows of the package zero, carries the honest public columns, and is in the norm bound, whatever the bit candidates are (`Export.BoundedDepth.NativeCompleteness.native_complete` (`8ea66c7824c62767`)). A value 0 or 1 is valid in a bit slot (`Export.Generic.NativeBlock.bit_valid_of_boolean` (`57dcbb4510bea8b0`)). A bit slot whose coordinate passes the norm check decodes to −1, 0 or 1 (`Export.Generic.NativeBlock.bitSlot_sound` (`e976afcbfa3c7e5c`)), so the row that forces the word makes it 0 or 1. The relation is sound for every slot layout (`Export.Generic.CircuitRelation.rowsZero_implies_spec` (`9b58c4fd0fc190b5`)). Rust: `StepPackage::with_application` in `crates/nightstream-fprime/src/bounded_depth/application.rs` and `check_blocks` in `crates/nightstream/src/application/poseidon2.rs` (not checked against Lean). Rust's rule (`StepPackage::rule_slots` in `crates/nightstream-fprime/src/bounded_depth/layout.rs`) gives the slots of each package that Lean emits (the test `lean_packages_follow_the_slot_rule`).

### 7.8 WASM table access traps

**Input.** A `table.get` or `table.set` row, its table id, its `u32` index, and the current table size.

**Output.** One in-bounds table access, or a trapped terminal state.

**Steps.**

1. Read the table size from the table-size RAM at the program-bound table id, and compare the index with that size.
2. If the index is below the size, read the table element for `table.get` or write it for `table.set`. Bind the value to the corresponding stack operand or result.
3. Otherwise, disable the table-element port, mark the execution as trapped, and capture no output. A trapping `table.get` has no table value to read.

**Rejects.** A normal return after an out-of-bounds access, a trapped return for an in-bounds access, or a table-element access on a trapping row.

**Source.** [WebAssembly table execution](https://webassembly.github.io/spec/core/exec/instructions.html); `crates/neo-wasm/src/ccs/trap.rs`, `crates/neo-wasm/src/ccs/stack_io.rs`, and `crates/neo-wasm/src/adapters/wasmtime/normalize/trace_build.rs` (not checked against Lean).

### Algorithm 7. Memory step

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

**Source.** `Lifecycle.Nebula.MemoryApp.program`, the step `Spec.Nebula.invoke`. Rows that hold imply the program's step and validity predicates (`Lifecycle.Nebula.MemoryApp.soundness` (`1a9c9027f88907c8`)). On a reachable input carry, they give one step of `invoke` (`Lifecycle.Nebula.StepWitness.RowsHold.invoke` (`43d83c7be2720429`)), and an honest step satisfies the rows (`Lifecycle.Nebula.MemoryApp.completeness` (`56a9eca2876bd0e0`)). Each execution of `S · N` steps with `1 ≤ S ≤ S_max` has a step history whose statement passes Algorithm 8 steps 3 and 4 (`Lifecycle.Nebula.MemoryApp.history_of_execution` (`f12147627b97b445`)). The first step is the base step of the light circuit (Algorithm 3, step 2).

### Algorithm 8. Memory verifier

**Input.** The package of the application at its geometry, a program, the initial and final application words, and a memory proof: the final carry with a step proof. The proof bytes start with the 39 carry words, and each word must be below `q`. The statement is the step count `T`, the two application words, the segment count `S`, the final timestamp and the final memory root. The verifier takes `S = seg_idx`, `T = S · N`, the final timestamp `ts` and the final memory root `D_mem` from the final carry, so the caller never gives a carry or a statement integer.

**Output.** Accept or reject.

**Steps.**

1. Require that the program has the package's geometry and valid images, and compute `D_init`.
2. Require that each state of the statement has the application's word count, and that each application word and the final timestamp is below `q`. Each other field and carry counter must equal a value below `q`.
3. Compute the initial state as the state digest of the initial words and the start carry, and the final state as the state digest of the final words and the final carry.
4. Require `idx = N` in the final carry, `1 ≤ S ≤ S_max`, `T = S · N` and `seg_idx = S`, and require that the final timestamp and the final memory root are `ts` and `D_mem` of the final carry. So `T = 0` is rejected.
5. Run Algorithm 6 on `(T, initial state, final state)` and the proof. A compressed memory proof is the final carry with a final proof of Algorithm 9. For it, run Algorithm 10 in place of Algorithm 6.

**Rejects.** Each step's failure.

**Source.** `nebula::Verifier::verify` and, for a compressed memory proof, `nebula::Verifier::verify_final` (`crates/nightstream/src/nebula/proving.rs`), with steps 2 to 4 in `terminal_states` (`crates/nightstream/src/nebula/mod.rs`; not checked against Lean). Steps 3 and 4 are `Lifecycle.Nebula.TerminalChecks`. A chain of valid steps with these checks gives a run that the memory model accepts, or a state-digest collision in the run (`Lifecycle.Nebula.accepts_or_collision` (`7237d81815f8356a`)). For an honest execution, steps 3 and 4 pass and Algorithm 6 accepts the honest proof (`Export.BoundedDepth.NebulaCompleteness.memory_complete` (`6f8f45de5be0373a`)). Algorithm 6 alone does not make steps 2 to 4, so `nightstream::Circuit::verifier` refuses a memory relation: a pinned Lean memory package, a circuit from `nebula::Circuit::compile`, or a saved memory circuit, whose file starts with `NSOPTM`, 0, 1 (`crates/nightstream/src/circuit.rs`; not checked against Lean). A verifier compiles its own circuit and never loads a saved file.

## 8. Differences from the papers

Each row is a place where Nightstream differs from SuperNeo v1.2 (ePrint 2026/242), HyperNova or Nebula.

| Paper | Nightstream | Record |
|---|---|---|
| SuperNeo v1.2 Section 8 gives three parameter sets for fields of 64 bits or less. None of them uses the prime `2^64 − 59`. Section 8.1 uses the ring `X^64 + 1` with a different prime, and Section 8.2 uses `k = 14` and `B = 2^14`. | The profile binds `q = 2^64 − 59`, the ring `X^64 + 1`, `T = 128`, `κ = 18`, `k_rho = 16` and `B = 2^16`, and the row cube is `2^28`. Theorem 4 holds with `z = 4`, and Theorem 5 gives `T`. The profile is called "Nightstream P64, k_rho = 16". | `AGENTS.md` (Protocol), `decisions/field-2p64-59-ring-x64.md`, `decisions/fprime-stage1-domain-2p28.md` |
| SuperNeo v1.2 Section 7 runs Π_CCS, Π_RLC and Π_DEC in each fold. | Two step circuits. A light fold has no Π_DEC and combines the accumulator with the weight 1, so its norm grows by `T(b − 1)` for each fold (Cyclo, ePrint 2026/359, Theorem 3). Theorem 3 writes the growth as `L·b·γ`. Here `L = 1` and `γ` is `T`. Cyclo's bound `b` is inclusive, so it is `b − 1` in the strict norm of this document. A refresh fold runs Π_DEC on the accumulator before Π_CCS and Π_RLC. The light circuit rejects a fold at depth 64 (Algorithms 3 and 4). | `Spec/Folding/Light/Verifier.lean`, `Spec/Folding/Refresh/Verifier.lean`, `Spec/Folding/Schedule.lean` |
| SuperNeo v1.2 Section 8 sets the MSIS norm to `8TB`. | Extraction recovers accumulators up to `B + 64·T·(b − 1)`, so binding needs MSIS at the chain norm 75,497,472 (section 3). | `Spec/Folding/Schedule.lean`, `SECURITY_MODEL.md` trust item 2 |
| SuperNeo v1.2 Definition 8 samples the commitment matrix uniformly at random. | Algorithm 1 expands the matrix from one fixed public seed with SHAKE128. The premises are SHAKE128 as a random oracle and MSIS for a uniform matrix. | `decisions/fprime-ajtai-shake128-setup.md`, `decisions/fprime-ajtai-shake128-security-argument.md` |
| HyperNova Construction 2 step 5 hashes `(vk, i, z_0, z_i, U_i, pc_i)` in this order, with the running instances `U_i` in full. | The state hash starts with a domain block and then hashes the two key digests, `i`, `z0`, `zi`, `pc` and the digest of each slot. Each slot digest hashes the accumulator and its depth under its own domain (section 4.2). | `Lifecycle/BoundedDepth/State.lean` |
| HyperNova Construction 3 hashes the public parameters and the structure into the first challenge, and it derives each later challenge from the prover's message and the challenge before it. The errata's Construction 3 also absorbs the instances before the first challenge. | The fold transcripts absorb a domain block, the fresh statement, the Π_DEC children of a refresh fold and the prover messages. The accumulator enters through the state hash in the fresh public input. | `decisions/piccs-prior-state-digest.md`, `SECURITY_MODEL.md` trust item 3 |
| HyperNova Definitions 7 and 11 let the adversary or the IVC prover choose the structure. | The verifier's packages fix one application, and their identity is pinned (Algorithm 2). | `SECURITY_MODEL.md` (Applications and setups) |
| HyperNova's history result uses Definition 7 knowledge soundness of the NIFS. | Lean states the history result for each slot under errata Assumption 1 for the light and refresh folds. | `SECURITY_MODEL.md` trust item 3 |
| HyperNova NIVC selects the circuit of a step with `φ(z_i, w_i)`. | The prover picks the circuit: Lean takes the selector that returns the circuit that ran, and Rust selects it from the depth of the selected slot (lifecycle rule 4). Both circuits run the same step, so the pick does not change the computed state. | `SECURITY_MODEL.md` trust item 4 |
| HyperNova Construction 2 `F′_j` computes `pc_{i+1} = φ(z_i, w_i)` and hashes it. It does not compare it with `j`. | The step relation of circuit `j` requires `φ(z_i, w_i) = j` and `pc_{i+1} = j`. | `Spec/HyperNova/Construction2/Paper.lean` |
| The HyperNova verifier accepts each proof at `i = 0` when `z_i = z_0`. | At `i = 0`, the terminal verifier accepts only the initial proof, and each other proof needs `i > 0` (Algorithm 6). The accepted states are the same. | `Spec/HyperNova/Construction2/Paper.lean` |
| The HyperNova verifier checks each running instance under one relation, and a SuperNeo v1.2 Definition 21 witness has norm below `B`. | The terminal verifier holds each slot to the bound of its depth, `accumulatorBound j ≤ 10,368` (Algorithm 6, Cyclo Theorem 3 item 1). So Definition 11 (i) holds for each proof that the verifier accepts. The terminal relation of the security results keeps `B`. | `Export/BoundedDepth/RunCompleteness.lean` |
| HyperNova Definition 11 (ii) is knowledge soundness. It compares the probability that the extractor's witnesses give the statement with the probability that the verifier accepts. | Lean bounds the joint event: the terminal relation accepts, and the witnesses that the reverse extractor returns are not `iteration` valid steps from `z0` to `zi` (`Export.BoundedDepth.IvcSecurity.ExtractionFailure`, `Export.BoundedDepth.IvcSecurity.extraction_bound` (`dd0c7f22441b3a9c`)). The bound is a sum of stage errors and collision probabilities, not a negligible function, and the running time of the extractor is an abstract premise. | `SECURITY_MODEL.md` adversary 4 |
| SuperNeo samples each Π_RLC challenge uniformly from the strong set. | Each challenge is five sponge lanes reduced modulo `5^64` (section 5.3), within `2^−173` of uniform. | `decisions/field-2p64-59-ring-x64.md` |
| Nebula Section 4.3 proves `F`, hashes the carried commitments into the challenges, and then proves `F_ops` and `F_scan`. These three proofs are layer 1. In layer 2, `F_final` folds them. | One F′ chain. At the start of a segment, the prover proposes the ops and FS roots, and `η` comes from them, `D_mem`, `plan_digest` and `ts` (section 7.4). The close of the segment makes the `F_final` checks: the chains equal the proposals, the memory continues, and the products balance. | `decisions/nebula-memory-checking.md`, `SECURITY_MODEL.md` trust item 6 |
| Nebula Sections 4.2 and 4.3 take one challenge pair in the base field. | `η1` and `η2` are in `K`. Over `F_q`, the fingerprint error of a geometry at the rule 6 limit is about `2^−45.9`. | `decisions/nebula-memory-checking.md` |
| Nebula Constructions 1 and 2 chain commitments to a split-committed witness. | The record chains are the hash chain of Nebula Section 3.1 over the packed lanes, computed inside F′, as in the generic construction of Section 3.2. | `decisions/nebula-memory-checking.md` |
| In Nebula Section 4.3, a segment has as many steps as the proof, and the scan has that many chunks. | A segment has `N` steps and the scan has `B_scan = (R + M)/N` slots in each step. Idle steps fill the last segment. | `decisions/nebula-memory-checking.md` |
| Nebula Section 4.2 has one memory. | ROM and RAM share one index space. O5 forbids a ROM write, and O6 keeps a ROM address below `R`. | `decisions/nebula-memory-checking.md` |
| Nebula Sections 2.3 and 4.2 take each address, value and timestamp as a field element. | An address has `μ` bits, a value has 32 bits and a timestamp has `W_ts ≤ 62` bits. The geometry bounds the number of operations of a proof: `S_max · N · B_ops < 2^W_ts` (section 7.1). | `Spec/Nebula/Records.lean`, `Spec/Nebula/Plan.lean` |
| Nebula Construction 2 hashes the carried commitment as its own field of the state hash. | The memory carry is in the application-state digest, so the F′ state hash and layout do not change. | `decisions/nebula-memory-checking.md` |

The `F_ops` checks of Nebula are O4 and the write timestamp `ts + cnt_j`, and the `F_scan` address check is the computed cell of a scan slot. The published `F_scan` has no timestamp range check, and Nightstream adds none: one segment is sequentially consistent without it (`Spec.Nebula.segment_sequential` (`b186065f1fe7272e`)).

The scope choices that the papers leave open are records too: the reference application (`decisions/fprime-stage1-poseidon2-hash-chain-v1.md`), the 41-trit opening of field values in binding maps (`decisions/41-trit-encoding.md`) the Poseidon2 instance over `F_q` (`decisions/poseidon2-p64-instance.md`) and the Poseidon2-only rule for protocol binding (`AGENTS.md`, Protocol). The compression of the final proof is such a record too (`decisions/akita-compression.md`). The papers of this section give no compression that Nightstream uses.

## 9. Conformance

An implementation conforms when it reproduces these vectors and passes these checks. `SHA256SUMS` in each artifact folder pins the bytes of the Lean exports. CI runs each check that this table marks "(CI)".

| Vectors | Check | Covers |
|---|---|---|
| `formal/nightstream-fprime/artifacts/` | `python3 -B formal/nightstream-fprime/scripts/check_artifacts.py` (CI) | the digests of the Lean exports and of their copies in `crates/nightstream-fprime/artifacts/` |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-parity-v1.json` | `cargo test -p nightstream-fprime --release` (CI) | Rust's expansion of the light and refresh packages against Lean's row count and checksum, Rust's identity code against Lean's probe packages, the witness program and the logical assignment against Lean (`crates/nightstream-fprime/tests/bounded_depth_packages.rs`) |
| `crates/nightstream/tests/fixtures/poseidon2-hash-chain-v1-digest.json` | `cargo test -p nightstream --release --lib application::` (CI) | the reference application's rows against the row digest of the Lean application (`crates/nightstream/tests/fixtures/README.md`) |
| `crates/nightstream/artifacts/` | `cargo test -p nightstream --release --lib artifacts::` (CI) | each embedded zstd copy against its Lean export (`crates/nightstream/artifacts/README.md`) |
| the light and refresh packages | `cargo test -p nightstream --release` (CI) | the base step of each circuit, light and refresh folds into both slots with the edits of lifecycle rule 3, the byte form of a final proof and of a compression key, and the proof of zero steps (`crates/nightstream/tests/bounded_depth_lifecycle/compression.rs`, `crates/nightstream/tests/bounded_depth_base_steps.rs`, `crates/nightstream/tests/bounded_depth_api.rs`, `crates/nightstream/tests/bounded_depth_lifecycle/mod.rs`). The tests that compress a step of the reference application are ignored, and CI does not run them (lifecycle rules). |
| the light and refresh packages | `cargo test -p nightstream --release -- --ignored`, by hand | final proofs at the cap 2 with the edits of lifecycle rule 3, and the public API with a key that the verifier takes and loads (`crates/nightstream/tests/bounded_depth_lifecycle/compression.rs`, `crates/nightstream/tests/compression.rs`) |
| `crates/neo-compress/artifacts/key20.aks`, `crates/neo-compress/artifacts/key30.aks` | `cargo test -p neo-compress --release` (CI) | the SHA-256 and the rows of each schedule file, the argument of Algorithm 9 with the key of Algorithm 11 on small claims with `n_A = 9`, no expansion of the Ajtai key in Algorithm 10, the key reduction on the block count of the light relation, and a ring equation against its inner product at a point (`crates/neo-compress/tests/unit/akita.rs`, `crates/neo-compress/tests/argument.rs`, `crates/neo-compress/tests/unit/key.rs`, `crates/neo-compress/tests/verifier_reads_no_key.rs`, `crates/neo-compress/tests/ring_equation.rs`) |
| the four bounded-depth memory packages | `cargo test -p nightstream --release --lib nebula::` (CI) | the Rust-built memory relation of the machine against the row digests of both Lean memory applications (`crates/nightstream/tests/fixtures/nebula-memory-v1-digest.json`, `crates/nightstream/tests/fixtures/nebula-memory-n1-digest.json`); the catalog rows; reads past the view. Ignored tests make memory proofs and check the rejection cases that need a bad witness (`crates/nightstream/tests/nebula/parity.rs`, `crates/nightstream/tests/nebula/conformance.rs`) |
| `crates/nightstream/tests/fixtures/nebula-memory-v1-terminal.json`, `crates/nightstream/tests/fixtures/nebula-memory-n1-terminal.json` | `cargo test -p nightstream --release --lib nebula::memory` (CI) | Algorithm 8 against the Lean terminal checks, `D_init` against Lean, and statements that are not canonical or have another word count |
| the WASM application | `cargo test -p neo-wasm --release` (CI) | the catalog rows of a WASM program at 1, 2 and 4 rows in each step, a load that reads a store of an earlier row of its step, and a step of rows that are not neighbors; ignored tests prove it at 1 and 2 rows in each step |

These checks cover their recorded inputs. No theorem covers arbitrary Rust execution (`SECURITY_MODEL.md`, trust item 5).
