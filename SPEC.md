# Nightstream specification

## 1. Status and authority

This document describes the protocol that the `nightstream` crate runs: Stage 1 recursion, which is HyperNova over SuperNeo v1.2 folding, Nebula memory checking (section 7), and the compressed final proof. The Lean packages in `formal/nightstream-fprime/` and `formal/compression-lean/` are normative. This document restates them for readers outside the project, such as an auditor or a second implementer. It is not a second authority. Where the two differ, Lean is correct and this document has a defect.

Rust in `crates/` implements the protocol and is not proved. Lean emits the circuit packages, and Rust embeds them and pins their identifiers. A statement that only Rust fixes is marked "(not checked against Lean)".

Lean proves these results. `formal/nightstream-fprime/SECURITY_MODEL.md` gives the adversary of each result and what you must trust.

- If the rows of the step circuit hold and the public input matches, the typed step holds (`Export.Stage1.ActualPiDECOutput.selectedRowsAndPublic_imply_step` (`3aad9b2575f40242`)). No hardness premise is used.
- If the terminal predicate accepts, the opened step is the authenticated NIFS step, or the base step, or a named Poseidon2 state-hash collision occurred (`Export.Stage1.ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision` (`455f5269f7b739ba`)).
- One fold is knowledge sound in the random-oracle model (`Export.Stage1.RandomOracleSetup.production_knowledge_error_lt` (`709bb7698524ce34`)).
- A false history is accepted with probability at most the sum of the per-stage errors, under HyperNova errata Assumption 1 (`Export.Stage1.HyperNovaFalseAcceptance.probability_bound` (`4106ba4f56189715`), `Export.Stage1.HyperNovaVisitedSecurity.history_failure_le` (`ea7ead2ec7b3627d`)). With the errors that the random-oracle result motivates, this bound is useful for one or two folds only (`SECURITY_MODEL.md`, trust item 4).
- The honest prover's first proof is accepted, and an accepted proof gives an accepted next proof, for a witness that the application accepts (`Export.Stage1.HyperNovaInitial.initial_accepted` (`cc4df317635879dd`), `Export.Stage1.HyperNovaAcceptedNext.base_extend` (`d7d06d7ddc3a8971`), `Export.Stage1.HyperNovaAcceptedNext.recursive_extend` (`b237776203d41507`)).
- If the final verifier accepts, the F′ claim of the last fold is true, except with probability `calls / 2^t` in the state-restoration experiment at Rust's layer bits `t` (`CompressionLean.Claims.production_bound` (`ab6fab37dd9e07c1`)). For the reference application, each premise of this result is met by a Lean proof, a Lean check of Rust's files, Rust, or a Lean value that compiled code evaluates and the kernel does not check (`CompressionLean.Claims.poseidon2_bound` (`3974c561a95f556a`)). For another application, four premises are open: the parent shape, the honest setup, and that the setup fits and that the key describes it (`formal/compression-lean/STATUS.md`). Idealized WHIR openings, Poseidon2 as a random oracle, Fiat–Shamir inside the final program and a key root that is the honest root of the claims' setup are assumptions (`formal/compression-lean/TRUST.md`, sections 3 and 5).
- If the memory verifier accepts, the history that the reverse extractor returns is one execution of the claimed memory statement, with memory that is sequentially consistent. The exceptions are the history terms of the fourth result, Poseidon2 collisions in the extracted run, and the error of the memory-round premise (`Export.Stage1.NebulaMemorySecurity.bad_le` (`b2c519e88f779c50`); `SECURITY_MODEL.md`, adversary 5 and trust item 7). Each honest execution has a history that the memory verifier accepts (`Lifecycle.Nebula.MemoryApp.history_of_execution` (`a1ef468b8bc6e701`)).

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
| 2, notation | `Spec/Algebra.lean`, `Spec/Phi81StrongSet.lean`, `Spec/Folding/Nifs/PaperNonInteractive/Types.lean` |
| 3, parameters | `Spec/Profile.lean`, `Export/Package.lean` |
| 4, encodings | `Lifecycle/XOut.lean`, `Lifecycle/VerifierContext.lean`, `Lifecycle/Stage1/VerificationKey.lean` |
| 5, transcript | `Spec/Poseidon2.lean`, `Lifecycle/Transcript.lean`, `Lifecycle/ProductionKey.lean` |
| 6, algorithms | `Spec/AjtaiSetupV1.lean`, `Spec/Folding/Nifs/PaperNonInteractive/Verifier.lean`, `Spec/HyperNova/Construction2/Paper.lean`, `Lifecycle/Stage1/`, `Final/` |
| 7, Nebula memory | `Spec/Nebula/`, `Lifecycle/Nebula/`, `Export/Stage1/NebulaMemorySecurity.lean` |

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

### 2.4 The relation

- The step relation is a CCS over `F_q` with `t = 4` matrices `M_0, …, M_3`. Each matrix has its rows on the cube `{0,1}^ℓ` and `m` columns, and only its first `n` rows are nonzero (`Lifecycle.ProductionKey.LogicalRelation`). The constraint polynomial `f` is in section 3 (`Spec.ProductionRelation.polynomial`). An assignment `z` satisfies the CCS when `f((M_0 z)_r, …, (M_3 z)_r) = 0` for every row `r`.
- The public input of `z` is its first 270 entries, five ring elements (`Spec.projectPublicInput`, `Lifecycle.PaperAlgebra.publicRingColumns`).
- `eq(x, y) = ∏_i (x_i y_i + (1 − x_i)(1 − y_i))` for `x, y ∈ K^ℓ`. `ỹ(r)` is the multilinear extension of a table `y` on `{0,1}^ℓ`, evaluated at `r ∈ K^ℓ`.

### 2.5 Claims and the NIFS proof

The NIFS folds one fresh claim and 16 running claims into 16 running claims (`Spec.Folding.Nifs.PaperNonInteractive.Fresh`, `Spec.Folding.Nifs.PaperNonInteractive.Running`). The sources are in the order fresh first, then running 0 to 15.

- A fresh claim is a CCS claim: a commitment `c ∈ R_q^κ` and a public input `x ∈ F_q^270`. Its witness satisfies the CCS, opens `c`, projects to `x` and has norm below `b`.
- A running claim is a CE claim (SuperNeo v1.2 Definition 21): a commitment, a public input and an evaluation family at the shared point `r ∈ K^ℓ`. The family has `Eval_K`, 54 elements of `K` (the Pad evaluations), and `Eval_A`, `4 × 54` elements of `K` (the matrix evaluations). The running claims are Π_DEC children, so a witness opens the commitment, projects to the public input, has norm below `b`, and gives these evaluations at `r` (`Lifecycle.runningStatement`). The Π_RLC parent of Algorithm 4 is a CE claim with norm below `B`.
- The default running claim `u_⊥` has the point 0 and zero commitments, public inputs and evaluations (`Lifecycle.defaultRunning`, HyperNova Definition 12).
- A NIFS proof has, in order: `ℓ` sum-check round polynomials of degree at most 8, each its 9 coefficients in `K` from degree 0 up; the Π_CCS output, one evaluation family for each of the 17 sources; and the 16 Π_DEC children, each a commitment and an evaluation family (`Spec.Folding.Nifs.PaperNonInteractive.Proof`).

## 3. Parameters

The table below is generated from Lean. Its values are the integer equations of `Spec.production_parameter_values` (`f9cfc75c6a143482`), which `decide` proves, of three more Lean theorems, and of the header of the Lean-emitted package `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json`. The names of the header fields come from their Lean formats in `Export/Package.lean`. The package is the one of `Poseidon2HashChainV1`, the reference application (`decisions/fprime-stage1-poseidon2-hash-chain-v1.md`). Another application has another package, with the same profile but other sizes (`decisions/fprime-stage1-per-application-packages.md`).

The profile's shape theorem gives the same values for the folding scheme: `ℓ = 28`, 4 matrices, `d = 54`, 1 fresh and 16 running claims, a public width of 270, `κ = 22` and a round degree bound of 8 (`Lifecycle.NifsProfile.selected_shape` (`410b7cb6db537cb7`)).

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
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.piRlcInputs` | — | 17 | The Π_RLC inputs: one fresh claim and the running claims. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.piDecChildren` | — | 16 | The Π_DEC children, which are the next running claims. |
| `Spec/Profile.lean`, `production_parameter_values` | `productionProfile.ccsMatrices` | `t` | 4 | The CCS matrices. |
| `Spec/Profile.lean`, `production_norm_stages` | `NormStage.bound productionGlobalParams .fresh` | — | 2 | The norm bound of a fresh witness, `b`. |
| `Spec/Profile.lean`, `production_norm_stages` | `NormStage.bound productionGlobalParams .combined` | — | 65536 | The norm bound of a Π_RLC parent, `B`. |
| `Spec/Profile.lean`, `production_norm_stages` | `NormStage.bound productionGlobalParams .ambient` | — | 9223372034707292161 | The ambient bound, `q/2 + 1`. |
| `Spec/Profile.lean`, `production_msis_norm_bound` | `productionGlobalParams.msisNormBound` | — | 113246208 | The MSIS norm of the binding reduction, `8TB`. |
| `Lifecycle/Nifs/SelectedTestNumerator.lean`, `numerator_eq` | `numerator` | — | 4589 | The numerator of the Π_CCS test error `ε_test`. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `sealed.schema` | — | 6 | The schema of the sealed package file. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.schemaVersion` | — | 8 | The schema of the circuit package. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.poseidon.width` | — | 16 | Poseidon2 state width. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.poseidon.rate` | — | 12 | Poseidon2 rate. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.poseidon.digestLength` | — | 4 | Poseidon2 digest length. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.poseidon.initialFullRounds` | — | 4 | Poseidon2 full rounds before the partial rounds. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.poseidon.partialRounds` | — | 22 | Poseidon2 partial rounds. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.poseidon.terminalFullRounds` | — | 4 | Poseidon2 full rounds after the partial rounds. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.poseidon.recipesPerPermutation` | — | 1096 | Witness recipes of the permutation template. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.poseidon.outputLocalStart` | — | 1080 | The first output local of the permutation template. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.layout.rowCount` | — | 11383401 | Rows of the physical layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.layout.privateColumnCount` | — | 11470080 | Private columns of the physical layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.layout.constantColumn` | — | 11470080 | The constant column of the physical layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.layout.publicColumnCount` | — | 278 | Public columns of the physical layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.layout.totalColumnCount` | — | 11470359 | All columns of the physical layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.relation.rowCount` | `n` | 1004131 | Rows of the CCS relation. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.relation.columnCount` | `m` | 43963750 | Columns of the CCS relation, the logical width. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.relation.cubeVariables` | `ℓ` | 28 | Variables of the row cube: `n ≤ 2^ℓ`, and Π_CCS has `ℓ` rounds. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.relation.degreeBound` | — | 8 | The degree bound of a Π_CCS round polynomial. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.relation.terms` | `f` | M_0·M_1 − M_2 + M_3^7 | The CCS polynomial over the matrix images `M_j z`. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.terminal.rowStart` | — | 0 | The first row of the terminal layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.terminal.rowCount` | — | 1004131 | Rows of the terminal layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.terminal.runningClaims` | — | 16 | Running claims of the terminal layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `package.terminal.freshClaims` | — | 1 | Fresh claims of the terminal layout. |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | `sealed.logicalPublicInputCount` | — | 270 | The public input of a claim, in field elements. |
| computed | `ε_test` | `ε_test` | 4589/q² ≈ 2^−115.83 | The Π_CCS test error, `numerator / q²` (`test_error_eq`). |

A probability is shown as `2^−x`, where `x` is `−log₂` of the exact fraction rounded down to two decimals, so the shown bound is never below the true value.
<!-- /params -->

The error terms are these. The Π_CCS test error `ε_test` is `(ℓ · 8 + 4365)/q² = 4589/q²` (`Lifecycle.Nifs.VerifierErrorBudget.test_error_eq` (`4826645c11e6724e`)), where 4365 is `4338 − 1 + ℓ` and 4338 is the number of joint coefficients (`Spec.Folding.PiCCS.PaperJoint.Shape.jointCoefficientCount`). The law of each Π_RLC challenge is within `2^−132` of the uniform law on `C` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt` (`ed2b1cc8c235a81e`)). In the random-oracle result, each query adds at most `50/(87 · 2^115)` to the statistical error (`Lifecycle.RandomOracleKnowledge.statisticalError_le` (`116e4b448e8d7136`)). The setup of the key from SHAKE128 chunks adds at most `2^−190` (`SECURITY_MODEL.md`, adversary 3). The Rust estimator of `nightstream` reports 114 bits for this profile. That is a statistical estimate, not a claim about the complete system (`crates/nightstream/README.md`; not checked against Lean).

## 4. Encodings

### 4.1 Words

Lean hashes and absorbs field elements, called words. A natural number `n` becomes the word `n mod q` (`Spec.Poseidon2.ofNat`).

- Bytes become one word per 8 bytes, little-endian, reduced modulo `q` (`Lifecycle.packBytes`).
- An element `c_0 + c_1 u` of `K` is the two words `c_0, c_1` (`Lifecycle.serializeK`).
- A ring element is its 54 coefficients in order (`Lifecycle.serializeRingF`). A commitment is its `κ` ring elements in row order, `κd = 1188` words (`Lifecycle.serializeCommitment`).
- A public input is its 270 words (`Lifecycle.serializePublicInput`). A point of `K^ℓ` is its coordinates in order, `2ℓ` words (`Lifecycle.serializePoint`).
- An evaluation family is `Eval_K`, 54 elements of `K`, then `Eval_A` matrix by matrix, each 54 elements of `K`: `2 · 5 · d = 540` words (`Lifecycle.serializeEvaluations`).
- A framed list is its length as one word, then the list (`Lifecycle.VerifierContext.framed`).

### 4.2 The state-hash preimage

The state hash of HyperNova Construction 2 binds the verifier context, the step counter, the initial and current states and the running claims (`Lifecycle.serializePreimage`, `Lifecycle.stateHash`). A state is four words (`Lifecycle.Stage1.Application.stateWordCount`).

| Part | Words | Content |
|---|---|---|
| domain | 12 | the ASCII bytes of `HyperNova/NIVC/state/v2`, packed 8 bytes to a word (`Lifecycle.stateDomainChunk`) |
| commitments | `16κd` | the 16 running commitments |
| `Eval_K` | `16 · 2d` | the 16 running Pad evaluations |
| `Eval_A` | `16 · 4 · 2d` | the 16 running matrix evaluations |
| point | `2ℓ` | the shared point `r` |
| parent public input | 90 | the recomposed public input `Σ_j 2^j x_j` of the 16 children, three 17-bit lanes to a word (`Lifecycle.serializeParentPublic`) |
| `vk` | 4 | the verifier-context digest (Algorithm 2) |
| `i` | 1 | the step counter |
| `z0` | 4 | the initial state |
| `zi` | 4 | the current state |

At `κ = 22`, `d = 54` and `ℓ = 28` the preimage has 27,819 words (computed). The packed word is `low + 2^17·middle + 2^34·high` (`Lifecycle.packWord`). Each lane value is in `(−2^16, 2^16)` when the children are the canonical split of the parent (`Lifecycle.ChildrenCanonical`). The verifier recomputes this hash from the running claims, so it rejects children that are not that split (`SECURITY_MODEL.md`, adversary 3).

### 4.3 The public input of a fresh claim

A fresh claim's public input encodes a four-word digest in 270 cells (`Lifecycle.encHash`, HyperNova Definition 12). Cell 0 is 1 (`Lifecycle.encHash_marker` (`d04bb6b10eebcdff`)). Cell `1 + 64w + k` is bit `k` of digest word `w`, for `w < 4` and `k < 64`. Cells 257 to 269 are 0. The verifier decodes the digest from bits 1 to 256 (`Lifecycle.decodeHash`).

### 4.4 The verifier context and the verification key

The verifier context binds the relation, the application, the NIFS key and the commitment-key authority (`Lifecycle.VerifierContext.Authority`, `Export.Stage1.PerApplicationCanonicalPackage.verifierContextDescriptor`). Component `i` has the digest `Poseidon2.hash(componentDomain(i) ++ framed(words_i))`, where `componentDomain(i)` is the ASCII bytes of `Nightstream/FPrime/context/v1_1`, one word each, then `i` (`Lifecycle.VerifierContext.componentDigest`). The components are 1 for the relation, 2 for the application, 3 for the NIFS key and 4 for the commitment key.

The descriptor is these words, and its digest `vk` is their Poseidon2 hash (`Lifecycle.VerifierContext.Descriptor.serialize`, `Lifecycle.VerifierContext.digest`). Each byte of a label is one word.

1. The 40 bytes of `Nightstream/FPrime/verifier-context/v1_1`.
2. Framed `profileWords`: `2^32 − 1, 1, 2, 16, 65536, 1, 16, 17, 16, 4, 28, 8, 54, 22` (`Lifecycle.VerifierContext.profileWords`).
3. Framed `scheduleWords`: `1, 1, 1, 28, 9, 17, 4, 54, 4, 1` (`Lifecycle.VerifierContext.scheduleWords`).
4. The four framed component digests, in component order.

Two descriptors with one serialization are equal (`Lifecycle.VerifierContext.Descriptor.serialize_injective` (`86735804b38f487e`)). The verification key binds the package identity, the descriptor and a formulas digest under the domain `Nightstream/FPrime/verifier-key/v1`, in 131 words (`Lifecycle.Stage1.VerificationKey.Binding.serialize`, `Lifecycle.Stage1.VerificationKey.Binding.serialize_length` (`4b622447d14368c7`)).

The commitment-key authority is 73 words: 37, the 37 bytes of the setup identifier, `κ`, the number of key columns, 32, and the 32 seed bytes (`Spec.AjtaiSetupV1.Setup.authorityWords`, `Spec.AjtaiSetupV1.Setup.authorityWords_length` (`22e8a499de205f29`)).

### 4.5 Proof bytes

These layouts are Rust's (not checked against Lean). Every word is 8 bytes, little-endian, and a field word must be below `q`.

A Stage 1 proof (`crates/nightstream/src/lifecycle/encoding.rs`) is:

| Part | Size | Content |
|---|---|---|
| tag | 16 | `NS-STAGE1-PROOF1` |
| kind | 8 | 0 for the initial proof, 1 for an active proof |
| initial proof | 32 | `z0` |
| active: state | 72 | `i`, `z0`, `zi` |
| active: running claims | `16 · 8(κd + 270 + 2ℓ + 10d)` | each claim's commitment, public projection, point, `Eval_K` and `Eval_A` |
| active: running witnesses | `16 · 16⌈m/d⌉` | per column, a 64-bit mask of the `+1` coefficients, then of the `−1` coefficients |
| active: fresh claim | `8(κd + 270) + 16⌈m/d⌉` | the commitment, the public input and the witness masks |

The decoder checks the exact length that the circuit fixes before it reads, and it rejects a word not below `q` and a malformed witness mask. Each accepted proof has one encoding.

A final proof (`FinalProof::to_bytes`, `crates/nightstream/src/lifecycle/finish.rs`) is the tag `NS-FINAL-PROOF02`, `i`, `z0` and `zi` as nine words, then the shrink proof (`crates/neo-spartan/src/shrink/mod.rs`). A compression key is its `bincode` encoding, and a decoder accepts only the canonical encoding.

## 5. Transcript

### 5.1 The permutation and the hash

The hash is Poseidon2 over `F_q` with width 16 and the S-box `x^7` (`Spec.Poseidon2.permute`). It has 4 full rounds, 22 partial rounds and 4 full rounds, with Plonky3's round constants and internal diagonal (`Spec.Poseidon2.initialConstants`, `Spec.Poseidon2.internalDiagonal`). The state is 16 words, and the rate is 12.

- To absorb a block, add its words to lanes 0 to 11, then permute (`Spec.Poseidon2.absorbBlock`).
- `Poseidon2.hash(words)` starts from the zero state, absorbs the words in blocks of 12, the last one short, then adds 1 to lane 0, permutes and outputs lanes 0 to 3 (`Spec.Poseidon2.hash`). It does not bind the length.

### 5.2 The fold transcript

The NIFS verifier and the step circuit run one Poseidon2 sponge (`Lifecycle.Transcript`). To absorb a list, add it in blocks of 12, each followed by one permutation (`Lifecycle.Transcript.absorb`). The profile fixes every absorbed length, so the stream has one parse without framing (`decisions/piccs-prior-state-digest.md`).

A challenge in `K` at position `p` reads lanes `2(p mod 6)` and `2(p mod 6) + 1` as `c_0, c_1`. After the read at a position with `p mod 6 = 5`, the sponge absorbs a zero block (`Lifecycle.Transcript.squeezeAt`). Coordinate `j` of `α` has position `j`, `γ` has position `ℓ`, and every round challenge has position 0 (`Lifecycle.Transcript.coinPosition`).

| # | Operation | Words or reads | Value |
|---|---|---|---|
| 1 | absorb | 12 | the ASCII bytes of `Nightstream/SuperNeo/fold/v2`, packed 8 bytes to a word (`Lifecycle.Transcript.foldDomainChunk`) |
| 2 | absorb | `4 + κd + 270` | the prior digest decoded from the fresh public input, the fresh commitment, the fresh public input (`Lifecycle.ProductionKey.absorbPublicInput`) |
| 3 | read | `ℓ` | `α_0, …, α_{ℓ−1}` at positions 0 to `ℓ − 1` |
| 4 | read | 1 | `γ` at position `ℓ` |
| 5 | for each round `j < ℓ`: absorb | 18 | the 9 coefficients of round polynomial `j` (`Lifecycle.Transcript.serializeMessage`) |
| 6 | then read | 1 | the round challenge `r'_j` at position 0 |
| 7 | absorb | `17 · 10d` | the Π_CCS output, source by source (`Lifecycle.ProductionKey.absorbFullOutput`) |
| 8 | for each source `s < 17`: Π_RLC | 4 lanes | `ρ_s` (section 5.3) |

The running claims do not enter the sponge directly. They enter through the prior digest in the fresh public input, which is their state hash (Algorithm 3). Distinct challenges read distinct addresses, a call list and a slot (`Lifecycle.TranscriptCoverage.challengeAddress_injective` (`6fac00b3edab0838`)). The state hash shares no call list with a challenge, because their first inputs absorb different domain blocks (`Lifecycle.TranscriptCoverage.stateHash_separated` (`8aad409286c506c0`)).

### 5.3 The Π_RLC sampler

Let `S_0` be the sponge state after row 7. For `s < 17`, absorb the block `[4, s]` into `S_s` to get `E_s` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.enter`). Read lanes 0 to 3 of `E_s` as `x_0, …, x_3`, and set `S_{s+1} = permute(E_s)` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.Transcript.stateAt`).

1. Set `N = Σ_{k<4} x_k · q^k` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.drawIndex`) and `R = N mod 5^54` (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.reduce`).
2. Coefficient `h` of `ρ_s` is `⌊R / 5^h⌋ mod 5`, minus 2 (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.sample`, `Spec.Phi81StrongSet.embedScalar`).

The sampler cannot fail (`Lifecycle.ProductionKey.piRlcResponse_ne_none` (`d42d0844252c6562`)), and each `ρ_s` is in `C` (`Lifecycle.Transcript.PiRlcSampler.piRlcChallenges_member` (`3c3147745741757a`)).

## 6. Algorithms

### Algorithm 1. Commitment-key setup

**Input.** A 32-byte seed, `κ` and the number of key columns `w = ⌈m/d⌉`.

**Output.** The Ajtai key `A ∈ R_q^{κ × w}`.

**Steps.**

1. For each row `i < κ` and column `j < w`, let `in` be the 37 ASCII bytes of `nightstream-ajtai-shake128-wide256-v1`, then the seed, then `i` as 4 bytes and `j` as 8 bytes, little-endian (`Spec.AjtaiSetupV1.elementInput`).
2. Read the first `4d` 64-bit lanes of SHAKE128 of `in` (`Spec.AjtaiSetupV1.elementLanes`).
3. Set coefficient `h` of `A_{i,j}` to the 256-bit little-endian value of lanes `4h` to `4h + 3`, reduced modulo `q` (`Spec.AjtaiSetupV1.wideCoefficientNat`).

**Rejects.** None.

**Source.** `Spec.AjtaiSetupV1.Setup.verifierKey`. Each coefficient is one SHAKE128 chunk reduced modulo `q` (`Spec.AjtaiSetupV1.Programming.verifierKey_eq` (`59e988e0131c808d`)). The production seed is `fc404984d44c1b878d68a6a80092d7d7ab44d81ac17b45a8e7bd4c1f1e371702` (`Export.Stage1.Poseidon2HashChainV1SetupAuthority.productionSeed`). Records `decisions/fprime-stage1-main-ajtai-setup.md` and `decisions/fprime-ajtai-shake128-setup.md`.

### Algorithm 2. Verifier context

**Input.** The package of one application.

**Output.** The verifier-context digest `vk`.

**Steps.**

1. Read the words of the relation, the application, the NIFS key and the commitment-key authority from the package (`Export.Stage1.PerApplicationCanonicalPackage.verifierContextDescriptor`).
2. Compute the descriptor and its digest (section 4.4).

**Rejects.** None.

**Source.** `Lifecycle.VerifierContext.digest`. Every state hash carries `vk`, so a proof for one application does not verify for another unless a state-hash collision occurs (`SECURITY_MODEL.md`, Applications and setups). Record `decisions/fprime-stage1-per-application-packages.md`.

### Algorithm 3. The step relation F′

Each step proves one instance of HyperNova Construction 2's augmented step (`Lifecycle.StepHoldsFor`, `Spec.HyperNova.Construction2.Paper.FixedAugmentedTransition`).

**Input.** The step counter `i`, the states `z0` and `zi`, the running claims `U_i`, a fresh claim `u_i`, a NIFS proof, and the application witness `w`.

**Output.** The next state `z_{i+1}`, the running claims `U_{i+1}`, and the digest `x` that the next fresh claim's public input encodes.

**Steps.**

1. Require that the application accepts the witness `w` at `zi`, and set `z_{i+1} = F(zi, w)`. `F` is the application's step, and the acceptance check is its `valid` predicate (`Lifecycle.Stage1.Application.Program`).
2. If `i = 0`: require `z0 = zi`, and set `U_{i+1}` to 16 copies of `u_⊥`.
3. If `i > 0`: require that the public input of `u_i` is `encHash(stateHash(vk, i, z0, zi, U_i))`. Then require that Algorithm 4 accepts `(U_i, u_i, proof)`, and set `U_{i+1}` to its output.
4. Set `x = stateHash(vk, i + 1, z0, z_{i+1}, U_{i+1})`.

**Rejects.** The relation does not hold.

**Source.** The step circuit checks these steps in eight children, in this order: the prior state hash, the output state hash, Π_CCS, Π_RLC, Π_DEC, the running transition, the application and the next preimage (`Lifecycle.Stage1.Formal`, `formal/nightstream-fprime/README.md`). Its rows imply the typed step (`Export.Stage1.ActualPiDECOutput.selectedRowsAndPublic_imply_step` (`3aad9b2575f40242`)). For the reference application, `F(z, w) = Poseidon2.hash(tag ++ z ++ w)`, where `tag` is the 40 bytes of `Nightstream/Stage1/Poseidon2HashChain/v1`, one word each (`decisions/fprime-stage1-poseidon2-hash-chain-v1.md`). HyperNova Construction 2, steps 4 and 5.

### Algorithm 4. NIFS verifier

**Input.** The running claims `U = (r, c_s, x_s, y_s)` for `s < 16`, the fresh claim `(c, x)`, and a NIFS proof.

**Output.** The next running claims, or reject.

**Steps.**

1. Order the 17 sources fresh first, then running 0 to 15. Run rows 1 to 6 of section 5.2 to get `α`, `γ` and `r' = (r'_0, …, r'_{ℓ−1})`.
2. Compute the initial claim `c_0 = Σ γ^{s + 16h} · Eval_K(y_s)_h + Σ γ^{864 + s + 16j + 64h} · Eval_A(y_s)_{j,h}`, over `s < 16`, `j < 4` and `h < 54` (`Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial.VerifierInput.initial`, `Spec.Folding.PiCCS.PaperJoint.PadCoordinate.gammaExponent`, `Spec.Folding.PiCCS.PaperJoint.MatrixCoordinate.gammaExponent`).
3. For each round `j`, require `g_j(0) + g_j(1) = c_j`, and set `c_{j+1} = g_j(r'_j)` (`Spec.SumCheck.Finite.FixedPhase.checkChain`).
4. Require `c_ℓ = V`, where `V` is computed from the Π_CCS output at `r'` (`Spec.Folding.PiCCS.PaperJoint.ProtocolPolynomial.terminalFromMessage`):

   ```text
   V = eq(r', r) · Σ_coord γ^e · Pad_coord
     + γ^864 · eq(r', r) · Σ_coord γ^e · Mat_coord
     + γ^4320 · eq(r', α) · ( f(M_0 z_0, …, M_3 z_0) + γ · Σ_{s<17} γ^s · N(z_s) )
   N(y) = (y + 1) · y · (y − 1)
   ```

   Here `Pad_coord`, `Mat_coord`, `M_j z_0` (the fresh source) and `z_s` are the values that the output claims at `r'`, and `e` is the local exponent of each coordinate (`Spec.Folding.PiCCS.PaperJoint.StrongReduction.Statement.projectOutput`). 864 is the Pad count `16d`, and 4320 is `16d · 5` (`Spec.Folding.PiCCS.PaperJoint.Shape.constraintOffset_eq` (`395ce359b5e699ad`)).
5. Absorb the Π_CCS output (row 7), and derive `ρ_0, …, ρ_16` (section 5.3).
6. Compute the parent `P = (r', Σ_s ρ_s c_s, Σ_s ρ_s x_s, Σ_s ρ_s y'_s)` over the 17 sources, where `c_s` and `x_s` are the source's commitment and public input and `y'_s` is its family in the Π_CCS output (`Spec.Folding.PiRLC.combinedOutput`, SuperNeo v1.2 Section 7.4).
7. Split the parent's public input into 16 digit vectors `x_0, …, x_15` with `x_P = Σ_k 2^k x_k` (`Lifecycle.PaperAlgebra.publicInputSplit`). Reject if it does not split.
8. Read the 16 children's commitments `c_k` and evaluation families `y_k` from the proof. Require `c_P = Σ_k 2^k c_k` and `y_P = Σ_k 2^k y_k` (`Spec.Folding.PiDEC.Accepted`, SuperNeo v1.2 Section 7.5).
9. Output the children `(r', c_k, x_k, y_k)` as the running claims.

**Rejects.** The verifier rejects when step 3, 4, 7 or 8 fails (`Spec.Folding.Nifs.PaperNonInteractive.verify`).

**Source.** `Spec.Folding.Nifs.PaperNonInteractive.verify` at `Lifecycle.ProductionKey.key`. It accepts exactly when the Π_CCS check and the Π_DEC check pass and the output exists (`Spec.Folding.Nifs.PaperNonInteractive.verify_eq_some_iff` (`542d30511c2cbf9e`), `Lifecycle.Stage1.Accumulator.holds_iff_checks` (`49e83160b0a7dee6`)). SuperNeo v1.2 Sections 7.3 to 7.5. Record `decisions/padded-row-identity-piccs.md`.

### Algorithm 5. Prove and extend

**Input.** For `prove`, the initial state `z0` and the application witness. For `extend`, a proof at step `i` and the next application witness.

**Output.** A Stage 1 proof at step `i + 1`.

**Steps.**

1. Fill the witness of Algorithm 3 for step `i`. At `i = 0` the running claims are `u_⊥` and the NIFS proof is zero, and the base branch does not read it.
2. Commit the step witness. Its commitment and public input are the fresh claim of step `i + 1`.
3. At `i > 0`, fold `U_i` and `u_i` by the NIFS prover, so that Algorithm 4 accepts. Keep the 16 children and their witnesses as `U_{i+1}`.
4. The proof is the state `(i + 1, z0, z_{i+1})`, the running claims with their witnesses, and the fresh claim with its witness.

**Rejects.** None. A prover that fails returns an error and leaves the supplied proof unchanged.

**Source.** `crates/nightstream/src/lifecycle/` (not checked against Lean). Lean proves that the honest step is accepted when the application accepts its witness (`Export.Stage1.HyperNovaInitial.initial_accepted` (`cc4df317635879dd`), `Export.Stage1.HyperNovaAcceptedNext.base_extend` (`d7d06d7ddc3a8971`), `Export.Stage1.HyperNovaAcceptedNext.recursive_extend` (`b237776203d41507`)).

### Algorithm 6. Terminal verifier

**Input.** The expected state `(i, z0, zi)` and a Stage 1 proof.

**Output.** Accept or reject.

**Steps.**

1. If the proof is the initial proof, accept exactly when `i = 0` and `zi = z0`.
2. Require `0 < i < q`.
3. Require that the public input of the fresh claim is `encHash(stateHash(vk, i, z0, zi, U))`, where `U` is the proof's running claims.
4. Require that each running witness opens its claim: the commitment, the public projection, norm below `b`, and the `Eval_K` and `Eval_A` values at the shared point.
5. Require that the fresh witness satisfies the CCS, opens the fresh commitment and projects to the fresh public input.

**Rejects.** Each step's failure. Rust checks them in this order: the state, the counter, the claim shapes, the public projections, the state hash, the commitments, then the openings and the fresh relation (`crates/nightstream/src/lifecycle/verify.rs`; not checked against Lean).

**Source.** `Lifecycle.Stage1.Terminal.HoldsFor`, the terminal check of HyperNova Construction 2 (`Spec.HyperNova.Construction2.Paper.OuterTerminalTransition`). The initial proof is accepted exactly at `i = 0` and `zi = z0` (`Lifecycle.Stage1.Terminal.holdsFor_bottom_iff` (`8838d0b9d5350a57`)). Acceptance implies the authenticated NIFS step, the base step or a state-hash collision (`Export.Stage1.ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision` (`455f5269f7b739ba`)).

### Algorithm 7. Finish

**Input.** An active Stage 1 proof and the compression setup of its circuit.

**Output.** A final proof.

**Steps.**

1. Run Π_CCS and Π_RLC of Algorithm 4 once more on the proof's running and fresh claims, and stop before Π_DEC. The result is one CE claim with norm bound `B`, the parent (SuperNeo v1.2 Lemma 1).
2. Prove the parent claim with layer 1, a sum-check and WHIR argument (`crates/neo-spartan/`).
3. Prove with the shrink layer, SuperSpartan and one WHIR commitment, that the final program accepts the terminal package's rows and the layer-1 transcript for the state.
4. The final proof is the state and the shrink proof. It carries no witness and cannot be extended.

**Rejects.** None, with one exception. A WHIR query sample of `p − 1` fails Algorithm 8, so `finish_with_spartan` returns an error and no proof (lifecycle rule 2). This has a probability of about `2^−64` for each query draw. The proof is deterministic, so a new call at the same state fails again, but the prover can extend one step and then finish.

**Source.** `crates/nightstream/src/lifecycle/finish.rs` (not checked against Lean). The terminal package is the F′ rows up to Π_RLC, without change (`CompressionLean.Terminal.circuitPackage_implies_terminal` (`575cb76bc24d795c`), `Terminal/Data.lean`).

### Algorithm 8. Final verifier

**Input.** The expected state `(i, z0, zi)`, a compression key and a final proof. The key comes from the verifier's circuit or a pinned copy, never from a prover.

**Output.** Accept or reject.

**Steps.**

1. Require that the key is the key of the verifier's circuit and meets the verifier's security minimum.
2. Require that the proof's state is the expected state, and `0 < i < q`.
3. Verify the shrink proof for the final program of the key and the state.

**Rejects.** Each step's failure. In step 3, a WHIR query sample of `p − 1` is a failure, where Plonky3 draws again. This check is stricter than Plonky3 and affects only completeness, not soundness (`crates/neo-spartan/src/whir/mod.rs`; the Lean final program makes the same check, `Sim/Whir.lean`).

**Source.** `CompressionKey::verify` and `Verifier::verify_final` (`crates/nightstream/src/lifecycle/finish.rs`, `crates/nightstream/src/circuit.rs`; not checked against Lean). Layer 1 and the shrink layer run in one state-restoration experiment, so one bad-set argument covers both. At Rust's layer bits `t`, it bounds an accepted false F′ claim by `calls / 2^t` (`CompressionLean.Final.final_restore_budget_parent` (`36f3e4c5d4c18992`)). That theorem has the premise that the shrink CCS encodes the final program, which is the terminal package and the layer-1 verifier. Lean proves this premise for the production final program (`Sim/Encodes.lean`), and the claim joins the two (`CompressionLean.Claims.production_bound` (`ab6fab37dd9e07c1`)). Its premises about the key and the setup are met for the reference application only (`CompressionLean.Claims.poseidon2_bound` (`3974c561a95f556a`), `formal/compression-lean/STATUS.md`). Three things are assumed, not proved: the bound includes no WHIR error; it covers a valid program witness only where its in-program challenges are the experiment's oracle answers (the read condition); and the key's root is the honest root of the claims' setup (`formal/compression-lean/TRUST.md`, sections 3 and 5).

### Lifecycle rules

These four rules join Algorithms 5 to 8. Rust keeps them, and no theorem covers them (not checked against Lean). The tests check them on fixed inputs. CI does not run a test marked "ignored", because it runs the full production profile.

1. **Each returned proof verifies.** Algorithm 6 accepts each proof that `prove` or `extend` returns, at the proof's own state. Before it folds, `extend` checks the counter, the state hash in the fresh public input and the running children. After the fold, it runs Algorithm 4 on its own NIFS proof (`crates/nightstream/src/lifecycle/extend.rs`, `crates/nightstream/src/lifecycle/step_inputs.rs`). Tests: `poseidon_base_step_matches_lean_and_verifies` and `poseidon_metal_recursive_lifecycle` in `crates/nightstream/tests/circuit_lifecycle.rs` (ignored).
2. **Each returned final proof verifies.** Algorithm 8 accepts each final proof that `finish_with_spartan` returns, with the key of the setup that made it, for a verifier of the same circuit and security minimum. `finish_with_spartan` checks that key against its circuit first. Before it returns, it verifies its own shrink proof, so a device fault gives an error and not a bad proof (`crates/nightstream/src/lifecycle/finish.rs`). Tests: `poseidon_finish_with_spartan_verifies` and `poseidon_metal_finish_with_spartan_verifies` in `crates/nightstream/tests/circuit_lifecycle.rs` (ignored, and they need the setup files of about 29 GB). `committed_final_proof_verifies_and_rejects_changes` (CI) checks one committed final proof.
3. **An edited proof fails, and `extend` does not carry the edit forward.** Algorithm 6 rejects each of 41 changes to a two-step proof. A change to the fold digests, which Algorithm 6 does not read, changes nothing. Algorithm 8 rejects a final proof after a changed byte, with a wrong state and with a wrong key. On each engine, `extend` refuses a proof whose fresh commitment was edited, at the Π_DEC commitment check of its replay. Tests: `terminal_rejects_every_authoritative_envelope_change` in `crates/nightstream/tests/lifecycle_native/terminal_sweep.rs` (ignored), `committed_final_proof_verifies_and_rejects_changes` (CI), and `crates/nightstream/tests/extend_edited_proof.rs` (ignored, CPU and Metal).
4. **A proof that verifies can be extended and finished, except at the last counter.** Algorithm 6 accepts `0 < i < q` (step 2). `extend` and `finish_with_spartan` also need the successor `i + 1 < q`, so they refuse a proof that Algorithm 6 accepts at `i = q − 1` (`crates/nightstream/src/lifecycle/extend.rs`, `crates/nightstream/src/lifecycle/step_inputs.rs`). A chain gets to `i = q − 1` only after `q − 2` calls to `extend`, which is about `2^64`. The tests of rules 1 and 2 extend and finish proofs that verify. No test covers the last counter.

## 7. Nebula memory

Nebula memory checking (Nebula, Sections 3 and 4) gives an application public ROM and RAM. A memory application is a Stage 1 application: its step circuit holds the memory rows, three record chains, the memory carry and the application's own rows, which meet at fixed memory ports (`Lifecycle.Nebula.MemoryApp.program`). It adds no commitment, no claim, no evaluation and no field to the F′ state, so Algorithms 1 to 8 do not change. The memory carry is in the application state: the four state words of section 4.2 are a digest of the application words and the carry (section 7.5).

The steps come in segments of `N` steps. At the start of a segment, the prover proposes two chain roots, and the relation derives the memory challenge `η` from them. Each step checks its memory accesses against fingerprint products and adds its own records to the chains. The step that completes the segment requires that the chains equal the proposed roots and that the products balance. The verifier then runs the Stage 1 terminal check and requires a closed segment (Algorithm 10). An accepted proof attests one execution of the application from the statement's initial words and the program's memory to the statement's final words, in which each memory access goes through a port and reads the last value written (`SECURITY_MODEL.md`, adversary 5).

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

Every memory package that `lake exe emit` writes gets its relation from `Export.Stage1.NebulaMemoryPackage.application`, which takes the proofs of rules 1 to 6. The verifier rejects a program whose ROM image does not have `R` words or whose RAM image does not have `M` words before it computes `D_init` (`Spec.Nebula.Plan.Valid`). The segment counter uses `W_seg = ⌈log₂(S_max + 1)⌉` bits.

The plan digest is `Digest("Nightstream/Nebula/v3/plan", [r, μ, W_ts, B_ops, B_scan, N, S_max])` (section 7.4, `Lifecycle.Nebula.planDigest`). The package rows hold it as a constant, as they hold the verifier-context digest.

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

When the plan names each checked port once, each slot shows at most one active port, and of two dependent active ports in different slots the port of the earlier slot has the lower rank, the events of the active ports in slot order are a reorder of the row's events that keeps each dependent pair (`Spec.Nebula.MemoryTrace.reorders_of_slotOrder` (`2b1d9f7b165426bd`)). The slot plan changes only completeness and cost. `CatalogBinding::program` puts each preloaded cell of a checked memory at `base + packParts(a)` of its image and rejects a cell outside its region. No Lean result shows that the five rules above enforce these conditions, or covers the address and value rows (`SECURITY_MODEL.md`, Out of scope).

**Several rows in one step.** One step can hold `k` rows of a catalog application, so that the rows share the cost of one fold. The step uses the batched catalog (`Spec.Nebula.MemoryTrace.batchCatalog`). It has the memories of the catalog in order. Each memory has the ports of row 0, then of row 1, up to row `k − 1`, and a map `shift i` moves the columns of row `i` to columns of the step witness. At `shift i c`, the step witness holds the value of row `i` at column `c`, for each column `c` that a port reads (`Spec.Nebula.MemoryTrace.HoldsRows`). So a step lists its events memory by memory, and its rows list them row by row. When the memory ids are distinct, each memory's events come in the same order in the two lists. So batched steps are memory consistent exactly when their rows, in order, are memory consistent (`Spec.Nebula.MemoryTrace.memoryConsistent_batchCatalog_iff` (`8884c58e0831ed34`)). An execution whose active ports, step after step, bind a reorder of the batched events of each step makes the trace of the rows memory consistent (`Spec.Nebula.MemoryTrace.memoryConsistent_of_batched_executes` (`d36926c9499e93e5`)). A slot plan that obeys the conditions above in each step gives this reorder (`Spec.Nebula.MemoryTrace.memoryConsistent_of_slotOrder_executes` (`763d05fb5fd4b167`)). Rust `neo_application::MemoryCatalog::batched` builds the batched catalog, and the catalog rejects two memories with one id. The WASM application `neo_wasm::nebula::WasmApplication` keeps column 0 (`z[0] = 1`) for all rows and moves each other column of row `i` by `i` times the row width. Its relation also makes each row's `_AFTER` columns equal to the next row's `_BEFORE` columns, and its slot plan is the one-row plan for each row in turn (`crates/neo-wasm/src/nebula.rs`; not checked against Lean).

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

**Source.** `Lifecycle.Nebula.MemoryApp.program`, the step `Spec.Nebula.invoke`. Rows that hold imply the program's step and validity predicates (`Lifecycle.Nebula.MemoryApp.soundness` (`f1f3e13012f71ac2`)). On a reachable input carry, they give one step of `invoke` (`Lifecycle.Nebula.StepWitness.RowsHold.invoke` (`fb47b498ac87650e`)), and an honest step satisfies the rows (`Lifecycle.Nebula.MemoryApp.completeness` (`292c1ed52f1fd1da`)). Each execution of `S · N` steps with `1 ≤ S ≤ S_max` has a step history whose statement passes Algorithm 10 (`Lifecycle.Nebula.MemoryApp.history_of_execution` (`a1ef468b8bc6e701`)). The first step is the Stage 1 base step (Algorithm 3, step 2).

### Algorithm 10. Memory verifier

**Input.** The package of the application at its geometry, a program, the statement and the final carry, and a Stage 1 proof or a final proof. The statement is the step count `T`, the initial and final application words, the segment count `S`, the final timestamp and the final memory root.

**Output.** Accept or reject.

**Steps.**

1. Require that the program has the package's geometry and valid images, and compute `D_init`.
2. Require that each state of the statement has the application's word count, and that each application word and the final timestamp is below `q`. Each other field and carry counter must equal a value below `q`.
3. Compute the initial state as the state digest of the initial words and the start carry, and the final state as the state digest of the final words and the final carry.
4. Require `idx = N` in the final carry, `1 ≤ S ≤ S_max`, `T = S · N` and `seg_idx = S`, and require that the final timestamp and the final memory root are `ts` and `D_mem` of the final carry. So `T = 0` is rejected.
5. Run Algorithm 6 on `(T, initial state, final state)` and the Stage 1 proof, or Algorithm 8 on the final proof.

**Rejects.** Each step's failure.

**Source.** `nebula::Verifier::verify` and `nebula::Verifier::verify_final`, with steps 2 to 4 in `Context::terminal_states` (`crates/nightstream/src/nebula/mod.rs`; not checked against Lean). Steps 3 and 4 are `Lifecycle.Nebula.TerminalChecks`. A chain of valid steps with these checks gives a run that the memory model accepts, or a state-digest collision in the run (`Lifecycle.Nebula.accepts_or_collision` (`31d12d5e456bee76`)). Algorithm 6 alone does not make steps 2 to 4, so `nightstream::Verifier` refuses a memory relation: a pinned Lean memory package, a circuit from `nebula::Circuit::compile`, or a saved memory circuit, whose file starts with `NSMEM` (`crates/nightstream/src/circuit.rs`; not checked against Lean).

## 8. Differences from the papers

Each row is a place where Nightstream differs from SuperNeo v1.2 (ePrint 2026/242) or HyperNova.

| Paper | Nightstream | Record |
|---|---|---|
| SuperNeo v1.2 Section 8.2 gives a Goldilocks example with `κ = 18`, `k = 14`, `B = 2^14` and `m = 2^30`. | The profile binds `κ = 22`, `k_rho = 16` and `B = 2^16`, and the row cube is `2^28`. It is called "Nightstream Goldilocks, k_rho = 16". | `AGENTS.md` (Protocol), `decisions/fprime-stage1-main-ajtai-setup.md`, `decisions/fprime-stage1-domain-2p28.md` |
| SuperNeo v1.2 Definition 8 samples the commitment matrix uniformly at random. | Algorithm 1 expands the matrix from one fixed public seed with SHAKE128. The premises are SHAKE128 as a random oracle and MSIS for a uniform matrix. | `decisions/fprime-ajtai-shake128-setup.md`, `decisions/fprime-ajtai-shake128-security-argument.md` |
| HyperNova Construction 3 absorbs the structure, the public parameters and the running instance before the first challenge. | The fold transcript absorbs a domain block, the fresh statement and the prover messages. The running claims enter through the state hash in the fresh public input. | `decisions/piccs-prior-state-digest.md`, `SECURITY_MODEL.md` trust item 4 |
| HyperNova Definitions 7 and 11 let the adversary or the IVC prover choose the structure. | The verifier's package fixes one application and its setup. | `decisions/fprime-stage1-per-application-packages.md` |
| HyperNova's history result uses Definition 7 knowledge soundness of the NIFS. | Lean states the history result under errata Assumption 1, which adds the prior-state link and differs from Definition 7 in six named points. | `SECURITY_MODEL.md` trust item 4 |
| SuperNeo samples each Π_RLC challenge uniformly from the strong set. | Each challenge is four sponge lanes reduced modulo `5^54` (section 5.3), within `2^−132` of uniform. | none |
| Nebula Section 4.3 proves `F`, hashes the carried commitments into the challenges, proves `F_ops` and `F_scan` in a second layer, and folds the three proofs with `F_final`. | One F′ chain. At the start of a segment, the prover proposes the ops and FS roots, and `η` comes from them, `D_mem`, `plan_digest` and `ts` (section 7.4). The close of the segment makes the `F_final` checks: the chains equal the proposals, the memory continues, and the products balance. | `decisions/nebula-memory-checking.md`, `SECURITY_MODEL.md` trust item 7 |
| Nebula Sections 4.2 and 4.3 take one challenge pair in the base field. | `η1` and `η2` are in `K`. Over `F_q`, the fingerprint error of a geometry at the rule 6 limit is about `2^−45.9`. | `decisions/nebula-memory-checking.md` |
| Nebula Constructions 1 and 2 chain commitments to a split-committed witness. | The record chains are the hash chain of Nebula Section 3.1 over the packed lanes, computed inside F′, as in the generic construction of Section 3.2. | `decisions/nebula-memory-checking.md` |
| In Nebula Section 4.3, a segment has as many steps as the proof, and the scan has that many chunks. | A segment has `N` steps and the scan has `B_scan = (R + M)/N` slots in each step. Idle steps fill the last segment. | `decisions/nebula-memory-checking.md` |
| Nebula Section 4.2 has one memory. | ROM and RAM share one index space. O5 forbids a ROM write, and O6 keeps a ROM address below `R`. | `decisions/nebula-memory-checking.md` |
| Nebula Construction 2 hashes the carried commitment as its own field of the state hash. | The memory carry is in the application-state digest, so the F′ state hash and layout do not change. | `decisions/nebula-memory-checking.md` |

The `F_ops` checks of Nebula are O4 and the write timestamp `ts + cnt_j`, and the `F_scan` address check is the computed cell of a scan slot. The published `F_scan` has no timestamp range check, and Nightstream adds none: one segment is sequentially consistent without it (`Spec.Nebula.segment_sequential` (`b186065f1fe7272e`)).

The scope choices that the papers leave open are records too: the reference application (`decisions/fprime-stage1-poseidon2-hash-chain-v1.md`), the 41-trit opening of field values in binding maps (`decisions/41-trit-encoding.md`) and the Poseidon2-only rule for protocol binding (`AGENTS.md`, Protocol).

## 9. Conformance

An implementation conforms when it reproduces these vectors and passes these checks. `SHA256SUMS` in each artifact folder pins the bytes of the Lean exports.

| Vectors | Check | Covers |
|---|---|---|
| `formal/nightstream-fprime/artifacts/` | `python3 -B formal/nightstream-fprime/scripts/check_artifacts.py` (CI) | the digests of the Lean exports and of their copies in `crates/nightstream/artifacts/` and `crates/nightstream-fprime/artifacts/` |
| `crates/nightstream/tests/fixtures/poseidon2-application-reference.json`, `crates/nightstream/tests/fixtures/poseidon2-application-execution.json` | `cargo test -p nightstream --release --test application_poseidon2` (CI) | the reference application's rows and witness against the Lean export (`crates/nightstream/tests/fixtures/README.md`) |
| `crates/nightstream/tests/fixtures/compression/` | `cargo test -p nightstream --release --test circuit_lifecycle committed_final_proof_verifies_and_rejects_changes` (CI) | a production compression key and final proof that verify, and changed bytes, states and keys that do not |
| `crates/nightstream/tests/fixtures/golden-v1.zip` | `crates/nightstream/tests/check_golden_vectors.py` | a two-fold vector replayed by the Lean and Rust verifiers |
| the golden conformance run | `python3.12 -B scripts/golden_conformance_ci.py --directory DIRECTORY` | two fresh native folds checked by Lean, with proof bytes, caller values and witnesses (`scripts/GOLDEN_CONFORMANCE.md`) |
| the independent Lean generation | `scripts/INDEPENDENT_GENERATION.md` | Π_CCS messages, Π_RLC and Π_DEC witnesses generated by Lean from the original openings |
| the selected package identity | `bash scripts/validate.sh identity`, from `formal/nightstream-fprime/` | the emitted binding against the fixture and the Rust pins |
| `formal/nightstream-fprime/artifacts/nightstream-fprime-stage2-nebula-memory-v1.json`, `formal/nightstream-fprime/artifacts/nightstream-fprime-stage2-nebula-memory-n1.json` | `cargo test -p nightstream --release --lib nebula::` (CI) | the Rust-built memory relation of the machine against both Lean memory packages, row for row; the catalog rows; reads past the view. Ignored tests prove both packages and check the rejection cases that need a bad witness (`crates/nightstream/tests/nebula/conformance.rs`) |
| `crates/nightstream/tests/fixtures/nebula-memory-v1-terminal.json`, `crates/nightstream/tests/fixtures/nebula-memory-n1-terminal.json` | `cargo test -p nightstream --release --test nebula_memory` (CI) | Algorithm 10 against the Lean terminal checks, `D_init` against Lean, and statements that are not canonical or have another word count |
| the WASM application | `cargo test -p neo-wasm --release` (CI) | the catalog rows of a WASM program at 1, 2 and 4 rows in each step, a load that reads a store of an earlier row of its step, and a step of rows that are not neighbors; ignored tests prove it at 1 and 2 rows in each step |

These checks cover their recorded inputs. No theorem covers arbitrary Rust execution (`SECURITY_MODEL.md`, trust item 5).
