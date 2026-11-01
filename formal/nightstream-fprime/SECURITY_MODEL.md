# Security model

This page states which adversaries the Nightstream F′ Lean results protect
against: what each adversary controls, when it wins, the Lean result that
bounds it, and what you must trust. It does not restate proofs; read the cited
declaration. The literal statements of the registered targets are in
`tests/EvidenceTargets.lean`.

Names omit the common `NightstreamFPrime` namespace.
`tests/EndpointCensus.lean` checks that every name on this page that starts
with `Export.`, `Layout.`, `Lifecycle.` or `Spec.` exists, and that every cited
theorem uses only `propext`, `Classical.choice` and `Quot.sound`. Shorter names
are not checked.

## What Lean checks

- The kernel of the official Lean v4.35.0 toolchain, with Mathlib v4.35.0 and
  the CompPoly revision that `lakefile.toml` pins.
- `scripts/validate.sh` runs on the developer Mac. Its `build`, `axioms`,
  `executables` and `all` phases use `lake build --wfail`, and single files
  run with `warningAsError`, so any warning fails a gate. `all` runs the
  boundary checks, both libraries, every `lean_exe` target and the census.
  CI runs no Lean (owner decision of 2026-09-26).
- Every audited theorem uses the three axioms above. The audit fails closed:
  `sorryAx` or a new axiom stops the build.
- One linter is off: `Circuit/Basic.lean` turns off `linter.dupNamespace`,
  because the `Circuit` monad has the name of its namespace. It is a naming
  lint, not a check of any proof.
- Strong-set low-norm invertibility (SuperNeo Theorem 4) is proved
  (`Spec.Phi81StrongSet.lowNormInvertibility`), so no security result takes it
  as a premise.

## Applications and setups

An application is a step program (`Lifecycle.Stage1.Application.Program`).
The results for adversaries 1 to 4 hold for every application that fits the
`2^28` profile (`Export.Stage1.PerApplicationFixedPoint.FitsTwoPow28`), at its
own relation (`Export.Stage1.PerApplicationFixedPoint.relation`). Adversary 5
also requires a valid memory plan and a `PortApp` model. Adversary 1 holds for
every Ajtai key, adversaries 2 and 4 for every commitment setup
(`Export.Stage1.PerApplicationCanonicalPackage.CommitmentSetup`, with key
`Export.Stage1.PerApplicationCanonicalPackage.commitmentKey`), and adversary 3
draws the key inside its game. `Poseidon2HashChainV1` is the reference
application of the golden runs and the executables.

No adversary chooses the application or the setup: the verifier's package
fixes both before the adversary runs. A memory application (adversary 5) is
such an application: its package also fixes the memory geometry, and the
verifier takes the program, which the adversary does not choose. The
verifier-context digest hashes the relation, the application, the NIFS key
and the setup authority (setup identifier, dimensions and seed)
(`Export.Stage1.PerApplicationCanonicalPackage.verifierContextDescriptor`),
and every state hash carries it. So when one application's verifier accepts a
proof, adversary 2's result gives that application's context or a state-hash
collision.

The verifier pins the package identity. Two packages with one package
identity have the same sealed envelope and the same authority, or one of four
named collisions: two structural values with one structural identity, two
application plans with one application digest, two authority components with
one component digest, or two final preimages with one package identity
(`Export.Stage1.PerApplicationSecurity.packageIdentity_identifies_package_authority_or_collision`).
The structural identity takes the application only through its application
digest (`Export.Stage1.PerApplicationCanonicalPackage.structuralValue`).

The adversary may replace all prover code, including witness generation and proof
serialization. Only the verifier's code and package are fixed. The Lean games
read arbitrary assignments, proof messages and envelopes, not outputs of an
honest prover algorithm.

## Adversaries

### 1. Malicious step assignment

- **Controls:** every value of the step circuit's assignment and the claimed
  public input.
- **Wins:** the package rows are zero and the public input matches, but the
  typed step does not hold.
- **Result:** no such win. For an arbitrary selected assignment, zero rows,
  the public-input equality and a four-word digest imply the typed step under
  the decoded context
  (`Export.Stage1.ActualPiDECOutput.selectedRowsAndPublic_imply_step`; for
  the fixed application,
  `Export.Stage1.Poseidon2HashChainV1Closure.rowsZero_implies_stepHoldsFor`).
  The compact matrix program has the rows of the structural plan
  (`Export.Stage1.PerApplicationCanonicalPackage.matrixProgram_exact`). The
  result is deterministic and needs no hardness premise. Adversary 2 binds
  the decoded context to the verifier's context.

### 2. Malicious terminal opening

- **Controls:** the terminal statement and every value of the terminal
  payload.
- **Wins:** the terminal predicate accepts, but the opened step is not the
  authenticated NIFS step of the verifier's context.
- **Result:** acceptance gives the verifier's context, the prior-state link
  and the exact NIFS output, or the base step, or a named state-hash collision
  (`Export.Stage1.ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision`;
  the selected rows come from
  `Export.Stage1.ActualContextSecurity.terminal_implies_rowsAndPublic`). A
  named collision is two preimages with different serializations and one
  Poseidon2 hash (`Layout.Stage1.PiCCSSecurity.StateHashCollision`). For two
  well-formed preimages, equal hashes give equal preimages or such a collision
  (`Layout.Stage1.PiCCSSecurity.stateHash_identifies_statement_or_collision`).
  The result is deterministic. It is about the Lean terminal predicate, not
  about the Rust verifier (trust item 5).

### 3. One-fold Fiat–Shamir adversary (random-oracle model)

- **Controls:** a classical algorithm that is a function of the setup chunks,
  so it may read the whole Ajtai key, and that makes at most `Q` queries to
  the oracle `H`. It may choose the running and fresh statements after any
  query. It outputs a claim (the statements, a NIFS proof and 16 child
  witnesses) and a prior preimage. The algorithm is deterministic
  (`Spec.RandomOracle.OracleComp` has no coins). A randomized adversary is a
  family over uniform coins, drawn independently of the chunks and of `H`;
  `Export.Stage1.RandomOracleSetup.randomized_knowledge_error_lt` states the
  result averaged over the coins. Each query is a call list no longer than a fixed
  bound and a slot (`Lifecycle.RandomOracleTest.Point`).
- **Does not control:** the application, the setup chunks, which are uniform
  256-bit values (trust item 2), and `H`, a uniform random function of each
  challenge's address, its exact call list and its slot (trust item 1). The
  adversary is fixed before the chunks and receives no advice about the
  sampled key. A kernel vector found after reading the chunks is charged to
  the MSIS term.
- **Wins:** the oracle verifier accepts the claim, the child witnesses open
  the returned children, the prior preimage links the claim, and extraction
  returns no source-valid witness. The link
  (`Export.Stage1.RandomOracleLink.linkedClaim`) requires that the prior
  preimage hashes to the digest in the fresh public input and is well formed
  (`Layout.Stage1.StateEncoding.WellFormed`). Well formed includes that the
  running children are the canonical split of the parent that the state hash
  stores (`Lifecycle.ChildrenCanonical`); the deployed verifier rejects every
  other split.
- **Result:** for every application that fits,
  `Export.Stage1.RandomOracleSetup.production_knowledge_error_lt`
  (lean-graph target `rom-knowledge-soundness`):

  ```
  𝔼 chunks, Pr_H[accept] < 𝔼 chunks, (Pr[extract] + hashCollisions)
                           + 17 (Q + 17) ε_sample + (Q + 74) ε_test
                           + msisAdvantage + 2^-190
  ```

  - `Export.Stage1.RandomOracleSetup.hashCollisions`: a retry or a rerun
    gives two different prior preimages of the adversary with one state hash.
  - `Lifecycle.RandomOracleKnowledge.statisticalError`: the two statistical
    terms. `ε_test = 4589/p² ≈ 2^-115.84`
    (`Lifecycle.Nifs.VerifierErrorBudget.test_error_eq`) and
    `ε_sample ≈ 2^-125.4`. Each query adds at most `50 / (87 · 2^115)`, which
    is below `2^-115.79` (computed outside Lean): the sum is at most
    `(Q + 74) · 50 / (87 · 2^115)`
    (`Lifecycle.RandomOracleKnowledge.statisticalError_le`), so at `Q = 2^64`
    it is below `2^-51` (about `2^-51.80`;
    `Lifecycle.RandomOracleKnowledge.statisticalError_two_pow_64`).
  - `Export.Stage1.RandomOracleSetup.msisAdvantage`: the success of an
    explicit MSIS solver on a uniform matrix. It runs the binding reduction
    `Lifecycle.RandomOracleBinding.rerunKernel`, which computes a nonzero
    kernel vector of the key with every coordinate below `8TB` from two runs.
  - `2^-190` bounds the cost of a uniform matrix in place of reduced chunks
    for every application that fits
    (`Export.Stage1.RandomOracleSetup.programmingError_lt_of_fits`).
- **Work:** the extractor takes `17 (Q + 17)` expected retries
  (`Lifecycle.RandomOracleExtraction.expected_retries_le`). The binding
  reduction is an oracle program that holds no adversary
  (`Lifecycle.RandomOracleReduction.reduction`);
  `Lifecycle.RandomOracleReduction.solve` runs it on the adversary's records.
  The program's queries are adversary runs: it builds the oracle of each run from
  its state and one fresh tape entry, and the answer is the run's record (its
  output and its queries). So the run count is a query bound of that program,
  as in Ironwood's cost certificates. With budget `T` it makes at most `T`
  runs on every tape and for every adversary
  (`Lifecycle.RandomOracleReduction.runs_length_le`). The oracle of each run
  depends on the adversary only through the records of the earlier runs
  (`Lifecycle.RandomOracleReduction.runs_take_congr`), and the output only
  through the records of the runs made
  (`Lifecycle.RandomOracleReduction.solve_congr`). For every budget it makes
  at most `35 Q + 653` expected runs
  (`Lifecycle.RandomOracleSolver.expected_runs_le`): the base run,
  `17 (Q + 17)` retries, `Q + 74` reruns
  (`Lifecycle.RandomOracleUniqueness.expected_reruns_le`) and `17 (Q + 17)`
  retries of the extractions inside the reruns
  (`Lifecycle.RandomOracleSolver.expected_second_retries_le`). Lean counts
  adversary runs, not machine work. Each step also applies the claim parse
  function to the run's record and checks the claim (the oracle verifier,
  and `PriorLink` for the prior preimage). Lean decides these checks
  classically. The strict-time readings in trust item 3 need these checks to
  be efficient, with their cost counted inside a run.
- **What it covers:** the oracle verifier is the Lean NIFS verifier
  `Spec.Folding.Nifs.PaperNonInteractive.verify`, with its coins read from
  `H`, plus the child openings. At an oracle that answers the 74 challenge
  points of one execution as the sponge does, the two agree
  (`Lifecycle.RandomOracleFidelity.accepts_iff_verify`), and every execution
  has such an oracle (`Lifecycle.RandomOracleFidelity.deployedOracle_deployed`).
  `Lifecycle.RandomOracleExtraction.Claim` contains an assignment for each
  child, and its `Accepts` event requires every child assignment to satisfy
  `CE.Holds`. The public NIFS proof does not carry those assignments. The
  adversary algorithm in this game can be arbitrary, but its success event is
  conditional on valid child openings; this theorem alone does not bound all
  proofs that the official verifier can receive. The reverse history walk
  obtains child openings from its current source result
  (`Export.Stage1.HyperNovaVisitedSecurity.Stage.nifs`) and then uses its
  separate Assumption 1 to extract the preceding source. Neither step follows
  from this one-fold oracle theorem alone.
  Oechsner, Pereira and Scholl (ePrint 2025/1835, Section 3.3) show how an
  unannounced restriction on malicious prover outputs can hide a missing
  verifier check. This restriction is explicit here, but it must not be
  treated as unconditional public-verifier soundness.
  Parno (ePrint 2015/437, Section 3) reuses a QAP proof for a false public
  statement when the verifier checks only a linearly dependent projection of
  its input/output values. The selected terminal verifier compares the whole
  externally expected state, recomputes the state hash from the running
  claims, requires the children's canonical split of their parent, and checks
  every fresh witness public coordinate. The compressed terminal also enters
  the whole state as public circuit words (`lifecycle/verify.rs`,
  `lifecycle/inputs.rs`, `lifecycle/terminal/mod.rs`). The paper's projection
  substitution therefore has no identified path here; Poseidon2 binding and
  the Rust-to-Lean refinement remain separate premises.
  Tang et al. (ePrint 2024/514, Section 2.2) describe an unused-public-input
  replay and a prover assignment that has no matching constraint. The
  selected fresh public input equals the rehashed terminal state and every
  coordinate equals its witness projection (`lifecycle/verify.rs`). The
  compressed terminal has nine fixed external statement words; `Shrink::start`
  absorbs all of them before the commitment, and `shrink::verify` includes
  all of them in its public-input evaluation and checks the shape-run public
  words against the statement. Lean's
  `CompressionLean.Terminal.circuitPackage_implies_terminal` connects the
  selected terminal circuit to the terminal check under its opening premise.
  This check finds no omitted external state word in that path; the concrete
  WHIR and Rust-to-Lean premises remain.
  The deployed system checks every fold only inside the step circuit of
  `F′`, with Poseidon2. The Rust terminal verifier checks the final openings
  and the state-hash link, and runs no fold
  (`crates/nightstream/src/lifecycle/verify.rs`). No random-oracle model
  covers a hash inside a circuit, so this result bounds no deployed verifier
  check. For the history, it only motivates the value of `error` in
  Assumption 1 (adversary 4). `Export.Stage1.RandomOracleSetup.contract`
  states the result as Ironwood's six-question `Spec.KnowledgeContract`.
- **Model limits:** the extractor and the binding reduction rerun the
  adversary with a changed oracle, so the oracle is programmable. The game
  draws the chunks after the adversary is fixed, but the deployed seed is
  fixed and public. An adversary that is precomputed for that seed, or an
  application that is chosen after it, is outside the model.

The proof follows the interactive SuperNeo v1.2 extraction and replaces each
fresh verifier coin by an oracle read. The Lean column names the declaration for each step:

| Lemma | Step | Lean |
|---|---|---|
| 1 | Each challenge reads the state after its exact call list at its slot (a lane pair, or four lanes for a Π_RLC scalar); distinct challenges have distinct addresses (call list and slot). | `Lifecycle.TranscriptCoverage.challenge_seal`, `Lifecycle.TranscriptCoverage.challengeAddress_injective` |
| 2 | Bad-set bounds for bad sets that do not read their own point: `Q ε` at the queried points (Ironwood `escapesDuringC_measure_le'`) and the pinned-squeeze `(Q + 1) ε` at an output point (Ironwood `xEscAtPoint_measure_le`). | `Spec.RandomOracle.escape_le`, `Spec.RandomOracle.pinned_le` |
| 3 | A false PiCCS acceptance has one bad coin. | `Spec.Folding.PiCCS.PaperJoint.RoundByRound.falseAcceptance_splits` |
| 4 | Test error in the oracle model. | `Lifecycle.RandomOracleTest.test_error_le` |
| 5 | Π_RLC coordinate fork extraction. | `Lifecycle.RandomOracleExtraction.fork_failure_le` |
| 6 | Forked uniqueness and the binding reduction. | `Lifecycle.RandomOracleUniqueness.source_error_le`, `Lifecycle.RandomOracleBinding.collisionChance_le_kernelChance` |
| — | One key. | `Lifecycle.RandomOracleKnowledge.knowledge_error_le` |
| — | With the prior-state link, a moved running statement is a state-hash collision. | `Export.Stage1.RandomOracleLink.knowledge_error_le_linked` |
| — | The key is drawn inside the game. | `Spec.AjtaiSetupV1.Programming.expect_le_programmed`, `Export.Stage1.RandomOracleSetup.knowledge_error_le_setup` |

### 4. Recursive-history adversary

- **Controls:** a random tape and, from it, a terminal statement and proof
  after any number of folds, up to a symbolic depth, for a fixed application
  and commitment setup. The adversary must be admitted (trust item 4).
- **Wins:** the terminal is accepted, and no valid history exists: no
  application witnesses of the circuit's witness length satisfy the
  application's validity predicate at each step and take `z0` to `zi` in
  `iteration` steps (`Export.Stage1.HyperNovaFalseAcceptance.FalseAcceptance`).
  Every history that the reverse run returns has that length
  (`Export.Stage1.HyperNovaHistory.run_witness_length`).
- **Result:** under Assumption 1 and the closure premise,
  `Pr[FalseAcceptance] ≤ Σ_j (h_j + error(stage j))` on the IVC adversary's
  own law, with no conditioning
  (`Export.Stage1.HyperNovaFalseAcceptance.probability_bound`, lean-graph
  target `hypernova-terminal-false-acceptance`). `h_j` is stage `j`'s marked
  state-hash collision mass.
- **Extraction form:** under the same premises, on the joint law of the
  adversary and the reverse extractor of HyperNova Lemma 17
  (`Export.Stage1.HyperNovaVisitedSecurity.reverseStages`),
  `Pr[accept ∧ no valid history returned] ≤ Σ_j (h_j + error(stage j))`
  (`Export.Stage1.HyperNovaVisitedSecurity.history_failure_le`, lean-graph
  target `hypernova-linear-security`). This is the failure event of
  HyperNova Definition 11 (ii), constant-step knowledge soundness, and it
  implies that definition's difference form. The joint law has the adversary's output law
  (`Export.Stage1.HyperNovaVisitedSecurity.reverse_input_event`), and the
  false-acceptance bound above follows from this one.
- **Depth:** with the error that adversary 3 motivates, the bound is useful
  only for one or two folds (trust item 4, *Concrete depth*).

### 5. Memory adversary

- **Controls:** what adversary 4 controls, for a memory application at a
  fixed geometry and program, and from the same tape a claimed memory
  statement and the final carry. The adversary must be admitted (trust
  item 4).
- **Wins:** the memory verifier accepts (`SPEC.md` Algorithm 10), and the
  history that the reverse extractor returns does not attest the claim: it
  is not an execution from the claim's initial words and the program's
  memory to its final words in which each memory access reads the last value
  written.
- **Result:** under Assumption 1, the closure premise and the memory round
  (trust item 7), the mass of wins on the joint law of the claim and the
  extractor's sample is at most `badBound`: the Stage 1 history terms of
  adversary 4, the Poseidon2 collision masses of the extracted run's state
  links and record chains, and the memory-round error of the last stage
  (`Export.Stage1.NebulaMemorySecurity.bad_le`). The claim and the Stage 1
  statement read the adversary's own tape
  (`Export.Stage1.NebulaMemorySecurity.input_adversaryTape`). For an
  adversary whose outputs can have more than `S_max · N` steps, the same
  bound holds for its clamped form when its start stage is admitted
  (`Export.Stage1.NebulaMemorySecurity.bad_le_clamp`).
- **Deterministic part:** an accepted run gives an execution, a Poseidon2
  collision among the run's own chain inputs and the `D_init` chain, or a
  segment that passes the product test with unbalanced multisets
  (`Spec.Nebula.soundness`). With the state links, an accepted chain of
  valid steps gives such a run or a state-digest collision
  (`Lifecycle.Nebula.accepts_or_collision`, joined in
  `Lifecycle.Nebula.EndToEnd.bad_subset`).
- **Challenge game:** in the interactive game where `η` comes after the
  roots and `ts`, segment slot `k` is bad with frequency at most
  `2 · m_mem/q²` and a retry term, and each retry disagreement is a Poseidon2
  transcript collision (`Lifecycle.Nebula.badAt_bound`). Here
  `m_mem = R + M + N · B_ops`, and setup rule 6 keeps `2 · m_mem/q²` at most
  `2^−109.91` (`Spec.Nebula.Geometry.secure_iff`).
- **Depth:** the depth is `S_max · N`, so the depth limit of adversary 4
  applies.

## What you trust

1. **The random-oracle model of the challenge reads.** The NIFS verifier
   reads each challenge from the Poseidon2 sponge state after the exact call
   list of that challenge. One state gives up to six coins: `α` and `γ` read
   consecutive lane pairs, and a round coin reads lanes 0 and 1 after its
   message (`Lifecycle.Transcript.squeezeAt`); a Π_RLC scalar reads four lanes
   from lane 0. The deployed
   system makes these reads only inside the step circuit, so this model
   applies to adversary 3's game, not to a deployed check.
   `Lifecycle.TranscriptCoverage.piCcsProbe_coins` and
   `Lifecycle.TranscriptCoverage.rho_seal` prove that the coins are one read
   of each challenge's address: its call list and its slot. The model
   replaces that read by a uniform random function of the address
   (`Lifecycle.RandomOracleTest.coins` for the Π_CCS coins,
   `Lifecycle.RandomOracleExtraction.rho` for the Π_RLC scalars): each
   address gets an independent
   four-word answer, and a model query is one address. In the sponge, the
   answers at neighbouring slots of one state share two lanes, but no two
   verifier reads use the same lane. This is
   a modelling assumption, and a stronger one than Ironwood's. Ironwood
   idealizes one hash call on the whole transcript. The standard
   justification would be duplex-sponge Fiat–Shamir in the ideal-permutation
   model, where the rate lanes of one permutation output are independent;
   Lean does not prove it. Chiesa and Orrù (ePrint 2025/536, Theorems 6.1 and
   6.2) bound an overwrite-mode duplex with a separate random input hash.
   `Poseidon2Transcript::absorb_v1_2` instead adds each chunk to the rate lanes
   and permutes after it, starting from zero. Their bound does not apply to
   this selected transcript without a new reduction for its exact schedule.
   Ashur and Bhati (ePrint 2024/911, Sections 5 and 6) include additive
   chaining over prime fields in their generalized sponge model. Their
   random-oracle analysis requires injective padding and treats one hash
   invocation. It does not directly justify this transcript's intermediate
   challenge reads, including several lane pairs from one state and the
   zero-chunk transition after the sixth pair.
   Dao, Miller, Wright and Grubbs (ePrint 2023/691, Section VI) break a weak
   Fiat–Shamir Spartan transcript by choosing one public input after its
   challenges. In Nightstream, `bind_and_sample_with_trace` absorbs the fresh
   public input before `α`; the official F′ constraints check that its prior
   digest matches the running claims. Layer 1 starts from its key and claim,
   and shrink starts from its key and state, before either first challenge.
   The paper's late-public-input attack does not directly apply to these
   configured paths. It does not prove the concrete transcript secure.
   Fenzi (ePrint 2026/1838) attacks generated R1CS when the first
   Fiat–Shamir challenge does not bind the generated statement. The first
   Π_CCS absorb includes the fresh public input, which the verifier requires
   to contain the state hash of the configured relation and application
   context. Layer 1 absorbs its trusted matrix-root key and statement before
   its first challenge. The paper's delayed-statement attack therefore needs
   a state-hash collision or a change to verifier-owned authority here; this
   does not settle the concrete Poseidon2 random-oracle premise.
   Merz and Rodríguez García (ePrint 2026/306) give faster algebraic attacks
   on Poseidon2's linear layers and on some sponge and compression modes,
   while reporting no break of the 128-bit sets they analyze. The selected
   permutation uses 16 Goldilocks words, `x^7`, eight full rounds and 22
   partial rounds. Their mode estimates do not bound this additive duplex's
   repeated intermediate lane reads. Hovhannisyan, Asaturyan and Hovhannisyan
   (ePrint 2026/409) report no forgery from their coupled Poseidon and
   Fiat–Shamir algebraic experiments, even with reduced rounds. Those
   negative experiments are not a security reduction for this transcript.
   Grassi, Koschatko and Rechberger (ePrint 2025/954, Section 7.2 and
   Table 3) analyze Poseidon2 sponge subspace attacks near `p = 2^64` with
   width 16, capacity four and digest four. Their table fixes a cubic S-box
   and six full rounds, unlike this `x^7`, eight-full-round profile, and they
   report no break of the instances they analyze. Their estimated costs do
   not price this selected tuple or its intermediate transcript reads.
   Vitto (ePrint 2026/1579, revised 2026-09-18, Sections 5.2 and 6.2)
   constructs a weak Poseidon matrix after seeing the round constants. The
   selected Poseidon2 external and internal matrices are fixed independently
   of the seed-derived constants. For the paper's direct, jointly independent
   four-full-round CICO-4 prefix construction, Proposition 4 requires
   `4 · (4 + 1) = 20` matrix actions at width 16, so that construction cannot
   fit. The paper does not rule out dependent constructions, and neither
   result bounds this additive duplex with intermediate challenge reads.
   Li, Liu and Wang (ePrint 2026/1792, Sections 4.2–4.3) extend algebraic
   subspace attacks to nonlinear trails. Their `p ≈ 2^64` sponge estimates use
   a cubic S-box and six full rounds; their highlighted `x^7` cases use
   `p ≈ 2^256`. Neither matches this 16-word Goldilocks, `x^7`, eight-full,
   22-partial-round profile. The method itself does reach the selected
   internal linear layer: with four capacity words and a four-word digest,
   its sponge CICO budget is `E_c = 16 - 4 - 4 = 8`. The Appendix B.7 even-
   construction core `N_e` is `7 × 7`; direct arithmetic modulo Goldilocks
   with the pinned internal diagonal gives
   `det(N_e) = 2691752801020228242 ≠ 0`. Thus the matrix-level balancing
   check supports that paper's 16-partial-round trail here. The selected
   permutation has 22 partial rounds and eight full rounds. The paper gives
   no cost for this selected tuple, and its CICO preimage model does not
   analyze the additive duplex's intermediate challenge reads. This check
   does not prove a break or the oracle premise.
   Li and Wang (ePrint 2026/967, Section 4, Proposition 3) show that the
   `diag(λ) + J` internal layer has no infinite linear subspace trail of
   their type when its 15 inactive-coordinate diagonal entries are pairwise
   distinct; the finite linear trail has length at most 15. The pinned
   Goldilocks diagonal has 15 distinct inactive entries, and their
   Vandermonde determinant is `11457322283648754730` modulo `p`, not zero.
   This excludes that paper's repeated-diagonal infinite linear trail here.
   It does not exclude nonlinear trails, settle the full permutation's
   algebraic cost, or justify the concrete transcript as a random oracle.
   Bak, Bariant, Boeuf, Hostettler and Jazeron (ePrint 2026/150, Tables 1–3)
   analyze a Goldilocks Poseidon2 instance with width eight and `x^7`:
   their concrete bounty targets have six full and at most 13 partial rounds,
   while their 8-full/22-partial CICO-1 estimate is for width eight, not the
   selected width 16. Sanso and Vitto (ePrint 2025/937, Table 2) solve the
   width-eight 6-full/7-partial Goldilocks bounty instance. Neither result
   gives a preimage, collision, or cost for this selected 16-word permutation
   or its multi-read transcript.
   Sanso and Vitto (ePrint 2026/1254, Sections 3–4) introduce degree
   annihilation for cubic Poseidon, with a reduced-round CICO-2 proof of
   concept. They identify FFT-friendly width-16 settings as natural further
   targets, but do not give control equations, a solution or an attack cost
   for this `x^7` Poseidon2 instance with eight full rounds. Their reported
   full-round estimates do not fall below the claimed security levels of the
   parameter sets they consider; this is not a bound for the selected tuple.
   The same sponge also computes the state hash, which the theorem
   treats as a concrete function. The state hash is a four-lane read after its
   own call list (`Lifecycle.TranscriptCoverage.stateHash_eq_run`), and the
   only common prefix of a challenge's call list and a state hash's call list
   is empty, because their first inputs absorb different domain chunks
   (`Lifecycle.TranscriptCoverage.stateHash_separated`). So no challenge
   read uses a call list that the state hash computes. The verifier-context
   digests (`Lifecycle.VerifierContext.componentDigest` and
   `Lifecycle.VerifierContext.Descriptor.digest4`) also call
   `Poseidon2.hash` from the zero state. No separation statement covers
   them; they hash only fixed words that the verifier owns. Each Π_RLC
   scalar is the existing sampler applied to an oracle answer; its law is
   within `2^-132` of uniform on the strong set
   (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt`).
   The F′ circuit also computes Poseidon2 while it checks the prior Π_CCS
   fold (`Lifecycle.Stage1.Formal.piCcsChild`), and the reference application
   computes Poseidon2 (`Lifecycle.Stage1.Poseidon2HashChainV1.step`). The
   random-oracle theorem fixes `ProductionKey.LogicalRelation` before drawing
   the ideal challenge oracle (`Lifecycle.RandomOracleKnowledge`); it does not
   model a circuit that queries that same ideal oracle. Arnon and Yogev
   (ePrint 2025/329, Section 2.4) and Ghosal et al. (ePrint 2026/2114,
   Section 2.1) explain how a computation that evaluates its own
   Fiat–Shamir hash can predict a challenge in diagonalization attacks on
   recursive proof systems. Their examples also need a way to turn that
   prediction into an accepted false output. No such path has been found in
   the fixed F′ or reference application, but the theorem above does not
   prove soundness when the ideal oracle is instantiated by the same
   Poseidon2 that the circuit computes. This concrete recursive
   Fiat–Shamir premise remains open even with the state-hash domain
   separation proved above.
2. **SHAKE128 as a random oracle (premise P1).** For every setup, each key
   coefficient is one SHAKE128 output chunk reduced modulo `p`
   (`Spec.AjtaiSetupV1.Programming.verifierKey_eq`). The
   game draws the chunks uniformly and independently of the Fiat–Shamir
   oracle. The context digest is computed from the public setup authority
   (setup identifier, dimensions and seed;
   `Spec.AjtaiSetupV1.Setup.authorityWords`), so it does not depend on the
   chunks.
3. **Two hardness terms.**
   - If MSIS is hard for a uniform matrix of `22 × blockCount` elements of
     `F_p[X]/Φ₈₁` (degree 54) at norm `8TB`, at the solver's work, then
     `msisAdvantage` is small. `blockCount` is at most 4,971,027 for every
     application that fits, and 814,144 for `Poseidon2HashChainV1`. MSIS
     hardness is stated for solvers with a strict time bound. The solver
     `Lifecycle.RandomOracleReduction.solve` with a strict budget of `T`
     adversary runs loses at most `(35 Q + 653) / T`:
     `msisAdvantage ≤ solverAdvantage T + (35 Q + 653) / T`
     (`Export.Stage1.RandomOracleSetup.msisAdvantage_le_solver`), where
     `solverAdvantage T` is the success of `solve`. So MSIS hardness at the
     work of `T` runs applies to `solverAdvantage T`. The
     2026-09-08 public-seed MSIS approval covered the ChaCha20 matrix only.
     For ordinary binding collisions in the SHAKE128 random-oracle setup
     game for the reference application,
     `Export.Stage1.Poseidon2HashChainV1Setup.production_binding_lt_solver` gives an
     MSIS solution for a uniform matrix, with error below `2^-190`.
     Wee and Wu (ePrint 2024/028, Section 4) attack extractability of other
     lattice functional commitments by deriving a short-preimage basis from
     their published setup. This setup publishes a matrix description, not
     those short preimages, and this theorem uses MSIS binding plus a
     rewind-based protocol extractor rather than an extractable Ajtai
     commitment premise. Their attack gives no false proof for this
     selected verifier and does not close this extractor's recursion bound.
     For the selected Goldilocks prime, `q mod 81 = 4` and
     `ord_81(q) = 27`; thus `Φ₈₁` has two degree-27 factors over `F_q`.
     At the selected MSIS norm `8TB = 113,246,208`, `3(8TB)² < q`.
     `Spec.Phi81StrongSet.lowNormInvertibility` therefore rules out a
     nonzero short ring word that vanishes in only one CRT factor.
     The unstructured lattice estimate does not price attacks that use this
     CRT split. The Module-SIS algorithm of ePrint 2026/2420 is proved for
     power-of-two cyclotomic rings and does not give a cost for this profile.
     Koo, Lee, Lee, No and Kim (ePrint 2020/1257, Theorem 3) give a
     same-prime-modulus reduction from module-SIS to ring-SIS, conditional on
     a ring-SIS solver and their rank and norm premises. For rank 22, its
     output Euclidean norm is `(t σ √n · m)^21 β^22`, where `m` is the number
     of ring columns and `β` is the assumed ring-SIS solution norm. This is
     not the selected coefficient-infinity bound `8TB`, and the theorem gives
     no fixed-key attack or cost for it.
     Porter, Mendelsohn and Ling (ePrint 2021/600, Theorem 8) reduce SVP in
     either prime ideal above `q` to a two-dimensional subfield problem here:
     `v₃(q − 1) = 1`, so their parameter `r` is zero and `φ(3) = 2`.
     This does not yield a short word under `8TB`, by the Lean invertibility
     bound above. Their module extension (Section 6) requires decomposition
     group symmetry of the target module; a kernel of a uniform matrix is
     not generally fixed by those automorphisms. This paper gives no cost
     for the selected random module-SIS kernel.
     Cramer, van Gent, Lesavourey and Pellet-Mary (ePrint 2026/1805,
     Theorems 3–4) likewise reduce module-lattice dimension only when a
     nontrivial Galois group fixes the module coordinate-wise as a set, under
     their unramified-discriminant premise. The indexed SHAKE128 key-generation
     rule imposes no such symmetry. For the selected fixed key, let `A₀` be
     its first 22 rows and blocks 0–21 over `R_q`, let `b` be block 22, and
     solve `A₀x = b` by unit-pivot elimination. Independent Python and Rust
     ring calculations confirmed that `(x, -1)` is in all 22 kernel rows.
     For each of the 53 nonidentity automorphisms `σ_a: X ↦ X^a` of
     `Q(ζ₈₁)`, the constant coefficient of row 0 of `Aσ_a(x, -1)` is nonzero;
     the product of these 53 coefficients modulo `q` is
     `11665745737130300385`. Thus no nontrivial Galois automorphism fixes
     this selected q-ary kernel as a set. This excludes the cited dimension
     reduction for this key; it does not bound other module-SIS attacks.
     De Perthuis and Trenkić (ePrint 2026/2527, Theorem 1) show that the
     relative coefficient and canonical embeddings agree up to scaling for
     `Q(ζ₈₁)` over its degree-two subfield `Q(ζ₃)`: their condition holds
     because `rad(81) / rad(3) = 1`. Their module-BKZ analysis (Lemma 7)
     identifies such subfields as favorable under its uSVP heuristics. This
     is a separate structured attack route that does not need the symmetry
     used in ePrint 2021/600. The paper's concrete NTRU+ estimates concern
     a short secret sampled in coefficient geometry; they do not price a
     short kernel of this uniform matrix under the coefficient infinity
     bound `8TB`. The selected module-SIS cost remains open.
     Ducas, Engelberts and de Perthuis (ePrint 2025/1904) predict
     module-BKZ slope gains for odd-conductor cyclotomic fields, but state
     that translating a slope gain into a module-SIS attack cost remains an
     open question. Their asymptotic block-size formula is not a cost for
     this kernel either.
     Bollauf, Pouly and Shen (ePrint 2026/225, Theorem 5) give a Gaussian-
     sampling algorithm for SIS in the infinity norm when the bound is below
     `q/2`, as `8TB` is here. The full coefficient expansion of this ring
     matrix has correlated rows. However, restricting each witness ring word
     to a constant coefficient gives a flat `22 · 54 × blockCount` matrix:
     its columns are independent and uniform in the uniform-ring-key game.
     A nonzero integer kernel vector below `8TB` for this flat matrix is a
     module-SIS kernel by using those integers as constant ring words.
     Thus their random-matrix attack applies to this restricted subproblem
     in that game. Their reported 50-bit gain compares with an older
     provable Wagner algorithm, not with the heuristic BKZ/sieving model
     used for this profile. Their Dilithium examples give no cost for the
     `1,188 × blockCount` subproblem or the fixed SHAKE128-expanded key.
   - If Poseidon2 state-hash collisions are hard to find, then
     `hashCollisions` is small. Collision hardness is also stated for
     algorithms with a strict time bound. Eighteen finders with a strict
     budget of `T` adversary runs, one for each Π_RLC coordinate and one that
     runs the binding reduction, lose at most `(52 Q + 959) / T`:
     `𝔼 chunks, hashCollisions ≤ 18 · hashAdvantage T + (52 Q + 959) / T`
     (`Export.Stage1.RandomOracleHashSolver.hashCollisions_le_solver`).
     `hashAdvantage T` is the success of one finder
     (`Export.Stage1.RandomOracleHashSolver.hashSolve`) that draws the
     chunks, one of the 18 targets and a tape. Each finder is a program that
     holds no adversary: it makes at most `T` runs on every tape
     (`Export.Stage1.RandomOracleHashSolver.hashRuns_length_le`), each run
     depends only on the records of the earlier runs
     (`Export.Stage1.RandomOracleHashSolver.hashRuns_take_congr`), and the
     output only on the records of the runs made
     (`Export.Stage1.RandomOracleHashSolver.hashSolve_congr`). The state
     hash is fixed and has no key, so an
     efficient algorithm that contains a collision exists. A bound therefore
     needs the human-ignorance reading (Rogaway): it applies to algorithms
     that a person can write without such knowledge.
   - Lean bounds neither term numerically.
4. **The history: HyperNova errata Assumption 1.**
   `Export.Stage1.HyperNovaVisitedSecurity.Assumption1` states Assumption 1 in
   the form of HyperNova Definition 7 knowledge soundness of the Poseidon2
   NIFS: for every admitted NIFS adversary (a random tape, and the NIFS input
   and real output it computes) there is an efficient extractor that reads
   the adversary's tape and its own coins, so it may rerun the adversary, and
   that fails after a real success (`NifsRealSuccess.RealSuccess`, with the
   prior-state link) with probability at most `error` of that adversary. Six
   differences from Definition 7:
   - The success event adds the prior-state link (`PriorLink`): the prior
     preimage that the adversary outputs must hash to the digest in the fresh
     public input and be well formed, including canonical children. The bare
     NIFS verifier does not check this, so without the link a prover could
     choose the running statement after `γ`. The deployed system checks it:
     the step circuit of the next step checks the link of every fold, and
     the terminal verifier makes the same check on the terminal pair, which
     no fold uses (`crates/nightstream/src/lifecycle/verify.rs`). Both
     recompute the state hash of the running claims, which rejects a split
     that is not canonical, and compare it with the fresh public input. So
     Assumption 1 is Definition 7 for the NIFS verifier together with this
     check.
   - The transcript is not the one of HyperNova Construction 3, which absorbs
     `sid` (the structure and the public parameters) and the running instance
     before the first challenge. The Nightstream transcript absorbs a domain
     tag, the fresh statement and the prover messages
     (`Lifecycle.TranscriptCoverage.statementCalls`,
     `Lifecycle.TranscriptCoverage.AgreeOnAbsorbed`). The structure, the keys
     and the running instance enter only through the prior digest in the
     fresh public input, a Poseidon2 state hash that carries the
     verifier-context digest. This is the cause of the difference above. So
     `Assumption1` is a premise about the Nightstream transcript, not the
     paper's premise about Construction 3.
   - This joint form implies Definition 7's difference form
     `Pr[success] − Pr[extraction] ≤ error`. The converse needs an adversary
     that stops when its own success check fails.
   - The public parameters are a fixed commitment setup, not sampled by the
     generator. With a fixed hash and a fixed key, an efficient algorithm that
     contains a collision exists. So the premise can hold only for algorithms
     that a person can write without such knowledge (Rogaway's
     human-ignorance reading).
   - The structure is a fixed application. Definition 7 lets the adversary
     choose the structure after the public parameters, and Definition 11 lets
     the IVC prover choose `F` from the public parameters and its coins. With
     fixed public parameters, a choice that depends only on them is one fixed
     application, which the result covers for every application that fits. A
     choice that depends on the adversary's coins is not covered.
   - `error` replaces `negl(λ)`.

   `Export.Stage1.HyperNovaVisitedSecurity.Closed` is the composition premise.
   A second class, `StageAdmitted`, holds whole stages: the reverse extractor
   so far, as one algorithm. An admitted stage gives an admitted NIFS
   adversary, and one reverse step of an admitted stage with an efficient
   extractor is again an admitted stage. The IVC adversary is the start stage,
   which must be admitted. To cover a modified prover, `StageAdmitted` must
   include its arbitrary efficient output algorithm, and `Admitted` must
   include each NIFS algorithm derived from that output. Neither class may
   require proof messages to come from the official prover. `Admitted`,
   `StageAdmitted` and `Efficient` are abstract; their intended meaning is
   expected polynomial time. Lean has no running-time model, so it states no
   false claim about running time, and
   the closure property is a premise. A `goodActive` visit is a real success
   (`Export.Stage1.HyperNovaVisitedSecurity.realSuccess_of_goodActive`).
   Assumption 1 is the paper's plain-model premise, not a theorem. The step
   circuit recomputes the challenges of the previous fold with Poseidon2, so a
   recursive argument uses the concrete hash, which no random-oracle model
   covers. Adversary 3's result motivates the value of `error`, but no Lean
   statement derives one from the other:
   - *Success event.* Assumption 1 uses `NifsRealSuccess.RealSuccess` under
     the stage's own law, a plain-model distribution. The random-oracle event
     equals it only at the sponge reads
     (`Export.Stage1.RandomOracleLink.succeeds_iff_realSuccess`).
   - *Relation instance.* The history checks the statement of
     `Export.Stage1.PiCCSStoredWitnessCheck.statement`, which is the key's
     statement (`Export.Stage1.PiCCSStoredWitnessCheck.statement_eq_key`).
     This difference is closed.
   - *Extractor.* The random-oracle extractor reruns the adversary on a
     changed oracle; the plain-model extractor of Assumption 1 is any
     efficient algorithm that reads the adversary's tape. Neither is derived
     from the other.
   - *Efficiency.* Lean proves that every stage stays in `StageAdmitted`
     under `Closed`. The meaning of that class (expected polynomial time for
     a constant depth, as in Lemma 17) is a premise outside Lean. The hash
     term `h_j` is the probability that stage `j`, an admitted stage, followed
     by the computation of its current visit, outputs a state-hash collision.
     So an external Poseidon2 collision bound applies to it, under the same
     human-ignorance reading. `Efficient` and `StageAdmitted` must have that
     reading too: under the plain expected-polynomial-time reading, the class
     contains an extractor that outputs a known collision, and then `h_j` has
     no useful bound. The paper's truncation argument for
     expected-time stages is outside Lean. Lean chooses the stages with
     `Classical.choose`, so a numerical bound on `h_j` and on `error` must
     hold for the whole class.
   - *Concrete depth.* The sum has one Assumption 1 error for each stage, and
     stage `j + 1` runs the extractor of stage `j`, which may rerun stage `j`.
     So the query count `Q_j` of stage `j` includes every earlier rerun. If
     the plain-model extractor reruns its adversary as the random-oracle
     extractor does (about `17 (Q + 17)` times), then `Q_{j+1} ≈ 17 Q_j²`.
     With the error that adversary 3's result motivates, `(Q_j + 74) ε_test`
     is then above 1 at `j = 1` when `Q_0 = 2^64`, and at `j = 2` when
     `Q_0 = 2^30`. So, with these values, the bound is useful only for a depth
     of one or two folds, and it gives no concrete security for long chains.
     This is the constant-depth limit of Lemma 17; the Lean statement is
     correct for every depth. As a provisional 64-fold planning case, even
     assigning each fold only the `Q = 0` one-fold test term gives the
     arithmetic sum `64 · 74 · 4589 / p² ≈ 2^-103.63`. This is not a
     recursive-history bound: it omits oracle queries, state-hash collisions,
     MSIS and the extractor's growing cost. A production fold cap is not yet
     fixed. Even a hypothetical charge of one `4589/p²` test term per fold
     gives `64 · 4589/p² ≈ 2^-109.84`, so this simple additive accounting
     cannot certify 110 bits at depth 64. This is not a lower bound on an
     attack's success probability.
     Lee and Seo, ePrint 2024/232 (revised 2026-02-13), give an artificial
     IVC scheme that meets bounded-depth knowledge soundness but is forgeable
     at polynomial depth. Their polynomial-depth result for Nova uses the
     extended algebraic group model and a general zero-testing hash premise.
     Neither premise has been established for this Ajtai and Poseidon2
     instantiation, so that result does not close this depth gap.
     Hall-Andersen and Nielsen, ePrint 2022/542 (Theorem 1), show that a
     succinct random-oracle IVC cannot have their stated knowledge-sound and
     zero-knowledge guarantees for every nondeterministic transition while
     keeping sublinear verification. Their theorem constructs a transition
     function and concerns sufficiently long polynomial depth. This verifier
     fixes one application, claims no zero-knowledge guarantee, and has only a
     provisional 64-fold planning case. The theorem therefore gives neither
     a forgery for this application nor a 64-fold bound; it reinforces the
     need to state the exact recursion and oracle assumptions above.
     Nguyen, Boneh and Setty (ePrint 2023/969, Appendix B) break an older
     two-curve Nova verifier by splicing valid current pairs from separate
     chains: its checks do not link a prior fresh pair in one curve to the
     corresponding running pair in the other. The selected Nightstream
     relation has one running family. Its F′ step checks the fresh public
     input against `stateHash(vk, i, z0, zi, U_i)` and checks the NIFS
     transition from that `U_i` to `U_{i+1}`; the terminal binds the current
     state to the running family (`SPEC.md`, Algorithms 3 and 6). This
     excludes that missing-link splice, but is not a separate proof of
     concrete security at depth 64.
     Campanelli, Fiore and Pancholi (ePrint 2025/1413, revised 2026-10-16,
     Theorems 2 and 6) show that an IVC already insecure beyond constant
     depth has no general graceful soundness degradation, and give a way to
     boost depth by constructing a different scheme from short chains.
     Neither result shows this protocol is insecure at 64 folds or gives it
     a 64-fold security bound.
     Maller, Mohnblatt and Zapico (ePrint 2025/1663, Theorem 4) prove
     polynomial-depth HyperNova knowledge soundness in their open-and-sign
     random-oracle model, over a two-curve cycle. Its oracle records
     decompositions of elliptic-curve commitments, and the paper does not
     instantiate it from a plain hash. This Ajtai and Poseidon2 path has no
     such oracle, so that theorem supplies no 64-fold bound here.
     Chiesa, Guan, Samocha and Yogev (ePrint 2023/1646, Theorem 1) give a
     small recursion loss for PCD from a relativized argument with a
     straightline knowledge extractor, including oracle calls in the proved
     computation. The extractor used by
     `Lifecycle.RandomOracleKnowledge.knowledge_error_le` reruns the
     adversary on changed oracle answers, so it is not straightline. Their
     theorem therefore does not remove this query growth or justify the
     provisional 64-fold case.
     Chiesa, Guan, Wang and Wu (ePrint 2026/2209, Theorems 1 and 2) show a
     strict-time black-box rewinding barrier for Kilian-style arguments and
     give an expected-time analysis for interactive, multi-round
     commit-and-open IOP arguments with position-binding commitments. The
     positive theorem does not analyze the Fiat–Shamir transform, this
     Ajtai/Poseidon2 NIFS inside F′, or repeated recursive extraction. It
     therefore supplies no concrete 64-fold bound; its expected-time model
     is consistent with, but does not prove, `StageAdmitted`'s intended
     reading.
     Külaots, Lipmaa, Parisella and Siim (ePrint 2026/326, Theorems 6 and 13)
     obtain tight Fiat–Shamir knowledge soundness from a special-sound
     univariate polynomial IOP and a polynomial commitment with the stated
     evaluation- and degree-binding properties. Theorem 13 treats an
     interactive univariate PCS with corresponding special-binding notions.
     Neither theorem supplies special soundness or those binding premises for
     this rewind-based SuperNeo NIFS or the multilinear WHIR compression.
     Thus this compiler result does not give a 64-fold history bound here.
5. **Rust-to-Lean coverage.** The golden conformance runs and recorded native
   evidence cover their recorded inputs. For the two pinned memory packages,
   executed tests compare every logical matrix row against the Lean artifact.
   No theorem covers arbitrary Rust execution. HyperNova Definition 12
   requires witness-independent, efficiently invertible structure and instance
   encodings, monotonicity, and default satisfying instances for its NIVC
   compiler. The selected base case has Lean proofs, but no theorem shows that
   every public Rust-built application meets those compiler premises. In
   particular,
   `Lifecycle.Stage1.Application.Program.spec_iff` is a premise of the Lean
   application interface: it makes circuit satisfaction imply both the step
   output and `valid`, which
   `Export.Stage1.ActualApplicationStep.selectedRowsZero_implies_decodedValid`
   uses. The public Rust `ApplicationBuilder` writes R1CS records and output
   rows, but no theorem constructs that Lean `Program` from every Rust-built
   application or identifies every Rust row check with Lean `RowsZero`.
   Ozdemir, Wahby, Brown and Barrett (ePrint 2023/778, Appendix G) find a
   field-blasting comparison bug where a prover can choose inconsistent
   decomposition bits to make a range test return false. The selected F′
   `CanonicalU64` gadget constrains all 64 bits, their recomposition and the
   exceptional high-word case (`Gadgets.Range.CanonicalU64.soundness`). The
   memory rows constrain operation, timestamp-difference and scan bits, and
   the catalog gates active addresses to those bits (`Lifecycle.Nebula.Rows.slotRows_iff`,
   `nebula/rows.rs`, `nebula/catalog.rs`). This sampled check finds no such
   false comparison in those paths; it does not prove the general Rust-built
   application interface correct.
   Stephens, Pailoor and Dillig (ePrint 2025/916, Section 10.5) show that a
   deterministic circuit can still accept inputs that its witness generator
   does not produce. For the selected Poseidon2 hash-chain application,
   `Lifecycle.Stage1.Poseidon2HashChainV1.spec_iff` states the exact step
   relation. The Rust release test `application_poseidon2` compares its rows
   and witness values with the saved Lean program on the recorded case and
   rejects changes to an input, output, private input and intermediate word.
   Those four tests passed on 2026-11-01. They do not establish consistency
   for every Rust-built application or every input.
6. **Runtime checks.** The verifier's identity and setup checks stay necessary.
7. **The memory round.** `Export.Stage1.NebulaMemorySecurity.MemoryRound`
   states that, for each stage that trust item 4 admits and the memory
   statement of its tape, the extracted run is accepted and has a segment
   that passes the product test with unbalanced multisets with probability at
   most `error_mem` of that stage. One term covers all segment slots. Lean
   proves the premise when `error_mem` is the probability of the event, so
   its content is that `error_mem` is small. The relation computes `η` with
   Poseidon2 inside the proven step, so no random-oracle proof covers it, as
   for the challenges of trust item 4. The challenge game of adversary 5
   motivates the target `(Q_η + 1) · (2 · m_mem/q² + ε_coll(t_1))`, where
   `Q_η` counts the evaluations of the `η` transcript, in the circuit and
   outside it, and `ε_coll(t_1)` is the Poseidon2 collision probability at
   twice the time of one run. The owner chose this form
   (`decisions/nebula-memory-checking.md`). The record chains need only
   Poseidon2 collision resistance.

## Out of scope

- Campanelli, Chakraborty and Ganesh (ePrint 2026/2631, Theorem 5.10)
  attack simulation soundness of zero-knowledge IVCs whose proofs can be
  merged after the fact. This review's soundness game asks whether the
  official verifier accepts a false state; it makes no simulation-soundness
  or proof-non-malleability claim. The selected HyperNova step links one
  previous running family to one fresh step (`SPEC.md`, Algorithm 3), and
  no proof-merging relation is selected. The paper gives no false-state
  proof for this verifier. A caller that needs non-malleability needs a
  separate security argument.
- The layer-1 WHIR query sampler permits two redraws. It fails if the first
  sample and both redraws equal `p − 1` at one query. This is a completeness
  limit. Successful queries consume the same words as Plonky3 and preserve
  its next transcript state (`CompressionLean.Circuit.Whir.queryIndex`,
  `decisions/whir-query-redraws.md`). No numerical bound for this failure
  event is proved for the concrete Poseidon2 transcript.
- The layer-1 leaf sponge and node compression have an exact cross-mode alias
  on an eight-word leaf: hashing two four-word digests as a leaf equals
  compressing them as a node. The selected setup leaf has 208 words, and the
  selected WHIR leaf has at least sixteen base words or 48 extension words.
  The public verifier fixes these lengths and the path depth, so this alias
  gives no identified public false-proof path. The Merkle binding premise is
  for these fixed shapes; a profile with eight-word leaves would need another
  argument (`formal/compression-lean/TRUST.md`, Merkle binding).
- Fenzi and Sanso (ePrint 2025/2197, Lemma 4.4) give a large-list adversary
  against a toy hash-based argument and say that finding its chosen list
  member appears computationally infeasible. Their high-list estimates use
  proximity near capacity. This verifier selects `JohnsonBound` with a
  positive margin and an explicit list-size charge (`neo-spartan/src/pcs.rs`,
  `p3-security` 0.8.0), so the paper gives no public false proof for this
  profile. Jo (ePrint 2026/891, Corollary 4.6) leaves list size separate for
  row-wise interleaved codes. The selected WHIR witness stacks its tables
  into one polynomial, but each commitment encodes `2^folding` columns as
  a row-wise interleaved Reed–Solomon code (`p3-sumcheck` 0.8.0
  `commit_base`, `p3-whir` 0.8.0 `commit_extension`). Its relative minimum
  row distance is at least `1 − (k − 1)/N` for column degree below `k` and
  domain height `N`. The classical Johnson list bound is independent of
  this vector alphabet at the selected below-Johnson radius, so the width
  does not itself add an uncharged list-size factor. This algebraic check
  does not prove WHIR's numeric report or its Rust-to-Lean premise.
- Completeness does not follow from a knowledge bound: a bound on the
  adversary is not evidence that an honest proof is accepted. The honest
  prover has separate results, for every application that fits the `2^28`
  profile and every commitment setup: an initial statement with a state of
  four words is accepted
  (`Export.Stage1.HyperNovaInitial.initial_accepted`), and an accepted proof
  with advice of the application's witness width that satisfies the
  application's validity predicate gives an accepted next proof
  (`Export.Stage1.HyperNovaAcceptedNext.recursive_extend`,
  `Export.Stage1.HyperNovaAcceptedNext.base_extend`, through
  `Export.Stage1.SelectedAssignmentCompleteness.complete` and
  `Lifecycle.Nifs.BaseCompleteness.zeroProof_verify`). Each application
  promises that its step keeps the state width
  (`Lifecycle.Stage1.Application.Program.step_length`); the counter must stay
  below `p`.
- The error is not one security level. The statistical part costs more
  than `87 · 2^115 / 50 ≈ 2^115.79` oracle queries (addresses) for each unit
  of success, linear in `Q` (`Lifecycle.RandomOracleKnowledge.statisticalError_le`). One
  Poseidon2 state answers up to six addresses, so in permutation calls this
  is about `2^113.2` (computed outside Lean). Even at `Q = 0`, the 74 test
  challenges contribute `74 · 4589/p² ≈ 2^-109.63` to this proof bound
  (`Lifecycle.RandomOracleUniqueness.challenges_length`,
  `Lifecycle.Nifs.VerifierErrorBudget.test_error_eq`). This bound cannot
  certify a 110-bit total error before the other terms. The other terms
  depend on MSIS at the solver's work and on Poseidon2 collisions. 128-bit
  security needs a larger challenge field for `γ`.
  The 74 include 17 Π_RLC challenges. `RandomOracleTest.coinBad` assigns
  them the empty bad set, but `RandomOracleUniqueness.checked` queries all
  74 to locate its stopping point. A separate proof that charged only the
  57 Π_CCS challenges might give a `Q = 0` test term of about `2^-110.003`.
  This is not the current theorem. Even its optimistic sum over 64 folds
  would be about `2^-104.003` before the other errors, below the requested
  110-bit level.
  Rust's profile gate computes one Π_CCS test term and one coordinate-fork
  term (`Params::for_ccs_shape`), and `compression_security_bits` subtracts
  that same estimate from the requested budget. It does not charge the 74
  tests in the random-oracle extraction result. Thus a public verifier can
  accept the 110-bit minimum and derive a compression key even though this
  Lean proof bound already exceeds the requested error budget. This is a
  proof-budget mismatch, not a demonstrated false proof.
- Concrete attacks on Poseidon2 as a random oracle, or on the additive duplex,
  are outside the model. Attacks that use the hash inside the circuit (KRS)
  are outside every random-oracle model; the verifier's package key fixes the
  relation.
  Benedikt and Fischlin (ePrint 2025/1846, Section 8.3) study cheaper
  multiple forgeries of Fiat–Shamir signatures when a sponge hashes the
  message before the signer's commitment. The selected Π_CCS transcript
  absorbs its fixed domain, prior-state digest, fresh commitment and public
  input before its first challenge (`bind_and_sample_with_trace`). The paper's
  signature game and hash-order result neither give a false proof for this
  fixed relation nor bound this transcript's concrete multi-proof security.
- The adversary is a classical algorithm. Quantum adversaries are not modelled.
  The four-field Poseidon2 state digest has about 256 output bits. A generic
  quantum collision search for an ideal uniform digest takes about
  `|F|^(4/3) = 2^85.33` queries, below the selected Ajtai component's
  roughly 110-bit quantum cost (ePrint 2021/1578 and 2026/2135;
  `decisions/piccs-prior-state-digest.md`). A collision between a valid
  one-step state `(1, z, z)` and a false one-step state
  `(1, z', z' + (1, 0, 0, 0))` for the same zero-private-input identity
  application would let a prover substitute the false outer state in a
  public proof: terminal verification checks that state through the digest.
  This is an idealized cost and a conditional attack path, not a concrete
  Poseidon2 collision or a produced false proof.
- No adversary chooses the application or the setup (Applications and
  setups). An application chosen from the adversary's coins, or after the
  setup chunks, is outside the results for adversaries 3 and 4.
- Concrete security for long recursive chains: the history bound is useful
  only for one or two folds (trust item 4, *Concrete depth*).
- Lean states no machine running time and no Rust execution result beyond the
  recorded inputs.
- A program that the adversary chooses after it sees the setup. Adversary 5
  fixes the program before the adversary runs.
- The Rust-built memory relation of an application or geometry that no Lean
  package fixes, the binding rows of the Rust catalog backend, and the
  relation that links the rows of a batched WASM step (`SPEC.md` section
  7.6). Lean proves what the slot plan conditions give
  (`Spec.Nebula.MemoryTrace.reorders_of_slotOrder`) and that batched steps
  are memory consistent exactly when their rows are
  (`Spec.Nebula.MemoryTrace.memoryConsistent_batchCatalog_iff`), but not
  that the Rust rows enforce those conditions. Tests check the rows on fixed
  inputs (trust item 5).
- Values for the adversary time and the query counts `Q` and `Q_η`. They
  stay symbolic (`decisions/nebula-memory-checking.md`).
