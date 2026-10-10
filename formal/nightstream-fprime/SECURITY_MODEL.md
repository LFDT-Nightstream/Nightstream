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

- The kernel of the official Lean v4.32.2 toolchain, with Mathlib v4.32.2 and
  the CompPoly revision that `lakefile.toml` pins. Final validation uses the
  official toolchain. The lean4-optimized fork is used only to iterate.
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
Every adversary result below holds for every application that fits the
`2^28` profile (`Export.Stage1.PerApplicationFixedPoint.FitsTwoPow28`), at its
own relation (`Export.Stage1.PerApplicationFixedPoint.relation`). Adversary 1
holds for
every Ajtai key, adversaries 2 and 4 for every commitment setup
(`Export.Stage1.PerApplicationCanonicalPackage.CommitmentSetup`, with key
`Export.Stage1.PerApplicationCanonicalPackage.commitmentKey`), and adversary 3
draws the key inside its game. `Poseidon2HashChainV1` is the reference
application of the golden runs and the executables.

No adversary chooses the application or the setup: the verifier's package
fixes both before the adversary runs. The verifier-context digest hashes the
relation, the application, the NIFS key and the setup authority (setup
identifier, dimensions and seed)
(`Export.Stage1.PerApplicationCanonicalPackage.verifierContextDescriptor`),
and every state hash carries it. So when one application's verifier accepts a
proof, adversary 2's result gives that application's context or a state-hash
collision.

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
  (`Export.Stage1.ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision`,
  `Export.Stage1.ActualTerminalSecurity.terminal_implies_securityOrCollision`;
  the selected rows come from
  `Export.Stage1.ActualContextSecurity.terminal_implies_rowsAndPublic`). A
  named collision is two different well-formed preimages with one Poseidon2
  hash (`Layout.Stage1.PiCCSSecurity.stateHash_identifies_statement_or_collision`).
  The result is deterministic. It is about the Lean terminal predicate, not
  about the Rust verifier (trust item 5).

### 3. One-fold Fiat–Shamir adversary (random-oracle model)

- **Controls:** a classical algorithm that is a function of the setup chunks,
  so it may read the whole Ajtai key, and that makes at most `Q` queries to
  the oracle `H`. It may choose the running and fresh statements after any
  query. It outputs a claim (the statements, a NIFS proof and 16 child
  witnesses) and a prior preimage. The algorithm is deterministic
  (`Spec.RandomOracle.OracleComp` has no coins). A randomized adversary is
  covered for each fixed value of its coins; the average over the coins is an
  argument outside Lean. Each query is a call list no longer than a fixed
  bound (`Lifecycle.RandomOracleTest.Point`).
- **Does not control:** the application, the setup chunks, which are uniform
  256-bit values (trust item 2), and `H`, a uniform random function of each
  challenge's exact call list (trust item 1). The adversary is chosen before
  the chunks, so it cannot contain a kernel vector of the key.
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
    `ε_sample ≈ 2^-125.4`; at `Q = 2^64` their sum is about `2^-51.84`. These
    two approximations are computed outside Lean.
  - `Export.Stage1.RandomOracleSetup.msisAdvantage`: the success of an
    explicit MSIS solver on a uniform matrix. It runs the binding reduction
    `Lifecycle.RandomOracleBinding.rerunKernel`, which computes a nonzero
    kernel vector of the key with every coordinate below `8TB` from two runs.
  - `2^-190` bounds the cost of a uniform matrix in place of reduced chunks
    for every application that fits
    (`Export.Stage1.RandomOracleSetup.programmingError_lt_of_fits`).
- **Work:** the extractor takes `17 (Q + 17)` expected retries
  (`Lifecycle.RandomOracleExtraction.expected_retries_le`), and the binding
  reduction takes `Q + 74` expected reruns
  (`Lifecycle.RandomOracleUniqueness.expected_reruns_le`). Lean does not bound
  the retries of the second extraction in the binding reduction. Lean counts
  runs, not machine work.
- **What it covers:** the oracle verifier is the Lean NIFS verifier
  `Spec.Folding.Nifs.PaperNonInteractive.verify`, with its coins read from
  `H`, plus the child openings. At an oracle that answers the 74 challenge
  points of one execution as the sponge does, the two agree
  (`Lifecycle.RandomOracleFidelity.accepts_iff_verify`), and every execution
  has such an oracle (`Lifecycle.RandomOracleFidelity.deployedOracle_deployed`).
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
fresh verifier coin by an oracle read. Each module header states its part:

| Lemma | Step | Lean |
|---|---|---|
| 1 | Each challenge reads its exact call list; distinct challenges have distinct call lists. | `Lifecycle.TranscriptCoverage.challenge_seal`, `Lifecycle.TranscriptCoverage.challengeCalls_injective` |
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
  application witnesses of the circuit's witness length take `z0` to `zi` in
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

## What you trust

1. **The random-oracle model of the challenge reads.** The NIFS verifier
   reads each challenge from the Poseidon2 sponge state after the exact call
   list of that challenge. The deployed system makes these reads only inside
   the step circuit, so this model applies to adversary 3's game, not to a
   deployed check. `Lifecycle.TranscriptCoverage.piCcsProbe_coins`
   and `Lifecycle.TranscriptCoverage.rho_seal` prove that the coins are one
   read of each call list. The model replaces that read by a uniform random
   function of the call list (`Lifecycle.RandomOracleTest.coins`). This is a
   modelling assumption, and a stronger one than Ironwood's. Ironwood
   idealizes one hash call on the whole transcript. A duplex read is related
   to the reads after nearby call lists: an extension challenge reads two
   permutation outputs, and the second is the first word of the next read
   (`Lifecycle.TranscriptCoverage.readK`). The standard justification would be
   duplex-sponge Fiat–Shamir in the ideal-permutation model; Lean does not
   prove it. The same sponge also computes the state hash, which the theorem
   treats as a concrete function. The two uses absorb different first words
   (their domain tags), but no Lean statement separates them. Each Π_RLC
   scalar is the existing sampler applied to an oracle answer; its law is
   within `2^-132` of uniform on the strong set
   (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt`).
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
     application that fits, and 835,936 for `Poseidon2HashChainV1`. The
     solver runs the adversary an expected number of times that grows with
     `Q`, and Lean does not bound its work. MSIS hardness is stated for
     solvers with a strict time bound, so this step also needs a Markov
     truncation of the expected work, whose loss Lean does not count. The
     2026-09-08 public-seed MSIS approval covered the ChaCha20 matrix only.
     For ordinary binding collisions under the SHAKE128 matrix of the
     reference application,
     `Export.Stage1.Poseidon2HashChainV1Setup.production_binding_lt_solver` gives an
     MSIS solution for a uniform matrix, with error below `2^-190`.
   - If Poseidon2 state-hash collisions are hard to find, then
     `hashCollisions` is small. The state hash is fixed and has no key, so an
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
     the terminal verifier recomputes the state hash of the running claims,
     which rejects a split that is not canonical, and compares it with the
     fresh public input (`crates/nightstream/src/lifecycle/verify.rs`), and
     for an inner fold the step circuit recomputes it. So Assumption 1 is
     Definition 7 for the NIFS verifier together with this check.
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
   which must be admitted. `Admitted`, `StageAdmitted` and `Efficient` are
   abstract; their intended meaning is expected polynomial time. Lean has no
   running-time model, so it states no false claim about running time, and
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
     correct for every depth.
5. **Rust and Lean agree on recorded inputs only.** The golden conformance runs
   and the recorded native evidence cover their recorded inputs. No theorem
   covers arbitrary Rust execution.
6. **Runtime checks.** The verifier's identity and setup checks stay necessary.

## Out of scope

- Completeness does not follow from a knowledge bound: a bound on the
  adversary is not evidence that an honest proof is accepted. The honest
  prover has separate results:
  `Export.Stage1.SelectedAssignmentCompleteness.complete`,
  `Export.Stage1.HyperNovaInitial.initial_accepted`,
  `Export.Stage1.HyperNovaAcceptedNext.recursive_extend` and
  `Lifecycle.Nifs.BaseCompleteness.zeroProof_verify`.
- The error is not one security level. The statistical part costs about
  `2^115.8` work for each unit of success, linear in `Q` (computed outside
  Lean). The other terms
  depend on MSIS at the solver's work and on Poseidon2 collisions. 128-bit
  security needs a larger challenge field for `γ`.
- Concrete attacks on Poseidon2 as a random oracle, or on the additive duplex,
  are outside the model. Attacks that use the hash inside the circuit (KRS)
  are outside every random-oracle model; the verifier's package key fixes the
  relation.
- The adversary is a classical algorithm. Quantum adversaries are not modelled.
- No adversary chooses the application or the setup (Applications and
  setups). An application chosen from the adversary's coins, or after the
  setup chunks, is outside the results for adversaries 3 and 4.
- Concrete security for long recursive chains: the history bound is useful
  only for one or two folds (trust item 4, *Concrete depth*).
- Lean states no machine running time and no Rust execution result beyond the
  recorded inputs.
