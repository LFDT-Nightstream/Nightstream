# Security model

This page states which adversaries the Nightstream F′ Lean results protect
against: what each adversary controls, when it wins, the Lean result that
bounds it, and what you must trust. It does not restate proofs; read the cited
declaration. `SPEC.md` describes the protocol: two step circuits, a light
circuit and a refresh circuit, that fold into two slots.

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
- Strong-set low-norm invertibility (SuperNeo v1.2 Theorem 4) is proved
  (`Spec.Phi81StrongSet.lowNormInvertibility`), so no security result takes it
  as a premise.

## Applications and setups

An application is a step program (`Lifecycle.Stage1.Application.Program`).
Both step circuits run the same application step (`Lifecycle.OptionB.machine`).
Adversary 1 holds for every application at the relation that Lean exports for
it, and adversary 2 for every setup whose circuit keys carry these relations.
Adversaries 3 and 4 hold for every key of the light and refresh folds.
`Poseidon2HashChainV1` (`Lifecycle.Stage1.Poseidon2HashChainV1.program`) is
the reference application of the packages and the executables.

No adversary chooses the application or the setup. The verifier's packages
fix the application, the fixed public seed of `SPEC.md` Algorithm 1 fixes the
commitment key, and the verifier pins the identity of the packages (`SPEC.md`
Algorithm 2). A memory application (adversary 5) is such an application: its
packages also fix the memory geometry, and the verifier takes the program,
which the adversary does not choose. The identity is the Poseidon2 hash of the
encoding of the light and refresh packages
(`Export.OptionB.StepPackage.identity`), and every format decodes what it
encodes. No Lean theorem states that equal identities give equal packages or
a Poseidon2 collision.

The state hash binds the key digest of each circuit
(`Lifecycle.OptionB.statePreimage`). Rust computes each key digest as the
Poseidon2 hash of a domain, the identity and the circuit index, and it
derives each circuit's relation from its package (`SPEC.md` section 4.4, not
checked against Lean). So when the verifier of one pair of packages accepts a
proof, adversary 2's result gives that verifier's keys or a Poseidon2
collision, if each key carries the relation that Lean exports. The Rust tests
compare Rust's expansion of each package with Lean's row checksum (trust
item 5).

## Adversaries

### 1. Malicious step assignment

- **Controls:** every value of the assignment of a step circuit and the
  claimed public input.
- **Wins:** the rows of the circuit's relation are zero and the public input
  encodes a digest, but the circuit's case of the step relation does not hold
  with that digest as its output hash.
- **Result:** no such win, for the light circuit
  (`Export.OptionB.StepRelation.light_stepHolds`) and for the refresh circuit
  (`Export.OptionB.StepRelation.refresh_stepHolds`). The step holds in an
  environment that agrees with the assignment on the public input and
  differs only on dead Poseidon2 locals. The package rows are the relation's
  rows (`Export.OptionB.StepPackage.relationRows_packageOf`), and the rows of
  either circuit give its case of the step relation
  (`Layout.OptionB.StepCircuits.soundness`). Each result takes as a premise
  that every valid step of the circuit's application has the selector value
  of that circuit. The selector that returns the circuit that runs
  (`Export.OptionB.TerminalBridge.chosen`) meets this premise for every
  application (trust item 4). The decoded key digests are the words that
  the assignment carries; adversary 2 binds them to the verifier's keys. The
  result is deterministic and needs no hardness premise.

### 2. Malicious terminal opening

- **Controls:** the terminal statement and every value of the proof: the
  program counter, both slots with their witnesses, and the fresh claim with
  its witness.
- **Wins:** the terminal relation accepts, but no step holds under the
  verifier's keys that ends in the statement.
- **Result:** acceptance gives the step that the fresh witness decodes to
  (`Export.OptionB.TerminalBridge.Exports.decode`). The step relation holds
  on it at the production cap, under the setup's keys, and it ends in the
  terminal's statement, or its next state and the terminal's state collide
  (`Export.OptionB.TerminalBridge.bridge`,
  `Export.OptionB.TerminalBridge.Collision`). The terminal is
  `Lifecycle.OptionB.Terminal.HoldsFor`. Two well-formed states with one state
  hash are equal or give a Poseidon2 collision on an input pair that the two
  states fix (`Lifecycle.OptionB.stateHash_collision`,
  `Lifecycle.OptionB.StateCollision`). The bridge reads each counter as
  one word, so it gives the depth words and not the depths
  (`Export.OptionB.TerminalBridge.stateHash_binds_words`). The result is
  deterministic. It is about the Lean terminal relation, not about the Rust
  verifier (trust item 5). It concludes the step relation with the selector
  that returns the circuit that ran, and the terminal does not read the
  selector (`Export.OptionB.TerminalBridge.holdsFor_recursive_control`), so
  its premises hold for the reference packages, whose two circuits run one
  application. The decode is a function of the proof, so the collision names
  two states that a reduction computes from the proof.

### 3. One-fold adversary

- **Controls:** the accumulator, the fresh claim and the proof of one light
  fold or one refresh fold, for a fixed key.
- **Wins:** the fold verifier accepts two outputs that differ only in the
  Π_RLC challenges (a fork), and extraction from the fork does not give the
  witnesses.
- **Result:** for a light fold, the extracted fresh witness opens the fresh
  claim with norm below `b`, the extracted accumulator witness opens the
  input accumulator at the output bound plus `T(b − 1)`, and the claims are
  true at the prior point, or a Π_CCS bad event occurred: a mixing root of
  `α` and `γ`, or a sum-check failure
  (`Spec.Folding.Light.accepted_implies_securityOutcome`; Cyclo, ePrint
  2026/359, Theorem 3, item 2; `Lifecycle.OptionB.lightExtractionNormLaw`).
  For a refresh fold, the extracted fresh witnesses satisfy the CCS and the
  recomposed children open the input accumulator below `B`, or a Π_CCS bad
  event occurred (`Spec.Folding.Refresh.accepted_implies_securityOutcome`).
  The fork outputs of a refresh fold need no norm bound. These results are
  deterministic.
- **Binding:** two openings of one commitment at a recovered bound give a
  kernel vector of the Ajtai key below the chain norm
  (`Spec.Phi81Relation.PiRLCAlgebra.Binding.chainCollision_to_shortKernel`,
  `Lifecycle.Nifs.BindingBridge.differenceBounded`). Every bound that a
  history recovers is at most `B + cap · T · (b − 1)`
  (`Spec.Folding.Schedule.recoveredBounds_le`). Trust item 2 gives the
  hardness premise.
- **Error terms:** the light Π_CCS test numerator is 524, against 4,589 for
  the refresh Π_CCS, and a light fork has one `ρ` coordinate, against 17 for
  a refresh fork (`Spec.Folding.Light.ErrorBudget.production_testNumerator`,
  `Spec.Folding.Light.ErrorBudget.production_forkCoordinates`,
  `Lifecycle.Nifs.VerifierErrorBudget.test_error_eq`). Each numerator is a
  per-call count over `q²`.
- **Not covered:** Lean has no random-oracle result for the light or the
  refresh fold. It does not bound the probability that a fork fails or that a
  Π_CCS bad event occurs in a Fiat–Shamir game, and it does not join these
  counts into one error.

### 4. IVC adversary

- **Controls:** a random tape and, from it, a terminal statement and every
  value of a proof, for a fixed application, setup and pair of exported step
  circuits. The adversary must be admitted (trust item 3).
- **Wins:** the terminal relation accepts, and no `iteration` steps of the
  application lead from `z0` to `zi` (a false acceptance).
- **Result:** under Assumption 1 and the closure premise,
  `Pr[false acceptance] ≤ Σ_j (Pr[collision at visit j] + error(stage j))`
  for `j` below a bound on the adversary's iteration
  (`Export.OptionB.IvcSecurity.probability_bound`). The reverse extractor
  walks back from the terminal one step per stage
  (`Export.OptionB.IvcWalk`). At each visit, the fresh witness decodes to the
  last step (`Export.OptionB.TerminalBridge.Exports.decode`). When that step
  is recursive, the stage's attempt is its fold of the slot that its prior
  program counter selects, linked to the step's prior state, with the slot's
  witness at the slot's bound. An answer that extracts the fold gives the
  prior visit, whose statement and slots are the step's input. Each slot keeps
  its own bound: a light fold adds `T(b − 1)` and a refresh fold returns it to
  `B`, so each attempt stays within `B + cap · T · (b − 1)`
  (`Export.OptionB.IvcWalk.DepthWithin`). At a visit without a collision, an
  answer that extracts gives a valid prior visit one iteration back, or the
  last step is the base step (`Export.OptionB.IvcWalk.walk_step`). So a false
  acceptance has a collision or a failed extraction at one of its first
  `iteration` visits (`Export.OptionB.IvcWalk.covered`).
- **Collision term:** the collision at a visit is a Poseidon2 collision on
  the next state of the decoded step and the visit's terminal state
  (`Export.OptionB.TerminalBridge.Collision`). The stage computes both states
  from its tape, so a reduction that runs the stage outputs the colliding
  pair. Lean does not bound this probability (trust item 2).
- **Per slot:** the earlier one-slot result,
  `Spec.HyperNova.Construction2.BoundedDepthSecurity.history_failure_le`,
  takes the history as an adversary output and has no collision term. Without
  probabilities, the reverse run of one slot opens each running claim at its
  recovered bound, or it names a failed fold
  (`Spec.HyperNova.Construction2.BoundedDepthKnowledge.execution_extracts`).
- **Gaps:** Lean does not derive the value of `error`. Adversary 3's counts
  are deterministic, and no random-oracle result for the light and refresh
  folds discharges Assumption 1. The advice of a false acceptance is any list
  of witnesses that `step` maps from `z0` to `zi`. The step relation does not
  carry the application's validity predicate, so the result does not say that
  the witnesses are valid for the application.

### 5. Memory adversary

- **Controls:** what adversary 4 controls, for a memory application at a
  fixed geometry and program, and a claimed memory statement and final carry.
- **Wins:** the memory verifier accepts (`SPEC.md` Algorithm 10), and the
  history that an extractor returns does not attest the claim: it is not an
  execution from the claim's initial words and the program's memory to its
  final words in which each memory access reads the last value written.
- **Result:** no probabilistic Lean result covers the two step circuits.
  `Lifecycle.Nebula.EndToEnd.bad_subset` reduces a win to a failure of the
  history extraction, a state-link collision, a chain collision, or an
  accepted run with an unbalanced segment that passes the product test, for
  every extraction that meets `Lifecycle.Nebula.EndToEnd.Stage1Extraction`.
  Lean proves that premise for no extractor of the two step circuits.
- **Deterministic part:** an accepted run gives an execution, a Poseidon2
  collision among the run's own chain inputs and the `D_init` chain, or a
  segment that passes the product test with unbalanced multisets
  (`Spec.Nebula.soundness`). With the state links, an accepted chain of valid
  steps gives such a run or a state-digest collision
  (`Lifecycle.Nebula.accepts_or_collision`).
- **Challenge game:** in the interactive game where `η` comes after the
  roots and `ts`, segment slot `k` is bad with frequency at most
  `2 · m_mem/q²` and a retry term, and each retry disagreement is a Poseidon2
  transcript collision (`Lifecycle.Nebula.badAt_bound`). Here
  `m_mem = R + M + N · B_ops`, and setup rule 6 keeps `2 · m_mem/q²` at most
  `2^−109.91` (`Spec.Nebula.Geometry.secure_iff`).

### 6. Final-proof adversary

- **Controls:** a final proof and the statement, against the verifier's
  compression key.
- **Wins:** the final verifier accepts, but no last fold exists whose finish
  checks hold and whose two slot claims hold under their circuits' relations.
- **Result:** the claims of `formal/compression-lean`
  (`CompressionLean.Claims.option_b_parent_bound`): such a win has probability
  at most `calls / 2^t` in the state-restoration experiment at Rust's layer
  bits `t`. For the reference application, a Lean proof, a Lean check of
  Rust's files, Rust or a Lean value meets each premise
  (`CompressionLean.Claims.option_b_poseidon2_bound`,
  `formal/compression-lean/STATUS.md`). The final program's checks on the
  finish package give the finish relation of the last fold
  (`Export.OptionB.FinishRelation.package_finish`). Trust item 7 lists what
  this result assumes.

## What you trust

1. **SHAKE128 as a random oracle (premise P1).** Each key coefficient is one
   SHAKE128 output chunk reduced modulo `q`
   (`Spec.AjtaiSetupV1.Programming.verifierKey_eq`). With uniform chunks, an
   event has at most its probability under a uniform matrix, plus
   `2 · (2^32 − 1)/2^256` for each coefficient
   (`Spec.AjtaiSetupV1.Programming.expect_le_programmed`), so MSIS for a
   uniform matrix applies (`decisions/fprime-ajtai-shake128-setup.md`,
   `decisions/fprime-ajtai-shake128-security-argument.md`). The seed is
   fixed and public, so an adversary that is precomputed for that seed is
   outside this premise. Lean does not join this step to the results above.
2. **Two hardness terms.**
   - *MSIS at the chain norm.* Binding needs MSIS for a uniform matrix of
     `κ = 22` rows of elements of `F_q[X]/Φ₈₁` (degree 54) at the chain norm
     `msisChainBound = 8 · T · (B + cap · T · (b − 1)) = 137,134,080`, about
     `2^27.03` (`Spec.Folding.Schedule.production_msisChainBound`). It is the
     SuperNeo v1.2 Section 8 norm `8TB` with `B` replaced by the largest
     recovered bound (`Spec.Folding.Schedule.msisChainBound_zero`). The
     decider checks each slot below `B`, and an honest slot stays below
     `accumulatorBound cap = 17,496` (`Spec.GlobalParams.accumulatorBound`,
     `SPEC.md` Algorithm 7). The recovered bound, and so the chain norm,
     comes from the decider bound `B`, not from the honest bound.
   - *The κ estimate.* A lattice-estimator run outside Lean (commit 38618db),
     with the method and the matrix of 22 × 4,708,530 ring elements of the
     baseline in `decisions/fprime-ajtai-public-seed-msis-assumption.md`,
     gives these costs at the chain norm: Core-SVP 123.22 bits classical and
     111.83 bits quantum (block size 422), and MATZOV 150.69 bits classical.
     At `8TB` the same method gives 125.85, 114.22 and 153.20. The owner kept
     `κ = 22` on 2026-10-31. The models are heuristic generic lattice attacks.
     They are not a proof for structured Module-SIS or for the fixed public
     seed, and the run's files are not in the repository.
   - *Poseidon2 collisions.* The state hash, the slot digests, the package
     identity, the Nebula record chains and the state digests of a memory
     application need Poseidon2 collision resistance. Each hash is fixed and
     has no key, so an efficient algorithm that contains a collision exists.
     A bound therefore needs the human-ignorance reading (Rogaway): it
     applies to algorithms that a person can write without such knowledge.
   - Lean bounds neither term numerically.
3. **The history: HyperNova errata Assumption 1 for the two fold kinds.**
   `Spec.HyperNova.ReverseStages.Assumption1` states Assumption 1 in the form
   of HyperNova Definition 7 knowledge soundness. Adversary 4 uses it with the
   success event `Export.OptionB.IvcWalk.WalkSucceeds` and the extraction
   event `Export.OptionB.IvcWalk.WalkExtracted`; the one-slot result uses
   `Spec.HyperNova.Construction2.BoundedDepthSecurity.Assumption1` with the
   same events.
   For every admitted fold adversary there is an efficient extractor that
   reads the adversary's tape and its own coins, so it may rerun the
   adversary. After a success, the extractor fails with probability at most
   `error` of that adversary. The adversary outputs an attempt: a fold, the
   running claim that it starts from, and a witness of its output at a
   bound. One premise covers both fold kinds. The attempt names its circuit,
   and a success needs the bounded-depth verifier of that circuit to accept,
   so a light fold at the cap is not a success. A success also needs an
   output bound of at most `B + cap · T · (b − 1)`, so the premise is needed
   only up to the chain norm of trust item 2
   (`Spec.HyperNova.Construction2.BoundedDepthReverse.Succeeds`). Extraction
   must open the fold's input at the output bound plus `T(b − 1)` after a
   light fold, and below `B` after a refresh fold, and it must open the fresh
   claim (`Spec.HyperNova.Construction2.BoundedDepthReverse.Extracted`,
   `Spec.Folding.Schedule.extractedBefore`). These are the differences from
   Definition 7:
   - The success event has a prior-state link. The light transcript does
     not absorb the running claim (`Lifecycle.OptionB.lightAbsorbPublicInput`),
     and the refresh transcript absorbs the children's commitments and
     evaluations but not the accumulator's point or public input
     (`Lifecycle.OptionB.refreshWords`). Both bind the accumulator only
     through the prior digest in the fresh public input. So the adversary
     also outputs a prior state, and a success needs `PriorLink` of that
     state, the running claim and the fold
     (`Spec.HyperNova.Construction2.BoundedDepthReverse.RunsBack`). The
     prior state is an output, as in PR #155: a link to some prior state
     holds for almost every running claim, because many states share one
     digest. In an accepted execution, each fold of a slot links to the
     prior hash preimage of its step: the fresh public input encodes the hash
     of that preimage, and its program counter selects the slot that holds
     the fold's running claim
     (`Spec.HyperNova.Construction2.BoundedDepthSecurity.StepLink`,
     `Spec.HyperNova.Construction2.BoundedDepthSecurity.slot_runsBack`).
     Option B's link also requires a well-formed prior state
     (`Lifecycle.OptionB.PriorLink`). Two linked folds with one fresh instance
     then start from one running claim, or their two prior states give a
     state-hash collision (`Lifecycle.OptionB.priorLink_binds`). So the
     transcript fixes the running claim before the challenges, up to a
     collision on a pair of states that the adversary outputs.
   - The relations have a key. A claim holds under the key of its slot, and
     every fold of an accepted history uses the slot's key.
   - The transcript is not the one of HyperNova Construction 3, which absorbs
     `sid` (the structure and the public parameters) and the running instance
     before the first challenge. The fold transcripts absorb a domain block,
     the fresh statement, the children of a refresh fold and the prover
     messages (`SPEC.md` section 5.2). The structure, the keys and the
     accumulator enter only through the prior digest. So `Assumption1` is a
     premise about the Nightstream transcripts, not the paper's premise about
     Construction 3.
   - This joint form implies Definition 7's difference form
     `Pr[success] − Pr[extraction] ≤ error`. The converse needs an adversary
     that stops when its own success check fails.
   - The public parameters are a fixed commitment setup, not sampled by the
     generator. With a fixed hash and a fixed key, an efficient algorithm that
     contains a collision exists. So the premise can hold only for algorithms
     that a person can write without such knowledge.
   - The structure is a fixed application. Definition 7 lets the adversary
     choose the structure after the public parameters, and Definition 11 lets
     the IVC prover choose `F` from the public parameters and its coins. A
     choice that depends on the adversary's coins is not covered.
   - `error` replaces `negl(λ)`.

   `Spec.HyperNova.ReverseStages.Closed` is the composition premise. A second class, `StageAdmitted`, holds whole stages:
   the reverse extractor so far, as one algorithm. An admitted stage gives an
   admitted fold adversary, and one reverse step of an admitted stage with an
   efficient extractor is again an admitted stage. The IVC adversary is the
   start stage, which must be admitted. `Admitted`, `StageAdmitted` and
   `Efficient` are abstract; their intended meaning is expected polynomial
   time. Lean has no running-time model, so it states no false claim about
   running time, and the closure property is a premise. Lean chooses the
   stages with `Classical.choose`, so a numerical bound on `error` must hold
   for the whole class. Assumption 1 is the paper's plain-model premise, not
   a theorem. The step circuits recompute the challenges of the previous fold
   with Poseidon2, so a recursive argument uses the concrete hash, which no
   random-oracle model covers. Adversary 3's error counts motivate the value
   of `error`, but no Lean statement derives one from the other. The sum has
   one Assumption 1 error for each fold of the slot, and stage `j + 1` runs
   the extractor of stage `j`, which may rerun stage `j`. So the cost of a
   stage grows with its depth. This is the constant-depth limit of HyperNova
   Lemma 17: the Lean statement holds for every depth, but it gives no
   concrete security for a long history.
4. **The circuit selection.** HyperNova NIVC selects the circuit of a step
   with `φ(z_i, w_i)`. The Lean step relation takes it from the machine's
   selector of the application state and witness (`Lifecycle.OptionB.machine`,
   `Lifecycle.OptionB.stepHolds_iff`). Both option-B circuits run the same
   application step, so the selector does not change the computed state.
   `Export.OptionB.TerminalBridge.bridge` takes the selector that returns the
   circuit that ran (`Export.OptionB.TerminalBridge.chosen`), so the prover
   picks the circuit. Rust picks it from the depth of the selected slot
   (`SPEC.md` Algorithm 5): a light fold below the cap and a refresh fold at
   the cap. The light depth guard rejects a light fold at the cap for every
   pick. The statement `(i, z0, zi)` does not fix the circuit of a step; the
   state hash binds the program counter of the last step only.
5. **Rust and Lean agree on recorded inputs only.** The Rust tests compare
   Rust's expansion of the packages, its identity code, its witness program
   and its logical assignment with Lean exports, and they check the
   lifecycle on fixed inputs (`SPEC.md` section 9). No theorem covers
   arbitrary Rust execution.
6. **The memory round.** No Lean premise states the memory round for the two
   step circuits. Its target is this: for each admitted stage and the memory
   statement of its tape, the extracted run is accepted and has a segment
   that passes the product test with unbalanced multisets with probability at
   most `error_mem` of that stage. The relation computes `η`
   with Poseidon2 inside the proven step, so no random-oracle proof covers
   it. The challenge game of adversary 5 motivates the target
   `(Q_η + 1) · (2 · m_mem/q² + ε_coll(t_1))`, where `Q_η` counts the
   evaluations of the `η` transcript, in the circuit and outside it, and
   `ε_coll(t_1)` is the Poseidon2 collision probability at twice the time of
   one run. The owner chose this form (`decisions/nebula-memory-checking.md`).
   The record chains need only Poseidon2 collision resistance.
7. **The compressed proof.** `formal/compression-lean/TRUST.md` lists the
   assumptions of adversary 6: WHIR openings are idealized, Poseidon2 is a
   random oracle, Fiat–Shamir runs inside the final program, and the layer-1
   key's root is the honest root of the claims' setup. The premises about the
   keys and the setups are met for the reference application only
   (`formal/compression-lean/STATUS.md`).

## Out of scope

- Completeness does not follow from a knowledge bound: a bound on the
  adversary is not evidence that an honest proof is accepted. The honest
  prover has separate results: an honest light fold and an honest refresh
  fold are accepted (`Spec.Folding.Light.complete`,
  `Spec.Folding.Refresh.complete`), each accumulator of an admissible
  schedule stays below `B` (`Spec.Folding.Schedule.honest_bounds`), and an
  honest step of either circuit has an assignment in which every row holds
  (`Layout.OptionB.LightAssemblerCompleteness.complete`,
  `Layout.OptionB.RefreshAssemblerCompleteness.complete`). Lean does not join
  these results into a statement that the honest prover's proofs are
  accepted. Each application promises that its step keeps the state width
  (`Lifecycle.Stage1.Application.Program.step_length`); the counter must stay
  below `q`.
- The error is not one security level. Lean bounds no fold error in a
  Fiat–Shamir game (adversary 3), and the other terms depend on MSIS at the
  solver's work and on Poseidon2 collisions.
- Concrete attacks on Poseidon2 as a random oracle, or on the additive duplex,
  are outside the model. Attacks that use the hash inside the circuit (KRS)
  are outside every random-oracle model; the verifier's packages fix the
  relation.
- The adversary is a classical algorithm. Quantum adversaries are not
  modelled. The κ estimate of trust item 2 gives a quantum Core-SVP cost, but
  no Lean result uses it.
- No adversary chooses the application or the setup (Applications and
  setups). An application chosen from the adversary's coins, or after the
  seed, is outside the results.
- Concrete security for long recursive histories (trust item 3).
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
- Values for the adversary time and the query counts `Q_η`. They stay
  symbolic (`decisions/nebula-memory-checking.md`).
