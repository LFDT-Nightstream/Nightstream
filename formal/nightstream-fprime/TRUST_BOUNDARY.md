# Trust boundary

This page states what a reader must trust to accept the Nightstream F′
security results, and which Lean declaration states each result. Names omit
the common `NightstreamFPrime` namespace. `tests/EndpointCensus.lean` checks
that every full name on this page, on [the assurance surface](ASSURANCE_SURFACE.md)
and in [the ROM note](../../docs/reviews/nightstream-fprime-requirements/ROM_KNOWLEDGE_SOUNDNESS.md)
exists, and that every cited theorem uses only `propext`, `Classical.choice`
and `Quot.sound`.

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

## The one-fold knowledge endpoint

`Lifecycle.RandomOracleKnowledge.knowledge_error_le` is knowledge soundness of
one production NIFS fold in the random-oracle model. The extractor succeeds
with probability at least the acceptance probability minus
`Lifecycle.RandomOracleKnowledge.knowledgeError`.
`Export.Stage1.RandomOracleLink.knowledge_error_le_linked` is the same theorem
when the verifier also checks the prior-state link (`PriorLink`); then two of
the three named events are state-hash collisions (below).
`Lifecycle.RandomOracleKnowledge.contract` states the result as a
`Spec.KnowledgeContract`, after Ironwood's record:

| Question | Answer | Lean |
|---|---|---|
| 1. What is a run? | A uniform random function on bounded call lists, then the extractor's retries. The adversary has at most `Q` queries and selects its statement. | `Lifecycle.RandomOracleKnowledge.runWeight` |
| 2. When does the verifier accept? | The production NIFS verifier accepts with every coin read from the oracle, the adversary's witnesses open the 16 returned children, and the claim passes its `linked` check. At the sponge reads this is `PaperNonInteractive.verify`, and with `PriorLink` it is `NifsRealSuccess.RealSuccess`. | `Lifecycle.RandomOracleExtraction.Succeeds`, `Lifecycle.RandomOracleFidelity.accepts_iff_verify`, `Export.Stage1.RandomOracleLink.succeeds_iff_realSuccess` |
| 3. What does extraction return? | The Π_CCS output witness of the complete Π_RLC fork, or nothing. | `Lifecycle.RandomOracleKnowledge.extract` |
| 4. What does a returned witness certify? | `SourceHolds` for the running and fresh statements that the adversary selected. | `Spec.KnowledgeContract` |
| 5. What is the failure event? | The verifier accepts, and extraction returns nothing. | `Spec.KnowledgeContract` |
| 6. What is the error? | `17 (Q + 17) ε_sample + (Q + 74) ε_test`, plus three named events (below). | `Lifecycle.RandomOracleKnowledge.knowledgeError` |

The extractor takes `17 (Q + 17)` expected retries
(`Lifecycle.RandomOracleExtraction.expected_retries_le`). The binding
reduction takes `Q + 74` expected reruns
(`Lifecycle.RandomOracleUniqueness.expected_reruns_le`;
`Lifecycle.RandomOracleUniqueness.challenges_length` pins the 74). With
`ε_test = 4589/p² ≈ 2^-115.84`
(`Lifecycle.Nifs.VerifierErrorBudget.test_error_eq`) and
`ε_sample ≈ 2^-125.4`, the statistical part at `Q = 2^64` is about
`2^-51.84`. The proof is in
[the ROM note](../../docs/reviews/nightstream-fprime-requirements/ROM_KNOWLEDGE_SOUNDNESS.md).

## What you trust

1. **The random-oracle model of the challenge reads.** The deployed verifier
   reads each challenge from the Poseidon2 sponge state after the exact call
   list of that challenge. `Lifecycle.TranscriptCoverage.piCcsProbe_coins`
   and `Lifecycle.TranscriptCoverage.rho_seal` prove that the coins are one
   read of each call list. The model replaces that read by a uniform random
   function of the call list (`Lifecycle.RandomOracleTest.coins`). This is a
   modelling assumption, as Ironwood's abstract squeeze is. It is not a
   theorem about Poseidon2 or the additive duplex. Nothing else in the
   verifier changes: at an oracle that answers the 74 challenge points of one
   execution as the sponge does, the oracle verifier is
   `PaperNonInteractive.verify` plus the child openings
   (`Lifecycle.RandomOracleFidelity.accepts_iff_verify`), and every execution
   has such an oracle (`Lifecycle.RandomOracleFidelity.deployedOracle_deployed`).
2. **Three named events.**
   - `Lifecycle.RandomOracleUniqueness.collisionChance` is the success of the
     binding reduction: two different witnesses for one running and fresh
     statement. Lean turns every such rerun into a `(2B, C)`-relaxed binding
     collision on one input commitment (SuperNeo v1.2 Appendix B), and then
     into a nonzero kernel vector of the same Ajtai key with every coordinate
     below `8TB`
     (`Lifecycle.RandomOracleBinding.collides_relaxedBindingCollision`,
     `Lifecycle.RandomOracleBinding.rerun_shortKernel`). For the production
     key this is
     `Export.Stage1.Poseidon2HashChainV1Setup.productionRelaxedBindingCollision_to_shortKernel`,
     which extends to the fixed-seed instance with the earlier approved
     dimensions
     (`Export.Stage1.Poseidon2HashChainV1Setup.productionShortKernel_to_approvedMsis`).
     The premise is MSIS for that public-seed matrix. The 2026-09-08
     public-seed MSIS approval covered the ChaCha20 matrix only. For the
     current SHAKE128 matrix, Lean reduces ordinary binding collisions to MSIS
     for a uniform matrix, with SHAKE128 as a random oracle
     (`Export.Stage1.Poseidon2HashChainV1Setup.production_binding_lt_solver`).
     It has no such statement for the kernel vectors of relaxed collisions.
   - `Lifecycle.RandomOracleExtraction.mismatchChance` and
     `Lifecycle.RandomOracleUniqueness.runningChance` count retries and
     reruns that change the running statement. When the claim carries
     `PriorLink` (`Export.Stage1.RandomOracleLink.linkedClaim`), each of them
     is a state-hash collision
     (`Export.Stage1.RandomOracleLink.mismatch_collision`,
     `Export.Stage1.RandomOracleLink.moves_collision`), and
     `Export.Stage1.RandomOracleLink.knowledge_error_le_linked` bounds the two
     terms by the collision events.
   - Lean bounds none of these events numerically.
3. **The history: HyperNova errata Assumption 1.**
   `Export.Stage1.HyperNovaVisitedSecurity.history_probability_bound` takes
   `Export.Stage1.HyperNovaVisitedSecurity.NifsKnowledgeSound`: at each visited
   step, the given source extractor fails after a real acceptance with
   probability at most `error j`. The bound is on the joint event (a
   `goodActive` visit and no checked source witness) under the unconditioned
   visited law. The history bound adds, at each visit `j`, the marked
   state-hash collision mass `h_j` and `error j`. A `goodActive` visit is a real acceptance
   (`Export.Stage1.HyperNovaVisitedAcceptance.realSuccess_iff_goodActive`);
   the visited law stops marking visits after the first failure.
   This is the paper's plain-model premise, not a theorem. The step circuit
   recomputes the challenges of the previous fold with Poseidon2, so a
   recursive argument uses the concrete hash, which no random-oracle model
   covers. `Export.Stage1.HyperNovaFalseAcceptance.probability_bound` is the
   false-acceptance form. The one-fold endpoint motivates the value
   `error j = knowledgeError(Q_j)`, but no Lean statement derives one from the
   other:
   - *Success event.* The history uses `goodActive` under the visited law, a
     plain-model distribution. The ROM event equals
     `NifsRealSuccess.RealSuccess` only at the sponge reads
     (`Export.Stage1.RandomOracleLink.succeeds_iff_realSuccess`).
   - *Relation instance.* The history checks the statement of
     `Export.Stage1.PiCCSStoredWitnessCheck.statement`, which is the key's
     statement (`Export.Stage1.PiCCSStoredWitnessCheck.statement_eq_key`).
     This difference is closed.
   - *Extractor.* `NifsKnowledgeSound` takes one memoryless source kernel of
     the visited statement and payload, the same at every visit. HyperNova
     Definition 7 gives an extractor for each adversary, with the prover's
     state, and that extractor may rewind; the ROM extractor reruns the
     adversary on a changed oracle. No Lean statement shows that Assumption 1
     gives a kernel of the Lean form.
   - *Efficiency.* `NifsKnowledgeSound` has no running-time condition;
     HyperNova Definition 7 asks for an expected polynomial-time extractor.
     Lean bounds the history's extractor calls by the iteration count
     (`Export.Stage1.HyperNovaHistoryWork.source_calls_le_iteration`), and the
     ROM extractor's retries by `17 (Q + 17)`. It counts calls, not machine
     work. The hash term `h_j` comes from the chain that the same
     kernel builds, so an external Poseidon2 collision bound applies to it only
     when that kernel is efficient.
   - *Valid history.* An application witness is a field list of any length
     (`AppWitness`), and the step hash takes any length, but a history that the
     reverse run returns has the circuit's witness length. So
     `FalseAcceptance` does not count an accepted statement that is valid only
     through witnesses of other lengths. Such a statement needs a Poseidon2
     output to agree across input lengths.
4. **Rust and Lean agree on recorded inputs only.** The golden conformance runs
   and the native evidence of the assurance surface cover their recorded
   inputs. No theorem covers arbitrary Rust execution.
5. **Runtime checks.** The verifier's identity and setup checks stay necessary.

## What this page does not say

- Completeness does not follow. A contract bounds the adversary; it is not
  evidence that an honest prover's proof is accepted.
- The error is not one security level. The statistical part costs about
  `2^115.8` work for each unit of success, linear in `Q`. The named events
  depend on the MSIS and Poseidon2 parameters.
