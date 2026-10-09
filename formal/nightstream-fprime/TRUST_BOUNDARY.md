# Trust boundary

This page states what a reader must trust to accept the Nightstream F′
security results, and which Lean declaration states each result. Names omit
the common `NightstreamFPrime` namespace. `tests/EndpointCensus.lean` checks
that every full name on this page, on [the assurance surface](ASSURANCE_SURFACE.md)
and in [the ROM note](ROM_KNOWLEDGE_SOUNDNESS.md)
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
[the ROM note](ROM_KNOWLEDGE_SOUNDNESS.md).

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
   `Export.Stage1.HyperNovaVisitedSecurity.Assumption1` states Assumption 1 in
   the form of HyperNova Definition 7 knowledge soundness of the Poseidon2
   NIFS: for every
   admitted NIFS adversary (a random tape, and the NIFS input and real output
   it computes) there is an efficient extractor that reads the adversary's
   tape and its own coins, so it may rerun the adversary, and that fails after
   a real success (`NifsRealSuccess.RealSuccess`, with the prior-state link)
   with probability at most `error` of that adversary. Three differences from
   Definition 7:
   - This joint form implies Definition 7's difference form
     `Pr[success] − Pr[extraction] ≤ error`. The converse needs an adversary
     that stops when its own success check fails.
   - The public parameters are the fixed production key and setup, not
     sampled by the generator. With a fixed hash and a fixed key, an efficient
     algorithm that contains a collision exists. So the premise can hold only
     for algorithms that a person can write without such knowledge
     (Rogaway's human-ignorance reading).
   - `error` replaces `negl(λ)`.

   `Export.Stage1.HyperNovaVisitedSecurity.Closed` is the composition premise.
   A second class, `StageAdmitted`, holds whole stages: the reverse extractor
   so far, as one algorithm. An admitted stage gives an admitted NIFS
   adversary, and one reverse step of an admitted stage with an efficient
   extractor is again an admitted stage. The IVC adversary is the start stage,
   which must be admitted. `Admitted`, `StageAdmitted` and `Efficient` are
   abstract; their intended meaning is expected polynomial time. Lean has no
   running-time model, so it states no false claim about running time, and
   the closure property is a premise.
   `Export.Stage1.HyperNovaVisitedSecurity.reverseStages` is the reverse
   extractor of HyperNova Lemma 17 (Appendix H.3): stage `j + 1` runs stage
   `j` and then the extractor that Assumption 1 gives for stage `j`'s NIFS
   adversary. Every stage is admitted.
   `Export.Stage1.HyperNovaVisitedSecurity.history_probability_bound` bounds
   the accepted terminal mass by the reverse extractor's returned-history mass
   plus, at each stage `j`, its marked state-hash collision mass `h_j` and the
   Assumption 1 error of stage `j`;
   `Export.Stage1.HyperNovaFalseAcceptance.probability_bound` is the
   false-acceptance form. A `goodActive` visit is a real acceptance
   (`Export.Stage1.HyperNovaVisitedSecurity.realSuccess_of_goodActive`).
   Assumption 1 is the paper's plain-model premise, not a theorem. The step
   circuit recomputes the challenges of the previous fold with Poseidon2, so a
   recursive argument uses the concrete hash, which no random-oracle model
   covers. The one-fold endpoint motivates the value of `error`, but no Lean
   statement derives one from the other:
   - *Success event.* Assumption 1 uses `NifsRealSuccess.RealSuccess` under
     the stage's own law, a plain-model distribution. The ROM event equals it
     only at the sponge reads
     (`Export.Stage1.RandomOracleLink.succeeds_iff_realSuccess`).
   - *Relation instance.* The history checks the statement of
     `Export.Stage1.PiCCSStoredWitnessCheck.statement`, which is the key's
     statement (`Export.Stage1.PiCCSStoredWitnessCheck.statement_eq_key`).
     This difference is closed.
   - *Extractor.* The ROM extractor reruns the adversary on a changed oracle;
     the plain-model extractor of Assumption 1 is any efficient algorithm that
     reads the adversary's tape. Neither is derived from the other.
   - *Efficiency.* Lean proves that every stage stays in `StageAdmitted`
     under `Closed`. The meaning of that class (expected polynomial time for a
     constant depth, as in Lemma 17) is a premise outside Lean. The hash term
     `h_j` is the probability that stage `j`, an admitted stage, followed by
     the computation of its current visit, outputs a state-hash collision. So
     an external Poseidon2 collision bound applies to it, under the same
     human-ignorance reading, because the state hash is fixed and has no key.
     The paper's truncation argument for expected-time stages is outside
     Lean. Lean
     chooses the stages with `Classical.choose`, so a numerical bound on `h_j`
     and on `error` must hold for the whole class.
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
