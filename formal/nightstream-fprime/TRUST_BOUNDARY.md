# Trust boundary

This page states what a reader must trust to accept the Nightstream F′
security results, and which Lean declaration states each result. Names omit
the common `NightstreamFPrime` namespace. `tests/EndpointCensus.lean` checks
that every full name on this page and on [the assurance surface](ASSURANCE_SURFACE.md)
exists, and that every cited theorem uses only `propext`, `Classical.choice`
and `Quot.sound`.

## What Lean checks

- The kernel of the official Lean v4.32.2 toolchain, with Mathlib v4.32.2 and
  the CompPoly revision that `lakefile.toml` pins. Final validation uses the
  official toolchain. The lean4-optimized fork is used only to iterate.
- `scripts/validate.sh static`, `build` and `axioms` run on the developer Mac.
  CI runs no Lean (owner decision of 2026-09-26).
- Every audited theorem uses the three axioms above. The audit fails closed:
  `sorryAx` or a new axiom stops the build.

## The one-fold knowledge endpoint

`Lifecycle.RandomOracleKnowledge.knowledge_error_le` is knowledge soundness of
one production NIFS fold in the random-oracle model. The extractor succeeds
with probability at least the acceptance probability minus
`Lifecycle.RandomOracleKnowledge.knowledgeError`.
`Lifecycle.RandomOracleKnowledge.contract` states the same result as a
`Spec.KnowledgeContract`, after Ironwood's record:

| Question | Answer | Lean |
|---|---|---|
| 1. What is a run? | A uniform random function on bounded call lists, then the extractor's retries. The adversary has at most `Q` queries and selects its statement. | `Lifecycle.RandomOracleKnowledge.runWeight` |
| 2. When does the verifier accept? | The production NIFS verifier accepts with every coin read from the oracle, and the adversary's witnesses open the 16 returned children. | `Lifecycle.RandomOracleExtraction.Succeeds` |
| 3. What does extraction return? | The Π_CCS output witness of the complete Π_RLC fork, or nothing. | `Lifecycle.RandomOracleKnowledge.extract` |
| 4. What does a returned witness certify? | `SourceHolds` for the running and fresh statements that the adversary selected. | `Spec.KnowledgeContract` |
| 5. What is the failure event? | The verifier accepts, and extraction returns nothing. | `Spec.KnowledgeContract` |
| 6. What is the error? | `17 (Q + 17) ε_sample + (Q + 74) ε_test`, plus three named events (below). | `Lifecycle.RandomOracleKnowledge.knowledgeError` |

The extractor takes `17 (Q + 17)` expected retries
(`Lifecycle.RandomOracleExtraction.expected_retries_le`). The binding
reduction takes `Q + 74` expected reruns
(`Lifecycle.RandomOracleUniqueness.expected_reruns_le`). With
`ε_test = 7209/p² ≈ 2^-115.2` and `ε_sample ≈ 2^-125.4`, the statistical part
at `Q = 2^64` is about `2^-51.2`. The proof is in
[the ROM note](../../docs/reviews/nightstream-fprime-requirements/ROM_KNOWLEDGE_SOUNDNESS.md).

## What you trust

1. **The random-oracle model of the challenge reads.** The deployed verifier
   reads each challenge from the Poseidon2 sponge state after the exact call
   list of that challenge. `Lifecycle.TranscriptCoverage.piCcsProbe_coins`
   proves that the coins are one read of each call list
   (`Lifecycle.TranscriptCoverage.coinsFrom`). The model replaces that read by
   a uniform random function of the call list
   (`Lifecycle.RandomOracleTest.coins`). This is a modelling assumption, as
   Ironwood's abstract squeeze is. It is not a theorem about Poseidon2 or the
   additive duplex.
2. **Three named events.** `Lifecycle.RandomOracleUniqueness.collisionChance`
   is the success of the binding reduction: two witnesses for one running
   statement, an MSIS break for the SHAKE128-expanded Ajtai matrix
   (`Export.Stage1.NifsBinding.bindingEvent_to_shortKernel`,
   `Export.Stage1.Poseidon2HashChainV1Setup.production_binding_lt_solver`).
   `Lifecycle.RandomOracleExtraction.mismatchChance` and
   `Lifecycle.RandomOracleUniqueness.runningChance` are retries that change
   the running statement. With `Layout.Stage1.PiCCSSecurity.PriorLink` they
   are state-hash collisions. Lean bounds none of these events numerically.
3. **The history: HyperNova errata Assumption 1.**
   `Export.Stage1.HyperNovaVisitedSecurity.history_probability_bound` takes
   `Export.Stage1.HyperNovaVisitedSecurity.NifsKnowledgeSound`: at each visited
   step, the source extractor of the Poseidon2 NIFS fails after a real
   acceptance with probability at most `error j`. This is the paper's
   plain-model premise, not a theorem. The step circuit recomputes the
   challenges of the previous fold with Poseidon2, so a recursive argument
   uses the concrete hash, which no random-oracle model covers. The one-fold
   endpoint above justifies the value `knowledgeError(Q_j)`.
   `Export.Stage1.HyperNovaFalseAcceptance.probability_bound` is the
   false-acceptance form.
4. **Rust and Lean agree on recorded inputs only.** The golden conformance runs
   and the native evidence of the assurance surface cover their recorded
   inputs. No theorem covers arbitrary Rust execution.
5. **Runtime checks.** The verifier's identity and setup checks stay necessary.

## What this page does not say

- Completeness does not follow. A contract bounds the adversary; it is not
  evidence that an honest prover's proof is accepted.
- The error is not one security level. The statistical part costs about
  `2^115` work for each unit of success, linear in `Q`. The named events
  depend on the MSIS and Poseidon2 parameters.
