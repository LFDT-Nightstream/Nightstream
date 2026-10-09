# Trust boundary

This page states what a reader must trust to accept the Nightstream F′
security results, and which Lean declaration states each result. Names omit
the common `NightstreamFPrime` namespace. `tests/EndpointCensus.lean` checks
that every name on this page, on [the assurance surface](ASSURANCE_SURFACE.md)
and in [the ROM note](ROM_KNOWLEDGE_SOUNDNESS.md) that starts with `Export.`,
`Layout.`, `Lifecycle.` or `Spec.` exists, and that every cited theorem uses
only `propext`, `Classical.choice` and `Quot.sound`. Shorter names are not
checked.

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

`Export.Stage1.RandomOracleSetup.production_knowledge_error_lt` is knowledge
soundness of one production NIFS fold in the random-oracle model. The
lean-graph target `rom-knowledge-soundness` pins its exact statement
(`scripts/lean_graph/ROM_KNOWLEDGE_TARGET.md`). For every adversary with at
most `Q` Fiat–Shamir oracle queries it states:

```
𝔼 chunks, Pr_H[the verifier accepts the linked claim]
  < 𝔼 chunks, (Pr[extraction returns a source-valid witness] + hashCollisions)
    + 17 (Q + 17) ε_sample + (Q + 74) ε_test + msisAdvantage + 2^-190
```

- *The game.* The Ajtai key is drawn inside the game: its coefficients are
  uniform 256-bit setup chunks reduced modulo `p`
  (`Export.Stage1.RandomOracleSetup.setupKey`; trust item 2). The adversary is
  a function of the chunks, chosen before them, so it may read the key but
  cannot contain a kernel vector of it. The Fiat–Shamir reads are a uniform
  random function `H` (trust item 1). The verifier also checks the prior-state
  link (`Export.Stage1.RandomOracleLink.linkedClaim`).
- *The binding term.* `Export.Stage1.RandomOracleSetup.msisAdvantage` is the
  success of an explicit MSIS solver on a uniform matrix. It draws chunks
  uniformly among the matrix's preimages, runs the adversary for those chunks,
  and returns the output of the binding reduction
  `Lifecycle.RandomOracleBinding.rerunKernel`. That output is a nonzero integer
  kernel vector of the matrix's key with every coordinate below `8TB`,
  computed from the two runs' `Π_RLC` forks
  (`Spec.Folding.PiRLC.PaperForkBinding.collisionAt`, then
  `Spec.Phi81Relation.PiRLCAlgebra.Binding.relaxedBindingCollision_to_shortKernel`).
  For each key, `Lifecycle.RandomOracleBinding.collisionChance_le_kernelChance`
  bounds the binding term of the knowledge error by the reduction's success,
  and `Spec.AjtaiSetupV1.Programming.expect_le_programmed` moves it from
  reduced chunks to a uniform matrix at a cost below `2^-190`.
- *The hash term.* `Export.Stage1.RandomOracleSetup.hashCollisions` is the
  chance that a retry or a rerun gives two different prior preimages of the
  adversary with one state hash
  (`Export.Stage1.RandomOracleLink.mismatch_collision`,
  `Export.Stage1.RandomOracleLink.moves_collision`).
- *The statistical term.* `Lifecycle.RandomOracleKnowledge.statisticalError`.
  With `ε_test = 4589/p² ≈ 2^-115.84`
  (`Lifecycle.Nifs.VerifierErrorBudget.test_error_eq`) and
  `ε_sample ≈ 2^-125.4`, it is about `2^-51.84` at `Q = 2^64`. The `74` is
  the number of challenges of one execution
  (`Lifecycle.RandomOracleUniqueness.challenges_length`).

`Export.Stage1.RandomOracleSetup.contract` states the theorem as a
`Spec.KnowledgeContract`, after Ironwood's record:

| Question | Answer | Lean |
|---|---|---|
| 1. What is a run? | The setup chunks, a uniform random function on bounded call lists, and the extractor's retries. The adversary is a function of the chunks with at most `Q` queries, and selects its statement. | `Export.Stage1.RandomOracleSetup.contract`, `Lifecycle.RandomOracleKnowledge.runWeight` |
| 2. When does the verifier accept? | The production NIFS verifier accepts with every coin read from the oracle, the adversary's witnesses open the 16 returned children, and the prior preimage links the claim. At the sponge reads this is `NifsRealSuccess.RealSuccess`. | `Lifecycle.RandomOracleExtraction.Succeeds`, `Lifecycle.RandomOracleFidelity.accepts_iff_verify`, `Export.Stage1.RandomOracleLink.succeeds_iff_realSuccess` |
| 3. What does extraction return? | The Π_CCS output witness of the complete Π_RLC fork, or nothing. | `Lifecycle.RandomOracleKnowledge.extract` |
| 4. What does a returned witness certify? | `SourceHolds` for the chunks' key and the running and fresh statements that the adversary selected. | `Lifecycle.RandomOracleKnowledge.extract_holds` |
| 5. What is the failure event? | The verifier accepts, and extraction returns nothing. | `Lifecycle.RandomOracleKnowledge.failure_eq` |
| 6. What is the error? | The statistical error, the expected state-hash collision chances, `msisAdvantage`, and the programming error `card(SetupIndex) · 2r/2^256`, which is below `2^-190` at the production size. | `Export.Stage1.RandomOracleSetup.knowledge_error_le_setup`, `Export.Stage1.RandomOracleSetup.production_knowledge_error_lt` |

The record states no running time. The extractor takes `17 (Q + 17)` expected
retries (`Lifecycle.RandomOracleExtraction.expected_retries_le`). The binding
reduction takes `Q + 74` expected reruns until one extracts
(`Lifecycle.RandomOracleUniqueness.expected_reruns_le`); Lean does not bound
the retries of that second extraction. Lean counts runs, not machine work, and
it states no time bound for the MSIS solver. The proof is in
[the ROM note](ROM_KNOWLEDGE_SOUNDNESS.md).

## What you trust

1. **The random-oracle model of the challenge reads.** The deployed verifier
   reads each challenge from the Poseidon2 sponge state after the exact call
   list of that challenge. `Lifecycle.TranscriptCoverage.piCcsProbe_coins`
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
   (their domain tags), but no Lean statement separates them. Nothing else in
   the verifier changes: at an oracle that answers the 74 challenge points of
   one execution as the sponge does, the oracle verifier is
   `PaperNonInteractive.verify` plus the child openings
   (`Lifecycle.RandomOracleFidelity.accepts_iff_verify`), and every execution
   has such an oracle (`Lifecycle.RandomOracleFidelity.deployedOracle_deployed`).
2. **SHAKE128 as a random oracle (premise P1).** Each key coefficient is one
   SHAKE128 output chunk reduced modulo `p`
   (`Export.Stage1.Poseidon2HashChainV1Setup.productionKey_eq_chunks`). The
   game draws the chunks uniformly and independently of the Fiat–Shamir
   oracle. The context digest is computed from the public setup authority
   (setup identifier, dimensions and seed;
   `Spec.AjtaiSetupV1.Setup.authorityWords`), so it does not depend on the
   chunks.
3. **Two hardness terms.**
   - If MSIS is hard for a uniform matrix of `22 × 835936` elements of
     `F_p[X]/Φ₈₁` (degree 54) at norm `8TB`, at the solver's work, then
     `msisAdvantage` is small. The solver runs the adversary an expected
     number of times that grows with `Q` (above); Lean does not bound its
     work, so this step also needs the paper argument that the work is
     polynomial. The 2026-09-08 public-seed MSIS approval covered the
     ChaCha20 matrix only.
   - If Poseidon2 state-hash collisions are hard to find, then
     `hashCollisions` is small. The state hash is fixed and has no key, so an efficient
     algorithm that contains a collision exists. A bound therefore needs the
     human-ignorance reading (Rogaway): it applies to algorithms that a person
     can write without such knowledge.
   - Lean bounds neither term numerically.
4. **The history: HyperNova errata Assumption 1.**
   `Export.Stage1.HyperNovaVisitedSecurity.Assumption1` states Assumption 1 in
   the form of HyperNova Definition 7 knowledge soundness of the Poseidon2
   NIFS: for every
   admitted NIFS adversary (a random tape, and the NIFS input and real output
   it computes) there is an efficient extractor that reads the adversary's
   tape and its own coins, so it may rerun the adversary, and that fails after
   a real success (`NifsRealSuccess.RealSuccess`, with the prior-state link)
   with probability at most `error` of that adversary. Four differences from
   Definition 7:
   - The success event adds the prior-state link (`PriorLink`): the prior
     preimage that the adversary outputs must hash to the digest in the fresh
     public input and be well formed. Well formed includes that the running
     children are the canonical split of the parent that the state hash
     stores (`Lifecycle.ChildrenCanonical`). The bare NIFS verifier does not check this, so without the
     link a prover could choose the running statement after `γ`. The deployed
     system checks it: the terminal verifier recomputes the state hash of the
     running claims, which rejects a split that is not canonical, and
     compares it with the fresh public input
     (`crates/nightstream/src/lifecycle/verify.rs`), and for an inner fold the
     step circuit recomputes it. So Assumption 1 is Definition 7 for the NIFS
     verifier together with this check.
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
   - *Concrete depth.* The sum has one Assumption 1 error for each stage, and
     stage `j + 1` runs the extractor of stage `j`, which may rerun stage `j`.
     So the query count `Q_j` of stage `j` includes every earlier rerun. If
     the plain-model extractor reruns its adversary as the random-oracle
     extractor does (about `17 (Q + 17)` times), then `Q_{j+1} ≈ 17 Q_j²`.
     With the error that the one-fold endpoint motivates, `(Q_j + 74) ε_test`
     is then above 1 at `j = 1` when `Q_0 = 2^64`, and at `j = 2` when
     `Q_0 = 2^30`. So, with these values, the bound is useful only for a depth
     of one or two folds, and it gives no concrete security for long chains.
     This is the constant-depth limit of Lemma 17; the Lean statement is
     correct for every depth.
   - *Valid history.* An application witness is a field list of any length
     (`AppWitness`), and the step hash takes any length, but a history that the
     reverse run returns has the circuit's witness length. So
     `FalseAcceptance` does not count an accepted statement that is valid only
     through witnesses of other lengths. Such a statement needs a Poseidon2
     output to agree across input lengths.
5. **Rust and Lean agree on recorded inputs only.** The golden conformance runs
   and the native evidence of the assurance surface cover their recorded
   inputs. No theorem covers arbitrary Rust execution.
6. **Runtime checks.** The verifier's identity and setup checks stay necessary.

## What this page does not say

- Completeness does not follow. A contract bounds the adversary; it is not
  evidence that an honest prover's proof is accepted.
- The error is not one security level. The statistical part costs about
  `2^115.8` work for each unit of success, linear in `Q`. The other terms
  depend on MSIS at the solver's work and on Poseidon2 collisions.
