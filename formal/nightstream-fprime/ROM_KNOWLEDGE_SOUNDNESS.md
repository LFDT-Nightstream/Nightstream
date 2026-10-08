# Knowledge soundness of the Nightstream NIFS in the random-oracle model

Status: **proved for one fold; history done (2026-10-07).** Owner decisions
of 2026-10-07: a direct random-oracle (ROM) knowledge theorem replaces the
external `FiatShamirModel` assumption; the oracle is a random function of each
challenge's exact absorbed prefix. Lemmas 1–6, the one-fold knowledge
theorem, the binding step of Lemma 6 and the fidelity of the oracle verifier
are proved in Lean (Section 6). `FiatShamirModel` is retired: the history
bound takes HyperNova errata Assumption 1 at each visit (Section 7,
decision 4). No Rust change depends on this note.

Lean names below omit the common `NightstreamFPrime` namespace.
`tests/EndpointCensus.lean` checks that every full name in this note exists.

Neither paper supplies this proof. SuperNeo is stated for the interactive
protocol only. The published HyperNova asserts a Fiat–Shamir lemma for
multi-folding without a multi-round knowledge argument; our corrected copy
(errata v2) states it as Assumption 1. The proof below therefore follows the
interactive SuperNeo extraction (Appendix B.1–B.3) and replaces each use of
fresh verifier coins by an oracle step.

## 1. Why `FiatShamirModel` cannot be discharged as stated

`FiatShamirModel` asked for `g Q p_real − δ Q ≤ p_interactive`, where
`p_interactive` is the success of the *causal* interactive game: the prover
sends each SumCheck message before it receives that round's coin. A
black-box translation from an oracle adversary must decide, before each coin,
which of the adversary's oracle queries carries the final round message. With
28 SumCheck rounds and the α/γ step, the guessing loss is about `(Q+1)^29`.
No useful `g` exists. The proof below never builds an interactive prover. It
analyzes the oracle experiment directly, as Ironwood does for Halo 2.

## 2. Model

**Prefixes.** `Lifecycle.TranscriptCoverage.challenge_seal` proves that each
production challenge `c` (28 α coordinates, γ, 28 SumCheck points, 17 Π_RLC
scalars) is the read of the Poseidon2 state after the call list
`prefix(c) = proverCalls c ++ fixedCalls c`.
`Lifecycle.TranscriptCoverage.proverCalls_identify` proves that equal
prover-dependent calls identify the fresh statement and every earlier prover
message. With `PriorLink`, they also identify the prior preimage, the
verifier context and the running statement, or exhibit a state-hash collision
(`Layout.Stage1.PiCCSSecurity.calls_identify_view_or_collision`).

**Oracle.** Let `T` be the call lists of bounded length (every production
prefix is shorter than a fixed bound) and `B = F^4` the four-word block. The
model replaces `prefix ↦ read(run(initialState, prefix))` by a uniformly
random function `H : T → B`:

- an α, γ or SumCheck challenge is the extension element formed by the first
  two words of `H(prefix(c))`;
- a Π_RLC scalar is the existing sampler applied to `H(prefix(c))`; its law
  is within `distance < 2^-132` of uniform on the strong set
  (`Spec.Folding.Nifs.NonInteractive.PiRlcSampler.distance_lt`).

This is the same idealization as Ironwood's abstract `squeeze`. It is a
modelling assumption, not a theorem about Poseidon2. Unlike Ironwood, the
argument of `H` is the proved concrete call list, not a typed abstraction.

**Distinct prefixes.** For one execution, distinct challenges have distinct
prefixes (`Lifecycle.TranscriptCoverage.challengeCalls_injective`), so their
oracle answers are independent and uniform. The proof uses the prefix
lengths.

**Adversary.** `A^H` is a classical oracle algorithm with at most `Q` oracle
queries (Ironwood's `OracleComp` with `QueryBound Q`). It outputs a claim
(running and fresh statement, a NIFS proof and 16 child witnesses) and a
prior preimage. The statement is adaptive: `A` may choose it after any
queries.

**Success** is `Lifecycle.RandomOracleExtraction.Succeeds`: the oracle
verifier accepts, the 16 child witnesses open the verifier-computed children,
and the claim passes its `linked` check. In production the check is
`PriorLink` (`Export.Stage1.RandomOracleLink.linkedClaim`): the output prior
preimage links the running statement and the verifier context digest to the
absorbed digest. At the sponge reads, this event is
`NifsRealSuccess.RealSuccess`
(`Export.Stage1.RandomOracleLink.succeeds_iff_realSuccess`).

**Extraction goal.** An expected-time extractor `E`, with oracle access to
`A` and the ability to rerun `A` on a modified table, returns a valid source
witness (the paper's `SourceHolds` for the fresh and running claims).

## 3. Main theorem (proved for one fold)

`Lifecycle.RandomOracleKnowledge.knowledge_error_le`: for every adversary `A`
with at most `Q` queries,

```
Pr[the claim of A succeeds]
  ≤ Pr[E extracts a source-valid witness]
  + 17 · (Q + 17) · ε_sample   -- Π_RLC coordinate retry, Lemma 5
  + (Q + 74) · ε_test          -- PiCCS challenges, Lemmas 4 and 6
  + Pr[mismatch]               -- a retry changes the running statement
  + Pr[collision]              -- binding reduction: two witnesses, Lemma 6
  + Pr[running]                -- binding reduction: other running statement
```

`E` is the Π_RLC fork extractor of Lemma 5; it takes `17 · (Q + 17)`
expected retries (`Lifecycle.RandomOracleExtraction.expected_retries_le`). Its
retry law has total mass one after each acceptance, so a missing retry counts
as a failure. `ε_sample = 1/5^54 + distance` is the mass of one sampler fiber,
and `ε_test = IndependentExecution.testError productionShape 8`. `Q + 17`
counts the adversary's queries and the 17 Π_RLC queries; `Q + 74` also
counts the 28 α, γ and 28 round queries
(`Lifecycle.RandomOracleUniqueness.challenges_length`).

The three named events:

- *Mismatch* and *running.* With `PriorLink`, every counted retry or rerun
  that changes the running statement is a state-hash collision
  (`Export.Stage1.RandomOracleLink.mismatch_collision`,
  `Export.Stage1.RandomOracleLink.moves_collision`).
  `Export.Stage1.RandomOracleLink.knowledge_error_le_linked` is the theorem
  with these two terms replaced by the collision events.
- *Collision.* Every counted rerun gives a `(2B, C)`-relaxed binding
  collision on one input commitment, and then a nonzero kernel vector of the
  Ajtai key with every coordinate below `8TB`
  (`Lifecycle.RandomOracleBinding.rerun_shortKernel`). The binding reduction
  takes `Q + 74` expected reruns
  (`Lifecycle.RandomOracleUniqueness.expected_reruns_le`).

Numerically (4 CCS matrices, `J = 4338`), `ε_test = 4589/p² ≈ 2^-115.84`
(`Lifecycle.Nifs.VerifierErrorBudget.test_error_eq`), and its γ term
`(J − 1)/p² ≈ 2^-115.92` dominates; `ε_sample ≈ 2^-125.4`. At `Q = 2^64`
oracle queries, `(Q + 74)·ε_test ≈ 2^-51.84` and
`17·(Q + 17)·ε_sample ≈ 2^-57.3`. This is the honest Fiat–Shamir cost; the
interactive analysis does not show it.

The history theorem cannot be a pure ROM theorem: the step circuit
recomputes the previous fold's challenges with Poseidon2, so a recursive
argument uses the concrete hash. It takes HyperNova errata Assumption 1,
plain-model part, as Definition 7 knowledge soundness of the NIFS
(`Export.Stage1.HyperNovaVisitedSecurity.Assumption1`), and Lean proves the
paper's reverse-extractor composition (Lemma 17,
`Export.Stage1.HyperNovaVisitedSecurity.history_probability_bound`). This
theorem motivates the error of Assumption 1, but no Lean statement derives one
from the other; `formal/nightstream-fprime/TRUST_BOUNDARY.md` lists the
differences.

## 4. Proof outline

The proof has six lemmas. Lemmas 1–3 are standard or already proved. Lemmas
4–6 carry the new argument and need the closest review.

**Lemma 1 (prefix oracle).** Each verifier challenge is `H` at a prefix that
identifies all earlier prover data. *Status:* proved. Coverage:
`Lifecycle.TranscriptCoverage.challenge_seal` and
`Lifecycle.TranscriptCoverage.proverCalls_identify`; the key's coins are one
read of each prefix (`Lifecycle.TranscriptCoverage.piCcsProbe_coins`,
`Lifecycle.TranscriptCoverage.rho_seal`); distinct challenges have distinct
prefixes (`Lifecycle.TranscriptCoverage.challengeCalls_injective`).

**Lemma 2 (pinned squeeze; after Ironwood `xEscAtPoint_measure_le`).** Let
`bad t H ⊆ B` read `H` only away from `t`, with `μ(bad t H) ≤ ε` for every
`t` and `H`. For any `Q`-query `A` with an output point `xpt`,
`Pr_H[H(xpt(A^H)) ∈ bad(xpt(A^H), H)] ≤ (Q+1)·ε`. *Proof:* at the first
query of a point, its answer is independent of the adversary's view and of
`H` elsewhere. *Status:* proved as `Spec.RandomOracle.pinned_le`
(`Spec.RandomOracle.escape_le` for every queried point). Ironwood proves the
case where `bad` does not read `H`; the local form is needed because a
round's bad set reads the earlier challenges.

**Lemma 3 (per-challenge test bound).** `ε_test` splits into point-indexed
round-by-round bounds. Fix a statement and a witness `w`. For each challenge
`c`, define `bad_w(prefix(c))` as the coins that turn a failing claim into a
passing one:

- α coordinate `j`: the multilinear `eq(α, ·)` batch is degree 1 in `α_j`,
  so the bad set has at most one value: `1/p²` per coordinate, `28/p²` total;
- γ: the joint residual is a polynomial in γ of degree `J − 1`, where
  `J = jointCoefficientCount = 16·54·5 + 1 + 17 = 4338`: `(J − 1)/p²`;
- SumCheck round `r`: a wrong degree-8 round polynomial agrees with the true
  one on at most 8 points: `8/p²` per round, `224/p²` total.

The sum is exactly `testError`. Each set depends only on `w`, the statement
and the prover data before `c`, which the prefix identifies (Lemma 1).
*Status:* proved as
`Spec.Folding.PiCCS.PaperJoint.RoundByRound.falseAcceptance_splits` with the
counts `Spec.Folding.PiCCS.PaperJoint.RoundByRound.alphaBad_probability_le`,
`Spec.Folding.PiCCS.PaperJoint.RoundByRound.gammaBad_probability_le` and
`Spec.Folding.PiCCS.PaperJoint.RoundByRound.roundBad_probability_le`. The `α`
split follows one fixed path of nonzero sub-tables, one coordinate per step.

**Lemma 4 (test error in the ROM).** For a fixed fork state `x` (below) and a
fixed witness `w` that is not source-valid, the probability that the
continuation of `A` from `x` outputs an accepted proof whose extracted
witness is `w` is at most `(Q+1)·ε_test`. *Proof:* if the output is accepted
with witness `w`, some challenge of the final transcript lies in its
`bad_w` set (the round-by-round argument of the interactive proof, applied to
the final transcript). Apply Lemma 2 to each of the 57 extension-field
challenges and sum. *Status:* proved as
`Lifecycle.RandomOracleTest.test_error_le` for an adaptive fresh statement,
with the running statement and `w` fixed. Lemma 6 removes the fixed `w`.

**Lemma 5 (Π_RLC coordinate retry in the ROM).** After an acceptance, the
extractor resamples the answer at one Π_RLC point and reruns `A` until that
point succeeds again, once for each of the 17 coordinates. The other
coordinates keep their answers, as the interactive retry (SuperNeo B.3;
Fenzi–Moghaddas–Nguyen Lemma 7.1) requires, because each Π_RLC scalar is a
separate oracle point. A retry is valid when it keeps the running statement
and draws a different scalar. The base and 17 valid retries form the paper's
complete coordinate fork, and the existing Appendix D.5 algebra
(`Spec.Folding.PiRLC.PaperForkExtraction.completeFork_implies_correctedAmbientHolds`)
extracts a witness that opens every Π_CCS output of the base probe. A retry
draws the base scalar with probability at most `ε_sample`; each queried point
is charged once (`Spec.RandomOracle.repeat_le`), so the loss is
`17 · (Q + 17) · ε_sample` plus the chance that a retry changes the running
statement. The loss is linear in `Q`, as in Attema–Fehr–Klooß–Resch.
*Status:* proved as `Lifecycle.RandomOracleExtraction.fork_failure_le`,
`Lifecycle.RandomOracleExtraction.expected_retries_le` and
`Lifecycle.RandomOracleExtraction.extracted_ambient`.

**Lemma 6 (forked uniqueness; replaces SuperNeo v1.2 B.2 retry).** The
interactive argument uses two independent executions at one fixed context.
In the ROM the statement is adaptive, so the second execution is a *fork*.
Run `A` followed by the verifier's 74 challenge queries. The fork index `J`
is the first query whose call list extends the output statement's calls; the
fork context is the set of points queried before `J`. The binding reduction
runs the extractor, then reruns `A` with the context answers kept and fresh
answers elsewhere until a rerun succeeds at index `J` again. Both runs output
the same fresh statement, because the query at `J` fixes it. The rerun then
changes the running statement (a state-hash collision under `PriorLink`),
extracts another witness for the same statement, or extracts the same
witness. In the second case the two complete Π_RLC forks give a relaxed
binding collision (SuperNeo v1.2 Appendix B,
`Spec.Folding.PiRLC.PaperForkBinding.two_forks_unique_or_collision`), and so
a short kernel vector of the key
(`Lifecycle.RandomOracleBinding.collides_relaxedBindingCollision`). The
extracted witnesses satisfy only the corrected ambient relation, so the step
needs relaxed binding, not ordinary binding. In the last case the rerun is a
false acceptance of the first run's witness. That witness depends on answers
after the context, so a bad set built from it would not be local. The proof
replaces it by the *worst witness* of the context: the extraction, over all
valid runs, that the reruns from this context falsely accept most often. The
context alone determines it. The bad set at a challenge point uses the worst
witness of the context where that point's statement first appears, so it is
local, and Lemma 2 charges each of the `Q + 74` checked queries once. The
division by the rerun success cancels on average over the base run
(`Spec.RandomOracle.expect_div_resampled`). Result:
`Pr[E's witness fails SourceHolds] ≤ (Q + 74)·ε_test + Pr[collision] +
Pr[running]`. *Status:* proved as
`Lifecycle.RandomOracleUniqueness.source_error_le` and
`Lifecycle.RandomOracleUniqueness.expected_reruns_le`.

## 5. What this changes and what it does not

- `FiatShamirModel` and its parameters `g`, `deltaFS` are retired. The
  history bound takes Assumption 1 with a per-visit error instead.
- The PriorLink, coverage and identify results of this PR are inputs
  (Lemma 1), not replaced.
- The MSIS assumption, the state-hash collision event and the
  low-norm-invertibility facts are unchanged.
- The model remains an idealization: a concrete attack on Poseidon2 as a
  random oracle, or on the additive duplex, is outside it, as BLAKE2b is
  outside Ironwood's theorem.
- The in-circuit hash attack class (KRS) is outside every random-oracle
  model; the relation is still fixed by the verifier's package key.

## 6. Formalization design

**Oracle model.** `H` is a uniformly random function from bounded call lists
to four-word blocks, averaged over all such functions (Ironwood's full-table
model). The lazily sampled `PiRlcSampler.OracleModel` is not reused: the bad
set of a challenge reads other challenges (later α coordinates, earlier
rounds), which a lazy tape may not have sampled yet. With a full table, the
only condition is *locality*: the bad set at a point may read `H` anywhere
except at that point. The domain bound covers every challenge prefix; no
result depends on its value.

**Verifier fidelity.** The oracle verifier
`Lifecycle.RandomOracleExtraction.Accepts` uses the key's own definitions and
reads only the coins and the Π_RLC challenges from the oracle. At an oracle
that answers the 74 challenge points of one execution as the sponge does
(`Lifecycle.RandomOracleFidelity.Deployed`), it is the deployed verifier
`PaperNonInteractive.verify` plus the child openings
(`Lifecycle.RandomOracleFidelity.accepts_iff_verify`), and every execution
has such an oracle
(`Lifecycle.RandomOracleFidelity.deployedOracle_deployed`). So the theorem is
about the deployed verifier with only the reads idealized. A key with a
call-list state was rejected: `readK` reads lane 0 of two consecutive states,
and the key's construction and coverage proofs would be duplicated.

**Modules.**

| Module | Owns |
|---|---|
| `Spec/RandomOracle.lean` | `OracleComp`, `run`, `queries`, `QueryBound`, locality, `escape_le` (`Q·ε`), `pinned_le` (`(Q+1)·ε`); credit Ironwood. Line retries (`repeat_le`, `retries_le`), the retry law (`retryWeight`, total mass one), stopping sets and resampling (`expect_resample`, `expect_div_resampled`) |
| `Spec/.../PaperJoint/RoundByRound.lean` | deterministic split of a false acceptance into one bad coin per α coordinate, γ or round; bad-set counts `1`, `J−1`, `8` |
| `Lifecycle/TranscriptCoverage.lean` | `point`, `coinsFrom`, distinct point lengths, the seals of the key's coins and Π_RLC challenges |
| `Lifecycle/RandomOracleTest.lean` | bounded domain, `decodeK`, oracle bad sets, `test_error_le`: `(Q+1)·testError`; `hits_error_le`, the per-coin sum |
| `Lifecycle/RandomOracleExtraction.lean` | Lemma 5: oracle verifier `Accepts`, the `linked` check, coordinate retries, `completeFork`, `extractedWitness`, `fork_failure_le`, `expected_retries_le` |
| `Lifecycle/RandomOracleFidelity.lean` | `Deployed`, `accepts_iff_verify`: the oracle verifier at the sponge reads is `PaperNonInteractive.verify` |
| `Lifecycle/RandomOracleUniqueness.lean` | Lemma 6: fork index and context, worst witness, local bad sets, `source_error_le`, `expected_reruns_le` |
| `Lifecycle/RandomOracleBinding.lean` | the binding step of Lemma 6: a counted rerun gives a relaxed binding collision and a short kernel vector |
| `Lifecycle/RandomOracleKnowledge.lean` | `knowledge_error_le`: Lemmas 5 and 6 together; `contract`, the `Spec.KnowledgeContract` instance (Ironwood's six questions) |
| `Export/Stage1/RandomOracleLink.lean` | `PriorLink` as the `linked` check; the two running events as state-hash collisions; `succeeds_iff_realSuccess`; `knowledge_error_le_linked` |

`tests/AxiomsStage1Security.lean` audits every theorem above for axioms.

## 7. Review decisions (owner, 2026-10-07)

1. **Lemma 5 citation.** Use Attema, Fehr, Klooß, Resch, *The Fiat–Shamir
   Transformation of (Γ₁,…,Γμ)-Special-Sound Interactive Proofs* (ePrint
   2023/1945): its loss is linear in the oracle queries and independent of
   the round count. Coordinate-wise special soundness is an instance of
   Γ-special soundness (Attema, Fehr, Resch, ePrint 2023/818).
   Fenzi–Moghaddas–Nguyen Lemma 7.1 stays the source of the interactive
   extractor.
2. **Lemma 6 fork point.** Keep "the first query that extends the output
   statement's calls". The output defines it, every challenge of that
   statement comes at or after it, and its call list fixes the fresh
   statement. The knowledge extractor does not fork; only the binding
   reduction forks. No re-fork bound is needed. *Correction from the Lean
   proof:* the reduction reruns from the fork context until a rerun succeeds
   at the fork index, not exactly twice; it takes at most `Q + 74` expected
   reruns (`Lifecycle.RandomOracleUniqueness.expected_reruns_le`).
3. **Deployment margin.** `(Q+1)·ε_test` is linear in `Q`, so the cost per
   unit of success stays `2^115.8`: about 115-bit security, equal to the
   interactive bound, and tight against a grinding attack. Accept it for this
   work. Reaching 128 bits is a separate parameter decision (a larger field
   for `γ`).

4. **History step (2026-10-07).** Replace `FiatShamirModel` in the history
   bound by HyperNova errata Assumption 1, plain-model part, at each visited
   step: the Poseidon2 NIFS is knowledge sound at error `knowledgeError(Q)`,
   which the ROM theorem motivates but does not prove. Delete the transfer
   code that only `FiatShamirModel` uses. Done in
   `Export.Stage1.HyperNovaVisitedSecurity.history_probability_bound` and
   `Export.Stage1.HyperNovaFalseAcceptance.probability_bound`.

5. **Assumption 1 as in the paper (2026-10-08).** State Assumption 1 as
   Definition 7: an extractor for each admitted adversary, which reads that
   adversary's tape. Prove Lemma 17's composition: the reverse extractor
   applies Assumption 1 to its own stages. Admission applies to whole stages
   (`StageAdmitted`), and an admitted stage gives an admitted NIFS adversary,
   because a stage's NIFS projection can hide data that the next stage reads.
   Done in
   `Export.Stage1.HyperNovaVisitedSecurity.Assumption1` and
   `Export.Stage1.HyperNovaVisitedSecurity.reverseStages`.

References: Attema, Fehr, Klooß, Resch, ePrint 2023/1945; Attema, Fehr,
Resch, ePrint 2023/818; Attema, Fehr, Klooß, *Fiat–Shamir Transformation of
Multi-Round Interactive Proofs*, J. Cryptology 2023; Fenzi, Moghaddas, Nguyen,
*Lattice-Based Polynomial Commitments*, J. Cryptology 2024, Lemma 7.1;
Bellare, Neven, *Multi-Signatures in the Plain Public-Key Model and a General
Forking Lemma*, CCS 2006; zcash/ironwood `86e3c7026db8`,
`Zcash/Snark/Soundness/FiatShamir/PinnedSqueeze.lean`.
