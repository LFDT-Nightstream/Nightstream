# Knowledge soundness of the Nightstream NIFS in the random-oracle model

Status: **DRAFT for review (2026-10-07).** Owner decisions of 2026-10-07: a
direct random-oracle (ROM) knowledge theorem replaces the external
`FiatShamirModel` assumption; the oracle is a random function of each
challenge's exact absorbed prefix; this written proof is reviewed before the
forking work. Lemmas 1–4 are proved in Lean (Section 6). Lemmas 5–6 and the
history capstone wait for the review in Section 7. No Rust change depends on
this note.

Neither paper supplies this proof. SuperNeo is stated for the interactive
protocol only. The published HyperNova asserts a Fiat–Shamir lemma for
multi-folding without a multi-round knowledge argument; our corrected copy
(errata v2) states it as Assumption 1. The proof below therefore follows the
interactive SuperNeo extraction that Lean already proves (Appendix B.1–B.3)
and replaces each use of fresh verifier coins by an oracle step.

## 1. Why `FiatShamirModel` cannot be discharged as stated

`FiatShamirModel` asks for `g Q p_real − δ Q ≤ p_interactive`, where
`p_interactive` is the success of the *causal* interactive game: the prover
sends each SumCheck message before it receives that round's coin. A
black-box translation from an oracle adversary must decide, before each coin,
which of the adversary's oracle queries carries the final round message. With
28 SumCheck rounds and the α/γ step, the guessing loss is about `(Q+1)^29`.
No useful `g` exists. The proof below never builds an interactive prover. It
analyzes the oracle experiment directly, as Ironwood does for Halo 2.

## 2. Model

**Prefixes.** `Lifecycle/TranscriptCoverage.lean` proves that each production
challenge `c` (28 α coordinates, γ, 28 SumCheck points, 17 Π_RLC scalars) is
the read of the Poseidon2 state after the call list
`prefix(c) = proverCalls c ++ fixedCalls c`, and that equal prover-dependent
calls identify the fresh statement and every earlier prover message
(`proverCalls_identify`). With `PriorLink`, they also identify the prior
preimage, the verifier context and the running statement, or exhibit a
state-hash collision (`calls_identify_view_or_collision`).

**Oracle.** Let `T` be the call lists of bounded length (every production
prefix is shorter than a fixed bound) and `B = F^4` the four-word block. The
model replaces `prefix ↦ read(run(initialState, prefix))` by a uniformly
random function `H : T → B`:

- an α, γ or SumCheck challenge is the extension element formed by the first
  two words of `H(prefix(c))`;
- a Π_RLC scalar is the existing sampler applied to `H(prefix(c))`; its law
  is within `distance < 2^-132` of uniform on the strong set
  (`PiRlcSampler.distance_lt`).

This is the same idealization as Ironwood's abstract `squeeze`. It is a
modelling assumption, not a theorem about Poseidon2. Unlike Ironwood, the
argument of `H` is the proved concrete call list, not a typed abstraction.

**Required lemma (new, phase 1): distinct prefixes.** For one execution,
distinct challenges have distinct prefixes. Then their oracle answers are
independent and uniform. The proof uses the label words (`[1,c]`, `[2]`,
`[3,r]`, `[4,i]`) and the prefix lengths.

**Adversary.** `A^H` is a classical oracle algorithm with at most `Q` oracle
queries (Ironwood's `OracleComp` with `QueryBound Q`). It outputs a context
(prior preimage, running and fresh statement), a NIFS proof and 16 child
witnesses. The statement is adaptive: `A` may choose it after any queries.

**Real success** is the approved event `FiatShamirTransfer.RealSuccess`: the
output prior preimage satisfies `PriorLink` for the running statement and the
verifier context digest, the verifier accepts, and the 16 child witnesses
open the verifier-computed children.

**Extraction goal.** An expected-time extractor `E`, with oracle access to
`A` and the ability to rerun `A` on a modified table, returns a valid source
witness (the paper's `SourceHolds` for the fresh and running claims).

## 3. Main theorem (target statement)

For every `A` with at most `Q` queries:

```
Pr[RealSuccess ∧ E fails]
  ≤ (Q + 1) · ε_test          -- PiCCS challenges, Lemma 4
  + (Q + 1) · ε_weak          -- Π_RLC coordinate retry, Lemma 5
  + Q · ε_sampler             -- strong-set sampler bias, per Π_RLC query
  + ε_uniq                    -- forked disagreement, Lemma 6
  + Pr[state-hash collision]  -- named event, unchanged
```

with `ε_test = IndependentExecution.testError productionShape 9`,
`ε_weak = 17 / |C|` (`|C| = 5^54`), and `ε_sampler = distance`. `ε_uniq` is
the forked disagreement probability that the existing adaptive binding
reduction turns into a fixed-seed MSIS solution (`AdaptiveBindingProbability`
term, multiplied by 17 as today).

Numerically (7 CCS matrices, `J = 6930`), `ε_test = 7209/p² ≈ 2^-115.2`,
and its γ term `(J − 1)/p² ≈ 2^-115.24` dominates; `ε_weak ≈ 2^-121.3`. At
`Q = 2^64` oracle queries, `(Q+1)·ε_test ≈ 2^-51.2`. This is the honest
Fiat–Shamir cost; the interactive analysis does not show it.

The history theorem then applies this bound at each visited step in place of
`FiatShamirModel`, with the visited-law composition already proved.

## 4. Proof outline

The proof has six lemmas. Lemmas 1–3 are standard or already proved. Lemmas
4–6 carry the new argument and need the closest review.

**Lemma 1 (prefix oracle).** Each verifier challenge is `H` at a prefix that
identifies all earlier prover data. *Status:* proved. Coverage:
`challenge_seal`, `proverCalls_identify`; the key's coins are one read of each
prefix (`piCcsProbe_coins`); distinct challenges have distinct prefixes
(`challengeCalls_injective`).

**Lemma 2 (pinned squeeze; after Ironwood `xEscAtPoint_measure_le`).** Let
`bad t H ⊆ B` read `H` only away from `t`, with `μ(bad t H) ≤ ε` for every
`t` and `H`. For any `Q`-query `A` with an output point `xpt`,
`Pr_H[H(xpt(A^H)) ∈ bad(xpt(A^H), H)] ≤ (Q+1)·ε`. *Proof:* at the first
query of a point, its answer is independent of the adversary's view and of
`H` elsewhere. *Status:* proved as `RandomOracle.pinned_le` (`escape_le` for
every queried point). Ironwood proves the case where `bad` does not read `H`;
the local form is needed because a round's bad set reads the earlier
challenges.

**Lemma 3 (per-challenge test bound).** `ε_test` splits into point-indexed
round-by-round bounds. Fix a statement and a witness `w`. For each challenge
`c`, define `bad_w(prefix(c))` as the coins that turn a failing claim into a
passing one:

- α coordinate `j`: the multilinear `eq(α, ·)` batch is degree 1 in `α_j`,
  so the bad set has at most one value: `1/p²` per coordinate, `28/p²` total;
- γ: the joint residual is a polynomial in γ of degree `J − 1`, where
  `J = jointCoefficientCount = 16·54·8 + 1 + 17 = 6930`: `(J − 1)/p²`;
- SumCheck round `r`: a wrong degree-9 round polynomial agrees with the true
  one on at most 9 points: `9/p²` per round, `252/p²` total.

The sum is exactly `testError`. Each set depends only on `w`, the statement
and the prover data before `c`, which the prefix identifies (Lemma 1).
*Status:* proved as `RoundByRound.falseAcceptance_splits` with the counts
`alphaBad_probability_le`, `gammaBad_probability_le` and
`roundBad_probability_le`. The `α` split follows one fixed path of nonzero
sub-tables, one coordinate per step.

**Lemma 4 (test error in the ROM).** For a fixed fork state `x` (below) and a
fixed witness `w` that is not source-valid, the probability that the
continuation of `A` from `x` outputs an accepted proof whose extracted
witness is `w` is at most `(Q+1)·ε_test`. *Proof:* if the output is accepted
with witness `w`, some challenge of the final transcript lies in its
`bad_w` set (the round-by-round argument of the interactive proof, applied to
the final transcript). Apply Lemma 2 to each of the 57 extension-field
challenges and sum. *Status:* proved as `RandomOracleTest.test_error_le` for an
adaptive fresh statement, with the running statement and `w` fixed. The fork
form needs the pre-fork answers held fixed; `escape_le` is proved through that
form.

**Lemma 5 (Π_RLC coordinate retry in the ROM).** The interactive weak
extractor (SuperNeo B.3; Fenzi–Moghaddas–Nguyen Lemma 7.1, as formalized in
`PiRLC/CoordinateRetry.lean`) fixes the other coordinates and resamples one
coordinate until acceptance. In the ROM, the extractor reruns `A` with the
same tape and the same table except at the prefix of that coordinate, which
it reprograms with a fresh block. The other 16 coordinates keep their
answers, as the interactive retry requires, because each Π_RLC scalar is a
separate oracle point (the existing `scalarQuery` schedule). Reprogramming
can change which history `A` outputs; a retry counts only when the history
is unchanged. The expected-time argument of Attema–Fehr–Klooß
(JoC 2023, Theorem 3) for multi-round special-sound protocols gives the
`(Q+1)` factor on the per-coordinate loss. *Status:* new composition of a
published technique; needs the most careful review. The expected-work
accounting of `InteractiveWork` must be restated with `Q`.

**Lemma 6 (forked uniqueness; replaces SuperNeo v1.2 B.2 retry).** The
interactive argument uses two independent executions at one fixed context,
giving `E·S ≤ D + E·ε_test` and so `E ≤ ε_test + D/S` (proved as
`StrongProbability.local_source_error_le_retry`). In the ROM the statement is
adaptive, so the second execution is a *fork*: rerun `A` with the same tape
and the same answers up to the first query whose prefix extends the final
statement calls (the fork index `J`), and fresh answers from there. The fork
state `x` is `A`'s state at `J`; it fixes the statement, and so (by the
commitment) the witness up to an MSIS break. At each fork state the
interactive inequality holds with Lemma 4 in place of `ε_test`:
`e(x)·s(x) ≤ d(x) + e(x)·(Q+1)·ε_test`. Sum over the `Q+1` possible fork
indices; the disagreement mass becomes `ε_uniq`, which the existing adaptive
binding reduction bounds by MSIS. *Status:* new; the fork-index definition
and the summation need review.

## 5. What this changes and what it does not

- `FiatShamirModel` and its parameters `g`, `deltaFS` are retired. The
  history bound takes `Q` and the ROM terms above instead.
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

**Verifier fidelity.** The verifier already accepts a `Probe` with explicit
coins (`piCcsCheck_eq_true_iff_fixedWidthAccepted`). One definition,
`TranscriptCoverage.coinsFrom read`, builds the coins from any read of the
challenge prefixes. The deployed coins are `coinsFrom (readK ∘ run)`; the
oracle coins are `coinsFrom (decodeK ∘ H)`. Nothing else in the verifier
changes, so the theorem is about the deployed verifier with only the reads
idealized. A key with a call-list state was rejected: `readK` reads lane 0
of two consecutive states, so no single oracle function reproduces it, and
the key's construction and coverage proofs would be duplicated.

**Modules.**

| Module | Owns |
|---|---|
| `Spec/RandomOracle.lean` | `OracleComp`, `run`, `queries`, `QueryBound`, locality, `escape_le` (`Q·ε`), `pinned_le` (`(Q+1)·ε`); credit Ironwood |
| `Spec/.../PaperJoint/RoundByRound.lean` | deterministic split of a false acceptance into one bad coin per α coordinate, γ or round; bad-set counts `1`, `J−1`, `9` |
| `Lifecycle/TranscriptCoverage.lean` | `point`, `coinsFrom`, distinct point lengths, the deployed-coins bridge |
| `Lifecycle/RandomOracleTest.lean` | bounded domain, `decodeK`, oracle bad sets, `test_error_le`: `(Q+1)·testError` |

Phases 1 and 2 of the plan become these four modules. The forking lemmas
(Lemmas 5–6) and the history capstone follow after the review in Section 7.
At a fork state, Lemma 4 needs `escape_le` with the pre-fork answers held
fixed; its proof already carries that form.

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
   reduction forks, and it runs the adversary exactly twice. No re-fork bound
   is needed.
3. **Deployment margin.** `(Q+1)·ε_test` is linear in `Q`, so the cost per
   unit of success stays `2^115.2`: about 115-bit security, equal to the
   interactive bound, and tight against a grinding attack. Accept it for this
   work. Reaching 128 bits is a separate parameter decision (a larger field
   for `γ`).

References: Attema, Fehr, Klooß, Resch, ePrint 2023/1945; Attema, Fehr,
Resch, ePrint 2023/818; Attema, Fehr, Klooß, *Fiat–Shamir Transformation of
Multi-Round Interactive Proofs*, J. Cryptology 2023; Fenzi, Moghaddas, Nguyen,
*Lattice-Based Polynomial Commitments*, J. Cryptology 2024, Lemma 7.1;
Bellare, Neven, *Multi-Signatures in the Plain Public-Key Model and a General
Forking Lemma*, CCS 2006; zcash/ironwood `86e3c7026db8`,
`Zcash/Snark/Soundness/FiatShamir/PinnedSqueeze.lean`.
