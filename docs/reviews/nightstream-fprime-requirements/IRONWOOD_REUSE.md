# Ironwood reference and reuse review

Reviewed on 2026-09-12. The [Zakura article](https://zakura.com/engineering/ironwood-zero-knowledge/)
links **zakura-core/ironwood-formal-verification, branch `establish-zk`**.
The downloaded snapshot is commit
[`22dfee003b639eff660f68ea69a98a00409a9cb1`](https://github.com/zakura-core/ironwood-formal-verification/commit/22dfee003b639eff660f68ea69a98a00409a9cb1),
which is also the article's explicit completeness anchor. It is a fork of
`zcash/ironwood`; it is not the current `zcash/ironwood` main-branch snapshot.
The complete archive, licenses, credits, and checksum are in
[external/ironwood/PROVENANCE.md](../../../external/ironwood/PROVENANCE.md).

This was a source and provenance review. No upstream Lean build, script, test,
fixture runner, dependency install, or production integration was performed.
The archive is reference material only. No existing assumption, requirement
status, package, profile, or runtime path changed.

The article-linked source has complete proof bodies for interactive statistical
zero knowledge and a separate Fiat–Shamir simulation result. In
[`actionFiatShamir_simulation_error_bound`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/ActionFiatShamir.lean#L66),
both directions of event-probability difference are bounded by
`plonkSimulationErrorBound actions + beforeBudget / scalarFieldOrder`.
The protocol has multiple rounds. The adversary can select its statement after
oracle preprocessing and can make further queries after the proof attempt.
The actual definitions erase the witness before the simulator call and retain
the final oracle cache, failed prefixes, and failure status. The proof composes
the pointwise continuation bound under the original preprocessing law.
See [`ActionOracleAdversary`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/ActionOracleAdversary.lean#L44).

Its conditions matter: the Action relation and matching public setup, eleven
IPA rounds, a nonidentity blinding generator, independent wide-reduced private
randomness, and a classical programmable random oracle. The field, group,
PLONK constraints, masking scheme, and transcript are those of the Orchard/
Zakura Halo2 construction. A query bound is not a machine-time bound. Seeded
randomness needs the separate stated PRNG security premise.

The following source is useful as a reference. The entries do not authorize a
new Nightstream framework or dependency.

| Reusable part | Exact source | Fit and limit |
| --- | --- | --- |
| Probability transport and mixture bounds | [`PMFEventBiasLE.bind_average`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Common/Oracle/Model.lean#L137), [`eventBias_map` and `mixedLaws_error_bound`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/Distribution.lean#L17) | Generic PMF mathematics. The latter retains arbitrary bad branches and charges their mass without conditioning the experiment on success. Use only if a concrete existing PMF proof needs that lemma. |
| Adaptive query accounting | [`OracleComp.runFreshPMF_eventBiasLE`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Common/Oracle/Hybrid.lean#L33), [`OracleComp`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Common/Oracle/OracleComp.lean#L17) | A bounded adaptive query tree accumulates a supplied one-answer bias as `Q * epsilon`. Repeated oracle queries must retain their answer; upstream uses deduplication/cache laws. This does not establish a random-oracle law for a concrete permutation. |
| Actual witness-to-circuit composition | [`actionWitnessRows_relation_capstone`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/ActionWitnessSimulation.lean#L25), [`ActionWitnessConstructionConditions`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/ActionWitnessConditions.lean#L38) | A useful proof pattern: application conditions produce the actual assignment, all rows, and copy equations; accepted emission is not an input premise. The Action/Sinsemilla/ECC circuit itself is not our selected F-prime circuit. |
| Prover replay with expected results kept separate | [`replay_eq_reference_capstone`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/Fixtures/Prover/Replay.lean#L48), [`Check.lean`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/Fixtures/Prover/Check.lean#L1) | The replay takes witness rows, RNG draws, and received challenges. Expected messages and bytes enter only later comparisons. This supports our existing same-input conformance method; their captured executions do not prove universal Rust equivalence or witness synthesis. |
| Explicit trust and coverage checks | [`assert_axioms`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Meta/AxiomCheck.lean#L390), [`check_endpoint_census.sh`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/scripts/check_endpoint_census.sh#L1), [`CensusCheck.lean`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/CensusCheck.lean#L1) | Direct endpoint coverage and explicit native-certificate owners make trust changes visible. Our existing axiom policy remains authoritative; importing their broader trust allowances is not part of reuse. |

The sparse IPA and linear multi-opening masking results are specific to their
polynomial commitment protocol. They are useful material for a separately
requested privacy study, not a proof of zero knowledge for our NIFS.
[`MaskSampling.lean`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/MaskSampling.lean#L1)
states the local sampled-mask comparisons and their nonzero/distinct-point
conditions. No masking or commitment randomization change is proposed here.

This source does **not** discharge Nightstream's approved Fiat–Shamir or MSIS
boundaries. Zero-knowledge simulation and knowledge extraction are different
claims. The fork's separate soundness development uses represented group
outputs in the algebraic group model and an idealized oracle for its transcript;
see [`ComputedAlgebraicFSFamily`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/Soundness/FiatShamir/Adversary/Algebraic.lean#L356)
and the [`oracle execution definition`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/Soundness/FiatShamir/Execution.lean#L1).
Neither supplies a concrete additive Poseidon2 transfer theorem, our `g` or
`deltaFS`, or a bound for the exact SuperNeo NIFS experiment. The reviewed tree
contains no Module-SIS/Goldilocks/SuperNeo result. Its elliptic-curve DLOG/IPA
reductions do not establish hardness of our fixed public-seed Ajtai matrix.
The existing [FS approval](FIAT_SHAMIR_MODEL.md) and
[fixed-seed MSIS assumption](PUBLIC_SEED_MSIS_ASSUMPTION.md) remain unchanged.

The checked-in trust declarations are more specific than a claim of no `sorry`.
The generic audit accepts `propext`, `Classical.choice`, and `Quot.sound`.
The concrete Action simulation endpoints also declare native-decision
dependencies on `CompElliptic.Curves.Pasta.Pallas.q_nsmul_Gpt` and
`CompElliptic.Curves.Pasta.Vesta.p_nsmul_Gpt`; see the
[HVZK audit](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/Action/TrustBoundary.lean#L227)
and [FS audit](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/Action/TrustBoundary.lean#L661).
The audit implementation checks the exact named native owners, rejects other
axioms, and checks compiled-body substitutions. Those allowances are part of
the upstream trust boundary, not our current three-axiom production policy.

A local lexical scan of all 1,351 archived Lean files, with comments and strings
removed, found no `sorry` or `admit` tokens. Its five explicit `axiom` declarations
are in the adversarial `Zcash/Meta/Tests` files. It found 341 `native_decide`
tokens across the archive. These are source observations, not an elaborated
axiom report; dependencies were not downloaded or checked.

The pinned snapshot must not be described as locally build-validated. Its
[README](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/README.md#L37)
marks the new completeness result as awaiting a final build. The
[`wideActionWitness_completeness_error_bound`](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/ActionProverCompleteness.lean#L110)
body exists and derives acceptance rather than assuming it, but this review
did not validate that body. At review, the pinned commit's
[aggregate Lean CI](https://github.com/zakura-core/ironwood-formal-verification/actions/runs/34645852179/job/103485331397)
reported failure and its
[formalization build](https://github.com/zakura-core/ironwood-formal-verification/actions/runs/34645852179/job/103416466737)
was cancelled. [Fixtures CI](https://github.com/zakura-core/ironwood-formal-verification/actions/runs/34645852159/job/103418024510)
reported success. Cancellation is not evidence of a false theorem; these
statuses simply do not establish a complete passing Lean build at this pin.

The upstream release provenance also records a permutation identity-set model
exception and invalid-witness lookup behavior. Its finite prover captures use
synthesized rows and captured challenges; they do not establish equality of
all Rust/Lean output distributions or the distribution of a concrete RNG.
These limits are stated in the
[release provenance](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/ZeroKnowledge/Zakura/PROVENANCE.md#L88)
and [fixture provenance](https://github.com/zakura-core/ironwood-formal-verification/blob/22dfee003b639eff660f68ea69a98a00409a9cb1/Zcash/Snark/Fixtures/Prover/PROVENANCE.md#L75).

Credit: Zakura contributors, including Tal Derei, for the article-linked work;
Zcash Protocol Developers and all other upstream contributors for Ironwood.
Upstream licenses the code under Apache-2.0 OR MIT. The complete license files
and all embedded notices are retained without changes. The downloaded branch
also includes the inherited `book/src/formal-verification/proof-map.md` and
`proof-map-embed.html`. The user's requested
[zcash proof map](https://zcash.github.io/ironwood/formal-verification/proof-map.html)
is a separate documentation reference; its adaptation and credit are tracked
by the map owner, not by this download.
