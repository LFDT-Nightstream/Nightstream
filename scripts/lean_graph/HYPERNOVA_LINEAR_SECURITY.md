# HyperNova linear-security milestone

This registration records the history-security criterion. Since 2026-10-07
(owner decision) it takes HyperNova errata Assumption 1, plain-model part, at
each visited step instead of the retired `FiatShamirModel`. The existing
lean-graph schema and review process are unchanged. The target's meaning
changed, so its recorded target-meaning and decomposition reviews must be
renewed.

The final declaration is
`NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity.history_probability_bound`.
The literal criterion is `LeanGraph.Targets.HyperNovaLinearSecurity` in
`tests/EvidenceTargets.lean`; `hyperNovaLinearSecurity` proves that criterion.

The required conclusion is

```
Pr[accepted terminal] ≤ Pr[returned history advice]
  + Σ j < depth, (hashCollision_j + error_j)
```

for every source extractor that satisfies
`HyperNovaVisitedSecurity.NifsKnowledgeSound source initial depth error`: after
a real NIFS acceptance at visit `j` (`goodActive`), the extractor returns no
checked source witness with probability at most `error_j`. The visited laws,
the marked hash-collision events and the source-failure event
(`HyperNovaFirstFailure.MarkedSourceFailure`) are the existing ones.

Assumption 1 is the paper's plain-model knowledge-soundness premise for the
Poseidon2 NIFS. It is not a theorem: the step circuit recomputes the previous
fold's challenges with Poseidon2, so a recursive argument uses the concrete
hash, which no random-oracle model covers.
`Lifecycle.RandomOracleKnowledge.knowledge_error_le` proves a random-oracle
analogue for one fold and motivates `error_j = knowledgeError(Q_j)`: linear in
the query count, about `(Q_j + 74) · 2^-115.84 + 17 (Q_j + 17) · 2^-125.4`
plus the named MSIS and state-hash events. No Lean statement derives `error_j` from it; the success
event, the extractor and the efficiency condition differ
(`formal/nightstream-fprime/TRUST_BOUNDARY.md`). Definition 7 also requires an
expected polynomial-time extractor. `NifsKnowledgeSound` does not state that
requirement, so the criterion is weaker than Assumption 1 there. Lean counts
the history's extractor calls
(`Export.Stage1.HyperNovaHistoryWork.source_calls_le_iteration`), and the
random-oracle extractor takes `17 (Q + 17)` expected retries; neither is a
machine-time bound.

Required premises:

- The initial counter is at most symbolic `depth` on its support.
- Assumption 1 at every visit of the same history law.

Dependencies, with namespace prefix `NightstreamFPrime`:

| Declaration | Required use |
| --- | --- |
| `Export.Stage1.HyperNovaFirstFailure.accepted_probability_le_first_failures` | Bound the accepted mass by the first marked failures at the actual visits. |
| `Export.Stage1.HyperNovaVisitedSecurity.NifsKnowledgeSound` | State Assumption 1 at the actual visited laws. |
| `Export.Stage1.HyperNovaVisitedSecurity.history_probability_bound` | Compose the first-failure bound with Assumption 1. |

`hypernova-terminal-false-acceptance` uses the same premise through
`Export.Stage1.HyperNovaFalseAcceptance.probability_bound`.

Use `explain hypernova-linear-security` for remaining validation and review.
The registered gate runs static, build, axioms, the exact target check and
declaration export, in order. General build success cannot replace the exact
target and correspondence checks. Reuse the existing evidence store and graph
queries. Rust lifecycle conformance remains a separate required check. Sampler integration
changes the selected artifacts, so old fixture receipts cannot validate it.

Reference: `docs/superneo-paper-v1_2`, `SUPERNEO_V1_2_DELTA.md`,
`FIAT_SHAMIR_MODEL.md` and `PUBLIC_SEED_MSIS_ASSUMPTION.md` under
`docs/reviews/nightstream-fprime-requirements` for the latter three documents.
