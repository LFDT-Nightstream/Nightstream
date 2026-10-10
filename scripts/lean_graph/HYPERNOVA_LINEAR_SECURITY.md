# HyperNova linear-security milestone

This registration records the history-security criterion, adversary 4 of
`formal/nightstream-fprime/SECURITY_MODEL.md`. It takes HyperNova errata
Assumption 1, plain-model part, in the form of the paper's Definition 7
knowledge soundness of the NIFS, with an extractor for each admitted
adversary that reads that adversary's tape, composed by the reverse extractor
of HyperNova Lemma 17 (Appendix H.3). It differs from Definition 7 in six
stated ways: the success event adds the prior-state link, the transcript is not
the one of Construction 3, the premise is the joint form, the public parameters
are a fixed commitment setup, the structure is a fixed application, and
`error` replaces `negl(λ)` (trust item 4 of the security model).

The final declarations are
`NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity.history_failure_le`
and `NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity.reverse_input_event`.
The literal criterion is `LeanGraph.Targets.HyperNovaLinearSecurity` in
`tests/EvidenceTargets.lean`; `hyperNovaLinearSecurity` proves that criterion.

The required conclusion is, on the joint law of the IVC adversary and the
reverse extractor,

```
Pr[accepted terminal ∧ no valid history returned]
  ≤ Σ j < depth, (hashCollision_j + error(stage_j))
```

and that law has the IVC adversary's output law. This holds for every
application that fits the `2^28` profile, every commitment setup, every class
of admitted NIFS adversaries and efficient extractors with
`Assumption1 Admitted Efficient error`, every class of admitted stages with
`Closed Admitted StageAdmitted Efficient`, and every IVC adversary whose start
stage is admitted and whose advertised iteration is at most `depth`.
Stage `0` is the IVC adversary. Stage `j + 1` runs stage `j` and then the
extractor that Assumption 1 gives for stage `j`'s NIFS adversary; its tape is
stage `j`'s tape and that extractor's coins. `hashCollision_j` is the marked
state-hash collision mass of stage `j`, and `error(stage_j)` is the
Assumption 1 error of stage `j`'s NIFS adversary. Every stage is admitted.

Assumption 1 is the paper's plain-model knowledge-soundness premise for the
Poseidon2 NIFS. It is not a theorem: the step circuit recomputes the previous
fold's challenges with Poseidon2, so a recursive argument uses the concrete
hash, which no random-oracle model covers. `Admitted`, `StageAdmitted` and
`Efficient` are abstract; their intended meaning is expected polynomial time.
`Closed` states that an admitted stage, as one algorithm, gives an admitted
NIFS adversary and that one reverse step keeps a stage admitted, as in
Lemma 17 for a constant depth. Lean has no running-time model.
`Lifecycle.RandomOracleKnowledge.knowledge_error_le` proves a random-oracle
analogue for one fold and motivates the value of `error`: linear in the query
count, about `(Q + 74) · 2^-115.84 + 17 (Q + 17) · 2^-125.4` plus the named
MSIS and state-hash events. No Lean statement derives `error` from it
(trust item 4 of the security model).

Required premises:

- The advertised iteration of the IVC adversary is at most symbolic `depth`
  on its support.
- Assumption 1 for one class of admitted NIFS adversaries, and `Closed` for
  one class of admitted stages that contains the IVC adversary's start
  stage.

Dependencies, with namespace prefix `NightstreamFPrime`:

| Declaration | Required use |
| --- | --- |
| `Export.Stage1.HyperNovaFirstFailure.unreturned_acceptance_le_first_failures` | Bound each tape's acceptance with no returned history by its first marked failures. |
| `Export.Stage1.HyperNovaVisitedLaw.visitedLaw_listSource` | Read each tape's deterministic reverse path from its extractor results. |
| `Export.Stage1.HyperNovaVisitedSecurity.Assumption1` | State Assumption 1 in the form of Definition 7. |
| `Export.Stage1.HyperNovaVisitedSecurity.reverseStages` | Build the reverse extractor of Lemma 17. |
| `Export.Stage1.HyperNovaVisitedSecurity.failure_term_le` | Bound each stage's source failure by its Assumption 1 failure. |
| `Export.Stage1.HyperNovaVisitedSecurity.history_failure_le` | Average the per-tape bound over the reverse extractor's tape. |
| `Export.Stage1.HyperNovaVisitedSecurity.reverse_input_event` | Keep the IVC adversary's output law in the joint law. |

`hypernova-terminal-false-acceptance` uses the same premises through
`Export.Stage1.HyperNovaFalseAcceptance.probability_bound`, which follows from
`history_failure_le`.

Use `explain hypernova-linear-security` for remaining validation and review.
The registered gate runs static, build, axioms, the exact target check and
declaration export, in order. General build success cannot replace the exact
target and correspondence checks. Rust lifecycle conformance remains a
separate required check.

Reference: `docs/superneo-paper-v1_2` and the HyperNova errata in
`docs/hypernova-paper`.
