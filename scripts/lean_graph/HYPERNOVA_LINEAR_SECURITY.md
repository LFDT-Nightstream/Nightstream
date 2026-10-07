# HyperNova linear-security milestone

This registration records the history-security criterion based on the
September 4 SuperNeo v1.2 source. The sampler replacement keeps the original
history event and reduction, and makes its statistical transfer loss explicit.
The existing lean-graph schema and review process are unchanged.

The final declaration is
`NightstreamFPrime.Export.Stage1.HyperNovaVisitedSecurity.history_probability_linear_bound`.
The literal criterion is `LeanGraph.Targets.HyperNovaLinearSecurity` in
`tests/EvidenceTargets.lean`; `hyperNovaLinearSecurity` proves that criterion.
It quantifies state and tape types in `Type`; the production theorem also
supports higher universes. The existing square-root theorem remains available.

The required conclusion is

```
Pr[accepted terminal] ≤ Pr[returned history advice]
  + Σ j < depth, (hashCollision_j + active_j - g(Q_j)(active_j) + deltaFS(Q_j)
      + sampleQueries(Q_j) * samplerDistance
      + weakLoss + testError + 17 * adaptiveMsisSuccess_j).
```

`samplerDistance < 2^-132` is proved for one uniform four-field draw.
`sampleQueries` counts all block-oracle calls of the translated experiment,
including adversarial calls and replays. It is not replaced by the 17 verifier
challenges. The cumulative sampler term is
`samplerDistance * sum_j sampleQueries(Q_j)`; `samplerTransferError_sum`
proves this accounting. Concrete Poseidon2/FS applicability remains external.

Each term uses the same actual guarded visit and source laws as the existing
history theorem. `adaptiveMsisSuccess_j` must be the success probability of the
computed adaptive reduction, including its uniform choice among the 17 inputs.
It must not be a free error parameter or a renamed disagreement event. The
factor 17 is `PaperProfile.arity.total`, not a new selected parameter.

Required premises:

- The initial counter is at most symbolic `depth` on its support.
- The fixed selected continuation, source program and tapes supply the same
  operational source and all visit laws. Existing storage and primitive clock
  bounds, correctness facts and summability stay visible. The concrete Phi81
  low-norm invertibility fact is supplied by its checked proof.
- Every guarded real experiment satisfies the approved classical
  additive-Poseidon2 `FiatShamirTransfer.FiatShamirModel`. Since 2026-10-06 its
  success event also requires a prior preimage, output by the adversary, that
  links the running statement and the verifier context to the absorbed digest.
  The model has shared `g` and
  combined error `samplerTransferError deltaFS sampleQueries` and its own total
  permutation-query count `Q_j`. `FiatShamirModel.of_blockOracle` derives this
  interface from an explicitly supplied raw/balanced block-oracle game transfer
  using the proved sampler comparison. It does not instantiate that transfer
  for concrete Poseidon2. No default query inflation or numerical model is supplied.
- The public fixed-seed MSIS assumption can bound the computed reduction only
  after its adaptive calls, termination, expected work, preprocessing and query
  use fit the approved model. Its hardness bound remains an explicit assumption.

The current history premises do not prove a global work bound. A hardness
corollary must retain the required context-weighted moments, preparation and
polynomial bounds for the same computed reduction. The approved FS boundary
also retains responsibility for total forward/inverse query applicability.
An expected invocation bound does not give a deterministic query bound for
unbounded retries. No cutoff or numerical tail allowance is selected here.

Dependencies retained from checked commit `0b42eed2`:

| Declaration, with namespace | Required use |
| --- | --- |
| `Spec.Folding.Nifs.SequentialObservationLaw.retryDisagreement_eq` | Identify the local ratio from actual retained observations. |
| `Lifecycle.Nifs.InteractiveComposition.source_success_retry_bound` | Supply the additive source inequality. |
| `Lifecycle.Nifs.AdaptiveBinding.check_correct`, `callClock_eq` | Check actual relaxed success and account for each complete call. |
| `Lifecycle.Nifs.BindingProbability.supported_binding_le_success` | Bound a supported selected-pair collision by the actual emitted vector's success. |
| `Spec.Folding.PiCCS.PaperJoint.AcceptedRetryLaw.entered_expectedWork_tendsto` | Sum entered retry work, including rejected calls. |
| `Export.Stage1.HyperNovaFirstFailure.accepted_probability_le_first_failures` | Compose losses at actual history visits. |
| `Export.Stage1.HyperNovaVisitedSecurity.history_probability_bound` | Reference for the unchanged final event, premises and history law. |

All namespaces in the table have prefix `NightstreamFPrime`. The new
`AdaptiveBindingProbability.retryDisagreement_le_success` connects the gated
pair to the actual original-context average; `successProbability_tendsto`
proves its finite-driver interpretation. The linear consumer chain is
`SupportedExtraction` → `FiatShamirTransfer` → `NifsClosure` → `NifsProviderLaw`
→ `HyperNovaVisitedSecurity`.

`AdaptiveBindingWork` supplies the actual-step mean equalities, convergence,
entered termination, context moment bound and prepared polynomial bound.
These are separate graph roots because a probability inequality alone does
not prove the resource or execution correspondence obligations.

Use `explain hypernova-linear-security` for remaining validation and review.
The registered gate runs static, build, axioms, the exact target check and
declaration export, in order. General build success cannot replace the exact
target and correspondence checks. Reuse the existing evidence store and graph
queries. Rust lifecycle conformance remains a separate required check. Sampler integration
changes the selected artifacts, so old fixture receipts cannot validate it.

Reference: `docs/superneo-paper-v1_2`, `SUPERNEO_V1_2_DELTA.md`,
`FIAT_SHAMIR_MODEL.md` and `PUBLIC_SEED_MSIS_ASSUMPTION.md` under
`docs/reviews/nightstream-fprime-requirements` for the latter three documents.
