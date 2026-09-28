# Selected SuperNeo v1.2 linear-security milestone

Validated code: `1ad23f557b73e0564a27b62e20fe9d01a688f6b0` on
`nico/f-prime-constraints-cuda-formal`. The committed Lean and graph source
matches snapshot `be5c5cebc5ce537c9735b200f99c7737a8a1051d1ef3049f176ddc19ea01b1d3`.

## Result

`HyperNovaVisitedSecurity.history_probability_linear_bound` proves the final
selected history inequality with additive test loss and 17 times the actual
adaptive MSIS success at each guarded visit. `LeanGraph.Targets.HyperNovaLinearSecurity`
states the full criterion; `hyperNovaLinearSecurity` discharges it.

The reduction keeps both executable relaxed-acceptance checks, emits its
actual integer vector, and retains zero-success contexts. Its probability is
the limit of the finite driver probabilities averaged over the original
context law. The checked lean-graph path is:

```
AdaptiveBindingProbability.retryDisagreement_le_success
  → SupportedExtraction.returned_source_bound_with_adaptive_msis
  → FiatShamirTransfer.returned_source_bound_with_adaptive_msis
  → NifsClosure.source_probability_linear_bound
  → NifsProviderLaw.source_probability_linear_bound
  → HyperNovaVisitedSecurity.source_failure_probability_linear_le
  → HyperNovaVisitedSecurity.history_probability_linear_bound
  → LeanGraph.Targets.hyperNovaLinearSecurity
```

`AdaptiveBindingWork` proves the actual-step clock correspondence, complete
mean convergence and entered termination. At each context its bound is
`2*m + 3 + H`, where `m` is the actual complete checked-call mean and `H` is
the existing vector-computation and control allowance. The original context
average and preparation are included under explicit source and preparation
moment premises. The prepared polynomial bound has the derived constant 22.
These results use the declared mathematical clock; they do not measure a
machine runtime.

## Validation

| Check | Result | Command time |
| --- | --- | --- |
| Static boundaries | Pass | 8.63 s |
| Full library, cold checker build | Pass | 493.55 s |
| Full axiom gate | Pass | 65.73 s |
| Exact target build | Pass | 0.87 s |
| Exact target acceptance | Pass | 1.97 s |
| Declaration export | Pass | 194.49 s |
| Existing lean-graph tests, including the build-target regression | 82 pass | 5.91 s |

The independent review passes statement, premises, argument, correspondence
and parent use on this same snapshot. Only `propext`, `Classical.choice` and
`Quot.sound` occur in the target audit. The complete graph reports the source
bound connected to the final target. Checker preparation, metadata processing
and artifact checks are additional to the command times above.

The requirements update changes links and claim text on seven NIFS/HyperNova
security records. Existing proof and assumption statuses are retained. The
local site build passes 2,130 source-location checks, 19 Python tests and 12
JavaScript tests. It does not publish the live site.

Evidence: [source, logs, normalized graph and review archive](SUPERNEO_V1_2_LINEAR_SECURITY_EVIDENCE.zip).
Archive size: 38,844,481 bytes. SHA-256:
`5eafcef9cf672b712d0f0a96251057ca9a5a67bc051632ee6aa3c2a69055c23b`.
Its manifest records every retained file hash. The duplicate raw graph log
remains in the external lean-graph store; its path and hash are in the manifest.
The graph results and review are local diagnostics. Protected-checker approval
is not configured, so the tool's separate accepted-closure flag remains open.

## Scope

The approved classical additive-Poseidon2 transfer and fixed public-seed MSIS
conditions remain explicit. No numerical hardness advantage, query cutoff,
efficient general FS translation or new cryptographic premise is supplied.
Expected invocation count is not a deterministic permutation-query bound.
Depth and query counts remain parameters. The existing square-root results
remain available; this is a new checked final consumer, not a formula edit.
The proof covers the adaptive strategy required by this selected consumer.
It does not claim that every abstract reduction interface has been ported to
the revised paper's universal uniqueness definition.

No Rust or package bytes changed. The next milestone in the parent goal is
the remaining selected Rust lifecycle conformance. Native execution results
retain their existing scope; this proof does not close those records.

The argument follows Wilson Nguyen and Srinath Setty's supplied September 4
SuperNeo v1.2 paper, especially Appendix B.2, and uses the pinned Mathlib PMF
and convergence results. The prior paper comparison is in
[SUPERNEO_V1_2_DELTA.md](SUPERNEO_V1_2_DELTA.md).
