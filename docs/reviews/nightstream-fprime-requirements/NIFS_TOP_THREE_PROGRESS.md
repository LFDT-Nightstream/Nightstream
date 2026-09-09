# NIFS top-three completion

All eight requested records are proved and connected at their stated
conditional interactive scope. NIFS now shows **Proof 14/21** and **Link 18/26**.

Base: `7099ea842171a9a53114644612766a2522b4c4d0`, plus the checked runtime closure.

## Closed records

| Record | Checked owner |
|---|---|
| N.profile.relation | NifsProfile.selected_relation / selected_matrix |
| N.profile.shape | NifsProfile.selected_shape |
| N.profile.arity | NifsProfile.selected_arity |
| N.profile.setup | NifsProfile.shared_setup / phases_preserve_relation |
| N.security.strong | StrongExtraction.probability_and_expected_work |
| N.security.aligned_fork | PaperAlignedExtraction.positive_return_implies_alignedFork |
| N.security.weak | WeakExtraction.weak_relaxed_success_bound; InteractiveCompleteness.exists_honest_execution; InteractiveAgreement.disagreement_le_bindingProbability; SupportedExtraction.probability_and_expected_work |
| N.security.composition | SupportedExtraction.probability_and_expected_work |

## Final result

The final theorem connects one actual checked PiCCS prefix, its captured
PiRLC/PiDEC continuation, and the source values returned by the checked
projection. Its first conjunct uses the actual call-result equality to align
the work calculation with the receipt used by the probability law.

The same theorem proves global work summability, an explicit polynomial
expected-work bound, and source extraction success at least the original
success probability minus

`17 / |C| + sqrt(binding-event probability + PiCCS test error)`.

Only positive, reachable continuation inputs need finite call moments.
Unreachable inputs use the proved abort extension. Coupling tables used in the
probability argument are not executed by the extractor. Two unequal actual
fork returns give a collision from their short response differences; there is
no arbitrary-ambient uniqueness premise.

The runtime repair removes a redundant finite-type instance and reduces the
two generated prefix matches by their `none` and `some` cases. It changes no
protocol relation, clock, layout, parameter, transcript or package identity.
The runtime module is now in the library and axiom audit. Earlier paused drafts
remain local historical evidence, not the current implementation.

## Explicit boundaries

The selected profile remains Goldilocks, b=2, k_rho=16, B=65536, one fresh
source and 16 running sources. Low-norm invertibility, exact call/check/access
correctness, and the displayed polynomial bounds on actual call and primitive
work remain explicit premises.

The measured binding-event probability is not replaced by a numerical MSIS
estimate. `N.security.binding` still owns the computational reduction for the
approved Nightstream-specific public-seed setup. `N.security.fiat_shamir`
still owns transfer from independent interactive coins to the selected
Poseidon2 transcript. HyperNova history extraction, context selection and
production acceptance keep their own open records.

## Validation

- Boundary gate: pass.
- Full Lean library: 3781 jobs, 7 seconds.
- Full test/axiom library: 3819 jobs, 5 seconds.
- NIFS audit: 118 declarations, only `propext`, `Classical.choice`, `Quot.sound`.
- Site build and all seven export tests: pass.
- No Rust changes; native conformance was not rerun for this proof-only change.

Evidence is in the local sibling directory
`../nightstream-stage1-evidence/nifs-top-three-2026-09-09/`:
`closure-runtime-check.log`, `closure-combined-check.log`,
`closure-full-audit.log`, and `closure-source-manifest.json`.

## Publication

The public map closes only the two requested interactive records. Its live JSON
matches the validated source. Site version 19; source
`5a892da188a4851efd5d5a8f6fe4d3a0d1bfc4cd`; deployment
`appgdep_6aa0e917f8e88191bcfd0dd5d1b1d39a`.

https://nightstream-requirements.nicarq.chatgpt.site/#group-N
