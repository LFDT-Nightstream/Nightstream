# Assurance surface

This document identifies the final row theorems and their scope. The selected
Nightstream Goldilocks profile uses `b = 2`, `k_rho = 16`, 17 sources, 16 children,
14 matrices, 28 rounds, and Poseidon2 binding. A successful build checks the
stated conclusions under their stated hypotheses.

The current production entrypoint selects the wide sampler package. Its
proof authority is listed in the wide-sampler section below. The earlier
baseline declarations and native trace records remain evidence for their
recorded package versions. The current native integration checks and open
fixture work are recorded in the integration report.

The selected native NIFS-to-successor bridge is checked at `68c5d94a`.
It constructs the complete caller packet from the actual saved NIFS output,
matches every caller word with Lean, and passes independent complete
physical/logical rows and child mutations. See [the exact execution scope](../../docs/reviews/nightstream-fprime-requirements/NATIVE_SUCCESSOR_EVIDENCE.md).
Child witness retention and terminal verification are supplied by the
subsequent checkpoints below.

The next native milestone, `50127ac9`, retains all actual child witnesses and
commits and retains the exact complete fresh matrix in the existing proof-state
type. Its initial empty case and actual successor pass the recorded checks.
See [the envelope evidence](../../docs/reviews/nightstream-fprime-requirements/NATIVE_ENVELOPE_EVIDENCE.md).

At `5b544af3`, the selected native terminal verifier accepts that same
complete envelope against the independent Lean endpoint. It recomputes
all running commitments and Pad/matrix evaluations, the full state/public
link, and the fresh commitment and complete CCS relation. Rehashed and
recommitted Pad/matrix mutations reject at their specific opening checks;
a recommitted private unit mutation fails a CCS row. Initial/counter/state
checks pass, and non-authoritative parent/frame/scalar caches are ignored.
All four full-profile cases pass within the native cap. See
[the terminal evidence](../../docs/reviews/nightstream-fprime-requirements/NATIVE_TERMINAL_EVIDENCE.md).
This closes the recorded initial/base/actual-C-R-D/successor/terminal trace.
It does not extend execution coverage to later running inputs, prove
arbitrary Rust semantics, select a new backend or grant production approval.

**Selected rows and completeness.** Declaration names below omit only the common `NightstreamFPrime` namespace.

| Exact declaration | Proved result and scope |
|---|---|
| `Export.Stage1.PerApplicationCanonicalPackage.matrixProgram_exact`, [source:226](NightstreamFPrime/Export/Stage1/PerApplicationCanonicalPackage.lean#L226) | The selected compact interpreter has the structural plan's exact row count and returns the same form at every row. The owner supplies source custody through `PerApplicationPackageSourceCustody.custody`; the selected theorem has no caller-supplied custody premise. This equality preserves row semantics in both directions. |
| `Export.Stage1.ActualPiDECOutput.selectedRowsAndPublic_imply_step`, [source:153](NightstreamFPrime/Export/Stage1/ActualPiDECOutput.lean#L153) | For an arbitrary selected assignment, `RowsZero`, the actual public-input equality, and a four-word digest imply the full typed `StepHoldsFor` under the decoded context. Its component theorems identify the actual PiDEC check and all 16 returned children. No honest encoder, assumed NIFS acceptance, or assumed child-output equality is required. Binding the decoded context to the verifier's context is a separate step. |
| `Export.Stage1.Poseidon2HashChainV1Closure.rowsZero_implies_stepHoldsFor`, [source:35](NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Closure.lean#L35) | Specializes the complete step to the fixed application and setup. Its input is `RawValues` after the actual `bound` operation, with a `RowsZero` premise. Keep this input scope distinct from the arbitrary-assignment theorem above. |
| `Layout.PiDEC.v1_1.physical_complete_production`, [source:161](NightstreamFPrime/Layout/PiDEC/v1_1/Preservation.lean#L161) | From `InputShapes`, `Formal.Assumptions`, and `Semantics.PhaseHolds`, constructs satisfying physical PiDEC values. It preserves the environment outside `Formal.logicalPrivateCount + exactFreshCount`. This is the completeness boundary that a PiDEC allocation or constraint change must retain. |
| `Export.Stage1.PiDECPackageCompleteness.completePackageRows`, [source:658](NightstreamFPrime/Export/Stage1/PiDECPackageCompleteness.lean#L658) | Completes the PiDEC and running-transition region of `Data.circuitPackage`. It requires the stated PiRLC/PiDEC phase and shape assumptions, running-transition specification, pilot checks, PiCCS rows, and PiRLC physical rows. It does not construct these preceding rows from an arbitrary valid complete step. |

`Export.Stage1.SelectedAssignmentCompleteness.complete` ([source](NightstreamFPrime/Export/Stage1/SelectedAssignmentCompleteness.lean)) constructs one canonical assignment satisfying the complete fixed-application structural plan, with strict carrier norm below two, the exact public output digest, and the actual application advice. Its hypotheses are the typed semantic step, well-formed prior and next preimages, the fresh public link, actual NIFS acceptance, agreement of the recursive result, and four-word advice. The same assignment satisfies every conclusion; no phase-row, environment-agreement or completeness callback is assumed. This is a mathematical existence theorem, not an executable Rust producer or unconditional sampler-success claim.

The constructor consumes `Export.Stage1.DirectApplicationPrefixPlan.rowsZero_iff` ([source:206](NightstreamFPrime/Export/Stage1/DirectApplicationPrefixPlan.lean#L206)), which gives the exact conjunction of prefix, application, next-preimage and public-output rows. `PerApplicationAssignmentTransportExecution.canonical_execute_eq_assignment` ([source:489](NightstreamFPrime/Export/Stage1/PerApplicationAssignmentTransportExecution.lean#L489)) separately identifies execution of the schema-6 interpreter with the canonical logical assignment.

`Export.Stage1.HyperNovaAcceptedNext.recursive_extend_of_sampler_success` and `base_extend_of_sampler_success` ([source](NightstreamFPrime/Export/Stage1/HyperNovaAcceptedNext.lean)) construct accepted successor envelopes for the selected application from accepted recursive and initial envelopes. They require four-word advice and success of the actual bounded sampler; the recursive branch also requires the successor counter below the field modulus. The C messages and full output are fixed before the recursive sampler condition. Both branches commit the exact constructed fresh carrier and retain it as the fresh opening. The recursive envelope retains the actual returned NIFS child witnesses; the base envelope retains default running claims with their proved zero openings, while its dummy verifier result is used only to construct the rows. These are existence proofs in the checked model. They do not claim sampler totality, unconditional perfect completeness, Rust execution correctness or machine runtime.

**Actual NIFS and security consumers.**

- `Export.Stage1.Wide.OpeningBinding.freshHolds_implies_rowsAndPublic` ([source:102](NightstreamFPrime/Export/Stage1/Wide/OpeningBinding.lean#L102)) reaches the wide package rows from semantic terminal membership. For each `Wide.Target`, `terminal_implies_nifsOrBaseOrCollision` and `terminal_implies_parentOrBaseOrCollision` ([source:166](NightstreamFPrime/Export/Stage1/Wide/TerminalSecurity.lean#L166), line 204 in the same file) consume the actual decoded input, proof, and terminal witnesses. They give the exact wide-key NIFS output, a valid recomposed parent, or the named base/collision alternatives. These are implications from `Terminal.HoldsFor`, not a theorem about an arbitrary Rust Boolean verifier result. The baseline-key `terminal_implies_securityOrCollision` was removed with the baseline security owners; the wide history bound replaces its role.
- `Poseidon2HashChainV1Closure.rowsZero_implies_base_or_securityOutcome` and `expectedBindingAndRowsZero_implies_securityOrCollision` ([source:64](NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Closure.lean#L64), line 107 in the same file) retain low-norm invertibility and explicit collision alternatives. `Spec.Folding.Nifs.PaperSecurityComposition.SecurityOutcome` ([source:466](NightstreamFPrime/Spec/Folding/Nifs/PaperSecurityComposition.lean#L466)) includes knowledge, mixing, sumcheck, parent-binding, missing-fork, and missing-child-opening cases. Its exhaustive case split is not a probability bound on those failures.
- `Export.Stage1.NifsClosure.source_probability_linear_bound` ([source:186](NightstreamFPrime/Export/Stage1/NifsClosure.lean#L186)) proves the selected v1.2 lower bound `g Q p_real − deltaFS Q − weakLoss − testError − 17 * adaptiveMsisSuccess` on the constructed `HyperNovaSourceLaw.law` PMF and its actual `SourceReturned` event. `AdaptiveBindingProbability.successProbability_tendsto` identifies the MSIS term with the limit of the actual driver's emitted-vector success, retaining both acceptance gates, aborts and the original context law. `AdaptiveBindingWork` proves the actual-step clock correspondence, entered termination, complete mean convergence and prepared polynomial bound under explicit moments. The existing square-root and source-extractor work theorem remains available as `finishValue_probability_and_expected_work`.
- `Export.Stage1.HyperNovaHistory.run_correct` proves correctness of the actual reverse run on a supplied list of NIFS source results. Under accepted terminal membership, the exact encountered `SourceReturned` events and absence of the encountered state-hash collisions, it returns advice of length `statement.iteration` whose forward execution reaches `statement.zi` from `statement.z0`. It uses the exact decoded claims, reconstructs accepted predecessors and consumes no source result at the base step. The result generator and its probability bound are described below; the selected probability and declared-work bounds are now composed below.

- `Export.Stage1.HyperNovaRealInput.realSuccess_of_terminal` supplies the exact real-game input from an accepted recursive terminal opening, absence of its state-hash collision, and a positive decoded predecessor iteration. It uses the actual decoded local proof and the terminal's current child witnesses. No separate child-correctness or verifier-output equality is assumed.
- `Export.Stage1.HyperNovaHistoryLaw.results` generates the source returns consumed by the reverse run under the supplied state-dependent source kernel. Its private structural counter is exactly `statement.iteration`; the proof excludes truncation. `accepted_probability_le` bounds initial accepted mass by complete returned-history mass, visited source-failure mass, and encountered state-hash failure mass. It preserves the initial marginal and aborts. The selected fixed source family and guarded security instances are now consumed by `HyperNovaVisitedSecurity`; `HyperNovaSourceWork` composes the unconditional operational source-work moments. No point-mass or conditional model is inferred from an unconditional game-transfer hypothesis.
- `Export.Stage1.HyperNovaVisitedAcceptance.marked_accepted` derives actual terminal membership from each supported history mark. `realSuccessProbability_eq_goodActive` identifies the real NIFS success mass. `NifsProviderLaw` constructs one fixed continuation from the same raw calls, tapes and clocks, and proves equality of its source law with the supported extension used by the selected NIFS consumer. Total-family finite call moments are explicit construction hypotheses; no value-check or source-law equality is assumed.
- `Export.Stage1.HyperNovaVisitedSecurity.history_probability_linear_bound` ([source:337](NightstreamFPrime/Export/Stage1/HyperNovaVisitedSecurity.lean#L337)) composes the actual history bound with additive test loss and 17 times adaptive MSIS success at each visit. It uses one shared `g` and `deltaFS` for the exact guarded model family at symbolic depth, and marked hash-collision masses. `HyperNovaFirstFailure` covers the base hash event and absent source results. The remaining hypotheses specify the exact models, invertibility, declared primitive/storage bounds, finite call moments, and initial depth support. The experiment is not conditioned on acceptance or successful extraction. Numerical cryptographic advantages and total query applicability remain external. `HyperNovaSourceWork.expected_work_polynomial_bound` supplies the separate source-work bound. The exact target and validation are registered as `hypernova-linear-security` in the existing lean-graph; see [the milestone report](../../docs/reviews/nightstream-fprime-requirements/SUPERNEO_V1_2_LINEAR_SECURITY.md).
- `Export.Stage1.HyperNovaHistoryWork` bounds all generated source calls by the advertised iteration, including abort and false-mark paths. Its separate declared orchestration allowance is one initial entry plus one processed source return; its expectation is at most `D + 1`. It excludes payload decoding, copying and advice evaluation, and is not a source-work or machine-time bound.
- `Layout.Stage1.PiCCSProtocolCompleteness.completePrefix` now constructs the local PiCCS prefix from typed canonical prior/output states, their verifier context, and actual PiCCS acceptance. The existing context slots are loaded while protocol input readback is preserved; serializer theorems derive state framing. The constructor derives the generated phase specification, including the round point and outgoing transcript state. The complete selected assignment is supplied by `SelectedAssignmentCompleteness.complete`; the honest outer prover remains separate.
- `Lifecycle.Nifs.BaseCompleteness.zeroProof_verify_of_sampler` proves complete NIFS acceptance of the canonical base dummy when its actual bounded sampler succeeds. It derives the parent public bound and the D checks. It supplies no fresh or child opening; the zero assignment does not open the base fresh claim's encoded hash.
- `Lifecycle.PiRLC.v1_1.SamplerChain.completePrefix_of_available` constructs all sampler children from actual bounded executions. `Formal.completePrefix_of_available` reuses the existing R assembler and derives its generated `PhaseHolds` from the rows. These take the existing source bounds and actual sampler availability, without a generated-output or parent-bound premise. `Layout.Stage1.PiRLCProtocolCompleteness.completePrefix` connects actual typed C inputs to C/R prefixes, preserves C rows and identifies the actual sampled challenges and production parent. `PiDECProtocolCompleteness.completePrefix` now constructs local C/R/D, preserves C/R rows and derives the exact D output. `SelectedAssignmentCompleteness.complete` supplies the complete selected assignment under its stated step and acceptance conditions.

- `Export.Stage1.HyperNovaSourceWork.expected_work_polynomial_bound` bounds expected source work on the actual operational history, including abort and false-mark paths. It proves the context/kernel equality and sums the existing source-clock bounds over unconditional visited laws at symbolic depth `D`. Primitive/storage bounds and finite raw-call moments remain explicit. Adding the existing control allowance gives the stated declared-clock bound; payload decoding, copying, advice evaluation and machine runtime are excluded.

- `Export.Stage1.HyperNovaInitial.initial_accepted` constructs the initial iteration-zero statement and bottom envelope from an initial state of the fixed application width. Equal endpoints and counter validity are derived.
- `Layout.Stage1.PiDECProofInputs` loads the actual D proof commitments, Pad/matrix evaluations and verifier-computed public digits into the existing four source ranges. Exact typed readback and preservation outside those ranges are proved. D acceptance, child openings and selected-row construction remain separate.

- `Layout.Stage1.PiDECStepCompleteness.recursive_completePrefix` derives the local C/R/D constructor inputs from an actual valid positive selected SuperNeo step. Its proof and fresh data come from that step's actual advice through `PiCCSProofReadback`. No generated phase, sampler-success, parent-bound or child-opening premise is added. It does not construct the pilot, application, transition, physical lowering or complete low-norm assignment.
- `Lifecycle.PiRLC.v1_1.Formal.retainedSamplerBits_of_rows` derives the actual retained reject/position Boolean facts from R rows and the existing circuit assumptions. Their selected-assignment norm consumer is separate.
- `Export.Stage1.HyperNovaEnvelopeSize.accepted_wordCount_le` bounds all existing running/fresh claims and complete opening domains by the fixed application dimensions, independent of iteration, and proves accepted recursive pc is one. It counts dense field words and framing; it makes no Rust wire-format, heap or runtime claim.

**What is proved locally, and what remains explicit.**

- The selected checker and primitive **value** obligations are discharged, not assumed by the final theorem: `PiCCSStoredWitnessCheck.finishValue_source_iff` ([source:153](NightstreamFPrime/Export/Stage1/PiCCSStoredWitnessCheck.lean#L153)), `PiCCSStoredSourceProbability.sourceProgram_correct` ([source:93](NightstreamFPrime/Export/Stage1/PiCCSStoredSourceProbability.lean#L93)), and `PiRLCExtractionPrimitives.program_correct` ([source:65](NightstreamFPrime/Export/Stage1/PiRLCExtractionPrimitives.lean#L65)). Abort and malformed-certificate rejection remain in the value semantics.
- The final consumer constructs its continuation with `NifsExtractionProvider.provider`. `ClaimCheck.check_eq_true_iff` checks the full CE opening; `suffixProgram_correct` checks the actual PiDEC attempt and all 16 child openings; `recompose_value` and `parentChecker_spec` check the exact recomposed parent. `batchAt_eq` connects the selected relation, statement and receipt to the existing continuation. These facts discharge the suffix and parent-check correctness fields. The final consumer executes the exact checked prefix and uses identity preparation under the supplied context law. It requires no free provider, `callCorrect` or prepared-context equality. Low-norm invertibility, raw adversary calls and tape laws, declared clock bounds, access bounds and summability/polynomial moment bounds remain explicit. Identity preparation does not implement a sampler or adversary translation. `Spec.Phi81StrongSet.LowNormInvertibility` ([source:213](NightstreamFPrime/Spec/Phi81StrongSet.lean#L213)) is an external mathematical theorem parameter, not a cryptographic hardness assumption. The delivered work claim uses its declared mathematical clock; it is not a Lean/Rust machine-time bound or a bound for an unspecified adversary translator.
- Named Poseidon2 collision events remain explicit. Treating them as negligible requires the applicable cryptographic security premise. The approved [public-seed MSIS assumption](../../docs/reviews/nightstream-fprime-requirements/PUBLIC_SEED_MSIS_ASSUMPTION.md) is specifically for the frozen generated matrix and strict norm below `8TB`; it is stronger than a setup-average or uniform-matrix assumption. `NifsBinding.bindingEvent_to_shortKernel` ([source:29](NightstreamFPrime/Export/Stage1/NifsBinding.lean#L29)) connects the actual binding event to that search problem. No numerical hardness bound follows from this approval.
- `Spec.AjtaiSetupV1.Prefix.extendShortKernel` preserves a nonzero short integer kernel when a same-seed key prefix is extended by zero blocks. Padding starts after the smaller complete carrier. This is the deterministic prerequisite for retaining the existing hardness premise after the proposed unused-allocation removal; it changes no package width or pin and proves no hardness.
- The owner approved the [Fiat–Shamir assumption boundary](../../docs/reviews/nightstream-fprime-requirements/FIAT_SHAMIR_MODEL.md) on 2026-09-11 UTC. This approves no numerical model instance. `Lifecycle.Nifs.WideFiatShamir.FiatShamirModel` ([source:108](NightstreamFPrime/Lifecycle/Nifs/WideFiatShamir.lean#L108)) assumes only the transfer inequality to the typed interactive game, for the selected wide key. `FiatShamirTransfer` keeps only the key-independent composition that consumes this inequality. Its real event requires actual verifier acceptance **and witnesses for the exact 16 children**. The shared context marginal does not construct an adversary translation. Applicable classical transfer, useful `g`/`deltaFS`, and total permutation-query accounting, including replays, must be supplied externally. This is an additional FS/SuperNeo contract, not ordinary collision resistance or an established instantiation of the overwrite-duplex theorem for the additive transcript.
- Finite sampler density and abort results concern the explicit independent-field comparison. They do not assign that law to Poseidon2. `Lifecycle.Nifs.VerifierErrorBudget.any_test_or_sampler_abort_le` ([source:62](NightstreamFPrime/Lifecycle/Nifs/VerifierErrorBudget.lean#L62)) unions the two specified events over caller-supplied `n` in one trace law. Actual per-call bounds, which can follow from bounds conditional on the preceding history, remain required; independence is not required. This bound excludes FS/hash/MSIS and extraction loss. It is not a complete security bound.
- Verifier-owned identity and setup checks remain necessary at the runtime trust boundary. The retained optimized Lean/Rust examples establish only their recorded inputs and rejection cases. This note adds no arbitrary-input Rust proof, backend claim, PaperExact authorization, or efficient probabilistic full-history extraction result.
- The selected native cache now derives its rows through `Poseidon2HashChainV1Package.build_superneo_cache`. Exact reservations control memory; every actual row still passes order, coefficient and coverage checks. The actual base witness and logical transport matched all 14 × 54 retained Lean matrix outputs in 197.77 seconds of test time, with 26.90 GiB peak RSS. This is one full-profile matrix comparison. The normal selected PiCCS prover now passes from the actual base witness and fixed-key commitment: all round messages, transcript states and complete outputs match retained assertions, and the normal optimized verifier accepts (231.52 seconds, 26.86 GiB peak RSS). The complete selected-base C → R → D path now passes through capped actual-witness stages: computed D openings, normal NIFS acceptance, all 43 NIFS mutations, and strict comparison with an independently computed Lean C/R/D result, including all 55 D mutations. The retained regression checks that result and its 945,983-byte proof encoding against the current pinned package. The unused-allocation mutation repair is recorded separately in `NIFS_UNUSED_ALLOCATIONS.md`. See [native evidence](../../docs/reviews/nightstream-fprime-requirements/NATIVE_NIFS_EVIDENCE.md) for exact sources, inputs and logs. Full-profile later running inputs, universal Rust semantics, the production proof backend and performance remain separate.

### Selected wide sampler

The owner approved the map and transcript change on 2026-09-24 and, on
2026-09-25, confirmed that the Fiat–Shamir boundary extends to the wide key
(recorded with a hash in the [FS boundary note](../../docs/reviews/nightstream-fprime-requirements/FIAT_SHAMIR_MODEL.md)). The selection
keeps Poseidon2 and the Nightstream Goldilocks profile `b=2`, `k_rho=16`,
`B=65536`. `Export.Entrypoint` uses `Wide.Emitter` for the selected package,
and the Rust native sampler reads the same joint four-field block.

- `Spec.Folding.Nifs.NonInteractive.PiRlcWideSampler.TranscriptHistory.queryAt_answer`
  proves that replaying a normalized block history from any seed reproduces
  the additive Poseidon2 domain entry, four rate lanes and digest advance.
  `Lifecycle.Nifs.WideSamplerSecurity.response_history` connects those values
  to the wide verification key. No theorem identifies the verifier's PiCCS
  output state with the replay of the complete transcript history.
- `WideSamplerSecurity.no_sampler_abort` proves totality for every initial
  state. `any_test_le` unions only PiCCS test errors in the supplied trace law.
  It includes no old sampler-shortfall event.
- `Lifecycle.Nifs.WideFiatShamir.returned_source_bound_with_adaptive_msis`
  consumes the same parametric classical transfer boundary for the wide key.
  The real event requires that key's actual acceptance and witnesses for its
  exact sixteen returned children. The source bound keeps `g Q p_real`,
  `deltaFS Q`, interactive extraction loss, PiCCS test loss and the actual
  same-key MSIS event separate.
- `WideSamplerSecurity.adaptive_bias_bound` is the general hybrid bound `q*δ`
  for a bounded test of a cached block-oracle run, including raw lanes and
  repeated queries. Its test runs the concrete verifier, which derives its
  challenges with concrete Poseidon2; the oracle reaches the test only through
  the supplied decoder, so the balanced run does not give the verifier uniform
  challenges. `concrete_bias_bound` adds the supplied `modelError`. Neither
  result is consumed by the extraction chain: the wide-versus-uniform challenge
  difference stays inside the FS term `deltaFS`. Here `δ < 2^-132`; `q` counts every block call, while
  `Q` in the FS boundary counts permutation calls including replays. Neither
  count is inferred from the other. Balanced raw replies are not asserted to
  be a Fiat–Shamir extractor, and a general `g` is not assumed to preserve
  additive error. The fresh-batch result alone allows `17*L*δ`; its extraction
  term `17*L/|C|` remains separate.
- `Export.Stage1.Wide.ContextBinding.step_or_collision` binds arbitrary
  accepted candidate rows and their actual public projection to the context
  in a verifier-checked state hash, or exhibits the existing state-hash
  collision event. No honest assignment or context-equality premise is used.
- `Export.Stage1.Wide.PackageAuthority.matrix_exact`
  ([source](NightstreamFPrime/Export/Stage1/Wide/PackageAuthority.lean)) proves
  that the matrix program in a successfully prepared package has the exact
  structural-plan rows, using that package's actual physical source archive.
  The constructor supplies source custody; no source-row equality is assumed.
- `Export.Stage1.Wide.AssignmentTransportCorrectness.canonical_execute_eq_assignment`
  and `canonical_carrier_eq`
  ([source](NightstreamFPrime/Export/Stage1/Wide/AssignmentTransportCorrectness.lean))
  identify execution of the emitted schema-4 transport with the direct wide
  assignment and its complete padded carrier. These theorems require the
  stated PiCCS/PiRLC physical conditions and completed range values. The
  package completeness theorem below constructs those conditions.
- `Export.Stage1.Wide.PackageCompleteness.complete`
  ([source](NightstreamFPrime/Export/Stage1/Wide/PackageCompleteness.lean))
  constructs physical values that the actual prepared transport accepts. The
  resulting assignment satisfies the complete structural plan, has strict
  carrier norm below two, and carries the exact public output digest and
  application advice. Its premises are successful package preparation, the
  typed semantic step, well-formed prior and next preimages, the fresh public
  link, actual wide-key NIFS acceptance, agreement of the recursive result,
  and four-word advice. It assumes no physical rows, transport equality or
  source-read equality.
- `Export.Stage1.Wide.SetupBinding.descriptor_recomputed` and
  `step_or_collision`
  ([source](NightstreamFPrime/Export/Stage1/Wide/SetupBinding.lean)) bind the
  relation and exact application child from the sealed package, the wide
  transcript schedule, and the indexed Ajtai setup with the approved seed
  and dimensions 22 × 2,543,368. The setup dimensions follow from the
  137,341,872-coordinate carrier. The context and verification-key
  serializers both use the wide schedule. The key uses the same sampler as
  `WideSamplerSecurity`; the explicit Fiat–Shamir and block-oracle model
  boundaries above remain in force. `shortKernel_to_approvedMsis` reduces the
  smaller same-seed key to the existing approved MSIS instance by zero
  extension.

All three selected costs are recorded in
[the integration report](../../tools/recursive-constraint-minimizer/experiments/wide-sampler-integration.md).
The selected layout has 3,248,956 rows, 137,341,872 committed coordinates,
and 2,607,606,765 normalized matrix entries. The complete independent Rust
matrix comparison and native binding parity pass. The current-package
golden conformance run at `48c8e0b9` passes for both folds and the terminal
checks: fresh Lean C/R/D verification, complete proof bytes and caller inputs
match native, with the stated mutation rejections. C messages and child claims
are native inputs; the physical witness is not compared. See
[the record](../../docs/reviews/nightstream-fprime-requirements/golden-conformance-wide/README.md).
No numerical security level or performance claim follows from these counts.
