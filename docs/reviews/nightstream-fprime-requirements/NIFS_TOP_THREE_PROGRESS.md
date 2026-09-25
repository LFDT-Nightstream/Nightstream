# NIFS interactive proof review

The incoming proofs are useful at their stated conditional interactive scope.
The review found no source-level proof regression. The combined
source passes the full Lean library, axiom and site export checks.

Reviewed incoming branch: `nico/nifs-interactive-proofs` at
`f256543cbce3e560f9464879645aad4cadda9a50`. Target base:
`nico/f-prime-constraints-cuda-formal` at
`861a5b8c272e6edb204623524219caf49027b303`.

## What the merge adds

Paths below are relative to
`formal/nightstream-fprime/NightstreamFPrime/`.

| Local obligation | Direct Lean evidence | Scope |
|---|---|---|
| Exact selected relation and matrices | `Lifecycle/NifsProfile.lean`, `selected_relation`, `selected_matrix` | Equality to the relation selected by `ProductionKey`; no digest substitute. |
| Common shape, arity and setup | `Lifecycle/NifsProfile.lean`, `selected_shape`, `selected_arity`, `shared_setup`, `phases_preserve_relation` | The same relation and setup connect all three phases. |
| Strong interactive prefix | `Lifecycle/Nifs/StrongExtraction.lean`, `probability_and_expected_work`; `Spec/Folding/Nifs/PaperStrongCompleteness.lean`, `exists_honest_piCcs_prover` | One causal honest prover, fixed commitment projection, actual source return and conditional expected work. |
| Exact fork alignment | `Spec/Folding/Nifs/PaperAlignedExtraction.lean`, `positive_return_implies_alignedFork` | Positive returned endpoint yields the fork under the stated observed probe and sampler equalities. |
| Weak interactive extraction | `Lifecycle/Nifs/WeakExtraction.lean`, `weak_relaxed_success_bound`; `Lifecycle/Nifs/InteractiveCompleteness.lean`, `exists_honest_execution` | Actual suffix results and checked child witnesses yield the relaxed source relation, with loss `17 / card(Challenge)`. |
| Agreement of two extractions | `Lifecycle/Nifs/InteractiveAgreement.lean`, `disagreement_le_bindingProbability`; `Spec/Folding/PiRLC/PaperForkBinding.lean`, `collisionAt` | Unequal returned values produce collision data from actual bounded response differences. |
| Probability and work of the same composition | `Lifecycle/Nifs/SupportedExtraction.lean`, `probability_and_expected_work` | The actual checked prefix, captured continuation and returned source values share one probability and work argument. |

The new modules separate causal execution, probability laws, output decoding,
work accounting and selected-key composition. The parents use these interfaces.
No phase gadget, physical assembler, transcript sampler or relation predicate
is replaced. The largest new Lean module has 474 lines; the existing averaging
module grows to 935 lines. The added helpers establish different proof steps,
including receipt equality, support transport, summability and returned-value
correctness. File count is not closure evidence.

## Exact result and remaining premises

The final theorem joins four facts: the costed prefix returns the exact checked
receipt; the returned source has the stated success probability; the global
work sum exists; and expected work satisfies the displayed polynomial bound.
The extraction loss is

`17 / card(Challenge) + sqrt(binding-event probability + PiCCS test error)`.

The review checked this against the local SuperNeo v1_1 Definitions 16 and 17,
Theorem 12 and Appendix B.1, B.3 and B.4. In particular, the coordinate loss is
`17 / card(Challenge)`, not a denominator raised to the seventeenth power.
Each response has its own strict parent bound. Their differences have the
strict `2 * B` bounds used by the collision argument.

| Claim that remains open | Exact missing connection | Effect |
|---|---|---|
| Negligible binding failure for the selected public-seed setup | The proof bounds disagreement by a collision event. It does not supply an efficient computational reduction with a numerical or negligible bound for that event. | Full security and production remain open. |
| Security of the selected Poseidon2 transcript | The probability law uses independent uniform interactive coins. It does not prove Fiat-Shamir transfer to the bounded fail-closed sampler. The alignment theorem retains explicit observed equalities. | Noninteractive security and production remain open. |
| A fully instantiated execution-cost claim | Low-norm invertibility, call/check/access correctness, primitive bounds and global base-work summability/polynomial bounds remain explicit premises. The theorem derives extractor work from them. It is not a measurement of native runtime or an instantiated uniform security-family theorem. | Keep the result conditional until the relevant implementation and security premises are supplied. |
| Complete Stage 1 acceptance | These additions do not prove HyperNova history extraction, select the full production context, or close native conformance. | Full Stage 1 and Production-closed status remain open. |

Only reachable continuations with positive probability need finite call
moments. The proved abort extension handles unreachable states. Finite suffix
tables couple the probability experiments; the extractor does not execute
those tables. The fork proof uses short differences of returned responses,
not uniqueness of arbitrary ambient openings.

The profile remains Goldilocks, `b = 2`, `k_rho = 16`, `B = 65536`, one fresh
source, 16 running sources, 17 inputs in `K + k` order, 16 PiDEC children,
14 matrices and 28 PiCCS rounds. The current joint-domain bound remains
`2^28`. This merge does not change the package, footprint or identity.

## Requirements map and prior fixes

The map's Proof and Link axes describe local facts and their local consuming
contracts. The eight NIFS records retain their proved/connected status with
explicit conditional scope. This grants no new Compiler-closed,
Conformance-closed or Production-closed phase status.

The map was merged by node. All prior nodes outside these eight records and
all prior update records were preserved. This retains the terminal-opening
proofs, the PiRLC transcript-state fix, current package pins and the open
full-assignment mutation failure. The existing failure concerns unused logical
blocks 12 (`piCcsPayload`), 13 (`runningRoundC0`) and 14 (`runningRoundC1`).
It was not rerun or closed by this proof-only merge.

No new NIFS closure target was registered in lean-graph. Its current registered
pilot/PiCCS and terminal checks cannot grant approval for these new security
results. This review and the following build results are local diagnostics;
they do not create authenticated review or conformance approvals.

## Validation and delivery

The combined-source boundary gate passed. A fresh project build completed
`NightstreamFPrime` (3,786 jobs, 800 seconds) and `NightstreamFPrimeTests`
(3,822 jobs, 43 seconds). The guarded validation took 880.745 seconds in total,
within the owner policy's 1,500-second cap. Only third-party dependency caches
were copied; `lean-toolchain` and `lake-manifest.json` matched exactly.

The NIFS audit checked all 118 listed declarations and accepted only
`propext`, `Classical.choice` and `Quot.sound`. The complete test library also
checked the retained Stage 1 security and terminal exports. The recorded source
manifest still matches the files that were built.

The site build produced 454 nodes, 11 Markdown files, HTML, JSON and the
download ZIP. All seven export tests and the JavaScript syntax check passed.
The reviewed reference check confirmed the eight NIFS records against the
combined code commit and confirmed preservation of every earlier node outside
that set and every earlier update record. Non-Lean checks used the shared
300-second guard.

The first added reference check failed because it treated a descriptive Rust
source label as a literal declaration name. The corrected check validates
source locations and literal Lean declaration names. It passed; both attempts
are retained. The seven export tests had already passed and were not rerun.

The map's code reference is the signed merge commit
`77713c1e711c7b1bca0a1b0e5941c2ac94f0656e`. The following documentation commit
only records this result and its evidence.

[NIFS_INTERACTIVE_EVIDENCE.zip](NIFS_INTERACTIVE_EVIDENCE.zip) contains the
source manifest, full build/audit log, site check logs and review findings.
The source manifest binds the checked Lean files to their exact bytes.
This local source update makes no claim that the live site was published.
The review does not adopt the incoming report's earlier timings or publication
record as evidence for this combined tree. No proof backend is used.
