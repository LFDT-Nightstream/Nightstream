# Independent NIFS conformance review

Reviewer: Codex Worker 3, task /root/conformance_review. The task coordinator
assigned this agent as the independent SuperNeo v1.1 reviewer. This agent did
not implement the reviewed source or produce the retained native execution.

Reviewed source cut: **773f3d0f29209b33e2325538d5f258f541569c25**, branch
nico/nifs-proof-links, at /home/nicoarq/develop/Nightstream-nifs-proof-links.
All source judgments below apply to that commit. Later commits and concurrent
stored-witness work are outside this review.

The review contract is to map every C/R/D verifier conjunct to its semantic,
circuit and proof owner, and to check the scope of the selected nonzero
Lean/optimized Rust chain. The evidence must identify the same relation, key,
profile, prior input, proof, complete phase results and final output. The
review must preserve the distinction between this task's selected execution
and the broader per-phase completion standard in
[FPRIME_STAGE1_GOAL.md](/home/nicoarq/develop/Nightstream-nifs-proof-links/FPRIME_STAGE1_GOAL.md:443).

The fixed input profile is the Nightstream Goldilocks profile with b = 2,
k_rho = 16, B = 65536, one fresh source, 16 running sources, 17 sources in
fresh-then-running order, 16 children, 14 matrices, 28 rounds, ring degree 54,
Ajtai rank 22 and Poseidon2 protocol binding. It is not the paper's reference
k_rho = 14 profile.

I find no mismatch between the inspected SuperNeo v1.1 verifier formulas and
the selected Lean C/R/D predicates and circuit owners. The retained records
substantiate the stated nonzero Lean/optimized NIFS execution and the linked
matrix and raw-assignment checks. I independently checked the archive bytes,
complete output links and canonical proof encoding. This is affirmative
independent review evidence for that scope.

**This review does not declare a phase Conformance-closed or
Production-closed.** Same-input three-way R/D evidence remains open. A
previously failed logical mutation diagnostic is retained below with its
exact authority scope. The security results retain their stated
premises. No build, proof backend, PaperExact action, fixture generator or
native test was run for this review. No owner approval or approved-checker
record is inferred from a successful local diagnostic.

The normative verifier source is
[SuperNeo v1.1 §7](/home/nicoarq/develop/Nightstream/docs/superneo-paper-v1_1/07_superneo_folding_scheme_for_ccs.md:1),
with the field, norm and sumcheck definitions in §4, the coefficient and
evaluation maps in §5, and
[Appendix B.1–B.4](/home/nicoarq/develop/Nightstream/docs/superneo-paper-v1_1/11_appendix_B_deferred_theorems_and_proofs.md:1).
The prior-state boundary is the one in
[HyperNova Construction 2](/home/nicoarq/develop/Nightstream/docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md:1).
The architecture contract is
[FPRIME_LEAN_ARCHITECTURE_SPEC.md](/home/nicoarq/develop/Nightstream-nifs-proof-links/FPRIME_LEAN_ARCHITECTURE_SPEC.md:1).
No frozen Nightstream Lean proof file was used.

The PiCCS coverage map follows. C denotes the namespace
NightstreamFPrime.Lifecycle.PiCCS.v1_1. Each listed leaf exports its circuit,
soundness and completeness under that leaf namespace. The last column names
the theorem that connects its result to the parent check.

| Exact verifier obligation | Lean predicate or value | Circuit owner and parent theorem |
|---|---|---|
| §7.3 input: one CCS source and 16 CE sources share the selected structure; the prior point and both prior evaluation families remain distinct. | PiCCS.Coverage.input_eval_K and input_eval_A; the typed Running/Fresh interfaces; StateBinding.SpecHolds. | [C.StatementBinding](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/StatementBinding.lean:72), including its StateBinding child; spec_implies_keyStatement. |
| The concrete noninteractive statement state binds the authenticated prior digest and fresh commitment/public input. This is the Nightstream instantiation of the public statement. | ProductionKey.publicInputBlocks, absorbPublicInput and key_publicInputState_eq. | [C.StatementAbsorption](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/StatementAbsorption.lean:990); spec_implies_keyInitialState. Authentication of the digest is a separate parent obligation below. |
| §7.3 step 1: the verifier owns alpha and gamma. The actual deterministic implementation derives them from the bound Poseidon2 state. | Transcript.deriveFromState; PiCCS.Coverage.transcript; C.ChallengeDerivation.SpecHolds. | [C.ChallengeDerivation](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/ChallengeDerivation.lean:761); spec_implies_derivePreSumcheck and spec_implies_keyExecution_challenges. These theorems prove derivation, not uniform randomness. |
| §7.3 step 2 and §4 Definition 11: each round message precedes its verifier challenge; all 28 rounds have the declared coefficient width. | RoundTranscript.SpecHolds; typed FixedPolynomial messages; FiatShamir.deriveRoundsFrom. | [C.RoundTranscript](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/RoundTranscript.lean:644); spec_implies_keyExecution_rounds. The shared round interface supplies the sumcheck child with those exact challenges. |
| §7.3 step 2: T is the Pad coefficient sum plus gamma^(k*d) times the matrix coefficient sum, with the stated exponent order. | ProtocolPolynomial.VerifierInput.initial; FinalIdentity.targetCoefficientList; initial_eq_eval_K_add_shifted_eval_A. | [C.InitialClaim](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/InitialClaim.lean:68); spec_implies_keyInitial. The Horner list contains all k*d Pad coefficients before all k*t*d matrix coefficients. |
| §4 Definition 11: p_j(0) + p_j(1) equals the current claim; the next claim is p_j(r_j); the last claim equals the terminal value. | SumCheck.Finite.FixedPhase.Chain, used by piCcsCheck. Degree at most 9 is supplied by the typed 10-coefficient messages. | [C.SumcheckChain](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/SumcheckChain.lean:76); spec_implies_keyChain and keyChain_implies_spec_and_terminal. |
| §7.3 step 4: E_K = eq(r',r) times the prior-source Pad coefficient sum. | ProtocolPolynomial.padAtMessage; FinalIdentity.padAtMessage_eq_pointEquality_mul_horner; the semantic EvalK.Holds corresponds to B.2 equation (9). | [C.EvalKTerminal](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/EvalKTerminal.lean:83); spec_implies_keyPadAtMessage. |
| §7.3 step 4: E_A = eq(r',r) times the prior-source, matrix, coefficient sum. Its local exponents exclude the outer k*d shift. | ProtocolPolynomial.matrixAtMessage; FinalIdentity.matrixAtMessage_eq_pointEquality_mul_horner; EvalA.Holds corresponds to B.2 equation (10). | [C.EvalATerminal](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/EvalATerminal.lean:84); spec_implies_keyMatrixAtMessage. The matrix index has exactly 14 entries and contains no Pad entry. |
| §7.3 step 4: F evaluates the selected CCS polynomial at the constant terms of the fresh matrix evaluations. | ProtocolPolynomial.ccsAtMessage; the selected 74-term ProductionRelation.polynomial; Phi81CoefficientKernel.constant. | [C.CcsTerminal](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/CcsTerminal.lean:87); spec_implies_keyCcsAtMessage. |
| §7.3 step 4: N sums all 17 signed-binary norm residuals, gamma^i times (a+1)*a*(a-1), where a is the Pad constant term. | ProtocolPolynomial.normAtMessage and strictNormResidual; semantic norm residuals correspond to B.2 equation (8). | [C.NormTerminal](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/NormTerminal.lean:68); spec_implies_keyNormAtMessage. |
| §7.3 step 4: v = E_K + gamma^(k*d)*E_A + gamma^(k*d*(t+1))*eq(r',alpha)*(F + gamma^K*N). Here k*d = 864, k*d*(t+1) = 12960 and K = 1. | ProtocolPolynomial.terminalFromMessage; FinalIdentity.Holds and terminal_eq_eval_K_add_shifted_eval_A_add_constraints. | [C.FinalIdentity](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/FinalIdentity.lean:173); spec_implies_keyTerminal. The final two base-field equalities use the same terminal wire as the sumcheck chain. |
| §7.3 steps 3 and 5: return all 17 original commitments and public inputs at r', with every supplied Pad and matrix coefficient; bind the full output before R. | FullOutputCoordinates.FullOutput; Key.piCcsCertificate; PiCCS.Coverage.output_eval_K and output_eval_A; ProductionKey.absorbFullOutput. | [C.OutputBinding](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/OutputBinding.lean:210); key_output_eval_K, key_output_eval_A and spec_implies_keyOutgoingState. |

The exact predicate is
[PiCCS.Accepted](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/PiCCS/Accepted.lean:46),
an abbreviation for the selected piCcsCheck result. It is not an alternate
relation. accepted_iff_coverage equates the audit view to that check.
C.Formal.spec_implies_phaseHolds constructs every Coverage field from the
shared interfaces, derives the complete outgoing state and calls that
equivalence. This includes the fresh/source ordering and terminal-to-chain
wiring. The algebraic validity of all input openings is not a deterministic
consequence of one accepted sumcheck transcript; the strong reduction owns
that separate conclusion under its stated probability and witness premises.

The PiRLC map follows. R denotes
NightstreamFPrime.Lifecycle.PiRLC.v1_1. The complete semantic record is
[PiRLC.Accepted](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/PiRLC.lean:126).
It contains all Equations fields and challengesValid.

| Exact verifier obligation | Lean conjunct | Circuit owner and parent theorem |
|---|---|---|
| §7.4 input: all 17 inputs are CE(b). | Equations.inputFresh. | [R.InputBinding](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/InputBinding.lean:125); parentCoverage and Semantics.inputBinding. This checks the stage; it does not prove private opening validity. |
| §7.4 input/output: all structures and points are the same. | Equations.sameStructure and samePoint. | R.InputBinding and [R.OutputBinding](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/OutputBinding.lean:69); parentCoverage and Semantics.inputBinding. Both use the shared relation and point. |
| §7.4 step 1: each rho belongs to the strong set. The concrete implementation must derive it in source order and fail on shortfall. | Accepted.challengesValid; Semantics.PhaseHolds.sampler. | [R.SamplerChain](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/SamplerChain.lean:560), its Sampler, First54 and digest children; sampleRingChallenge_eq, Semantics.challengesValid and PhaseHolds.response. |
| §7.4 step 1: c = sum rho_i*c_i, with every ordered input included. | Equations.commitmentEquation. | [R.CommitmentCombination](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/CommitmentCombination.lean:115); parentCoverage and Semantics.commitmentEquation. |
| §7.4 step 1: pack(x) = sum rho_i*pack(x_i), followed by unpacking. | Equations.publicInputEquation. | [R.PublicInputCombination](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/PublicInputCombination.lean:228); parentCoverage and Semantics.publicInputEquation. All five public rings are retained. |
| §7.4 step 1: y = sum rho_i*y_i in the separate Pad family. | The pad projection of Equations.evaluationEquation. | [R.EvalKCombination](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/EvalKCombination.lean:1); parentCoverage and Semantics.outputFamily_eq_combine. |
| §7.4 step 1: for every one of 14 matrices, y_j = sum rho_i*y_i,j. | The matrix projection of Equations.evaluationEquation. | [R.EvalACombination](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/EvalACombination.lean:86); parentCoverage and Semantics.evaluationEquation. |
| §7.4 step 3: return the exact combined CE(B) claim with the shared structure and point. The next phase receives the full sampler endpoint. | Equations.outputCombined; Semantics.PhaseHolds.outgoingState. | R.OutputBinding, Semantics.outputCombined, output_eq_combinedOutput and PhaseHolds.outgoingState. |

The four arithmetic families use the same
[CombinationStep.circuit](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/CombinationStep.lean:405).
Its equation is next = prior + rho*value in the Phi81 ring. cellCount = 1
handles base-field ring coefficients; cellCount = 2 handles extension
coefficients. CombinationFamily proves the ordered 17-source composition.
Semantics.spec_implies_equations fills all seven Equations fields;
spec_implies_phaseHolds derives membership from sampler replay.

[Lifecycle.Transcript.PiRlcSampler](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Transcript.lean:132)
uses two little-endian 16-bit candidates from each of four lanes, eight
complete digest blocks, and the first 54 accepted candidates from the
64-candidate source. Those constants come from the selected protocol schedule.
Each source starts with the domain separator [4, source]. Shortfall returns
none; membership is proved only on success. The outgoing state follows every
scheduled block. Membership and exact replay do not establish the paper's
uniform independent coin law.

The PiDEC map follows. D denotes
NightstreamFPrime.Lifecycle.PiDEC.v1_1. Its exact semantic verifier is
[PiDEC.PaperVerifier.Accepted](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/PiDEC/PaperVerifier.lean:226).

| Exact verifier obligation | Lean conjunct or computed field | Circuit owner and parent theorem |
|---|---|---|
| §7.5 input: the parent is at the combined norm stage. | Accepted.parentCombined. | [D.InputBinding](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/InputBinding.lean:153); soundness, completeness and D.Semantics.accepted. |
| §7.5 typed messages: the parent and every one of 16 messages have the complete evaluation family. | Accepted.parentEvaluationSize and messageEvaluationSize. | D.InputBinding.accepted_parentEvaluationSize and accepted_messageEvaluationSize. The concrete array has one EvaluationFamily record, which contains one Pad family and 14 matrix families. |
| §7.5 step 2: reject when any centered public coefficient has magnitude at least B. | Accepted.parentBounded; PublicInputSplit.checked. | [D.PublicInputSplit](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/PublicInputSplit.lean:546), with SignedSplitScalar; parentBounded and SignedSplitScalar.spec_parentBounded. |
| §7.5 step 2 and §4 decomposition: compute the 16 signed-binary public digits with magnitude less than 2 and exact radix recomposition. | PublicInputSplit.SpecHolds; UniformSignedDigits.Accepted; OutputAccepted.outputComputed. | [D.SignedSplitScalar](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/SignedSplitScalar.lean:517); spec_digits_eq_splitScalar, spec_uses_bounded_branch; PublicInputSplit.children_eq_splitPublicInput. |
| §7.5 step 2: c = sum 2^i*c_i. | Accepted.commitmentEquation. | [D.CommitmentRecomposition](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/CommitmentRecomposition.lean:143); parentCoverage, soundness and completeness. |
| §7.5 step 2: y = sum 2^i*y_i for Pad. | The pad projection of Accepted.evaluationEquation. | [D.EvalKRecomposition](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/EvalKRecomposition.lean:106); parentCoverage and D.Semantics.evaluationFamily_eq. |
| §7.5 step 2: y_j = sum 2^i*y_i,j for each matrix. | The matrix projection of Accepted.evaluationEquation. | [D.EvalARecomposition](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/EvalARecomposition.lean:97); parentCoverage and D.Semantics.accepted. |
| §7.5 step 3: return exactly 16 children with the computed x_i, copied structure and point, supplied c_i/y_i/y_i,j, and fresh stage. | PaperVerifier.children; OutputAccepted.outputComputed; D.Semantics.PhaseHolds. | [D.OutputBinding](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/OutputBinding.lean:1); D.Semantics.spec_implies_phaseHolds and phaseHolds_implies_spec. The phase has no challenge and preserves the R endpoint. |

SignedSplitScalar constrains a Boolean sign, every digit as zero or that
common sign, and the weighted sum. UniformSignedDigits proves both the strict
parent bound and equality with the canonical split. A total semantic fallback
definition can still exist outside the accepted domain; spec_uses_bounded_branch
proves that accepted rows cannot use it. PaperAlgebra.piDecDecision uses the
computable PaperVerifier.acceptedDecision. The old
Classical.propDecidable debt is not present at this selected decision boundary.

The generic one-element evaluation array used by R and D is not a compressed
matrix list. Its element is a record with pad and matrix fields.
[PaperAlgebra.evaluationFamily](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PaperAlgebra.lean:156)
computes Pad from cubeLayout.paddedIdentityEntry and each genuine matrix from
the selected matrix source. combineEvaluationFamily and recomposeEvaluationFamily
act on these fields separately. The canonical Lean path and native C/R/D
verifier path reviewed here contain no Pad-as-matrix-zero substitution.
Legacy native circuit modules elsewhere in the repository are outside the
selected replay; this review does not establish their removal or the exclusive
production lifecycle.

The compiler and consumer links are as follows. These are source-checked
theorem connections with retained Lean audit evidence, not newly executed
kernel checks by this reviewer.

| Link | Exact owner and proof |
|---|---|
| PiCCS leaf composition and exact phase coverage | C.Formal.soundness and spec_implies_phaseHolds in [Formal](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/Formal.lean:1193); Formal.circuit and completeness in [Completeness](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiCCS/v1_1/Completeness.lean:251). The parent uses opaque child contracts and shared expressions. |
| PiRLC and PiDEC composition | Each phase's Formal.soundness, Formal.completeness and Formal.circuit; [R.Semantics.spec_implies_phaseHolds](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiRLC/v1_1/Semantics.lean:554); [D.Semantics.spec_implies_phaseHolds](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/PiDEC/v1_1/Semantics.lean:364). D also proves phaseHolds_implies_spec for the exact computed outputs. |
| Physical preservation | Layout.PiCCS.v1_1.physical_implies_phaseHolds and physical_complete in [C preservation](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Layout/PiCCS/v1_1/Preservation.lean:53); the corresponding physical_implies_phaseHolds, physical_complete and physical_complete_production in [R preservation](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Layout/PiRLC/v1_1/Preservation.lean:57) and [D preservation](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Layout/PiDEC/v1_1/Preservation.lean:53). |
| Canonical package rows | [Package](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Package.lean:432).circuitPackage_implies_piCcsPhaseHolds, circuitPackage_implies_piRlcPhaseHolds and circuitPackage_implies_piDecPhaseHolds; PackageCompleteness.complete_piCcsRows and complete_piRlcRows; PiRLCPackageCompleteness.completePackets; PiDECPackageCompleteness.completeRows and completePackageRows. |
| Same C result, R input, R parent and D output | [AccumulatorSemantics](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Layout/Stage1/AccumulatorSemantics.lean:1101).phases_imply_holds_of_wiring and phases_imply_holds. Its named intermediate theorems equate the C output to R inputs, sampled challenges to the key challenges, R output to the key parent, D parent to R output, and the returned NIFS output to the accumulator. AccumulatorPackage.circuitPackage_implies_accumulatorHolds consumes the package rows. |
| Arbitrary accepted opening and authenticated prior input | [ActualTerminalSecurity.terminal_implies_nifsOrBaseOrCollision](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/ActualTerminalSecurity.lean:30) consumes ActualContextSecurity.terminal_implies_matchingStepOrCollision. It derives the selected context, full prior encHash equation, exact absorbed prior digest and exact advertised NIFS output, or the named state-hash collision. It does not take canonical-assignment or output-match premises. The base branch performs no NIFS call. |
| Executed C acceptance to the strong probe | [PiCCSInputCheck.execute_accepted_iff](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PiCCSInputCheck.lean:539) proves equality to FixedWidthAccepted on the exact messages, full output and derived challenges. PiRLCInputCheck.sampled_fixedWidthAccepted consumes it on a successful actual handoff. |
| Executed R parent and weak-success endpoint | [PiRLCParent.inputBatch_eq_probe](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PiRLCParent.lean:96) identifies the entire ordered batch. computedParent_correct proves that the canonical traces always return the exact combined claim and full sampler state. checked_children_imply_rlc_success consumes the accepted D result and the valid openings of its actual children. |
| D knowledge reconstruction | PiDECInputCheck.accepted_reduces_knowledge and ActualTerminalSecurity.terminal_implies_parentOrBaseOrCollision consume the existing B.4 recomposition argument. Parent opening validity is a conclusion. The local checker theorem still needs child opening validity; the terminal theorem obtains it from Terminal.HoldsFor. |
| Selected public extraction checker | [SupportedExtraction.publicCheck_correct](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Lifecycle/Nifs/SupportedExtraction.lean:57) proves the concrete checker equivalent to the selected predicate for every probe, including malformed messages. It does not discharge the witness-operation, ambient-checker or work contracts. |

The selected package geometry is proved in
[Poseidon2HashChainV1Package](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/Poseidon2HashChainV1Package.lean:140):
29,225,729 physical rows, 29,344,425 physical columns including the constant
and 278 public columns, 6,377,559 active logical rows and 254,260,583 logical
columns. The ring carrier has 254,260,620 coordinates, including 37 alignment
zeros. plan_fixedPoint and jointDomain_le_twoPow28 prove the selected recursive
plan connection and domain bound. These size theorems do not prove a security
transfer or a production backend result.

The selected structural identifier is
[12850397830186002711, 3288783999059851307, 4378874948253040911,
12528041036879069119]. The package identifier in the retained binding is
[6335883518996883063, 12495920096635129097, 13503539403510063271,
2659339960413338175]. The committed Git LFS record identifies a 128,466,966-byte
candidate with SHA-256
fb865ee053060ac5bb52dcc86ea8289c10425b38c9ccf21c11f2598c26e4c5c4.
SHA-256 in this review identifies files only. Protocol binding remains Poseidon2.

The retained evidence was inspected as follows.

| Evidence | Independently substantiated scope |
|---|---|
| [NIFS_COMPLETE_REPLAY_EVIDENCE.zip](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/NIFS_COMPLETE_REPLAY_EVIDENCE.zip) | At source 9dcb4e7b9d7b16ac15e7f19c2ab4a4db007708f5, child-nifs-driver calls the optimized C prover, native R/D provers and nifs::verify. The record compares all final children, the returned R parent cache, the full eight-word transcript state and empty verifier witness vector. The source and log show 43 NIFS and 55 D rejection cases. |
| [NIFS_DEC_AND_FINAL_OUTPUT_EVIDENCE.zip](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/NIFS_DEC_AND_FINAL_OUTPUT_EVIDENCE.zip) | Executable Lean accepts the exact C/R/D envelope. Native D compares all 17 result fields and every private digit of the actual R witness. The independent selected-key check covers all 19,008 child commitment coefficients; the independent Pad/14-matrix check covers all 12,960 extension coefficients. Lean also rejects 34 public mutations, 10 encoding errors, an unbounded parent and a rejected C prefix. |
| [NIFS_RUNNING_AND_NATIVE_RLC_EVIDENCE.zip](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/NIFS_RUNNING_AND_NATIVE_RLC_EVIDENCE.zip) | Independent opening checks cover the 16 running children at both the prior and new C points, their commitments, and the fresh commitment and CCS rows. Native R compares all 254,260,620 private coefficients to the recorded integer combination. Its maximum magnitude is 113. Changing the final private coefficient fails the comparison. |
| [NIFS_MATRIX_CALLER_EVIDENCE_LINKS.json](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/NIFS_MATRIX_CALLER_EVIDENCE_LINKS.json) and [conformance-fixes-evidence.zip](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/conformance-fixes-evidence.zip) | The candidate physical record compares final A/B/C entries. The logical record checks all 14 matrix families. The recursive record checks every physical row, every transported logical coordinate, alignment zeros and all 6,377,559 active logical rows for the caller input reused here. The original records state development diagnostic only. Their failed checks remain failed. |
| [NIFS_RLC_PARENT_EVIDENCE.zip](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/NIFS_RLC_PARENT_EVIDENCE.zip) | The checked R-parent equality, full-state and total-return proofs replace the failed earlier draft. These are deterministic proofs and conditional child-opening reconstruction, not a sampler distribution theorem. |
| [NIFS_EXECUTED_PROBE_EVIDENCE.zip](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/NIFS_EXECUTED_PROBE_EVIDENCE.zip) and [NIFS_SELECTED_CHECKER_EVIDENCE.zip](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/NIFS_SELECTED_CHECKER_EVIDENCE.zip) | The first archive's four source files match the reviewed cut. The second archive's seven source files match the reviewed cut. Their successful explicit audits cover the executed C probe, complete R input equality and selected public checker. Only propext, Classical.choice and Quot.sound are reported. |

I checked all manifest-listed source, fixture and log hashes in the six newer
archives above. I also compared actual small-record bytes across archives.
The complete C prefix equals the historical honest recursive Lean result.
The complete C/R prefix equals the running archive result. The complete
C/R/D envelope equals the D archive result.

The native C input/proof, R output and D running-output files retain exactly
662,424, 61,612 and 446,109 bytes. The final NIFS child output equals the Lean
result's entire returned running value. Its parent equals the D input parent,
and its full state equals both the R and D outgoing states. The common C
input has SHA-256
3f230a887bad5bce6a05f4b8c4fb2a487b9557ec3102a793a7aafc7f251580e0.

I separately encoded the native wire format in memory from the retained raw
Lean fields. This check includes the magic/version, C round messages, all 17
C claims, R parent, all 16 D children, every length, commitment, packed public
input, point, separate evaluation family, zero padding and carried frame
digest. **All 945,983 bytes equal the retained native proof.** The proof file
SHA-256 is
3cc9d9d9f58fb4a11d30fff99c632ab657f3f2747e8a94ea685ac5deb0387f3e.
This independent byte check does not call either Rust encoder.

The source-level result encoders cover the goal's complete phase values.
PiCCSInputCheck.execute emits 15 fields: acceptance, alpha, gamma, pre-state,
all round challenges and states, r', initial and intermediate claims, the six
terminal values, all 17 commitments/public inputs, the separate Pad and matrix
families, and outgoing state. PiRLCInputCheck emits 11 fields, including every
rho, membership result and all 17 indexed partial combinations. PiDECInputCheck
emits 17 fields, including the parent bound, all public digits/ranges, each
recomposition family, all 16 claims, unchanged state and exact returned running
value. Diagnostic nonzero flags are not acceptance conditions. Zero high digit
planes and the selected zero matrix slot are permitted. The actual proof,
Pad evaluations and matrix evaluations contain nonzero values.

The native prior-parent check is substantive:
[nifs::verify](/home/nicoarq/develop/Nightstream-nifs-proof-links/crates/neo-fold-clean/src/paper/nifs/verifier.rs:32)
revalidates all carried children through PiDEC against the supplied parent
before C. It then feeds each accepted phase result into the next phase.
The test driver supplies the parent commitment, public input, point and both
evaluation families from the previously checked C/R result. Its legacy
fold_digest is the incoming caller frame digest. Equality of this digest
alone does not prove the parent equations or opening validity.

The public NIFS verifier takes the caller-selected relation header. The
public C equations do not read an evaluator cache. The fixed NIFS, F-prime
and finalization callers retain cache validation where they own preprocessing.
The reviewed native run therefore needs no dummy matrix cache. This removes
no matrix-conformance obligation and gives the header digest no authority.
The retained fixed-NIFS cache-substitution regression checks a same-shape
cache from another relation.

For the matrix and raw-assignment links, I checked the bodies of
[compare_raw_matrices](/home/nicoarq/develop/Nightstream-nifs-proof-links/crates/nightstream-fprime/src/bin/check_package_conformance/independent_assignment.rs:142),
[evaluate_canonical_assignment](/home/nicoarq/develop/Nightstream-nifs-proof-links/crates/nightstream-fprime/src/bin/check_package_conformance/canonical_assignment.rs:426)
and the
[recursive caller checker](/home/nicoarq/develop/Nightstream-nifs-proof-links/crates/nightstream-fprime/tests/support/recursive_step.rs:52).
The matrix comparison reads every scheduled row of all three final matrix
objects and compares sorted indices and coefficients, including the constant
and public mapping; it checks padded rows as zero. The independent evaluator
decodes canonical rows and computes their field residuals from the already
generated raw assignment. It does not call the witness generator, Rust matrix
expander or runtime constraint evaluator. The caller checker binds the full
prior/output preimages, exact previous C proof, all children and public hash
encoding. It does not accept a self-consistent digest chain as opening proof.

Source comparison from 2c63f41de09cfc47e6e4be4afd17e3e8a7726049 to the reviewed
cut shows no change to the relevant matrix, row, layout, phase-circuit,
profile, transcript or package owners. In the compared Rust matrix/primitive
crates, only the two separate opening-test files changed. From 9dcb4e7b to the
reviewed cut, no Rust protocol source changed. The C/R checker changes add
the probe view and its proofs; the prior execution definitions are unchanged.
The R-parent change adds complete batch equality and uses it in the weak
success proof. These comparisons justify reuse at the exact stated scopes.
Large raw witness caches are external to the ZIP files. Their recorded
identities and regeneration commands support reproducibility; this reviewer
did not re-evaluate those large buffers.

The mutation evidence has separate scopes. Historical honest PiCCS checks
cover proof, statement, output and point mutations. The actual C/R archive
adds 62 native R mutations and seven serialized result mutations. The
injected sampler test direct_decoder_matches_lean_boundaries_and_fails_closed
passes, including a stream with only 53 accepted coefficients. The complete
run's 43 NIFS mutations cover missing/changed prior authority, all prior child
commitments, fresh public input, C message/evaluation/order and R/D outputs.
Its 55 D mutations cover every child commitment, public input, point, Pad,
all matrix families, shape, padding, frame digest and digit range. Serialized
comparison failure and native verifier rejection are kept distinct.

The exact limits on approval and closure are:

| Status | Finding |
|---|---|
| Independently substantiated at this cut | The C/R/D formula-to-predicate-to-circuit map; separate Pad and 14-matrix families; same selected context and prior-input owner; complete recorded Lean/optimized phase and NIFS outputs; exact canonical proof bytes; the stated retained matrix, raw-assignment and mutation results. |
| Source and retained-audit evidence | The named compiler, physical-preservation, package and terminal-input proof links. This reviewer checked their statements and consuming code and inspected the retained successful audits. No new kernel or implementation execution was performed. |
| Conditional proof | The strong/weak extraction and probability/work results still require their stated continuation, private-witness operations, ambient checkers, accessors and work bounds. The approved same-key MSIS premise supplies no numerical success bound. A deterministic Poseidon2 replay does not discharge Fiat–Shamir transfer. |
| Broader per-phase item 4 remains open | The historical recursive PiCCS accept action does compare the same honest input in Lean, PaperExact and optimized Rust. The older R/D three-way tests use different synthetic fixture inputs/results. I compared those artifact values against the retained honest C/R/D envelope and they differ. The new full native run uses optimized C/R/D. It therefore does not supply same-input PaperExact R/D results. This does not weaken the established Lean/optimized equality requested by the active NIFS task. |
| Historical blanket mutation diagnostic remains failed | The retained final logical-assignment test reports no effective mutation for assignment blocks 12–14, which are absent from canonical rows: piCcsPayload, runningRoundC0 and runningRoundC1. [CONFORMANCE_FIXES.md](/home/nicoarq/develop/Nightstream-nifs-proof-links/docs/reviews/nightstream-fprime-requirements/CONFORMANCE_FIXES.md:99) records the failure. The source analysis below identifies these as unused duplicate allocations. This failed diagnostic does not identify an authoritative NIFS value that can change without rejection. |
| Approval not granted here | Conformance-closed or Production-closed phase status, a complete production lifecycle, full HyperNova history extraction, a new Fiat–Shamir model, PaperExact execution, or a proof backend. The named reviews and owner decisions retain their own scope. |

The three failed mutation blocks were checked against their actual consumers.
PerApplicationAssignmentPlan.block still emits piCcsPayload, runningRoundC0
and runningRoundC1 as blocks 12, 13 and 14. The selected semantic path reads
other values:

| Retained duplicate | Actual consumer and proof of its source |
|---|---|
| piCcsPayload | [DirectPiDECPrefixPlan.piCcsPayload](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/DirectPiDECPrefixPlan.lean:75) selects PiCCSPayloadWiring.form. [PiCCSPayloadWiring.form_eval](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PiCCSPayloadWiring.lean:184) proves that each selected form equals the declared action expression through the ordinary source map for every assignment with the required constant. It does not read the separately allocated payload block. |
| runningRoundC0 and runningRoundC1 | [RunningTransitionDirectPlan.Location.form](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/RunningTransitionDirectPlan.lean:96) selects PiCCSTranscriptOutputForms.pointForm for both components. [pointForm_eq_outputState](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Export/Stage1/PiCCSTranscriptOutputForms.lean:67) identifies them with the actual constrained Poseidon outputs. The separate roundC0Block and roundC1Block allocations are not selected. |

ActualStep.withDecodedPiCCS reads PiCCSAssignmentSoundness.decodedEnv, which
evaluates PiCCSOrdinaryDirectPlan.sourceMap. Its proofLogical transcript case
selects the same transcript output forms. ActualPiDECMessages.proof installs
this decoded C proof. ActualRunningTransition.decodedEnv uses the running
source map above. Thus the statement/proof/point consumers do not obtain
authority from these three duplicate allocations. No missing C/R/D conjunct
was identified.

These coordinates still increase the carrier width and can affect a
commitment and Pad evaluation when an opening changes. They are not absent
from the committed vector. Their values have no independent decoded Step
meaning. Requiring an effective row mutation for every nonempty allocation
is stronger than requiring rejection for every authoritative NIFS family.
The historical diagnostic remains failed, but a production repair of these
duplicates is rejected as unnecessary for this review contract.

For the broader three-way R/D gap, the smallest existing computational owners
are paper_exact::sample_rho_n, paper_exact::verify_pi_rlc and
paper_exact::verify_pi_dec, plus the corresponding public phase wrappers.
The existing external-input checker exposes only optimized R; the older R/D
test helpers read fixed synthetic artifact paths. A public-only external
R/D comparison can reuse the retained honest fields and existing complete
result encodings after a narrow runner change and a release build. It needs
no prover, private witness cache or backend. Execution requires the owner's
explicit PaperExact approval under the project rule. The coordinator later
requested preparation of such a validation runner. Any code, build or
execution from that later work has a separate source cut and does not change
the evidence status for 773f3d0f. This report supplies no execution approval.

Rejected closure claims: a green diagnostic count is not per-phase closure;
file hashes are not semantic authority; synthetic three-way R/D results are
not same-input honest-chain results; terminal acceptance with named premises
is not an unconditional security or history-extraction theorem.

---

**Separate committed storage-delta review.** This section reviews
773f3d0f29209b33e2325538d5f258f541569c25 through
6e4b08f1a9acd21d8a01489fb066b87910714c84. It does not extend the original
execution records to a changed verifier. The diff contains only
CostedWitnessProjection, CheckedWitnessExtraction, the two new stored-witness
modules, and their explicit AxiomsNifsClosure entries. It changes no C/R/D
predicate, circuit, transcript, layout, package, profile or Rust protocol
implementation.

| Claim | Review result and exact scope |
|---|---|
| Stored input representation | [StoredWitnessProjection.StoredWitness](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/PiCCS/PaperJoint/StoredWitnessProjection.lean:19) is a nested Vector with the exact source count and carrier width. In the selected Lean 4.30.0 library, Vector wraps Array. The view erases storage to the same coordinate function. It does not convert an arbitrary semantic function into an array at constant cost. |
| Projection value | StoredWitnessProjection.project_value proves equality with WitnessProjection.project on the erased stored input. Fresh sources return their private tails; running sources return their full vectors. reconstruct_project needs the existing fresh public-prefix equality. It does not assume the complete source is valid. |
| Preserved old path | CostedWitnessProjection.projectReads factors out the existing copier. The semantic project passes the same accessor to it. CheckedWitnessExtraction.finishChecked factors out the prior branch logic. On abort, rejection and success, the old values and returned counter expressions are preserved. |
| Concrete stored read counter | StoredWitnessProjection.read returns the selected field and counts the outer array read, inner array read and result construction. project_work_le supplies this reader to the existing projection bound. No arbitrary-function access bound is assumed for this reader. This is an operation-counter theorem under the stated accounting convention, not a compiler-runtime theorem. |
| Checked stored return | CheckedWitnessExtraction.finishStored_source_iff proves the exact existing B.2 source-success event. It still requires correctness of the supplied public/ambient checker. Checker acceptance provides the fresh public-prefix equality used by reconstruction. The theorem does not assume source truth or infer it from storage. |
| One-run value connection | [StoredOneRunExtraction.run_source_iff](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/PiCCS/PaperJoint/StoredOneRunExtraction.lean:83) uses callCorrect to identify the erased stored call with the same CausalExecution.run and checkCorrect to identify the checker. Those premises remain explicit. This is not a concrete construction of the producer or ambient checker. |
| Returned work connection | run_work_le retains the producing call's counter, the checker counter on a returned candidate, and projection/dispatch costs. expected_work_bound assumes summability of that same base clock and takes the mean under the existing verifier-coin distribution. It does not supply the producer's cost, a polynomial base bound, or a Fiat–Shamir transfer. |

The source statements keep storage construction, validation and coin-reading
cost at the producing call. The theorems add the call's supplied counter; they
do not independently prove that a particular producer has charged those
operations. That concrete cost proof remains required.

The existing projection copier returns List.Vector values and recursively
passes a read closure of the form fun index => read index.succ. Its returned
counter does not account for general Lean closure traversal, allocation
overhead or later access to the returned list representation. The source
does not provide a compiler/runtime cost theorem for this path. Output
representation and projection therefore remain **unresolved runtime
evidence**. A counter equality or bound must not be reported as closure of
the full representation-cost obligation. The value theorems are unaffected.

I inspected the successful stored-projection, checked-return and one-run
audit records in /tmp/nightstream-nifs-stored-projection-1.log,
/tmp/nightstream-nifs-stored-return-1.log and
/tmp/nightstream-nifs-stored-one-run-2.log. The final one-run audit ends with
exit 0 after four seconds. The stored-interface boundary log reports a pass.
The new explicit theorem audits report only propext, Classical.choice and
Quot.sound. These are coordinator-produced records; this reviewer did not
launch a Lean command.

This committed delta is accepted as value-preserving storage/projection and
returned-counter proof work at the stated scope. It does not close the
remaining producer, ambient-checker, output-runtime or cryptographic-model
obligations. A later stored-arithmetic implementation requires its own final
source and audit identity.

**Separate stored-arithmetic source review.** The additional
[StoredAssignmentArithmetic module](/home/nicoarq/develop/Nightstream-nifs-proof-links/formal/nightstream-fprime/NightstreamFPrime/Spec/Folding/Nifs/StoredAssignmentArithmetic.lean:1)
was reviewed after the coordinator finished its 222-line version and paused
edits. The base was 6e4b08f1a9acd21d8a01489fb066b87910714c84. The new file's
SHA-256 at review was
5286553017b90af1c5f50ce2b85103da4c776e4103fd55dcc6f7591f71371362.
It was not yet committed. This file identity applies only to this separate
delta and does not change the original 773f3d0f verdict.

The value claims are substantiated by the inspected definitions and proofs:

| Claim | Exact meaning |
|---|---|
| build_value | Every coordinate of the returned Vector equals the value returned by the supplied coordinate action at that same Fin index. |
| subtract_value | Erasing storage gives pointwise field subtraction of the two supplied arrays. It is not an implementation of the complete B.3 extractor. |
| combine_value | Erasing storage gives the existing BaseLinear.Raw.combineAssignments with the same stored weights and ordered source arrays. The helper is generic in source count. |
| recompose_value | The selected entrypoint first constructs the 16 canonical binary weights and then returns exactly PiDEC.Raw.recomposeAssignment. The existing raw_recomposeAssignment_eq connects that operation to typed B.4 recomposition. No norm or child-validity conclusion is claimed by this value equality. |

The build implementation uses Vector.ofFnM. I checked the selected Lean
4.30.0 library source: this function starts with Array.emptyWithCapacity,
keeps the same callback, and advances a direct Fin index while it pushes each
result. The successor decomposition used inside the proof is a library
equality; it does not replace the executable loop with nested read closures.
Thus this new builder avoids the assignment-width closure chain in the older
List.Vector projection copier.

build_work_le bounds the returned counter by the initial width/capacity
charge, each coordinate action's returned work, two loop charges per
coordinate, and the final return. subtract_work_le and combine_work_le fill
in their explicit read and arithmetic charges. recompose_work_le additionally
includes the construction of every binary weight and the final dispatch.
Its source count is fixed to productionGlobalParams.k = 16. The private
binaryPower call receives exponents from 0 through 15; it cannot select a
different decomposition profile.

These are explicit **returned-counter bounds**. No theorem here links the
counter to compiled Lean execution time or proves the complete extractor's
runtime. The generic combine counter should not be described as a full runtime
bound for arbitrary source count. The selected 16-child recomposition also
does not settle the output representation and projection costs identified
above. No new cryptographic or challenge-distribution assumption is added.

The coordinator's
[/tmp/nightstream-nifs-stored-recompose-2.log](/tmp/nightstream-nifs-stored-recompose-2.log)
records a successful focused module build, with the module job reported as
411 ms and bounded command exit 0. The earlier arithmetic build is also
retained. The final combined axiom audit was still pending when this source
review was first written. Its later result is recorded separately below.

The module adds storage arithmetic and proof connections only. It changes no
existing C/R/D acceptance predicate, transcript, matrix or package definition,
and no Rust protocol path. This delta is accepted for the stated coordinate
value and returned-counter claims. Concrete use by the complete extractor
and full representation/runtime evidence retain their separate obligations.

**Separate audit-record update.** I checked
[/tmp/nightstream-nifs-stored-sampler-axioms-2.log](/tmp/nightstream-nifs-stored-sampler-axioms-2.log).
It records `lake build tests.AxiomsNifsClosure` under the 1500-second Lean
cap, successful completion, exit 0 and elapsed time of four seconds. The
records for build_value, build_work_le, subtract_value, subtract_work_le,
combine_value, combine_work_le, recompose_value and recompose_work_le list
only propext, Classical.choice and Quot.sound. The other printed theorem
records also use only that allowed set. This closes the pending combined
axiom gate for the reviewed arithmetic declarations. It is a coordinator
build record; this reviewer launched no Lean command.

At this update, `wc -l` reports 222 lines and the arithmetic file's SHA-256
remains exactly
5286553017b90af1c5f50ce2b85103da4c776e4103fd55dcc6f7591f71371362.
The earlier 213-line statement was a reporting error, not a different
reviewed source. This audit update does not extend the source review to the
new stored checker or ShortfallBound files. It changes neither the original
773f3d0f verdict nor the remaining runtime obligations.

**Separate inversion research note.** CompPoly tag v4.30.0-patch1, commit
050f0bc7e9780703beb8d178ec533e52bd87d649, uses
[Lean 4.30.0](https://github.com/Verified-zkEVM/CompPoly/blob/050f0bc7e9780703beb8d178ec533e52bd87d649/lean-toolchain)
and [the active Mathlib commit](https://github.com/Verified-zkEVM/CompPoly/blob/050f0bc7e9780703beb8d178ec533e52bd87d649/lake-manifest.json).
[EuclideanAlgorithm.lean](https://github.com/Verified-zkEVM/CompPoly/blob/050f0bc7e9780703beb8d178ec533e52bd87d649/CompPoly/Univariate/EuclideanAlgorithm.lean)
provides executable xgcd and normXgcd with Bezout and Mathlib agreement
proofs. The inspected import closure contains 15 CompPoly files and 6539
source lines. CPolynomial.ofArray and coeff_ofArray support dense inputs.

The candidate inverse uses threshold zero, then modByMonic to reduce the
input cofactor modulo X^54 + X^27 + 1. The generic remainder is scaled for a
nonmonic divisor. Fuel sufficiency is proved; a full operation-clock bound
is not. The RingF multiplication bridge, inverse value proof and complete
representation/work bounds remain required. This is a source research
result. No dependency or copied source was added and no build was run.
Upstream source carries
[Apache-2.0 licensing and author notices](https://github.com/Verified-zkEVM/CompPoly/blob/050f0bc7e9780703beb8d178ec533e52bd87d649/LICENSE).
