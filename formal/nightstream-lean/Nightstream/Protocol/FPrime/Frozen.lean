import Nightstream.SuperNeo.Folding.PiCCS.PaperCorrections
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrongBoundary
import Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec
import Nightstream.SuperNeo.Folding.PiDEC.PaperReduction
import Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections
import Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.OracleSoundness
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableContinuation
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.InteractiveCompositionBridge
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPrefixCoupling
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.CausalPostPrefixBridge
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.RewindableOracleSoundness
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.PostPrefixWorldSoundness
import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.FullOracleSoundness
import Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract
import Nightstream.HyperNova.NIVCCompatibility
import Nightstream.HyperNova.Construction2.Paper
import Nightstream.Protocol.FPrime.CanonicalVerifier
import Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne
import Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement
import Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier
import Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne
import Nightstream.Protocol.FPrime.Frozen.NonInteractiveContinuationObstruction
import Nightstream.Protocol.FPrime.Frozen.NonInteractiveAdaptiveWitnessObstruction
import Nightstream.Protocol.FPrime.Frozen.NonInteractiveFixedKeyObstruction
import Nightstream.Protocol.FPrime.Frozen.NonInteractiveOracleObstruction
import Nightstream.Protocol.FPrime.Frozen.PiCcsAsymptoticObstruction
import Nightstream.Protocol.FPrime.Frozen.PiCcsFirstSuccessBridge
import Nightstream.Protocol.FPrime.Frozen.NifsNonInteractiveBridge
import Nightstream.Protocol.FPrime.Frozen.PiDecTargetWitnessObstruction
import Nightstream.Protocol.FPrime.Frozen.SumCheckEncodingObstruction
import Nightstream.Protocol.FPrime.Frozen.Obligations

/-!
Frozen paper-authoritative facade for the F-prime verification program.

Authority:

- SuperNeo Sections 4--7 and D.3--D.6;
- HyperNova Sections 3--4, 6.2--6.3, H.1, and H.3.

Owns: curated access to the frozen target propositions, the proved Pi_CCS
formula obstructions/corrections, exact deterministic `NIFS.V` graph, exact
Construction-2 `F'_j` transition, and exact terminal verifier transition.

Does not own: old candidate NIFS semantics, legacy implementation semantics,
Rust, R1CS, artifacts, concrete implementation cost claims, or proofs of the
remaining security targets.

Emits constraints: no.

The `Option` result in `NIFS.V` is the frozen reject-totalization of the
paper's deterministic verifier notation: `none` means rejection and
`some U'` is the unique computed accepted output.

| Child path | Mathematical obligation | Excluded boundary |
|---|---|---|
| `Frozen.Obligations` | fixed signatures for the SuperNeo reduction and NIFS targets | no proof or implementation premise discharges a target |
| `Frozen.NonInteractiveFixedKeyObstruction` | distinct D.5 base samples cannot share one fixed realized PiRLC oracle | no oracle-world distribution |
| `Frozen.PiCcsAsymptoticObstruction` | exact opacity of an arbitrary frozen PiCCS runtime field | does not obstruct an operationally linked game |
| `Frozen.PiCcsFirstSuccessBridge` | exact unbounded finite-alphabet success-gated law, almost-sure termination, EPT, and frozen `PiCcsStrong` bridge | no Fiat--Shamir or downstream composition premise |
| `Frozen.NifsNonInteractiveBridge` | exact `NifsNonInteractiveSound` theorem for the full correlated oracle experiment | named interactive, extraction, and collision contracts remain primitive premises |
| SuperNeo paper corrections/reductions | corrected quantitative boundaries and the finite `Pi_DEC o Pi_RLC o Pi_CCS` knowledge theorem | no Fiat--Shamir or concrete primitive bound |
| `CanonicalVerifier.PaperNonInteractiveNifs` | exact paper NIFS and selected Construction-2 recursive fold | no Rust/R1CS refinement |
| `CanonicalVerifier` | executable base/recursive `F'_j` graph equals Construction 2 | no concrete NIFS semantics by itself |
| `CanonicalTerminalVerifier` | explicit base/recursive terminal relation with no final fold | no concrete relation checker by itself |
| `FixedOne` | payload-minimal one-slot step/terminal exactness | no Rust/R1CS or global arithmetization lower bound |
-/

namespace Nightstream.Protocol.FPrime.Frozen

export Obligations
  (SuperNeoGames strongWeakKnowledgeGame superNeoCompositionGame
    PiCcsStrong PiRlcWeak SharedCommitmentProjection
    PiDecReductionOfKnowledge SuperNeoCompositionReductionOfKnowledge
    superNeoCompositionReductionOfKnowledge SuperNeoPaperObligations
    superNeoPaperObligations_of_components
    NifsSoundModulo NifsComplete
    NifsSoundAndCompleteModulo NifsNonInteractiveSound)


namespace SuperNeo

open Nightstream.SuperNeo.InteractiveReduction.Paper

export Nightstream.SuperNeo.InteractiveReduction.Paper
  (NifsExtractionErrorBudget nifsInteractiveTotal nonInteractiveTotal)

export Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract
  (EventPredicates ExplicitRandomOracleContract
    anyFailure_iff_exists_event anyFailure_probability_le_total)

/- Deterministic obligation-5 core for the typed paper SuperNeo NIFS. -/
export Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
  (nifsSoundAndCompleteModulo)

/- Independent valid source claims construct an accepted paper NIFS fold. -/
export Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
  (sourceValid_exists_verifiedTransition)

/- Finite operational component theorems and the exact boundary that
separates them from the unbounded sampler. -/
export Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrong
  (finitePaperStrong)

export Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.FinitePaperStrongBoundary
  (extractorRuntime_iff_uniformTruncatedWorkBound
    extractorRuntime_iff_all_finite_cutoffs)

export PiCcsAsymptoticObstruction
  (piCcsStrong_iff_runtime
    frozenTarget_without_samplerLink_countermodel
    not_attemptedBridgeWithoutSamplerLink)

/- Headline obligation-1 theorem. The game is definitionally linked to the
success-gated trace law and explicit security-parameter cost family;
almost-sure termination, fresh-initial/conditioned-retry law, and extractor
EPT are conclusions rather than caller premises. -/
export PiCcsFirstSuccessBridge
  (Completion PiCcsSecurityFamily
    piCcsStrong_of_successGatedRetry)

/- Headline obligation-5 theorems. The combined theorem constructs the
deterministic soundness/completeness core and the quantitative explicit-RO
target; only named event contracts remain premises. -/
export NifsNonInteractiveBridge
  (fullOracleMixtureNifsNonInteractiveSound
    paperNifsSoundCompleteAndNonInteractive)

/- Fixed-width paper-polynomial gate and deterministic strong-reduction
core.  The width is verifier-owned; high zero coefficient slots are valid
paper messages rather than a canonicalization failure. -/
export Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.ProtocolPolynomial.FixedWidth
  (accepted_implies_tableTruth_or_badEvent)

export Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongReduction
  (fixedWidthAcceptedProbe_extracts_source_or_badEvent)

export Nightstream.SuperNeo.Folding.PiRLC.PaperWeakFiniteUniform
  (paperWeak)

/- Literal ordered-commitment alignment used by the concrete Theorem-6
coupling. -/
export Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcBatch.CompatibleContext
  (batchPhi_eq_piCcsOutputPhi repeatedBatch_samePhi)

/- Kernel obstruction separating a typed transcript schedule from the
still-required random-oracle and concrete Poseidon2 contracts. -/
export NonInteractiveOracleObstruction
  (distinct_contexts_same_derived distinct_labels_same_squeeze
    distinct_public_inputs_same_bound_state)

/- Kernel obstruction separating batch alignment from the still-required
rewindable-continuation alignment. -/
export NonInteractiveContinuationObstruction
  (replaceForkOracle replaceForkOracle_acceptedOutcome_iff
    replaceForkOracle_transitionOutcome_iff
    distinct_replacement_oracles_same_nifs_execution)

/- Kernel obstruction proving that a non-degenerate D.5 challenge experiment
must vary the realized PiRLC oracle world. -/
export NonInteractiveFixedKeyObstruction
  (outcomeForSample programmingReceipts_force_same_base
    distinct_bases_force_programming_failure)

/- Kernel obstructions separating fixed-witness D.4 contracts and public
PiDEC acceptance from the stronger events their extractors consume. -/
export NonInteractiveAdaptiveWitnessObstruction
  (adaptiveWitnessBad_iff_exists
    fixed_witness_bound_does_not_bound_adaptive_existential)

export PiDecTargetWitnessObstruction
  (child_not_target noTargetWitnessFamily
    accepted_without_piDec_target_witness)

/- Historical kernel obstruction separating the selected paper fixed-width
message relation from canonical variable-length trimming. -/
export SumCheckEncodingObstruction
  (fixed_width_acceptance_is_not_canonical_raw_acceptance)

/- Exact typed non-interactivity boundary.  These equations pin the full
public-input absorption, replay, post-output state, and PiRLC coordinate
alignment without asserting collision probabilities. -/
export Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
  (piCcsExecution_coins_eq_replayInput
    piCcsExecution_outgoingState_eq_postOutput
    piRlcChallenge_eq_response_after_piCcsOutput
    not_piRlcSamplingSetFailure
    AlignedForkOutcome CoordinateProgrammingReceipt
    MultiForkProgrammingFailure CoordinateForkSamplingFailure
    nifsEventPredicates
    transcriptSecurityEvent_implies_eventPredicate
    acceptedFork_implies_ambientTargetOpenings
    piRlcExtractionFailure_implies_forkSampling_or_programmingFailure
    ResidualBadEvent verify_sound_or_residual_or_multiFork
    ResidualFailure InteractiveResidualContract
    NifsExplicitRandomOracleContract
    AcceptedOutcome TransitionOutcome NonInteractiveFailure
    residualBadEvent_iff_residualFailure
    residualFailure_probability_le_total
    acceptedOutcome_implies_transition_or_failure
    nonInteractiveFailure_probability_le_total
    accepted_probability_sub_total_le_transition
    PrefixMessage ContinuationReply RewindableProver
    RewindableForkOutcome
    Key.strongExecutionContext Key.compatibleContext
    Key.compatiblePiDecContext
    Key.compatibleContext_piRlc
    Key.compatiblePiDecContext_paper
    CausalPrefixAlignment
    acceptedCheck_eq_piCcsCheck
    mixingFailure_iff_piCcsMixingRoot
    sumCheckFailure_iff_piCcsSumCheckCollision
    piCcsCheck_extracts_sourceValid_or_badEvent
    batchOfPrefix_eq_nifsPiRlcBatch
    combinedParent_eq_nifsParent
    RewindableProver.toInteractivePiDecReply
    RewindableProver.toInteractivePiDecReply_childAssignments
    RewindableProver.continuationPiDecExecution
    RewindableProver.interactivePiDecExecution_eq_continuation
    RewindableProver.continuationPiDecExecution_baseChallenges_attempt
    RewindableForkOutcome.piDecExecutionAt_eq_continuation
    RewindableForkOutcome.continuationSuccessAt_baseChallenges_iff
    causalPiRlcAdversary causalPrefixRun
    CausalPrefixCouplingContract
    CausalPrefixCouplingContract.support
    CausalPrefixCouplingContract.toPrefixExperiment
    CausalPrefixCouplingContract.support_eq_product
    CausalPrefixCouplingContract.mem_support_iff
    CausalPrefixCouplingContract.support_cardinality
    CausalPrefixCouplingContract.toPrefixExperiment_prefixAligned
    CausalPrefixCouplingContract.toPrefixExperiment_piCcsCheck_extracts_sourceValid_or_badEvent
    CausalPrefixCouplingContract.toPrefixExperiment_batch_eq
    CausalPrefixCouplingContract.interactivePiDecExecution_eq_continuation
    causalPostPrefixOutcomeOfSeed
    CausalPrefixCouplingContract.interactivePiDecExecution_eq_postPrefix
    RewindablePiRlcWorldOutcome.piDecExecutionAt_world_attempt_eq_nifs
    RewindablePiRlcWorldOutcome.continuationSuccessAt_world_iff_nifs_target
    CausalPrefixCouplingContract.interactivePiDecSuccess_iff_postPrefixNifsTarget
    RewindableProver.proofAt RewindableProver.baseChallenges
    RewindableProver.baseProof RewindableProver.piRlcChallenges_baseProof
    RewindableProver.assignmentAt
    RewindableForkOutcome.toAlignedForkOutcome
    RewindableForkOutcome.toAlignedForkOutcome_oracle
    RewindableForkOutcome.toAlignedForkOutcome_batch
    RewindableForkOutcome.ContinuationSuccessAt
    RewindableForkOutcome.continuationSuccessAt_implies_parentOpening
    RewindableForkOutcome.continuationSuccessAt_implies_piRlcVerifies
    RewindableForkOutcome.continuationSuccesses_imply_acceptedFork
    rewindableAlignedExperiment rewindableAlignedUnionBound
    RewindableAcceptedOutcome RewindableTransitionOutcome
    RewindableNonInteractiveFailure AllContinuationsSuccessful
    PiDecContinuationFailure
    piRlcForkSamplingFailure_implies_piDecContinuationFailure
    rewindable_accepted_probability_sub_total_le_transition
    PiRlcVectorWorld RewindablePiRlcWorldOutcome
    RewindableProver.acceptsInPiRlcWorld
    postPrefixForkExperiment
    postPrefixForkExperiment_expectedQueriesAtMost
    PiRlcWorldAcceptedOutcome PiRlcWorldTransitionOutcome
    PiRlcWorldNonInteractiveFailure
    PostPrefixCollisionBudget PostPrefixCollisionContract
    postPrefixFiatShamirBudget
    postPrefixOutcome_worldAccepted_iff
    postPrefixChallengeSamplingFailure_probability_eq_zero
    piRlcWorldProgrammingFailure_probability_le_paper
    postPrefixExplicitRandomOracleContract
    piRlcWorldAccepted_probability_sub_total_le_transition
    postPrefixAccepted_probability_sub_total_le_transition
    PiCcsPrefixOracleWorld PiCcsPrefixExperiment
    PiCcsPrefixExperiment.realizedKey
    FullOracleOutcome fullOracleForkMixture
    FullOracleAcceptedOutcome FullOracleTransitionOutcome
    fullOracleEventPredicates FullOracleNonInteractiveFailure
    fullOracleAccepted_implies_transition_or_failure
    fullOracleChallengeSamplingFailure_probability_eq_zero
    fullOracleProgrammingFailure_probability_le_paper
    FullOracleInteractiveResidualContract
    FullOracleCollisionContract
    fullOracleMixtureExplicitRandomOracleContract
    fullOracleAccepted_probability_sub_total_le_transition
    fullOracleMixtureAccepted_probability_sub_total_le_transition)

universe uStructure uAssignment uPublicInput uPoint uEvaluation uCommitment
  uWeight

/-- Headline obligation-3 theorem: the exact paper Pi_DEC verifier has the
straight-line, zero-loss reduction of knowledge from Section 7.5 / D.6. -/
theorem piDec_reductionOfKnowledge
    {Structure : Type uStructure}
    {Assignment : Type uAssignment}
    {PublicInput : Type uPublicInput}
    {Point : Type uPoint}
    {Evaluation : Type uEvaluation}
    {Commitment : Type uCommitment}
    {Weight : Type uWeight}
    (context :
      Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.Context
        Structure Assignment PublicInput Point Evaluation Commitment)
    (scale : ProbabilityScale Weight) :
    ReductionOfKnowledge scale
      (Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.knowledgeGame
        context scale)
      scale.zero := by
  exact Nightstream.SuperNeo.Folding.PiDEC.PaperReduction.reductionOfKnowledge
    context scale

/- Headline obligation-4 theorem for the exact finite operational
`Pi_DEC ∘ Pi_RLC ∘ Pi_CCS` composition.  The adversary supplies only
`Pi_DEC` child messages and child assignments; the combined parent is
verifier-computed, and Theorem 7 adds zero loss to the finite Theorem-6
budget. -/
export Nightstream.SuperNeo.Folding.PiCCS.PaperJoint.StrongExecution.PiRlcComposition.PiDec
  (finiteReductionOfKnowledge)

/-- Frozen paper obstruction: strict `q / 2` does not contain every centered
Goldilocks residue, contrary to Appendix D.5's universal-coverage step. -/
theorem piRlc_literalAmbientBound_obstruction :
    ¬ Nightstream.SuperNeo.Concrete.centeredMagnitude
        Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.midpointResidue <
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.literalAmbientBound := by
  exact Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.midpointResidue_not_literalAmbientBounded

/-- The corrected strict ambient bound `floor(q / 2) + 1` contains every
production field residue. -/
theorem piRlc_correctedAmbientBound_covers
    (value : Nightstream.SuperNeo.Concrete.F) :
    Nightstream.SuperNeo.Concrete.centeredMagnitude value <
      Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.correctedAmbientBound := by
  exact Nightstream.SuperNeo.Folding.PiRLC.PaperCorrections.all_centeredMagnitude_lt_correctedAmbientBound
    value

end SuperNeo

namespace HyperNova

open Nightstream.HyperNova.NonInteractiveMultiFold
open Nightstream.HyperNova.Construction2.Paper

/- Concrete no-premise Construction-2 refinement using the paper SuperNeo
NIFS verifier. -/
export Nightstream.Protocol.FPrime.CanonicalVerifier.PaperNonInteractiveNifs
  (canonicalFprime_accepts_implies_paperTransition_or_nifsBadEvent
    canonicalFprime_paperTransition_implies_exists_nifsProof_accepts)

universe uKey uRunning uFresh uProof

/- Fixed-one paper-only executable surfaces.  These exports retain the
model-level boundary: they neither refine Rust/R1CS nor claim a global
arithmetization lower bound. -/
namespace FixedOne

namespace Step

export Nightstream.Protocol.FPrime.CanonicalVerifier.FixedOne
  (eval_eq_generic accepts_iff_transition)

end Step

namespace Terminal

export Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.FixedOne
  (eval_eq_generic accepts_iff_transition)

end Terminal

end FixedOne

/-- Frozen exact graph of the deterministic, one-message NIFS verifier. -/
theorem nifsV_accepts_iff
    {Key : Type uKey}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    (verifier : Verifier Key Running Fresh Proof)
    (key : Key)
    (running : Running)
    (fresh : Fresh)
    (proof : Proof)
    (output : Running) :
    Accepts verifier key running fresh proof output <->
      verifier.verify key running fresh proof = some output := by
  exact accepts_iff_verify verifier key running fresh proof output

universe uDigest uState uWitness uEncoded

/-- Headline obligation-6 equation: `F'_j` accepts exactly the independently
expanded Construction-2 transition. -/
theorem fprime_accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) :
    Holds setup machine functionIndex input output <->
      Transition setup machine functionIndex input output := by
  exact holds_iff_transition setup machine functionIndex input output

/-- Headline obligation-7 equation: the compact executable verifier accepts
exactly the independently expanded Construction-2 transition. -/
theorem canonicalFprime_accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount) :
    Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts
        setup machine functionIndex input output <->
      Transition setup machine functionIndex input output := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.accepts_iff_transition
    setup machine functionIndex input output

/-- Canonical augmented-function soundness against the independent selected
NIFS transition. The only admitted failure is the bad event returned by that
selected NIFS verification. -/
theorem canonicalFprime_accepts_implies_semanticTransition_or_selectedNifsBadEvent
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (nifsTransition : Key -> Running -> Fresh -> Running -> Prop)
    (nifsBadEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop)
    (nifsCorrect : NifsSoundAndCompleteModulo setup.nifs
      nifsTransition nifsBadEvent)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount)
    (accepted : Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts
      setup machine functionIndex input output) :
    Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
        setup machine nifsTransition functionIndex input output \/
      Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SelectedNifsBadEvent
        setup nifsBadEvent input output := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.accepts_implies_semanticTransition_or_selectedNifsBadEvent
      setup machine nifsTransition nifsBadEvent nifsCorrect functionIndex input
      output accepted

/-- Honest semantic Construction-2 execution is accepted after replacing
only the single recursive NIFS proof. Every other input field is preserved. -/
theorem canonicalFprime_semanticTransition_implies_exists_nifsProof_accepts
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {Fresh : Type uFresh}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (nifsTransition : Key -> Running -> Fresh -> Running -> Prop)
    (nifsBadEvent : Key -> Running -> Fresh -> Proof -> Running -> Prop)
    (nifsCorrect : NifsSoundAndCompleteModulo setup.nifs
      nifsTransition nifsBadEvent)
    (functionIndex : Fin slotCount)
    (input : Input Key State Witness Running Fresh Proof slotCount)
    (output : Output Digest State Running slotCount)
    (semantic :
      Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.SemanticTransition
        setup machine nifsTransition functionIndex input output) :
    exists nifsProof : Proof,
      Nightstream.Protocol.FPrime.CanonicalVerifier.Accepts setup machine
        functionIndex
        (Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.withNifsProof
          input nifsProof)
        output := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.semanticTransition_implies_exists_nifsProof_accepts
      setup machine nifsTransition nifsBadEvent nifsCorrect functionIndex input
      output semantic

universe uRunningWitness uFreshWitness

/-- Headline terminal equation: base checks only the endpoint; recursive
terminal acceptance checks all instance/witness relations and performs no
additional NIFS fold. -/
theorem terminal_accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    TerminalHolds setup machine relations statement proof <->
      TerminalTransition setup machine relations statement proof := by
  exact terminalHolds_iff_transition setup machine relations statement proof

/-- Headline obligation-7 terminal equation: the compact executable terminal
checker accepts exactly the independent Construction-2 terminal relation and
performs no final NIFS fold. -/
theorem canonicalTerminal_accepts_iff_transition
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (checks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        relations)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.eval
        setup machine relations checks statement proof = true <->
      TerminalTransition setup machine relations statement proof := by
  exact Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.eval_eq_true_iff_transition
    setup machine relations checks statement proof

/-- Executable terminal exactness is independent of NIFS soundness and
performs no final fold. -/
theorem canonicalTerminal_exact_without_nifs
    {Key : Type uKey}
    {Digest : Type uDigest}
    {State : Type uState}
    {Witness : Type uWitness}
    {Running : Type uRunning}
    {RunningWitness : Type uRunningWitness}
    {Fresh : Type uFresh}
    {FreshWitness : Type uFreshWitness}
    {Proof : Type uProof}
    {Encoded : Type uEncoded}
    {slotCount : Nat}
    [DecidableEq State]
    [DecidableEq Encoded]
    (setup : Setup Key Running Fresh Proof slotCount)
    (machine : Machine Key Digest State Witness Running Fresh Encoded slotCount)
    (relations : TerminalRelations Key Running RunningWitness Fresh FreshWitness
      slotCount)
    (checks :
      Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.RelationChecks
        relations)
    (statement : TerminalStatement State)
    (proof : TerminalProof Running RunningWitness Fresh FreshWitness slotCount) :
    Nightstream.Protocol.FPrime.CanonicalTerminalVerifier.eval
        setup machine relations checks statement proof = true <->
      TerminalTransition setup machine relations statement proof := by
  exact Nightstream.Protocol.FPrime.CanonicalVerifier.NifsRefinement.terminal_exact_without_nifs
    setup machine relations checks statement proof

end HyperNova

end Nightstream.Protocol.FPrime.Frozen
