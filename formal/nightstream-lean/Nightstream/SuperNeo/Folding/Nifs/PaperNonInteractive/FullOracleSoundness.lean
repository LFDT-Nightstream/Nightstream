import Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive.FullOracleExperiment

/-!
Global conditional soundness for the full finite NIFS oracle experiment.

Source: SuperNeo Sections 7.3--7.5 and Appendices D.3--D.6.

Owns: exact owners for one accepted-NIFS-without-target-witness predicate and
the four nonzero interactive residual predicates over the two-level oracle
outcome; their composition-ordered union bound; the complete six-event
random-oracle contract; a four-field collision-only contract for the actual
mixture; and the final subtractive soundness theorem. The final theorem proves
zero ideal sampling loss and the Appendix-D.5 programming bound internally.

Does not own: construction of an ideal-random-oracle support, the accepted
target-witness extraction bound, any of the four interactive-event bounds,
any of the four transcript-collision bounds, Poseidon2, Ajtai, Rust, R1CS,
artifacts, minimality, or costs.

Emits constraints: no.

No contract field contains acceptance, transition, their union, or the final
inequality. Exactly nine event bounds remain premises: one target-witness
extraction event, four interactive reduction events, and four typed
transcript collisions; rejected outcomes do not consume the extraction
budget, and the D.5 programming bound is proved internally.
-/

set_option autoImplicit false

namespace Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive

open Nightstream.SuperNeo
open Nightstream.SuperNeo.Folding
open Nightstream.SuperNeo.Folding.PiCCS.PaperJoint
open Nightstream.SuperNeo.InteractiveReduction.FiatShamirContract
open Nightstream.SuperNeo.InteractiveReduction.FiniteUniform
open Nightstream.SuperNeo.InteractiveReduction.Paper
open Nightstream.SuperNeo.InteractiveReduction.ProbabilityCalculus

universe uExtension uCommitment uPublicInput uScalar uState uWeight

/-- Exact accepted global `Pi_DEC` child-extraction predicate. -/
def FullOraclePiDecChildExtractionEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) : Prop :=
  PiRlcWorldPiDecChildExtractionEvent outcome.inner

/-- Exact global `Pi_RLC` accepted-fork sampling predicate. -/
def FullOracleForkSamplingEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) : Prop :=
  PiRlcWorldForkSamplingEvent outcome.inner

/-- Exact global `Pi_CCS` SumCheck predicate. -/
def FullOraclePiCcsSumCheckEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) : Prop :=
  PiRlcWorldPiCcsSumCheckEvent outcome.inner

/-- Exact global `Pi_CCS` mixing-root predicate. -/
def FullOraclePiCcsMixingRootEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) : Prop :=
  PiRlcWorldPiCcsMixingRootEvent outcome.inner

/-- Exact global parent-opening binding predicate. -/
def FullOracleParentBindingCollisionEvent
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    (outcome : FullOracleOutcome prefixExperiment) : Prop :=
  PiRlcWorldParentBindingCollisionEvent outcome.inner

/-- Independent bounds for one accepted target-witness extraction predicate
and four exact interactive residual predicates over any experiment on the
complete two-level oracle outcome. -/
structure FullOracleInteractiveResidualContract
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    {probabilityScale : ProbabilityScale Weight}
    (experiment :
      ProbabilityExperiment probabilityScale
        (FullOracleOutcome prefixExperiment))
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (budget : InteractiveErrorBudget Weight) : Prop where
  piDecChildExtraction :
    probabilityScale.le
      (experiment.probability FullOraclePiDecChildExtractionEvent)
      extractionBudget.piDecTargetWitnessFailure
  piRlcForkSampling :
    probabilityScale.le
      (experiment.probability FullOracleForkSamplingEvent)
      budget.piRlcForkSampling
  piCcsSumCheck :
    probabilityScale.le
      (experiment.probability FullOraclePiCcsSumCheckEvent)
      budget.piCcsSumCheck
  piCcsMixingRoot :
    probabilityScale.le
      (experiment.probability FullOraclePiCcsMixingRootEvent)
      budget.piCcsSchwartzZippel
  parentBindingCollision :
    probabilityScale.le
      (experiment.probability FullOracleParentBindingCollisionEvent)
      budget.adjustedRelaxedBinding

/-- The accepted target-witness extraction bound and four exact global
interactive residual bounds imply the NIFS extraction plus
composition-ordered interactive total. -/
theorem fullOracleResidualFailure_probability_le_total
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    {probabilityScale : ProbabilityScale Weight}
    (scaleLaws : ScaleLaws probabilityScale)
    (experiment :
      ProbabilityExperiment probabilityScale
        (FullOracleOutcome prefixExperiment))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (budget : InteractiveErrorBudget Weight)
    (contract :
      FullOracleInteractiveResidualContract experiment extractionBudget
        budget) :
    probabilityScale.le
      (experiment.probability FullOracleResidualFailure)
      (nifsInteractiveTotal probabilityScale extractionBudget budget) := by
  have sumCheckMixing :
      probabilityScale.le
        (experiment.probability fun outcome =>
          FullOraclePiCcsSumCheckEvent outcome \/
            FullOraclePiCcsMixingRootEvent outcome)
        (probabilityScale.add budget.piCcsSumCheck
          budget.piCcsSchwartzZippel) :=
    probabilityScale.le_trans
      (unionLaw.unionBound FullOraclePiCcsSumCheckEvent
        FullOraclePiCcsMixingRootEvent)
      (scaleLaws.add_mono contract.piCcsSumCheck
        contract.piCcsMixingRoot)
  have piCcsBinding :
      probabilityScale.le
        (experiment.probability fun outcome =>
          (FullOraclePiCcsSumCheckEvent outcome \/
              FullOraclePiCcsMixingRootEvent outcome) \/
            FullOracleParentBindingCollisionEvent outcome)
        (probabilityScale.add
          (probabilityScale.add budget.piCcsSumCheck
            budget.piCcsSchwartzZippel)
          budget.adjustedRelaxedBinding) :=
    probabilityScale.le_trans
      (unionLaw.unionBound
        (fun outcome =>
          FullOraclePiCcsSumCheckEvent outcome \/
            FullOraclePiCcsMixingRootEvent outcome)
        FullOracleParentBindingCollisionEvent)
      (scaleLaws.add_mono sumCheckMixing
        contract.parentBindingCollision)
  have forkTail :
      probabilityScale.le
        (experiment.probability fun outcome =>
          FullOracleForkSamplingEvent outcome \/
            (FullOraclePiCcsSumCheckEvent outcome \/
                FullOraclePiCcsMixingRootEvent outcome) \/
              FullOracleParentBindingCollisionEvent outcome)
        (probabilityScale.add budget.piRlcForkSampling
          (probabilityScale.add
            (probabilityScale.add budget.piCcsSumCheck
              budget.piCcsSchwartzZippel)
            budget.adjustedRelaxedBinding)) :=
    probabilityScale.le_trans
      (unionLaw.unionBound FullOracleForkSamplingEvent
        (fun outcome =>
          (FullOraclePiCcsSumCheckEvent outcome \/
              FullOraclePiCcsMixingRootEvent outcome) \/
            FullOracleParentBindingCollisionEvent outcome))
      (scaleLaws.add_mono contract.piRlcForkSampling piCcsBinding)
  exact
    probabilityScale.le_trans
      (unionLaw.unionBound FullOraclePiDecChildExtractionEvent
        (fun outcome =>
          FullOracleForkSamplingEvent outcome \/
            (FullOraclePiCcsSumCheckEvent outcome \/
                FullOraclePiCcsMixingRootEvent outcome) \/
              FullOracleParentBindingCollisionEvent outcome))
      (by
        simpa [FullOracleResidualFailure, PiRlcWorldResidualFailure,
          ResidualFailure, FullOraclePiDecChildExtractionEvent,
          FullOracleForkSamplingEvent, FullOraclePiCcsSumCheckEvent,
          FullOraclePiCcsMixingRootEvent,
          FullOracleParentBindingCollisionEvent,
          PiRlcWorldPiDecChildExtractionEvent,
          PiRlcWorldForkSamplingEvent,
          PiRlcWorldPiCcsSumCheckEvent,
          PiRlcWorldPiCcsMixingRootEvent,
          PiRlcWorldParentBindingCollisionEvent,
          nifsInteractiveTotal,
          InteractiveErrorBudget.strongWeakTotal] using
          scaleLaws.add_mono contract.piDecChildExtraction forkTail)

/-- Exact six-event random-oracle contract on the complete two-level
outcome. -/
abbrev FullOracleExplicitRandomOracleContract
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    {probabilityScale : ProbabilityScale Weight}
    (experiment :
      ProbabilityExperiment probabilityScale
        (FullOracleOutcome prefixExperiment))
    (budget : FiatShamirErrorBudget Weight) :=
  ExplicitRandomOracleContract experiment
    (fullOracleEventPredicates prefixExperiment) budget

/-- The interactive and random-oracle contracts bound the exact complete
failure union over the global outcome. -/
theorem fullOracleNonInteractiveFailure_probability_le_total
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    {probabilityScale : ProbabilityScale Weight}
    (scaleLaws : ScaleLaws probabilityScale)
    (experiment :
      ProbabilityExperiment probabilityScale
        (FullOracleOutcome prefixExperiment))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (interactiveBudget : InteractiveErrorBudget Weight)
    (fiatShamirBudget : FiatShamirErrorBudget Weight)
    (interactiveContract :
      FullOracleInteractiveResidualContract experiment extractionBudget
        interactiveBudget)
    (randomOracleContract :
      FullOracleExplicitRandomOracleContract experiment fiatShamirBudget) :
    probabilityScale.le
      (experiment.probability FullOracleNonInteractiveFailure)
      (nonInteractiveTotal probabilityScale extractionBudget
        interactiveBudget fiatShamirBudget) := by
  have interactiveBound :=
    fullOracleResidualFailure_probability_le_total scaleLaws experiment
      unionLaw extractionBudget interactiveBudget interactiveContract
  have fiatShamirBound :=
    anyFailure_probability_le_total scaleLaws experiment unionLaw
      (fullOracleEventPredicates prefixExperiment) fiatShamirBudget
      randomOracleContract
  exact
    probabilityScale.le_trans
      (unionLaw.unionBound FullOracleResidualFailure
        (AnyFailure (fullOracleEventPredicates prefixExperiment)))
      (by
        simpa [FullOracleNonInteractiveFailure, nonInteractiveTotal] using
          scaleLaws.add_mono interactiveBound fiatShamirBound)

/-- Generic subtractive soundness for any probability experiment over the
complete dependent outcome. -/
theorem fullOracleAccepted_probability_sub_total_le_transition
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    {Weight : Type uWeight}
    [DecidableEq Extension]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    {prefixExperiment : PiCcsPrefixExperiment key}
    {probabilityScale : ProbabilityScale Weight}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (scaleLaws : ScaleLaws probabilityScale)
    (experiment :
      ProbabilityExperiment probabilityScale
        (FullOracleOutcome prefixExperiment))
    (unionLaw : UnionBound experiment)
    (extractionBudget : NifsExtractionErrorBudget Weight)
    (interactiveBudget : InteractiveErrorBudget Weight)
    (fiatShamirBudget : FiatShamirErrorBudget Weight)
    (interactiveContract :
      FullOracleInteractiveResidualContract experiment extractionBudget
        interactiveBudget)
    (randomOracleContract :
      FullOracleExplicitRandomOracleContract experiment fiatShamirBudget) :
    probabilityScale.le
      (probabilityScale.subtract
        (experiment.probability FullOracleAcceptedOutcome)
        (nonInteractiveTotal probabilityScale extractionBudget
          interactiveBudget fiatShamirBudget))
      (experiment.probability FullOracleTransitionOutcome) := by
  apply loss_le_of_cover scaleLaws experiment unionLaw
    FullOracleAcceptedOutcome FullOracleTransitionOutcome
    FullOracleNonInteractiveFailure
    (nonInteractiveTotal probabilityScale extractionBudget
      interactiveBudget fiatShamirBudget)
  · exact fullOracleAccepted_implies_transition_or_failure laws strongSet
  · exact
      fullOracleNonInteractiveFailure_probability_le_total scaleLaws
        experiment unionLaw extractionBudget interactiveBudget
        fiatShamirBudget interactiveContract randomOracleContract

/-- The four remaining random-oracle premises over the actual complete
mixture.  Sampling and programming are proved internally. -/
structure FullOracleCollisionContract
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq Scalar]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prefixExperiment : PiCcsPrefixExperiment key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (budget : PostPrefixCollisionBudget) : Prop where
  publicInputBindingCollision :
    (fullOracleForkMixture prefixExperiment alphabet
      alphabetValid).probability
        (fullOracleEventPredicates
          prefixExperiment).publicInputBindingCollision ≤
      budget.publicInputBindingCollision
  transcriptReplayCollision :
    (fullOracleForkMixture prefixExperiment alphabet
      alphabetValid).probability
        (fullOracleEventPredicates
          prefixExperiment).transcriptReplayCollision ≤
      budget.transcriptReplayCollision
  transcriptStateCollision :
    (fullOracleForkMixture prefixExperiment alphabet
      alphabetValid).probability
        (fullOracleEventPredicates
          prefixExperiment).transcriptStateCollision ≤
      budget.transcriptStateCollision
  outputAbsorptionCollision :
    (fullOracleForkMixture prefixExperiment alphabet
      alphabetValid).probability
        (fullOracleEventPredicates
          prefixExperiment).outputAbsorptionCollision ≤
      budget.outputAbsorptionCollision

/-- The four global collision bounds, exact zero sampling theorem, and
averaged D.5 theorem instantiate the complete six-event oracle contract. -/
theorem fullOracleMixtureExplicitRandomOracleContract
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq Scalar]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (prefixExperiment : PiCcsPrefixExperiment key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (collisionBudget : PostPrefixCollisionBudget)
    (collisionContract :
      FullOracleCollisionContract prefixExperiment alphabet alphabetValid
        collisionBudget) :
    FullOracleExplicitRandomOracleContract
      (fullOracleForkMixture prefixExperiment alphabet
        alphabetValid).toProbabilityExperiment
      (postPrefixFiatShamirBudget key alphabet collisionBudget) := by
  refine {
    publicInputBindingCollision :=
      collisionContract.publicInputBindingCollision
    transcriptReplayCollision :=
      collisionContract.transcriptReplayCollision
    transcriptStateCollision :=
      collisionContract.transcriptStateCollision
    outputAbsorptionCollision :=
      collisionContract.outputAbsorptionCollision
    challengeSamplingFailure := ?_
    multiForkProgrammingFailure := ?_
  }
  · change
      (fullOracleForkMixture prefixExperiment alphabet
        alphabetValid).probability
          (fullOracleEventPredicates
            prefixExperiment).challengeSamplingFailure ≤ 0
    rw [fullOracleChallengeSamplingFailure_probability_eq_zero]
    exact Rat.le_refl
  · change
      (fullOracleForkMixture prefixExperiment alphabet
        alphabetValid).probability
          (fullOracleEventPredicates
            prefixExperiment).multiForkProgrammingFailure ≤
        ratio (key.arity.total + 1) alphabet.cardinality
    exact
      fullOracleProgrammingFailure_probability_le_paper prefixExperiment
        alphabet alphabetValid

/-- Headline soundness over the complete finite correlated prefix and
post-prefix oracle experiment.

The outcome carrier, public NIFS inputs, malicious prover prefix and
continuation, both oracle realizations, and coordinate forks are all owned by
one experiment. The only remaining assumptions are one accepted target-
witness extraction bound, four exact interactive-reduction bounds, and four
exact transcript-collision bounds. -/
theorem fullOracleMixtureAccepted_probability_sub_total_le_transition
    {Extension : Type uExtension}
    {Commitment : Type uCommitment}
    {PublicInput : Type uPublicInput}
    {Scalar : Type uScalar}
    {State : Type uState}
    [DecidableEq Extension]
    [DecidableEq Scalar]
    [DecidableEq State]
    {shape : Shape}
    {columns blockCount degreeBound : Nat}
    {key : Key Extension Commitment PublicInput Scalar State shape
      columns blockCount degreeBound}
    (laws : PiRLC.PaperForkExtraction.ExtractionAlgebra
      key.nifsPiRlcContext.semantics key.nifsPiRlcContext.params
      key.nifsPiRlcContext.algebra)
    (strongSet : PiRLC.PaperForkExtraction.StrongSetUnits laws.ring
      key.nifsPiRlcContext.algebra.challengeValid)
    (prefixExperiment : PiCcsPrefixExperiment key)
    (alphabet : Support Scalar)
    (alphabetValid : forall scalar, scalar ∈ alphabet.values ->
      key.piRlcAlgebra.challengeValid scalar)
    (extractionBudget : NifsExtractionErrorBudget Rat)
    (interactiveBudget : InteractiveErrorBudget Rat)
    (collisionBudget : PostPrefixCollisionBudget)
    (interactiveContract :
      FullOracleInteractiveResidualContract
        (fullOracleForkMixture prefixExperiment alphabet
          alphabetValid).toProbabilityExperiment
        extractionBudget interactiveBudget)
    (collisionContract :
      FullOracleCollisionContract prefixExperiment alphabet alphabetValid
        collisionBudget) :
    scale.le
      (scale.subtract
        ((fullOracleForkMixture prefixExperiment alphabet
          alphabetValid).probability FullOracleAcceptedOutcome)
        (nonInteractiveTotal scale extractionBudget interactiveBudget
          (postPrefixFiatShamirBudget key alphabet collisionBudget)))
      ((fullOracleForkMixture prefixExperiment alphabet
        alphabetValid).probability FullOracleTransitionOutcome) := by
  exact
    fullOracleAccepted_probability_sub_total_le_transition laws strongSet
      scaleLaws
      (fullOracleForkMixture prefixExperiment alphabet
        alphabetValid).toProbabilityExperiment
      (fullOracleForkMixture prefixExperiment alphabet
        alphabetValid).toProbabilityUnionBound
      extractionBudget interactiveBudget
      (postPrefixFiatShamirBudget key alphabet collisionBudget)
      interactiveContract
      (fullOracleMixtureExplicitRandomOracleContract prefixExperiment alphabet
        alphabetValid collisionBudget collisionContract)

end Nightstream.SuperNeo.Folding.Nifs.PaperNonInteractive
